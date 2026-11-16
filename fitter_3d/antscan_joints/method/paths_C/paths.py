"""Round 6, PATHS-C: limb centrelines by tube decomposition.

Input: a trunk product (S6 TRUNK layout: trunk.npz + trunk.json) and the
envelope mesh (scan frame, um). Output: the round-6 limb product (round-6 limb layout) limbs.json + limbs.npz.

Method
  1. fine grid    the mesh is re-voxelised in the trunk's working frame at pitch BL / 600 (capped at MAX_VOX voxels);
                  the trunk core labels are looked up from the trunk grid and grown by ~1 trunk voxel (removes the
                  one-voxel rind the opening leaves on the trunk surface). Appendage solid A = occupancy - core.
  2. tubes        watershed of the distance transform on A; adjacent pieces whose saddle is shallow and spans a full
                  cross-section are merged (plateau noise along a tube), so segments end at constrictions (articulations)
                  and at contacts. Each pair of touching segments (and each segment touching a trunk part) shares one or
                  more face patches ("necks"). Per patch and side: the local tube axis (PCA of the segment's voxels
                  near the patch), align = |neck normal . axis|, offset = distance of the neck centroid from the axis
                  line / local radius, neck ratio = neck radius / tube radius.
  3. contacts     every cycle of the segment graph (trunk parts joined) must contain a contact: touching limbs, a
                  folded tibia against its femur, a tarsus on the head, an antenna on a leg all close a loop through
                  the trunk. A maximum spanning tree on the patches, weighted by how much each patch looks like a
                  cross-section along a tube (align high, offset low, neck ratio high), keeps exactly one route per
                  loop; the patches left out are contacts and are cut (a voxel layer on one side is removed).
  4. paths        per family (legs from the mesosoma core, antennae / mandibles from the head core): geodesic
                  extremities in the cut solid (the annotation tool's tip rule), min-cost medial paths (cost 1 / D^2) to the core,
                  dense smoothed polylines with the tip marched to the surface and a straight extension into the core
                  (build_path), duplicate routes dropped (dedup).
  5. naming       anatomical axes from the trunk (head direction along the mesosoma, dorsal = away from the leg roots,
                  left = dorsal x head) - never the working frame's axes; legs: the leg-like candidates, side from where
                  coxa + femur go, fore / mid / hind by root order along the mesosoma within a side (pairs checked
                  against the other side); antennae: the long head paths, one per side; mandibles: short front paths.
  6. status       path / partial (tip ends at a cut contact, or much shorter than its mirror limb) / no_path, with
                  confidence from the evidence (side margin, slot certainty, symmetry, purity checks).

  python paths.py --trunk-dir DIR --out-dir DIR --mesh-root DIR [--workers 4] [NAME ...]
"""

import argparse
import json
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tubes
from geomlib import (
    Surface,
    arclen,  # noqa: F401
    build_path,
    dedup,
    load_mesh,
    mcp_paths,
    tips_from,  # noqa: F401
    tips_persistence,
    voxelize,
)

VERSION = "S6 paths_C 2026-11-10"
LEGS = ["L1", "L2", "L3", "R1", "R2", "R3"]
ANTS, MANS = ["AL", "AR"], ["ML", "MR"]
LIMBS = LEGS + ANTS + MANS
CFG = dict(
    k_fine=500,  # fine pitch = BL / k_fine
    max_vox=120e6,  # cap on the fine grid size
    rind_trunk_vox=1.0,  # core growth in trunk-grid voxels
    ws_smooth=0.7,
    merge_ratio=0.9,
    merge_fill=0.5,
    tip_nms_pct=4.0,  # minimum persistence of a geodesic extremity, % BL (the annotation tool's NMS radius)
    tip_min_pct=6.0,  # annotation tool
    ext_pct=6.0,  # straight extension into the core (annotation tool)
    ref_nvox=300.0,  # reference pitch (BL / 300) for path smoothing and surface sampling (annotation tool)
    tiny_patch_faces=2,
    strip_split=2.5,  # chunk length of long seg-seg necks in units of the neck radius (0 = off)
    core_bonus=1.0,  # added to the keep-weight of segment-trunk patches: a cut limb root sends the limb through a
    # neighbour (wrong limb), a kept trunk contact only shortens it (partial)
)


# ----------------------------------------------------------------------------------------------- inputs
def load_trunk(trunk_dir, name):
    d = os.path.join(trunk_dir, name)
    tj = json.load(open(os.path.join(d, "trunk.json")))
    z = np.load(os.path.join(d, "trunk.npz"))
    shape = tuple(int(x) for x in z["shape"])
    occ = np.unpackbits(z["occ"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    R = np.asarray(tj["frame"]["rotation_scan_to_canonical"], float)
    o = np.asarray(tj["frame"]["origin_scan"], float)
    T = {
        "occ": occ,
        "shape": shape,
        "lo": np.asarray(z["lo"], float),
        "pitch": float(z["pitch"]),
        "tpart": np.asarray(z["tpart"]).astype(np.int8),
        "R": R,
        "o": o,
        "BL": float(tj["BL_um"]),
        "json": tj,
    }
    for k in ("T_s", "T_r", "T_part"):
        if k in z.files:
            T[k] = np.asarray(z[k])
    if "T_X_frame" in z.files:
        T["T_Xw"] = np.asarray(z["T_X_frame"], float)
    elif "T_X" in z.files:
        T["T_Xw"] = (np.asarray(z["T_X"], float) - o) @ R.T
    js = tj.get("joints_scan") or tj.get("trunk_joints_scan") or {}  # older layout / S6 trunk layout
    T["joints_w"] = {
        k: (np.asarray(v, float) - o) @ R.T for k, v in js.items() if v is not None and np.ndim(v) == 1 and len(v) == 3
    }
    return T


# ----------------------------------------------------------------------------------------------- fine grid
class Grid:
    """Fine grid axis-aligned in the working frame; voxel centre = lo + ijk * h."""

    def __init__(self, Vw, F, BL, T):
        ext = Vw.max(0) - Vw.min(0)
        self.h = h = float(max(BL / CFG["k_fine"], (np.prod(ext + 6 * BL / CFG["k_fine"]) / CFG["max_vox"]) ** (1 / 3)))
        occ, lo = voxelize(Vw, F, h)
        self.occ, self.lo, self.shape = occ, lo + 0.5 * h, occ.shape
        # trunk core labels from the trunk grid (nearest trunk voxel)
        idx = np.argwhere(occ)
        q = np.round((self.lo + idx * h - T["lo"]) / T["pitch"]).astype(int)
        ok = np.all((q >= 0) & (q < np.array(T["shape"])), 1)
        tp = np.full(len(idx), -1, np.int8)
        tp[ok] = T["tpart"][tuple(q[ok].T)]
        lab = np.full(self.shape, -1, np.int8)
        lab[tuple(idx.T)] = tp
        n = max(1, int(np.ceil(CFG["rind_trunk_vox"] * T["pitch"] / h)))
        for _ in range(n):  # grow the core into the rind (label of the max neighbour; ties go to the higher part id)
            g = ndi.grey_dilation(lab, size=3)
            m = occ & (lab < 0) & (g >= 0)
            lab[m] = g[m]
        self.tpart = lab
        self.D = ndi.distance_transform_edt(occ).astype(np.float32)

    def to_w(self, ijk):
        return self.lo + np.asarray(ijk, float) * self.h

    def to_ijk(self, Xw):
        return np.clip(np.round((np.asarray(Xw) - self.lo) / self.h).astype(int), 0, np.array(self.shape) - 1)


# ----------------------------------------------------------------------------------------------- tubes + contacts
def split_patches(I):
    """Split every interface into face-connected patches. Returns per-patch arrays (pairs, n_faces, vec_area, centroid,
    neck) and per-patch face index lists."""
    inv, mid = I["face_inv"], I["face_p0"].astype(float)
    ax = I["face_ax"]
    mid = mid.copy()
    mid[np.arange(len(mid)), ax] += 0.5
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(I["pairs"])))]
    out = {"pairs": [], "faces": []}
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    for q in range(len(I["pairs"])):
        fi = order[starts[q] : starts[q + 1]]
        if len(fi) <= 3:
            comps = [fi]
        else:
            pr = cKDTree(mid[fi]).query_pairs(1.5, output_type="ndarray")
            if len(pr) == 0:
                comps = [fi[[i]] for i in range(len(fi))]
            else:
                M = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(fi), len(fi)))
                nc, cl = connected_components(M, directed=False)
                comps = [fi[cl == c] for c in range(nc)]
        for c in comps:
            out["pairs"].append(I["pairs"][q])
            out["faces"].append(c)
    return out


def build_patches(lab, D, I, core_ids=()):
    P = split_patches(I)
    p0, ax = I["face_p0"], I["face_ax"]
    la = lab[tuple(p0.T)]
    e = np.eye(3, dtype=int)[ax]
    lb = lab[tuple((p0 + e).T)]
    sign = np.where(la < lb, 1.0, -1.0)
    Dm = np.minimum(D[tuple(p0.T)], D[tuple((p0 + e).T)])
    mid = p0.astype(float) + 0.5 * e
    if CFG["strip_split"] > 0:
        # a long strip-shaped neck between two segments (a folded tibia pressed along its femur up to the knee, two
        # legs lying side by side) is cut into chunks along its length, so that the spanning tree can keep the one
        # chunk where the tubes really join (the knee) and cut the side contact
        pairs, faces = [], []
        for pr, fi in zip(P["pairs"], P["faces"]):
            if int(pr[0]) in core_ids or int(pr[1]) in core_ids or len(fi) < 12:
                pairs.append(pr)
                faces.append(fi)
                continue
            M = mid[fi]
            c = M.mean(0)
            w_, U = np.linalg.eigh(np.cov((M - c).T))
            t = (M - c) @ U[:, 2]
            L = t.max() - t.min()
            cl = max(4.0, CFG["strip_split"] * float(Dm[fi].max()))
            if L <= 2.0 * cl:
                pairs.append(pr)
                faces.append(fi)
                continue
            nb = int(np.ceil(L / cl))
            b = np.minimum(((t - t.min()) / L * nb).astype(int), nb - 1)
            for k in range(nb):
                if (b == k).any():
                    pairs.append(pr)
                    faces.append(fi[b == k])
        P = {"pairs": pairs, "faces": faces}
    k = len(P["faces"])
    out = {
        "pairs": np.array(P["pairs"], dtype=np.int64).reshape(-1, 2),
        "n_faces": np.zeros(k, int),
        "vec_area": np.zeros((k, 3)),
        "centroid": np.zeros((k, 3)),
        "neck": np.zeros(k),
        "faces": P["faces"],
    }
    for q, fi in enumerate(P["faces"]):
        out["n_faces"][q] = len(fi)
        out["vec_area"][q] = (e[fi] * sign[fi, None]).sum(0)
        out["centroid"][q] = mid[fi].mean(0)
        out["neck"][q] = Dm[fi].max()
    return out


FEATS = ["a_min", "a_max", "off_max", "off_min", "nr_min", "nr_max", "log_fill", "elong_min", "log_rel", "tiny"]


def patch_features(Pt, Fe, core_ids):
    """Feature matrix (k, len(FEATS)) per patch from the non-trunk sides, and the is-core flag."""
    k = len(Pt["pairs"])
    X = np.zeros((k, len(FEATS)))
    is_core = np.zeros(k, bool)
    for q in range(k):
        sides = [s for s in (0, 1) if int(Pt["pairs"][q, s]) not in core_ids]
        is_core[q] = len(sides) < 2
        ok = [s for s in sides if np.isfinite(Fe["align"][q, s]) and Fe["elong"][q, s] >= 1.15]
        al = [Fe["align"][q, s] for s in ok]
        of = [min(Fe["offset"][q, s], 3.0) for s in ok]
        el = [Fe["elong"][q, s] for s in sides] or [1.0]
        rad = [max(Fe["rad"][q, s], 1.0) for s in sides] or [1.0]
        nk = max(Pt["neck"][q], 0.5)
        X[q] = [
            min(al) if al else 0.5,
            max(al) if al else 0.5,
            max(of) if of else 1.0,
            min(of) if of else 1.0,
            min(nk / max(rad), 1.5),
            min(nk / min(rad), 1.5),
            np.log(Pt["n_faces"][q] / (np.pi * nk**2)),
            min(el),
            np.log(Pt["n_faces"][q] / (np.pi * min(rad) ** 2)),
            float(Pt["n_faces"][q] <= CFG["tiny_patch_faces"]),
        ]
    return X, is_core


def load_contact_model(path=None):
    path = path or os.path.join(HERE, "contact_model.json")
    if os.path.isfile(path):
        return json.load(open(path))
    return None


def patch_weights(Pt, Fe, core_ids, model=None):
    """Keep-likeness of each patch (higher = more like a cross-section along one tube): the logistic model's
    log-odds of 'along' (contact_model.json, fitted on the dev truth, leave-one-specimen-out validated) or, without a
    model, a hand rule (align + (1 - offset) + neck ratio)."""
    X, is_core = patch_features(Pt, Fe, core_ids)
    if model is not None:
        w = np.zeros(len(X))
        for key, m in (("seg", ~is_core), ("core", is_core)):
            mm = model[key]
            Z = (X[m] - np.asarray(mm["mean"])) / np.asarray(mm["std"])
            w[m] = Z @ np.asarray(mm["coef"]) + mm["intercept"]
        return w + CFG["core_bonus"] * is_core, X
    o = 1.0 - np.clip((X[:, 2] - 0.5) / 1.0, 0, 1)
    w = X[:, 0] + o + 0.5 * np.clip(X[:, 5], 0, 1) - X[:, 9]
    return w + CFG["core_bonus"] * is_core, X


def decompose(Gd, A):
    """Watershed tube segments + trunk part labels -> patches with features."""
    tp = Gd.tpart
    lab = tubes.watershed_segments(Gd.D, A, CFG["ws_smooth"])[0]
    I = tubes.interfaces(lab, Gd.D)
    if I is not None:
        lab = tubes.merge_shallow(lab, Gd.D, I, CFG["merge_ratio"], CFG["merge_fill"])
    N = int(lab.max())
    parts = [p for p in (0, 1, 2, 3) if (tp == p).any()]
    core_id = {p: N + 1 + p for p in parts}
    for p in parts:
        lab[tp == p] = core_id[p]
    I = tubes.interfaces(lab, Gd.D)
    cid = set(core_id.values())
    Pt = build_patches(lab, Gd.D, I, cid)
    # drop trunk-trunk patches and patches with the waist / gaster: those parts are impassable for both families, and
    # joining them into the graph would make a hind leg resting on the gaster close a loop that the tree may break at
    # the leg's mesosoma root
    drop = {core_id[p] for p in (2, 3) if p in core_id}
    keep = np.array(
        [not ((int(a) in cid and int(b) in cid) or int(a) in drop or int(b) in drop) for a, b in Pt["pairs"]], bool
    )
    Pt = {k: (v[keep] if isinstance(v, np.ndarray) else [x for x, kk in zip(v, keep) if kk]) for k, v in Pt.items()}
    Fe = tubes.interface_features(lab, Gd.D, Pt, skip=cid)
    return lab, I, Pt, Fe, core_id


def contact_cuts(lab, I, Pt, Fe, core_id, model=None):
    """Maximum spanning tree over the patches (trunk parts joined by fixed edges) -> cut mask and evidence."""
    import networkx as nx

    core_ids = set(core_id.values())
    chain = [core_id[p] for p in (0, 1) if p in core_id]  # head and mesosoma joined; waist / gaster patches dropped
    w, X = patch_weights(Pt, Fe, core_ids, model)
    G = nx.MultiGraph()
    for q, (a, b) in enumerate(Pt["pairs"]):
        G.add_edge(int(a), int(b), key=q, weight=float(w[q]))
    for a, b in zip(chain, chain[1:]):
        G.add_edge(int(a), int(b), key=-1, weight=1e9)
    Tr = nx.maximum_spanning_tree(G, weight="weight")
    kept = {k for _, _, k in Tr.edges(keys=True)}
    cut = [q for q in range(len(Pt["pairs"])) if q not in kept]
    mask = np.zeros(lab.shape, bool)
    p0, ax = I["face_p0"], I["face_ax"]
    e = np.eye(3, dtype=int)[ax]
    sh = np.array(lab.shape)
    for q in cut:
        a, b = (int(x) for x in Pt["pairs"][q])
        if a in core_ids or b in core_ids:
            x, y = (b, a) if a in core_ids else (a, b)
        else:
            x, y = (a, b) if Fe["rad"][q, 0] >= Fe["rad"][q, 1] else (b, a)
        fi = Pt["faces"][q]
        V = np.vstack([p0[fi], p0[fi] + e[fi]])
        lo_b = np.maximum(V.min(0) - 3, 0)
        hi_b = np.minimum(V.max(0) + 4, sh)
        sl = tuple(slice(lo_b[d], hi_b[d]) for d in range(3))
        L = lab[sl]
        fm = np.zeros(L.shape, bool)
        fm[tuple((V - lo_b).T)] = True
        near = ndi.binary_dilation(fm, structure=np.ones((3, 3, 3)))
        cv = (L == x) & near & ndi.binary_dilation(L == y, structure=np.ones((3, 3, 3)))
        mask[sl] |= cv

    def rec(q):
        return {
            "pair": [int(v) for v in Pt["pairs"][q]],
            "faces": int(Pt["n_faces"][q]),
            "centroid_vox": Pt["centroid"][q].round(1).tolist(),
            "w": round(float(w[q]), 3),
            "feat": [round(float(v), 3) for v in X[q]],
        }

    info = {
        "n_segments": int(lab.max()) - len(core_ids),
        "n_patches": int(len(Pt["pairs"])),
        "n_cut": len(cut),
        "model": "logistic contact_model.json" if model else "hand rule",
        "kept": [rec(q) for q in sorted(kept) if q >= 0],
        "cut": [rec(q) for q in cut],
    }
    return mask, info


# ----------------------------------------------------------------------------------------------- families
def family_paths(Gd, passable, src, other_core, BL, surf, fam, ref_pitch):
    from skimage.graph import MCP_Geometric

    A = passable & ~other_core
    sb = src & ~ndi.binary_erosion(src) & ndi.binary_dilation(A & ~src)
    if not sb.any():
        return []
    reg = ndi.binary_dilation(A, iterations=1) | sb
    ii = np.argwhere(reg)
    lo_b, hi_b = ii.min(0), ii.max(0) + 1
    sl = tuple(slice(lo_b[d], hi_b[d]) for d in range(3))
    Ab, sbb, srcb = A[sl], sb[sl], src[sl]
    pas = Ab & ~srcb | sbb
    cost1 = np.where(pas, 1.0, np.inf)
    starts = np.argwhere(sbb)
    mcp = MCP_Geometric(cost1, fully_connected=True)
    Gc, _ = mcp.find_costs([tuple(s) for s in starts])
    Gc[srcb] = 0
    tips, pers = tips_persistence(
        Gc, Ab & ~srcb, CFG["tip_nms_pct"] / 100 * BL / Gd.h, CFG["tip_min_pct"] / 100 * BL / Gd.h
    )
    if len(tips) == 0:
        return []
    Db = np.maximum(Gd.D[sl], 0.5)
    cost = np.where(pas, 1.0 / Db**2, np.inf)
    paths, _ = mcp_paths(cost, starts, tips)
    centroid = Gd.to_w(np.argwhere(src)).mean(0)
    other_b = other_core[sl]
    out = []
    for t, p in zip(tips, paths):
        if p is None:
            continue
        p = p[::-1] + lo_b
        bp = build_path(Gd.to_w(p), ref_pitch, BL, surf, core_target=centroid, ext_pct=CFG["ext_pct"])
        tt = t
        lo_i, hi_i = np.maximum(tt - 3, 0), np.minimum(tt + 4, np.array(other_b.shape))
        pseudo = bool(other_b[lo_i[0] : hi_i[0], lo_i[1] : hi_i[1], lo_i[2] : hi_i[2]].any())
        bp.update(
            {
                "family": fam,
                "ijk": p,
                "tip_ijk": tt + lo_b,
                "G_tip_um": float(Gc[tuple(tt)] * Gd.h),
                "centroid": centroid,
                "pseudo_tip": pseudo,
            }
        )
        out.append(bp)
    return out


# ----------------------------------------------------------------------------------------------- anatomy + naming
NAMING = dict(
    leg_min_free_pct=15.0,  # leg candidates: free arc >= this (% BL) ...
    leg_min_away=0.35,  # ... and this share of the free arc farther than 4 % BL from the trunk core
    u_slot=(
        0.59,
        0.20,
        0.03,
    ),  # expected root position along the mesosoma (2nd..98th pct of the core, 0 back .. 1 front)
    # of fore / mid / hind legs: means of the dev candidates matched to truth (dev r4)
    u_sigma=0.12,
    u_weight=0.3,  # weak: the prior only breaks ties the order / pairing cannot decide
    pair_sigma=0.1,  # mirror legs: difference of root positions along the mesosoma
    empty_slot_cost=1.5,
    shared_root_pct=1.5,  # two leg candidates whose roots are closer than this (% BL) share a coxa
    side_margin_pct=1.5,  # side score (% BL) beyond which a leg's side is taken as certain
    ant_min_free_pct=18.0,
    man_max_free_pct=50.0,
)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def at_arc(X, s, a):
    a = float(np.clip(a, 0, s[-1]))
    return np.array([np.interp(a, s, X[:, k]) for k in range(3)])


def anatomy(T, st):
    """Anatomical axes in the working frame from the trunk parts and the leg roots (never the frame's own axes):
    x = head direction along the mesosoma (neck - b_a_1, refined by the mesosoma's principal axis), z = dorsal = away
    from the roots of the long mesosoma paths (legs attach ventrally), y = z x x (animal's left)."""
    BL, P = st["BL"], st["part_pts"]
    J = T["joints_w"]
    cent = {p: v.mean(0) for p, v in P.items()}
    front = J.get("neck", cent.get(0))
    back = J.get("b_a_1", cent.get(2, cent.get(3)))
    if front is None or back is None or 1 not in P:
        raise RuntimeError("trunk product lacks head / mesosoma / waist-gaster parts")
    meso = P[1]
    x = unit(front - back)
    c = meso.mean(0)
    w_, U = np.linalg.eigh(np.cov((meso - c).T))
    if w_[2] > 2.0 * w_[1]:
        ax = U[:, 2] * np.sign(U[:, 2] @ x)
        if ax @ x > 0.8:
            x = unit(0.5 * x + 0.5 * ax)
    legs = sorted(
        [q for q in st["cands"] if q["family"] == "leg" and q["s_main_end"] > 0.2 * BL], key=lambda q: -q["s_main_end"]
    )[:8]
    radial = []
    for q in legs:
        r = at_arc(q["X"], q["s"], q["s_main_end"]) - c
        r = r - (r @ x) * x
        if np.linalg.norm(r) > 1e-9:
            radial.append(unit(r))
    if radial:
        v = np.mean(radial, 0)
        z = -unit(v - (v @ x) * x)
        z_src = f"away from {len(radial)} leg roots (mean radial length {np.linalg.norm(v):.2f})"
    else:
        z = unit(np.cross(x, np.array([1.0, 0, 0])) if abs(x[0]) < 0.9 else np.cross(x, np.array([0, 1.0, 0])))
        z_src = "no leg roots: arbitrary (side unresolved)"
    y = np.cross(z, x)
    Rb = np.stack([x, y, z])
    um = (meso - c) @ x
    lo_u, hi_u = np.percentile(um, [2, 98])
    return {
        "Rb": Rb,
        "c": c,
        "u_lo": float(lo_u),
        "u_hi": float(hi_u),
        "info": {
            "z_source": z_src,
            "x_source": "neck - b_a_1" if ("neck" in J and "b_a_1" in J) else "part centroids",
            "meso_length_pct_BL": round(100 * float(hi_u - lo_u) / BL, 2),
        },
    }


def cand_features(c, An, BL):
    X, s = c["X"], c["s"]
    core = c["s_main_end"]
    Rb, cc = An["Rb"], An["c"]
    rel = lambda p: Rb @ (p - cc)  # noqa: E731
    root = rel(at_arc(X, s, core))
    p8 = rel(at_arc(X, s, core - 0.08 * BL))
    p15 = rel(at_arc(X, s, core - 0.15 * BL))
    tip = rel(X[0])
    free = (s <= core) & (s >= 0.02 * BL)
    rr = c["r"][free]
    return {
        "free_pct": 100 * core / BL,
        "u": float((root[0] - An["u_lo"]) / max(An["u_hi"] - An["u_lo"], 1e-9)),
        "root_x_pct": 100 * root[0] / BL,
        "root_y_pct": 100 * root[1] / BL,
        "root_z_pct": 100 * root[2] / BL,
        "p8_y_pct": 100 * p8[1] / BL,
        "p15_y_pct": 100 * p15[1] / BL,
        "tip_y_pct": 100 * tip[1] / BL,
        "tip_x_pct": 100 * tip[0] / BL,
        "side_pct": 0.25 * 100 * root[1] / BL + 0.35 * 100 * p8[1] / BL + 0.4 * 100 * p15[1] / BL,
        "r_med_pct": 100 * float(np.median(rr)) / BL if len(rr) else 0.0,
        "r_min_rel": float(np.min(rr) / max(np.median(rr), 1e-9)) if len(rr) else 0.0,
        "away": float(c.get("away", 1.0)),
        "pseudo_tip": float(bool(c.get("pseudo_tip"))),
    }


def assign_legs(cands, feats, BL):
    """Best joint assignment of leg candidates to L1..R3. Within a side the slots follow the root order along the
    mesosoma (front to back, hard constraint). Cost = side evidence + pair consistency (mirror legs root at similar
    positions along the body) + a weak prior on the root position along the mesosoma core (the core's extent varies a
    lot between taxa) + unassigned leg-like candidates (long, standing off the body) + empty slots. Two candidates
    sharing their root (one reached the coxa of the other through a contact) cannot both be named."""
    import itertools

    N = NAMING
    idx = [i for i, f in enumerate(feats) if f["free_pct"] >= N["leg_min_free_pct"] and f["away"] >= N["leg_min_away"]]
    idx = sorted(idx, key=lambda i: -feats[i]["free_pct"])[:9]
    mu, sg, mrg = N["u_slot"], N["u_sigma"], N["side_margin_pct"]
    sides = {i: ([1] if feats[i]["side_pct"] > mrg else [-1] if feats[i]["side_pct"] < -mrg else [1, -1]) for i in idx}
    root = {i: np.array([feats[i]["root_x_pct"], feats[i]["root_y_pct"], feats[i]["root_z_pct"]]) for i in idx}
    share = {(i, j) for i in idx for j in idx if i < j and np.linalg.norm(root[i] - root[j]) < N["shared_root_pct"]}

    def side_cost(i, sgn):
        return 2.0 * max(0.0, mrg - sgn * feats[i]["side_pct"]) / mrg

    def pinch(i):  # a near-zero radius on the free part: the path probably crossed a contact
        return 1.5 * max(0.0, 0.2 - feats[i]["r_min_rel"]) / 0.2

    def legness(i):
        f = feats[i]
        return 1.0 + 2.0 * float(f["free_pct"] >= 35.0) + 1.0 * float(f["away"] >= 0.6)

    best = None
    for lab in itertools.product(*[sides[i] + [0] for i in idx]):
        Ls = [i for i, l in zip(idx, lab) if l == 1]
        Rs = [i for i, l in zip(idx, lab) if l == -1]
        if len(Ls) > 3 or len(Rs) > 3:
            continue
        used = set(Ls) | set(Rs)
        if any(i in used and j in used for i, j in share):
            continue
        cost = 0.0
        slots = {}
        for sgn, grp, nm in ((1, Ls, "L"), (-1, Rs, "R")):
            grp = sorted(grp, key=lambda i: -feats[i]["u"])
            bc, bs = None, None
            for comb in itertools.combinations([1, 2, 3], len(grp)):
                cc = sum(N["u_weight"] * min(((feats[i]["u"] - mu[k - 1]) / sg) ** 2, 4.0) for i, k in zip(grp, comb))
                if bc is None or cc < bc:
                    bc, bs = cc, comb
            cost += bc + sum(side_cost(i, sgn) + pinch(i) for i in grp) + N["empty_slot_cost"] * (3 - len(grp))
            for i, k in zip(grp, bs):
                slots[f"{nm}{k}"] = i
        for k in (1, 2, 3):
            if f"L{k}" in slots and f"R{k}" in slots:
                du = feats[slots[f"L{k}"]]["u"] - feats[slots[f"R{k}"]]["u"]
                cost += min((du / N["pair_sigma"]) ** 2, 6.0)
        for i, l in zip(idx, lab):
            if l == 0:
                dup = [j for j in used if (min(i, j), max(i, j)) in share]
                # a root-sharing duplicate costs little to drop, but more when it is the longer of the two
                cost += (
                    (0.3 + float(any(feats[i]["free_pct"] > feats[j]["free_pct"] + 3.0 for j in dup)))
                    if dup
                    else legness(i)
                )
        if best is None or cost < best[0]:
            best = (cost, slots)
    named = {}
    if best:
        for k, i in best[1].items():
            ns = sum(1 for kk in best[1] if kk[0] == k[0])
            named[k] = (
                i,
                {
                    "side_score_pct": round(feats[i]["side_pct"], 2),
                    "u": round(feats[i]["u"], 3),
                    "n_on_side": ns,
                    "assign_cost": round(best[0], 2),
                },
            )
    return named


def assign_head(cands, feats, BL):
    """Antennae: per side the longest head path standing off the head (side from the proximal part: scape + first
    funiculus segments); mandibles: per side the bulkiest remaining head path (closed mandibles lie on the head and
    their tips cross the midline, so the side comes from the root alone). Of head paths sharing a root, only the
    longest is considered (the shorter ones end at a contact or a branch of the same appendage)."""
    N = NAMING
    idx = [i for i, f in enumerate(feats) if f["free_pct"] >= 5.0]
    root = {i: np.array([feats[i]["root_x_pct"], feats[i]["root_y_pct"], feats[i]["root_z_pct"]]) for i in idx}
    idx = sorted(idx, key=lambda i: -feats[i]["free_pct"])
    kept = []
    for i in idx:
        if all(np.linalg.norm(root[i] - root[j]) >= N["shared_root_pct"] for j in kept):
            kept.append(i)
    side = {i: 0.3 * feats[i]["root_y_pct"] + 0.7 * feats[i]["p8_y_pct"] for i in kept}
    named = {}
    ants = [i for i in kept if feats[i]["free_pct"] >= N["ant_min_free_pct"] and feats[i]["away"] >= 0.2]
    for s_name, sgn in (("L", 1), ("R", -1)):
        mine = sorted([i for i in ants if sgn * side[i] > 0], key=lambda i: -feats[i]["free_pct"])
        if mine:
            named["A" + s_name] = (mine[0], {"side_score_pct": round(float(side[mine[0]]), 2), "n_on_side": len(mine)})
    # an antenna lying along the head capsule (scape in a scrobe, funiculus folded back) does not stand off the head:
    # accept it on a side without an antenna when its length matches the other side's antenna (mirror evidence)
    for s_name, sgn, other in (("L", 1, "AR"), ("R", -1, "AL")):
        if "A" + s_name in named or other not in named:
            continue
        lo_ = feats[named[other][0]]["free_pct"]
        mine = sorted(
            [
                i
                for i in kept
                if i != named[other][0]
                and sgn * side[i] > 0
                and feats[i]["free_pct"] >= N["ant_min_free_pct"]
                and 0.6 <= feats[i]["free_pct"] / lo_ <= 1.6
            ],
            key=lambda i: -feats[i]["free_pct"],
        )
        if mine:
            named["A" + s_name] = (
                mine[0],
                {"side_score_pct": round(float(side[mine[0]]), 2), "n_on_side": len(mine), "lies_on_head": True},
            )
    used = {v[0] for v in named.values()}
    amin = min([feats[v[0]]["free_pct"] for v in named.values()], default=np.inf)
    mans = [i for i in kept if i not in used and feats[i]["free_pct"] <= min(N["man_max_free_pct"], 0.85 * amin)]
    for s_name, sgn in (("L", 1), ("R", -1)):
        mine = sorted(
            [i for i in mans if sgn * feats[i]["root_y_pct"] > 0],
            key=lambda i: -feats[i]["free_pct"] * feats[i]["r_med_pct"] ** 2,
        )
        if mine:
            named["M" + s_name] = (
                mine[0],
                {"side_score_pct": round(float(feats[mine[0]]["root_y_pct"]), 2), "n_on_side": len(mine)},
            )
    return named


def limb_status(limb, c, f, ev, named, feats, cut_tree, h, BL):
    """Status, confidence and evidence notes of a named limb; for a partial limb ev['missing'] declares what is
    probably not covered."""
    conf = 0.9
    notes = []
    status = "path"
    missing = []
    distal = "pt, ta (tarsus / leg tip)" if limb in LEGS else ("an_3 (antenna tip)" if limb in ANTS else "mandible tip")
    if cut_tree is not None and cut_tree.query(c["X"][min(2, len(c["X"]) - 1)])[0] < 3 * h:
        status, conf = "partial", conf - 0.3
        notes.append("tip ends at a cut contact")
        missing.append(f"{distal}: the path ends where a contact was cut; the part beyond it was not resolved")
    mirror = (
        (("R" if limb[0] == "L" else "L") + limb[1:])
        if limb in LEGS
        else ({"AL": "AR", "AR": "AL", "ML": "MR", "MR": "ML"}[limb])
    )
    if mirror in named:
        fm = feats[named[mirror][0]]
        ratio = f["free_pct"] / max(fm["free_pct"], 1e-9)
        ev["mirror_length_ratio"] = round(ratio, 3)
        if ratio < 0.7:
            status = "partial"
            conf -= 0.3
            notes.append("much shorter than its mirror limb")
            missing.append(
                f"distal part: free arc {f['free_pct']:.0f} % BL vs {fm['free_pct']:.0f} % BL on the mirror "
                f"limb (folded, hidden, broken, or the path ends at a contact)"
            )
        elif ratio > 1.4:
            conf -= 0.2
            notes.append("much longer than its mirror limb")
    else:
        conf -= 0.1
        notes.append("mirror limb not found")
    if abs(ev.get("side_score_pct", 0.0)) < NAMING["side_margin_pct"]:
        conf -= 0.2
        notes.append("side margin small")
    if limb in LEGS and ev.get("n_on_side", 3) < 3:
        conf -= 0.15
        notes.append("slot inferred (fewer than 3 legs on this side)")
    if ev.get("lies_on_head"):
        conf -= 0.15
        notes.append("antenna lies along the head; accepted by the length of the other antenna")
    if limb in MANS:
        conf -= 0.2
        notes.append("mandible: chosen among the short head paths by bulk and root side only")
    if f["r_min_rel"] < 0.25:
        conf -= 0.15
        notes.append("near-zero radius on the free part (possible contact crossing)")
    if missing:
        ev["missing"] = missing
    return status, float(np.clip(conf, 0.05, 1.0)), notes


# ----------------------------------------------------------------------------------------------- main
def geometry_stage(name, T, V, F, contact_model=None):
    """Everything up to the candidate paths (working frame). Returns a picklable dict."""
    t0 = time.time()
    BL = T["BL"]
    Vw = (V - T["o"]) @ T["R"].T
    Gd = Grid(Vw, F, BL, T)
    t_grid = time.time() - t0
    tp = Gd.tpart
    A = Gd.occ & (tp < 0)
    labc, I, Pt, Fe, core_id = decompose(Gd, A)
    cmodel = load_contact_model(contact_model)
    cutmask, cinfo = contact_cuts(labc, I, Pt, Fe, core_id, cmodel)
    cinfo["model_file"] = contact_model or "contact_model.json"
    if os.environ.get("PATHS_C_DEBUG"):
        np.savez_compressed(
            os.path.join(os.environ["PATHS_C_DEBUG"], f"{name}_debug.npz"),
            lab=labc,
            cut=cutmask,
            lo=Gd.lo,
            h=Gd.h,
            R=T["R"],
            o=T["o"],
        )
        json.dump(cinfo, open(os.path.join(os.environ["PATHS_C_DEBUG"], f"{name}_cuts.json"), "w"))
    t_tubes = time.time() - t0 - t_grid
    passable = Gd.occ & ~cutmask
    ref_pitch = BL / CFG["ref_nvox"]
    surf = Surface(Vw, F, ref_pitch)
    cands = []
    for fam, sp in (("leg", 1), ("head", 0)):
        if sp not in core_id:
            continue
        src = tp == sp
        other = (tp >= 0) & ~src
        cands += family_paths(Gd, passable, src, other, BL, surf, fam, ref_pitch)
    keep, dropped = dedup(cands, ref_pitch, BL)
    kc = [
        {k: c[k] for k in ("X", "s", "r", "s_main_end", "family", "pseudo_tip", "G_tip_um")}
        for c in (cands[i] for i in keep)
    ]
    core_tree = cKDTree(Gd.to_w(np.argwhere((tp >= 0) & ~ndi.binary_erosion(tp >= 0))))
    for c in kc:  # share of the free arc farther than 4 % BL from the trunk core (limbs stand off the body)
        fr = (c["s"] <= c["s_main_end"]) & (c["s"] >= 0.02 * BL)
        c["away"] = float(np.mean(core_tree.query(c["X"][fr])[0] > 0.04 * BL)) if fr.any() else 0.0
    rng = np.random.default_rng(0)

    def sub(P, n):
        return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]

    core = tp >= 0
    stage = {
        "name": name,
        "BL": BL,
        "h": Gd.h,
        "cands": kc,
        "n_dropped": len(dropped),
        "contacts": cinfo,
        "cut_pts": sub(Gd.to_w(np.argwhere(cutmask)), 60000) if cutmask.any() else np.zeros((0, 3)),
        "core_surface_pts": sub(Gd.to_w(np.argwhere(core & ~ndi.binary_erosion(core))), 60000),
        "part_pts": {p: sub(Gd.to_w(np.argwhere(tp == p)), 20000) for p in (0, 1, 2, 3) if (tp == p).any()},
        "grid": {"h_um": Gd.h, "shape": list(Gd.shape)},
        "seconds": {
            "grid": round(t_grid, 1),
            "tubes": round(t_tubes, 1),
            "paths": round(time.time() - t0 - t_grid - t_tubes, 1),
        },
    }
    return stage


def name_stage(name, T, st, out_dir):
    """Anatomy, naming, status and the limb product (scan frame)."""
    t0 = time.time()
    BL, kc = st["BL"], st["cands"]
    An = anatomy(T, st)
    feats = [cand_features(c, An, BL) for c in kc]
    li = [i for i, c in enumerate(kc) if c["family"] == "leg"]
    hi = [i for i, c in enumerate(kc) if c["family"] == "head"]
    named = {}
    nl = assign_legs([kc[i] for i in li], [feats[i] for i in li], BL)
    named.update({k: (li[v[0]], v[1]) for k, v in nl.items()})
    nh = assign_head([kc[i] for i in hi], [feats[i] for i in hi], BL)
    named.update({k: (hi[v[0]], v[1]) for k, v in nh.items()})
    cut_tree = cKDTree(st["cut_pts"]) if len(st["cut_pts"]) else None
    limbs, arrays = {}, {}
    R, o = T["R"], T["o"]
    for limb in LIMBS:
        if limb not in named:
            limbs[limb] = {"status": "no_path", "confidence": 0.0, "evidence": "no candidate path assigned"}
            continue
        i, ev = named[limb]
        c, f = kc[i], feats[i]
        status, conf, notes = limb_status(limb, c, f, ev, named, feats, cut_tree, st["h"], BL)
        Xs = c["X"] @ R + o
        arrays[limb + "_X"], arrays[limb + "_s"], arrays[limb + "_r"] = Xs, c["s"], c["r"]
        limbs[limb] = {
            "status": status,
            "confidence": round(conf, 3),
            "evidence": "; ".join(notes) if notes else "complete pair, clear side and slot",
            "core_arc_um": float(c["s_main_end"]),
            "arc_um": float(c["s"][-1]),
            "n_points": int(len(Xs)),
            "tip_scan": Xs[0].tolist(),
            "root_scan": (at_arc(c["X"], c["s"], c["s_main_end"]) @ R + o).tolist(),
            "free_arc_pct_BL": round(f["free_pct"], 2),
            "assign": ev,
            "candidate": int(i),
            "family": "P" if limb in LEGS else "H",
        }
    if "T_Xw" in T:
        arrays["T_X"] = T["T_Xw"] @ R + o
        for k in ("T_s", "T_r", "T_part"):
            if k in T:
                arrays[k] = T[k]
    for i, c in enumerate(kc):
        arrays[f"C{i}_X"] = c["X"] @ R + o
    out = {
        "name": name,
        "units": "um",
        "version": VERSION,
        "method": "tube decomposition + contact cuts (PATHS-C)",
        "frame": {"rotation_scan_to_canonical": R.tolist(), "origin_scan": o.tolist()},
        "anatomical_axes_scan": (An["Rb"] @ R).tolist(),
        "anatomy": An["info"],
        "BL_um": BL,
        "BL_source": T["json"].get("BL_source"),
        "limbs": limbs,
        "candidates": [
            {
                "i": i,
                "family": c["family"],
                "named": next((k for k, v in named.items() if v[0] == i), None),
                **{k: (round(float(v), 3) if np.ndim(v) == 0 else None) for k, v in f.items() if np.ndim(v) == 0},
            }
            for i, (c, f) in enumerate(zip(kc, feats))
        ],
        "n_dropped_duplicates": st["n_dropped"],
        "contacts": st["contacts"],
        "grid": st["grid"],
        "seconds": {**st["seconds"], "naming": round(time.time() - t0, 1)},
    }
    out["seconds"]["total"] = round(sum(out["seconds"].values()), 1)
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "limbs.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(d, "limbs.npz"), **arrays)
    return out


def run(name, trunk_dir, mesh_path, out_dir, contact_model=None, stage_dir=None):
    """Limb product for one specimen: trunk product (trunk_dir/<name>/trunk.npz + .json) + envelope mesh ->
    out_dir/<name>/limbs.json + limbs.npz. stage_dir (dev): cache / reuse the geometry stage (pickle)."""
    import pickle

    T = load_trunk(trunk_dir, name)
    sp = os.path.join(stage_dir, f"{name}_stage.pkl") if stage_dir else None
    if sp and os.path.isfile(sp) and os.environ.get("PATHS_C_REUSE_STAGE"):
        st = pickle.load(open(sp, "rb"))
    else:
        V, F = load_mesh(mesh_path)
        st = geometry_stage(name, T, V, F, contact_model)
        if sp:
            os.makedirs(stage_dir, exist_ok=True)
            pickle.dump(st, open(sp, "wb"))
    return name_stage(name, T, st, out_dir)


def _one(args):
    name, trunk_dir, mesh_root, out_dir, skip, cm_dir, stage_dir = args
    if skip and os.path.isfile(os.path.join(out_dir, name, "limbs.json")):
        return name, "skipped", 0.0
    t0 = time.time()
    try:
        mesh = os.path.join(mesh_root, name, f"{name}_envelope.stl")
        cm = os.path.join(cm_dir, f"{name}.json") if cm_dir else None
        if cm and not os.path.isfile(cm):
            raise FileNotFoundError(f"no LOSO contact model for {name} in {cm_dir}")
        o = run(name, trunk_dir, mesh, out_dir, cm, stage_dir)
        st = " ".join(f"{k}:{v['status'][0]}" for k, v in o["limbs"].items())
        return name, f"ok {st} cut {o['contacts'].get('n_cut')} t {o['seconds']}", time.time() - t0
    except Exception as e:
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        json.dump(
            {
                "name": name,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
                "limbs": {k: {"status": "failed"} for k in LIMBS},
            },
            open(os.path.join(d, "limbs.json"), "w"),
            indent=1,
        )
        return name, f"FAILED {type(e).__name__}: {e}", time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--names-file", default=None)
    ap.add_argument("--trunk-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--stage-dir",
        default=None,
        help="dev: cache the geometry stage here; with PATHS_C_REUSE_STAGE=1 "
        "set, an existing cache is reused and only the naming stage runs",
    )
    ap.add_argument(
        "--contact-model-dir",
        default=None,
        help="per-specimen contact models <dir>/<name>.json (LOSO dev runs); default contact_model.json",
    )
    a = ap.parse_args()
    names = list(a.names)
    if a.names_file:
        names += open(a.names_file).read().split()
    if not names:
        names = sorted(d for d in os.listdir(a.trunk_dir) if os.path.isfile(os.path.join(a.trunk_dir, d, "trunk.json")))
    jobs = [(n, a.trunk_dir, a.mesh_root, a.out_dir, a.skip_existing, a.contact_model_dir, a.stage_dir) for n in names]
    if a.workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(min(a.workers, 4), maxtasksperchild=1) as pool:
            for n, msg, dt in pool.imap_unordered(_one, jobs):
                print(f"{n:48s} {dt:6.1f}s {msg}", flush=True)
    else:
        for j in jobs:
            n, msg, dt = _one(j)
            print(f"{n:48s} {dt:6.1f}s {msg}", flush=True)


if __name__ == "__main__":
    main()
