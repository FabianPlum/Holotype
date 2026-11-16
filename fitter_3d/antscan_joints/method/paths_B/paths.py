"""Round-6 PATHS-B: limb centrelines from a curve-skeleton graph of the scan, named by anatomy.

Input: a trunk product (<trunk_dir>/<name>/trunk.npz + trunk.json: occupancy grid in a working frame, opened trunk
core labels head 0 / mesosoma 1 / waist 2 / gaster 3, trunk path, BL) and the envelope mesh (scan frame, um).
Output: <out_dir>/<name>/limbs.json + limbs.npz in the round-6 limb interface (round-6 limb format).

Method (no truth, no posed model, no specimen rules):
  1. re-voxelise the mesh finer (BL / N_VOX) in the working frame; carry the trunk core labels over by nearest voxel
  2. 3D thinning skeleton of the whole solid; the part outside the trunk core becomes a graph (limbgraph.py); nodes
     touching the core are trunk attachments (roots) labelled with the core part
  3. prune spurs (claws, spurs, thinning hairs); cluster junctions; pair the edges that continue each other through
     a junction (tubes.py): a contact between two limbs is a junction that two tubes pass straight through
  4. tubes rooted on the mesosoma (legs) or head (antennae, mandibles) are limb candidates; tubes with no root are
     fragments; stitching and folded-limb handling (see below)
  5. an anatomical frame from the trunk (head direction from the core labels, ventral = side of the leg roots,
     left = ventral x head) names the limbs: legs by side of the proximal leg and root order along the mesosoma,
     antennae / mandibles by length, root position and side
  6. polylines tip -> root (smoothed, tip marched to the surface, root extended straight into the core like the
     truth paths), honest status and confidence from evidence

  python paths.py --trunk-dir DIR --out-dir DIR [--names a,b] --mesh-root DIR
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import trimesh
from scipy import ndimage as ndi

import limbgraph
import proximal
import route as rt
import tubes as tb
import vox
from limbgraph import arclen

VERSION = "S6 paths_B 2026-11-10"
LEGS = ["L1", "L2", "L3", "R1", "R2", "R3"]
ANTS = ["AL", "AR"]
MANS = ["ML", "MR"]
LIMBS = LEGS + ANTS + MANS

CFG = {
    "N_VOX": 400,  # fine grid pitch = BL / N_VOX
    "prune_c_r": 2.0,
    "prune_c_bl": 1.5,  # spur pruning
    "c_short": 1.5,  # junction clustering
    "max_turn": 60.0,
    "max_ratio": 2.5,  # pairing through a junction
    "ext_pct": 6.0,  # straight root extension into the core, % BL (as the truth paths)
    "leg_min_pct": 12.0,
    "leg_lateral_pct": 0.3,
    "leg_dorsal_pct": 4.0,
    "near_body_pct": 3.0,
    "ant_min_pct": 20.0,
    "man_min_pct": 6.0,
    "man_max_pct": 60.0,
    "cand_min_pct": 3.0,
    "mode": "route",
    "lam_turn": 0.05,
    "mu_rad": 0.05,
    "rad_free": 1.3,
    "block_factor": 3.0,
    "pair_short": 0.6,
    "conf_pen_pct": 2.5,
    "conf_pair": 0.1,
    "select": "flow",
    "kappa": 1.5,
    "secondary": True,
    "spur_max_pct": 4.0,
    "anchor_leg_pct": 8.0,
    "anchor_head_pct": 4.0,
    "tip_march_r": 3.0,
}


# ------------------------------------------------------------------------------------------------ io
def load_trunk(trunk_dir, name):
    tj = json.load(open(os.path.join(trunk_dir, name, "trunk.json")))
    tz = np.load(os.path.join(trunk_dir, name, "trunk.npz"))
    sh = tuple(int(x) for x in tz["shape"])
    t = {k: tz[k] for k in tz.files}
    t["occ"] = np.unpackbits(tz["occ"])[: int(np.prod(sh))].reshape(sh).astype(bool)
    t["R"] = np.asarray(tj["frame"]["rotation_scan_to_canonical"], float)
    t["o"] = np.asarray(tj["frame"]["origin_scan"], float)
    t["BL"] = float(tj["BL_um"])
    t["json"] = tj
    return t


def load_mesh(path):
    m = trimesh.load(path, process=True, force="mesh")
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda q: len(q.faces))
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


# ------------------------------------------------------------------------------------------------ grid
def fine_grid(Vw, F, trunk, N):
    BL = trunk["BL"]
    h = BL / N
    occ, lo = vox.voxelize(Vw, F, h)
    lo = lo + 0.5 * h  # centre of voxel 0
    tp, lo_c, pc = trunk["tpart"], np.asarray(trunk["lo"], float), float(trunk["pitch"])
    ijk = np.argwhere(occ)
    q = np.round((lo + ijk * h - lo_c) / pc).astype(int)
    ok = np.all((q >= 0) & (q < np.array(tp.shape)), axis=1)
    core = np.full(occ.shape, -1, np.int8)
    core[tuple(ijk[ok].T)] = tp[tuple(q[ok].T)]
    D = ndi.distance_transform_edt(occ).astype(np.float32) * h
    return occ, lo, h, core, D


def lookup(grid, lo, h, X, default=0.0):
    q = np.round((np.asarray(X) - lo) / h).astype(int)
    ok = np.all((q >= 0) & (q < np.array(grid.shape)), axis=1)
    out = np.full(len(q), default, dtype=float)
    out[ok] = grid[tuple(q[ok].T)]
    return out


# ------------------------------------------------------------------------------------------------ frame
def trunk_axis(trunk, core, lo, h):
    """Anterior direction (working frame) along the mesosoma, from the core labels (head vs gaster / waist), and the
    mesosoma centre. Independent of the working frame's axes."""
    cen = {}
    for k in range(4):
        m = core == k
        if m.any():
            cen[k] = lo + np.argwhere(m).mean(0) * h
    if 1 not in cen:
        raise RuntimeError("trunk product has no mesosoma core")
    meso = np.argwhere(core == 1)
    Pm = lo + meso * h
    c = Pm.mean(0)
    # principal axis of the mesosoma core, oriented towards the head (or away from the waist / gaster)
    w, U = np.linalg.eigh(np.cov((Pm - c).T))
    ax = U[:, -1]
    ref = None
    if 0 in cen:
        ref = cen[0] - c
    elif 3 in cen or 2 in cen:
        ref = c - cen.get(2, cen.get(3))
    if ref is not None:
        # blend the principal axis with the head - gaster direction (a short mesosoma can have a poor PCA axis)
        hg = (cen[0] if 0 in cen else c) - (cen[3] if 3 in cen else cen.get(2, c))
        if np.linalg.norm(hg) > 0:
            hg = hg / np.linalg.norm(hg)
            if np.dot(ax, hg) < 0:
                ax = -ax
            if abs(np.dot(ax, hg)) < 0.8:
                ax = hg
    return ax / np.linalg.norm(ax), c, cen


def symmetry_normal(core, lo, h, ax, n_pts=20000, step_deg=2.0, seed=0):
    """Lateral axis: the normal of the bilateral symmetry plane of the trunk core (which contains the anterior axis
    `ax`), found by reflecting the core voxels across planes rotated about `ax` and keeping the best match."""
    from scipy.spatial import cKDTree

    idx = np.argwhere(core >= 0)
    rng = np.random.default_rng(seed)
    if len(idx) > n_pts:
        idx = idx[rng.choice(len(idx), n_pts, replace=False)]
    P = lo + idx * h
    P = P - P.mean(0)
    tree = cKDTree(P)
    e1 = np.cross(ax, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(ax, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    best = None
    for th in np.radians(np.arange(0.0, 180.0, step_deg)):
        nrm = np.cos(th) * e1 + np.sin(th) * e2
        # the plane through the core's centroid; offset along the normal refined by the mean reflected mismatch
        Q = P - 2.0 * np.outer(P @ nrm, nrm)
        d, _ = tree.query(Q, workers=2)
        sc = float(np.mean(d))
        if best is None or sc < best[0]:
            best = (sc, nrm)
    return best[1], best[0]


def anatomical_frame(ax, c, cands, BL, core=None, lo=None, h=None):
    """Rows (X anterior, Y left, Z dorsal). The lateral axis is the normal of the bilateral symmetry plane of the
    head + mesosoma core; its sign (which side is ventral) is the side of the mesosoma where long tubes rooted on the mesosoma core
    attach (coxae attach ventrally; left and right cancel sideways; every candidate counts once). Without a core
    the attachment vote alone gives the ventral direction."""
    acc = np.zeros(3)
    for cd in cands:
        if cd["root_part"] != 1 or cd["free_arc"] < 0.15 * BL:
            continue
        v = cd["root_pos"] - c  # coxae attach ventrally on the mesosoma (leg postures vary, attachments do not)
        v = v - np.dot(v, ax) * ax
        if np.linalg.norm(v) > 1e-9:
            acc += v / np.linalg.norm(v)
    if np.linalg.norm(acc) < 1e-9:
        acc = np.array([0.0, 0.0, -1.0]) - np.dot([0.0, 0.0, -1.0], ax) * ax
    vent = acc / np.linalg.norm(acc)
    if core is not None:
        hm = np.where(np.isin(core, (0, 1)), core, -1).astype(np.int8)  # head + mesosoma: the most rigid part
        nrm, _ = symmetry_normal(hm if (hm >= 0).any() else core, lo, h, ax)
        zc = np.cross(ax, nrm)
        zc /= np.linalg.norm(zc)
        vent = zc if np.dot(zc, vent) > 0 else -zc
    Z = -vent
    Y = np.cross(Z, ax)
    Y /= np.linalg.norm(Y)
    Z = np.cross(ax, Y)
    return np.stack([ax, Y, Z])


# ------------------------------------------------------------------------------------------------ candidates
def make_candidates(tubes, BL, cfg):
    cands, frags = [], []
    for i, t in enumerate(tubes):
        P, r = t["P"], t["r"]
        if len(P) < 2:
            continue
        st, en = t["start"], t["end"]
        roots = [x for x in (st, en) if x["kind"] == "root"]
        if not roots:
            frags.append({"tube": i, "P": P, "r": r, "start": st, "end": en, "len": float(arclen(P)[-1])})
            continue
        if len(roots) == 2:
            # both ends on the trunk: the root is the end on the limb family core with the thicker tube
            s = arclen(P)
            Lw = min(0.05 * BL, 0.3 * s[-1])
            r0 = np.median(r[s <= Lw])
            r1 = np.median(r[s >= s[-1] - Lw])
            pref = lambda x: x["root_part"] in (0, 1)  # noqa: E731
            if pref(st) != pref(en):
                root_at_start = pref(st)
            else:
                root_at_start = r0 > r1
        else:
            root_at_start = st["kind"] == "root"
        if root_at_start:
            P, r = P[::-1], r[::-1]
            root, distal = st, en
        else:
            root, distal = en, st
        s = arclen(P)
        if s[-1] < cfg["cand_min_pct"] / 100 * BL:
            continue
        cands.append(
            {
                "tube": i,
                "P": P,
                "r": r,
                "s": s,
                "free_arc": float(s[-1]),
                "root_part": root["root_part"],
                "root_node": root["node"],
                "distal": distal,
                "root_pos": P[-1].copy(),
                "tip_pos": P[0].copy(),
            }
        )
    return cands, frags


# ------------------------------------------------------------------------------------------------ naming
def point_from_root(cd, d):
    s = cd["s"]
    sr = s[-1] - s
    t = min(d, 0.5 * s[-1])
    return np.array([np.interp(t, sr[::-1], cd["P"][::-1, k]) for k in range(3)])


def _seg_dist(P, X):
    """Distance of points P to polyline X."""
    if len(X) < 2:
        return np.linalg.norm(P - X[0], axis=1)
    a, ab = X[:-1], X[1:] - X[:-1]
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    u = np.clip(((P[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0.0, 1.0)
    return np.linalg.norm(P[:, None, :] - (a[None] + u[..., None] * ab[None]), axis=2).min(1)


def proximal_samples(cd, BL, lo_pct=2.0, hi_pct=12.0, n=12):
    s = cd["s"]
    sr = s[-1] - s
    t = np.linspace(lo_pct / 100 * BL, min(hi_pct / 100 * BL, 0.6 * s[-1]), n)
    return np.stack([np.interp(t, sr[::-1], cd["P"][::-1, k]) for k in range(3)], 1)


def whole_samples(cd, BL, step_pct=1.0):
    s = cd["s"]
    t = np.arange(0.0, s[-1], step_pct / 100 * BL)
    return np.stack([np.interp(t, s, cd["P"][:, k]) for k in range(3)], 1) if len(t) else cd["P"][:1]


def conflicts(cands, idx, BL, tol_pct=1.5, share=0.5, shared_roots=None):
    """Pairs of candidates that cannot both be limbs: duplicates of one limb (they share their proximal part -
    coxa / trochanter / femur base, or scape base - and most of their length) or routes ending at the same tip. Two
    candidates that share only the proximal part (two legs whose proximal completion entered the same coxa) do not
    conflict; they are recorded in `shared_roots` and flagged."""
    C = set()
    prox = {i: proximal_samples(cands[i], BL) for i in idx}
    stretch = {i: proximal_samples(cands[i], BL, 0.0, 16.0, 40) for i in idx}
    whole = {i: whole_samples(cands[i], BL) for i in idx}
    tol = tol_pct / 100 * BL
    for a in idx:
        for b in idx:
            if a >= b:
                continue
            fa = (_seg_dist(prox[a], stretch[b]) <= tol).mean()
            fb = (_seg_dist(prox[b], stretch[a]) <= tol).mean()
            tip = np.linalg.norm(cands[a]["P"][0] - cands[b]["P"][0]) <= 2 * tol
            short, other = (a, b) if cands[a]["free_arc"] <= cands[b]["free_arc"] else (b, a)
            wshare = (_seg_dist(whole[short], cands[other]["P"]) <= tol).mean()
            if (max(fa, fb) > share and wshare > share) or tip:
                C.add((a, b))
                C.add((b, a))
            elif max(fa, fb) > share and shared_roots is not None:
                shared_roots.setdefault(a, set()).add(b)
                shared_roots.setdefault(b, set()).add(a)
    return C


def quality(cd, BL, w_pen=2.0, w_ov=0.5):
    return min(cd["free_arc"], 1.3 * BL) - w_pen * cd.get("pen", 0.0) - w_ov * cd.get("overlap", 0.0) * cd["free_arc"]


def name_limbs(cands, Fr, c, BL, cfg):
    """Structural prior: six legs rooted on the mesosoma (side by where the proximal leg runs, fore / mid / hind by
    root order along the mesosoma), two antennae and up to two mandibles on the head. Candidates sharing their
    proximal part or tip conflict; the selection is greedy by quality (length minus turn penalties and overlap)."""
    to_a = lambda X: (np.asarray(X) - c) @ Fr.T  # noqa: E731  anatomical coords about the mesosoma centre
    out = {}
    notes = {}
    legs = [
        i for i, cd in enumerate(cands) if cd["root_part"] in (1, 2) and cd["free_arc"] >= cfg["leg_min_pct"] / 100 * BL
    ]
    for i in legs:
        cd = cands[i]
        pa = to_a(point_from_root(cd, 0.12 * BL))
        ra = to_a(cd["root_pos"])
        cd["side_y"] = float(pa[1])
        # order key: trochanter region along the body (the femur 30 % BL out breaks ties between legs whose
        # proximal completions entered the same coxa)
        cd["root_x"] = float(to_a(point_from_root(cd, 0.08 * BL))[0]) + 0.1 * float(
            to_a(point_from_root(cd, 0.30 * BL))[0]
        )
        cd["root_z"] = float(ra[2])
        cd["quality"] = quality(cd, BL)
    # legs attach ventrally and run out sideways: dorsal roots (spines, nodes) and routes hugging the midline
    # (surface routes over the body) are not legs
    notes["not_leg"] = {
        i: ("dorsal root" if cands[i]["root_z"] >= cfg["leg_dorsal_pct"] / 100 * BL else "on the midline")
        for i in legs
        if cands[i]["root_z"] >= cfg["leg_dorsal_pct"] / 100 * BL
        or abs(cands[i]["side_y"]) < cfg["leg_lateral_pct"] / 100 * BL
    }
    legs = [i for i in legs if i not in notes["not_leg"]]
    shared = {}
    C = conflicts(cands, legs, BL, shared_roots=shared)
    for i, js in shared.items():
        cands[i]["shares_root"] = sorted(js)
    # a candidate sharing its proximal part or tip with a cleaner one (length minus turn penalties) is dropped;
    # the rest are taken by length, at most three per side
    dom = {i for i in legs if any((i, j) in C and cands[j]["quality"] > cands[i]["quality"] for j in legs)}
    sel = {1: [], -1: []}
    for i in sorted([i for i in legs if i not in dom], key=lambda i: -cands[i]["free_arc"]):
        sd = 1 if cands[i]["side_y"] >= 0 else -1
        if len(sel[sd]) >= 3 or any((i, j) in C for j in sel[1] + sel[-1]):
            continue
        sel[sd].append(i)
    for side, lab in ((1, "L"), (-1, "R")):
        mine = sorted(sel[side], key=lambda i: -cands[i]["root_x"])
        if len(mine) == 3:
            for k, i in enumerate(mine):
                out[f"{lab}{k + 1}"] = i
            continue
        other = sorted(sel[-side], key=lambda i: -cands[i]["root_x"])
        if len(other) == 3:
            ref = [cands[i]["root_x"] for i in other]
        elif len(other) + len(mine) >= 3:
            allx = sorted([cands[i]["root_x"] for i in other + mine], reverse=True)
            ref = [allx[0], float(np.median(allx)), allx[-1]]
        else:
            ref = None
        if ref is None:
            for k, i in enumerate(mine):
                out[f"{lab}{k + 1}"] = i
            continue
        best = None
        import itertools

        for slots in itertools.permutations(range(3), len(mine)):
            if list(slots) != sorted(slots):
                continue  # keep the front-to-back order
            cst = sum(abs(ref[k] - cands[i]["root_x"]) for k, i in zip(slots, mine))
            if best is None or cst < best[0]:
                best = (cst, slots)
        for k, i in zip(best[1], mine):
            out[f"{lab}{k + 1}"] = i
    head = [i for i, cd in enumerate(cands) if cd["root_part"] == 0 and cd["free_arc"] >= cfg["man_min_pct"] / 100 * BL]
    for i in head:
        cd = cands[i]
        cd["side_y"] = float(to_a(point_from_root(cd, 0.04 * BL))[1])
        cd["root_a"] = to_a(cd["root_pos"])
        cd["quality"] = quality(cd, BL)
    Ch = conflicts(cands, head, BL)
    ants = []
    for i in sorted(
        [i for i in head if cands[i]["free_arc"] >= cfg["ant_min_pct"] / 100 * BL], key=lambda i: -cands[i]["quality"]
    ):
        if len(ants) < 2 and not any((i, j) in Ch for j in ants):
            ants.append(i)
    if len(ants) == 2:
        a, b = ants
        if cands[a]["side_y"] >= cands[b]["side_y"]:
            out["AL"], out["AR"] = a, b
        else:
            out["AL"], out["AR"] = b, a
    elif len(ants) == 1:
        out["AL" if cands[ants[0]]["side_y"] > 0 else "AR"] = ants[0]
    mans = [
        i
        for i in head
        if i not in ants
        and cands[i]["free_arc"] <= cfg["man_max_pct"] / 100 * BL
        and not any((i, j) in Ch for j in ants)
    ]
    for side, lab in ((1, "ML"), (-1, "MR")):
        mine = [i for i in mans if np.sign(cands[i]["side_y"]) == side]
        if mine:
            out[lab] = max(mine, key=lambda i: cands[i]["quality"])
    return out, notes


# ------------------------------------------------------------------------------------------------ polylines
def resample(P, step):
    s = arclen(P)
    if s[-1] < step:
        return P.copy()
    q = np.linspace(0, s[-1], int(np.ceil(s[-1] / step)) + 1)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(3)], 1)


def geodesic_from_core(occ, core, part):
    """Geodesic distance (voxels, unit cost, 26-connected) inside the solid from the boundary of trunk core `part`,
    the other trunk cores impassable. Used to pick the limb's tip (the free end furthest from the body, the truth
    convention for claws and clubs)."""
    from skimage.graph import MCP_Geometric

    src = core == part
    if not src.any():
        return None
    sb = np.argwhere(src & ~ndi.binary_erosion(src))
    cost = np.where(occ & ~((core >= 0) & ~src), 1.0, np.inf)
    g, _ = MCP_Geometric(cost, fully_connected=True).find_costs([tuple(x) for x in sb])
    return g.astype(np.float32)


def refine_tip(P, occ, lo, h, D, BL, Gd):
    """Within a ball around the skeleton end (thinning stops short of blunt ends and picks an arbitrary claw), the
    voxel of the solid connected to the end inside the ball with the largest geodesic distance from the body."""
    if Gd is None:
        return None
    r_tip = float(lookup(D, lo, h, P[:1])[0])
    rb = max(3.0 * r_tip + 3 * h, 0.04 * BL)
    c = P[0]
    i0 = np.maximum(np.floor((c - rb - lo) / h).astype(int), 0)
    i1 = np.minimum(np.ceil((c + rb - lo) / h).astype(int) + 1, np.array(occ.shape))
    sl = tuple(slice(a, b) for a, b in zip(i0, i1))
    sub = occ[sl].copy()
    ijk = np.argwhere(sub)
    X = lo + (ijk + i0) * h
    far = np.linalg.norm(X - c, axis=1) > rb
    sub[tuple(ijk[far].T)] = False
    q = np.round((c - lo) / h).astype(int) - i0
    if not (np.all(q >= 0) and np.all(q < np.array(sub.shape))) or not sub[tuple(q)]:
        return None
    lab, _ = ndi.label(sub, structure=np.ones((3, 3, 3)))
    comp = np.argwhere(lab == lab[tuple(q)])
    g = Gd[sl][tuple(comp.T)]
    g = np.where(np.isfinite(g), g, -1)
    k = int(np.argmax(g))
    if g[k] <= Gd[sl][tuple(q)] + 1:
        return None
    return lo + (comp[k] + i0) * h


def finalize(cd, occ, lo, h, D, core_centroid, BL, cfg, Gd=None):
    """Tip -> root polyline: resampled, smoothed, tip marched along its tangent to the surface, root extended
    straight towards the core centroid. Returns X, s, r, core_arc."""
    P0 = cd["P"]
    if cd.get("distal", {}).get("kind", "tip") == "tip" and len(P0) >= 3:
        t = refine_tip(P0, occ, lo, h, D, BL, Gd)
        if t is not None and np.linalg.norm(t - P0[0]) > h:
            P0 = np.vstack([t, P0])
    P = resample(P0, h / 2)
    if len(P) >= 5:
        P = ndi.gaussian_filter1d(P, 3.0, axis=0, mode="nearest")
        P[0] = resample(P0, h / 2)[0]  # keep the refined tip on the surface
    s = arclen(P)
    k = min(int(np.searchsorted(s, 2 * h)), len(P) - 1)
    u = P[0] - P[k]
    if np.linalg.norm(u) > 0:
        u /= np.linalg.norm(u)
        ext = []
        r_tip = float(lookup(D, lo, h, P[:1])[0])
        for t in np.arange(h / 2, max(8 * h, cfg["tip_march_r"] * r_tip), h / 2):
            q = P[0] + t * u
            ext.append(q)
            if lookup(occ, lo, h, q[None])[0] < 0.5:
                break
        if ext:
            P = np.vstack([np.array(ext)[::-1], P])
    core_arc = float(arclen(P)[-1])
    if core_centroid is not None:
        d = core_centroid - P[-1]
        L = min(cfg["ext_pct"] / 100 * BL, 0.85 * float(np.linalg.norm(d)))
        if L > h / 2:
            m = int(np.ceil(L / (h / 2)))
            P = np.vstack([P, P[-1] + np.outer(np.arange(1, m + 1) * (L / m), d / np.linalg.norm(d))])
    s = arclen(P)
    r = lookup(D, lo, h, P)
    return P, s, r, core_arc


# ------------------------------------------------------------------------------------------------ status, confidence
PAIR = {
    "L1": "R1",
    "L2": "R2",
    "L3": "R3",
    "R1": "L1",
    "R2": "L2",
    "R3": "L3",
    "AL": "AR",
    "AR": "AL",
    "ML": "MR",
    "MR": "ML",
}


def limb_features(cd, X, s, r, core_arc, BL):
    """Evidence measured without truth (all lengths in % BL)."""
    pct = 100.0 / BL
    rr = np.asarray(cd["r"], float)
    k = max(1, len(rr) // 10)
    return {
        "free_arc_pct": round(core_arc * pct, 2),
        "tip_r_pct": round(float(np.median(rr[:k])) * pct, 3),
        "max_r_pct": round(float(np.percentile(rr, 95)) * pct, 3),
        "turn_penalty_pct": round(cd.get("pen", 0.0) * pct, 2),
        "overlap": round(float(cd.get("overlap", 0.0)), 3),
        "secondary": bool(cd.get("secondary", False)),
        "side_y_pct": round(cd.get("side_y", 0.0) * pct, 2),
        "root_z_pct": round(cd.get("root_z", np.nan) * pct, 2) if "root_z" in cd else None,
        "root_part": int(cd["root_part"]),
        "anchor_arc_pct": round(cd["proximal"]["anchor_arc"] * pct, 2) if cd.get("proximal") else None,
        "near_body": round(cd.get("near_body", np.nan), 3),
        "reach_pct": round(cd.get("reach_pct", np.nan), 2),
        "shares_root": bool(cd.get("shares_root")),
    }


def assess(limbs, BL, cfg):
    """Status and confidence from evidence measured without truth. Confidence = 0.95 x exp(-turn penalty / a) x
    exp(-|ln(arc / partner arc)| / b) (a missing partner counts as |ln| = 0.5): a route that turned through contacts,
    or a limb whose length disagrees with its bilateral partner, is less likely to be the right whole limb. a and b
    were chosen leave-one-specimen-out on the dev set (pooled held-out AUC 0.74). A limb
    much shorter than its partner (< pair_short x) is 'partial' (its distal part is probably lost in a contact or
    folded away); limbs without any candidate are 'no_path'. Other evidence (overlap with other limbs, secondary
    route, side signal, free-end radius) is recorded in `features` and `flags` but did not predict errors on dev."""
    for L, v in limbs.items():
        if "features" not in v:
            v.update(
                {
                    "status": "no_path",
                    "confidence": 0.0,
                    "evidence": "no candidate limb route found",
                    "missing": ["whole limb (no route from a free tip to the trunk could be given this name)"],
                }
            )
            continue
        f = v["features"]
        miss, flags = [], []
        P = limbs.get(PAIR[L], {}).get("features")
        ratio = None
        if P:
            ratio = f["free_arc_pct"] / max(P["free_arc_pct"], 1e-6)
            if ratio < cfg["pair_short"]:
                miss.append(f"distal part (free arc {ratio:.2f} x its partner {PAIR[L]})")
        pair_dev = abs(np.log(ratio)) if ratio else 0.5
        conf = (
            0.95
            * float(np.exp(-f["turn_penalty_pct"] / cfg["conf_pen_pct"]))
            * float(np.exp(-pair_dev / cfg["conf_pair"]))
        )
        if f["turn_penalty_pct"] > 2.0:
            flags.append("route turns through junctions (contacts)")
        if ratio is not None and not 0.8 <= ratio <= 1.25:
            flags.append(f"length differs from partner ({ratio:.2f})")
        if P is None:
            flags.append("no bilateral partner found")
        if f["overlap"] > 0:
            flags.append(f"shares {f['overlap']:.2f} of its length with another limb")
        if f["secondary"]:
            flags.append("secondary route (left out by the joint selection)")
        if f.get("shares_root"):
            flags.append("proximal part shared with another leg candidate (coxa / trochanter uncertain)")
        ev = (
            f"free arc {f['free_arc_pct']:.1f} % BL, turn penalty {f['turn_penalty_pct']:.1f} % BL"
            + ("" if ratio is None else f", arc / partner {ratio:.2f}")
            + (f"; flags: {', '.join(flags)}" if flags else "")
        )
        v.update(
            {
                "status": "partial" if miss else "path",
                "confidence": round(conf, 3),
                "evidence": ev,
                "missing": miss,
                "flags": flags,
            }
        )


# ------------------------------------------------------------------------------------------------ main
def run(name, trunk_dir, mesh_path, out_dir, cfg=None, debug=False):
    cfg = dict(CFG, **(cfg or {}))
    t0 = time.time()
    trunk = load_trunk(trunk_dir, name)
    R, o, BL = trunk["R"], trunk["o"], trunk["BL"]
    V, F = load_mesh(mesh_path)
    Vw = (V - o) @ R.T
    occ, lo, h, core, D = fine_grid(Vw, F, trunk, cfg["N_VOX"])
    G, sk = limbgraph.build(occ, core, lo, h, D)
    limbgraph.prune_spurs(G, BL, cfg["prune_c_r"], cfg["prune_c_bl"])
    cl, internal = tb.clusters(G, cfg["c_short"])
    J = tb.pair_junctions(G, cl, internal, BL, cfg["max_turn"], cfg["max_ratio"])
    T = tb.extract_tubes(G, cl, internal, J)
    if cfg["mode"] == "route":
        router = rt.Router(G, cl, internal, J, BL, cfg["lam_turn"], cfg["mu_rad"], cfg["rad_free"])
        if cfg["select"] == "flow":
            tips_ = rt.tips_of(G, internal)
            routes, claimed = rt.flow_routes(router, tips_, BL, kappa=cfg["kappa"], min_len_pct=cfg["cand_min_pct"])
            if cfg["secondary"]:
                routes = rt.add_secondary(
                    router,
                    routes,
                    claimed,
                    tips_,
                    BL,
                    block_factor=cfg["block_factor"],
                    min_len_pct=cfg["cand_min_pct"],
                    spur_max_pct=cfg["spur_max_pct"],
                )
        else:
            routes, claimed = rt.claim_routes(
                router,
                rt.tips_of(G, internal),
                BL,
                block_factor=cfg["block_factor"],
                min_len_pct=cfg["cand_min_pct"],
                spur_max_pct=cfg["spur_max_pct"],
            )
        cands, frags = [], []
        for ro in routes:
            P, r = ro["P"], ro["r"]
            s_ = arclen(P)
            cands.append(
                {
                    "tube": ro["id"],
                    "P": P,
                    "r": r,
                    "s": s_,
                    "free_arc": float(s_[-1]),
                    "root_part": ro["root_part"],
                    "root_node": ro["root_node"],
                    "distal": {"kind": "tip"},
                    "root_pos": P[-1].copy(),
                    "tip_pos": P[0].copy(),
                    "pen": ro["pen"],
                    "overlap": ro["overlap"],
                }
            )
        cps = {}
        for cd in cands:
            fam = 1 if cd["root_part"] in (1, 2) else 0
            if fam not in cps:
                try:
                    cps[fam] = proximal.CorePaths(occ, core, lo, h, D, fam, BL)
                except RuntimeError:
                    cps[fam] = None
            cp = cps[fam]
            if cp is None:
                continue
            anc = (cfg["anchor_leg_pct"] if fam == 1 else cfg["anchor_head_pct"]) / 100 * BL
            dc = cp.dist(cd["P"])
            k = np.where(dc <= anc)[0]
            if not len(k):
                continue
            k = int(k[0])
            tr_ = cp.trace(cd["P"][k])
            if tr_ is None or len(tr_) < 1:
                continue
            P = np.vstack([cd["P"][: k + 1], tr_[1:]]) if len(tr_) > 1 else cd["P"][: k + 1]
            cd["P"], cd["r"] = P, lookup(D, lo, h, P)
            cd["s"] = arclen(P)
            cd["free_arc"] = float(cd["s"][-1])
            cd["root_pos"] = P[-1].copy()
            cd["proximal"] = {"anchor_arc": float(cd["s"][k]), "trace_n": int(len(tr_))}
            cd["core_centroid"] = cp.centroid
    else:
        cands, frags = make_candidates(T, BL, cfg)
    # how much of each candidate lies on the body surface (within near_body_pct of the trunk core): routes over the
    # body (gaster, mesosoma flanks) are not limbs
    distc = ndi.distance_transform_edt(core < 0).astype(np.float32) * h
    for cd in cands:
        dd = lookup(distc, lo, h, cd["P"])
        sr = cd["s"][-1] - cd["s"]
        m = sr > 0.08 * BL  # leave out the proximal part (coxa / scape base at the body)
        cd["near_body"] = float(np.mean(dd[m] < cfg["near_body_pct"] / 100 * BL)) if m.any() else 1.0
        cd["reach_pct"] = float(dd.max() / BL * 100)
    ax, c, cen = trunk_axis(trunk, core, lo, h)
    Fr = anatomical_frame(ax, c, cands, BL, core, lo, h)
    names, notes = name_limbs(cands, Fr, c, BL, cfg)
    # ---- output
    arrays, limbs = {}, {}
    Gds = {}
    to_scan = lambda X: np.asarray(X) @ R + o  # noqa: E731
    for L in LIMBS:
        if L not in names:
            limbs[L] = {"status": "no_path", "confidence": 0.0, "evidence": "no candidate"}
            continue
        cd = cands[names[L]]
        part = 1 if L in LEGS else 0
        if part not in Gds:
            Gds[part] = geodesic_from_core(occ, core, part)
        X, s, r, core_arc = finalize(cd, occ, lo, h, D, cd.get("core_centroid", cen.get(part)), BL, cfg, Gds[part])
        arrays[L + "_X"], arrays[L + "_s"], arrays[L + "_r"] = to_scan(X), s, r
        limbs[L] = {
            "core_arc_um": core_arc,
            "n_points": int(len(X)),
            "arc_um": float(s[-1]),
            "tip_scan": to_scan(X[0]).tolist(),
            "root_scan": to_scan(X[-1]).tolist(),
            "features": limb_features(cd, X, s, r, core_arc, BL),
        }
    assess(limbs, BL, cfg)
    for k in ("T_X", "T_s", "T_r", "T_part"):
        if k in trunk:
            arrays[k] = trunk[k]
    Fs = Fr @ R  # anatomical rows in scan coordinates: x_anat = Fr @ (x_w - c) = Fr @ R @ (x_scan - o) - Fr @ c
    origin_scan = c @ R + o
    out = {
        "name": name,
        "units": "um",
        "version": VERSION,
        "source": "curve-skeleton graph (PATHS-B)",
        "frame": {
            "rotation_scan_to_canonical": Fs.tolist(),
            "origin_scan": origin_scan.tolist(),
            "note": "anatomical frame from the trunk core labels and the leg roots (not the trunk product's working frame)",
        },
        "BL_um": BL,
        "BL_source": trunk["json"].get("BL_source", "trunk product"),
        "limbs": limbs,
        "other_candidates": [
            {
                "tube": cd["tube"],
                "root_part": int(cd["root_part"]),
                "free_arc_pct": round(cd["free_arc"] / BL * 100, 1),
                "distal": cd["distal"]["kind"],
            }
            for i, cd in enumerate(cands)
            if i not in names.values()
        ],
        "n_fragments": len(frags),
        "seconds": round(time.time() - t0, 1),
        "cfg": cfg,
    }
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "limbs.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(d, "limbs.npz"), **arrays)
    if debug:
        return out, dict(
            Gds=Gds,
            occ=occ,
            lo=lo,
            h=h,
            core=core,
            D=D,
            G=G,
            J=J,
            T=T,
            cands=cands,
            frags=frags,
            names=names,
            Fr=Fr,
            c=c,
            cen=cen,
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trunk-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--names", default=None, help="comma-separated (default: every specimen in --trunk-dir)")
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    names = (
        a.names.split(",")
        if a.names
        else sorted(d for d in os.listdir(a.trunk_dir) if os.path.isfile(os.path.join(a.trunk_dir, d, "trunk.json")))
    )
    if a.skip_existing:
        names = [n for n in names if not os.path.isfile(os.path.join(a.out_dir, n, "limbs.json"))]
    jobs = [(n, a.trunk_dir, os.path.join(a.mesh_root, n, f"{n}_envelope.stl"), a.out_dir) for n in names]
    if a.workers > 1:
        from multiprocessing import Pool

        with Pool(min(a.workers, 4)) as p:
            for r in p.imap_unordered(_job, jobs):
                print(r, flush=True)
    else:
        for j in jobs:
            print(_job(j), flush=True)


def _job(j):
    n, td, mp, od = j
    try:
        out = run(n, td, mp, od)
        st = {L: v["status"] for L, v in out["limbs"].items()}
        return f"{n} {out['seconds']}s " + " ".join(f"{L}:{v[0]}" for L, v in st.items())
    except Exception as e:
        d = os.path.join(od, n)
        os.makedirs(d, exist_ok=True)
        json.dump(
            {
                "name": n,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
                "limbs": {L: {"status": "failed"} for L in LIMBS},
            },
            open(os.path.join(d, "limbs.json"), "w"),
            indent=1,
        )
        return f"{n} FAILED {type(e).__name__}: {e}"


if __name__ == "__main__":
    main()
