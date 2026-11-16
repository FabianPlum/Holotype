"""Canonical orientation of AntScan worker envelopes (Stage 2a): head +X, animal's left +Y, dorsal +Z, det +1.

The longest axis is not enough: PCA leaves head/gaster and dorsal/ventral undecided, long legs
(Cataglyphis) or a curled gaster tilt or even dominate it. This module decides the frame from anatomy and says how
sure it is, so that doubtful specimens get rendered and looked at instead of silently accepted.

Pipeline
 1. occupancy grid of the watertight envelope (column parity, exact for closed surfaces; pitch = V^(1/3) / 55,
    i.e. ~210 voxels along the body of a typical worker, independent of how far the legs stick out),
 2. body core vs appendages: a ball opening with radius r = OPEN_RADIUS_FRAC * V^(1/3) removes what is thinner
    than 2r (legs, antennae, mandibles, hairs, spines); head, mesosoma, waist and gaster stay. The opening also
    strips a thin surface layer in concave corners (between the coxae), which glues neighbouring appendages
    together: appendage components are therefore found after a 1-voxel erosion and the stripped voxels are given
    back to the nearest surviving component,
 3. appendage components with their attachment blobs on the core, volume, reach and tip,
 4. frame from anatomical cues, each with a signed margin:
      long axis   principal axis of the body core, or, with >= 3 coxa attachments, the axis through the coxa
                  attachment points (the mesosoma = OmniAnt's kinematic root), independent of head/gaster posture
      dorsal      coxae attach on the ventral mesosoma (offset of the attachments from the local core centre line);
                  leg tips lie ventrally ("legs beneath the body")
      anterior    (a) the coxae sit in the anterior half of the core,
                  (b) antennae and mandibles attach in front of the fore coxae, nothing thin attaches behind the
                      hind coxae (sting aside),
                  (c) hind legs are longer/bulkier than fore legs,
                  (d) the waist (petiole) is a deeper constriction than the neck,
                  (e) the core thickness profile matched to the OmniAnt template, forward vs reversed,
 5. confidence per decision; disagreeing cues or small margins -> flags and status "review".

Rotation convention: R rows = canonical axes in scan coordinates, x_canonical = R @ (x_scan - origin_scan).
Units: input in micrometres, never rescaled.

Run (pytorch3d env, repo root):
  python -m custom_processing.antscan_align <envelope.stl> <out_dir> [--write-mesh]  # -> <name>_frame.json
"""

import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(os.path.dirname(HERE), "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")

VERSION = "2026-11-09h"
VOX_PER_CBRT_V = 55.0  # grid pitch = V^(1/3) / this (~210 voxels along the body of a typical worker)
GRID_MAX = 320  # cap on the grid size along the longest extent (memory)
OPEN_RADIUS_FRAC = 0.09  # ball-opening radius in units of V^(1/3) (template: legs/antennae vanish, waist is cut)
SPLIT_EROSION = 1.0  # erosion radius (voxels) used to separate appendages glued by stripped surface layers
MIN_APPENDAGE_FRAC = 0.001  # components below this fraction of the solid volume are sculpture, hairs, spines
END_EXCLUSION = 0.10  # attachments closer than this (x core length) to an end of the core are not coxae
LEG_CLUSTER_GAP = 0.20  # coxa attachments further apart than this (x core length) belong to different clusters
MIN_LEG_REACH = 0.15  # a leg reaches at least this far (x core length) from its attachment; spines and hairs do not (0.10 scored worse)
MIN_END_REACH = 0.12  # thin parts at the core ends count as antennae/mandibles only when at least this long
MIN_COXA_FOOTPRINT = (
    0.03  # a coxa/femur footprint on the core is at least (this x core length)^2 voxels; scapes and hairs are smaller
)
S26 = np.ones((3, 3, 3), bool)
# Roll-decision variants, ablated on the labelled real set (47 workers with frames judged by eye, 2026-11-09):
# whole-core symmetry + coxae-and-feet sign (defaults) 42/47 heads, 38/45 dorsals; mesosoma-only symmetry, a coxa
# footprint filter and an attachments-only sign each scored lower (39-41 / 34-37). Kept as switches for re-testing.
USE_COXA_FOOTPRINT = False
USE_MESOSOMA_SYMMETRY = False
SIGN_MODE = "mixed"  # "mixed" (coxae + feet together) or "attachments" (coxae decide, feet/mass break ties)


# ----------------------------------------------------------------------------------------------- geometry


def load_mesh(path):
    import trimesh

    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def mesh_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def occupancy(v, f, pitch, pad=3):
    """Solid voxels enclosed by the closed surface (v, f): parity of surface crossings along z columns.

    Voxel k spans [k, k+1) grid units with its centre at k + 0.5; a crossing at height z is counted for all voxel
    centres above it. Pixel centres get an irrational sub-voxel offset so that none sits exactly on a shared edge.
    """
    import trimesh

    origin = v.min(0) - (pad + np.array([0.3819660, 0.2360680, 0.1234568])) * pitch
    g = (v - origin) / pitch
    dims = np.ceil(g.max(0)).astype(int) + pad + 1
    g, f = trimesh.remesh.subdivide_to_size(g, f, max_edge=3.0, max_iter=10)  # short candidate loops below
    tri = g[f]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    d = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep = np.abs(d) > 1e-12  # zero projected area: the column runs along the face, no crossing
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]
    lo = np.ceil(np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2]) - 0.5).astype(int)
    hi = np.floor(np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2]) - 0.5).astype(int)
    span = hi - lo
    cnt = np.zeros(dims, dtype=np.int32)
    for di in range(int(span[:, 0].max()) + 1):
        for dj in range(int(span[:, 1].max()) + 1):
            sel = (span[:, 0] >= di) & (span[:, 1] >= dj)
            if not sel.any():
                continue
            i, j = lo[sel, 0] + di, lo[sel, 1] + dj
            px, py = i + 0.5, j + 0.5
            A, B, C, D = a[sel], b[sel], c[sel], d[sel]
            w0 = ((B[:, 0] - px) * (C[:, 1] - py) - (B[:, 1] - py) * (C[:, 0] - px)) / D
            w1 = ((C[:, 0] - px) * (A[:, 1] - py) - (C[:, 1] - py) * (A[:, 0] - px)) / D
            w2 = 1.0 - w0 - w1
            ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not ins.any():
                continue
            z = w0[ins] * A[ins, 2] + w1[ins] * B[ins, 2] + w2[ins] * C[ins, 2]
            kz = np.clip(np.ceil(z - 0.5).astype(int), 0, dims[2] - 1)
            np.add.at(cnt, (i[ins], j[ins], kz), 1)
    occ = (np.cumsum(cnt, axis=2) & 1).astype(bool)
    return occ, origin


def ball_open(mask, r):
    """Morphological opening with a Euclidean ball of radius r voxels (exact, via two distance transforms)."""
    if r <= 0:
        return mask.copy()
    eroded = ndi.distance_transform_edt(mask) > r
    if not eroded.any():
        return eroded
    return ndi.distance_transform_edt(~eroded) <= r


def principal_axes(pts):
    c = pts.mean(0)
    w, e = np.linalg.eigh(np.cov((pts - c).T))
    order = np.argsort(w)[::-1]
    return c, e[:, order].T, w[order]  # rows = axes, descending variance


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


# ----------------------------------------------------------------------------------------------- parts


def label_appendages(app, erosion=SPLIT_EROSION):
    """Connected appendage components, separated where only thin bridges join them.

    Components are labelled after an erosion by `erosion` voxels (bridges thinner than ~2*erosion disappear), then
    every removed appendage voxel is handed to the nearest surviving component.
    """
    if erosion <= 0:
        return ndi.label(app, structure=S26)
    core_app = ndi.distance_transform_edt(app) > erosion
    lab, n = ndi.label(core_app, structure=S26)
    if n == 0:
        return ndi.label(app, structure=S26)
    _, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
    full = lab[tuple(idx)]
    full[~app] = 0
    return full, n


def split_core_and_appendages(v, f, pitch=None, open_frac=OPEN_RADIUS_FRAC):
    """Occupancy, body core (opened) and appendage components with attachment information. Coordinates in voxels."""
    vol_mesh = mesh_volume(v, f)
    extent = v.max(0) - v.min(0)
    if pitch is None:
        pitch = max(vol_mesh ** (1 / 3) / VOX_PER_CBRT_V, float(extent.max()) / GRID_MAX)
    occ, origin = occupancy(v, f, pitch)
    n_solid = int(occ.sum())
    vol = n_solid * pitch**3
    r_vox = open_frac * vol ** (1 / 3) / pitch
    core = ball_open(occ, r_vox)
    if not core.any():
        raise RuntimeError("opening removed everything (mesh thinner than the opening diameter?)")
    core_lab, n_core = ndi.label(core, structure=S26)
    app = occ & ~core
    lab, n = label_appendages(app)
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(app, lab, idx) if n else np.zeros(0)
    min_size = MIN_APPENDAGE_FRAC * n_solid
    near_core = ndi.binary_dilation(core, structure=S26) & app
    comps = []
    for k in idx[sizes >= min_size]:
        m = lab == k
        pts = np.argwhere(m).astype(np.float64)
        blobs = []
        bl, nb = ndi.label(m & near_core, structure=S26)
        for q in range(1, nb + 1):
            p = np.argwhere(bl == q).astype(np.float64)
            blobs.append({"n": int(len(p)), "centre": p.mean(0)})
        blobs.sort(key=lambda d: -d["n"])
        att_c = blobs[0]["centre"] if blobs else pts.mean(0)
        dist = np.linalg.norm(pts - att_c, axis=1)
        comps.append(
            {
                "label": int(k),
                "n": int(len(pts)),
                "volume_frac": float(len(pts) / n_solid),
                "attached": bool(blobs),
                "attach": att_c,
                "attach_blobs": blobs,
                "tip": pts[np.argmax(dist)],
                "reach": float(dist.max()),
                "centroid": pts.mean(0),
            }
        )
    comps.sort(key=lambda d: -d["n"])
    return {
        "occ": occ,
        "core": core,
        "core_labels": core_lab,
        "n_core_components": int(n_core),
        "app_labels": lab,
        "pitch": pitch,
        "origin": origin,
        "open_radius_vox": float(r_vox),
        "components": comps,
        "n_small_dropped": int((sizes < min_size).sum()),
        "volume_um3": float(vol),
        "mesh_volume_um3": vol_mesh,
    }


def core_profile(core_pts, c0, e1, bin_vox=1.0):
    """Per-slice centroid and equivalent radius along the axis e1 (slices of bin_vox thickness)."""
    s = (core_pts - c0) @ e1
    lo, hi = s.min(), s.max()
    nb = max(int(np.ceil((hi - lo) / bin_vox)), 1)
    b = np.clip(((s - lo) / (hi - lo + 1e-9) * nb).astype(int), 0, nb - 1)
    cnt = np.bincount(b, minlength=nb).astype(float)
    cx = np.stack([np.bincount(b, weights=core_pts[:, i], minlength=nb) for i in range(3)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = cx / cnt[:, None]
    s_mid = lo + (np.arange(nb) + 0.5) * (hi - lo) / nb
    return {"s": s_mid, "count": cnt, "centroid": cen, "r_eq": np.sqrt(cnt / np.pi), "s_min": lo, "s_max": hi}


def slice_centroid(prof, s):
    j = int(np.clip(np.searchsorted(prof["s"], s), 0, len(prof["s"]) - 1))
    ok = np.where(prof["count"] > 0)[0]
    if prof["count"][j] <= 0:
        j = ok[np.argmin(np.abs(ok - j))]
    return prof["centroid"][j], prof["r_eq"][j]


# ----------------------------------------------------------------------------------------------- template

_TEMPLATE = {}


def template_core():
    """OmniAnt template body core (same opening as the specimens): voxel grid, centroid, core length along +X
    (the template frame is the canonical frame: +X head, +Z dorsal) and a subsample of its voxels."""
    if "core" not in _TEMPLATE:
        dd = pickle.load(open(MODEL, "rb"), encoding="latin1")
        v, f = np.asarray(dd["v_template"], np.float64), np.asarray(dd["f"], np.int64)
        parts = split_core_and_appendages(v, f)
        pts = np.argwhere(parts["core"]).astype(np.float64)
        c = pts.mean(0)
        L = float(np.ptp(pts[:, 0]))
        sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 100_000), replace=False)]
        _TEMPLATE.update(
            core=parts["core"], centre=c, L=L, pts=sub, profile=core_profile(pts, c, np.array([1.0, 0, 0]))
        )
    return _TEMPLATE


def template_profile(n_bins=64):
    """Equivalent-radius profile of the OmniAnt template body core along +X (head first), mean-normalised."""
    if "profile_resampled" not in _TEMPLATE:
        _TEMPLATE["profile_resampled"] = resample(template_core()["profile"]["r_eq"], n_bins)
    return _TEMPLATE["profile_resampled"]


def template_shape_scores(core, pts, c0, L, X, Z):
    """Overlap of the specimen's body core with the template core under the four sign hypotheses
    (head +-X, dorsal +-Z), after normalising both by their core length. Returns {(sx, sz): score} with
    score = harmonic mean of (specimen voxels inside the template core) and (template voxels inside the specimen
    core); the comparison is a whole-shape check of head/gaster proportions, the mesosoma hump and the ventral
    coxal side, independent of the appendage segmentation."""
    T = template_core()
    tc, tL, tpts = T["core"], T["L"], T["pts"]
    tshape, shape = np.array(tc.shape), np.array(core.shape)
    out = {}
    q_t = (tpts - T["centre"]) / tL  # template voxels, normalised canonical coords
    for sx in (1, -1):
        for sz in (1, -1):
            Xh, Zh = sx * X, sz * Z
            R = np.stack([Xh, np.cross(Zh, Xh), Zh])
            q = ((pts - c0) @ R.T) / L  # specimen core in normalised canonical coords
            g = np.rint(q * tL + T["centre"]).astype(int)
            ok = np.all((g >= 0) & (g < tshape), axis=1)
            prec = float(tc[g[ok, 0], g[ok, 1], g[ok, 2]].sum() / max(len(g), 1))
            h = np.rint(c0 + (q_t * L) @ R).astype(int)
            ok = np.all((h >= 0) & (h < shape), axis=1)
            rec = float(core[h[ok, 0], h[ok, 1], h[ok, 2]].sum() / max(len(h), 1))
            out[(sx, sz)] = 2 * prec * rec / max(prec + rec, 1e-9)
    return out


def resample(y, n):
    y = np.asarray(y, float)
    out = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y)
    return out / max(out.mean(), 1e-9)


# ----------------------------------------------------------------------------------------------- frame


def cluster_1d(values, gap):
    """Split 1-D values into clusters separated by more than gap. Returns a list of index arrays."""
    order = np.argsort(values)
    groups, cur = [], [order[0]]
    for i0, i1 in zip(order[:-1], order[1:]):
        if values[i1] - values[i0] > gap:
            groups.append(np.array(cur))
            cur = [i1]
        else:
            cur.append(i1)
    groups.append(np.array(cur))
    return groups


def perp_offset(prof, p, s, e1):
    """Offset of point p from the core centre line at axial position s, perpendicular to e1, and the local radius."""
    cen, r_loc = slice_centroid(prof, s)
    d = p - cen
    return d - (d @ e1) * e1, r_loc


def symmetry_plane(core, c0, e1, axes, n_max=150_000, step_deg=3.0, seed=0):
    """Roll angle of the sagittal plane: the plane through the axis e1 about which the body core is most mirror
    symmetric. Returns the in-plane unit vector t (dorsal axis up to sign), the best overlap score, the score of
    the perpendicular plane and the fine-step angle.

    Ants are bilaterally symmetric; the coxae then only have to decide the sign of t, which removes the lateral
    bias of a mean-offset estimate when legs are missing or merged on one side.
    """
    pts = np.argwhere(core).astype(np.float64)
    if len(pts) > n_max:
        pts = pts[np.random.default_rng(seed).choice(len(pts), n_max, replace=False)]
    d = pts - c0
    s, u, w = d @ e1, d @ axes[1], d @ axes[2]
    base = c0 + s[:, None] * e1
    shape = np.array(core.shape)

    def score(phi):
        c, sn = np.cos(phi), np.sin(phi)
        proj = u * c + w * sn
        u2, w2 = 2 * proj * c - u, 2 * proj * sn - w
        q = np.rint(base + u2[:, None] * axes[1] + w2[:, None] * axes[2]).astype(int)
        ok = np.all((q >= 0) & (q < shape), axis=1)
        hit = np.zeros(len(q), bool)
        hit[ok] = core[q[ok, 0], q[ok, 1], q[ok, 2]]
        return float(hit.mean())

    coarse = np.radians(np.arange(0.0, 180.0, step_deg))
    sc = np.array([score(p) for p in coarse])
    k = int(np.argmax(sc))
    fine = coarse[k] + np.radians(np.arange(-step_deg, step_deg + 0.25, 0.5))
    sf = np.array([score(p) for p in fine])
    phi = fine[int(np.argmax(sf))]
    perp = score(phi + np.pi / 2)
    t = np.cos(phi) * axes[1] + np.sin(phi) * axes[2]
    return t, float(sf.max()), perp, float(np.degrees(phi))


def decide_frame(parts):
    """Anatomical cues -> rotation (rows = canonical axes in grid coordinates), confidences, flags."""
    core, comps = parts["core"], parts["components"]
    pts = np.argwhere(core).astype(np.float64)
    c0, axes, var = principal_axes(pts)
    e1 = axes[0]
    prof = core_profile(pts, c0, e1)
    L = prof["s_max"] - prof["s_min"]
    s_mid = 0.5 * (prof["s_min"] + prof["s_max"])
    flags, cues = [], {}
    attached = [c for c in comps if c["attached"]]
    for c in attached:
        c["s"] = float((c["attach"] - c0) @ e1)
        c["s_all"] = [float((b["centre"] - c0) @ e1) for b in c["attach_blobs"]]
        c["off"], c["r_loc"] = perp_offset(prof, c["attach"], c["s"], e1)
    # --- coxae: long attachments away from the core ends (spines and hairs are short), densest cluster along the
    #     axis, one-sided offset (ventral)
    interior = [
        c
        for c in attached
        if prof["s_min"] + END_EXCLUSION * L < c["s"] < prof["s_max"] - END_EXCLUSION * L
        and c["reach"] >= MIN_LEG_REACH * L
        and (not USE_COXA_FOOTPRINT or c["attach_blobs"][0]["n"] >= (MIN_COXA_FOOTPRINT * L) ** 2)
    ]
    legs = []
    if interior:
        groups = cluster_1d(np.array([c["s"] for c in interior]), LEG_CLUSTER_GAP * L)
        groups.sort(key=lambda g: (-len(g), -sum(interior[i]["n"] for i in g)))
        legs = [interior[i] for i in groups[0]]
        if len(legs) >= 3:  # second pass: drop attachments not on the side where most coxae sit
            vdir = unit(np.sum([unit(c["off"]) for c in legs], 0))
            kept = [c for c in legs if unit(c["off"]) @ vdir > 0.5]  # within 60 deg of the majority side
            if len(kept) >= 2:
                legs = kept
    others = [c for c in attached if c not in legs]
    if len(legs) < 3:
        flags.append("few_leg_attachments")
    if not attached:
        flags.append("no_attached_appendages")
    leg_s = np.array([c["s"] for c in legs]) if legs else np.zeros(0)
    # ---------------------------------------------------------------- anterior (sign of e1)
    # Cue weights from the synthetic benchmark (40 posed OmniAnt instances, 2026-11-09): template profile 40/40
    # correct, coxae in the anterior half 33/40, thin appendages at the ends 24/40, waist-vs-neck 25/40, leg bulk
    # gradient 18/37 (dropped, kept as a diagnostic). Real specimens get re-weighted once the visual labels exist.
    votes, diag = {}, {}
    curl = float(var[1] / max(var[0], 1e-9))  # core PCA eigenvalue ratio: ~0.05 straight, >0.25 strongly curled
    if curl > 0.25:
        flags.append("body_curled")
    if legs:
        votes["coxae_anterior_half"] = 0.6 * float(
            np.clip((leg_s.mean() - s_mid) / (0.08 * L), -1, 1)
        )  # >0: +e1 anterior
        lo, hi = leg_s.min() - 0.04 * L, leg_s.max() + 0.04 * L
        # long thin appendages attached only beyond the coxa span: antennae (and long mandibles) in front,
        # nothing behind the hind coxae; counted per component so that hairs and leg fragments touching the gaster
        # do not outweigh an antenna
        ahead = sum(1 for c in others if min(c["s_all"]) > hi and c["reach"] >= MIN_END_REACH * L)
        behind = sum(1 for c in others if max(c["s_all"]) < lo and c["reach"] >= MIN_END_REACH * L)
        if ahead + behind:
            votes["end_appendages"] = 0.6 * float((ahead - behind) / (ahead + behind))
        if len(legs) >= 4 and np.ptp(leg_s) > 0.05 * L:
            bulk = np.array([c["n"] for c in legs], float)
            rk = lambda x: np.argsort(np.argsort(x))  # noqa: E731
            rho = np.corrcoef(rk(leg_s), rk(bulk))[0, 1]
            if np.isfinite(rho):
                diag["leg_bulk_gradient"] = float(np.clip(-rho, -1, 1))
        r, s = prof["r_eq"], prof["s"]
        r_body = np.median(r[r > 0]) if (r > 0).any() else 1.0
        plus = r[(s > hi) & (s < prof["s_max"] - 0.03 * L)]
        minus = r[(s < lo) & (s > prof["s_min"] + 0.03 * L)]
        if len(plus) and len(minus):
            votes["waist_deeper_than_neck"] = 0.3 * float(np.clip((plus.min() - minus.min()) / (0.25 * r_body), -1, 1))
        # Diagnostics only: leg tips reaching further behind than in front (hind legs longest) scored 34/44 on the
        # labelled real set and the bulkier core end (gaster) 33/47, but adding either as a vote lowered the
        # combined head accuracy (42 -> 40 of 47); on synthetic they are near chance (20/34, 22/40).
        tip_s = np.array([(c["tip"] - c0) @ e1 for c in legs])
        reach_plus, reach_minus = max(float((tip_s - hi).max()), 0.0), max(float((lo - tip_s).max()), 0.0)
        diag["leg_tip_reach"] = float(np.clip((reach_minus - reach_plus) / (0.1 * L), -1, 1))
        cnt = prof["count"]
        vol_plus, vol_minus = float(cnt[s > hi].sum()), float(cnt[s < lo].sum())
        if vol_plus + vol_minus > 0:
            diag["end_volume"] = float((vol_minus - vol_plus) / (vol_minus + vol_plus))
    try:
        tp = template_profile()
        pr = resample(prof["r_eq"], len(tp))
        cf, cr = np.corrcoef(pr, tp)[0, 1], np.corrcoef(pr[::-1], tp)[0, 1]
        votes["template_profile"] = 1.0 * float(np.clip((cf - cr) / 0.3, -1, 1))
        cues["profile_corr"] = {"forward": round(float(cf), 3), "reversed": round(float(cr), 3)}
    except Exception as ex:  # template unavailable: skip the cue
        cues["profile_corr"] = {"error": str(ex)}
    cues["diagnostics"] = {k: round(v, 3) for k, v in diag.items()}
    score = float(sum(votes.values()))
    signs = [np.sign(v) for v in votes.values() if abs(v) > 0.15]
    agree = (max(signs.count(1), signs.count(-1)) / len(signs)) if signs else 0.0
    anterior_sign = 1.0 if score >= 0 else -1.0
    ant_conf = float(np.tanh(abs(score) / 0.8) * (0.5 + 0.5 * agree))
    if curl > 0.25:
        ant_conf *= 0.7  # axial cues are unreliable on a curled body
    if ant_conf < 0.5:
        flags.append("anterior_uncertain")
    # ---------------------------------------------------------------- dorsal (perpendicular to e1)
    dorsal = {}
    vent = np.zeros(3)
    dor_conf = 0.0
    tips = []
    if legs:
        v_att = np.mean([c["off"] / max(c["r_loc"], 1.0) for c in legs], 0)  # coxae sit on the ventral surface
        tips = []
        for c in legs:
            t, _ = perp_offset(prof, c["tip"], c["s"], e1)
            tips.append(t / max(L, 1.0))
        v_tip = np.mean(tips, 0)  # feet hang below the body
        dorsal["coxa_attachment_offset"] = float(np.linalg.norm(v_att))
        dorsal["leg_tip_offset"] = float(np.linalg.norm(v_tip))
        w_att = float(np.clip(np.linalg.norm(v_att) / 0.3, 0, 1))
        w_tip = float(np.clip(np.linalg.norm(v_tip) / 0.08, 0, 1))
        vent = w_att * unit(v_att) + w_tip * unit(v_tip)
        if w_att > 0 and w_tip > 0:
            ang = float(np.degrees(np.arccos(np.clip(unit(v_att) @ unit(v_tip), -1, 1))))
            dorsal["attachment_vs_tip_angle_deg"] = ang
            if ang > 100 and w_tip > 0.5:  # feet clearly on the other side than the coxae say: look at it
                flags.append("legs_not_beneath_body")
        dor_conf = float(np.clip(np.linalg.norm(vent) / 1.2, 0, 1))
    if np.linalg.norm(vent) < 1e-6:  # fallback: thinnest core axis, towards the side with more appendage mass
        app_pts = np.argwhere(parts["occ"] & ~core).astype(np.float64)
        e3 = axes[2]
        side = float(((app_pts - c0) @ e3).mean()) if len(app_pts) else 1.0
        vent = e3 * np.sign(side if side != 0 else 1.0)
        flags.append("dorsal_from_pca_fallback")
        dor_conf = 0.1
    ventral = unit(vent - (vent @ e1) * e1)
    # roll from the bilateral symmetry of the MESOSOMA (the rigid root segment: head and gaster posture break the
    # symmetry of the whole core; measured on the labelled real set the whole-core plane was within 30 deg of the
    # truth in only 22 of 33 cases), sign from the coxae (they sit on the ventral surface); feet and appendage
    # mass only break ties.
    sym_mask = core
    if USE_MESOSOMA_SYMMETRY and len(legs) >= 2:
        s_all = (pts - c0) @ e1
        keep = (s_all >= leg_s.min() - 0.12 * L) & (s_all <= leg_s.max() + 0.12 * L)
        if keep.sum() > 2000:
            sym_mask = np.zeros_like(core)
            k = pts[keep].astype(int)
            sym_mask[k[:, 0], k[:, 1], k[:, 2]] = True
    t_sym, s_best, s_perp, phi = symmetry_plane(sym_mask, c0, e1, axes)
    contrast = s_best - s_perp
    ang_sym = float(np.degrees(np.arccos(min(1.0, abs(t_sym @ ventral)))))
    dorsal.update(
        {
            "symmetry_score": round(s_best, 3),
            "symmetry_score_perpendicular": round(s_perp, 3),
            "symmetry_region": "mesosoma" if sym_mask is not core else "core",
            "symmetry_vs_cue_angle_deg": round(ang_sym, 1),
            "roll_source": "cues",
        }
    )
    if contrast > 0.015 and legs:
        att_votes = np.clip(np.array([(c["off"] @ t_sym) / max(c["r_loc"], 1.0) for c in legs]), -1, 1)
        v_sign = float(att_votes.sum())
        margin = abs(v_sign) / len(legs)
        if SIGN_MODE == "mixed":
            tip_votes = np.clip(np.array([t @ t_sym for t in tips]) / 0.08, -1, 1)
            v_sign = v_sign + 0.5 * float(tip_votes.sum())
            margin = abs(v_sign) / (1.5 * len(legs))
        elif margin < 0.3:  # coxae undecided: feet and appendage mass break the tie
            tip_votes = np.clip(np.array([t @ t_sym for t in tips]) / 0.08, -1, 1)
            app_pts = np.argwhere(parts["occ"] & ~core).astype(np.float64)
            d = app_pts - c0
            d = d - np.outer(d @ e1, e1)
            mass = float(np.clip(np.mean(d @ t_sym) / (0.03 * L), -1, 1)) if len(app_pts) else 0.0
            v_sign = v_sign + 0.5 * float(tip_votes.sum()) + 0.5 * len(legs) * mass
            margin = abs(v_sign) / (2.0 * len(legs))
        dorsal["roll_sign_margin"] = round(margin, 3)
        if margin >= 0.25:
            ventral = unit(t_sym * np.sign(v_sign))
            dorsal["roll_source"] = "symmetry"
            dor_conf = max(dor_conf, min(1.0, 0.5 + margin))
        else:
            flags.append("roll_sign_uncertain")  # legs do not say which side of the sagittal axis is ventral
            dor_conf = min(dor_conf, 0.4)
        if ang_sym > 45:
            flags.append("roll_cues_disagree")  # mean leg direction and symmetry plane differ: look at it
            dor_conf = min(dor_conf, 0.45)
    elif contrast <= 0.015:
        flags.append("symmetry_plane_weak")
    if dor_conf < 0.5:
        flags.append("dorsal_uncertain")
    # ---------------------------------------------------------------- axes
    x_axis, axis_source = anterior_sign * e1, "core_pca"
    if len(legs) >= 3:
        ap = np.array([c["attach"] for c in legs])
        _, lax, lvar = principal_axes(ap)
        xm = lax[0] * np.sign(lax[0] @ x_axis)
        ang = float(np.degrees(np.arccos(np.clip(xm @ x_axis, -1, 1))))
        cues["mesosoma_axis_vs_core_axis_deg"] = round(ang, 1)
        if ang < 30 and lvar[0] > 3 * max(lvar[1], 1e-9) and np.ptp(leg_s) > 0.08 * L:
            x_axis, axis_source = xm, "coxa_line"
        elif ang >= 30:
            flags.append("mesosoma_axis_disagrees")
    z_axis = unit(-ventral - (-ventral @ x_axis) * x_axis)
    # ---------------------------------------------------------------- whole-shape overlap with the template (recorded, not used)
    sub = pts if len(pts) <= 100_000 else pts[np.random.default_rng(0).choice(len(pts), 100_000, replace=False)]
    try:
        ts = template_shape_scores(core, sub, c0, L, x_axis, z_axis)
        v_ant = (max(ts[(1, 1)], ts[(1, -1)]) - max(ts[(-1, 1)], ts[(-1, -1)])) / 0.04
        v_dor = (max(ts[(1, 1)], ts[(-1, 1)]) - max(ts[(1, -1)], ts[(-1, -1)])) / 0.04
        # Diagnostic only (2026-11-09): on the labelled real set the overlap confirmed the wrong orientation in all
        # three unflagged anterior errors and contradicted correct dorsal decisions; rest-pose template vs tucked
        # legs and curled gasters is too crude a comparison to decide anything.
        cues["template_shape"] = {
            "scores": {
                f"head{'+' if sx > 0 else '-'}_dorsal{'+' if sz > 0 else '-'}": round(v, 4)
                for (sx, sz), v in ts.items()
            },
            "anterior_vote": round(float(np.clip(v_ant, -1, 1)), 3),
            "dorsal_vote": round(float(np.clip(v_dor, -1, 1)), 3),
            "used": False,
        }
    except Exception as ex:
        cues["template_shape"] = {"error": str(ex)}
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis])  # rows: canonical axes in grid coords; det +1
    cues.update(
        {
            "anterior_votes": {k: round(v, 3) for k, v in votes.items()},
            "anterior_score": round(score, 3),
            "anterior_agreement": round(float(agree), 2),
            "dorsal": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dorsal.items()},
            "n_attached_appendages": len(attached),
            "n_leg_cluster": len(legs),
            "n_other_attached": len(others),
            "core_components": parts["n_core_components"],
            "axis_source": axis_source,
            "curl": round(curl, 3),
            "core_length_vox": round(float(L), 1),
            "open_radius_vox": round(parts["open_radius_vox"], 2),
        }
    )
    return {
        "R": R,
        "centre": c0,
        "anterior_confidence": round(ant_conf, 3),
        "dorsal_confidence": round(dor_conf, 3),
        "flags": flags,
        "cues": cues,
        "legs": legs,
        "others": others,
        "profile": prof,
        "e1": e1,
    }


# ----------------------------------------------------------------------------------------------- driver


def align(stl_path, out_dir=None, write_mesh=False, pitch=None):
    t0 = time.time()
    v, f = load_mesh(stl_path)
    name = os.path.basename(stl_path).replace("_envelope.stl", "").replace(".stl", "")
    parts = split_core_and_appendages(v, f, pitch=pitch)
    fr = decide_frame(parts)
    p, o = parts["pitch"], parts["origin"]
    R = fr["R"]
    origin_scan = o + fr["centre"] * p  # core centroid, micrometres
    vc = (v - origin_scan) @ R.T
    conf = min(fr["anterior_confidence"], fr["dorsal_confidence"])
    status = "ok" if conf >= 0.5 and not fr["flags"] else "review"
    L = fr["profile"]["s_max"] - fr["profile"]["s_min"]
    legs = [
        {
            "attach_um": (o + c["attach"] * p).tolist(),
            "tip_um": (o + c["tip"] * p).tolist(),
            "volume_frac": round(c["volume_frac"], 5),
            "reach_um": round(c["reach"] * p, 1),
            "s_frac": round(float(c["s"] / max(L, 1)), 3),
        }
        for c in fr["legs"]
    ]
    out = {
        "name": name,
        "units": "micrometre",
        "stage": "antscan_align (canonical frame)",
        "version": VERSION,
        "frame": {
            "rotation": R.tolist(),
            "origin_scan": origin_scan.tolist(),
            "det": float(np.linalg.det(R)),
            "note": "x_canonical = rotation @ (x_scan - origin_scan); +X head, +Y left, +Z dorsal",
        },
        "status": status,
        "confidence": {
            "anterior": fr["anterior_confidence"],
            "dorsal": fr["dorsal_confidence"],
            "overall": round(conf, 3),
        },
        "flags": fr["flags"],
        "cues": fr["cues"],
        "leg_attachments": legs,
        "core_length_um": round(float(L * p), 1),
        "canonical_bounds_um": {"min": vc.min(0).tolist(), "max": vc.max(0).tolist()},
        "grid": {
            "pitch_um": p,
            "shape": list(parts["occ"].shape),
            "solid_volume_um3": parts["volume_um3"],
            "mesh_volume_um3": parts["mesh_volume_um3"],
            "n_components_dropped_small": parts["n_small_dropped"],
        },
        "seconds": round(time.time() - t0, 1),
    }
    out["mesh"] = os.path.abspath(stl_path)
    out["mesh_md5"] = hashlib.md5(open(stl_path, "rb").read()).hexdigest()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}_frame.json")
        out = merge_previous_review(out, path)
        if out["status"] in ("reviewed", "excluded"):  # the reviewed rotation defines the canonical mesh
            R_use = np.asarray(out["frame"]["rotation"])
            vc = (v - np.asarray(out["frame"]["origin_scan"])) @ R_use.T
        json.dump(out, open(path, "w"), indent=1)
        if write_mesh:
            write_stl(os.path.join(out_dir, f"{name}_canonical.stl"), vc, f)
    return out, vc, f


def merge_previous_review(out, path):
    """Keep a manual or automatic review across re-runs while the envelope is unchanged (same md5): the reviewed rotation,
    status and review entries stay, the new automatic decision goes to "automatic". A review made on another mesh is
    moved to "review_invalidated" and the specimen needs a new review."""
    if not os.path.isfile(path):
        return out
    try:
        prev = json.load(open(path))
    except (OSError, ValueError):
        return out
    reviews = prev.get("review") or []
    if not reviews:
        return out
    if prev.get("mesh_md5") == out["mesh_md5"] and reviews[-1].get("mesh_md5") in (None, out["mesh_md5"]):
        auto = {
            "rotation": out["frame"]["rotation"],
            "status": out["status"],
            "confidence": out["confidence"],
            "flags": out["flags"],
            "version": out["version"],
        }
        out["automatic"] = auto
        out["review"] = reviews
        if prev.get("status") in ("reviewed", "excluded"):
            out["status"] = prev["status"]
            out["frame"]["rotation"] = prev["frame"]["rotation"]
            out["frame"]["det"] = prev["frame"].get("det")
            if prev.get("reason"):
                out["reason"] = prev["reason"]
            out.pop("canonical_bounds_um", None)
    else:
        out["review_invalidated"] = reviews
        out["flags"] = out["flags"] + ["review_invalidated_mesh_changed"]
        out["status"] = "review"
    return out


def write_stl(path, v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, np.stack([a, b, c], axis=1)
    with open(path, "wb") as fh:
        fh.write(b"antscan_align canonical frame, micrometre".ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("out_dir")
    ap.add_argument("--write-mesh", action="store_true")
    ap.add_argument(
        "--pitch", type=float, default=None, help="grid pitch in micrometres (default: V^(1/3) / %g)" % VOX_PER_CBRT_V
    )
    a = ap.parse_args()
    out, _, _ = align(a.stl, a.out_dir, a.write_mesh, a.pitch)
    print(json.dumps({k: out[k] for k in ("name", "status", "confidence", "flags", "seconds")}))


if __name__ == "__main__":
    main()
