"""PATHS-A (round 6): limb centrelines from geodesic extremity paths with contact cutting, named by anatomy.

  run(name, trunk_dir, mesh_path, out_dir, cfg=None) -> summary dict
  python paths.py --trunk-dir DIR --out-dir DIR [--names a,b | --names-file F] [--mesh-root DIR | --mesh-tmpl T] [--workers 4]

Input: a round-6 TRUNK product (<trunk_dir>/<name>/trunk.npz + trunk.json: occupancy grid in a working frame, opened
trunk core labels tpart (0 head, 1 mesosoma, 2 waist, 3 gaster), trunk path, trunk joints, BL) and the envelope mesh
(scan frame, um). Nothing else (no truth, no model, no round-3 output). The working frame's axes are never used as
anatomy: anterior comes from the trunk (b_a_1 -> neck), dorsal from where the legs attach (coxa roots are ventral),
left = dorsal x anterior.

Steps:
 1. per family (P: paths ending in the mesosoma core, H: in the head core; other trunk parts + a skin impassable):
    geodesic extremity tips, min-cost (1/D^2) medial paths tip -> core (the annotation tool's machinery);
 2. contact cutting: every radius minimum of every path is tested (pa_contacts); a neck where the path turns from one
    tube into another one that continues past the neck is a contact; a wall is cut between the two tubes and the paths
    are recomputed (iterated);
 3. duplicate routes are dropped (the annotation tool's dedup rules);
 4. anatomical frame from the trunk + leg roots; limb candidates scored; legs named by a constrained assignment
    (fore -> hind roots front to back on each side, left of a pair = root further left, side from root + femur
    direction), antennae / mandibles on the head by length, root position and side;
 5. limb product (round-6 limb format) with status, confidence and evidence.
"""

import argparse
import itertools  # noqa: F401
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import pa_contacts as pc
import pa_geom as pg
import pa_name

VERSION = "S6 paths_A 2026-11-10 v15"
VERSION_STAGE1 = "stage1 v15"
LEGS = ["L1", "L2", "L3", "R1", "R2", "R3"]
ANTS = ["AL", "AR"]
MANS = ["ML", "MR"]
LIMBS = LEGS + ANTS + MANS

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {
    # fitted naming priors (pa_name.fit on all 21 dev specimens, TRUNK-trunk candidates); None = heuristic fallback
    "priors": os.path.join(HERE, "priors_final", "priors_all_trunk.json"),
    "tip_nms_pct": 3.0,
    "tip_min_pct": 5.0,
    "skin_frac": 0.25,
    "open_pct": 4.0,
    "ext_pct": 6.0,
    "smooth_sigma_pitch": 1.5,
    "cut": True,
    "cut_iters": 4,
    "cut_ratio": 0.6,
    "cut_turn": 35.0,
    "cut_cont": 1.4,
    "dedup_cross_family": True,
    "block_head_conflicts": False,
    "own_same_family": True,
    "dedup_front_head_exempt": True,
    "tip_centre": True,
    "tip_march_pitch": 2.5,
    "leg_min_free_pct": 12.0,
    "leg_min_own_pct": 8.0,
    "ant_min_free_pct": 15.0,
    "man_min_free_pct": 4.0,
    "man_max_free_pct": 22.0,
}


# ====================================================================== paths
def family_paths(occ, Dc, tpart, src_part, cut, BL, pitch, k_skin, cfg):
    src = tpart == src_part
    if not src.any():
        return [], None, src
    trunk = tpart >= 0
    blocked = ndimage.binary_dilation(trunk & ~src, iterations=k_skin) & occ & ~src
    passable = occ & ~blocked & ~cut
    sb = np.argwhere(src & ~ndimage.binary_erosion(src))
    G = pg.geodesic(passable, sb)
    G[src] = 0
    nms = max(2, int(round(cfg["tip_nms_pct"] / 100 * BL / pitch)))
    tips = pg.tips_from(G, passable & ~src, nms, cfg["tip_min_pct"] / 100 * BL / pitch)
    cost = np.where(passable, 1.0 / Dc**2, np.inf)
    paths, _ = pg.mcp_paths(cost, sb, tips) if len(tips) else ([], None)
    out = []
    for t, p in zip(tips, paths):
        if p is None:
            continue
        rec = {"tip_ijk": t, "ijk": np.asarray(p[::-1], int), "G_tip": float(G[tuple(t)])}
        if cfg.get("tip_centre", True):
            rec["tip_centre_ijk"] = tip_centre(G, passable, t, BL, pitch)
        out.append(rec)
    return out, G, src


def tip_centre(G, passable, t, BL, pitch, rad_pct=2.5, band_pct=0.8):
    """Centre of a split limb end (e.g. two claws): the geodesic maxima of near-equal height around the tip that join
    the tip's own cap within a short geodesic band. Returns the mean voxel position (float) or None (single end)."""
    rc = max(2, int(round(rad_pct / 100 * BL / pitch)))
    w = max(1.5, band_pct / 100 * BL / pitch)
    lo = np.maximum(t - rc, 0)
    hi = np.minimum(t + rc + 1, np.array(G.shape))
    Gs = G[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    Ps = passable[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] & np.isfinite(Gs)
    gt = float(G[tuple(t)])
    ball = np.linalg.norm(np.indices(Gs.shape).reshape(3, -1).T + lo - t, axis=1).reshape(Gs.shape) <= rc
    st = np.ones((3, 3, 3))
    caps, n = ndimage.label(Ps & ball & (Gs >= gt - w), structure=st)
    if n < 2:
        return None
    joined, _ = ndimage.label(Ps & ball & (Gs >= gt - 3 * w), structure=st)
    tl = tuple(t - lo)
    own_cap, own_join = caps[tl], joined[tl]
    pts = [t.astype(float)]
    for k in range(1, n + 1):
        if k == own_cap:
            continue
        m = caps == k
        gk = np.where(m, Gs, -np.inf)
        a = np.unravel_index(int(np.argmax(gk)), Gs.shape)
        if gk[a] >= gt - 0.5 * w and joined[a] == own_join:
            pts.append(np.array(a, float) + lo)
    return np.mean(pts, axis=0) if len(pts) > 1 else None


def contact_scan(paths, occ, cut, D, trunk_dil, cfg, tried=None):
    """Test the radius minima of every path; cut the contacts found (in place). Returns the list of cut events.
    `tried`: list of neck voxels already cut (a neck within 2 voxels of one is not cut again)."""
    events = []
    occ_c = occ & ~cut
    tried = [] if tried is None else tried
    for p in paths:
        ijk = p["ijk"]
        Dp = D[tuple(ijk.T)]
        intr = trunk_dil[tuple(ijk.T)]
        n_free = int(np.argmax(intr)) if intr.any() else len(ijk)
        for i in pc.path_minima(Dp, 2, n_free - 2):
            if cut[tuple(ijk[i])]:
                continue
            if any(np.abs(ijk[i] - q).max() <= 2 for q in tried):
                continue
            j = pc.junction2(occ_c, D, ijk, i, Dp=Dp, trunk=trunk_dil, need_box=True)
            if j.get("skip") or j["ratio"] > cfg["cut_ratio"] or j.get("turn_deg", 0) < cfg["cut_turn"]:
                continue
            if j.get("trunk_in_box", 0) > 0:
                continue
            ce, le = pc.continuation(D, ijk, i, -1, j["r_E"], j["r_neck"])
            cx, lx = pc.continuation(D, ijk, i, +1, j["r_X"], j["r_neck"])
            if max(le, lx) < cfg["cut_cont"]:
                continue
            nw = pc.cut_wall2(j, cut)
            tried.append(ijk[i].copy())
            if nw == 0:
                continue
            ev = {k: v for k, v in j.items() if not k.startswith("_")}
            ev.update(
                {
                    "fam": p["fam"],
                    "tip_ijk": p["tip_ijk"].tolist(),
                    "wall_vox": nw,
                    "contE": ce,
                    "contX": cx,
                    "lenE": le,
                    "lenX": lx,
                }
            )
            events.append(ev)
            break  # one cut per path per pass; the path changes after it
    return events


def dense_path(ijk, lo, pitch, BL, surf, core_target, cfg, tip_centre_ijk=None):
    """the annotation tool's build_path logic: resample / smooth, tip marched to the surface, straight core extension. A split end
    (tip_centre_ijk: centre of near-equal geodesic maxima, e.g. two claws) starts the path at that centre instead."""
    h = pitch / 2.0
    Xv = pg.to_um(ijk, lo, pitch)
    X = pg.smooth(pg.resample(Xv, h), cfg["smooth_sigma_pitch"] * pitch / h)
    u0 = pg.end_dir(X, 2.0 * pitch, at_start=True)
    # march the tip to the surface; at most tip_march_pitch pitches: a tip touching another part (e.g. a tarsus on the
    # gaster) must not be marched into that part
    tipx = pg.march_out(X[0], u0, surf, h / 2, cfg.get("tip_march_pitch", 2.5) * pitch)
    if len(tipx) and surf.signed(tipx[-1])[0] < 0:
        tipx = tipx[:0]  # never reached the surface: the tip is in contact, keep the voxel tip
    if len(tipx):
        X = np.vstack([tipx[::-1], X])
    if tip_centre_ijk is not None:
        c = lo + np.asarray(tip_centre_ijk, float) * pitch
        sep = 2.0 * float(np.linalg.norm(c - X[0]))  # ~ distance between the split ends
        k = int(np.searchsorted(pg.arclen(X), 1.5 * sep))
        k = min(max(k, 1), len(X) - 2)
        X = np.vstack([c[None], X[k:]])
    s_main_end = float(pg.arclen(X)[-1])
    if core_target is not None:
        d = core_target - X[-1]
        nd = float(np.linalg.norm(d))
        L = min(cfg["ext_pct"] / 100 * BL, 0.85 * nd)
        if L > h and nd > 0:
            m = int(np.ceil(L / h))
            X = np.vstack([X, X[-1] + np.outer(np.arange(1, m + 1) * (L / m), d / nd)])
    X = pg.resample(X, h)
    s = pg.arclen(X)
    r = -surf.signed(X)
    return X, s, r, s_main_end


# ---------------------------------------------------------------------- dedup (the annotation tool's dedup logic, copied)
def coverage_share(A, B, pitch):
    ma = A["s"] <= A["s_main_end"]
    mb = B["s"] <= B["s_main_end"]
    XA, rA = A["X"][ma], A["r"][ma]
    if len(XA) < 2 or mb.sum() < 2:
        return 0.0
    d, _ = cKDTree(B["X"][mb]).query(XA)
    cov = d <= np.maximum(2.5 * pitch, 0.75 * np.maximum(rA, 0))
    w = np.r_[0.0, np.linalg.norm(np.diff(XA, axis=0), axis=1)]
    return float((w * cov).sum() / max(w.sum(), 1e-9))


def covered_mask(A, others, pitch):
    ma = A["s"] <= A["s_main_end"]
    XA, rA = A["X"][ma], A["r"][ma]
    tol = np.maximum(2.5 * pitch, 0.75 * np.maximum(rA, 0))
    cov = np.zeros(len(XA), bool)
    for B in others:
        mb = B["s"] <= B["s_main_end"]
        if mb.sum() < 2:
            continue
        cov |= cKDTree(B["X"][mb]).query(XA)[0] <= tol
    return cov, XA


def dedup(cands, pitch, BL, cross=0.7, twin=0.9, tip_pct=3.0, cross_family=True, exempt=()):
    tip_tol = max(4 * pitch, tip_pct / 100 * BL)
    keep = list(range(len(cands)))
    dropped = []
    C = {(i, j): coverage_share(cands[i], cands[j], pitch) for i in keep for j in keep if i != j}

    def share(i, js):
        cov, XA = covered_mask(cands[i], [cands[j] for j in js], pitch)
        w = np.r_[0.0, np.linalg.norm(np.diff(XA, axis=0), axis=1)]
        return float((w * cov).sum() / max(w.sum(), 1e-9))

    for victim, cover in (("H", "P"), ("P", "H")) if cross_family else ():
        while True:
            best = None
            for i in keep:
                if cands[i]["fam"] != victim or i in exempt:
                    continue
                other = [j for j in keep if j != i and cands[j]["fam"] == cover]
                if victim == "P" and exempt:
                    # a mesosoma path is only a detour of a head limb when that head path is rooted in the front of
                    # the head (antenna / mandible); head paths entering the back of the head are leg detours
                    other = [j for j in other if j in exempt]
                if not other:
                    continue
                c = share(i, other)
                shared = [j for j in other if np.linalg.norm(cands[j]["X"][0] - cands[i]["X"][0]) <= tip_tol]
                if c >= cross and shared and (best is None or c > best[1]):
                    best = (i, c, f"{victim}_covered_by_{cover}_shared_tip")
            if best is None:
                break
            keep.remove(best[0])
            dropped.append({"index": best[0], "coverage": round(best[1], 3), "why": best[2]})
    while True:
        best = None
        for i in keep:
            for j in keep:
                if i < j and cands[i]["fam"] == cands[j]["fam"] and min(C[(i, j)], C[(j, i)]) >= twin:
                    lose = i if cands[i]["s_main_end"] < cands[j]["s_main_end"] else j
                    c = min(C[(i, j)], C[(j, i)])
                    if best is None or c > best[1]:
                        best = (lose, c, "same_family_twin")
        if best is None:
            break
        keep.remove(best[0])
        dropped.append({"index": best[0], "coverage": round(best[1], 3), "why": best[2]})
    while True:
        cand = [(i, share(i, [j for j in keep if j != i])) for i in keep]
        cand = [(i, c) for i, c in cand if c >= twin]
        if not cand:
            break
        i, c = max(cand, key=lambda ic: (bool(cands[ic[0]].get("pseudo_tip")), ic[1]))
        keep.remove(i)
        dropped.append(
            {
                "index": i,
                "coverage": round(c, 3),
                "why": "union_covered" + ("_pseudo_tip" if cands[i].get("pseudo_tip") else ""),
            }
        )
    return keep, dropped


# ====================================================================== anatomy
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def trunk_axes(tr, cores_c):
    J = tr["joints_w"]
    a = None
    if J.get("neck") is not None and J.get("b_a_1") is not None:
        a = unit(J["neck"] - J["b_a_1"])
        p0, p1 = J["b_a_1"], J["neck"]
    if a is None or not np.isfinite(a).all():
        a = unit(cores_c[0] - cores_c[1])
        p0, p1 = cores_c[1] - 0.5 * (cores_c[0] - cores_c[1]) * 0.6, cores_c[0]
    return a, np.asarray(p0, float), np.asarray(p1, float)


class TrunkCoords:
    """Curvilinear trunk coordinates: t = arc position along the trunk path T between b_a_1 (t = 0) and the neck (t = 1);
    offsets are taken from the nearest T point, perpendicular to T's local tangent (robust to an arched mesosoma).
    Falls back to the straight b_a_1 -> neck axis when there is no trunk path."""

    def __init__(self, tr, a, p0, p1):
        self.a, self.p0, self.p1 = a, p0, p1
        T = tr.get("T") or {}
        X = T.get("X")
        self.ok = X is not None and len(X) > 5
        if self.ok:
            X = pg.resample(np.asarray(X, float), tr["pitch"] / 2)
            s = pg.arclen(X)
            tan = np.gradient(X, axis=0)
            tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
            tan = ndimage.gaussian_filter1d(tan, 4.0, axis=0, mode="nearest")
            tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
            if (X[-1] - X[0]) @ a < 0:  # orient T from the gaster to the head
                X, tan = X[::-1].copy(), -tan[::-1].copy()
                s = pg.arclen(X)
            tree = cKDTree(X)
            self.s0 = float(s[tree.query(p0)[1]])
            self.s1 = float(s[tree.query(p1)[1]])
            if self.s1 - self.s0 < 1e-6:
                self.ok = False
            else:
                L = self.s1 - self.s0
                m = (s >= self.s0 - 0.35 * L) & (s <= self.s1 + 0.15 * L)  # mesosoma (+ a margin) only
                self.X, self.s, self.tan = X[m], s[m], tan[m]
                self.tree = cKDTree(self.X)

    def coords(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        if not self.ok:
            return axis_coords(P, self.a, self.p0, self.p1)
        _, j = self.tree.query(P)
        along = np.sum((P - self.X[j]) * self.tan[j], axis=1)
        end = (j == 0) | (j == len(self.X) - 1)  # beyond the segment: extrapolate along the end tangent
        sj = self.s[j] + np.where(end, along, 0.0)
        t = (sj - self.s0) / (self.s1 - self.s0)
        off = P - self.X[j]
        off = off - np.sum(off * self.tan[j], axis=1, keepdims=True) * self.tan[j]
        return t, off


def symmetry_lateral_axis(tr, a, tc, lat_guess=None, max_dev_deg=50.0, n_pts=30000, step_deg=2.0):
    """Lateral axis (up to sign) of the bilaterally symmetric trunk core (head + mesosoma): the normal of the plane
    through the trunk path that maps the core onto itself best (reflection overlap), searched over roll angles about
    the local trunk tangent. Returns (unit lateral vector or None, best overlap, overlap of the worst angle)."""
    if not tc.ok:
        return None, 0.0, 0.0
    tpart, lo, pitch = tr["tpart"], tr["lo"], tr["pitch"]
    core = tpart == 1  # the mesosoma core: rigid, carries the legs (the head may be turned)
    idx = np.argwhere(core)
    if len(idx) < 100:
        return None, 0.0, 0.0
    rng = np.random.default_rng(0)
    if len(idx) > n_pts:
        idx = idx[rng.choice(len(idx), n_pts, replace=False)]
    P = pg.to_um(idx, lo, pitch)
    _, j = tc.tree.query(P)
    C = tc.X[j]
    Tn = tc.tan[j]
    # a reference perpendicular basis per point (from a fixed helper vector)
    hlp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(Tn, hlp)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9)
    e2 = np.cross(Tn, e1)
    best, best_phi, worst = -1.0, 0.0, 2.0
    shape = np.array(core.shape)
    jm = int(np.argmin(np.abs(tc.s - 0.5 * (tc.s0 + tc.s1))))
    tn = tc.tan[jm]
    e1m = np.cross(tn, hlp)
    e1m /= np.linalg.norm(e1m)
    e2m = np.cross(tn, e1m)
    phi_guess = None
    if lat_guess is not None:
        phi_guess = float(np.degrees(np.arctan2(lat_guess @ e2m, lat_guess @ e1m))) % 180.0
    for phi in np.arange(0.0, 180.0, step_deg):
        if phi_guess is not None:
            dev = abs((phi - phi_guess + 90.0) % 180.0 - 90.0)
            if dev > max_dev_deg:
                continue
        ph = np.radians(phi)
        nrm = np.cos(ph) * e1 + np.sin(ph) * e2
        Q = P - 2.0 * np.sum((P - C) * nrm, axis=1, keepdims=True) * nrm
        q = np.round((Q - lo) / pitch).astype(int)
        ok = np.all((q >= 0) & (q < shape), axis=1)
        sc = float(core[tuple(q[ok].T)].sum()) / len(P)
        if sc > best:
            best, best_phi = sc, phi
        worst = min(worst, sc)
    # global lateral axis: the plane normal at the mesosoma centre for the best roll
    ph = np.radians(best_phi)
    lat = np.cos(ph) * e1m + np.sin(ph) * e2m
    lat = lat - (lat @ a) * a
    return unit(lat), best, worst


def axis_coords(P, a, p0, p1, d=None, l=None):
    """t along the mesosoma (0 = b_a_1, 1 = neck) and the perpendicular offset vectors."""
    P = np.atleast_2d(P)
    L = max(float((p1 - p0) @ a), 1e-6)
    t = ((P - p0) @ a) / L
    off = (P - p0) - np.outer((P - p0) @ a, a)
    return t, off


# ====================================================================== main
STAGE1_KEYS = (
    "tip_nms_pct",
    "tip_min_pct",
    "skin_frac",
    "open_pct",
    "ext_pct",
    "smooth_sigma_pitch",
    "cut",
    "cut_iters",
    "dedup_cross_family",
    "cut_cont",
    "dedup_front_head_exempt",
    "tip_centre",
    "tip_march_pitch",
    "cut_ratio",
    "cut_turn",
)


def stage1(name, trunk_dir, mesh_path, C):
    """Paths with contact cutting, dense paths, dedup, anatomical frame, per-candidate features (cacheable)."""
    t0 = time.time()
    tr = pg.load_trunk(trunk_dir, name)
    occ, tpart, pitch, BL, lo = tr["occ"], tr["tpart"], tr["pitch"], tr["BL"], tr["lo"]
    R, o = tr["R"], tr["o"]
    V, F = pg.load_mesh_working(mesh_path, R, o)
    surf = pg.Surface(V, F, pitch)
    D = ndimage.distance_transform_edt(occ)
    Dc = np.maximum(D, 0.5)
    trunk = tpart >= 0
    trunk_dil = ndimage.binary_dilation(trunk, iterations=2)
    r_open = max(1, int(round(C["open_pct"] / 100 * BL / pitch)))
    k_skin = max(2, int(round(C["skin_frac"] * r_open)))
    cut = np.zeros_like(occ)
    timing = {"load": time.time() - t0}
    events = []
    tried = []
    n_pass = C["cut_iters"] if C["cut"] else 0
    for it in range(n_pass + 1):
        fams = {}
        allp = []
        for fam, sp in (("P", 1), ("H", 0)):
            ps, G, src = family_paths(occ, Dc, tpart, sp, cut, BL, pitch, k_skin, C)
            for p in ps:
                p["fam"] = fam
            fams[fam] = (ps, src)
            allp += ps
        if it == n_pass:
            break
        ev = contact_scan(allp, occ, cut, D, trunk_dil, C, tried)
        for e in ev:
            e["pass"] = it
        events += ev
        if not ev:
            break
    timing["paths"] = time.time() - t0
    cores_c = {}
    for k in range(4):
        m = tpart == k
        cores_c[k] = pg.to_um(np.argwhere(m), lo, pitch).mean(0) if m.any() else None
    cands = []
    for fam, (ps, src) in fams.items():
        cen = cores_c[1 if fam == "P" else 0]
        other_core = tpart == (0 if fam == "P" else 1)
        for p in ps:
            X, s, r, sme = dense_path(p["ijk"], lo, pitch, BL, surf, cen, C, p.get("tip_centre_ijk"))
            t = p["tip_ijk"]
            lo_i, hi_i = np.maximum(t - 3, 0), np.minimum(t + 4, np.array(occ.shape))
            pseudo = bool(other_core[lo_i[0] : hi_i[0], lo_i[1] : hi_i[1], lo_i[2] : hi_i[2]].any())
            cands.append(
                {
                    "fam": fam,
                    "ijk": p["ijk"],
                    "tip_ijk": t,
                    "X": X,
                    "s": s,
                    "r": r,
                    "s_main_end": sme,
                    "pseudo_tip": pseudo,
                    "G_tip_um": p["G_tip"] * pitch,
                }
            )
    # head-family paths rooted in the FRONT of the head (where antennal sockets and mandible bases are) are not dropped
    # as "a leg reached from the head through a contact" (those enter the head at its back or sides)
    exempt = set()
    if C.get("dedup_front_head_exempt", True) and cores_c[0] is not None:
        J = tr["joints_w"]
        hf, nk = J.get("head_front"), J.get("neck")
        if hf is not None and nk is not None:
            ha = unit(hf - nk)
            for k, c in enumerate(cands):
                if c["fam"] != "H":
                    continue
                ime = min(int(np.searchsorted(c["s"], c["s_main_end"])), len(c["X"]) - 1)
                if 100 * float((c["X"][ime] - cores_c[0]) @ ha) / BL > 0.0:
                    exempt.add(k)
    keep, dropped = dedup(cands, pitch, BL, cross_family=C.get("dedup_cross_family", True), exempt=exempt)
    timing["dense+dedup"] = time.time() - t0
    kept = [cands[i] for i in keep]
    frame, feats = frame_and_features(kept, tr, cores_c)
    # purity check without truth: the narrowest remaining neck on each candidate's free part that looks like a contact
    occ_c = occ & ~cut
    for p, f in zip(kept, feats):
        ijk = p["ijk"]
        Dp = D[tuple(ijk.T)]
        intr = trunk_dil[tuple(ijk.T)]
        n_free = int(np.argmax(intr)) if intr.any() else len(ijk)
        worst = None
        for i in pc.path_minima(Dp, 2, n_free - 2):
            if Dp[i] > 0.6 * max(1.0, float(np.percentile(Dp[:n_free], 75))):
                continue
            j = pc.junction2(occ_c, D, ijk, i, Dp=Dp, trunk=trunk_dil)
            if j.get("skip"):
                continue
            le = pc.continuation(D, ijk, i, -1, j["r_E"], j["r_neck"])[1]
            lx = pc.continuation(D, ijk, i, +1, j["r_X"], j["r_neck"])[1]
            susp = j["ratio"] <= 0.5 and j.get("turn_deg", 0) >= 60 and (j.get("loop") or max(le, lx) >= 0.8)
            if susp and (worst is None or j["ratio"] < worst["ratio"]):
                worst = {
                    "arc_frac": round(i / max(1, n_free), 3),
                    "ratio": round(j["ratio"], 3),
                    "turn_deg": round(j.get("turn_deg", 0), 1),
                    "loop": bool(j.get("loop")),
                    "cont": round(max(le, lx), 2),
                    "trunk_near": bool(j.get("trunk_in_box", 0) > 0),
                }
        f["suspect_neck"] = worst
        f["n_cuts_on_path"] = int(sum(1 for e in events if np.min(np.abs(ijk - np.asarray(e["c"])).max(axis=1)) <= 3))
    conflicts = same_limb_pairs(kept, pitch, BL)
    timing["features"] = time.time() - t0
    small_tr = {k: tr[k] for k in ("name", "R", "o", "BL", "pitch", "lo", "joints_w", "T", "meta")}
    small_tr["shape"] = occ.shape
    return {
        "tr": small_tr,
        "kept": kept,
        "feats": feats,
        "frame": frame,
        "conflicts": conflicts,
        "events": events,
        "dropped": dropped,
        "n_candidates": len(cands),
        "timing": timing,
        "cores_c": cores_c,
        "cut_packed": np.packbits(cut),
    }


def run(name, trunk_dir, mesh_path, out_dir, cfg=None):
    import hashlib
    import pickle

    C = dict(CFG)
    C.update(cfg or {})
    t0 = time.time()
    S1 = None
    cache = None
    if C.get("stage1_cache"):
        key = json.dumps({k: C[k] for k in STAGE1_KEYS}, sort_keys=True) + VERSION_STAGE1
        fz = os.path.join(trunk_dir, name, "trunk.npz")
        key += f"|{os.path.getmtime(fz):.0f}|{os.path.abspath(trunk_dir)}|{os.path.abspath(mesh_path)}"
        h = hashlib.sha1(key.encode()).hexdigest()[:12]
        cache = os.path.join(C["stage1_cache"], f"{name}_{h}.pkl")
        if os.path.isfile(cache):
            S1 = pickle.load(open(cache, "rb"))
    if S1 is None:
        S1 = stage1(name, trunk_dir, mesh_path, C)
        if cache:
            os.makedirs(C["stage1_cache"], exist_ok=True)
            pickle.dump(S1, open(cache, "wb"))
    # own arc (not covered by other paths) against the candidate's OWN family only: a leg duplicated by a head-family
    # route through a contact (or an antenna by a mesosoma route) keeps its own arc
    if C.get("own_same_family", True):
        BLs, pitch_s = S1["tr"]["BL"], S1["tr"]["pitch"]
        for p, f in zip(S1["kept"], S1["feats"]):
            others = [q for q in S1["kept"] if q is not p and q["fam"] == p["fam"]]
            cov, XA = covered_mask(p, others, pitch_s)
            w = np.r_[0.0, np.linalg.norm(np.diff(XA, axis=0), axis=1)]
            f["own_pct"] = 100 * float((w * ~cov).sum()) / BLs
    # distance of every candidate's free voxel path to the contact cuts (rerouted paths run along a cut wall)
    shape = tuple(S1["tr"]["shape"])
    cut = np.unpackbits(S1["cut_packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    if cut.any():
        dcut = ndimage.distance_transform_edt(~cut)
        pitch = S1["tr"]["pitch"]
        for p, f in zip(S1["kept"], S1["feats"]):
            ijk = p["ijk"]
            # distance to the nearest cut voxel, in units of the path's own tube radius
            r_vox = max(1.0, f.get("r_med", 0.0) / pitch)
            f["cut_dist_vox"] = float(dcut[tuple(ijk.T)].min())
            f["cut_dist_rel"] = f["cut_dist_vox"] / r_vox
    else:
        for f in S1["feats"]:
            f["cut_dist_vox"] = None
    pri = None
    if C.get("priors"):
        pth = C["priors"].format(name=name)
        if os.path.isfile(pth):
            pri = pa_name.load_priors(pth)
    names, scores = name_limbs(S1, C, pri)
    timing = dict(S1["timing"])
    timing["stage2"] = time.time() - t0
    out = write_product(name, out_dir, S1, names, scores, C, timing, t0)
    return out


# ====================================================================== naming
def path_features(p, tc, d, l, BL, pitch, others, tdist=None, lo=None, dhead=None, dwaist=None):
    a = tc.a
    ime = int(np.searchsorted(p["s"], p["s_main_end"]))
    ime = min(max(ime, 1), len(p["X"]) - 1)
    root = p["X"][ime]
    tip = p["X"][0]
    # coxa end region: the path point 8 % BL distal of the core boundary
    kcx = int(np.searchsorted(p["s"], max(0.0, p["s_main_end"] - 0.08 * BL)))
    cx = p["X"][kcx]
    t_cx, off_cx = tc.coords(cx)
    t_tip, off_tip = tc.coords(tip)
    extra = {
        "t_cx": float(t_cx[0]),
        "y_cx_pct": 100 * float(off_cx[0] @ l) / BL,
        "z_cx_pct": 100 * float(off_cx[0] @ d) / BL,
        "t_tip": float(t_tip[0]),
        "y_tip_pct": 100 * float(off_tip[0] @ l) / BL,
        "z_tip_pct": 100 * float(off_tip[0] @ d) / BL,
    }
    if dhead is not None and dwaist is not None:
        for nm_, P_ in (("root", root), ("cx", cx)):
            q = tuple(np.clip(np.round((P_ - lo) / pitch).astype(int), 0, np.array(dhead.shape) - 1))
            dh, dw = float(dhead[q]) * pitch, float(dwaist[q]) * pitch
            extra[f"dhead_{nm_}_pct"] = 100 * dh / BL
            extra[f"dwaist_{nm_}_pct"] = 100 * dw / BL
            extra[f"u_{nm_}"] = dh / max(dh + dw, 1e-9)  # 0 at the head core, 1 at the waist / gaster core
    if tdist is not None:
        q = np.clip(np.round((p["X"][:ime] - lo) / pitch).astype(int), 0, np.array(tdist.shape) - 1)
        dd = tdist[tuple(q.T)] * pitch
        extra["tip_trunk_pct"] = 100 * float(dd[0]) / BL
        extra["reach_pct"] = 100 * float(dd.max()) / BL if len(dd) else 0.0
    r_free = np.maximum(p["r"][:ime], 0)
    if len(r_free) > 4:
        n = len(r_free)
        extra["r_tip10_pct"] = 100 * float(np.median(r_free[: max(2, n // 10)])) / BL
        extra["r_max_pct"] = 100 * float(np.percentile(r_free, 95)) / BL
        extra["r_prox_pct"] = 100 * float(np.median(r_free[int(0.6 * n) :])) / BL
    free = p["s_main_end"]
    cov, XA = covered_mask(p, others, pitch)
    w = np.r_[0.0, np.linalg.norm(np.diff(XA, axis=0), axis=1)]
    own = float((w * ~cov).sum())
    t_root, off = tc.coords(root)
    y_root = float(off[0] @ l)
    z_root = float(off[0] @ d)
    # proximal direction of the limb: from the root outwards over ~12 % BL of free part
    k2 = int(np.searchsorted(p["s"], max(0.0, p["s_main_end"] - 0.12 * BL)))
    prox = unit(p["X"][k2] - root)
    rmed = float(np.median(np.maximum(p["r"][:ime], 0))) if ime > 2 else 0.0
    return {
        **extra,
        "free_um": float(free),
        "free_pct": 100 * free / BL,
        "own_pct": 100 * own / BL,
        "t_root": float(t_root[0]),
        "y_root_pct": 100 * y_root / BL,
        "z_root_pct": 100 * z_root / BL,
        "prox_y": float(prox @ l),
        "prox_z": float(prox @ d),
        "prox_a": float(prox @ a),
        "root": root,
        "tip": tip,
        "r_med": rmed,
        "r_med_pct": 100 * rmed / BL,
        "ime": ime,
        "tip_dist_core_pct": None,
    }


def frame_and_features(kept, tr, cores_c):
    BL, pitch = tr["BL"], tr["pitch"]
    a, p0, p1 = trunk_axes(tr, cores_c)
    tc = TrunkCoords(tr, a, p0, p1)
    P = [p for p in kept if p["fam"] == "P"]
    # ---- dorsal from leg roots: long P paths attach ventrally (length-weighted mean root offset = ventral)
    offs, ws = [], []
    for p in P:
        if p["s_main_end"] < 0.10 * BL:
            continue
        ime = min(int(np.searchsorted(p["s"], p["s_main_end"])), len(p["X"]) - 1)
        _, off = tc.coords(p["X"][ime])
        k2 = int(np.searchsorted(p["s"], max(0.0, p["s_main_end"] - 0.15 * BL)))
        prox = p["X"][k2] - p["X"][ime]
        prox = prox - (prox @ a) * a
        v = unit(off[0]) + 0.5 * unit(prox)
        offs.append(unit(v))
        ws.append(min(p["s_main_end"] / BL, 0.6))
    if offs:
        vent = unit(np.average(np.array(offs), axis=0, weights=np.array(ws)))
        dorsal_src = f"{len(offs)} long mesosoma paths (roots + proximal directions)"
    else:
        vent = -unit(np.cross(a, np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])))
        dorsal_src = "none (no long mesosoma path): arbitrary"
    d_leg = unit(-(vent - (vent @ a) * a))
    lat, sym_best, sym_worst = symmetry_lateral_axis(tr, a, tc, lat_guess=unit(np.cross(d_leg, a)))
    if lat is not None and sym_best - sym_worst > 0.05:
        d_sym = unit(np.cross(a, lat))  # dorsal up to sign from the symmetry plane
        if d_sym @ d_leg < 0:
            d_sym = -d_sym  # sign: legs attach ventrally
        d = d_sym
        dorsal_src += (
            f"; roll from trunk bilateral symmetry (overlap {sym_best:.3f}, worst {sym_worst:.3f}, "
            f"angle to the leg-root estimate {np.degrees(np.arccos(np.clip(d_sym @ d_leg, -1, 1))):.0f} deg)"
        )
    else:
        d = d_leg
    l = unit(np.cross(d, a))
    J = tr["joints_w"]
    hf, nk = J.get("head_front"), J.get("neck")
    ha = unit(hf - nk) if (hf is not None and nk is not None) else a
    hl = unit(l - (l @ ha) * ha)
    hd = unit(np.cross(ha, hl))
    hc = cores_c[0] if cores_c[0] is not None else (nk if nk is not None else p1)
    frame = {
        "anterior": a,
        "dorsal": d,
        "left": l,
        "p0": p0,
        "p1": p1,
        "head_axis": ha,
        "head_centre": hc,
        "dorsal_source": dorsal_src,
    }
    tdist = ndimage.distance_transform_edt(~(tr["tpart"] >= 0))
    hcore = tr["tpart"] == 0
    wcore = tr["tpart"] >= 2
    dhead = ndimage.distance_transform_edt(~hcore) if hcore.any() else None
    dwaist = ndimage.distance_transform_edt(~wcore) if wcore.any() else None
    feats = []
    for p in kept:
        others = [q for q in kept if q is not p]
        f = path_features(p, tc, d, l, BL, pitch, others, tdist=tdist, lo=tr["lo"], dhead=dhead, dwaist=dwaist)
        rt = f["root"] - hc
        f["h_a"] = float(rt @ ha) / BL * 100
        f["h_l"] = float(rt @ hl) / BL * 100
        f["h_d"] = float(rt @ hd) / BL * 100
        feats.append(f)
    return frame, feats


def same_limb_pairs(kept, pitch, BL, thr=0.4):
    """Pairs of candidates whose free parts overlap by >= thr of the shorter one (same limb twice)."""
    out = {}
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a = coverage_share(kept[i], kept[j], pitch)
            b = coverage_share(kept[j], kept[i], pitch)
            if max(a, b) >= thr:
                out[(i, j)] = (round(a, 3), round(b, 3))
    return out


def name_limbs(S1, C, pri):
    feats, kept = S1["feats"], S1["kept"]
    conflicts = set(S1["conflicts"].keys())
    names, scores = {}, {}
    margins = {}
    scores["_margins"] = margins
    iP = [i for i, p in enumerate(kept) if p["fam"] == "P"]
    iH = [i for i, p in enumerate(kept) if p["fam"] == "H"]
    if pri is None:
        best = assign_legs_heuristic(
            [
                i
                for i in iP
                if feats[i]["free_pct"] >= C["leg_min_free_pct"] and feats[i]["own_pct"] >= C["leg_min_own_pct"]
            ],
            feats,
        )
        for nm, i in best.items():
            names[i] = nm
        return names, scores
    # head limbs first (anatomically gated); a P path that is the same limb as a named head limb is not a leg
    if iH:
        ant, man, side = pa_name.head_scores([feats[i] for i in iH], pri)
        HN = ["AL", "AR", "ML", "MR"]
        S = np.c_[ant + side[:, 0], ant + side[:, 1], man + side[:, 0], man + side[:, 1]]
        conf_loc = {(iH.index(i), iH.index(j)) for i, j in conflicts if i in iH and j in iH}
        res = pa_name.hungarian(S, conf_loc, HN)
        hm = pa_name.hungarian_margins(S, conf_loc, HN, res)
        for nm, r in res.items():
            names[iH[r]] = nm
            margins[nm] = hm[nm]
        for r, i in enumerate(iH):
            scores[i] = {
                "antenna_logodds": round(float(ant[r]), 3),
                "mandible_logodds": round(float(man[r]), 3),
                "side_llr": side[r].round(2).tolist(),
                "score_by_name": dict(zip(HN, S[r].round(2).tolist())),
            }
    head_named = [i for i in names]
    blocked_P = set()
    if C.get("block_head_conflicts"):
        blocked_P = {i for i in iP for j in head_named if (i, j) in conflicts or (j, i) in conflicts}
    iP = [i for i in iP if i not in blocked_P]
    if iP:
        fP = [feats[i] for i in iP]
        lo, _, _ = pa_name.leg_scores(fP, pri)
        conf_loc = {(iP.index(i), iP.index(j)) for i, j in conflicts if i in iP and j in iP}
        res, tot, sl, sd, lm = pa_name.assign_legs_enum(
            fP,
            lo,
            pri,
            conf_loc,
            slot_clip=C.get("slot_clip"),
            name_penalty=C.get("name_penalty", pa_name.NAME_PENALTY),
        )
        for nm, r in res.items():
            names[iP[r]] = nm
            margins[nm] = lm[nm]
        for r, i in enumerate(iP):
            scores[i] = {
                "legness_logodds": round(float(lo[r]), 3),
                "slot_llr": [round(x, 2) for x in sl[r]],
                "side_llr": [round(x, 2) for x in sd[r]],
            }
        scores["_legs_total"] = round(float(tot), 3)
        for i in blocked_P:
            scores[i] = {"blocked": "same limb as a named head limb"}
    return names, scores


def assign_legs_heuristic(cands, feats):
    """Fallback without fitted priors: side from the root / coxa-end lateral offset, fore -> hind by the coxa end
    position along the mesosoma (t_cx), at most three per side."""
    names = {}
    for sd, sgn in (("L", 1), ("R", -1)):
        c = [i for i in cands if sgn * (feats[i]["y_root_pct"] + 0.5 * feats[i]["y_cx_pct"]) > 0]
        c = sorted(c, key=lambda i: -feats[i]["own_pct"])[:3]
        c = sorted(c, key=lambda i: -feats[i]["t_cx"])
        slots = [1, 2, 3] if len(c) == 3 else ([1, 3] if len(c) == 2 else [2])
        for k, i in zip(slots, c):
            names[f"{sd}{k}"] = i
    return names


# ====================================================================== status / confidence
TIP_MAX_LEG = 0.9  # % BL: a leg path ending in a thicker tip than this does not end at the tarsus (knee, stump)
PAIR_MIN = 0.7  # free arc / pair partner's free arc below this: shorter than its mirror leg


def limb_status(ch, i, names, info, margins, C, alt=None):
    """-> (status, confidence, flags, missing): identity confidence from the assignment margin; completeness and purity
    checks that need no truth (tip kind, mirror-pair arc ratio, a remaining contact-like neck)."""
    f = info[i]
    m = float(margins.get(ch, 0.0))
    c_id = 1.0 / (1.0 + np.exp(-(m - 1.0)))
    flags, missing = [], []
    partial = False
    if ch in LEGS:
        rt = f.get("r_tip10_pct")
        if rt is not None and rt > TIP_MAX_LEG:
            partial = True
            flags.append(f"thick tip (r {rt:.2f} % BL): the path does not end at the tarsus")
            missing.append("distal segments beyond the path tip (folded at the knee, contact or damage)")
        mate = ("R" if ch[0] == "L" else "L") + ch[1]
        by = {nm: k for k, nm in names.items()}
        if mate in by:
            ratio = f["free_um"] / max(info[by[mate]]["free_um"], 1e-9)
            f["pair_arc_ratio"] = round(float(ratio), 3)
            if ratio < PAIR_MIN:
                partial = True
                flags.append(f"free arc {100 * ratio:.0f} % of the mirror leg {mate}")
                if not missing:
                    missing.append("part of the limb (shorter than its mirror leg)")
    elif ch in ANTS:
        rt = f.get("r_tip10_pct")
        if rt is not None and rt < 0.45:
            flags.append(f"thin tip (r {rt:.2f} % BL): funiculus may end in a contact or be broken")
    sn = f.get("suspect_neck")
    c_pur = 1.0
    if sn and not sn.get("trunk_near"):
        c_pur = 0.6
        flags.append(
            f"contact-like neck left at {100 * sn['arc_frac']:.0f} % of the free arc (ratio {sn['ratio']}, turn {sn['turn_deg']})"
        )
    cd = f.get("cut_dist_rel")
    c_cut = 1.0
    if cd is not None and cd <= 2.0:
        c_cut = 0.7
        flags.append("path rerouted along a contact cut (check the region next to the cut)")
    c_alt = 1.0
    if alt:
        c_alt = 0.7
        flags.append(f"alternative route(s) {alt} share the proximal part and diverge distally (contact not resolved?)")
    conf = c_id * (0.6 if partial else 1.0) * c_pur * c_cut * c_alt
    status = "partial" if partial else "path"
    if c_id < C.get("ambiguous_cid", 0.0):
        status = "no_path"
        flags.append(f"ambiguous name (assignment margin {m:.2f})")
    return status, float(round(conf, 3)), flags, missing, float(round(c_id, 3)), m


# ====================================================================== output
EVID_KEYS = (
    "free_pct",
    "own_pct",
    "reach_pct",
    "tip_trunk_pct",
    "t_root",
    "t_cx",
    "y_root_pct",
    "y_cx_pct",
    "z_root_pct",
    "prox_y",
    "h_a",
    "h_l",
    "h_d",
    "r_tip10_pct",
    "r_max_pct",
)


def write_product(name, out_dir, S1, names, scores, C, timing, t0):
    tr, kept, info, frame, events = S1["tr"], S1["kept"], S1["feats"], S1["frame"], S1["events"]
    R, o, BL = tr["R"], tr["o"], tr["BL"]
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    arrays = {}
    limbs = {}
    by_name = {nm: i for i, nm in names.items()}
    for ch in LIMBS:
        if ch not in by_name:
            limbs[ch] = {
                "status": "no_path",
                "family": "P" if ch in LEGS else "H",
                "confidence": 0.0,
                "evidence": {"reason": "no candidate path has positive evidence for this name"},
            }
            continue
        i = by_name[ch]
        p = kept[i]
        f = info[i]
        Xs = pg.to_scan(p["X"], R, o)
        arrays[ch + "_X"] = Xs
        arrays[ch + "_s"] = p["s"]
        arrays[ch + "_r"] = p["r"]
        alt = []
        for a_, b_ in S1["conflicts"]:
            j = b_ if a_ == i else (a_ if b_ == i else None)
            if j is None or j in names or kept[j]["fam"] != p["fam"]:
                continue
            si, sj = scores.get(i, {}), scores.get(j, {})
            ki = "legness_logodds" if ch in LEGS else ("antenna_logodds" if ch in ANTS else "mandible_logodds")
            if si.get(ki) is not None and sj.get(ki) is not None and sj[ki] > si[ki] - 2.0:
                alt.append(int(j))
        status, conf, flags, missing, c_id, margin = limb_status(ch, i, names, info, scores.get("_margins", {}), C, alt)
        ev = {k: round(float(f[k]), 3) for k in EVID_KEYS if f.get(k) is not None}
        ev.update(
            {
                "candidate": i,
                "name_margin": round(margin, 3),
                "identity_confidence": c_id,
                "flags": flags,
                "suspect_neck": f.get("suspect_neck"),
                "cut_dist_vox": f.get("cut_dist_vox"),
                "cut_dist_rel": f.get("cut_dist_rel"),
                "pair_arc_ratio": f.get("pair_arc_ratio"),
            }
        )
        if i in scores:
            ev["scores"] = scores[i]
        if status == "no_path":
            arrays.pop(ch + "_X"), arrays.pop(ch + "_s"), arrays.pop(ch + "_r")
            limbs[ch] = {
                "status": "no_path",
                "family": p["fam"],
                "confidence": 0.0,
                "evidence": dict(ev, reason="ambiguous name; candidate kept in `candidates`"),
            }
            continue
        limbs[ch] = {
            "status": status,
            "family": p["fam"],
            "n_points": int(len(Xs)),
            "arc_um": float(p["s"][-1]),
            "core_arc_um": float(p["s_main_end"]),
            "tip_scan": Xs[0].tolist(),
            "root_scan": pg.to_scan(f["root"], R, o).tolist(),
            "confidence": conf,
            "missing": missing,
            "evidence": ev,
        }
    T = tr["T"]
    if T:
        arrays["T_X"] = pg.to_scan(T["X"], R, o)
        if T.get("s") is not None:
            arrays["T_s"] = T["s"]
        if T.get("part") is not None:
            arrays["T_part"] = T["part"]
    others = []
    cand_rec = []
    for i, p in enumerate(kept):
        f = info[i]
        rec = {
            "k": i,
            "fam": p["fam"],
            "name": names.get(i),
            "tip_scan": pg.to_scan(p["X"][0], R, o).tolist(),
            "root_scan": pg.to_scan(f["root"], R, o).tolist(),
            "core_arc_um": float(p["s_main_end"]),
        }
        rec.update(
            {
                k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                for k, v in f.items()
                if not isinstance(v, np.ndarray) and v is not None
            }
        )
        if i in scores:
            rec["scores"] = scores[i]
        cand_rec.append(rec)
        arrays[f"C{i}_X"] = pg.to_scan(p["X"], R, o)
        arrays[f"C{i}_s"] = p["s"]
        arrays[f"C{i}_r"] = p["r"]
        if i not in names:
            others.append(
                {"k": i, "fam": p["fam"], "free_pct": round(f["free_pct"], 1), "own_pct": round(f["own_pct"], 1)}
            )
    fr = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in frame.items()}
    fr_scan = {
        k: (np.asarray(v) @ R).tolist() for k, v in frame.items() if k in ("anterior", "dorsal", "left", "head_axis")
    }
    out = {
        "name": name,
        "units": "um",
        "version": VERSION,
        "source": "PATHS-A geodesic paths with contact cutting",
        "frame": {"rotation_scan_to_canonical": R.tolist(), "origin_scan": o.tolist()},
        "anatomy_axes_scan": fr_scan,
        "anatomy_axes_working": fr,
        "BL_um": BL,
        "BL_source": "trunk product",
        "limbs": limbs,
        "other_paths": others,
        "candidates": cand_rec,
        "same_limb_pairs": [[int(i), int(j), *v] for (i, j), v in S1["conflicts"].items()],
        "contacts_cut": [
            {k: v for k, v in e.items() if k not in ("c",)}
            | {"c_working_um": (np.asarray(e["c"]) * tr["pitch"] + tr["lo"]).tolist()}
            for e in events
        ],
        "dropped_paths": S1["dropped"],
        "n_candidates": S1["n_candidates"],
        "cfg": C,
        "timing_s": timing,
        "seconds": time.time() - t0,
    }
    json.dump(out, open(os.path.join(od, "limbs.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(od, "limbs.npz"), **arrays)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--names-file", default=None)
    ap.add_argument("--mesh-tmpl", required=True, help="envelope mesh path with {name}")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--cfg", default=None, help="JSON dict of CFG overrides")
    a = ap.parse_args()
    if a.names:
        names = a.names.split(",")
    elif a.names_file:
        names = [s.strip() for s in open(a.names_file) if s.strip()]
    else:
        names = sorted(n for n in os.listdir(a.trunk_dir) if os.path.isfile(os.path.join(a.trunk_dir, n, "trunk.npz")))
    cfg = json.loads(a.cfg) if a.cfg else None
    todo = [n for n in names if not (a.skip_existing and os.path.isfile(os.path.join(a.out_dir, n, "limbs.json")))]
    jobs = [(n, a.trunk_dir, a.mesh_tmpl.format(name=n), a.out_dir, cfg) for n in todo]
    if a.workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(_job, jobs):
                print(r, flush=True)
    else:
        for j in jobs:
            print(_job(j), flush=True)


def _job(args):
    n, trunk_dir, mesh, out_dir, cfg = args
    t0 = time.time()
    try:
        out = run(n, trunk_dir, mesh, out_dir, cfg)
        st = {k: v["status"] for k, v in out["limbs"].items()}
        return f"{n:48s} ok {time.time() - t0:5.1f}s cuts {len(out['contacts_cut'])} named " + " ".join(
            k for k in LIMBS if st[k] != "no_path"
        )
    except Exception as e:
        os.makedirs(os.path.join(out_dir, n), exist_ok=True)
        json.dump(
            {"name": n, "status": "failed", "error": repr(e), "traceback": traceback.format_exc()},
            open(os.path.join(out_dir, n, "failed.json"), "w"),
            indent=1,
        )
        return f"{n:48s} FAILED {e!r}"


if __name__ == "__main__":
    main()
