"""S7 THIN-LIMBS, part 2: discovery of limbs that no PATHS approach found (thin antennae / legs that dropped out of
every voxel grid entirely), on the mesh surface.

  1. coverage: a mesh vertex is covered by the trunk when it lies within CORE_MARGIN voxels of TRUNK's opened trunk core
     (tpart >= 0), and by a limb when it lies within 1.8 x the local radius of any limb polyline of the (extended)
     PATHS A / B / C products.
  2. the uncovered vertices split into connected surface patches; a patch that touches the trunk-covered region is a
     protrusion of the body; patches shorter than DET_MIN_PCT (geodesic inside the patch) are ignored.
  3. from the patch's trunk attachment, a surface geodesic is grown over the whole mesh except the trunk-covered region
     (so a limb whose surface is crossed by another limb's coverage near a contact is still followed), and traced with
     the same iso-contour tracker as the extension (thin.MeshGraph.track): ring centres from the attachment out to the
     free tip. Kept: traces that end at a free tip, are >= DISC_MIN_PCT long and do not lie on an existing limb
     polyline (share <= OVERLAP_MAX); traces ending at the same tip are one limb (the longer is kept).
  4. naming, only for limb names that are no_path in every PATHS product: attached to the head and >= ANT_MIN_PCT long
     -> antenna, side from TRUNK's anatomical frame (left = +Y of the attachment relative to the trunk path); attached
     to the mesosoma / waist and >= LEG_MIN_PCT -> leg, side as above, fore / mid / hind by the attachment's order
     along the trunk axis among this side's named legs (a slot that is already taken is not overwritten).
  5. the discovered limb is written into ONE approach product (the first available in A, C, B order), so that the S6
     combine sees one unsupported candidate: its limb confidence stays <= 0.3 (min(0.3, 0.3 x own), own = 0.5).
Nothing here reads truth, oracle limbs, development-set geometry or specimen names; thresholds are geometric, hand-set, in % BL.
"""

import json
import os

import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

import thin as TH

DCFG = {
    "core_margin_vox": 2.0,  # trunk coverage: within this many trunk voxels of the opened core
    "cover_r_factor": 1.8,  # limb coverage: within this x the local radius of a polyline point
    "cover_add_vox": 1.0,  # ... plus this many trunk voxels
    "cover_max_pct": 3.0,  # cover radius cap (% BL)
    "det_min_pct": 4.0,  # an attached uncovered patch at least this long (inside the patch) is traced (% BL)
    "disc_min_pct": 12.0,  # shortest traced limb kept (% BL)
    "overlap_max": 0.3,  # a trace lying on existing limb polylines for more than this share is a duplicate
    "overlap_tol_pct": 1.0,  # ... within this distance (% BL)
    "dup_tip_pct": 2.0,  # two traces with tips closer than this are the same limb (the longer is kept)
    "ant_min_pct": 18.0,  # head patch at least this long: antenna
    "leg_min_pct": 20.0,  # mesosoma / waist patch at least this long: leg
    "root_ext_pct": 4.0,  # straight root extension towards the attached part's core centroid (% BL)
    "own_conf": 0.5,  # the inserted candidate's own confidence (combine caps an unsupported limb at 0.3 x own)
}


def core_maps(trunk):
    tp = trunk["tpart"]
    core = tp >= 0
    dt, ind = ndimage.distance_transform_edt(~core, return_indices=True)
    lab = tp[tuple(ind)]
    return dt * trunk["pitch"], lab


def lookup(grid, trunk, P, default):
    W = (np.atleast_2d(P) - trunk["o"]) @ trunk["R"].T
    q = np.round((W - trunk["lo"]) / trunk["pitch"]).astype(int)
    ok = np.all((q >= 0) & (q < np.array(grid.shape)), axis=1)
    out = np.full(len(q), default, dtype=float)
    out[ok] = grid[tuple(q[ok].T)]
    return out


def coverage(mg, trunk, polys, BL, cfg):
    dcore, lab = core_maps(trunk)
    pitch = trunk["pitch"]
    d = lookup(dcore, trunk, mg.V, np.inf)
    part = lookup(lab, trunk, mg.V, -1).astype(int)
    trunk_cov = d <= cfg["core_margin_vox"] * pitch
    limb_cov = np.zeros(mg.n, bool)
    rmax = cfg["cover_max_pct"] / 100 * BL
    for X in polys:
        X = np.asarray(X, float)
        s = TH.arclen(X)
        if s[-1] <= 0:
            continue
        step = max(pitch, 0.25 / 100 * BL)
        P = np.stack([np.interp(np.arange(0, s[-1] + step, step), s, X[:, k]) for k in range(3)], 1)
        r, _ = mg.tree.query(P)
        rad = np.minimum(cfg["cover_r_factor"] * r + cfg["cover_add_vox"] * pitch, rmax)
        for p, rr in zip(P, rad):
            idx = mg.tree.query_ball_point(p, rr)
            if idx:
                limb_cov[idx] = True
    return trunk_cov, limb_cov, part


def patches(mg, covered):
    """Connected components of the uncovered vertices (labels, -1 covered)."""
    unc = ~covered
    rows = np.repeat(np.arange(mg.n), np.diff(mg.indptr))
    m = unc[rows] & unc[mg.nbr]
    A = sparse.csr_matrix((np.ones(int(m.sum())), (rows[m], mg.nbr[m])), shape=(mg.n, mg.n))
    ncomp, lab = sparse.csgraph.connected_components(A, directed=False)
    lab = np.where(unc, lab, -1)
    return lab


def discover(mg, trunk, polys, BL, cfg_thin, cfg=None, in_trunk=None):
    """Candidate limbs on the mesh not covered by the trunk or any limb polyline. Returns a list of dicts."""
    cfg = dict(DCFG, **(cfg or {}))
    trunk_cov, limb_cov, part = coverage(mg, trunk, polys, BL, cfg)
    covered = trunk_cov | limb_cov
    lab = patches(mg, covered)
    rows = np.repeat(np.arange(mg.n), np.diff(mg.indptr))
    # boundary: uncovered vertex with a trunk-covered (not limb-covered) neighbour
    bmask = (lab[rows] >= 0) & trunk_cov[mg.nbr] & ~limb_cov[mg.nbr]
    att = {}
    for v, w in zip(rows[bmask], mg.nbr[bmask]):
        att.setdefault(int(lab[v]), []).append((int(v), int(part[w])))
    sizes = np.bincount(lab[lab >= 0])
    cands = []
    h = cfg_thin["step_pct"] / 100 * BL
    for c, lst in att.items():
        if sizes[c] < 20:
            continue
        src = np.unique([v for v, _ in lst])
        parts = [p for _, p in lst if p >= 0]
        if not parts:
            continue
        pmaj = int(np.bincount(parts).argmax())
        # patch length: geodesic inside the patch from its trunk attachment (edges leaving the patch blocked)
        inside = lab == c
        data = mg.data.copy()
        data[~(inside[rows] & inside[mg.nbr])] = 1e18
        W = sparse.csr_matrix((data, mg.nbr, mg.indptr), shape=(mg.n, mg.n))
        g = dijkstra(W, directed=True, indices=src, min_only=True, limit=cfg_thin["max_ext_pct"] / 100 * BL + 2 * h)
        gin = g[inside]
        ext = float(np.nanmax(np.where(np.isfinite(gin), gin, np.nan))) if np.isfinite(gin).any() else 0.0
        rec = {
            "patch": int(c),
            "n_vertices": int(sizes[c]),
            "attach_part": pmaj,
            "patch_pct": round(100 * ext / BL, 2),
            "n_attach": int(len(src)),
        }
        if ext < cfg["det_min_pct"] / 100 * BL:
            continue
        # trace geodesic over everything but the trunk-covered region
        data = mg.data.copy()
        data[trunk_cov[rows] | trunk_cov[mg.nbr]] = 1e18
        W = sparse.csr_matrix((data, mg.nbr, mg.indptr), shape=(mg.n, mg.n))
        g = dijkstra(W, directed=True, indices=src, min_only=True, limit=cfg_thin["max_ext_pct"] / 100 * BL + 2 * h)
        g = TH.behind(mg, g, src)
        # starting radius: the first contour above the attachment
        first = mg.contours(g - h, mg.faces_of(src), max_faces=20000)
        r0 = float(np.median([f["radius"] for f in first])) if first else 0.01 * BL
        tr = mg.track(g, src, r0, BL, cfg_thin, in_trunk=None)
        rec.update(
            stop=tr["stop"],
            levels=tr["levels"],
            r0_um=round(r0, 2),
            r_median_um=round(float(np.median(tr["radii"])), 2) if len(tr["radii"]) else None,
        )
        if not str(tr["stop"]).startswith("tip") or tr["tip"] is None or len(tr["centres"]) < 3:
            rec["kept"] = False
            cands.append(rec)
            continue
        C = np.vstack([tr["centres"], tr["tip"][None]])
        free = C[::-1]  # tip first, ends at the attachment
        core_arc = float(TH.arclen(free)[-1])
        rec["extent_pct"] = round(100 * core_arc / BL, 2)
        if core_arc < cfg["disc_min_pct"] / 100 * BL:
            rec.update(kept=False, why="shorter than disc_min_pct")
            cands.append(rec)
            continue
        if polys:
            sf = TH.arclen(free)
            step = 0.5 / 100 * BL
            S = np.stack([np.interp(np.arange(0, sf[-1], step), sf, free[:, k]) for k in range(3)], 1)
            near = np.zeros(len(S), bool)
            for Y in polys:
                near |= TH.dist_poly(S, Y) <= cfg["overlap_tol_pct"] / 100 * BL
            rec["overlap"] = round(float(near.mean()), 3)
            if near.mean() > cfg["overlap_max"]:
                rec.update(kept=False, why="lies on an existing limb polyline")
                cands.append(rec)
                continue
        # root: straight towards the attached part's core centroid (the S6 paths enter the core the same way)
        cc = np.argwhere(trunk["tpart"] == pmaj)
        if len(cc):
            cen = (trunk["lo"] + cc.mean(0) * trunk["pitch"]) @ trunk["R"] + trunk["o"]
            dv = cen - free[-1]
            L = min(cfg["root_ext_pct"] / 100 * BL, 0.85 * float(np.linalg.norm(dv)))
            m = max(2, int(np.ceil(L / h)))
            root = free[-1] + np.outer(np.arange(1, m + 1) * (L / m), dv / max(np.linalg.norm(dv), 1e-9))
            X = np.vstack([free, root])
        else:
            X = free
        rec.update(
            kept=True,
            X=X,
            core_arc_um=core_arc,
            attach_scan=free[-1].tolist(),
            tip_scan=free[0].tolist(),
            radii=tr["radii"],
        )
        cands.append(rec)
    # traces ending at the same tip are one limb: keep the longer
    keep = sorted([c for c in cands if c.get("kept")], key=lambda c: -c["core_arc_um"])
    for i, c in enumerate(keep):
        for c2 in keep[:i]:
            if (
                c2.get("kept")
                and np.linalg.norm(np.subtract(c["tip_scan"], c2["tip_scan"])) <= cfg["dup_tip_pct"] / 100 * BL
            ):
                c.update(kept=False, why=f"same tip as patch {c2['patch']}")
                break
    return cands, {
        "n_trunk_cov": int(trunk_cov.sum()),
        "n_limb_cov": int(limb_cov.sum()),
        "n_patches_attached": len(att),
    }


def name_candidates(cands, trunk, present, roots, BL, cfg=None):
    """present: set of limb names that some product has; roots: {leg name: root point (scan)} of present legs.
    Returns {name: cand}."""
    cfg = dict(DCFG, **(cfg or {}))
    fr = trunk["json"].get("anatomical_frame_final") or trunk["json"]["frame"]
    Rf = np.asarray(fr["rotation_scan_to_canonical"], float)
    Tx = trunk.get("T_X")
    out = {}

    def side(p):
        p = np.asarray(p, float)
        if Tx is not None and len(Tx):
            q = Tx[int(np.argmin(np.linalg.norm(Tx - p, axis=1)))]
        else:
            q = np.asarray(fr["origin_scan"], float)
        return float((p - q) @ Rf[1])

    keep = [c for c in cands if c.get("kept")]
    # antennae
    ants = [c for c in keep if c["attach_part"] == 0 and c["extent_pct"] >= cfg["ant_min_pct"]]
    for nm, sg in (("AL", 1), ("AR", -1)):
        if nm in present:
            continue
        mine = [c for c in ants if np.sign(side(c["attach_scan"])) == sg and "name" not in c]
        if mine:
            c = max(mine, key=lambda x: x["extent_pct"])
            c["name"], c["side_um"] = nm, side(c["attach_scan"])
            out[nm] = c
    # legs
    legs = [c for c in keep if c["attach_part"] in (1, 2) and c["extent_pct"] >= cfg["leg_min_pct"]]
    for lab, sg in (("L", 1), ("R", -1)):
        mine = [c for c in legs if np.sign(side(c["attach_scan"])) == sg]
        missing = [f"{lab}{i}" for i in (1, 2, 3) if f"{lab}{i}" not in present]
        if not mine or not missing:
            continue
        have = [
            (float(np.asarray(roots[f"{lab}{i}"]) @ Rf[0]), f"{lab}{i}", None)
            for i in (1, 2, 3)
            if f"{lab}{i}" in present and f"{lab}{i}" in roots
        ]
        allx = sorted(
            have + [(float(np.asarray(c["attach_scan"]) @ Rf[0]), None, c) for c in mine], key=lambda x: -x[0]
        )
        for rank, (x, nm, c) in enumerate(allx[:3]):
            slot = f"{lab}{rank + 1}"
            if c is not None and slot in missing and slot not in out:
                c["name"], c["side_um"] = slot, side(c["attach_scan"])
                out[slot] = c
    return out


def insert(work, name, order, named, BL, cfg=None):
    """Write the named candidates into the first available approach product (order) under work/paths_X/name."""
    cfg = dict(DCFG, **(cfg or {}))
    for a in order:
        d = os.path.join(work, f"paths_{a}", name)
        fj, fz = os.path.join(d, "limbs.json"), os.path.join(d, "limbs.npz")
        if not (os.path.isfile(fj) and os.path.isfile(fz)):
            continue
        sd = os.path.join(d, "stage_done.json")
        if os.path.isfile(sd) and json.load(open(sd)).get("status") != "ok":
            continue
        meta = json.load(open(fj))
        z = np.load(fz)
        arrays = {k: z[k] for k in z.files}
        for nm, c in named.items():
            X = np.asarray(c["X"], float)
            arrays[nm + "_X"], arrays[nm + "_s"] = X, TH.arclen(X)
            r = np.full(len(X), np.nan)
            arrays[nm + "_r"] = r
            meta["limbs"][nm] = {
                "status": "path",
                "confidence": cfg["own_conf"],
                "core_arc_um": c["core_arc_um"],
                "arc_um": float(TH.arclen(X)[-1]),
                "n_points": int(len(X)),
                "tip_scan": X[0].tolist(),
                "root_scan": X[-1].tolist(),
                "missing": [],
                "evidence": f"S7 discovery: uncovered surface patch attached to trunk part {c['attach_part']}, "
                f"{c['extent_pct']} % BL long, traced to a free tip on the mesh (not found by any PATHS approach)",
                "thin_discovered": {k: v for k, v in c.items() if k not in ("X", "radii")},
            }
        meta.setdefault("thin", {})["discovered"] = sorted(named)
        json.dump(meta, open(fj, "w"), indent=1, default=float)
        np.savez_compressed(fz, **arrays)
        return a
    return None
