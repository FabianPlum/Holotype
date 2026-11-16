"""S7 THIN-LIMBS: mesh-native extension of limb centrelines through parts thinner than the voxel grid.

The round-6 limb centrelines (PATHS A / B / C) live on voxel grids of pitch BL/350 to BL/700. A limb segment thinner
than about one pitch (synthetic tarsi: radius ~0.2 % BL; antennal funiculi ~0.5 % BL) drops out of the occupancy, the
path ends where the tube became too thin, and ALONG then reads that end as the leg tip. The envelope mesh itself still
carries the thin tube. This module follows the tube on the mesh surface, independent of any grid:

  1. cut: at the path tip (or a little behind it) the mesh is sectioned by the plane normal to the path's end tangent;
     the closed section loop that contains the centreline (else the nearest closed loop) is the limb's cross-section.
  2. the mesh edges crossing that loop are removed and a surface geodesic distance g (Dijkstra on the mesh edge graph)
     is grown from the loop's distal side: g is the arc along the limb's surface beyond the cut.
  3. iso-contours of g (every STEP_PCT % BL) are exact piecewise-linear level sets on the mesh; on a tube each is one
     closed ring around the axis, its length-weighted centroid is a centreline point and its mean distance to the
     centroid the local radius. Contours are followed level by level (only the ones reachable from the previous ring
     through the band between them), so bends at joints are followed without any direction search.
  4. stop when the rings vanish (the tip: the vertex of largest g in the last band), when a ring is much wider than
     the recent rings (the tube merged into another part: a contact or the body), when the ring splits into two long
     branches (ambiguous), when the ring centre enters the trunk core, or after MAX_EXT_PCT.
  5. an extension is kept only when the trace ended at a free tip, it adds at least MIN_GAIN_PCT of arc beyond the old
     tip, does not run along another limb polyline of any approach (a contact) or into the trunk core, and the cut section is
     not much wider than the limb section BACK_CHECK_PCT behind it (a limb tapers towards its tip; a wider section at
     the path tip means the tip sits in a fused contact, and the trace would follow the other part). Legs and antennae
     only (mandibles are not thin limbs).

Inputs per specimen: a round-6 trunk product (trunk.json / trunk.npz: BL, working frame, trunk core labels) and limb
products (limbs.json / limbs.npz of PATHS A / B / C, round-6 limb layout), the envelope mesh (scan frame, um).
Output: the same limb products with extended polylines (tip first), `core_arc_um` increased by the added arc, an
`thin_extension` evidence record per limb, and (status rule, see `restatus`) 'partial' limbs whose only missing part was
the distal part and whose new free arc matches their bilateral partner set to 'path'.

No truth, no oracle limbs, no development-set geometry, no specimen names, nothing fitted: the constants below are geometric and
hand-set (step, tolerances), stated in % BL or as ratios.

  python thin.py extend NAME --s6-work W6 --mesh M --work W7 [--approaches A,B,C]
     reads W6/trunk/NAME, W6/paths_X/NAME; writes W7/paths_X/NAME/limbs.{json,npz} + stage_done.json, links W7/trunk/NAME
"""

import argparse
import json
import os
import sys
import time
import traceback

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
sys.dont_write_bytecode = True

import numpy as np
import trimesh
from scipy import ndimage
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

VERSION = "S7 thin 2026-11-11"
LEGS = [f"{s}{i}" for s in "LR" for i in (1, 2, 3)]
LIMBS = LEGS + ["AL", "AR", "ML", "MR"]
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

CFG = {
    "step_pct": 0.25,  # contour spacing along the surface geodesic (% BL)
    "max_ext_pct": 70.0,  # longest extension (% BL): a hind tibia + tarsus is ~55 % BL
    "cut_back_pct": (0.0, 0.5, 1.0, 2.0, 3.0),  # cut positions tried, arc behind the path tip (% BL)
    "dir_pct": 1.5,  # end tangent over this arc (% BL)
    "loop_ball_pct": 4.0,  # the cut loop is searched within this radius of the cut point (% BL)
    "loop_max_r_pct": 3.0,  # a section loop wider than this (mean radius, % BL) is not a limb section
    "loop_near": 2.5,  # a loop not containing the cut point is accepted within this x its radius
    "min_gain_pct": 3.0,  # an extension must add this much arc beyond the old tip (% BL): ALONG's own tip march
    # (geom_profile TIP_MAX_PCT) already covers up to 3 % BL; shorter gains are claw choices
    "jump_factor": 1.8,  # ring radius > jump_factor x recent median + jump_add_pct % BL: merged with another part
    "jump_add_pct": 0.25,
    "jump_hist": 6,  # rings in the recent median
    "long_branch_pct": 1.5,  # a branch reaching this far beyond its split (% BL) is long (claws are short)
    "branch_cap": 40000,  # forward-flood cap (vertices) when sizing a branch
    "contact_tol_pct": 1.0,  # an extension sample this close to another limb's polyline is on that limb (% BL)
    "contact_run_pct": 2.0,  # ... for this much consecutive arc: the extension is cut there (a contact)
    "partner_ratio": 0.8,  # partial -> path when the only missing part was distal and arc >= this x partner arc
    "smooth_levels": 1.5,  # Gaussian smoothing of the ring centres (levels)
    "require_tip": True,  # keep an extension only when the trace ended at a free tip (not at a merge / branch)
    "back_check_pct": 2.0,  # the limb section this far (or 3 x the cut radius) behind the cut ...
    "cut_wide_factor": 1.5,  # ... must not be narrower than cut radius / factor - add: else the cut is in a contact
    "cut_wide_add_pct": 0.1,  #     (a limb tapers towards its tip; a wider section at the tip is a fused contact)
    "limbs": ("L1", "L2", "L3", "R1", "R2", "R3", "AL", "AR"),  # thin limbs: legs and antennae (not mandibles)
    "discover": True,  # part 2 (discover.py): limbs no approach found, traced from uncovered surface patches
    "order": ("A", "C", "B"),  # the S6 combine order; a discovered limb goes into the first available product
}


# ------------------------------------------------------------------------------------------------ mesh graph
class MeshGraph:
    """Envelope mesh as an edge graph (CSR, edge lengths) with face <-> edge maps for exact level-set contours."""

    def __init__(self, mesh_path):
        tm = trimesh.load(mesh_path, process=True, force="mesh")
        self.tm = tm
        self.V = np.asarray(tm.vertices, float)
        self.F = np.asarray(tm.faces, np.int64)
        self.E = np.asarray(tm.edges_unique, np.int64)
        self.L = np.asarray(tm.edges_unique_length, float)
        n, m = len(self.V), len(self.E)
        self.n = n
        self.FE = np.asarray(tm.faces_unique_edges, np.int64)
        EF = np.full((m, 2), -1, np.int64)
        fe = self.FE.ravel()
        ff = np.repeat(np.arange(len(self.F)), 3)
        o = np.argsort(fe, kind="stable")
        fe_s, ff_s = fe[o], ff[o]
        first = np.r_[True, fe_s[1:] != fe_s[:-1]]
        EF[fe_s[first], 0] = ff_s[first]
        EF[fe_s[~first], 1] = ff_s[~first]
        self.EF = EF
        rows = np.r_[self.E[:, 0], self.E[:, 1]]
        cols = np.r_[self.E[:, 1], self.E[:, 0]]
        eid = np.r_[np.arange(m), np.arange(m)]
        o = np.argsort(rows, kind="stable")
        self.nbr, self.nbr_e = cols[o], eid[o]
        self.indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
        self.data = self.L[self.nbr_e].copy()
        pos = np.argsort(self.nbr_e, kind="stable")  # two CSR positions per edge
        self.edge_pos = pos.reshape(m, 2)
        self.tree = cKDTree(self.V)
        vf = np.asarray(tm.vertex_faces)
        self.vf = vf
        self.med_edge = float(np.median(self.L))
        self.p90_edge = float(np.percentile(self.L, 90))

    def neighbours(self, v):
        return self.nbr[self.indptr[v] : self.indptr[v + 1]]

    def faces_of(self, verts):
        f = self.vf[np.asarray(list(verts), np.int64)].ravel()
        return np.unique(f[f >= 0])

    # -------------------------------------------------------------------------------------------- level-set contours
    def contours(self, val, seed_faces, allowed=None, max_faces=20000):
        """Connected components of the zero level set of the per-vertex function `val` that pass through the seed
        faces (BFS from face to face across crossing edges). allowed: set of faces the walk may enter (else open).
        Returns a list of dicts: faces, centroid, perimeter, radius (mean distance of the contour to its centroid),
        segments (in, out points; oriented consistently), closed, huge (max_faces hit)."""
        F, FE, E, EF = self.F, self.FE, self.E, self.EF
        neg = val < 0
        fin = np.isfinite(val)  # a face touching an unreachable vertex (inf) is a boundary, never part of a contour
        seen = set()
        comps = []
        for f0 in seed_faces:
            f0 = int(f0)
            if f0 in seen:
                continue
            s = neg[F[f0]]
            if s.all() or not s.any() or not fin[F[f0]].all():
                continue
            comp, q, open_, huge = [], [f0], False, False
            seen.add(f0)
            while q:
                f = q.pop()
                comp.append(f)
                if len(comp) > max_faces:
                    huge = True
                    break
                for e in FE[f]:
                    a, b = E[e]
                    if neg[a] == neg[b]:
                        continue
                    for g in EF[e]:
                        g = int(g)
                        if g < 0:
                            open_ = True
                            continue
                        if g == f or g in seen:
                            continue
                        if (allowed is not None and g not in allowed) or not fin[F[g]].all():
                            open_ = True
                            continue
                        seen.add(g)
                        q.append(g)
            c = self._geom(np.array(comp, np.int64), val)
            c.update(closed=not open_ and not huge, huge=huge)
            comps.append(c)
        return comps

    def _geom(self, comp, val):
        Fc = self.F[comp]
        vv = val[Fc]
        ng = vv < 0
        pin = np.zeros((len(comp), 3))
        pout = np.zeros((len(comp), 3))
        for j in range(3):
            a, b = Fc[:, j], Fc[:, (j + 1) % 3]
            cr = ng[:, j] != ng[:, (j + 1) % 3]
            va, vb = vv[:, j], vv[:, (j + 1) % 3]
            t = np.where(cr, va / np.where(cr, va - vb, 1.0), 0.0)
            P = self.V[a] + t[:, None] * (self.V[b] - self.V[a])
            enter = cr & ng[:, j]  # neg -> pos along the face's edge order
            leave = cr & ~ng[:, j]
            pin[enter] = P[enter]
            pout[leave] = P[leave]
        mid = 0.5 * (pin + pout)
        ln = np.linalg.norm(pout - pin, axis=1)
        w = ln.sum()
        cen = (mid * ln[:, None]).sum(0) / w if w > 0 else mid.mean(0)
        rad = float((np.linalg.norm(mid - cen, axis=1) * ln).sum() / w) if w > 0 else 0.0
        up = np.unique(Fc[~ng])
        return {
            "faces": comp,
            "centroid": cen,
            "perimeter": float(w),
            "radius": rad,
            "pin": pin,
            "pout": pout,
            "upper": up,
        }

    @staticmethod
    def winding(comp, c, u):
        """Winding number (turns) of the oriented contour around point c, seen along the plane normal u."""
        a = comp["pin"] - c
        b = comp["pout"] - c
        a = a - np.outer(a @ u, u)
        b = b - np.outer(b @ u, u)
        ang = np.arctan2(np.cross(a, b) @ u, (a * b).sum(1))
        return float(ang.sum() / (2 * np.pi))

    def plane_loop(self, c, u, BL, cfg):
        """The closed section loop of the plane (c, u) that contains c (else the nearest closed loop within
        loop_near x its radius); None when there is none."""
        R = max(cfg["loop_ball_pct"] / 100 * BL, 6 * self.p90_edge)
        vids = self.tree.query_ball_point(c, R)
        if not vids:
            return None
        fl = self.faces_of(vids)
        val = (self.V - c) @ u
        loops = self.contours(val, fl, allowed=set(fl.tolist()), max_faces=5000)
        best = None
        rmax = cfg["loop_max_r_pct"] / 100 * BL
        for lp in loops:
            if not lp["closed"] or lp["radius"] > rmax or lp["perimeter"] <= 0:
                continue
            wn = abs(self.winding(lp, c, u))
            d = float(np.linalg.norm(lp["centroid"] - c))
            inside = wn > 0.5
            ok = inside or d <= cfg["loop_near"] * max(lp["radius"], self.med_edge)
            if not ok:
                continue
            key = (0 if inside else 1, lp["radius"] if inside else d)
            if best is None or key < best[0]:
                best = (key, lp, inside, d)
        if best is None:
            return None
        lp = best[1]
        lp["contains"] = best[2]
        lp["dist"] = best[3]
        lp["val"] = val
        return lp

    # -------------------------------------------------------------------------------------------- tracing
    def geodesic_beyond(self, loop, limit):
        """Surface geodesic distance from the distal side of a cut loop, with the loop's crossing edges removed."""
        val = loop["val"]
        Fc = self.F[loop["faces"]]
        cut = set()
        for f in loop["faces"]:
            for e in self.FE[f]:
                a, b = self.E[e]
                if (val[a] < 0) != (val[b] < 0):
                    cut.add(int(e))
        cut = np.array(sorted(cut), np.int64)
        src = np.unique(np.where(val[self.E[cut, 0]] >= 0, self.E[cut, 0], self.E[cut, 1]))
        data = self.data.copy()
        data[self.edge_pos[cut].ravel()] = 1e18
        W = sparse.csr_matrix((data, self.nbr, self.indptr), shape=(self.n, self.n))
        g = dijkstra(W, directed=True, indices=src, min_only=True, limit=limit)
        g = behind(self, g, src)
        return g, src, Fc

    def forward_extent(self, start, g, t, cap):
        """Flood from vertices `start` through vertices with g >= t: (max g reached, vertex of it, size, capped)."""
        seen = set(int(v) for v in start if np.isfinite(g[v]) and g[v] >= t)
        q = list(seen)
        best, bv = -1.0, None
        while q:
            v = q.pop()
            if g[v] > best:
                best, bv = float(g[v]), v
            if len(seen) > cap:
                return best, bv, len(seen), True
            for w in self.neighbours(v):
                w = int(w)
                if w not in seen and np.isfinite(g[w]) and g[w] >= t:
                    seen.add(w)
                    q.append(w)
        return best, bv, len(seen), False

    def trace(self, c, u, BL, cfg, in_trunk=None, loop=None):
        """Follow the tube beyond the cut at c (plane normal u, pointing distally). Returns dict with centres
        (m, 3) from the cut outwards, radii, tip (3,) or None, stop reason, levels, cut loop radius."""
        if loop is None:
            loop = self.plane_loop(c, u, BL, cfg)
        if loop is None:
            return {"ok": False, "stop": "no closed section loop at the cut"}
        h = cfg["step_pct"] / 100 * BL
        limit = cfg["max_ext_pct"] / 100 * BL
        g, src, _ = self.geodesic_beyond(loop, limit + 2 * h)
        out = self.track(g, src, loop["radius"], BL, cfg, in_trunk)
        out.update(cut_contains=bool(loop.get("contains")), cut_dist=float(loop.get("dist", 0.0)))
        return out

    def track(self, g, src, r0, BL, cfg, in_trunk=None):
        """Follow the iso-contours of the surface geodesic g outwards from the source vertices (module docstring,
        steps 3-4). r0: radius of the starting section."""
        h = cfg["step_pct"] / 100 * BL
        limit = cfg["max_ext_pct"] / 100 * BL
        jump_add = cfg["jump_add_pct"] / 100 * BL
        long_br = cfg["long_branch_pct"] / 100 * BL
        prev_upper = set(int(v) for v in src)
        cen, rad, info = [], [], []
        hist = [r0]
        stop, tip = None, None
        K = int(np.ceil(limit / h))
        t_prev = 0.0
        last_region = set(prev_upper)
        for k in range(1, K + 1):
            t = k * h
            # band region between the previous ring and this level, reachable from the previous ring
            q = [v for v in prev_upper if np.isfinite(g[v]) and g[v] < t]
            region = set(q)
            while q:
                v = q.pop()
                for w in self.neighbours(v):
                    w = int(w)
                    if w not in region and np.isfinite(g[w]) and t_prev <= g[w] < t:
                        region.add(w)
                        q.append(w)
            seeds = self.faces_of(region | prev_upper) if (region or prev_upper) else np.zeros(0, np.int64)
            comps = self.contours(g - t, seeds, max_faces=20000)
            if region:
                last_region = region  # noqa: F841
            if not comps:
                pool = list(region | prev_upper)
                gv = np.array([g[v] for v in pool])
                tip = self.V[pool[int(np.argmax(gv))]].copy()
                stop = "tip"
                break
            if len(comps) > 1:
                ext = []
                for cp in comps:
                    mg, mv, sz, capped = self.forward_extent(cp["upper"], g, t, cfg["branch_cap"])
                    ext.append((mg - t, mv, sz, capped, cp))
                longs = [e for e in ext if e[0] >= long_br or e[3]]
                if len(longs) >= 2:
                    stop = f"ambiguous branch ({len(longs)} long branches)"
                    info.append({"level": k, "branches": [(round(float(e[0]), 1), e[2], e[3]) for e in ext]})
                    break
                if len(longs) == 1:
                    comps = [longs[0][4]]
                else:  # only short branches (claws): the farthest one is the tip
                    e = max(ext, key=lambda x: x[0])
                    tip = self.V[e[1]].copy() if e[1] is not None else None
                    stop = "tip (short branches: claws)"
                    cen.append(comps[0]["centroid"] if len(ext) == 1 else np.mean([x[4]["centroid"] for x in ext], 0))
                    rad.append(float(np.mean([x[4]["radius"] for x in ext])))
                    break
            cp = comps[0]
            if cp["huge"] or not cp["closed"]:
                stop = "ring open or huge (merged into the body / mesh boundary)"
                break
            ref = float(np.median(hist[-cfg["jump_hist"] :]))
            if cp["radius"] > cfg["jump_factor"] * ref + jump_add:
                stop = f"ring radius jump ({cp['radius']:.1f} vs {ref:.1f} um: contact / body)"
                break
            if in_trunk is not None and in_trunk(cp["centroid"][None])[0]:
                stop = "entered the trunk core"
                break
            cen.append(cp["centroid"])
            rad.append(cp["radius"])
            hist.append(cp["radius"])
            prev_upper = set(int(v) for v in cp["upper"])
            t_prev = t
        else:
            stop = "max extension"
        C = np.array(cen) if cen else np.zeros((0, 3))
        if len(C) >= 5 and cfg["smooth_levels"] > 0:
            Cs = ndimage.gaussian_filter1d(C, cfg["smooth_levels"], axis=0, mode="nearest")
            Cs[0] = C[0]
            C = Cs
        return {
            "ok": True,
            "centres": C,
            "radii": np.array(rad),
            "tip": tip,
            "stop": stop,
            "levels": len(cen),
            "r_cut": r0,
            "info": info,
        }


def behind(mg, g, src):
    """Unreachable neighbours of the source vertices (the far side of the cut, or the blocked region next to an
    attachment) get g = -1: below every level, so the first rings close cleanly next to the cut instead of running
    along it; every other unreachable vertex stays inf and makes a ring touching it 'open' (a boundary / contact)."""
    g = np.array(g, float, copy=True)
    for v in src:
        for w in mg.neighbours(int(v)):
            if not np.isfinite(g[w]):
                g[w] = -1.0
    return g


# ------------------------------------------------------------------------------------------------ polylines
def arclen(X):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))] if len(X) > 1 else np.zeros(len(X))


def point_at(X, s, a):
    return np.array([np.interp(a, s, X[:, k]) for k in range(3)])


def dist_poly(P, X):
    P, X = np.atleast_2d(np.asarray(P, float)), np.asarray(X, float)
    if len(X) == 1:
        return np.linalg.norm(P - X[0], axis=1)
    a, ab = X[:-1], X[1:] - X[:-1]
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.empty(len(P))
    for i0 in range(0, len(P), 256):
        Q = P[i0 : i0 + 256]
        uu = np.clip(((Q[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0.0, 1.0)
        out[i0 : i0 + 256] = np.linalg.norm(Q[:, None, :] - (a[None] + uu[..., None] * ab[None]), axis=2).min(1)
    return out


def extend_one(mg, X, s, BL, cfg, in_trunk=None, others=()):
    """Try to extend one centreline (tip first). Returns (new X, record)."""
    X = np.asarray(X, float)
    s = arclen(X) if s is None else np.asarray(s, float)
    rec = {"extended": False}
    if len(X) < 3 or s[-1] <= 0:
        rec["stop"] = "polyline too short"
        return X, rec
    tries = []
    for back in cfg["cut_back_pct"]:
        a = back / 100 * BL
        if a >= 0.5 * s[-1]:
            break
        c = point_at(X, s, a)
        b = min(a + cfg["dir_pct"] / 100 * BL, s[-1])
        u = c - point_at(X, s, b)
        if np.linalg.norm(u) <= 0:
            continue
        u = u / np.linalg.norm(u)
        loop = mg.plane_loop(c, u, BL, cfg)
        if loop is None:
            tries.append({"back_pct": back, "loop": None})
            continue
        # contact check: the section behind the cut must not be much narrower than the cut section
        b2 = a + max(cfg["back_check_pct"] / 100 * BL, 3 * loop["radius"])
        r_back = None
        if b2 < 0.6 * s[-1]:
            c2 = point_at(X, s, b2)
            u2 = c2 - point_at(X, s, min(b2 + cfg["dir_pct"] / 100 * BL, s[-1]))
            if np.linalg.norm(u2) > 0:
                lp2 = mg.plane_loop(c2, u2 / np.linalg.norm(u2), BL, cfg)
                r_back = None if lp2 is None else float(lp2["radius"])
        if r_back is not None and loop["radius"] > cfg["cut_wide_factor"] * r_back + cfg["cut_wide_add_pct"] / 100 * BL:
            tries.append(
                {
                    "back_pct": back,
                    "loop_r": round(loop["radius"], 2),
                    "r_back": round(r_back, 2),
                    "rejected": "cut section wider than the limb behind it (contact at the tip)",
                }
            )
            rec.update(
                stop=f"rejected: cut section {loop['radius']:.1f} um wider than the limb behind it "
                f"({r_back:.1f} um): contact at the tip",
                tries=tries,
            )
            return X, rec
        tr = mg.trace(c, u, BL, cfg, in_trunk=in_trunk, loop=loop)
        tries.append(
            {
                "back_pct": back,
                "loop_r": round(loop["radius"], 2),
                "r_back": None if r_back is None else round(r_back, 2),
                "contains": bool(loop["contains"]),
                "stop": tr.get("stop"),
                "levels": tr.get("levels"),
            }
        )
        if not tr["ok"]:
            continue
        E = [c] + list(tr["centres"]) + ([tr["tip"]] if tr["tip"] is not None else [])
        E = np.array(E)
        # drop centres that double back behind the cut plane at the first levels only (noise next to the cut); a limb
        # that folds back further on (a tibia hanging back from a knee) is kept
        m0 = 1 + int(np.ceil(2.0 * max(loop["radius"], mg.med_edge) / (cfg["step_pct"] / 100 * BL)))
        keep = np.ones(len(E), bool)
        keep[1 : 1 + m0] = (E[1 : 1 + m0] - c) @ u > -0.5 * loop["radius"]
        E = E[keep]
        ext = E[::-1]  # tip first, ends at the cut point
        # extension vs other limbs (contacts) and the cut-point overlap rule
        if len(others) and len(ext) > 2:
            se = arclen(ext[::-1])
            P = ext[::-1]
            near = np.zeros(len(P), bool)
            for Y in others:
                near |= dist_poly(P, Y) <= cfg["contact_tol_pct"] / 100 * BL
            run, start, cut_at = 0.0, None, None
            for i in range(1, len(P)):
                if near[i]:
                    start = i if start is None else start
                    run = se[i] - se[start]
                    if run >= cfg["contact_run_pct"] / 100 * BL:
                        cut_at = start
                        break
                else:
                    start, run = None, 0.0
            if cut_at is not None:
                P = P[: max(cut_at, 1)]
                rec["contact_cut"] = True
                ext = P[::-1]
                tr["stop"] = str(tr["stop"]) + "; cut where it runs along another limb"
        if cfg.get("require_tip", True) and (not str(tr["stop"]).startswith("tip") or rec.get("contact_cut")):
            tries[-1].update(rejected="the trace did not end at a free tip")
            rec.update(stop=f"rejected: {tr['stop']}", tries=tries)
            return X, rec
        gain = float(arclen(ext)[-1]) - a
        tip_move = float(np.linalg.norm(ext[0] - X[0]))
        tries[-1].update(gain_pct=round(100 * gain / BL, 2), tip_move_pct=round(100 * tip_move / BL, 2))
        if gain < cfg["min_gain_pct"] / 100 * BL or tip_move < cfg["min_gain_pct"] / 100 * BL:
            rec.update(stop=f"no tube beyond the tip (gain {100 * gain / BL:.2f} % BL)", tries=tries)
            return X, rec
        i0 = int(np.searchsorted(s, a))
        newX = np.vstack([ext[:-1], c[None], X[i0:]]) if i0 < len(X) else np.vstack([ext, X[-1:]])
        # remove duplicate consecutive points
        keep = np.r_[True, np.linalg.norm(np.diff(newX, axis=0), axis=1) > 1e-9]
        newX = newX[keep]
        rec.update(
            extended=True,
            gain_um=round(gain, 2),
            gain_pct=round(100 * gain / BL, 3),
            cut_back_pct=back,
            removed_um=round(a, 2),
            stop=tr["stop"],
            levels=tr["levels"],
            r_cut_um=round(tr["r_cut"], 2),
            r_back_um=None if r_back is None else round(r_back, 2),
            r_ext_median_um=round(float(np.median(tr["radii"])), 2) if len(tr["radii"]) else None,
            tip_reached=str(tr["stop"]).startswith("tip"),
            n_new=int(len(ext) - 1),
            tries=tries,
        )
        return newX, rec
    rec.update(stop="no usable cut" if not tries else "no section loop at any cut", tries=tries)
    return X, rec


# ------------------------------------------------------------------------------------------------ products
def load_trunk(trunk_dir, name):
    tj = json.load(open(os.path.join(trunk_dir, name, "trunk.json")))
    tz = np.load(os.path.join(trunk_dir, name, "trunk.npz"))
    sh = tuple(int(x) for x in tz["shape"])
    occ = np.unpackbits(tz["occ"])[: int(np.prod(sh))].reshape(sh).astype(bool)
    t = {
        "json": tj,
        "BL": float(tj["BL_um"]),
        "R": np.asarray(tj["frame"]["rotation_scan_to_canonical"], float),
        "o": np.asarray(tj["frame"]["origin_scan"], float),
        "lo": np.asarray(tz["lo"], float),
        "pitch": float(tz["pitch"]),
        "tpart": np.asarray(tz["tpart"]),
        "occ": occ,
        "T_X": np.asarray(tz["T_X"], float) if "T_X" in tz.files else None,
    }
    return t


def trunk_test(trunk):
    """in_trunk(P scan) -> bool array: the point's voxel is in TRUNK's opened trunk core (tpart >= 0)."""
    tp, lo, h, R, o = trunk["tpart"], trunk["lo"], trunk["pitch"], trunk["R"], trunk["o"]

    def f(P):
        W = (np.atleast_2d(P) - o) @ R.T
        q = np.round((W - lo) / h).astype(int)
        ok = np.all((q >= 0) & (q < np.array(tp.shape)), axis=1)
        out = np.zeros(len(q), bool)
        out[ok] = tp[tuple(q[ok].T)] >= 0
        return out

    return f


def restatus(meta, arcs, cfg):
    """partial -> path when every declared missing part is the distal part and the new free arc reaches
    partner_ratio x the bilateral partner's free arc (both after extension). Returns the changed limb names."""
    changed = []
    for ch, m in meta.items():
        if m.get("status") != "partial" or not m.get("thin_extension", {}).get("extended"):
            continue
        miss = m.get("missing") or (m.get("assign") or {}).get("missing") or []
        if not miss or not all("distal" in str(x) for x in miss):
            continue
        p = PAIR[ch]
        if ch not in arcs or p not in arcs:
            continue
        if arcs[ch] >= cfg["partner_ratio"] * arcs[p]:
            m["status_before_thin"] = m["status"]
            m["missing_before_thin"] = miss
            m["status"] = "path"
            m["missing"] = []
            changed.append(ch)
    return changed


def free_part(X, core):
    s = arclen(X)
    k = len(X) if core is None else max(2, int(np.searchsorted(s, float(core))) + 1)
    return X[:k]


def gather_free(s6_work, name, approaches):
    """{limb name: [free parts of that name's polylines in every approach product]} (the S6 inputs, unextended)."""
    out = {}
    for a in approaches:
        fj = os.path.join(s6_work, f"paths_{a}", name, "limbs.json")
        fz = os.path.join(s6_work, f"paths_{a}", name, "limbs.npz")
        if not (os.path.isfile(fj) and os.path.isfile(fz)):
            continue
        try:
            meta = json.load(open(fj))
            z = np.load(fz)
        except Exception:
            continue
        for ch, m in (meta.get("limbs") or {}).items():
            if (m or {}).get("status") in ("path", "partial") and ch + "_X" in z.files:
                X = np.asarray(z[ch + "_X"], float)
                if X.ndim == 2 and len(X) >= 2:
                    out.setdefault(ch, []).append(free_part(X, m.get("core_arc_um")))
    return out


def obstacles(ch, own_free, all_free, BL, cfg):
    """Other limbs' free parts (every approach) an extension of `ch` must not run along; a polyline of another name
    that covers most of ch's own free part is the same physical limb named differently by another approach and is not
    an obstacle."""
    tol = cfg["contact_tol_pct"] / 100 * BL
    out = []
    step = 0.5 / 100 * BL
    so = arclen(own_free)
    S = np.stack([np.interp(np.arange(0, so[-1] + step, step), so, own_free[:, k]) for k in range(3)], 1)
    for c2, L in all_free.items():
        if c2 == ch:
            continue
        for Y in L:
            if (dist_poly(S, Y) <= tol).mean() >= 0.5:
                continue
            out.append(Y)
    return out


def extend_product(mg, src_dir, dst_dir, name, BL, cfg, in_trunk=None, cache=None, all_free=None):
    """Extend every limb polyline of one limb product (round-6 limb layout). cache: {(tip, dir) key: result}
    shared across products of the same specimen. all_free: gather_free() of the specimen (contact obstacles from every
    approach; default: this product's other limbs)."""
    meta = json.load(open(os.path.join(src_dir, name, "limbs.json")))
    z = np.load(os.path.join(src_dir, name, "limbs.npz"))
    arrays = {k: z[k] for k in z.files}
    limbs = meta.get("limbs") or {}
    polys = {
        ch: arrays[ch + "_X"]
        for ch in LIMBS
        if (limbs.get(ch) or {}).get("status") in ("path", "partial")
        and ch + "_X" in arrays
        and np.asarray(arrays[ch + "_X"]).ndim == 2
        and len(arrays[ch + "_X"]) >= 2
    }
    free = {}
    for ch, X in polys.items():
        core = limbs[ch].get("core_arc_um")
        s = arclen(X)
        k = len(X) if core is None else max(2, int(np.searchsorted(s, float(core))) + 1)
        free[ch] = X[:k]
    recs, arcs = {}, {}
    for ch, X in polys.items():
        if ch not in cfg["limbs"]:
            continue
        s = arrays.get(ch + "_s")
        s = arclen(X) if s is None or len(s) != len(X) else np.asarray(s, float)
        others = (
            obstacles(ch, free[ch], all_free, BL, cfg)
            if all_free is not None
            else [free[c2] for c2 in polys if c2 != ch]
        )
        key = tuple(np.round(X[0] / (0.2 / 100 * BL)).astype(int)) + tuple(
            np.round(X[min(len(X) - 1, 5)] / (0.2 / 100 * BL)).astype(int)
        )
        if cache is not None and key in cache and cache[key][2] == len(others):
            newX, rec = cache[key][0].copy(), dict(cache[key][1], cached=True)
        else:
            t0 = time.time()
            try:
                newX, rec = extend_one(mg, X, s, BL, cfg, in_trunk=in_trunk, others=others)
            except Exception as e:
                newX, rec = (
                    X,
                    {
                        "extended": False,
                        "stop": f"error {type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-800:],
                    },
                )
            rec["seconds"] = round(time.time() - t0, 2)
            if cache is not None:
                cache[key] = (newX.copy(), dict(rec), len(others))
        m = limbs[ch]
        m["thin_extension"] = {k: v for k, v in rec.items() if k != "trace"}
        if rec.get("extended"):
            gain = float(arclen(newX)[-1] - s[-1])
            r_old = arrays.get(ch + "_r")
            n_new = len(newX) - len(X) + int(np.searchsorted(s, rec["removed_um"]))
            r_new = np.r_[
                np.full(max(n_new, 0), rec.get("r_ext_median_um") or np.nan),
                (
                    r_old[int(np.searchsorted(s, rec["removed_um"])) :]
                    if r_old is not None and len(r_old) == len(X)
                    else np.full(len(newX) - max(n_new, 0), np.nan)
                ),
            ]
            if len(r_new) != len(newX):
                r_new = np.full(len(newX), np.nan)
            arrays[ch + "_X"], arrays[ch + "_s"], arrays[ch + "_r"] = newX, arclen(newX), r_new
            if m.get("core_arc_um") is not None:
                m["core_arc_um_before_thin"] = m["core_arc_um"]
                m["core_arc_um"] = float(m["core_arc_um"]) + gain
            m["arc_um"] = float(arclen(newX)[-1])
            m["tip_scan"] = newX[0].tolist()
            m["n_points"] = int(len(newX))
        recs[ch] = rec
        core = m.get("core_arc_um")
        arcs[ch] = float(core) if core is not None else float(arclen(arrays[ch + "_X"])[-1])
    changed = restatus(limbs, arcs, cfg)
    meta["thin"] = {
        "version": VERSION,
        "cfg": {k: v for k, v in cfg.items()},
        "restatus": changed,
        "n_extended": int(sum(1 for r in recs.values() if r.get("extended"))),
    }
    d = os.path.join(dst_dir, name)
    os.makedirs(d, exist_ok=True)
    json.dump(meta, open(os.path.join(d, "limbs.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(d, "limbs.npz"), **arrays)
    return recs, changed


def run_extend(name, s6_work, mesh_path, work, approaches=("A", "B", "C"), cfg=None):
    cfg = dict(CFG, **(cfg or {}))
    t0 = time.time()
    trunk_src = os.path.join(s6_work, "trunk", name)
    tdst = os.path.join(work, "trunk", name)
    os.makedirs(os.path.dirname(tdst), exist_ok=True)
    if not os.path.lexists(tdst):  # relative, so that the output folder can be moved
        os.symlink(os.path.relpath(os.path.realpath(trunk_src), os.path.realpath(os.path.dirname(tdst))), tdst)
    trunk = load_trunk(os.path.join(s6_work, "trunk"), name)
    BL = trunk["BL"]
    mg = MeshGraph(mesh_path)
    in_trunk = trunk_test(trunk)
    cache = None  # every polyline traced on its own (exact; tracing is ~0.1-1 s per limb)
    all_free = gather_free(s6_work, name, approaches)
    summary = {
        "name": name,
        "BL_um": BL,
        "mesh": {"V": mg.n, "median_edge_um": round(mg.med_edge, 2)},
        "approaches": {},
    }
    for a in approaches:
        src = os.path.join(s6_work, f"paths_{a}")
        dst = os.path.join(work, f"paths_{a}")
        sd = os.path.join(src, name, "stage_done.json")
        rec = {"stage": f"paths_{a}", "name": name, "status": "ok", "error": None}
        try:
            st = json.load(open(sd)) if os.path.isfile(sd) else {"status": "missing"}
            if st.get("status") != "ok":
                # the S6 stage failed: pass its record through unchanged
                os.makedirs(os.path.join(dst, name), exist_ok=True)
                json.dump(st, open(os.path.join(dst, name, "stage_done.json"), "w"), indent=1)
                summary["approaches"][a] = {"s6_status": st.get("status"), "error": st.get("error")}
                continue
            recs, changed = extend_product(
                mg, src, dst, name, BL, cfg, in_trunk=in_trunk, cache=cache, all_free=all_free
            )
            summary["approaches"][a] = {
                "extended": {ch: r.get("gain_pct") for ch, r in recs.items() if r.get("extended")},
                "restatus": changed,
                "stops": {ch: r.get("stop") for ch, r in recs.items()},
            }
        except Exception as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
            summary["approaches"][a] = {"error": rec["error"]}
        rec["seconds"] = round(time.time() - t0, 2)
        os.makedirs(os.path.join(dst, name), exist_ok=True)
        if rec["status"] == "ok":
            json.dump(rec, open(os.path.join(dst, name, "stage_done.json"), "w"), indent=1)
        else:
            json.dump(rec, open(os.path.join(dst, name, "stage_done.json"), "w"), indent=1)
    if cfg.get("discover"):
        try:
            summary["discovery"] = run_discovery(mg, trunk, name, work, BL, cfg, in_trunk)
        except Exception as e:
            summary["discovery"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    summary["seconds"] = round(time.time() - t0, 2)
    os.makedirs(os.path.join(work, "thin"), exist_ok=True)
    json.dump(summary, open(os.path.join(work, "thin", f"{name}.json"), "w"), indent=1, default=float)
    return summary


def run_discovery(mg, trunk, name, work, BL, cfg, in_trunk):
    """discover.py on the extended products in work/paths_X/name (see its docstring)."""
    import discover as DS

    polys, present, roots = [], set(), {}
    for a in cfg["order"]:
        fj = os.path.join(work, f"paths_{a}", name, "limbs.json")
        fz = os.path.join(work, f"paths_{a}", name, "limbs.npz")
        if not (os.path.isfile(fj) and os.path.isfile(fz)):
            continue
        meta = json.load(open(fj))
        z = np.load(fz)
        for ch in LIMBS:
            m = (meta.get("limbs") or {}).get(ch) or {}
            if m.get("status") in ("path", "partial") and ch + "_X" in z.files and np.asarray(z[ch + "_X"]).ndim == 2:
                X = np.asarray(z[ch + "_X"], float)
                polys.append(X)
                present.add(ch)
                if ch in LEGS and ch not in roots:
                    core = m.get("core_arc_um")
                    s = arclen(X)
                    roots[ch] = point_at(X, s, min(float(core), s[-1])) if core is not None else X[-1]
    missing = [ch for ch in LIMBS[:8] if ch not in present]
    rep = {"missing_before": missing}
    if not missing:
        return rep
    cands, cov = DS.discover(mg, trunk, polys, BL, cfg, in_trunk=in_trunk)
    named = DS.name_candidates(cands, trunk, present, roots, BL)
    rep.update(
        coverage=cov,
        candidates=[
            {k: (v if not isinstance(v, np.ndarray) else None) for k, v in c.items() if k not in ("X", "radii")}
            for c in cands
        ],
        named={
            k: {"extent_pct": c["extent_pct"], "attach_part": c["attach_part"], "side_um": c.get("side_um")}
            for k, c in named.items()
        },
    )
    if named:
        rep["inserted_into"] = DS.insert(work, name, cfg["order"], named, BL)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extend"])
    ap.add_argument("name")
    ap.add_argument("--s6-work", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--approaches", default="A,B,C")
    ap.add_argument("--cfg", default=None, help="JSON dict of CFG overrides")
    a = ap.parse_args()
    cfg = json.loads(a.cfg) if a.cfg else None
    s = run_extend(a.name, a.s6_work, a.mesh, a.work, tuple(a.approaches.split(",")), cfg)
    print(
        json.dumps(
            {
                "name": a.name,
                "seconds": s["seconds"],
                "extended": {k: v.get("extended") for k, v in s["approaches"].items()},
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
