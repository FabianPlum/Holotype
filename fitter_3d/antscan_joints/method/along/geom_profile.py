"""Centreline preparation and cross-section profiles on the envelope mesh (round 6, ALONG).

Everything here works in the scan frame (micrometres) and uses only the envelope mesh and the given centreline; no
truth, no model, no frame. Lengths are normalised by the body length BL that the caller passes.

prepare(X, s, tm, BL) -> dict with the cleaned centreline (resampled every STEP_PCT % BL, tip first):
  1. resample + light smoothing of the input polyline
  2. distal points outside the mesh are dropped (a tip extended beyond the surface)
  3. re-centring: each inside point moves towards the centroid of its ray-cast cross-section (2 passes, shift
     clipped to 0.6 x the section radius and smoothed along the arc; not where the section is open), so a centreline
     offset sideways inside the tube returns to the tube's middle
  4. tip: the distal end is marched along its tangent to the mesh surface (ray cast), so arc 0 = the limb's tip
  5. cross-section profile on the final centreline: K rays in the plane normal to the tangent, first hit each;
     r_eq = sqrt(area / pi) of the star polygon, r_min / r_max, open fraction (rays without a hit within rcap),
     inside flag, bend angle at several scales.
"""

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

STEP_PCT = 0.25  # centreline sample spacing, % BL
K_RAYS = 32
SMOOTH_PCT = 0.35  # Gaussian sigma of the centreline smoothing, % BL
RCAP_PCT = 6.0  # ray length cap, % BL
TRIM_PCT = 0.5  # leading tip points in air further than this from the surface are dropped, % BL
SNAP_PCT = 1.0
TIP_MAX_PCT = 3.0  # longest tip march along the end tangent, % BL
TIP_TURN_DEG = 40.0  # the tip march stops when its direction has turned this far  # points outside but within this of the surface are moved just inside, % BL
_MESH_CACHE = {}


def as_trimesh(mesh):
    """Accept a trimesh.Trimesh, a (V, F) tuple or a file path; cached by identity / path."""
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    key = mesh if isinstance(mesh, str) else id(mesh)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    if isinstance(mesh, str):
        tm = trimesh.load(mesh, process=True, force="mesh")
    else:
        V, F = mesh
        tm = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, int), process=True)
    if len(_MESH_CACHE) > 8:
        _MESH_CACHE.clear()
    _MESH_CACHE[key] = tm
    return tm


def arclen(X):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]


def resample(X, h):
    s = arclen(X)
    if s[-1] < h or len(X) < 2:
        return X.copy()
    q = np.linspace(0, s[-1], int(np.ceil(s[-1] / h)) + 1)
    return np.stack([np.interp(q, s, X[:, k]) for k in range(3)], 1)


def smooth(X, sigma_pts):
    if len(X) < 5 or sigma_pts <= 0:
        return X.copy()
    return ndimage.gaussian_filter1d(X, sigma_pts, axis=0, mode="nearest")


def tangents(X, w=2):
    n = len(X)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    T = X[i1] - X[i0]
    nt = np.linalg.norm(T, axis=1, keepdims=True)
    nt[nt == 0] = 1.0
    return T / nt


def normal_basis(T, ref=None):
    """Two unit vectors orthogonal to each tangent (parallel-transported, so the basis does not spin). The start
    vector comes from `ref` (a direction fixed to the specimen geometry, e.g. the centreline's chord) so that the ray
    directions do not depend on the coordinate axes (rotation test); axis fallback only when ref is missing or parallel."""
    n = len(T)
    A = np.zeros((n, 3))
    B = np.zeros((n, 3))
    a = None
    if ref is not None:
        a = np.asarray(ref, float) - (np.asarray(ref, float) @ T[0]) * T[0]
        if np.linalg.norm(a) < 1e-6 * max(np.linalg.norm(ref), 1e-12):
            a = None
    if a is None:
        ax = np.array([1.0, 0.0, 0.0]) if abs(T[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        a = np.cross(T[0], ax)
    a /= np.linalg.norm(a)
    for i in range(n):
        a = a - (a @ T[i]) * T[i]
        na = np.linalg.norm(a)
        if na < 1e-9:
            ref = np.array([1.0, 0.0, 0.0]) if abs(T[i, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            a = np.cross(T[i], ref)
            na = np.linalg.norm(a)
        a /= na
        A[i] = a
        B[i] = np.cross(T[i], a)
    return A, B


def first_hits(tm, O, D, rcap):
    """Distance along unit rays D from origins O to the first mesh hit (rcap where none within rcap)."""
    dist = np.full(len(O), np.inf)
    if len(O) == 0:
        return dist
    loc, iray, _ = tm.ray.intersects_location(O, D, multiple_hits=False)
    if len(iray):
        d = np.linalg.norm(loc - O[iray], axis=1)
        np.minimum.at(dist, iray, d)
    return np.minimum(dist, rcap)


def sections(tm, X, T, rcap, k=K_RAYS, ref=None):
    """Ray-cast cross-sections. Returns dict: R (n, k) hit distances (capped), open (n, k) bool, r_eq, r_min,
    r_max, centroid offset C (n, 3)."""
    n = len(X)
    A, B = normal_basis(T, ref)
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    D = A[:, None, :] * ct[None, :, None] + B[:, None, :] * st[None, :, None]  # (n, k, 3)
    O = np.repeat(X, k, axis=0)
    rc = np.broadcast_to(np.asarray(rcap, float).reshape(-1, 1), (n, k)).reshape(-1)
    d = first_hits(tm, O, D.reshape(-1, 3), np.inf)
    R = d.reshape(n, k)
    rcm = rc.reshape(n, k)
    opn = R >= rcm
    R = np.minimum(R, rcm)
    dth = 2 * np.pi / k
    R2 = np.roll(R, -1, axis=1)
    area = 0.5 * np.sin(dth) * (R * R2).sum(1)
    r_eq = np.sqrt(area / np.pi)
    # polygon centroid (in the plane), triangle fan from the origin
    P = R[:, :, None] * D  # (n, k, 3) hit vectors
    P2 = np.roll(P, -1, axis=1)
    tri_a = 0.5 * np.sin(dth) * R * R2  # (n, k)
    tri_c = (P + P2) / 3.0
    C = (tri_a[:, :, None] * tri_c).sum(1) / np.maximum(tri_a.sum(1), 1e-12)[:, None]
    return {"R": R, "open": opn, "r_eq": r_eq, "r_min": R.min(1), "r_max": R.max(1), "C": C}


def bend_angle(X, w):
    """Angle (deg) between the chords X[i-w]->X[i] and X[i]->X[i+w] (w samples), 0 at the ends."""
    n = len(X)
    out = np.zeros(n)
    if n < 2 * w + 1 or w < 1:
        return out
    a = X[w : n - w] - X[: n - 2 * w]
    b = X[2 * w :] - X[w : n - w]
    c = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    out[w : n - w] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


class Surface:
    """Area-weighted surface samples with face normals; signed distance by the vote of the 8 nearest samples
    (trimesh's ray-parity `contains` is unreliable on these meshes: ~25 % True for points well inside)."""

    def __init__(self, tm, spacing):
        n = int(np.clip(tm.area / spacing**2, 300000, 3000000))
        self.p, fi = trimesh.sample.sample_surface(tm, n, seed=1)
        self.nrm = tm.face_normals[fi]
        self.tree = cKDTree(self.p)

    def query(self, X):
        """(closest sample, distance, outward normal of the closest sample, inside flag)."""
        X = np.atleast_2d(X)
        d, ii = self.tree.query(X, k=8)
        v = np.einsum("nkc,nkc->nk", X[:, None, :] - self.p[ii], self.nrm[ii])
        inside = np.sign(v).sum(1) < 0
        return self.p[ii[:, 0]], d[:, 0], self.nrm[ii[:, 0]], inside


_SURF_CACHE = {}


def get_surface(tm, BL):
    key = id(tm)
    if key not in _SURF_CACHE:
        if len(_SURF_CACHE) > 4:
            _SURF_CACHE.clear()
        _SURF_CACHE[key] = Surface(tm, BL / 1200.0)
    return _SURF_CACHE[key]


def prepare(X, s, tm, BL, recentre=True, tip_march=True, step_pct=STEP_PCT):
    """Clean centreline + cross-section profile (see the module docstring). X tip first, scan um."""
    tm = as_trimesh(tm)
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), axis=1)]
    h = step_pct / 100.0 * BL
    info = {}
    X = smooth(resample(X, h), SMOOTH_PCT / step_pct)
    ref = X[-1] - X[0]  # geometry-fixed start vector for the ray bases (rotation invariance)
    if np.linalg.norm(ref) < 1e-9:
        ref = None
    rcap = RCAP_PCT / 100.0 * BL
    surf = get_surface(tm, BL)
    q, dsurf, nrm, inside = surf.query(X)
    # 2. drop the leading run of points that lie in air (outside and > TRIM_PCT % BL from the surface): a tip extended
    #    beyond the limb; thin tarsi whose centreline grazes the surface stay
    far = (~inside) & (dsurf > TRIM_PCT / 100.0 * BL)
    n_drop = 0
    while n_drop < len(X) - 3 and far[n_drop]:
        n_drop += 1
    if n_drop:
        X, inside, q, dsurf, nrm = X[n_drop:], inside[n_drop:], q[n_drop:], dsurf[n_drop:], nrm[n_drop:]
    info["tip_arc_dropped_pct"] = round(100.0 * n_drop * h / BL, 3)
    # 3. points just outside (thin parts, grazing paths): put them just inside the nearest surface, the re-centring
    #    then finds the tube's middle; points further out (gaps at contacts) stay
    near = (~inside) & (dsurf <= SNAP_PCT / 100.0 * BL)
    X = X.copy()
    X[near] = q[near] - nrm[near] * (0.05 / 100.0 * BL)
    info["snapped_inside_pct"] = round(100.0 * near.mean(), 1)
    if recentre:
        inside = surf.query(X)[3]
        for it in range(3):
            T = tangents(X, max(1, int(round(0.5 / step_pct))))
            sec = sections(tm, X, T, rcap, ref=ref)
            ok = inside & (~sec["open"].any(1))
            C = sec["C"].copy()
            nc = np.linalg.norm(C, axis=1)
            lim = (0.6 if it else 1.0) * sec["r_eq"]
            f = np.where(nc > lim, lim / np.maximum(nc, 1e-12), 1.0)
            C = C * f[:, None]
            C[~ok] = 0.0
            C = ndimage.gaussian_filter1d(C, 0.25 / step_pct, axis=0, mode="nearest")
            X = X + C
            inside = surf.query(X)[3]
        X = smooth(resample(X, h), 0.5 * SMOOTH_PCT / step_pct)
        inside = surf.query(X)[3]
    # 4. tip to the surface along the end tangent
    n_ext = 0
    if tip_march and len(X) > 3:
        k = min(len(X) - 1, max(2, int(round(1.0 / step_pct))))
        u = X[0] - X[k]
        nu = np.linalg.norm(u)
        if nu > 0 and inside[0]:
            u = u / nu
            # follow the tube beyond the given tip: step along the current direction, re-centre in the cross-section,
            # stop when the point leaves the mesh (then add the surface point) or the section grows beyond the tip
            # region's radius (a part folded against this one, e.g. the tibia at a knee tip); <= TIP_MAX_PCT % BL
            kk = min(len(X), max(2, int(round(1.0 / step_pct))))
            r0 = float(np.median(sections(tm, X[:kk], tangents(X, 2)[:kk], rcap, ref=ref)["r_eq"]))
            p = X[0].copy()
            u0 = u.copy()
            ext = []
            for _ in range(int(np.ceil(TIP_MAX_PCT / step_pct))):
                d = first_hits(tm, p[None], u[None], np.inf)[0]
                if not np.isfinite(d) or d <= h:
                    if np.isfinite(d) and d > 1e-9:
                        ext.append(p + d * u)
                    break
                q = p + h * u
                sq = sections(tm, q[None], u[None], rcap, ref=ref)
                if sq["r_eq"][0] > 1.5 * r0 + 0.001 * BL or sq["open"].any():
                    ext.append(p + min(d, max(r0, h)) * u)
                    break
                C = sq["C"][0]
                nc = np.linalg.norm(C)
                if nc > 0.6 * sq["r_eq"][0]:
                    C = C * (0.6 * sq["r_eq"][0] / nc)
                q = q + C
                un = q - p
                un /= max(np.linalg.norm(un), 1e-12)
                u = 0.5 * u + 0.5 * un
                u /= np.linalg.norm(u)
                if u @ u0 < np.cos(np.radians(TIP_TURN_DEG)):
                    break  # the tube turns back (a part folded against this one): the tip is here
                ext.append(q)
                p = q
            if ext:
                E = np.array(ext)[::-1]
                X = np.vstack([E, X])
                n_ext = len(E)
    info["tip_extension_pct"] = round(100.0 * n_ext * h / BL, 3) if n_ext else 0.0
    X = resample(X, h)
    s2 = arclen(X)
    T = tangents(X, max(1, int(round(0.5 / step_pct))))
    sec = sections(tm, X, T, rcap, ref=ref)
    _, dsurf, _, inside = surf.query(X)
    gap = (~inside) & (dsurf > 0.05 / 100.0 * BL)
    gap[0] = False  # the tip point sits on the surface
    r_eq = np.where(gap, 0.0, sec["r_eq"])
    prof = {
        "X": X,
        "s": s2,
        "T": T,
        "h": h,
        "BL": BL,
        "inside": inside,
        "gap": gap,
        "rcap": rcap,
        "r_eq": r_eq,
        "r_min": np.where(gap, 0.0, sec["r_min"]),
        "r_max": sec["r_max"],
        "open_frac": sec["open"].mean(1),
        "info": info,
    }
    for w_pct in (1.0, 2.0, 4.0):
        prof[f"bend_{w_pct:g}"] = bend_angle(X, max(1, int(round(w_pct / step_pct))))
    return prof


def project_arc(P, X, s):
    """Arc (and distance) of the point of polyline X nearest to P."""
    P = np.asarray(P, float)
    best = (None, np.inf)
    A = X[:-1]
    B = X[1:]
    AB = B - A
    L2 = (AB * AB).sum(1)
    u = np.clip(((P - A) * AB).sum(1) / np.maximum(L2, 1e-12), 0, 1)
    Q = A + u[:, None] * AB
    d = np.linalg.norm(Q - P, axis=1)
    i = int(np.argmin(d))
    best = (float(s[i] + u[i] * (s[i + 1] - s[i])), float(d[i]))
    return best


def point_at(X, s, a):
    return np.array([np.interp(a, s, X[:, k]) for k in range(3)])


def kdtree(X):
    return cKDTree(X)
