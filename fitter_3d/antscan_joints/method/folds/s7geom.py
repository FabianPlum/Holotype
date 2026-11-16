"""S7 FOLDS-AND-TIPS: local fine occupancy of the envelope mesh and geodesic tools (scan frame, micrometres).

parity_occupancy / _toggle / _prefix_xor are copied unchanged from an earlier high-resolution helper (exact ray-parity
voxelisation of a closed surface on any axis-aligned sub-box; the envelope is one closed outer shell).
"""

import numba
import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------------------------- exact ray-parity voxelisation
@numba.njit(cache=True)
def _toggle(tri, nx, ny, nz, out):
    jx, jy = 1.234567e-6, 2.345678e-6
    for t in range(tri.shape[0]):
        x0, y0, z0 = tri[t, 0, 0], tri[t, 0, 1], tri[t, 0, 2]
        x1, y1, z1 = tri[t, 1, 0], tri[t, 1, 1], tri[t, 1, 2]
        x2, y2, z2 = tri[t, 2, 0], tri[t, 2, 1], tri[t, 2, 2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0:
            continue
        i0 = max(int(np.ceil(min(x0, x1, x2) - jx)), 0)
        i1 = min(int(np.floor(max(x0, x1, x2) - jx)), nx - 1)
        j0 = max(int(np.ceil(min(y0, y1, y2) - jy)), 0)
        j1 = min(int(np.floor(max(y0, y1, y2) - jy)), ny - 1)
        for i in range(i0, i1 + 1):
            px = i + jx
            for j in range(j0, j1 + 1):
                py = j + jy
                w0 = (x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)
                w1 = (x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)
                w2 = (x0 - px) * (y1 - py) - (x1 - px) * (y0 - py)
                if area > 0:
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue
                else:
                    if w0 > 0 or w1 > 0 or w2 > 0:
                        continue
                z = (w0 * z0 + w1 * z1 + w2 * z2) / area
                k = int(np.ceil(z))
                if k < 0:
                    k = 0
                if k > nz:
                    k = nz
                out[i, j, k] ^= 1
    return out


@numba.njit(cache=True)
def _prefix_xor(a, nz):
    nx, ny = a.shape[0], a.shape[1]
    occ = np.zeros((nx, ny, nz), np.bool_)
    for i in range(nx):
        for j in range(ny):
            s = 0
            for k in range(nz):
                s ^= a[i, j, k]
                occ[i, j, k] = s == 1
    return occ


def parity_occupancy(tri, lo, shape, pitch):
    """Solid occupancy of the closed triangle soup tri (m, 3, 3) (scan frame) on the axis-aligned grid lo + ijk*pitch."""
    shape = tuple(int(s) for s in shape)
    g = (tri - np.asarray(lo, float)) / pitch
    mn, mx = g[:, :, :2].min(1), g[:, :, :2].max(1)
    sel = (mx[:, 0] >= -1) & (mn[:, 0] <= shape[0]) & (mx[:, 1] >= -1) & (mn[:, 1] <= shape[1])
    g = np.ascontiguousarray(g[sel])
    buf = np.zeros((shape[0], shape[1], shape[2] + 1), np.uint8)
    _toggle(g, shape[0], shape[1], shape[2], buf)
    return _prefix_xor(buf, shape[2])


# ------------------------------------------------------------------------------------- local grids
class Box:
    """Axis-aligned (scan frame) local grid around a set of points with its solid occupancy and distance transform."""

    def __init__(self, tri, pts, margin, pitch, max_vox=60e6):
        pts = np.atleast_2d(pts)
        lo = pts.min(0) - margin
        hi = pts.max(0) + margin
        shape = np.ceil((hi - lo) / pitch).astype(int) + 1
        while np.prod(shape) > max_vox:
            pitch *= 1.15
            shape = np.ceil((hi - lo) / pitch).astype(int) + 1
        self.lo, self.pitch, self.shape = lo, float(pitch), tuple(int(v) for v in shape)
        self.occ = parity_occupancy(tri, lo, self.shape, self.pitch)
        self._dt = None

    @property
    def dt(self):
        if self._dt is None:
            self._dt = ndimage.distance_transform_edt(self.occ) * self.pitch
        return self._dt

    def ijk(self, P):
        return np.rint((np.atleast_2d(P) - self.lo) / self.pitch).astype(int)

    def xyz(self, ijk):
        return self.lo + np.atleast_2d(ijk) * self.pitch

    def inside(self, ijk):
        ijk = np.atleast_2d(ijk)
        return np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)

    def solid_at(self, P):
        q = self.ijk(P)
        ok = self.inside(q)
        out = np.zeros(len(q), bool)
        out[ok] = self.occ[tuple(q[ok].T)]
        return out

    def snap(self, P, mask=None, r_vox=4):
        """Nearest voxel of mask (default solid) to each point within r_vox; None if there is none."""
        m = self.occ if mask is None else mask
        out = []
        for q in self.ijk(P):
            lo_ = np.maximum(q - r_vox, 0)
            hi_ = np.minimum(q + r_vox + 1, np.array(self.shape))
            sub = m[lo_[0] : hi_[0], lo_[1] : hi_[1], lo_[2] : hi_[2]]
            idx = np.argwhere(sub)
            if len(idx) == 0:
                out.append(None)
                continue
            idx = idx + lo_
            out.append(idx[np.argmin(((idx - q) ** 2).sum(1))])
        return out


def tube_mask(box, X, radius):
    """Voxels within radius of the polyline X (dense samples)."""
    from scipy.spatial import cKDTree
    from s7lib import resample

    Xd = resample(np.asarray(X, float), 0.5 * box.pitch)[0]
    # candidate voxels: bounding region of the dilated polyline
    idx = np.argwhere(np.ones(box.shape, bool)) if np.prod(box.shape) < 2e6 else None
    if idx is None:
        m = np.zeros(box.shape, bool)
        q = box.ijk(Xd)
        q = q[box.inside(q)]
        m[tuple(q.T)] = True
        r_vox = radius / box.pitch
        return ndimage.distance_transform_edt(~m) <= r_vox
    d, _ = cKDTree(Xd).query(box.xyz(idx), k=1, distance_upper_bound=radius)
    m = np.zeros(box.shape, bool)
    m[tuple(idx[np.isfinite(d)].T)] = True
    return m


def geodesic(passable, seeds, weight=None):
    """Geodesic distance (voxel units, 26-connected Euclidean steps) from the seed voxels within passable.
    weight: optional per-voxel cost multiplier (e.g. 1/DT^2 for centred routes). Returns (cum, mcp)."""
    from skimage.graph import MCP_Geometric

    cost = np.where(passable, 1.0 if weight is None else weight, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    cum, _ = mcp.find_costs([tuple(int(v) for v in s) for s in seeds])
    return cum, mcp


def route(passable, start, end, dt=None, eps=0.5):
    """Centred min-cost route (cost 1/DT^2) from start to end voxel; (n,3) ijk or None."""
    from skimage.graph import MCP_Geometric

    if dt is None:
        dt = ndimage.distance_transform_edt(passable)
    w = 1.0 / np.maximum(dt, eps) ** 2
    cost = np.where(passable, w, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    cum, _ = mcp.find_costs([tuple(int(v) for v in start)], ends=[tuple(int(v) for v in end)])
    if not np.isfinite(cum[tuple(int(v) for v in end)]):
        return None
    return np.asarray(mcp.traceback(tuple(int(v) for v in end)))


def local_maxima(G, valid, nms_vox, gmin):
    """Extremity tips: local maxima of G over valid voxels (copied logic of the annotation tool's tips_from)."""
    Gm = np.where(valid & np.isfinite(G), G, -1.0)
    mx = ndimage.maximum_filter(Gm, size=2 * nms_vox + 1)
    cand = np.argwhere((Gm >= mx) & (Gm > gmin))
    if len(cand) == 0:
        return np.zeros((0, 3), int)
    cand = cand[np.argsort(-Gm[tuple(cand.T)])]
    keep = []
    for c in cand:
        if all(np.linalg.norm(c - k) > nms_vox for k in keep):
            keep.append(c)
    return np.array(keep)


def smooth_poly(X, sigma_pts):
    if len(X) < 5 or sigma_pts <= 0:
        return np.asarray(X, float).copy()
    return ndimage.gaussian_filter1d(np.asarray(X, float), sigma_pts, axis=0, mode="nearest")


def bend_profile(X, s, w_um):
    """Bend angle (deg) at every sample between the chords to the points w_um before and after (by arc)."""
    n = len(X)
    out = np.zeros(n)
    i0 = np.searchsorted(s, s - w_um)
    i1 = np.minimum(np.searchsorted(s, s + w_um), n - 1)
    for i in range(n):
        a, b = X[i] - X[i0[i]], X[i1[i]] - X[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        out[i] = np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))
    return out
