"""PATHS-A geometry helpers: trunk-product loading, mesh in the working frame, surface distances, geodesics.

Several helpers are copied from the annotation tool's geometry helpers (Surface, arclen, resample, smooth, march_out, end_dir,
geodesic, tips_from, mcp_paths); the copies are marked. Coordinates: X_scan = X_working @ R + origin, with
R = trunk.json frame.rotation_scan_to_canonical (the working frame of the trunk product; its axes are NOT used as
anatomy anywhere in PATHS-A).
"""

import json
import os

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree


# ------------------------------------------------------------------ trunk product + mesh
def load_trunk(trunk_dir, name):
    d = os.path.join(trunk_dir, name)
    z = np.load(os.path.join(d, "trunk.npz"))
    meta = json.load(open(os.path.join(d, "trunk.json")))
    shape = tuple(int(x) for x in np.asarray(z["shape"]).ravel())
    occ = np.unpackbits(np.asarray(z["occ"]))[: int(np.prod(shape))].reshape(shape).astype(bool)
    fr = meta["frame"]
    R = np.asarray(fr["rotation_scan_to_canonical"], float)
    o = np.asarray(fr["origin_scan"], float)
    lo = np.asarray(z["lo"], float).ravel()
    pitch = float(np.asarray(z["pitch"]).ravel()[0])
    tpart = np.asarray(z["tpart"]).reshape(shape).astype(np.int8)
    T = {}
    if "T_X" in z.files:
        TX = np.asarray(z["T_X"], float)
        # The trunk path may be stored in the working frame (as geom.npz) or in the scan frame: take the frame in
        # which it lies inside the occupancy (robust to either convention).
        cand = {"working": TX, "scan": (TX - o) @ R.T}
        best, best_in = None, -1.0
        for k, X in cand.items():
            q = np.round((X - lo) / pitch).astype(int)
            ok = np.all((q >= 0) & (q < np.array(shape)), axis=1)
            frac = float(occ[tuple(q[ok].T)].sum()) / max(1, len(X))
            if frac > best_in:
                best, best_in = k, frac
        T = {
            "X": cand[best],
            "s": np.asarray(z["T_s"], float) if "T_s" in z.files else None,
            "part": np.asarray(z["T_part"]) if "T_part" in z.files else None,
            "frame_detected": best,
        }
    joints = {}
    src = meta.get("trunk_joints_scan") or meta.get("joints_scan") or {}
    for k, v in src.items():  # 3-vectors (or {"xyz": ...} records) in scan um; other entries are metadata
        if isinstance(v, dict):
            v = v.get("xyz")
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            joints[k] = (np.asarray(v, float) - o) @ R.T
    BL = float(meta.get("BL_um"))
    return {
        "name": name,
        "occ": occ,
        "shape": shape,
        "lo": lo,
        "pitch": pitch,
        "tpart": tpart,
        "R": R,
        "o": o,
        "T": T,
        "joints_w": joints,
        "BL": BL,
        "meta": meta,
    }


def load_mesh_working(mesh_path, R, o):
    if mesh_path.endswith(".npz"):
        z = np.load(mesh_path)
        V = np.asarray(z["v"], float)
        F = np.asarray(z["f"], np.int64)
        if "R" in z.files:  # scan_canonical.npz: v already canonical = the development set's working frame
            Rz, oz = np.asarray(z["R"], float), np.asarray(z["origin"], float)
            Vs = V @ Rz + oz
            return (Vs - o) @ R.T, F
        return (V - o) @ R.T, F
    m = trimesh.load(mesh_path, process=True, force="mesh")
    V = np.asarray(m.vertices, float)
    return (V - o) @ R.T, np.asarray(m.faces, np.int64)


def to_scan(X, R, o):
    return np.asarray(X, float) @ R + o


# ------------------------------------------------------------------ copied from the annotation tool's helpers (unchanged logic)
class Surface:
    """Area-weighted surface samples of the envelope (~4 per pitch along a line) for signed distances."""

    def __init__(self, V, F, pitch, nmin=400000, nmax=4000000):
        mesh = trimesh.Trimesh(V, F, process=False)
        n = int(np.clip(mesh.area / (pitch / 4.0) ** 2, nmin, nmax))
        self.p, fi = trimesh.sample.sample_surface(mesh, n, seed=1)
        self.nrm = mesh.face_normals[fi]
        self.tree = cKDTree(self.p)
        self.n = n

    def signed(self, P):
        """Signed distance (um) to the scan surface, negative inside (sign = vote of the 8 nearest samples)."""
        P = np.atleast_2d(P)
        d, ii = self.tree.query(P, k=8, workers=4)
        s = np.einsum("nkc,nkc->nk", P[:, None, :] - self.p[ii], self.nrm[ii])
        inside = np.sign(s).sum(1) < 0
        return np.where(inside, -d[:, 0], d[:, 0])


def arclen(X):
    X = np.asarray(X, float)
    if len(X) < 2:
        return np.zeros(len(X))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]


def resample(X, h):
    s = arclen(X)
    if len(X) < 2 or s[-1] < h:
        return np.asarray(X, float).copy()
    q = np.linspace(0, s[-1], int(np.ceil(s[-1] / h)) + 1)
    return np.stack([np.interp(q, s, X[:, k]) for k in range(3)], 1)


def smooth(X, sigma):
    if len(X) < 5 or sigma <= 0:
        return np.asarray(X, float).copy()
    return ndimage.gaussian_filter1d(np.asarray(X, float), sigma, axis=0, mode="nearest")


def march_out(p, u, surf, step, max_len):
    """Points from p along unit u (excluding p) until the first point outside the scan (included), at most max_len."""
    out = []
    t = step
    while t <= max_len:
        q = p + t * u
        out.append(q)
        if surf.signed(q)[0] >= 0:
            break
        t += step
    return np.array(out).reshape(-1, 3)


def end_dir(X, arc, at_start=True):
    s = arclen(X)
    if at_start:
        k = int(np.searchsorted(s, arc))
        k = min(max(k, 1), len(X) - 1)
        u = X[0] - X[k]
    else:
        k = int(np.searchsorted(s, s[-1] - arc))
        k = min(max(k, 0), len(X) - 2)
        u = X[-1] - X[k]
    nu = np.linalg.norm(u)
    return u / nu if nu > 0 else np.array([1.0, 0.0, 0.0])


def geodesic(passable, starts):
    from skimage.graph import MCP_Geometric

    cost = np.where(passable, 1.0, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    cum, _ = mcp.find_costs([tuple(s) for s in starts])
    return cum


def tips_from(G, valid, nms_vox, gmin):
    Gm = np.where(valid & np.isfinite(G), G, -1.0)
    mx = ndimage.maximum_filter(Gm, size=2 * nms_vox + 1)
    cand = np.argwhere((Gm >= mx) & (Gm > gmin))
    if len(cand) == 0:
        return np.zeros((0, 3), int)
    order = np.argsort(-Gm[tuple(cand.T)])
    cand = cand[order]
    keep = []
    for c in cand:
        if all(np.linalg.norm(c - k) > nms_vox for k in keep):
            keep.append(c)
    return np.array(keep)


def mcp_paths(cost, starts, ends):
    from skimage.graph import MCP_Geometric

    mcp = MCP_Geometric(cost, fully_connected=True)
    cum, _ = mcp.find_costs([tuple(s) for s in starts], ends=[tuple(e) for e in ends], find_all_ends=True)
    paths = []
    for e in ends:
        if not np.isfinite(cum[tuple(e)]):
            paths.append(None)
            continue
        paths.append(np.asarray(mcp.traceback(tuple(e))))
    return paths, cum


# ------------------------------------------------------------------ small helpers
def to_um(ijk, lo, pitch):
    return lo + np.asarray(ijk, dtype=np.float64) * pitch


def to_ijk(X, lo, pitch, shape):
    q = np.round((np.asarray(X) - lo) / pitch).astype(int)
    return np.clip(q, 0, np.array(shape) - 1)


def dist_poly(P, X):
    """Distance of points P (m,3) to the polyline X (n,3) and the arc of the nearest point."""
    P, X = np.atleast_2d(np.asarray(P, float)), np.asarray(X, float)
    if len(X) == 1:
        return np.linalg.norm(P - X[0], axis=1), np.zeros(len(P))
    s = arclen(X)
    a, ab = X[:-1], X[1:] - X[:-1]
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    u = np.clip(((P[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0.0, 1.0)
    d = np.linalg.norm(P[:, None, :] - (a[None] + u[..., None] * ab[None]), axis=2)
    k = d.argmin(1)
    i = np.arange(len(P))
    return d[i, k], s[k] + u[i, k] * (s[k + 1] - s[k])


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ]
    )
