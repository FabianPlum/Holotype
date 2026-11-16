"""S7 FOLDS-AND-TIPS: shared helpers (loading S6 products, occupancy grid, orthographic renders).

Nothing in this file reads truth. The rendering helpers are not used by the prediction path.
"""

import json
import os

import numpy as np

LEGS = [f"{s}{i}" for s in "LR" for i in (1, 2, 3)]
ANTS, MANS = ["AL", "AR"], ["ML", "MR"]
LIMBS = LEGS + ANTS + MANS
CHAIN_J = {
    **{c: ["pt", "ta", "ti", "fe", "tr", "co"] for c in LEGS},
    **{c: ["an_3", "an_2", "an_1"] for c in ANTS},
    **{c: ["ma"] for c in MANS},
}
PARTNER = {
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


def omni(ch, j):
    if ch in LEGS:
        return f"l_{ch[1]}_{j}_{ch[0].lower()}"
    if ch in ANTS:
        return f"{j}_{ch[1].lower()}"
    return f"ma_{ch[1].lower()}"


def arclen(X):
    X = np.asarray(X, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dist_poly(P, X):
    """Distance of points P (m,3) to polyline X (n,3) and the arc at the nearest point."""
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


def resample(X, step):
    s = arclen(X)
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, X[:, k]) for k in range(3)], 1), t


def load_mesh(name, tmpl):
    import trimesh

    return trimesh.load(tmpl.format(name=name), process=False)


def load_limbs(d, name):
    fj, fz = os.path.join(d, name, "limbs.json"), os.path.join(d, name, "limbs.npz")
    if not (os.path.isfile(fj) and os.path.isfile(fz)):
        return None, None
    return json.load(open(fj)), dict(np.load(fz))


def load_trunk(work, name):
    t = json.load(open(os.path.join(work, "trunk", name, "trunk.json")))
    z = np.load(os.path.join(work, "trunk", name, "trunk.npz"))
    return t, z


def load_prop(d, name):
    f = os.path.join(d, name, f"{name}_joints.json")
    return json.load(open(f)) if os.path.isfile(f) else None


class Grid:
    """The trunk product's voxel occupancy (working frame, axis-aligned): x_frame = R (x_scan - origin)."""

    def __init__(self, trunk_json, trunk_npz):
        z = trunk_npz
        self.shape = tuple(int(v) for v in z["shape"])
        self.occ = np.unpackbits(z["occ"])[: int(np.prod(self.shape))].reshape(self.shape).astype(bool)
        self.lo = np.asarray(z["lo"], float)
        self.pitch = float(z["pitch"])
        fr = trunk_json["frame"]
        self.R = np.asarray(fr["rotation_scan_to_canonical"], float)
        self.o = np.asarray(fr["origin_scan"], float)
        self.tpart = np.asarray(z["tpart"]) if "tpart" in z.files else None

    def to_frame(self, P):
        return (np.atleast_2d(P) - self.o) @ self.R.T

    def to_scan(self, F):
        return np.atleast_2d(F) @ self.R + self.o

    def to_ijk(self, P):
        return np.rint((self.to_frame(P) - self.lo) / self.pitch).astype(int)

    def ijk_to_scan(self, ijk):
        return self.to_scan(self.lo + np.atleast_2d(ijk) * self.pitch)

    def inside(self, ijk):
        ijk = np.atleast_2d(ijk)
        return np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)


# ------------------------------------------------------------------------------------------------ rendering
def plane_axes(P):
    """Best-fit plane of points: (centre, e1 in-plane long axis, e2 in-plane, n normal)."""
    P = np.asarray(P, float)
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0], vt[1], vt[2]


def crop_near(mesh, pts, r):
    """Faces with a vertex within r of any of the points pts (KD-tree)."""
    import trimesh
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(pts, float)).query(mesh.vertices, k=1, distance_upper_bound=r)
    vc = np.isfinite(d)
    fm = vc[mesh.faces].any(1)
    return trimesh.Trimesh(mesh.vertices, mesh.faces[fm], process=False)


def render_ortho(mesh, centre, e1, e2, half_w, res=520, light=None, crop_r=None):
    """Orthographic render looking along -n (n = e1 x e2) at centre, image x = e1, image y = e2. Returns
    (rgb float image HxWx3, proj(P)->(px, py, depth)). Uses embree ray casting (trimesh)."""
    import trimesh

    e1, e2 = unit(np.asarray(e1, float)), unit(np.asarray(e2, float))
    n = unit(np.cross(e1, e2))
    m = mesh
    if crop_r is not None:
        vc = np.linalg.norm(mesh.vertices - centre, axis=1) < crop_r
        fm = vc[mesh.faces].any(1)
        m = trimesh.Trimesh(mesh.vertices, mesh.faces[fm], process=False)
    W = H = res
    xs = (np.arange(W) + 0.5) / W * 2 * half_w - half_w
    ys = half_w - (np.arange(H) + 0.5) / H * 2 * half_w
    gx, gy = np.meshgrid(xs, ys)
    far = 1e5
    org = centre[None] + gx.reshape(-1, 1) * e1 + gy.reshape(-1, 1) * e2 + far * n
    dirs = np.tile(-n, (len(org), 1))
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        rmi = RayMeshIntersector(m)
    except Exception:
        rmi = m.ray
    loc, ir, it = rmi.intersects_location(org, dirs, multiple_hits=False)
    img = np.ones((H * W, 3)) * np.array([1.0, 1.0, 1.0])
    depth = np.full(H * W, np.inf)
    if len(ir):
        nr = m.face_normals[it]
        L = unit(np.array(light if light is not None else [0.3, 0.4, 1.0]))
        Lw = unit(L[0] * e1 + L[1] * e2 + L[2] * n)
        sh = np.abs(nr @ Lw)
        base = np.array([0.80, 0.77, 0.72])
        img[ir] = base[None] * (0.35 + 0.65 * sh[:, None])
        depth[ir] = far - ((loc - org[ir]) * (-n)).sum(1) * -1 + 0  # distance along -n from the origin plane
        depth[ir] = (loc - centre) @ n
    img = img.reshape(H, W, 3)
    depth = depth.reshape(H, W)

    def proj(P):
        P = np.atleast_2d(P) - centre
        x, y, z = P @ e1, P @ e2, P @ n
        return (x + half_w) / (2 * half_w) * W, (half_w - y) / (2 * half_w) * H, z

    return img, proj, depth


def visible(proj, depth, P, tol):
    """Is point P in front of the rendered surface (depth along n larger = nearer the camera)?"""
    px, py, z = proj(P)
    H, W = depth.shape
    out = []
    for x, y, zz in zip(px, py, z):
        i, j = int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))
        d = depth[i, j]
        out.append(not np.isfinite(d) or zz >= d - tol)
    return np.array(out)
