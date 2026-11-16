"""Geometry helpers for SKEL-REG.

Reused ideas (re-implemented here):
- exact point-to-triangle distance through pytorch3d
- flood-fill solid voxelisation, distance transform, min-cost medial paths (as in the joint-proposal method)
"""

import numpy as np
import trimesh

_TORCH = None


def ray_intersector(mesh):
    """embree ray caster of a trimesh mesh. The package `embreex` is required: trimesh's own caster needs `rtree`, is
    far slower and does not give the same hits."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError as e:
        raise ImportError(
            "the registration pipeline needs the package 'embreex' for ray casting (pip install embreex)"
        ) from e
    return RayMeshIntersector(mesh, scale_to_box=False)


def torch_dev():
    """torch + device (always the CPU)."""
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = (torch, torch.device("cpu"))  # the pipeline uses these helpers on the CPU only
    return _TORCH


# ---------------------------------------------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------------------------------------------
def load_mesh(path):
    """STL -> merged vertices (float64), faces (int64). No component filtering (the envelope is one surface)."""
    m = trimesh.load(path, process=False, force="mesh")
    m.merge_vertices()
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-30)[:, None], 0.5 * a


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area weighted
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1), 1e-30)[:, None]


def vertex_areas(V, F):
    _, a = face_normals(V, F)
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a / 3.0)
    return out


def winding_number(P, V, F, chunk=None):
    """Generalised winding number of points P w.r.t. the closed mesh (van Oosterom-Strackee solid angles)."""
    torch, dev = torch_dev()
    o = V.mean(0)
    tri = torch.tensor(V[F] - o, dtype=torch.float64, device=dev)  # (F,3,3)
    out = []
    if chunk is None:
        chunk = max(1, int(4e7 // max(len(F), 1)))
    for k in range(0, len(P), chunk):
        p = torch.tensor(P[k : k + chunk] - o, dtype=torch.float64, device=dev)
        a = tri[None, :, 0] - p[:, None]
        b = tri[None, :, 1] - p[:, None]
        c = tri[None, :, 2] - p[:, None]
        la, lb, lc = a.norm(dim=2), b.norm(dim=2), c.norm(dim=2)
        det = (a * torch.cross(b, c, dim=2)).sum(2)
        den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
        w = 2.0 * torch.atan2(det, den)
        out.append((w.sum(1) / (4 * np.pi)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


RAY_DIRS = np.array(
    [[0.36, 0.48, 0.80], [-0.80, 0.36, 0.48], [0.48, -0.80, -0.36], [-0.48, -0.36, 0.80], [0.80, 0.48, -0.36]]
)


def point_to_mesh(P, V, F, chunk=200000, return_face=False):
    """Exact unsigned distance from points to a triangle mesh (pytorch3d point_face_distance)."""
    torch, dev = torch_dev()
    from pytorch3d import _C

    o = V.mean(0)
    tris = torch.tensor((V - o)[F], dtype=torch.float32, device=dev)
    tfi = torch.zeros(1, dtype=torch.int64, device=dev)
    d_all, f_all = [], []
    for k in range(0, len(P), chunk):
        p = torch.tensor(P[k : k + chunk] - o, dtype=torch.float32, device=dev)
        d2, idx = _C.point_face_dist_forward(p, torch.zeros(1, dtype=torch.int64, device=dev), tris, tfi, len(p), 1e-9)
        d_all.append(torch.sqrt(d2.clamp(min=0)).cpu().numpy())
        f_all.append(idx.cpu().numpy())
    d = np.concatenate(d_all).astype(np.float64)
    if return_face:
        return d, np.concatenate(f_all)
    return d


def closest_point_on_triangles(P, A, B, C):
    """Closest points on triangles (A,B,C) to P, all (N,3). Returns points and barycentric coords (Ericson)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = P - B
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = P - C
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    n = len(P)
    bc = np.zeros((n, 3))
    done = np.zeros(n, bool)

    def setv(mask, u, v, w):
        m = mask & ~done
        bc[m, 0], bc[m, 1], bc[m, 2] = u[m] if np.ndim(u) else u, v[m] if np.ndim(v) else v, w[m] if np.ndim(w) else w
        done[m] = True

    one, zero = np.ones(n), np.zeros(n)
    setv((d1 <= 0) & (d2 <= 0), one, zero, zero)
    setv((d3 >= 0) & (d4 <= d3), zero, one, zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_ab = d1 / (d1 - d3)
        setv((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - v_ab, v_ab, zero)
        setv((d6 >= 0) & (d5 <= d6), zero, zero, one)
        w_ac = d2 / (d2 - d6)
        setv((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - w_ac, zero, w_ac)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        setv((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - w_bc, w_bc)
        den = va + vb + vc
        v, w = vb / den, vc / den
        setv(np.ones(n, bool), 1 - v - w, v, w)
    bc = np.nan_to_num(bc, nan=1.0 / 3)
    Q = bc[:, :1] * A + bc[:, 1:2] * B + bc[:, 2:3] * C
    return Q, bc


def closest_on_mesh(P, V, F, chunk=200000):
    """Distance, face index, closest point and barycentric coordinates of P on mesh (V,F)."""
    d, fi = point_to_mesh(P, V, F, chunk=chunk, return_face=True)
    T = V[F[fi]]
    Q, bc = closest_point_on_triangles(P, T[:, 0], T[:, 1], T[:, 2])
    return np.linalg.norm(P - Q, axis=1), fi, Q, bc


# ---------------------------------------------------------------------------------------------------------------
# voxel solid (union semantics: everything not reachable from outside is inside)
# ---------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------------------------------------------


def tangents(X):
    T = np.gradient(X, axis=0)
    return T / np.maximum(np.linalg.norm(T, axis=1), 1e-30)[:, None]


def rmf(X, n0):
    """Rotation-minimising frames along polyline X (double reflection, Wang et al. 2008).
    Returns T (K,3), N (K,3), B (K,3) with N[0] = n0 made perpendicular to T[0]."""
    T = tangents(X)
    N = np.zeros_like(X)
    n = n0 - (n0 @ T[0]) * T[0]
    N[0] = n / np.linalg.norm(n)
    for i in range(len(X) - 1):
        v1 = X[i + 1] - X[i]
        c1 = v1 @ v1
        if c1 < 1e-30:
            N[i + 1] = N[i]
            continue
        rL = N[i] - (2 / c1) * (v1 @ N[i]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        if c2 < 1e-30:
            N[i + 1] = rL
        else:
            N[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2
        N[i + 1] -= (N[i + 1] @ T[i + 1]) * T[i + 1]
        N[i + 1] /= np.linalg.norm(N[i + 1])
    return T, N, np.cross(T, N)
