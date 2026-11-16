"""Small geometry helpers (numpy). Mesh loading follows skelreg.geom.load_mesh (trimesh, vertices merged) so that the
vertex order of an envelope equals the order of the baseline inputs and of SKEL-REG's scan labels."""

import numpy as np


def load_mesh(path):
    import trimesh

    m = trimesh.load(path, process=True, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-30)[:, None], 0.5 * a


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


def vertex_areas(V, F):
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a / 3)
    return out


def sample_surface(V, F, n, rng):
    """area-weighted surface samples: points, face index"""
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    fi = rng.choice(len(F), n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    P = (w[:, :, None] * V[F[fi]]).sum(1)
    return P, fi


def rot_between(u, v):
    """smallest rotation matrix taking unit vector u to unit vector v"""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(u @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        a = np.eye(3)[np.argmin(np.abs(u))]
        k = np.cross(u, a)
        k /= np.linalg.norm(k)
        return 2 * np.outer(k, k) - np.eye(3)
    k = np.cross(u, v)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def rotvec(R):
    """rotation matrix -> axis-angle vector"""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-9:  # angle ~ pi
        w, v = np.linalg.eigh((R + np.eye(3)) / 2)
        return v[:, -1] * ang
    return ax / n * ang


def rodrigues(rv):
    rv = np.asarray(rv, float)
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.eye(3)
    k = rv / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def polyline_arclen(X):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])


def polyline_at(X, s, q):
    """points of polyline X (arc s, increasing) at arcs q"""
    q = np.clip(q, s[0], s[-1])
    return np.stack([np.interp(q, s, X[:, k]) for k in range(3)], 1)
