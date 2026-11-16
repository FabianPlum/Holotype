"""Geometry helpers copied (unchanged logic) from round 3's geometry helpers (load_mesh, voxelize) and from
the annotation tool's geometry helpers (Surface, dense paths, tips, min-cost paths, duplicate-route removal)."""

import os

import numba
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree


# ------------------------------------------------------------------------------------------- round-3 helpers
class MeshUnreadable(RuntimeError):
    pass


def load_mesh(path):
    """Largest connected component of the STL, float64 vertices, int64 faces (round 3's load_mesh)."""
    if not os.path.isfile(path):
        raise MeshUnreadable(f"mesh file not found: {path}")
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda q: len(q.faces))
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


@numba.njit(cache=True)
def _parity_counts(tri, nx, ny, nz):
    cnt = np.zeros((nx, ny, nz + 1), np.int16)
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
                cnt[i, j, k] += 1
    return cnt


def voxelize(v, f, h, pad=3):
    """Exact parity solid voxelisation of a closed mesh. Voxel (i,j,k) centre = lo + (ijk + 0.5) h (round 3)."""
    lo = v.min(0) - pad * h
    hi = v.max(0) + pad * h
    n = np.ceil((hi - lo) / h).astype(int)
    g = (v - lo) / h - 0.5
    cnt = _parity_counts(g[f].astype(np.float64), int(n[0]), int(n[1]), int(n[2]))
    occ = (np.cumsum(cnt, axis=2)[:, :, :-1] % 2) == 1
    return occ, lo


# ------------------------------------------------------------------------------------------- annotation-tool helpers
SMOOTH_SIGMA_PITCH = 1.5


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


def tips_from(G, valid, nms_vox, gmin):
    Gm = np.where(valid & np.isfinite(G), G, -1.0)
    mx = ndimage.maximum_filter(Gm, size=2 * nms_vox + 1)
    cand = np.argwhere((Gm >= mx) & (Gm > gmin))
    if len(cand) == 0:
        return np.zeros((0, 3), int)
    order = np.argsort(-Gm[tuple(cand.T)])
    cand = cand[order]
    keep = []
    for c in cand:  # plateau / near-duplicate suppression
        if all(np.linalg.norm(c - k) > nms_vox for k in keep):
            keep.append(c)
    return np.array(keep)


class Surface:
    """Area-weighted surface samples of the envelope (~4 per pitch along a line) for signed distances."""

    def __init__(self, V, F, pitch):
        mesh = trimesh.Trimesh(V, F, process=False)
        n = int(np.clip(mesh.area / (pitch / 4.0) ** 2, 400000, 4000000))
        self.p, fi = trimesh.sample.sample_surface(mesh, n, seed=1)
        self.nrm = mesh.face_normals[fi]
        self.tree = cKDTree(self.p)
        self.n = n

    def signed(self, P):
        """Signed distance (um) to the scan surface, negative inside (sign = vote of the 8 nearest samples)."""
        P = np.atleast_2d(P)
        d, ii = self.tree.query(P, k=8, workers=2)
        s = np.einsum("nkc,nkc->nk", P[:, None, :] - self.p[ii], self.nrm[ii])
        inside = np.sign(s).sum(1) < 0
        return np.where(inside, -d[:, 0], d[:, 0])


def arclen(X):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]


def resample(X, h):
    s = arclen(X)
    if s[-1] < h:
        return X.copy()
    q = np.linspace(0, s[-1], int(np.ceil(s[-1] / h)) + 1)
    return np.stack([np.interp(q, s, X[:, k]) for k in range(3)], 1)


def smooth(X, sigma):
    if len(X) < 5 or sigma <= 0:
        return X.copy()
    return ndimage.gaussian_filter1d(X, sigma, axis=0, mode="nearest")


def march_out(p, u, surf, step, max_len):
    out = []
    t = step
    while t <= max_len:
        q = p + t * u
        out.append(q)
        if surf.signed(q)[0] >= 0:
            break
        t += step
    return np.array(out).reshape(-1, 3)


def end_dir(X, idx_from_end_arc, at_start=True):
    s = arclen(X)
    if at_start:
        k = int(np.searchsorted(s, idx_from_end_arc))
        k = min(max(k, 1), len(X) - 1)
        u = X[0] - X[k]
    else:
        k = int(np.searchsorted(s, s[-1] - idx_from_end_arc))
        k = min(max(k, 0), len(X) - 2)
        u = X[-1] - X[k]
    nu = np.linalg.norm(u)
    return u / nu if nu > 0 else np.array([1.0, 0.0, 0.0])


def build_path(Xvox, pitch, BL, surf, core_target=None, ext_pct=6.0, root_to_surface=False):
    """Dense smoothed medial path (tip -> root) with the tip marched to the surface and an optional straight
    extension into the core (build_path)."""
    h = pitch / 2.0
    X = smooth(resample(Xvox, h), SMOOTH_SIGMA_PITCH * pitch / h)
    u0 = end_dir(X, 2.0 * pitch, at_start=True)
    tipx = march_out(X[0], u0, surf, h / 2, 6 * pitch)
    if len(tipx):
        X = np.vstack([tipx[::-1], X])
    n_tip_ext = len(tipx)
    s_main_end = float(arclen(X)[-1])
    n_core_ext = 0
    if core_target is not None:
        d = core_target - X[-1]
        L = min(ext_pct / 100 * BL, 0.85 * float(np.linalg.norm(d)))
        if L > h:
            m = int(np.ceil(L / h))
            ext = X[-1] + np.outer(np.arange(1, m + 1) * (L / m), d / np.linalg.norm(d))
            X = np.vstack([X, ext])
            n_core_ext = m
    n_root_ext = 0
    if root_to_surface:
        u1 = end_dir(X, 2.0 * pitch, at_start=False)
        rx = march_out(X[-1], u1, surf, h / 2, 0.15 * BL)
        if len(rx):
            X = np.vstack([X, rx])
        n_root_ext = len(rx)
    X = resample(X, h)
    s = arclen(X)
    r = -surf.signed(X)
    return {
        "X": X,
        "s": s,
        "r": r,
        "s_main_end": s_main_end if core_target is not None else float(s[-1]),
        "n_tip_ext": n_tip_ext,
        "n_core_ext": n_core_ext,
        "n_root_ext": n_root_ext,
    }


def coverage(A, B, pitch):
    ma = A["s"] <= A["s_main_end"]
    mb = B["s"] <= B["s_main_end"]
    XA, rA = A["X"][ma], A["r"][ma]
    if len(XA) < 2 or mb.sum() < 2:
        return 0.0
    d, _ = cKDTree(B["X"][mb]).query(XA)
    cov = d <= np.maximum(2.5 * pitch, 0.75 * np.maximum(rA, 0))
    w = np.r_[0.0, np.linalg.norm(np.diff(XA, axis=0), axis=1)]
    return float((w * cov).sum() / max(w.sum(), 1e-9))


def dedup(cands, pitch, BL, cross=0.7, twin=0.9, tip_pct=3.0):
    """dedup: drop duplicate routes (cross-family shared-tip routes, same-family twins, union-covered paths,
    pseudo-tip paths first). Returns (kept indices, dropped records)."""
    tip_tol = max(4 * pitch, tip_pct / 100 * BL)
    keep = list(range(len(cands)))
    dropped = []
    C = {(i, j): coverage(cands[i], cands[j], pitch) for i in keep for j in keep if i != j}

    def cov_by(i, fam):
        other = [j for j in keep if j != i and cands[j]["family"] == fam]
        if not other:
            return 0.0, []
        A = cands[i]
        ma = A["s"] <= A["s_main_end"]
        XA, rA = A["X"][ma], A["r"][ma]
        tol = np.maximum(2.5 * pitch, 0.75 * np.maximum(rA, 0))
        cov = np.zeros(len(XA), bool)
        for j in other:
            B = cands[j]
            mb = B["s"] <= B["s_main_end"]
            cov |= cKDTree(B["X"][mb]).query(XA)[0] <= tol
        w = np.r_[0.0, np.linalg.norm(np.diff(XA, axis=0), axis=1)]
        return float((w * cov).sum() / max(w.sum(), 1e-9)), other

    for victim, cover in (("head", "leg"), ("leg", "head")):
        while True:
            best = None
            for i in keep:
                if cands[i]["family"] != victim:
                    continue
                c, other = cov_by(i, cover)
                shared = [j for j in other if np.linalg.norm(cands[j]["X"][0] - cands[i]["X"][0]) <= tip_tol]
                if c >= cross and shared and (best is None or c > best[1]):
                    best = (i, c, f"{victim}_path_covered_by_{cover}_paths_shared_tip", other)
            if best is None:
                break
            i, c, why, other = best
            keep.remove(i)
            dropped.append({"index": i, "coverage": round(c, 3), "why": why, "covered_by_index": other})
    while True:
        best = None
        for i in keep:
            for j in keep:
                if i < j and cands[i]["family"] == cands[j]["family"] and min(C[(i, j)], C[(j, i)]) >= twin:
                    lose = i if cands[i]["s_main_end"] < cands[j]["s_main_end"] else j
                    c = min(C[(i, j)], C[(j, i)])
                    if best is None or c > best[1]:
                        best = (lose, c, "same_family_twin", [j if lose == i else i])
        if best is None:
            break
        i, c, why, other = best
        keep.remove(i)
        dropped.append({"index": i, "coverage": round(c, 3), "why": why, "covered_by_index": other})

    def cov_union(i):
        A = cands[i]
        ma = A["s"] <= A["s_main_end"]
        XA, rA = A["X"][ma], A["r"][ma]
        tol = np.maximum(2.5 * pitch, 0.75 * np.maximum(rA, 0))
        cov = np.zeros(len(XA), bool)
        for j in keep:
            if j == i:
                continue
            B = cands[j]
            mb = B["s"] <= B["s_main_end"]
            cov |= cKDTree(B["X"][mb]).query(XA)[0] <= tol
        w = np.r_[0.0, np.linalg.norm(np.diff(XA, axis=0), axis=1)]
        return float((w * cov).sum() / max(w.sum(), 1e-9))

    while True:
        cand = [(i, cov_union(i)) for i in keep]
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
                "covered_by_index": [j for j in keep],
            }
        )
    return keep, dropped


# ------------------------------------------------------------------------------------------- new (PATHS-C)
@numba.njit(cache=True)
def _persistence_maxima(G, order, offs):
    """Union-find over voxels in decreasing G (26-neighbourhood, flat indices into a padded array). Returns arrays
    (max voxel index, persistence) for every component that died, plus the survivors with persistence = inf."""
    n = G.size
    parent = np.full(n, -1, np.int64)
    birth = np.zeros(n, np.int64)  # index of the max voxel of the component (stored at the root)
    out_v = np.empty(order.size, np.int64)
    out_p = np.empty(order.size, np.float64)
    m = 0
    for v in order:
        parent[v] = v
        birth[v] = v
        roots = np.empty(26, np.int64)
        nr = 0
        for o in offs:
            u = v + o
            if parent[u] < 0:
                continue
            r = u
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            dup = False
            for k in range(nr):
                if roots[k] == r:
                    dup = True
                    break
            if not dup:
                roots[nr] = r
                nr += 1
        if nr == 0:
            continue
        # the root with the highest birth survives
        best = roots[0]
        for k in range(1, nr):
            if G[birth[roots[k]]] > G[birth[best]]:
                best = roots[k]
        for k in range(nr):
            r = roots[k]
            if r != best:
                out_v[m] = birth[r]
                out_p[m] = G[birth[r]] - G[v]
                m += 1
                parent[r] = best
        parent[v] = best
    # survivors
    for v in order:
        if parent[v] == v:
            out_v[m] = birth[v]
            out_p[m] = np.inf
            m += 1
    return out_v[:m], out_p[:m]


def tips_persistence(G, valid, min_persistence, gmin):
    """Geodesic extremities as maxima of G whose topological persistence (height above the saddle where their branch
    joins a higher one) is at least min_persistence and whose G exceeds gmin. Unlike a spatial non-maximum
    suppression, a limb tip lying next to another limb with larger G is not suppressed."""
    Gp = np.pad(np.where(np.isfinite(G), G, -np.inf), 1, constant_values=-np.inf)
    Vp = np.pad(valid & np.isfinite(G), 1)
    sh = np.array(Gp.shape)
    st = np.array([sh[1] * sh[2], sh[2], 1])
    offs = np.array(
        [
            dx * st[0] + dy * st[1] + dz * st[2]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ],
        np.int64,
    )
    flat = Gp.ravel()
    idx = np.flatnonzero(Vp.ravel() | np.isfinite(flat))
    idx = idx[np.isfinite(flat[idx])]
    order = idx[np.argsort(-flat[idx], kind="stable")]
    v, p = _persistence_maxima(flat, order, offs)
    keep = (p >= min_persistence) & (flat[v] > gmin) & Vp.ravel()[v]
    v, p = v[keep], p[keep]
    o = np.argsort(-flat[v])
    ijk = np.stack(np.unravel_index(v[o], Gp.shape), 1) - 1
    return ijk, p[o]
