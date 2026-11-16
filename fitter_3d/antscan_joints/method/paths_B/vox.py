"""Solid voxelisation (copied from round 3's geometry helpers, unchanged logic)."""

import numba
import numpy as np


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
    """Exact parity (scanline) solid voxelisation of a closed mesh. Voxel (i,j,k) centre = lo + (ijk + 0.5) h."""
    lo = v.min(0) - pad * h
    hi = v.max(0) + pad * h
    n = np.ceil((hi - lo) / h).astype(int)
    g = (v - lo) / h - 0.5
    cnt = _parity_counts(g[f].astype(np.float64), int(n[0]), int(n[1]), int(n[2]))
    occ = (np.cumsum(cnt, axis=2)[:, :, :-1] % 2) == 1
    return occ, lo
