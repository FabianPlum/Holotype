"""Proximal completion of a limb route: from an anchor point on the route (where the route first comes within
`anchor` of the limb family's trunk core) the route is replaced by the voxel min-cost medial path (cost 1 / D^2, the
other trunk cores and a skin around them impassable) to that core, as the truth paths are built. Near the body the
skeleton graph is a mesh of short edges (coxae touching each other and the body), and the min-cost medial path is
the robust way in.
"""

import numpy as np
from scipy import ndimage as ndi


class CorePaths:
    def __init__(self, occ, core, lo, h, D_um, part, BL, margin_pct=25.0, skin_pct=1.0):
        self.lo, self.h, self.part = lo, h, part
        src_full = core == part
        if not src_full.any():
            raise RuntimeError(f"no core voxels with part {part}")
        idx = np.argwhere(src_full)
        m = int(np.ceil(margin_pct / 100 * BL / h))
        self.i0 = np.maximum(idx.min(0) - m, 0)
        self.i1 = np.minimum(idx.max(0) + m + 1, np.array(occ.shape))
        sl = tuple(slice(a, b) for a, b in zip(self.i0, self.i1))
        o = occ[sl]
        c = core[sl]
        src = c == part
        k_skin = max(2, int(round(skin_pct / 100 * BL / h)))
        other = (c >= 0) & ~src
        blocked = ndi.binary_dilation(other, iterations=k_skin) & ~src if other.any() else np.zeros_like(o)
        passable = o & ~blocked
        Dv = np.maximum(D_um[sl] / h, 0.5)
        cost = np.where(passable, 1.0 / Dv**2, np.inf)
        from skimage.graph import MCP_Geometric

        self.mcp = MCP_Geometric(cost, fully_connected=True)
        sb = np.argwhere(src & ~ndi.binary_erosion(src))
        self.cum, _ = self.mcp.find_costs([tuple(s) for s in sb])
        self.dist_core = ndi.distance_transform_edt(~src) * h  # Euclidean distance to the source core (um)
        self.centroid = lo + idx.mean(0) * h
        self.shape = o.shape

    def local(self, X):
        q = np.round((np.asarray(X) - self.lo) / self.h).astype(int) - self.i0
        return q

    def in_box(self, X):
        q = self.local(X)
        return np.all((q >= 0) & (q < np.array(self.shape)), axis=1)

    def dist(self, X):
        q = self.local(X)
        ok = np.all((q >= 0) & (q < np.array(self.shape)), axis=1)
        out = np.full(len(q), np.inf)
        out[ok] = self.dist_core[tuple(q[ok].T)]
        return out

    def trace(self, x):
        """Min-cost medial path (um) from point x to the core boundary, or None."""
        q = self.local(np.asarray(x)[None])[0]
        if not np.all((q >= 0) & (q < np.array(self.shape))):
            return None
        # snap to the nearest finite-cost voxel in a small neighbourhood
        if not np.isfinite(self.cum[tuple(q)]):
            best = None
            for d in range(1, 4):
                lo_ = np.maximum(q - d, 0)
                hi_ = np.minimum(q + d + 1, np.array(self.shape))
                sub = self.cum[lo_[0] : hi_[0], lo_[1] : hi_[1], lo_[2] : hi_[2]]
                f = np.argwhere(np.isfinite(sub))
                if len(f):
                    k = f[np.argmin(np.linalg.norm(f + lo_ - q, axis=1))]
                    best = k + lo_
                    break
            if best is None:
                return None
            q = best
        p = np.asarray(self.mcp.traceback(tuple(q)))[::-1]  # traceback runs source -> point; return point -> core
        return self.lo + (p + self.i0) * self.h
