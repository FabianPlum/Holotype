"""The scan side: envelope mesh, its outer surface, ray casting and closest-point queries.

Real envelopes are true outer surfaces. The synthetic sets v3-v6 were processed with `--envelope-voxels 0`, so their
"envelope" is the decimated LBS surface including interior sheets where parts intersect. To make the synthetic test
behave like a real scan, faces that lie inside the solid are removed here (generalised winding number of a point just
outside the face >= 0.5). A subsample decides whether the full test is needed, so real envelopes pay ~1 s.
"""

import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import geom

OUTER_EPS = 5e-4  # probe offset along the normal, relative to the size scale (body length proxy)


def outer_faces(V, F, size, force=False, n_probe=3000, seed=0):
    """Boolean mask of faces on the outer surface. `size`: length scale (body length)."""
    fn, fa = geom.face_normals(V, F)
    C = V[F].mean(1) + OUTER_EPS * size * fn
    if not force:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(F), size=min(n_probe, len(F)), replace=False)
        wn = geom.winding_number(C[sel], V, F, chunk=max(1, int(1e7 // len(F))))
        if (wn >= 0.5).mean() < 0.002:
            return np.ones(len(F), bool), float((wn >= 0.5).mean())
    wn = geom.winding_number(C, V, F, chunk=max(1, int(1e7 // len(F))))
    return wn < 0.5, float((wn >= 0.5).mean())


class Scan:
    """Envelope mesh with the acceleration structures used by all stages. Units: micrometres, scan frame."""

    def __init__(self, V, F, size, cache=None, force_outer=False, assume_outer=False):
        self.Vfull, self.Ffull = V, F
        self.size = float(size)
        keep = None
        if cache and os.path.exists(cache):
            z = np.load(cache)
            if len(z["keep"]) == len(F):
                keep, self.inner_share = z["keep"], float(z["inner_share"])
        if keep is None and assume_outer:  # real envelopes are outer surfaces by construction (Stage 1)
            keep, self.inner_share = np.ones(len(F), bool), 0.0
        elif keep is None:
            keep, self.inner_share = outer_faces(V, F, size, force=force_outer)
            if cache:  # written under a private name and renamed: several workers may build it at the same time
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                tmp = f"{cache}.{os.getpid()}.tmp.npz"
                np.savez_compressed(tmp, keep=keep, inner_share=self.inner_share)
                os.replace(tmp, cache)
        self.face_keep = keep
        self.F = F[keep]
        self.V = V  # vertex indexing unchanged; vertices used by no outer face are flagged
        used = np.zeros(len(V), bool)
        used[self.F.reshape(-1)] = True
        self.vertex_outer = used
        self.fn, self.fa = geom.face_normals(self.V, self.F)
        self.vn = geom.vertex_normals(self.V, self.F)
        self.varea = geom.vertex_areas(self.V, self.F)
        self.fc = self.V[self.F].mean(1)
        self.mesh = trimesh.Trimesh(self.V, self.F, process=False)
        self._ray = None
        self._kd = None
        self.bbox = (V.min(0), V.max(0))

    @property
    def ray(self):
        if self._ray is None:
            self._ray = geom.ray_intersector(self.mesh)
        return self._ray

    @property
    def kd(self):
        if self._kd is None:
            self._kd = cKDTree(self.V[self.vertex_outer])
            self._kd_ids = np.where(self.vertex_outer)[0]
        return self._kd

    def first_hit(self, O, D):
        """First intersection of rays (O, D unit). Returns distance (inf = none), face index (-1), cos between the ray
        and the face normal (> 0: the ray leaves the solid through this face)."""
        O = np.ascontiguousarray(O, dtype=np.float64)
        D = np.ascontiguousarray(D, dtype=np.float64)
        n = len(O)
        dist = np.full(n, np.inf)
        face = np.full(n, -1, dtype=np.int64)
        if n == 0:
            return dist, face, np.zeros(0)
        loc, idx_ray, idx_tri = self.ray.intersects_location(O, D, multiple_hits=False)
        if len(idx_ray):
            d = np.linalg.norm(loc - O[idx_ray], axis=1)
            dist[idx_ray] = d
            face[idx_ray] = idx_tri
        cosn = np.zeros(n)
        ok = face >= 0
        cosn[ok] = (self.fn[face[ok]] * D[ok]).sum(1)
        return dist, face, cosn

    def exit_hit(self, O, D, max_skip=3, eps=None):
        """First intersection through which the ray LEAVES the solid (cos > 0), skipping up to `max_skip` entering
        hits (origins that sit just outside the surface, thin gaps). Returns distance, face, n_skipped."""
        eps = 1e-6 * self.size if eps is None else eps
        O = np.asarray(O, dtype=np.float64).copy()
        D = np.asarray(D, dtype=np.float64)
        n = len(O)
        total = np.zeros(n)
        out_d = np.full(n, np.inf)
        out_f = np.full(n, -1, dtype=np.int64)
        skipped = np.zeros(n, dtype=np.int64)
        active = np.arange(n)
        for _ in range(max_skip + 1):
            if not len(active):
                break
            d, f, c = self.first_hit(O[active], D[active])
            hit = f >= 0
            leave = hit & (c > 0)
            ids = active[leave]
            out_d[ids] = total[ids] + d[leave]
            out_f[ids] = f[leave]
            enter = hit & ~leave
            ids = active[enter]
            total[ids] += d[enter] + eps
            O[ids] = O[ids] + (d[enter] + eps)[:, None] * D[ids]
            skipped[ids] += 1
            active = ids
        return out_d, out_f, skipped

    def closest(self, P):
        """Closest point on the outer surface: distance, face, point."""
        d, f, Q, _ = geom.closest_on_mesh(np.asarray(P, dtype=np.float64), self.V, self.F)
        return d, f, Q


def load_scan(path, size, cache=None, force_outer=False, assume_outer=False):
    V, F = geom.load_mesh(path)
    return Scan(V, F, size, cache=cache, force_outer=force_outer, assume_outer=assume_outer)
