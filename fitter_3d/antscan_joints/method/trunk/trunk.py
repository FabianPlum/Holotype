"""S6 TRUNK: the trunk of an AntScan worker from its envelope mesh alone (round 6 of the AntScan joint localiser).

Input: the envelope mesh (scan frame, micrometres). No posed model; an input frame (e.g. round 3's) is used only as a
tie-breaker for the head end when the anatomy is ambiguous, and is reported against.

Steps (each pass on a voxel grid that is axis-aligned in a working frame; geometry code copied from
the annotation tool's and round 3's geometry helpers):
  1. coarse pass (PCA frame, pitch = volume^(1/3) / 45, opening radius 0.35 x the largest inscribed radius):
     occupancy (exact parity voxelisation), distance transform, ball-opened core; the two ends of the trunk = the
     farthest pair of large core components under the medial cost 1 / D^2 (double sweep); trunk medial path between
     them; radius profile (distance to the scan surface); bulbs and constrictions; head end by anatomy; trunk joints;
     body length estimate; anatomical frame.
  2. fine pass (anatomical frame, pitch = BL / 350, opening radius 12 voxels = the annotation tool's 4 % of the straightened BL,
     whose pitch is BL_straight / 300 and BL_straight ~ BL / 1.19): the same steps starting from the coarse ends, then
     the ends are moved to the extreme core voxel of the head / gaster part and the path recomputed (as the annotation tool
     does with its model-labelled cores).
Head end (anatomy, no input frame): the trunk path's radius profile splits into bulbs; the mesosoma is the interior
bulb that carries the most attached appendage volume (legs); the head is the end bulb joined to it by one simple
constriction and carrying appendages (antennae, mandibles); the gaster is the end bulb joined through the waist
(node(s), a longer thin stretch) and carrying (almost) nothing. Each cue votes; the confidence is their agreement.
Trunk joints (annotation convention): neck = deepest constriction between mesosoma and head bulbs;
b_a_1 = propodeum-petiole constriction (the waist constriction next to the mesosoma); b_a_3 = (post)petiole-gaster
constriction (next to the gaster); b_a_2 = the petiole node (one-segment waist) or the petiole-postpetiole constriction
(two segments; segment count from the profile); gaster_tip / head_front = the trunk path's ends on the surface;
b_a_4 / b_a_5 = the gaster centre-of-volume convention of round 3's gaster_landmarks (template constants f 0.368 / 0.903,
ventral offsets 1.116 / 1.114). BL = sum of the trunk chords gaster_tip -> b_a_3 -> b_a_2 -> b_a_1 -> neck -> head_front.

Outputs per specimen in <out_dir>/<name>/: trunk.npz, trunk.json (the S6 trunk product, see TRUNK_DOC) and
<name>_joints.json (the joint product contract with only trunk joints filled).

  python trunk.py --names NAME[,NAME..] | --names-file FILE  --out-dir DIR [--workers 4] [--s3-dir DIR] [--skip-existing]
"""

import argparse
import json
import os
import sys
import time
import traceback

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
sys.dont_write_bytecode = True

import numba
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy import ndimage as ndi
from scipy.ndimage import uniform_filter1d
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

VERSION = "S6 trunk 2026-11-10"
J_NAMES = [
    "b_t",
    "b_a_1",
    "b_a_2",
    "b_a_3",
    "b_a_4",
    "b_a_5",
    "l_1_co_r",
    "l_1_tr_r",
    "l_1_fe_r",
    "l_1_ti_r",
    "l_1_ta_r",
    "l_1_pt_r",
    "l_2_co_r",
    "l_2_tr_r",
    "l_2_fe_r",
    "l_2_ti_r",
    "l_2_ta_r",
    "l_2_pt_r",
    "l_3_co_r",
    "l_3_tr_r",
    "l_3_fe_r",
    "l_3_ti_r",
    "l_3_ta_r",
    "l_3_pt_r",
    "l_1_co_l",
    "l_1_tr_l",
    "l_1_fe_l",
    "l_1_ti_l",
    "l_1_ta_l",
    "l_1_pt_l",
    "l_2_co_l",
    "l_2_tr_l",
    "l_2_fe_l",
    "l_2_ti_l",
    "l_2_ta_l",
    "l_2_pt_l",
    "l_3_co_l",
    "l_3_tr_l",
    "l_3_fe_l",
    "l_3_ti_l",
    "l_3_ta_l",
    "l_3_pt_l",
    "b_h",
    "ma_r",
    "ma_l",
    "an_1_r",
    "an_2_r",
    "an_3_r",
    "an_1_l",
    "an_2_l",
    "an_3_l",
    "w_1_r",
    "w_2_r",
    "w_1_l",
    "w_2_l",
]
TRUNK_JOINTS = ["gaster_tip", "b_a_5", "b_a_4", "b_a_3", "b_a_2", "b_a_1", "neck", "head_front"]
PART = {"head": 0, "mesosoma": 1, "waist": 2, "gaster": 3}

CFG = dict(
    coarse_div=45.0,  # coarse pitch = volume^(1/3) / coarse_div
    coarse_open_frac=0.35,  # coarse opening radius = frac x largest inscribed radius (round 3's body_core)
    nvox_bl=350.0,  # fine pitch = BL_est / nvox_bl
    open_vox=12,  # fine opening radius (voxels)
    end_comp_share=0.03,  # core components smaller than this share of the opened volume cannot be trunk ends
    smooth_sigma_pitch=1.5,  # dense path smoothing (annotation tool)
    rw_bl=0.012,  # light profile smoothing window (fraction of BL): constrictions
    rs_bl=0.03,  # heavy profile smoothing window: bulbs
    bulb_prom=0.12,  # bulb = peak of rs with prominence >= bulb_prom x max(rs)
    min_prom=0.06,  # constriction = minimum of rw whose depth below both neighbouring maxima >= min_prom x r
    node_prom=0.06,  # waist node = maximum of rw rising >= node_prom x r above both neighbouring minima
    face_rule="largest_drop",  # gaster face: 'largest_drop' (v2) or 'level' (v1, uses face_level)
    face_level=0.5,  # the gaster face ends at a radius <= face_level x the gaster bulb radius
    drop_frac=0.25,  # gaster face drop ends where the slope magnitude falls below drop_frac x its steepest value
    min_waist_bl=0.04,  # b_a_1 - b_a_3 must span >= this fraction of BL
    min_segment_bl=0.04,  # two-segment waist: petiole and postpetiole each >= this fraction of BL
    waist_prior_bl=0.111,  # fallback b_a_1 - b_a_3 (median one-segment waist of the 21 dev truths)
    end_core_exit=False,  # gaster tip / head front at the core exit when the end march crosses thin material
    end_skin_vox=3.0,  # ... longer than this many voxels
    head_votes={"thin_stretch": 1.0, "end_length": 1.0, "gap": 0.5, "appendages": 0.5},
    gaster_calib={"b_a_4": (0.368, 1.116), "b_a_5": (0.903, 1.114)},  # round 3's gaster_landmarks template constants
)

TRUNK_DOC = {
    "frames": "Working frame: x_frame = R @ (x_scan - origin_scan), R = frame.rotation_scan_to_canonical (rows = frame "
    "axes in scan coordinates, det +1). The voxel grid is axis-aligned in the working frame: voxel (i,j,k) "
    "centre = lo + (i,j,k) * pitch (frame coordinates, um), as in geom.npz. All point arrays named *_X and all "
    "joints are in the SCAN frame (um) unless the key ends in _frame. When frame.anatomical is true the working "
    "frame is this method's anatomical estimate (+X head, +Y animal's left, +Z dorsal).",
    "npz": "occ (np.packbits of the boolean occupancy, C order), shape, lo (frame um, centre of voxel 0), pitch (um), "
    "tpart (int8 grid: -1 not trunk core, 0 head, 1 mesosoma, 2 waist, 3 gaster; the ball-opened trunk core only), "
    "T_X (n,3) scan um gaster tip first, T_X_frame (n,3) frame um, T_s arc from the gaster tip (um), T_r distance "
    "to the scan surface (um, negative outside), T_part (part of the path point: tpart, or the part by arc where "
    "the path leaves the core, e.g. in the waist), D (no; recompute with ndimage.distance_transform_edt(occ)).",
}


# ============================================================================================ mesh and voxels
class MeshError(RuntimeError):
    pass


def load_mesh(path):
    """Largest connected component of the STL (the envelopes' other components are 1-2 face debris)."""
    if not os.path.isfile(path):
        raise MeshError(f"mesh file not found: {path}")
    m = trimesh.load(path, process=True, force="mesh")
    if not hasattr(m, "faces") or len(m.faces) < 4:
        raise MeshError(f"mesh has no usable faces: {path}")
    parts = m.split(only_watertight=False)
    n_parts = len(parts)
    if n_parts > 1:
        m = max(parts, key=lambda q: len(q.faces))
    v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    if not np.isfinite(v).all():
        raise MeshError("non-finite vertices")
    return v, f, {"n_parts": n_parts, "watertight": bool(m.is_watertight), "n_vertices": int(len(v))}


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


def voxelize(Vf, F, h, pad=3):
    """Exact parity solid voxelisation (copied from round 3's voxelize). Returns occ and lo = CENTRE of voxel 0."""
    corner = Vf.min(0) - pad * h
    hi = Vf.max(0) + pad * h
    n = np.ceil((hi - corner) / h).astype(int)
    g = (Vf - corner) / h - 0.5
    cnt = _parity_counts(g[F].astype(np.float64), int(n[0]), int(n[1]), int(n[2]))
    occ = (np.cumsum(cnt, axis=2)[:, :, :-1] % 2) == 1
    return occ, corner + 0.5 * h


def mesh_volume(v, f):
    return abs(float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()) / 6.0)


class Surface:
    """Area-weighted surface samples for signed distances (copied from the annotation tool's Surface)."""

    def __init__(self, V, F, pitch):
        mesh = trimesh.Trimesh(V, F, process=False)
        n = int(np.clip(mesh.area / (pitch / 4.0) ** 2, 400000, 4000000))
        self.p, fi = trimesh.sample.sample_surface(mesh, n, seed=1)
        self.nrm = mesh.face_normals[fi]
        self.tree = cKDTree(self.p)
        self.n = n

    def signed(self, P):
        P = np.atleast_2d(P)
        d, ii = self.tree.query(P, k=8, workers=2)
        s = np.einsum("nkc,nkc->nk", P[:, None, :] - self.p[ii], self.nrm[ii])
        inside = np.sign(s).sum(1) < 0
        return np.where(inside, -d[:, 0], d[:, 0])


def pca_frame(V, F):
    """Rotation whose rows are the principal axes of the surface (largest first), det +1; origin = area centroid."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    w = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    cen = (a + b + c) / 3.0
    o = (w[:, None] * cen).sum(0) / w.sum()
    C = ((cen - o).T * w) @ (cen - o) / w.sum()
    _, ev = np.linalg.eigh(C)
    R = ev[:, ::-1].T.copy()
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return R, o


# ============================================================================================ paths
def to_ijk(Xf, lo, pitch, shape):
    q = np.round((np.asarray(Xf) - lo) / pitch).astype(int)
    return np.clip(q, 0, np.array(shape) - 1)


def mcp_from(cost, starts, ends=None):
    from skimage.graph import MCP_Geometric

    m = MCP_Geometric(cost, fully_connected=True)
    if ends is None:
        cum, _ = m.find_costs([tuple(s) for s in starts])
    else:
        cum, _ = m.find_costs([tuple(s) for s in starts], ends=[tuple(e) for e in ends], find_all_ends=True)
    return m, cum


def arclen(X):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]


def resample(X, h):
    s = arclen(X)
    if s[-1] < h:
        return X.copy()
    q = np.linspace(0, s[-1], int(np.ceil(s[-1] / h)) + 1)
    return np.stack([np.interp(q, s, X[:, k]) for k in range(3)], 1)


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


def end_dir(X, arc, at_start=True):
    s = arclen(X)
    if at_start:
        k = min(max(int(np.searchsorted(s, arc)), 1), len(X) - 1)
        u = X[0] - X[k]
    else:
        k = min(max(int(np.searchsorted(s, s[-1] - arc)), 0), len(X) - 2)
        u = X[-1] - X[k]
    nu = np.linalg.norm(u)
    return u / nu if nu > 0 else np.array([1.0, 0.0, 0.0])


def dense_path(Xvox, pitch, surf, BL):
    """the annotation tool's build_path for the trunk: resample every pitch/2, Gaussian smoothing (sigma 1.5 pitch), both ends
    marched along their end tangent to the scan surface. Returns X, s, r, n_tip_ext, n_root_ext."""
    h = pitch / 2.0
    X = resample(Xvox, h)
    if len(X) >= 5:
        X = ndi.gaussian_filter1d(X, CFG["smooth_sigma_pitch"] * pitch / h, axis=0, mode="nearest")
    u0 = end_dir(X, 2.0 * pitch, True)
    tipx = march_out(X[0], u0, surf, h / 2, 0.15 * BL)
    if len(tipx):
        X = np.vstack([tipx[::-1], X])
    u1 = end_dir(X, 2.0 * pitch, False)
    rx = march_out(X[-1], u1, surf, h / 2, 0.15 * BL)
    if len(rx):
        X = np.vstack([X, rx])
    X = resample(X, h)
    s = arclen(X)
    # 5-point running median: removes isolated sign-vote failures of the surface distance (single points at -300 um
    # deep inside the body on 3 of 103 production specimens), keeps the true sign change at the marched ends
    r = ndi.median_filter(-surf.signed(X), size=5, mode="nearest")
    return X, s, r, len(tipx), len(rx)


# ============================================================================================ core and ends
def opened_core(occ, D, r):
    core = D > r
    opened = (ndi.distance_transform_edt(~core) <= r) & occ
    lab, k = ndi.label(opened)
    vols = np.bincount(lab.ravel(), minlength=k + 1)[1:] if k else np.zeros(0, int)
    return opened, lab, vols


def far_ends(occ, D, lab, vols, share):
    """Double sweep under the medial cost 1 / max(D, 0.5)^2 restricted to the solid: from the deepest voxel the
    farthest voxel of a large core component (A), then from A the farthest voxel of a large core component (B)."""
    big = [i + 1 for i, v in enumerate(vols) if v >= share * vols.sum()]
    cand = np.isin(lab, big)
    Dc = np.maximum(D, 0.5)
    cost = np.where(occ, 1.0 / Dc**2, np.inf)
    seed = np.unravel_index(int(np.argmax(D)), D.shape)
    _, cum = mcp_from(cost, [seed])
    cc = np.where(cand & np.isfinite(cum), cum, -1)
    A = np.unravel_index(int(np.argmax(cc)), D.shape)
    _, cum = mcp_from(cost, [A])
    cc = np.where(cand & np.isfinite(cum), cum, -1)
    B = np.unravel_index(int(np.argmax(cc)), D.shape)
    return np.array(A), np.array(B), cost, {"n_big_components": len(big), "n_components": int(len(vols))}


def trace(cost, a, b):
    m, cum = mcp_from(cost, [a], [b])
    if not np.isfinite(cum[tuple(b)]):
        return None
    return np.asarray(m.traceback(tuple(b)))


# ============================================================================================ labels by arc
def arc_labels(mask, src_ijk, src_arc):
    """Geodesic Voronoi of the voxels of `mask` (6-connected graph) with the trunk path voxels as sources: arc of the
    nearest source per mask voxel (np.nan where unreachable). Returns an arc grid (float32, nan outside)."""
    idx = -np.ones(mask.shape, np.int64)
    pts = np.argwhere(mask)
    idx[tuple(pts.T)] = np.arange(len(pts))
    rows, cols = [], []
    for ax in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        a = idx[tuple(sl_a)]
        b = idx[tuple(sl_b)]
        ok = (a >= 0) & (b >= 0)
        rows.append(a[ok])
        cols.append(b[ok])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    n = len(pts)
    G = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    sid = idx[tuple(src_ijk.T)]
    ok = sid >= 0
    sid, sarc = sid[ok], np.asarray(src_arc)[ok]
    out = np.full(mask.shape, np.nan, np.float32)
    if len(sid) == 0:
        return out
    uniq, first = np.unique(sid, return_index=True)
    _, _, srcs = dijkstra(G, directed=False, indices=uniq, min_only=True, return_predecessors=True)
    arc_of = dict(zip(uniq.tolist(), sarc[first].tolist()))
    lut = np.full(n, np.nan)
    good = srcs >= 0
    lut[good] = np.array([arc_of[int(x)] for x in srcs[good]])
    out[tuple(pts.T)] = lut
    return out


# ============================================================================================ profile analysis
def smooth_profile(s, r, win):
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    k = max(3, int(round(win / max(ds, 1e-9))) | 1)
    return uniform_filter1d(np.asarray(r, float), size=k, mode="nearest")


def local_max(y, lo, hi):
    return [q for q in range(max(lo, 1), min(hi, len(y) - 1)) if y[q] >= y[q - 1] and y[q] > y[q + 1]]


def local_min(y, lo, hi):
    return [q for q in range(max(lo, 1), min(hi, len(y) - 1)) if y[q] <= y[q - 1] and y[q] < y[q + 1]]


def prominent_minima(rw, lo, hi, prom):
    """Local minima of rw in (lo, hi) whose depth below the lower of the neighbouring maxima (up to the next kept
    minimum or the interval end) is >= prom x their own radius (weakest dropped first; srefine._prominent_minima)."""
    M = local_min(rw, lo + 1, hi)
    while M:
        pr = []
        for j, q in enumerate(M):
            a = M[j - 1] if j > 0 else lo
            b = M[j + 1] if j < len(M) - 1 else hi
            pr.append(min(float(rw[a : q + 1].max()), float(rw[q : b + 1].max())) - float(rw[q]))
        rel = [p / max(float(rw[q]), 1e-9) for p, q in zip(pr, M)]
        j = int(np.argmin(rel))
        if rel[j] >= prom:
            break
        M.pop(j)
    return M


def bulbs(s, rs, prom):
    """Peaks of rs padded with zeros at both ends with prominence >= prom x max (scipy find_peaks)."""
    from scipy.signal import find_peaks

    pr = np.r_[0.0, rs, 0.0]
    pk, props = find_peaks(pr, prominence=prom * float(rs.max()))
    return [int(p - 1) for p in pk], props["prominences"].tolist()


def attachment_profile(occ, trunk, arcg, s):
    """Appendage volume attached along the trunk path: the thin solid (occ minus the trunk core) is split into
    6-connected components; each component's volume is spread over the trunk-core voxels it touches (the interface),
    and those are binned by their arc label. Returns att (len(s), voxels), the interface voxels (ijk), their weights
    and arcs, and a summary."""
    thin = occ & ~trunk
    lab, k = ndi.label(thin)
    att = np.zeros(len(s))
    empty = (att, np.zeros((0, 3), int), np.zeros(0), np.zeros(0), {"n_thin_components": int(k)})
    if k == 0:
        return empty
    vol = np.bincount(lab.ravel(), minlength=k + 1)
    nb = ndi.maximum_filter(lab, footprint=ndi.generate_binary_structure(3, 1))
    inter = trunk & (nb > 0) & np.isfinite(arcg)
    if not inter.any():
        return empty
    ijk = np.argwhere(inter)
    li = nb[inter]
    ai = arcg[inter].astype(float)
    cnt = np.bincount(li, minlength=k + 1)
    w = vol[li] / np.maximum(cnt[li], 1)
    q = np.clip(np.searchsorted(s, ai), 0, len(s) - 1)
    np.add.at(att, q, w)
    return att, ijk, w, ai, {"n_thin_components": int(k), "attached_vox": float(w.sum()), "thin_vox": int(thin.sum())}


# ============================================================================================ grid
class Grid:
    """Occupancy, distance transform and medial cost on a grid axis-aligned in the frame (R, o)."""

    def __init__(self, V, F, R, o, pitch):
        self.R, self.o, self.pitch = np.asarray(R, float), np.asarray(o, float), float(pitch)
        self.occ, self.lo = voxelize((V - self.o) @ self.R.T, F, self.pitch)
        self.shape = self.occ.shape
        self.D = ndi.distance_transform_edt(self.occ)
        self.cost = np.where(self.occ, 1.0 / np.maximum(self.D, 0.5) ** 2, np.inf)

    def core(self, r):
        """Ball-opened core (cached per radius)."""
        if not hasattr(self, "_core"):
            self._core = {}
        if r not in self._core:
            self._core[r] = opened_core(self.occ, self.D, r)
        return self._core[r]

    def ijk_to_scan(self, ijk):
        return (self.lo + np.asarray(ijk, float) * self.pitch) @ self.R + self.o

    def scan_to_frame(self, X):
        return (np.asarray(X, float) - self.o) @ self.R.T

    def scan_to_ijk(self, X):
        return to_ijk(self.scan_to_frame(X), self.lo, self.pitch, self.shape)


def trunk_route(G, a, b, surf, BL, lab):
    """Voxel route a -> b under the medial cost, the dense path in scan coordinates, the trunk core mask (opened core
    components the route passes through) and the arc label grid of that mask."""
    # the route runs through the trunk: search it inside the bounding box of the opened core (+ margin) for speed,
    # on the whole grid only when that fails
    core_ijk = np.argwhere(lab > 0)
    m_ = 10
    lo_b = np.maximum(np.minimum(core_ijk.min(0), np.minimum(a, b)) - m_, 0)
    hi_b = np.minimum(np.maximum(core_ijk.max(0), np.maximum(a, b)) + m_ + 1, np.array(G.shape))
    sl = tuple(slice(int(x), int(y)) for x, y in zip(lo_b, hi_b))
    path = trace(G.cost[sl], np.asarray(a) - lo_b, np.asarray(b) - lo_b)
    if path is not None:
        path = path + lo_b
    else:
        path = trace(G.cost, a, b)
    if path is None:
        raise RuntimeError("no trunk route between the ends")
    X, s, r, nt, nr = dense_path(G.ijk_to_scan(path), G.pitch, surf, BL)
    lp = lab[tuple(path.T)]
    comps = [int(c) for c in np.unique(lp) if c > 0 and (lp == c).sum() >= 3]
    trunk = np.isin(lab, comps)
    q = G.scan_to_ijk(X)
    arcg = arc_labels(trunk, q, s)
    return {
        "route": path,
        "X": X,
        "s": s,
        "r": r,
        "n_end_ext": (nt, nr),
        "trunk": trunk,
        "arcg": arcg,
        "trunk_components": comps,
    }


def flip_route(T):
    L = float(T["s"][-1])
    T = dict(T)
    T["route"] = T["route"][::-1]
    T["X"] = T["X"][::-1].copy()
    T["r"] = T["r"][::-1].copy()
    T["s"] = L - T["s"][::-1]
    T["arcg"] = (L - T["arcg"]).astype(np.float32)
    T["n_end_ext"] = T["n_end_ext"][::-1]
    return T


# ============================================================================================ anatomy on the profile
def profile_features(s, r, att, BL):
    """Bulbs (peaks of the heavily smoothed radius), their basins and attached appendage volume, and per end the
    structure between the end bulb and the mesosoma bulb (interior bulbs = waist nodes, length of the thin stretch).
    The mesosoma bulb = the interior bulb with the most attached appendage volume (legs)."""
    rw = smooth_profile(s, r, CFG["rw_bl"] * BL)
    rs = smooth_profile(s, r, CFG["rs_bl"] * BL)
    pk, prom = bulbs(s, rs, CFG["bulb_prom"])
    f = {
        "bulbs_pctL": [round(100 * s[p] / BL, 1) for p in pk],
        "bulb_r_um": [round(float(rs[p]), 1) for p in pk],
        "bulb_prom_um": [round(float(x), 1) for x in prom],
        "L_pctBL": round(100 * s[-1] / BL, 1),
    }
    cuts = [0] + [pk[i] + int(np.argmin(rs[pk[i] : pk[i + 1] + 1])) for i in range(len(pk) - 1)] + [len(s) - 1]
    att_b = [float(att[cuts[i] : cuts[i + 1] + 1].sum()) for i in range(len(pk))]
    tot = max(float(att.sum()), 1e-9)
    f["basin_cuts_pctL"] = [round(100 * s[c] / BL, 1) for c in cuts]
    f["basin_att_share"] = [round(a / tot, 3) for a in att_b]
    f["pk"], f["cuts"] = pk, cuts
    if len(pk) >= 3:
        mi = 1 + int(np.argmax(att_b[1:-1]))
    elif len(pk) == 2:
        mi = int(np.argmax(att_b))
    else:
        mi = 0
    f["meso_bulb"] = mi
    ends = {}
    for end, e in (("A", 0), ("B", len(pk) - 1)):
        if len(pk) < 2 or e == mi:
            ends[end] = {
                "n_nodes": None,
                "thin_pctBL": None,
                "att_share": round(att_b[e] / tot, 3) if pk else None,
                "end_is_meso": True,
            }
            continue
        i0, i1 = sorted((pk[e], pk[mi]))
        r_lo = min(rs[pk[e]], rs[pk[mi]])
        nodes = [j for j in range(len(pk)) if i0 < pk[j] < i1]
        seg = rw[i0 : i1 + 1]
        thin = float(np.sum(seg < 0.5 * r_lo) * np.median(np.diff(s)))
        rmin = float(seg.min())
        ends[end] = {
            "n_nodes": len(nodes),
            "thin_pctBL": round(100 * thin / BL, 2),
            "att_share": round(att_b[e] / tot, 3),
            "min_r_rel": round(rmin / r_lo, 3),
            "end_r_um": round(float(rs[pk[e]]), 1),
            "gap_pctBL": round(100 * (s[i1] - s[i0]) / BL, 2),
            "end_is_meso": False,
        }
    f["ends"] = ends
    return f, rw, rs


def decide_head(f):
    """Votes for 'end A is the gaster' (+) or 'end A is the head' (-) from the anatomy cues (weights in CFG):
      thin_stretch  the thin stretch between the end bulb and the mesosoma bulb (radius below half the smaller bulb) is
                    longer on the gaster side (waist) than on the head side (neck); full vote at 2 % BL difference
      end_length    the end basin (end -> first basin cut) is longer for the gaster than for the head; full at 10 % BL
      gap           the end bulb's peak lies farther from the mesosoma bulb on the gaster side; full at 10 % BL
      appendages    the head basin carries antennae and mandibles, the gaster basin ~nothing (log ratio, full at 4x)
    (v1 also voted with the number of interior bulbs; dropped: wrong on 5 of 34 production specimens where it voted,
    e.g. a pronotum bulb on the head side.) Returns (A_is_gaster or None, confidence 0..1, votes)."""
    A, B = f["ends"]["A"], f["ends"]["B"]
    W = CFG["head_votes"]
    votes = {}
    if A["thin_pctBL"] is not None and B["thin_pctBL"] is not None:
        votes["thin_stretch"] = W["thin_stretch"] * float(np.clip((A["thin_pctBL"] - B["thin_pctBL"]) / 2.0, -1, 1))
        votes["gap"] = W["gap"] * float(np.clip((A["gap_pctBL"] - B["gap_pctBL"]) / 10.0, -1, 1))
    cuts = f["basin_cuts_pctL"]
    if len(cuts) >= 3:
        votes["end_length"] = W["end_length"] * float(
            np.clip(((cuts[1] - cuts[0]) - (cuts[-1] - cuts[-2])) / 10.0, -1, 1)
        )
    a, b = A["att_share"], B["att_share"]
    if a is not None and b is not None:
        votes["appendages"] = W["appendages"] * float(np.clip(np.log((b + 0.01) / (a + 0.01)) / np.log(4.0), -1, 1))
    if not votes:
        return None, 0.0, votes
    tot = sum(votes.values())
    wsum = sum(W[k] for k in votes)
    return (tot > 0), float(min(1.0, abs(tot) / wsum)), votes


def place_trunk_joints(s, r, rw, rs, f, BL):
    """Trunk articulations on the gaster-first profile. Returns ({joint: (arc, rule, conf)}, info)."""
    pk, mi = f["pk"], f["meso_bulb"]
    n = len(s)
    out, info = {}, {}
    i_g = pk[0]
    i_m = pk[mi] if mi is not None else int(np.argmax(rs))
    i_h = pk[-1]
    ds = float(np.median(np.diff(s)))  # noqa: F841
    # ---- neck: deepest point of rw between the mesosoma bulb and the head bulb
    if i_h > i_m + 2:
        q = i_m + int(np.argmin(rw[i_m : i_h + 1]))
        rel = float(rw[q] / min(rs[i_m], rs[i_h]))
        out["neck"] = (float(s[q]), "neck_constriction", float(np.clip(1.5 - rel * 1.5, 0.1, 1.0)))
        info["neck_r_rel"] = round(rel, 3)
    else:
        a = int(np.searchsorted(s, s[-1] - 0.35 * BL))
        b = int(np.searchsorted(s, s[-1] - 0.08 * BL))
        q = a + int(np.argmin(rw[a : max(b, a + 1)]))
        out["neck"] = (float(s[q]), "neck_no_head_bulb_min", 0.2)
    i_n = q  # noqa: F841
    # ---- waist between the gaster bulb and the mesosoma bulb
    if i_m - i_g < 4:
        info["why"] = "no_waist_interval"
        return out, info
    M = prominent_minima(rw, i_g, i_m, CFG["min_prom"])
    r_g, r_m = float(rs[i_g]), float(rs[i_m])
    info["waist_minima_pctBL"] = [round(100 * s[q] / BL, 2) for q in M]
    info["waist_minima_r"] = [round(float(rw[q]), 1) for q in M]
    ant = [q for q in M if rw[q] <= 0.6 * r_m]
    if not ant:
        info["why"] = "no_propodeum_constriction"
        return out, info
    q1 = max(ant)
    out["b_a_1"] = (float(s[q1]), "propodeum_petiole_constriction", float(np.clip(1.2 - 1.5 * rw[q1] / r_m, 0.2, 0.9)))
    # ---- b_a_3: where the gaster's anterior face stops dropping (srefine._drop_end idea), snapped to a constriction
    g = np.gradient(rw, s)
    min_w = CFG["min_waist_bl"] * BL
    if CFG["face_rule"] == "level":
        # v1: the face must end at a waist-like radius (<= face_level x the gaster bulb); a drop that ends higher (a
        # step on a long tapering gaster) is not the face; search again from there
        start, steps = i_g, []
        for _ in range(6):
            ih = next((i for i in range(start, q1) if rw[i] <= CFG["face_level"] * r_g), q1)
            i_s = start + int(np.argmin(g[start : max(ih, start + 1) + 1]))
            thr = CFG["drop_frac"] * abs(float(g[i_s]))
            i_e = next((i for i in range(i_s, q1) if -g[i] < thr), q1)
            steps.append(round(100 * s[min(i_e, n - 1)] / BL, 2))
            if rw[min(i_e, n - 1)] <= CFG["face_level"] * r_g or i_e >= q1 or i_e <= start:
                break
            start = i_e
    else:
        # v2: every descent episode between the gaster bulb and b_a_1 (a descending run of rw split where its slope
        # flattens below drop_frac x its steepest value); the gaster face = the episode with the largest radius drop
        # among those ending >= min_waist in front of b_a_1
        eps, start = [], i_g
        while start < q1 - 1:
            k = next((i for i in range(start, q1) if g[i] < 0), None)
            if k is None:
                break
            m = next((i for i in range(k, q1 + 1) if g[i] >= 0), q1)
            i_s = k + int(np.argmin(g[k : max(m, k + 1)]))
            thr = CFG["drop_frac"] * abs(float(g[i_s]))
            i_e = next((i for i in range(i_s, m + 1) if -g[i] < thr), m)
            eps.append((float(rw[k : i_s + 1].max() - rw[i_e]), k, i_s, i_e))
            start = max(i_e, k) + 1
        ok_eps = [e for e in eps if s[q1] - s[e[3]] >= min_w]
        steps = [round(100 * s[e[3]] / BL, 2) for e in eps]
        info["gaster_face_drops_um"] = [round(e[0], 1) for e in eps]
        if ok_eps:
            _, k, i_s, i_e = max(ok_eps, key=lambda e: e[0])
        else:
            i_s, i_e = (eps[-1][2], eps[-1][3]) if eps else (i_g, q1)
    info["gaster_drop_ends_tried_pctBL"] = steps
    info["gaster_face_steepest_pctBL"] = round(100 * s[i_s] / BL, 2)
    info["gaster_drop_end_pctBL"] = round(100 * s[i_e] / BL, 2)
    near = [q for q in M if q < q1 and s[i_e] - 0.01 * BL <= s[q] <= s[i_e] + 0.03 * BL and s[q1] - s[q] >= min_w]
    q3, rule3 = None, None
    if near:
        q3 = min(near, key=lambda q: abs(s[q] - s[i_e]))
        rule3 = "gaster_constriction"
    elif s[q1] - s[i_e] >= min_w:
        q3, rule3 = i_e, "gaster_shoulder"
    if q3 is None:
        # no gaster face ends in front of b_a_1 (a gaster tapering into a broadly attached petiole): b_a_3 by the
        # median one-segment waist length of the dev truths (proportion), b_a_2 in the middle; low confidence
        info["why_b_a_3"] = (
            f"gaster drop end {100 * (s[q1] - s[i_e]) / BL:.1f} % BL from b_a_1 (< {100 * CFG['min_waist_bl']:.0f}): no waist feature"
        )
        info["waist_segments"] = None
        s3 = max(float(s[q1]) - CFG["waist_prior_bl"] * BL, float(s[i_g]))
        out["b_a_3"] = (s3, "proportion_waist_length", 0.15)
        out["b_a_2"] = (0.5 * (s3 + float(s[q1])), "proportion_waist_middle", 0.1)
        return out, info
    # ---- segments: an interior constriction between two real nodes -> petiole + postpetiole
    dlt = 0.015 * BL
    M2 = [q for q in M if s[q3] + dlt <= s[q] <= s[q1] - dlt]
    seg, q2 = 1, None
    if M2:
        qq = min(M2, key=lambda q: rw[q])
        m1 = float(rw[q3 : qq + 1].max())
        m2 = float(rw[qq : q1 + 1].max())
        ok1 = m1 - rw[qq] >= CFG["node_prom"] * m1 and (
            rule3 == "gaster_shoulder" or m1 - rw[q3] >= CFG["node_prom"] * m1
        )
        ok2 = m2 - max(rw[qq], rw[q1]) >= CFG["node_prom"] * m2
        info["interior_constriction_pctBL"] = round(100 * s[qq] / BL, 2)
        info["interior_nodes_ok"] = [bool(ok1), bool(ok2)]
        # both segments must have a plausible length (a double minimum at b_a_1 is not a postpetiole)
        ok3 = min(s[q1] - s[qq], s[qq] - s[q3]) >= CFG["min_segment_bl"] * BL
        info["interior_segments_long_enough"] = bool(ok3)
        if ok1 and ok2 and ok3:
            seg, q2 = 2, qq
    info["waist_segments"] = seg
    info["waist_length_pctBL"] = round(100 * (s[q1] - s[q3]) / BL, 2)
    out["b_a_3"] = (
        float(s[q3]),
        ("postpetiole" if seg == 2 else "petiole") + "_" + rule3,
        0.6 if rule3 == "gaster_constriction" else 0.45,
    )
    if seg == 2:
        out["b_a_2"] = (float(s[q2]), "petiole_postpetiole_constriction", 0.6)
    else:
        qn = q3 + int(np.argmax(rw[q3 : q1 + 1]))
        if q3 < qn < q1 and rw[qn] - max(rw[q3], rw[q1]) >= CFG["node_prom"] * rw[qn]:
            out["b_a_2"] = (float(s[qn]), "petiole_node", 0.5)
        else:  # no node rises inside the petiole: its middle
            out["b_a_2"] = (0.5 * float(s[q3] + s[q1]), "petiole_middle_no_node", 0.25)
    return out, info


def gaster_landmarks(Xg, a, dorsal, h, calib):
    """round 3's gaster_landmarks (copied): b_a_4 / b_a_5 at the template's axial fractions of the gaster core's extent
    along the axis from b_a_3 through the gaster core centroid, offset ventrally by a fraction of the slab's ventral
    half extent. Xg: gaster core voxel centres (scan um)."""
    if len(Xg) < 30:
        return {}, {"status": "no_gaster_core"}
    ax = Xg.mean(0) - a
    if np.linalg.norm(ax) < 1e-9:
        return {}, {"status": "degenerate_axis"}
    ax = ax / np.linalg.norm(ax)
    t = (Xg - a) @ ax
    t0, t1 = np.percentile(t, [1, 99])
    dz = dorsal - (dorsal @ ax) * ax
    if np.linalg.norm(dz) < 1e-9:
        return {}, {"status": "no_dorsal_direction"}
    dz = dz / np.linalg.norm(dz)
    span = t1 - t0
    out, info = {}, {"status": "ok", "gaster_length_um": round(float(span), 1)}
    for name, (fr, o) in calib.items():
        tt = t0 + fr * span
        slab = np.abs(t - tt) <= max(1.5 * h, 0.03 * span)
        if slab.sum() < 5:
            continue
        S = Xg[slab]
        c = S.mean(0)
        half = float(np.percentile((S - c) @ (-dz), 97)) + 0.5 * h
        out[name] = c + o * half * (-dz)
        info[name] = {"fraction": fr, "ventral_offset": o, "slab_vox": int(slab.sum())}
    return out, info


def point_at(X, s, a):
    return np.array([np.interp(a, s, X[:, k]) for k in range(3)])


# ============================================================================================ passes
def analyse(G, A, B, surf, BL, r_open, orient=None):
    """One analysis on grid G between voxel ends A, B: route, attachments, features, head decision (or the given
    orientation 'A_gaster'), oriented gaster first, joints. Returns a dict."""
    opened, lab, vols = G.core(r_open)
    T = trunk_route(G, A, B, surf, BL, lab)
    att, iijk, iw, iarc, ainfo = attachment_profile(G.occ, T["trunk"], T["arcg"], T["s"])
    f, rw, rs = profile_features(T["s"], T["r"], att, BL)
    A_gaster, conf, votes = decide_head(f)
    dec = {
        "A_is_gaster": A_gaster,
        "confidence": round(conf, 3),
        "votes": {k: round(v, 3) for k, v in votes.items()},
        "features": {k: v for k, v in f.items() if k not in ("pk", "cuts")},
    }
    use = A_gaster if orient is None else orient
    if use is None:
        use = True
    if not use:
        T = flip_route(T)
        att = att[::-1].copy()
        iarc = float(T["s"][-1]) - iarc
        f, rw, rs = profile_features(T["s"], T["r"], att, BL)
    joints, jinfo = place_trunk_joints(T["s"], T["r"], rw, rs, f, BL)
    return {
        "T": T,
        "att": att,
        "iface": (iijk, iw, iarc),
        "ainfo": ainfo,
        "f": f,
        "rw": rw,
        "rs": rs,
        "decision": dec,
        "used_A_gaster": bool(use),
        "joints": joints,
        "jinfo": jinfo,
        "lab": lab,
        "vols": vols,
        "r_open": r_open,
        "ends_vox": (T["route"][0], T["route"][-1]),
    }


def part_bounds(R, BL):
    """Arc boundaries (gaster|waist, waist|mesosoma, mesosoma|head) from the joints, basin cuts as fallbacks."""
    s, f, J = R["T"]["s"], R["f"], R["joints"]
    cuts = [s[c] for c in f["cuts"]]
    mi = f["meso_bulb"]
    b3 = J.get("b_a_3", (None,))[0]
    b1 = J.get("b_a_1", (None,))[0]
    nk = J.get("neck", (None,))[0]
    if b1 is None:
        b1 = cuts[mi] if mi is not None and mi >= 1 else 0.4 * s[-1]
    if b3 is None:
        b3 = b1
    if nk is None:
        nk = cuts[mi + 1] if mi is not None and mi + 1 < len(cuts) - 1 else 0.85 * s[-1]
    return float(b3), float(b1), float(nk)


def tpart_grid(R, BL):
    b3, b1, nk = part_bounds(R, BL)
    arcg, trunk = R["T"]["arcg"], R["T"]["trunk"]
    tp = np.full(arcg.shape, -1, np.int8)
    ok = trunk & np.isfinite(arcg)
    a = arcg[ok]
    lab = np.where(a >= nk, 0, np.where(a >= b1, 1, np.where(a >= b3, 2, 3))).astype(np.int8)
    tp[ok] = lab
    return tp, (b3, b1, nk)


def refined_ends(G, R, tp, BL):
    """Gaster / head end voxels: the core voxel of the part farthest along (part centroid - articulation)."""
    X, s = R["T"]["X"], R["T"]["s"]
    b3, b1, nk = part_bounds(R, BL)
    out = []
    for part, a in ((3, b3), (0, nk)):
        ijk = np.argwhere(tp == part)
        if len(ijk) < 5:
            out.append(R["ends_vox"][0] if part == 3 else R["ends_vox"][1])
            continue
        P = G.ijk_to_scan(ijk)
        J = point_at(X, s, a)
        u = P.mean(0) - J
        if np.linalg.norm(u) < 1e-9:
            out.append(R["ends_vox"][0] if part == 3 else R["ends_vox"][1])
            continue
        out.append(ijk[int(np.argmax((P - J) @ (u / np.linalg.norm(u))))])
    return out[0], out[1]


def end_points(G, R):
    """Gaster tip and head front on the trunk path. The path is marched along its end tangents to the scan surface;
    when that march runs on through thin material outside the opened trunk core for more than end_skin_vox voxels (a
    closed mandible in front of the clypeus, a protruding sting), the end is where the path leaves the core instead.
    Returns ({name: (index, rule)})."""
    X = R["T"]["X"]
    q = G.scan_to_ijk(X)
    ins = R["T"]["trunk"][tuple(q.T)]
    out = {}
    idx = np.where(ins)[0]
    if len(idx) == 0 or not CFG["end_core_exit"]:
        return {"gaster_tip": (0, "surface"), "head_front": (len(X) - 1, "surface")}
    s = R["T"]["s"]
    i0, i1 = int(idx[0]), int(idx[-1])
    skin = CFG["end_skin_vox"] * G.pitch
    out["gaster_tip"] = (i0, "core_exit") if s[i0] - s[0] > skin else (0, "surface")
    out["head_front"] = (i1, "core_exit") if s[-1] - s[i1] > skin else (len(X) - 1, "surface")
    return out


def joints_xyz(R, G=None):
    X, s = R["T"]["X"], R["T"]["s"]
    out = {"gaster_tip": X[0].copy(), "head_front": X[-1].copy()}
    if G is not None:
        ep = end_points(G, R)
        R["end_rules"] = {k: v[1] for k, v in ep.items()}
        R["end_arcs"] = {k: float(s[v[0]]) for k, v in ep.items()}
        for k, (i, rule) in ep.items():
            out[k] = X[i].copy()
    for j, (a, rule, c) in R["joints"].items():
        out[j] = point_at(X, s, a)
    return out


def body_length(P):
    seq = ["gaster_tip", "b_a_3", "b_a_2", "b_a_1", "neck", "head_front"]
    pts = [P[k] for k in seq if P.get(k) is not None]
    if len(pts) < 2:
        return None, []
    missing = [k for k in seq if P.get(k) is None]
    return float(sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1))), missing


def anatomical_frame(G, R, P):
    """+X: mesosoma chord b_a_1 -> neck (towards the head); +Z dorsal: away from the mean attachment direction of the
    appendages on the mesosoma (coxae attach ventrally), weighted by the attached volume; +Y = Z x X (animal's left).
    Origin: the trunk path point midway between b_a_1 and the neck. Returns (R rows = axes in scan coords, origin,
    info)."""
    X, s = R["T"]["X"], R["T"]["s"]
    b3, b1, nk = part_bounds(R, None)
    pn, p1 = point_at(X, s, nk), point_at(X, s, b1)
    ex = pn - p1
    if np.linalg.norm(ex) < 1e-9:
        ex = X[-1] - X[0]
    ex = ex / np.linalg.norm(ex)
    iijk, iw, iarc = R["iface"]
    sel = (iarc >= b1) & (iarc <= nk)
    info = {"x_from": "b_a_1 -> neck"}
    ez = None
    if sel.sum() >= 10:
        Pi = G.ijk_to_scan(iijk[sel])
        Q = np.stack([np.interp(iarc[sel], s, X[:, k]) for k in range(3)], 1)
        v = Pi - Q
        v = v - np.outer(v @ ex, ex)
        nv = np.linalg.norm(v, axis=1)
        ok = nv > 1e-9
        u = v[ok] / nv[ok, None]
        m = (u * iw[sel][ok, None]).sum(0) / iw[sel][ok].sum()
        info["dorsal_strength"] = round(float(np.linalg.norm(m)), 3)
        info["ventral_attach_scan"] = m.tolist()
        # + head flexion: the head front hangs ventrally of the mesosoma chord line (per BL)
        BLx = float(s[-1])
        dh = X[-1] - p1
        dh = dh - (dh @ ex) * ex
        dvec = -m - dh / max(BLx, 1e-9)
        info["dorsal_from_head_flexion"] = round(float(np.linalg.norm(dh) / max(BLx, 1e-9)), 3)
        if np.linalg.norm(dvec) > 1e-6:
            ez = dvec / np.linalg.norm(dvec)
            info["dorsal_fused_strength"] = round(float(np.linalg.norm(dvec)), 3)
    if ez is None:
        # fallback: any vector perpendicular to X
        tmp = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ez = tmp - (tmp @ ex) * ex
        ez /= np.linalg.norm(ez)
        info["dorsal_strength"] = 0.0
        info["dorsal_from"] = "arbitrary (no mesosoma attachments)"
    else:
        info["dorsal_from"] = "mesosoma appendage attachments (ventral) + head flexion (ventral)"
    # flexion cues (recorded): head front and gaster tip relative to the mesosoma chord line, perpendicular to X
    for nm, q in (("flex_head_scan", X[-1]), ("flex_gaster_scan", X[0])):
        d = q - p1
        d = d - (d @ ex) * ex
        info[nm] = d.tolist()
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    # roll: the trunk path of a dorso-ventrally flexed worker lies in the sagittal plane; its plane normal is a second
    # estimate of the left-right axis (weighted by the path's planarity), signed like the attachment estimate
    c = X.mean(0)
    w_, v_ = np.linalg.eigh(np.cov((X - c).T))
    plan = float(np.sqrt(max(w_[1], 1e-12) / max(w_[0], 1e-12)))
    yp = v_[:, 0] - (v_[:, 0] @ ex) * ex
    info["path_planarity"] = round(plan, 2)
    if np.linalg.norm(yp) > 1e-9:
        yp /= np.linalg.norm(yp)
        yp = yp if yp @ ey >= 0 else -yp
        wa = float(np.clip(info.get("dorsal_fused_strength", 0.0) / 0.3, 0.0, 1.0))
        wp = float(np.clip((plan - 3.0) / 10.0, 0.0, 1.0))
        info["roll_weights"] = {"attachments": round(wa, 3), "path_plane": round(wp, 3)}
        info["roll_attachments_vs_plane_deg"] = round(float(np.degrees(np.arccos(np.clip(yp @ ey, -1, 1)))), 1)
        if wa + wp > 0:
            ey = wa * ey + wp * yp
            ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    Rf = np.stack([ex, ey, ez])
    org = point_at(X, s, 0.5 * (b1 + nk))
    return Rf, org, info


def rot_angle(Ra, Rb):
    c = (np.trace(np.asarray(Ra) @ np.asarray(Rb).T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ============================================================================================ run
def s3_frame(name, s3_dir=None):
    if not s3_dir:
        return None
    f = os.path.join(s3_dir, name, f"{name}_joints.json")
    if not os.path.isfile(f):
        return None
    d = json.load(open(f))
    fr = d.get("frame") or {}
    if fr.get("rotation") is None:
        return None
    return {
        "rotation": np.asarray(fr["rotation"], float),
        "origin_scan": np.asarray(fr["origin_scan"], float),
        "source": f"round3 {f}",
    }


def snap_to_core(G, lab, big, Xscan):
    cand = np.argwhere(np.isin(lab, big))
    tr = cKDTree(G.lo + cand * G.pitch)
    return cand[tr.query(G.scan_to_frame(Xscan))[1]]


def route_len(G, a, b):
    p = trace(G.cost, a, b)
    if p is None:
        raise RuntimeError("no trunk route between the ends")
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * G.pitch)


def jconf(R):
    return {j: round(float(c), 3) for j, (a, rule, c) in R["joints"].items()}


def run(name, mesh_path, out_dir, frame_prior=None, write=True):
    """Trunk product for one specimen. frame_prior: optional {'rotation': 3x3 scan->canonical, 'origin_scan'} used only
    as a tie-breaker for the head end when the anatomy cues have no vote, and reported against."""
    t0 = time.time()
    tm = {}
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    V, F, minfo = load_mesh(mesh_path)
    vol = mesh_volume(V, F)
    size = vol ** (1 / 3)
    surf = Surface(V, F, size / CFG["coarse_div"] / 2.6)
    tm["load"] = round(time.time() - t0, 2)
    # ---------------- coarse pass (PCA frame)
    R0, o0 = pca_frame(V, F)
    G0 = Grid(V, F, R0, o0, size / CFG["coarse_div"])
    r0 = max(2, int(round(CFG["coarse_open_frac"] * G0.D.max())))
    _, lab0, vols0 = G0.core(r0)
    A, B, _, einfo = far_ends(G0.occ, G0.D, lab0, vols0, CFG["end_comp_share"])
    BLg = route_len(G0, A, B) / 1.1
    Ra = analyse(G0, A, B, surf, BLg, r0)
    orient_prior = None
    if Ra["decision"]["A_is_gaster"] is None and frame_prior is not None:
        hx = np.asarray(frame_prior["rotation"])[0]
        orient_prior = bool((Ra["T"]["X"][-1] - Ra["T"]["X"][0]) @ hx > 0)  # path end B ahead -> A gaster
        Ra = analyse(G0, A, B, surf, BLg, r0, orient=orient_prior)
    tp0, _ = tpart_grid(Ra, BLg)
    g0, h0 = refined_ends(G0, Ra, tp0, BLg)
    Rb = analyse(G0, g0, h0, surf, BLg, r0, orient=True)
    P0 = joints_xyz(Rb)
    BL0, _ = body_length(joints_xyz(Rb, G0))
    if BL0 is None or not np.isfinite(BL0) or BL0 <= 0:
        BL0 = float(Rb["T"]["s"][-1]) / 1.1
    Rf, of, finfo0 = anatomical_frame(G0, Rb, P0)
    tm["coarse"] = round(time.time() - t0, 2)
    coarse = {
        "pitch_um": G0.pitch,
        "open_vox": r0,
        "BL_route_guess_um": round(BLg, 1),
        "BL_um": round(BL0, 1),
        "decision": Ra["decision"],
        "ends": einfo,
    }
    # ---------------- fine pass (anatomical frame of the coarse pass)
    pitch = BL0 / CFG["nvox_bl"]
    G = Grid(V, F, Rf, of, pitch)
    rf = CFG["open_vox"]
    _, lab1, vols1 = G.core(rf)
    big1 = [i + 1 for i, v in enumerate(vols1) if v >= CFG["end_comp_share"] * vols1.sum()]
    if not big1:
        raise RuntimeError("no large core component on the fine grid")
    a1 = snap_to_core(G, lab1, big1, Rb["T"]["X"][0])
    b1 = snap_to_core(G, lab1, big1, Rb["T"]["X"][-1])
    Rc = analyse(G, a1, b1, surf, BL0, rf)
    fine_dec = Rc["decision"]
    if fine_dec["A_is_gaster"] is None:
        Rc = analyse(G, a1, b1, surf, BL0, rf, orient=True)  # keep the coarse orientation
    BL1, _ = body_length(joints_xyz(Rc, G))
    BL1 = BL1 if BL1 and np.isfinite(BL1) else BL0
    tp1, _ = tpart_grid(Rc, BL1)
    g1, h1 = refined_ends(G, Rc, tp1, BL1)
    R = analyse(G, g1, h1, surf, BL1, rf, orient=True)
    tm["fine"] = round(time.time() - t0, 2)
    P = joints_xyz(R, G)
    BL, missing_bl = body_length(P)
    tp, bounds = tpart_grid(R, BL)
    Rfin, ofin, finfo = anatomical_frame(G, R, P)
    gpts, ginfo = ({}, {"status": "no_b_a_3"})
    if P.get("b_a_3") is not None:
        Xg = G.ijk_to_scan(np.argwhere(tp == 3))
        gpts, ginfo = gaster_landmarks(Xg, P["b_a_3"], Rfin[2], G.pitch, CFG["gaster_calib"])
    P.update(gpts)
    T = R["T"]
    q = G.scan_to_ijk(T["X"])
    T_part = tp[tuple(q.T)]
    b3, b1_, nk = bounds
    T_part_arc = np.where(T["s"] >= nk, 0, np.where(T["s"] >= b1_, 1, np.where(T["s"] >= b3, 2, 3))).astype(np.int8)
    # ---------------- confidence and evidence
    dec_coarse, dec_fine = Ra["decision"], fine_dec
    head_conf = max(dec_fine["confidence"], 0.0)
    ch, cg = Rb["T"]["X"][-1], Rb["T"]["X"][0]  # coarse head / gaster ends (oriented)
    agree = bool(np.linalg.norm(P["head_front"] - ch) < np.linalg.norm(P["head_front"] - cg))
    conf = {"head_end": round(head_conf, 3)}
    # a joint's confidence = its feature clarity x the head-end confidence factor (a wrong head end moves every joint)
    hf_ = 0.6 + 0.4 * head_conf
    conf.update({j: round(c * hf_, 3) for j, c in jconf(R).items()})
    # the ends of a blunt gaster / head are ambiguous along the surface (dev: 3D error median ~2 % BL)
    conf["gaster_tip"] = round(0.5 * hf_, 3)
    conf["head_front"] = round(0.5 * hf_, 3)
    for j in ("b_a_4", "b_a_5"):
        conf[j] = round(0.4 * hf_, 3) if j in gpts else 0.0
    prior_cmp = None
    if frame_prior is not None:
        Rp = np.asarray(frame_prior["rotation"])
        prior_cmp = {
            "head_end_agrees": bool(Rfin[0] @ Rp[0] > 0),
            "x_angle_deg": round(float(np.degrees(np.arccos(np.clip(Rfin[0] @ Rp[0], -1, 1)))), 1),
            "z_angle_deg": round(float(np.degrees(np.arccos(np.clip(Rfin[2] @ Rp[2], -1, 1)))), 1),
            "frame_angle_deg": round(rot_angle(Rfin, Rp), 1),
            "source": frame_prior.get("source"),
        }
    seconds = time.time() - t0
    js = {k: (None if v is None else np.asarray(v).tolist()) for k, v in P.items()}
    trunk_joints = {
        k: js.get(k) for k in ("gaster_tip", "b_a_5", "b_a_4", "b_a_3", "b_a_2", "b_a_1", "neck", "head_front")
    }
    rules = {j: rule for j, (a, rule, c) in R["joints"].items()}
    arcs = {j: round(float(a), 2) for j, (a, rule, c) in R["joints"].items()}
    arcs.update({k: round(v, 2) for k, v in R.get("end_arcs", {}).items()})
    rules.update(R.get("end_rules", {}))
    res = {
        "name": name,
        "version": VERSION,
        "units": "micrometre",
        "status": "ok",
        "error": None,
        "conventions": TRUNK_DOC,
        "frame": {
            "rotation_scan_to_canonical": Rf.tolist(),
            "origin_scan": of.tolist(),
            "anatomical": True,
            "source": "coarse-pass anatomical estimate (+X head, +Y left, +Z dorsal); the voxel grid is axis-aligned in it",
        },
        "anatomical_frame_final": {
            "rotation_scan_to_canonical": Rfin.tolist(),
            "origin_scan": ofin.tolist(),
            **finfo,
            "angle_to_working_frame_deg": round(rot_angle(Rfin, Rf), 2),
        },
        "BL_um": BL,
        "BL_missing_joints": missing_bl,
        "pitch_um": G.pitch,
        "open_radius_vox": rf,
        "shape": list(G.shape),
        "trunk_joints_scan": trunk_joints,
        **trunk_joints,
        "b_t": trunk_joints["neck"],
        "b_h": trunk_joints["neck"],
        "waist_segments": R["jinfo"].get("waist_segments"),
        "joint_arcs_from_gaster_tip_um": arcs,
        "joint_rules": rules,
        "confidence": conf,
        "evidence": {
            "head_end": {
                "coarse": dec_coarse,
                "fine": dec_fine,
                "coarse_fine_agree": agree,
                "prior_used": orient_prior is not None,
            },
            "profile": {**{k: v for k, v in R["f"].items() if k not in ("pk", "cuts")}, **R["jinfo"]},
            "attachments": R["ainfo"],
            "gaster_landmarks": ginfo,
            "frame_prior": prior_cmp,
            "part_bounds_arc_um": {"gaster|waist": b3, "waist|mesosoma": b1_, "mesosoma|head": nk},
            "trunk_components": R["T"]["trunk_components"],
            "end_extension_points": list(map(int, R["T"]["n_end_ext"])),
        },
        "coarse": coarse,
        "mesh": minfo,
        "T_arc_um": float(T["s"][-1]),
        "timing_s": tm,
        "seconds": round(seconds, 1),
    }
    if write:
        np.savez_compressed(
            os.path.join(od, "trunk.npz"),
            occ=np.packbits(G.occ),
            shape=np.array(G.shape),
            lo=G.lo,
            pitch=G.pitch,
            tpart=tp,
            T_X=T["X"],
            T_X_frame=G.scan_to_frame(T["X"]),
            T_s=T["s"],
            T_r=T["r"],
            T_part=T_part,
            T_part_arc=T_part_arc,
            T_att=R["att"],
            R=Rf,
            origin=of,
        )
        json.dump(res, open(os.path.join(od, "trunk.json"), "w"), indent=1, default=float)
        write_joints(name, od, P, Rfin, ofin, conf, res, seconds)
    return res, (G, R, tp, P)


def write_joints(name, od, P, Rfin, ofin, conf, res, seconds):
    """Joint product contract with only the trunk joints filled (b_t = b_h = neck)."""
    js = {k: None for k in J_NAMES}
    for k in ("b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"):
        js[k] = None if P.get(k) is None else np.asarray(P[k]).tolist()
    js["b_t"] = js["b_h"] = None if P.get("neck") is None else np.asarray(P["neck"]).tolist()
    js["head_front"] = np.asarray(P["head_front"]).tolist()
    js["gaster_tip"] = np.asarray(P["gaster_tip"]).tolist()
    jc = {k: (None if v is None else (Rfin @ (np.asarray(v) - ofin)).tolist()) for k, v in js.items()}
    rules = res.get("joint_rules", {})
    prov = {}
    for k in js:
        if js[k] is None:
            prov[k] = None
            continue
        kk = "neck" if k in ("b_t", "b_h") else k
        rule = rules.get(kk, "gaster_centre_of_volume_convention" if k in ("b_a_4", "b_a_5") else "")
        prov[k] = {"kind": "proportion" if rule.startswith("proportion") else "scan feature", "rule": rule}
    c = {k: 0.0 for k in js}
    for k in js:
        kk = "neck" if k in ("b_t", "b_h") else k
        if kk in conf:
            c[k] = conf[kk]
    out = {
        "name": name,
        "units": "micrometre",
        "method": VERSION + " (trunk only)",
        "frame": {"rotation": Rfin.tolist(), "origin_scan": ofin.tolist()},
        "joints_scan": js,
        "joints_canonical": jc,
        "confidence": c,
        "provenance": prov,
        "body_length_um": res["BL_um"],
        "completeness": {
            "missing_joints": [k for k, v in js.items() if v is None],
            "complete": False,
            "flags": ["trunk_only"],
            "parts": {},
        },
        "status": "ok",
        "error": None,
        "seconds": round(seconds, 1),
    }
    json.dump(out, open(os.path.join(od, f"{name}_joints.json"), "w"), indent=1, default=float)


def run_safe(args):
    name, mesh, out_dir, use_prior, s3_dir, timeout = args
    import signal

    def _alarm(*_):
        raise TimeoutError(f"timeout after {timeout} s")

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout))
    t0 = time.time()
    try:
        fp = s3_frame(name, s3_dir) if use_prior else None
        res, _ = run(name, mesh, out_dir, frame_prior=fp)
        signal.alarm(0)
        return name, "ok", round(time.time() - t0, 1), None
    except Exception as e:
        signal.alarm(0)
        od = os.path.join(out_dir, name)
        os.makedirs(od, exist_ok=True)
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        fail = {
            "name": name,
            "version": VERSION,
            "status": "failed",
            "error": err,
            "traceback": tb,
            "seconds": round(time.time() - t0, 1),
        }
        json.dump(fail, open(os.path.join(od, "trunk.json"), "w"), indent=1)
        json.dump(
            {"name": name, "status": "failed", "error": err, "seconds": fail["seconds"], "joints_scan": None},
            open(os.path.join(od, f"{name}_joints.json"), "w"),
            indent=1,
        )
        return name, "failed", fail["seconds"], err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="")
    ap.add_argument("--names-file", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mesh-root", required=True, help="<root>/<name>/<name>_envelope.stl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-prior", action="store_true", help="do not read the round-3 frame (tie-breaker / report only)")
    ap.add_argument("--s3-dir", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    names = [x for x in a.names.split(",") if x]
    if a.names_file:
        names += [x.strip() for x in open(a.names_file) if x.strip() and not x.startswith("#")]
    jobs = []
    for n in names:
        if a.skip_existing and os.path.isfile(os.path.join(a.out_dir, n, "trunk.json")):
            continue
        jobs.append(
            (n, os.path.join(a.mesh_root, n, f"{n}_envelope.stl"), a.out_dir, not a.no_prior, a.s3_dir, a.timeout)
        )
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    if a.workers <= 1:
        res = [run_safe(j) for j in jobs]
        for r in res:
            print(*r, flush=True)
    else:
        import multiprocessing as mp

        res = []
        with mp.get_context("fork").Pool(min(a.workers, 4), maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run_safe, jobs):
                print(*r, flush=True)
                res.append(r)
    nf = sum(1 for r in res if r[1] != "ok")
    print(f"done {len(res)} specimens, {nf} failed, wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
