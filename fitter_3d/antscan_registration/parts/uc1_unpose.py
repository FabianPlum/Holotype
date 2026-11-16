"""ONE unposing convention for AntScan registrations (any route), round 2. numpy only; no route parameters needed.

Input per specimen: the 10,235 registered template vertices V (any frame, micrometres) in template order, and
optionally a per-vertex `valid` mask (route's registered status). Output: the specimen's REST mesh in OmniAnt model
units, with pose, global rotation, translation and size removed and every per-bone proportion, girth and part shape
kept. Definition ("UC1", fixed by the template alone):

  pivots     P = J_regressor @ V   (the model's own 10-vertex regressor; on posed template meshes it reproduces the
             posed pivot to 0.009 % extent median, 0.10 % max)
  rotations  per bone j, R_j = rotation of bone j's vertices relative to the TEMPLATE's (weights w_ij^2, valid
             vertices). Chain bones (exactly one non-wing child c): the minimal rotation taking the template bone
             axis J0_c - J0_j onto P_c - P_j, then the twist about that axis that best aligns the bone's vertices
             (closed form). Bones with several children (b_t = root, b_h) and leaves (pretarsi, an_3, b_a_5,
             mandibles): weighted Kabsch. Wings (no vertices): the parent's rotation.
  rest skel. Jr_root = 0, Jr_c = Jr_p + R_p^T (P_c - P_p): the template's rest bone directions for every chain bone,
             the specimen's own bone lengths, the specimen's own (rigid) placement of the coxae / head appendages.
  unpose     posed bone transform x -> R_j (x - Jr_j) + P_j, blended with the template skinning weights (rows
             renormalised): V_i = M_i x_i + t_i, M_i = sum_j w_ij R_j, t_i = sum_j w_ij (P_j - R_j Jr_j), so
             x_i = M_i^-1 (V_i - t_i). Where cond(M_i) > 8 (a joint folded by more than ~150 deg) the dominant bone's
             rigid inverse is used (flagged `fallback`; these vertices do not round-trip exactly).
  size       s = centroid size of the rest TRUNK vertices (dominant bone b_t, b_h, b_a_1..5) / the template's
             (micrometres per model unit). Rest mesh in model units: x / s + J0_root (template root pivot).
Per-bone scales are KEPT in the rest mesh (a decision of the shape-model work: the add-on reads them as shape).

The pipeline's export step (fitter_3d/antscan_registration/fill.py, parts/rs/export.py) writes the add-on bundle
from `unpose`; its keys (extra keys are ignored by the add-on):
   verts (N,10235,3) float32 model units (rest, UC1) | labels (N,) | valid (N,10235) bool | fallback (N,10235) bool
   um_per_model_unit (N,) | J_rest (N,55,3) model units | bone_len_um (N,55) pivot chord bone j -> its chain child
   (NaN for non-chain bones) | R (N,55,3,3) posed-from-rest bone rotations | roundtrip_max_um (N,) | convention
   <out>.json: per-specimen QA + the convention text.
"""

import pickle

import numpy as np

from fitter_3d.antscan_registration.parts import partconf

TRUNK = ("b_t", "b_h", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5")
WINGS = ("w_1_r", "w_2_r", "w_1_l", "w_2_l")
COND_MAX = 8.0
CONVENTION = "UC1 (round 2): J_regressor pivots; vertex-based bone rotations (axis-constrained chain bones, Kabsch otherwise); template rest directions + own lengths; blended inverse LBS; trunk centroid size; root at template root pivot"


def _min_rotation(a, b):
    """Rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-10:  # antiparallel: 180 deg about any axis perpendicular to a
        p = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _axis_rot(u, psi):
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(psi) * K + (1 - np.cos(psi)) * K @ K


def _kabsch(X, Y, w):
    """R minimising sum w |R (y - cy) - (x - cx)|^2."""
    w = w / w.sum()
    cx, cy = w @ X, w @ Y
    H = ((Y - cy) * w[:, None]).T @ (X - cx)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


class Template:
    def __init__(self, path=None):
        path = path or partconf.model()
        dd = pickle.load(open(path, "rb"), encoding="latin1")
        self.path = path
        self.names = [str(n) for n in dd["J_names"]]
        self.idx = {n: i for i, n in enumerate(self.names)}
        par = np.asarray(dd["kintree_table"])[0].astype(np.int64).copy()
        par[0] = -1
        self.parents = par
        self.nJ = len(par)
        self.children = [[int(c) for c in np.where(par == j)[0]] for j in range(self.nJ)]
        self.V0 = np.asarray(dd["v_template"], np.float64)
        self.F = np.asarray(dd["f"], np.int64)
        W = np.asarray(dd["weights"], np.float64)
        self.W = W / W.sum(1, keepdims=True)  # left-mandible rows sum to 0.40-1 in the pkl
        self.Jreg = np.asarray(dd["J_regressor"], np.float64)
        self.J0 = self.Jreg @ self.V0
        self.dom = self.W.argmax(1)
        self.nV = len(self.V0)
        self.trunk_v = np.isin(self.dom, [self.idx[n] for n in TRUNK])
        c0 = self.V0[self.trunk_v].mean(0)
        self.trunk_cs0 = float(np.sqrt(((self.V0[self.trunk_v] - c0) ** 2).sum(1).mean()))
        self.chain_kid = np.full(self.nJ, -1)
        for j in range(self.nJ):
            kids = [c for c in self.children[j] if self.names[c] not in WINGS]
            if len(kids) == 1:
                self.chain_kid[j] = kids[0]
        self.bone_v = [np.where(self.W[:, j] > 0)[0] for j in range(self.nJ)]
        # mirror map (y -> -y): exact on OmniAnt
        from scipy.spatial import cKDTree

        self.mirror_v = cKDTree(self.V0).query(self.V0 * np.array([1, -1, 1]))[1]
        self.mirror_j = np.array(
            [
                self.idx.get(n[:-2] + ("_l" if n.endswith("_r") else "_r"), j) if n.endswith(("_r", "_l")) else j
                for j, n in enumerate(self.names)
            ]
        )
        # rest body length of the template (skelreg definition: trunk extent along b_a_4 -> b_h), for % BL figures
        u = self.J0[self.idx["b_h"]] - self.J0[self.idx["b_a_4"]]
        u /= np.linalg.norm(u)
        pr = self.V0[self.trunk_v] @ u
        self.rest_bl = float(pr.max() - pr.min())


_T = {}


def template(path=None):
    path = path or partconf.model()
    if path not in _T:
        _T[path] = Template(path)
    return _T[path]


def bone_rotations(t, V, P, valid, mode="uc1"):
    """Posed-from-template rotation of every bone. mode "uc1" (the convention); experimental variants: "kabsch" (weighted
    Kabsch for every bone, no axis constraint), "minrot" (chain bones: minimal rotation of the PARENT-carried axis, no
    twist estimate from the vertices, as a joints-only unposer would do)."""
    R = np.tile(np.eye(3), (t.nJ, 1, 1))
    for j in range(t.nJ):  # parents precede children in OmniAnt's order (checked in unpose())
        ids = t.bone_v[j]
        if len(ids) == 0:
            R[j] = R[t.parents[j]] if j > 0 else np.eye(3)
            continue
        ok = valid[ids]
        if ok.sum() >= 3:
            ids = ids[ok]
        w = t.W[ids, j] ** 2
        c = t.chain_kid[j]
        chain = c >= 0 and np.linalg.norm(P[c] - P[j]) > 1e-9 * max(1.0, np.abs(P).max())
        if chain and mode == "minrot" and j > 0:
            Rp = R[t.parents[j]]
            R[j] = _min_rotation(Rp @ (t.J0[c] - t.J0[j]), P[c] - P[j]) @ Rp
        elif chain and mode == "uc1":
            a0 = t.J0[c] - t.J0[j]
            a = P[c] - P[j]
            u = a / np.linalg.norm(a)
            Rm = _min_rotation(a0, a)
            x = V[ids] - P[j]
            y = (t.V0[ids] - t.J0[j]) @ Rm.T
            x = x - np.outer(x @ u, u)
            y = y - np.outer(y @ u, u)
            psi = np.arctan2((w * (np.cross(y, x) @ u)).sum(), (w * (y * x).sum(1)).sum())
            R[j] = _axis_rot(u, psi) @ Rm
        else:
            R[j] = _kabsch(V[ids], t.V0[ids], w)
    return R


def rest_skeleton(t, P, R):
    Jr = np.zeros((t.nJ, 3))
    for j in range(1, t.nJ):
        p = t.parents[j]
        Jr[j] = Jr[p] + R[p].T @ (P[j] - P[p])
    return Jr


def blend_inverse(t, V, R, P, Jr):
    M = np.einsum("vj,jab->vab", t.W, R)
    T = P - np.einsum("jab,jb->ja", R, Jr)
    tv = t.W @ T
    cond = np.linalg.cond(M)
    bad = ~(cond <= COND_MAX)
    Mi = np.linalg.inv(np.where(bad[:, None, None], np.eye(3), M))
    X = np.einsum("vab,vb->va", Mi, V - tv)
    if bad.any():
        d = t.dom[bad]
        X[bad] = np.einsum("vba,vb->va", R[d], V[bad] - P[d]) + Jr[d]
    return X, bad, M, tv


def unpose(V, valid=None, t=None, mode="uc1", P=None, iters=0):
    """V (10235,3) micrometres, any frame. Returns dict (see module docstring). `mode` / `P` only for experiments."""
    t = t or template()
    assert all(t.parents[j] < j for j in range(1, t.nJ)), "parents must precede children"
    V = np.asarray(V, np.float64)
    assert V.shape == t.V0.shape and np.isfinite(V).all()
    valid = np.ones(t.nV, bool) if valid is None else np.asarray(valid, bool)
    P = t.Jreg @ V if P is None else np.asarray(P, np.float64)
    for it in range(1 + iters):
        R = bone_rotations(t, V, P, valid, mode)
        Jr = rest_skeleton(t, P, R)
        X, bad, M, tv = blend_inverse(t, V, R, P, Jr)
        if it < iters:  # pivots := J_regressor of the REST mesh, carried back to the posed mesh by the parent bone
            Jx = t.Jreg @ X
            Pn = P.copy()
            Pn[0] = R[0] @ (Jx[0] - Jr[0]) + P[0]
            for j in range(1, t.nJ):
                p = t.parents[j]
                Pn[j] = R[p] @ (Jx[j] - Jr[p]) + P[p]
            P = Pn
    Vb = np.einsum("vab,vb->va", M, X) + tv
    err = np.linalg.norm(Vb - V, axis=1)
    tr = t.trunk_v & valid if (t.trunk_v & valid).sum() > 100 else t.trunk_v
    c, c0 = X[tr].mean(0), t.V0[tr].mean(0)  # same vertex set on both sides
    s = float(np.sqrt(((X[tr] - c) ** 2).sum(1).mean()) / np.sqrt(((t.V0[tr] - c0) ** 2).sum(1).mean()))
    root = t.J0[0]
    blen = np.full(t.nJ, np.nan)
    for j in range(t.nJ):
        if t.chain_kid[j] >= 0:
            blen[j] = np.linalg.norm(P[t.chain_kid[j]] - P[j])
    return {
        "V_rest": X / s + root,
        "J_rest": Jr / s + root,
        "s": s,
        "R": R,
        "P": P,
        "valid": valid,
        "fallback": bad,
        "roundtrip_max_um": float(err[~bad].max()) if (~bad).any() else float("nan"),
        "roundtrip_fallback_max_um": float(err[bad].max()) if bad.any() else 0.0,
        "bone_len_um": blen,
        "V_rest_um": X,
    }


# ---------------------------------------------------------------------------------------------------------------
