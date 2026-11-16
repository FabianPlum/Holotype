"""OmniAnt25 template access for SKEL-REG (numpy only; no import of the rest of the repository).

Sources: field layout of the pkl and the LBS convention were read from
smal_model/smal_torch.py (SMAL.__call__) and custom_processing/omniant_synthetic.py (load_model: weight rows
renormalised).
"""

import functools
import pickle

import numpy as np

from fitter_3d.antscan_registration.parts import partconf

JOINT_KIND = (
    "medial"  # what the input joints are when template() is called without a kind (the joint method's products)
)

TRUNK = ["b_t", "b_h", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]
WINGS = ["w_1_r", "w_2_r", "w_1_l", "w_2_l"]

# evaluation / reporting groups (by dominant bone)
GROUPS = {
    "head": ["b_h"],
    "mandible": ["ma_r", "ma_l"],
    "scape": ["an_1_r", "an_1_l"],
    "funiculus": ["an_2_r", "an_2_l", "an_3_r", "an_3_l"],
    "mesosoma": ["b_t"],
    "waist": ["b_a_1", "b_a_2"],
    "gaster": ["b_a_3", "b_a_4", "b_a_5"],
    "coxa": [f"l_{k}_co_{s}" for k in "123" for s in "rl"],
    "femur": [f"l_{k}_tr_{s}" for k in "123" for s in "rl"],
    "tibia": [f"l_{k}_fe_{s}" for k in "123" for s in "rl"],
    "basitarsus": [f"l_{k}_ti_{s}" for k in "123" for s in "rl"],
    "tarsus": [f"l_{k}_ta_{s}" for k in "123" for s in "rl"] + [f"l_{k}_pt_{s}" for k in "123" for s in "rl"],
}
TUBE_GROUPS = ["scape", "funiculus", "coxa", "femur", "tibia", "basitarsus", "tarsus"]


class Template:
    def __init__(self, path=None):
        path = path or partconf.model()
        dd = pickle.load(open(path, "rb"), encoding="latin1")
        self.path = path
        self.names = list(dd["J_names"])
        self.idx = {n: i for i, n in enumerate(self.names)}
        par = np.asarray(dd["kintree_table"])[0].astype(np.int64).copy()
        par[0] = -1
        self.parents = par
        self.children = [np.where(par == i)[0].tolist() for i in range(len(par))]
        self.V0 = np.asarray(dd["v_template"], dtype=np.float64)
        self.F = np.asarray(dd["f"], dtype=np.int64)
        W = np.asarray(dd["weights"], dtype=np.float64)
        self.W = W / W.sum(1, keepdims=True)
        self.Jreg = np.asarray(dd["J_regressor"], dtype=np.float64)  # (55, V)
        self.J0 = self.Jreg @ self.V0
        self.shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)  # (V, 3, 25)
        self.shape_cov = np.asarray(dd["shape_cov"], dtype=np.float64)
        self.limits = np.asarray(dd["joint_limits"], dtype=np.float64)
        self.dom = self.W.argmax(1)
        self.nJ = len(self.names)
        self.nV = len(self.V0)
        self.group_of_bone = {}
        for g, bones in GROUPS.items():
            for b in bones:
                self.group_of_bone[self.idx[b]] = g
        self.group_names = list(GROUPS)
        self.vgroup = np.array([self.group_names.index(self.group_of_bone[b]) for b in self.dom])
        # rest body length (model units), the generator's definition: trunk extent along b_a_4 -> b_h
        body = np.isin(self.dom, [self.idx[k] for k in TRUNK])
        u = self.J0[self.idx["b_h"]] - self.J0[self.idx["b_a_4"]]
        u /= np.linalg.norm(u)
        pr = self.V0[body] @ u
        self.rest_body_length = float(pr.max() - pr.min())

    # ---- virtual markers: the articulation as a feature of the SURFACE (comparable with medial-axis joints) --------
    def marker_regressor(self):
        """Mreg (nJ, nV), rows sum to 1: `Mreg @ V` gives, for any posed / registered vertex set V, the centroid of the
        vertex ring at each articulation (the vertices whose summed skinning weight of the joint's subtree is between
        0.25 and 0.75), i.e. a point on the medial axis at the articulation. Special rows: b_t = the neck ring (as b_h);
        leaf joints without a ring of their own are tips: l_*_pt (leg tip), an_3 (antenna tip), b_a_5 (gaster apex,
        the five extreme gaster vertices along b_a_3 -> b_a_5); b_a_4 lies on the line b_a_3 ring -> gaster apex at the
        rest fraction of its own ring; wings keep the joint regressor."""
        if getattr(self, "_Mreg", None) is not None:
            return self._Mreg
        _nJ, nV = self.nJ, self.nV
        tri = self.V0[self.F]
        fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        va = np.zeros(nV)
        for k in range(3):
            np.add.at(va, self.F[:, k], fa / 3)
        M = self.Jreg.copy()

        def ring(j, lo=0.25, hi=0.75):
            d = self.W[:, self.descendants(j)].sum(1)
            for lo_, hi_ in ((lo, hi), (0.1, 0.9), (0.02, 0.98)):
                m = (d > lo_) & (d < hi_)
                if m.sum() >= 3:
                    w = va * m * (1 - np.abs(2 * d - 1) + 0.05)
                    return w / w.sum()
            return None

        for j, n in enumerate(self.names):
            if n in WINGS or j == 0:
                continue
            leaf = not self.children[j]
            if leaf and (n.startswith("an_3") or "_pt_" in n):
                u = self.J0[j] - self.J0[self.parents[j]]
                ids = np.where(np.isin(self.dom, [j, self.parents[j]]))[0]
                row = np.zeros(nV)
                row[ids[np.argmax(self.V0[ids] @ u)]] = 1.0
                M[j] = row
                continue
            if n in ("b_a_4", "b_a_5"):
                continue
            r = ring(j)
            if r is not None:
                M[j] = r
        i3, i4, i5 = self.idx["b_a_3"], self.idx["b_a_4"], self.idx["b_a_5"]
        u = self.J0[i5] - self.J0[i3]
        ids = np.where(np.isin(self.dom, [i3, i4, i5]))[0]
        top = ids[np.argsort(self.V0[ids] @ u)[-5:]]
        row = np.zeros(nV)
        row[top] = 0.2
        M[i5] = row
        a, b = M[i3] @ self.V0, M[i5] @ self.V0
        r4 = ring(i4)
        c4 = (r4 @ self.V0) if r4 is not None else self.J0[i4]
        f = float(np.clip(((c4 - a) @ (b - a)) / ((b - a) @ (b - a)), 0.2, 0.8))
        M[i4] = (1 - f) * M[i3] + f * M[i5]
        self.gaster_fraction = f
        M[0] = M[self.idx["b_h"]]
        self._Mreg = M
        return M

    def descendants(self, j):
        out, stack = [], [j]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(self.children[k])
        return out


@functools.lru_cache(maxsize=4)
def _template(kind, path):
    t = Template(path)
    t.kind = kind
    t.J0_pivot = t.J0.copy()
    if kind == "medial":
        # joints given on the scan's medial axis (annotated truth, path-based localisers): the template's reference
        # points are its virtual markers, so that like is compared with like
        t.J0 = t.marker_regressor() @ t.V0
    return t


def template(kind=None):
    """The template of the model file set for the pipeline. `kind` (default JOINT_KIND) says what the input joints are:
    "pivot" = model pivots (bone heads; oracle joints, round-3 proposals), "medial" = articulation points on the
    scan's medial axis (annotated truth, round-6 proposals)."""
    return _template(kind or JOINT_KIND, partconf.model())


def rodrigues(theta):
    """Axis-angle (N,3) -> rotation matrices (N,3,3)."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1, 3)
    ang = np.linalg.norm(theta, axis=1)
    out = np.tile(np.eye(3), (len(theta), 1, 1))
    ok = ang > 1e-12
    k = theta[ok] / ang[ok, None]
    K = np.zeros((ok.sum(), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(ang[ok])[:, None, None], np.cos(ang[ok])[:, None, None]
    out[ok] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return out


def fk(J, parents, R):
    """Global joint transforms. J (N,3) rest joints, R (N,3,3) local rotations. Returns GR (N,3,3), X (N,3)."""
    n = len(J)
    GR = np.zeros((n, 3, 3))
    X = np.zeros((n, 3))
    for j in range(n):
        p = parents[j]
        if p < 0:
            GR[j], X[j] = R[j], J[j]
        else:
            GR[j] = GR[p] @ R[j]
            X[j] = GR[p] @ (J[j] - J[p]) + X[p]
    return GR, X


def lbs(v, J, parents, R, W):
    """Linear blend skinning. Returns posed vertices and pivots (J_transformed)."""
    GR, X = fk(J, parents, R)
    T = X - np.einsum("jab,jb->ja", GR, J)  # x -> GR x + T
    WR = np.einsum("vj,jab->vab", W, GR)
    WT = W @ T
    return np.einsum("vab,vb->va", WR, v) + WT, X


def forward(tpl, betas, theta):
    """The generator's forward: shape blend, regressed rest joints, LBS. Returns verts, pivots, v_shaped, J."""
    v = tpl.V0 + tpl.shapedirs @ np.asarray(betas, dtype=np.float64)
    J = tpl.Jreg @ v
    R = rodrigues(theta)
    verts, piv = lbs(v, J, tpl.parents, R, tpl.W)
    return verts, piv, v, J
