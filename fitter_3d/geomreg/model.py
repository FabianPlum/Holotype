"""OmniAnt template structure for the geometric registration core (numpy; no dependency on SKEL-REG at run time).
Extended from the template class of the earlier surface registration: mirror maps,
rest markers, mandible tip rows, mandible limbs.


Facts read from the model file: 10235 vertices, 20466 faces, closed genus-0
manifold, one connected component, 55 joints (J_names), kintree_table[0] = parents (root b_t = -1), LBS weights with at
most 4 non-zero entries per vertex; 154 rows (left mandible; 134 of them by more than 1e-3, minimum 0.40) do not sum
to 1 in the pkl and are renormalised here (`W`) and in every harness that produced the baseline fits; `W_raw` keeps the
stored rows, which fitter_3d's own forward uses. Wings (w_*) own no vertices.
"""

import functools
import pickle

import numpy as np

_DEFAULT_PKL = [None]


def set_default_model(path):
    """model .pkl used by template() when no path is given (inside fitter_3d: config.SMAL_FILE)"""
    _DEFAULT_PKL[0] = path
    _template.cache_clear()


WINGS = ("w_1_r", "w_2_r", "w_1_l", "w_2_l")
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
TUBE_GROUPS = ("scape", "funiculus", "femur", "tibia", "basitarsus", "tarsus")
# limb ids of the limb product -> joint chain, DISTAL first (the limb polylines are stored tip first)
LIMB_CHAINS = {
    **{
        f"{S}{k}": [f"l_{k}_{j}_{s}" for j in ("pt", "ta", "ti", "fe", "tr", "co")]
        for k in "123"
        for S, s in (("L", "l"), ("R", "r"))
    },
    "AL": ["an_3_l", "an_2_l", "an_1_l"],
    "AR": ["an_3_r", "an_2_r", "an_1_r"],
    # the gaster as a tube along the scan's TRUNK path (base -> apex): gasters curl, a straight bone cannot
    "G": ["b_a_5", "b_a_4", "b_a_3"],
}


class Template:
    def __init__(self, path):
        dd = pickle.load(open(path, "rb"), encoding="latin1")
        self.names = [str(n) for n in dd["J_names"]]
        self.idx = {n: i for i, n in enumerate(self.names)}
        par = np.asarray(dd["kintree_table"])[0].astype(np.int64).copy()
        par[0] = -1
        self.parents = par
        self.nJ = len(par)
        self.children = [np.where(par == i)[0].tolist() for i in range(self.nJ)]
        self.V0 = np.asarray(dd["v_template"], np.float64)
        self.F = np.asarray(dd["f"], np.int64)
        self.nV, self.nF = len(self.V0), len(self.F)
        W = np.asarray(dd["weights"], np.float64)
        self.W_raw = W  # as stored: fitter_3d's forward (SMAL3DFitter) blends with these rows unchanged
        self.W = W / W.sum(1, keepdims=True)
        self.Jreg = np.asarray(dd["J_regressor"], np.float64)
        self.J0 = self.Jreg @ self.V0
        self.shapedirs = np.asarray(dd["shapedirs"], np.float64)  # (nV, 3, nB)
        self.nB = self.shapedirs.shape[2]
        self.shape_cov = np.asarray(dd["shape_cov"], np.float64) if "shape_cov" in dd else np.eye(self.nB)
        self.limits = np.asarray(dd["joint_limits"], np.float64) if "joint_limits" in dd else None
        self.dom = self.W.argmax(1)
        self.group_names = list(GROUPS)
        gob = {self.idx[b]: gi for gi, g in enumerate(GROUPS) for b in GROUPS[g]}
        self.bone_group = np.array([gob.get(j, -1) for j in range(self.nJ)])
        self.vgroup = self.bone_group[self.dom]
        self.tube_bone = np.array([self.group_names[g] in TUBE_GROUPS if g >= 0 else False for g in self.bone_group])
        # ---- topology
        F = self.F
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        es = np.sort(e, 1)
        self.E, inv = np.unique(es, axis=0, return_inverse=True)  # (nE, 2) unique edges
        self.nE = len(self.E)
        fidx = np.tile(np.arange(self.nF), 3)
        opp = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])  # vertex opposite to each half edge
        order = np.argsort(inv, kind="stable")
        inv_s, f_s, o_s = inv[order], fidx[order], opp[order]
        assert (np.bincount(inv_s) == 2).all(), "template is not a closed manifold"
        self.EF = f_s.reshape(-1, 2)  # the two faces of each edge
        self.EO = o_s.reshape(-1, 2)  # the two opposite vertices of each edge
        self.face_bone = self.dom[F[:, 0]]
        # ---- bone adjacency: same bone, parent / child, or siblings whose skins share an edge
        adj = np.eye(self.nJ, dtype=bool)
        for j in range(self.nJ):
            if par[j] >= 0:
                adj[j, par[j]] = adj[par[j], j] = True
        a, b = self.dom[self.E[:, 0]], self.dom[self.E[:, 1]]
        adj[a, b] = adj[b, a] = True
        # leaf tips belong with their parent segment
        self.adj = adj
        self.adj2 = (adj.astype(int) @ adj.astype(int)) > 0  # neighbours of neighbours
        # ---- per-vertex rest scale: mean incident edge length (model units)
        el = np.linalg.norm(self.V0[self.E[:, 0]] - self.V0[self.E[:, 1]], axis=1)
        s, c = np.zeros(self.nV), np.zeros(self.nV)
        for k in range(2):
            np.add.at(s, self.E[:, k], el)
            np.add.at(c, self.E[:, k], 1.0)
        self.vscale0 = s / c
        self.deg = c
        # ---- mirror maps: the template is exactly symmetric under y -> -y (nearest-vertex distance 0.0,
        # bijective); bones pair by their name suffix _l / _r
        from scipy.spatial import cKDTree

        Vm = self.V0 * np.array([1.0, -1.0, 1.0])
        dm, im = cKDTree(self.V0).query(Vm)
        self.mirror_exact = bool(dm.max() < 1e-7 and len(np.unique(im)) == self.nV)
        self.mirror_vert = im
        mb = np.arange(self.nJ)
        for j, n in enumerate(self.names):
            if n.endswith("_l") or n.endswith("_r"):
                o = n[:-1] + ("r" if n.endswith("_l") else "l")
                if o in self.idx:
                    mb[j] = self.idx[o]
        self.mirror_bone = mb

    def descendants(self, j):
        out, st = [], [j]
        while st:
            k = st.pop()
            out.append(k)
            st += self.children[k]
        return out

    def marker_regressor(self):
        """(55, V) virtual-marker regressor and the gaster fraction: port of SKEL-REG's Template.marker_regressor /
        the proposal's fitter_3d.joint_init.marker_regressor (checked equal)."""
        if getattr(self, "_M", None) is not None:
            return self._M
        V0, F, W = self.V0, self.F, self.W
        tri = V0[F]
        fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        va = np.zeros(self.nV)
        for k in range(3):
            np.add.at(va, F[:, k], fa / 3)
        self.varea0 = va

        def ring(j):
            d = W[:, self.descendants(j)].sum(1)
            for lo, hi in ((0.25, 0.75), (0.1, 0.9), (0.02, 0.98)):
                m = (d > lo) & (d < hi)
                if m.sum() >= 3:
                    r = va * m * (1 - np.abs(2 * d - 1) + 0.05)
                    return r / r.sum()
            return None

        M = self.Jreg.copy()
        for j, n in enumerate(self.names):
            if n in WINGS or j == 0:
                continue
            if not self.children[j] and (n.startswith("an_3") or "_pt_" in n):
                u = self.J0[j] - self.J0[self.parents[j]]
                ids = np.where(np.isin(self.dom, [j, self.parents[j]]))[0]
                M[j] = 0.0
                M[j, ids[np.argmax(V0[ids] @ u)]] = 1.0
                continue
            if n in ("b_a_4", "b_a_5"):
                continue
            r = ring(j)
            if r is not None:
                M[j] = r
        i3, i4, i5 = self.idx["b_a_3"], self.idx["b_a_4"], self.idx["b_a_5"]
        u = self.J0[i5] - self.J0[i3]
        ids = np.where(np.isin(self.dom, [i3, i4, i5]))[0]
        M[i5] = 0.0
        M[i5, ids[np.argsort(V0[ids] @ u)[-5:]]] = 0.2
        a, b = M[i3] @ V0, M[i5] @ V0
        r4 = ring(i4)
        c4 = (r4 @ V0) if r4 is not None else self.J0[i4]
        f = float(np.clip(((c4 - a) @ (b - a)) / ((b - a) @ (b - a)), 0.2, 0.8))
        M[i4] = (1 - f) * M[i3] + f * M[i5]
        M[0] = M[self.idx["b_h"]]
        self.gaster_fraction = f
        self._M = M
        return M

    def tip_rows(self):
        """{bone index: (nV,) one-hot row} for the one-joint parts that have a limb centreline in the joint method's limb
        product: the mandibles (limbs ML / MR). The tip is the vertex of the part (dominant bone) farthest from the
        part's base marker in the rest template; `tip_rows()[j] @ V` is the tip of the registered part."""
        if getattr(self, "_tips", None) is not None:
            return self._tips
        M = self.marker_regressor()
        out = {}
        for n in ("ma_l", "ma_r"):
            j = self.idx[n]
            ids = np.where(self.dom == j)[0]
            base = M[j] @ self.V0
            row = np.zeros(self.nV)
            row[ids[np.argmax(np.linalg.norm(self.V0[ids] - base, axis=1))]] = 1.0
            out[j] = row
        self._tips = out
        return out

    def head_front_row(self):
        """fixed vertex weights of the comparison's 'head front' marker (the same rows as the registration benchmark's)."""
        M = self.marker_regressor()
        ih = self.idx["b_h"]
        ids = np.where(self.dom == ih)[0]
        neck = M[ih] @ self.V0
        c = self.V0[ids].mean(0)
        u = (c - neck) / np.linalg.norm(c - neck)
        top = ids[np.argsort((self.V0[ids] - neck) @ u)[-5:]]
        row = np.zeros(self.nV)
        row[top] = 0.2
        return row

    # ------------------------------------------------------------------ limb stations (rings along the limb chains)
    def limb_stations(self, per_bone=4, min_ring=4):
        """For every limb chain of LIMB_CHAINS (stored proximal -> distal here): a sequence of 'stations' = vertex rings
        ordered along the limb. The joints' marker rings are stations; between two consecutive joints the vertices of
        the bone are cut into up to `per_bone` slabs along the rest marker-marker axis. Returns
        {limb: dict(joints=[joint index, proximal first], stations=[dict(w (nV,) ring weights, joint (index or -1),
        seg (index of the segment's proximal joint in `joints`), t (0..1 along the segment))])}."""
        if getattr(self, "_stations", None) is not None and self._stations[0] == (per_bone, min_ring):
            return self._stations[1]
        M = self.marker_regressor()
        mk = M @ self.V0
        out = {}
        for limb, chain in LIMB_CHAINS.items():
            js = [self.idx[n] for n in chain][::-1]  # proximal first
            st = []
            for a in range(len(js)):
                st.append(
                    dict(w=M[js[a]].copy(), joint=js[a], seg=min(a, len(js) - 2), t=0.0 if a < len(js) - 1 else 1.0)
                )
                if a == len(js) - 1:
                    break
                ja, jb = js[a], js[a + 1]
                A, B = mk[ja], mk[jb]
                u = B - A
                L = np.linalg.norm(u)
                u = u / L
                ids = np.where(self.dom == ja)[0]
                t = ((self.V0[ids] - A) @ u) / L
                sel = ids[(t > 0.08) & (t < 0.92)]
                ts = t[(t > 0.08) & (t < 0.92)]
                k = int(min(per_bone, len(sel) // max(min_ring, 1)))
                if k < 1:
                    continue
                edges = np.quantile(ts, np.linspace(0, 1, k + 1))
                for q in range(k):
                    m = (ts >= edges[q]) & (ts <= edges[q + 1])
                    if m.sum() < min_ring:
                        continue
                    w = np.zeros(self.nV)
                    w[sel[m]] = self.varea0[sel[m]]
                    w /= w.sum()
                    st.append(dict(w=w, joint=-1, seg=a, t=float(((w @ self.V0) - A) @ u / L)))
            # order: by segment, then t
            st.sort(key=lambda s: (s["seg"], s["t"]) if s["joint"] < 0 or s["t"] == 0.0 else (s["seg"], 2.0))
            out[limb] = dict(joints=js, stations=st)
        self._stations = ((per_bone, min_ring), out)
        return out


@functools.lru_cache(maxsize=2)
def _template(path):
    t = Template(path)
    t.marker_regressor()
    return t


def template(path=None):
    if path is None:
        path = _DEFAULT_PKL[0]
    if path is None:
        raise RuntimeError("geomreg.model.template(): no model path; call set_default_model(path) first")
    t = _template(path)
    if getattr(t, "mk0", None) is None:
        t.mk0 = t.marker_regressor() @ t.V0  # rest virtual markers (model units)
        t.path = path
    return t
