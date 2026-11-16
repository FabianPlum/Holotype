"""Population prior on limb segment lengths, estimated WITHOUT truth, taxonomy or the specimen it is applied to.

Rows: one per specimen, log chord lengths between DIRECTLY SUPPORTED joints of a joint product (confidence >= 0.3,
not declared missing): 15 leg segment types (leg pair x segment, the two sides averaged), scape, funiculus, and three
trunk chords (mesosoma neck -> b_a_1, head neck -> head_front, gaster b_a_3 -> gaster_tip). Model: a low-rank Gaussian
(rank 3 + diagonal noise, fitted by iterated SVD imputation). A missing segment of a specimen is predicted from the
specimen's own supported segments and trunk chords (conditional mean and sd of its log length); the specimen's own row
is left out of the fit when it is in the population (`without=name`).
Also stored: |log L/R| of every supported left/right pair per segment type; the partner tolerance is
max(PARTNER_TOL_MIN, 4.45 x median) (three robust standard deviations of the supported pairs).

    python -m fitter_3d.antscan_registration.build_data priors --names NAMES.txt --joints JOINTS_DIR --out priors.json
"""

import json
import os

import numpy as np

from .common import ANT_SEG, CHAINS, LEG_SEG, jdump, seg_key
from .product import Product


def key_of(name):
    """specimens are keyed by the md5 of their file name: the prior file holds no specimen or taxon names, and a
    specimen is still recognised (and left out) when the prior is applied to it."""
    import hashlib

    return None if name is None else hashlib.md5(str(name).encode()).hexdigest()


TRUNK = [("mesosoma", "b_t", "b_a_1"), ("head", "b_t", "head_front"), ("gaster", "b_a_3", "gaster_tip")]
KEYS = [f"leg{k}_{s}" for k in "123" for s in LEG_SEG] + list(ANT_SEG) + [t[0] for t in TRUNK]
RANK = 3
SD_MIN = 0.06
PARTNER_TOL_MIN = 0.10
PARTNER_TOL_MAX = 0.40
COXA_Z_MAX = 3.5  # a coxa base farther than this many robust sd from its leg pair's place on the mesosoma axis
# (or nearer to another pair's place) is not that leg's coxa


def supported_lengths(prod):
    """-> per-side chord lengths {(chain, i): um} of directly supported segments, and the specimen row {key: log um}."""
    side = {}
    for ch, js in CHAINS.items():
        for i in range(len(js) - 1):
            if prod.usable(js[i]) and prod.usable(js[i + 1]):
                side[(ch, i)] = float(np.linalg.norm(prod.pos(js[i + 1]) - prod.pos(js[i])))
    row = {}
    acc = {}
    for (ch, i), L in side.items():
        acc.setdefault(seg_key(ch, i), []).append(L)
    for k, v in acc.items():
        row[k] = float(np.log(np.mean(v)))
    for key, a, b in TRUNK:
        if prod.usable(a) and prod.usable(b):
            row[key] = float(np.log(max(np.linalg.norm(prod.pos(b) - prod.pos(a)), 1e-9)))
    return side, row


def _fit(A, rank=RANK, it=40):
    mu = np.nanmean(A, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Z = np.where(np.isnan(A), mu, A)
    rec = Z
    for _ in range(it):
        mu = Z.mean(0)
        U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
        rec = mu + (U[:, :rank] * S[:rank]) @ Vt[:rank]
        Z = np.where(np.isnan(A), rec, A)
    resid = np.where(np.isnan(A), np.nan, A - rec)
    with np.errstate(all="ignore"):
        psi = np.nanvar(resid, 0)
    psi = np.where(np.isfinite(psi), psi, 0.05) + 1e-4
    W = Vt[:rank].T * (S[:rank] / np.sqrt(max(len(A) - 1, 1)))
    return mu, W, psi


def coxa_fraction(prod, x):
    """position of a point along the mesosoma axis neck (b_t) -> petiole joint (b_a_1), as a share of that chord."""
    if not (prod.usable("b_t") and prod.usable("b_a_1")):
        return None
    a, b = prod.pos("b_t"), prod.pos("b_a_1")
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return None
    return float((np.asarray(x, float) - a) @ (b - a) / L / L)


class Population:
    def __init__(self, names, A, asym, coxa_t=None):
        self.names = list(names)
        self.A = np.asarray(A, float)
        self.asym = {k: list(v) for k, v in asym.items()}
        self.coxa_t = coxa_t or {}  # specimen -> {"1": [t, ...], "2": [...], "3": [...]} of its usable coxa bases
        self._cache = {}

    def coxa_identity(self, k, t, without=None):
        """Is a coxa base at axis fraction t typical for leg pair k ("1" / "2" / "3")? Robust z against the supported
        coxae of the population (the specimen itself left out). -> dict(t, own_z, other_z, ok) or None."""
        if t is None or not self.coxa_t:
            return None
        st = {}
        for j in "123":
            v = np.array([x for nm, d in self.coxa_t.items() if nm != key_of(without) for x in d.get(j, [])], float)
            if len(v) < 10:
                return None
            med = float(np.median(v))
            st[j] = (med, max(1.4826 * float(np.median(np.abs(v - med))), 1e-3))
        z = {j: abs(t - st[j][0]) / st[j][1] for j in "123"}
        other = min(z[j] for j in "123" if j != k)
        return dict(t=float(t), own_z=float(z[k]), other_z=float(other), ok=bool(z[k] <= COXA_Z_MAX and z[k] <= other))

    @classmethod
    def build(cls, names, joints_dir):
        rows, used, asym, coxa_t = [], [], {k: [] for k in KEYS}, {}
        for nm in names:
            p = f"{joints_dir}/{nm}/{nm}_joints.json"
            if not os.path.isfile(p):
                continue
            try:
                prod = Product(p)
                side, row = supported_lengths(prod)
            except Exception:
                continue
            if len(row) < 6:
                continue
            for (ch, i), L in side.items():
                if ch[0] in "LA" and ch.endswith(("1", "2", "3", "L")):
                    pc = {"L": "R"}.get(ch[0], "A") + (ch[1] if ch[0] == "L" else "R")
                    if (pc, i) in side:
                        asym[seg_key(ch, i)].append(abs(float(np.log(L / side[(pc, i)]))))
            rows.append([row.get(k, np.nan) for k in KEYS])
            used.append(key_of(nm))
            ct = {}
            for k in "123":
                for sd in "lr":
                    n = f"l_{k}_co_{sd}"
                    if prod.usable(n):
                        t = coxa_fraction(prod, prod.pos(n))
                        if t is not None:
                            ct.setdefault(k, []).append(t)
            coxa_t[key_of(nm)] = ct
        return cls(used, np.array(rows, float).reshape(len(used), len(KEYS)), asym, coxa_t)

    def save(self, path):
        jdump(
            {
                "keys": KEYS,
                "names": self.names,
                "A": [[None if not np.isfinite(x) else float(x) for x in r] for r in self.A],
                "asym": self.asym,
                "coxa_t": self.coxa_t,
                "note": "log chord lengths (um) of directly supported segments; `names` = md5 of the specimen file names; see limbrec/priors.py",
            },
            path,
        )

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        assert d["keys"] == KEYS, "prior file made with other keys"
        A = np.array([[np.nan if x is None else x for x in r] for r in d["A"]], float).reshape(
            len(d["names"]), len(KEYS)
        )
        return cls(d["names"], A, d["asym"], d.get("coxa_t"))

    def n(self, without=None):
        return len(self.names) - (1 if key_of(without) in self.names else 0)

    def model(self, without=None):
        key = key_of(without) if key_of(without) in self.names else None
        if key not in self._cache:
            A = self.A if key is None else np.delete(self.A, self.names.index(key), 0)
            self._cache[key] = _fit(A) if len(A) >= 8 else None
        return self._cache[key]

    def predict(self, obs, target, without=None):
        """obs: {key: log length} of the specimen (target excluded automatically) -> (mean log, sd) or None."""
        m = self.model(without)
        if m is None or target not in KEYS:
            return None
        mu, W, psi = m
        t = KEYS.index(target)
        o = [KEYS.index(k) for k in obs if k in KEYS and k != target and np.isfinite(obs[k])]
        Sig = W @ W.T + np.diag(psi)
        if not o:
            return float(mu[t]), float(max(np.sqrt(Sig[t, t]), SD_MIN))
        x = np.array([obs[KEYS[i]] for i in o])
        So = Sig[np.ix_(o, o)]
        st = Sig[t, o]
        w = np.linalg.solve(So, st)
        mean = mu[t] + w @ (x - mu[o])
        var = Sig[t, t] - w @ st
        return float(mean), float(max(np.sqrt(max(var, 0.0)), SD_MIN))

    def partner_tol(self, key):
        v = self.asym.get(key) or []
        if len(v) < 8:
            return 0.2
        return float(np.clip(4.45 * np.median(v), PARTNER_TOL_MIN, PARTNER_TOL_MAX))
