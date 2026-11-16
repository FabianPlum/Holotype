"""Round 6, ALONG: joints on a named limb centreline (scan frame, micrometres).

Round 7 (FOLDS) adds one option to the round-6 module: optional
`pins` ({joint: xyz}) in place() / solve(): a pinned joint's classifier unary is replaced by a narrow Gaussian
(PIN_SD_PCT) around the arc of the pinned point on the prepared centreline, and start hypotheses that would drop a
pinned joint are excluded. Without pins the behaviour is that of the round-6 module.

    place(limb_name, X, s, r, mesh, BL, core_arc_um, model=None) -> {joint: {"xyz", "conf", "provenance", ...}}

limb_name: L1..L3, R1..R3 (legs: pt ta ti fe tr co), AL / AR (antennae: an_3 an_2 an_1), ML / MR (mandibles: ma).
X: (n, 3) centreline tip first (scan um); s: arc from the tip (optional, recomputed); r: unused (the profile is
measured on the mesh); mesh: trimesh / (V, F) / path of the envelope mesh (scan frame); BL: body length (um);
core_arc_um: arc from the tip where the centreline enters the trunk (optional; used as one feature).

Method
  1. geom_profile.prepare: the centreline is resampled (0.25 % BL), re-centred in its ray-cast cross-sections, its tip
     marched to the mesh surface, and the cross-section profile (equivalent radius, min radius, open rays, gaps) and
     bend angles are measured on the mesh.
  2. features per sample (scale-free: lengths in % BL, radii as logs of ratios): multi-scale constriction depth,
     radius steps (proximal / distal mean), bend angles, relative thickness, distance from the tip, distance to the
     body (first open / thick section; core_arc_um), gap flags.
  3. a classifier per joint type (gradient boosting, fitted on the dev truth, leave-one-specimen-out for every dev
     number) gives p_j(sample) = P(joint j within 0.5 % BL); unary = log p.
  4. a chain dynamic programme over all samples places the joints in anatomical order (distal -> proximal) with
     log-normal segment-length priors (% BL, per leg pair): MAP by max-product, posterior marginals by sum-product.
     Distal parts may be absent (folded or missing tibia / tarsus): the chain then starts at fe / ti / ta at the tip;
     the start hypothesis is chosen by the DP evidence plus a learned prior, and joints of absent parts are null.
  5. confidence per joint = posterior mass within +-1 % BL of the MAP position x the start-hypothesis probability.
Provenance per joint: "scan_feature" (sample with a clear feature: classifier p above its median positive level),
"proportion" (placed mostly by the segment prior), and the start hypothesis / nulls with their reasons.
"""

import os
import pickle

import numpy as np
from scipy import ndimage
from scipy.special import logsumexp

import geom_profile as G

HERE = os.path.dirname(os.path.abspath(__file__))
LEGS = [f"{s}{i}" for s in "LR" for i in (1, 2, 3)]
ANTS, MANS = ["AL", "AR"], ["ML", "MR"]
JOINTS = {"leg": ["pt", "ta", "ti", "fe", "tr", "co"], "ant": ["an_3", "an_2", "an_1"], "man": ["ma"]}
SEGS = {"leg": ["pt-ta", "ta-ti", "ti-fe", "fe-tr", "tr-co"], "ant": ["an_3-an_2", "an_2-an_1"], "man": []}
# start hypotheses: index of the first present joint (the one at the path tip)
STARTS = {"leg": [0, 1, 2, 3, 4], "ant": [0], "man": [0]}
THICK_STARTS = (3, 4)  # the tip is a knee (folded tibia) or the trochanter (femur hidden / missing)
VERSION = "s6 along 2026-11-10 (+S7 pins)"
PIN_SD_PCT = 0.3
_MODEL = {}


def family(limb):
    return "leg" if limb in LEGS else ("ant" if limb in ANTS else "man")


def pair_key(limb):
    return limb[1] if limb in LEGS else limb[0]


# ------------------------------------------------------------------------------------------------- features
def _win_stat(x, w, fn):
    """fn over a centred window of +-w samples (edges clipped)."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = fn(x[max(0, i - w) : min(n, i + w + 1)])
    return out


def _side_mean(x, w):
    """Mean over the w samples distal (before i) and proximal (after i), edge-padded."""
    c = np.r_[0.0, np.cumsum(x)]
    n = len(x)
    i = np.arange(n)
    a0 = np.clip(i - w, 0, n)
    dist = (c[i] - c[a0]) / np.maximum(i - a0, 1)
    b1 = np.clip(i + 1 + w, 0, n)
    prox = (c[b1] - c[i + 1]) / np.maximum(b1 - i - 1, 1)
    dist[i == 0] = x[0]
    prox[i == n - 1] = x[-1]
    return dist, prox


def first_arc(mask, s, sustain_pts=1):
    """Arc of the first sample from the tip where mask holds for sustain_pts consecutive samples (inf if never)."""
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= sustain_pts:
            return float(s[i - sustain_pts + 1])
    return float("inf")


FEATURE_NAMES = None


def features(prof, core_arc_um=None, limb=None):
    """(n, d) feature matrix and the names, from a prepared profile (+ leg pair one-hot when limb is given)."""
    BL, h, s = prof["BL"], prof["h"], prof["s"]
    n = len(s)
    st = 100.0 / BL * h  # % BL per sample
    eps = 0.02 / 100 * BL
    r = np.maximum(prof["r_eq"], 0.0)
    rs = ndimage.gaussian_filter1d(r, 1.0, mode="nearest")
    lr = np.log(rs + eps)
    sp = 100.0 * s / BL
    cols, names = [], []

    def add(nm, v):
        cols.append(np.asarray(v, float))
        names.append(nm)

    add("s_pct", sp)
    add("log_r_pctbl", np.log(100.0 * rs / BL + 0.02))
    # body entry estimates
    s_open = first_arc(prof["open_frac"] >= 0.25, s, max(1, int(round(0.5 / st))))
    s_thick = first_arc(100.0 * rs / BL > 2.5, s, max(1, int(round(1.0 / st))))
    for nm, sv in (("to_open_pct", s_open), ("to_thick_pct", s_thick)):
        add(nm, np.where(np.isfinite(sv), 100.0 * (sv - s) / BL, np.nan))
    add("to_core_pct", 100.0 * (core_arc_um - s) / BL if core_arc_um is not None else np.full(n, np.nan))
    # relative thickness against the limb's own distal level
    s_body = min(s_open, s_thick, core_arc_um if core_arc_um is not None else np.inf)
    lim = s < s_body if np.isfinite(s_body) else np.ones(n, bool)
    if lim.sum() < 5:
        lim = np.ones(n, bool)
    rref = float(np.median(rs[lim])) + eps
    add("log_r_rel", np.log((rs + eps) / rref))
    add("u_body", sp / (100.0 * s_body / BL) if np.isfinite(s_body) and s_body > 0 else np.full(n, np.nan))
    for w_pct in (0.5, 1.0, 2.0, 4.0, 8.0):
        w = max(1, int(round(w_pct / st)))
        mx = ndimage.maximum_filter1d(rs, 2 * w + 1, mode="nearest")
        mn = ndimage.minimum_filter1d(rs, 2 * w + 1, mode="nearest")
        add(f"constr_{w_pct:g}", lr - np.log(mx + eps))
        add(f"ismin_{w_pct:g}", (rs <= mn + 1e-9).astype(float))
        d_, p_ = _side_mean(rs, w)
        add(f"step_{w_pct:g}", np.log(p_ + eps) - np.log(d_ + eps))
        dmx = ndimage.maximum_filter1d(np.r_[np.full(w, rs[0]), rs][:n], w + 1, mode="nearest")  # distal max
        pmx = ndimage.maximum_filter1d(np.r_[rs, np.full(w, rs[-1])][w:], w + 1, mode="nearest")  # proximal max
        add(f"valley_{w_pct:g}", lr - np.log(np.minimum(dmx, pmx) + eps))
    for k in ("bend_1", "bend_2", "bend_4"):
        add(k, prof[k])
    add("open_frac", prof["open_frac"])
    add("gap", prof["gap"].astype(float))
    add("gap_near", ndimage.maximum_filter1d(prof["gap"].astype(float), 2 * max(1, int(round(1.0 / st))) + 1))
    add("ellip", np.log((prof["r_min"] + eps) / (rs + eps)))
    dlr = np.gradient(lr) / st
    add("dlogr", dlr)
    add("d2logr", np.gradient(ndimage.gaussian_filter1d(dlr, 1.0)) / st)
    if limb is not None:
        pk = pair_key(limb)
        for q in ("1", "2", "3"):
            add(f"pair_{q}", np.full(n, float(pk == q)))
    F = np.stack(cols, 1)
    return F, names


# ------------------------------------------------------------------------------------------------- dynamic programme
def _seg_logprior(dpct, mu, sd):
    """log-normal log density (up to a constant) of a segment of dpct % BL; -inf for dpct <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.log(np.maximum(dpct, 1e-6)) - mu) / sd
        out = -0.5 * z * z - np.log(np.maximum(dpct, 1e-6))
    out[dpct <= 0] = -np.inf
    return out


def chain_dp(U, seg_prior, sp, w_seg=1.0, min_gap_pct=0.5):
    """U: (K, n) unary log scores for K joints in order (distal -> proximal), sp: (n,) arc % BL. seg_prior: list of
    (mu, sd) for K-1 segments. Returns (map indices (K,), log evidence, marginals (K, n))."""
    K, n = U.shape
    D = sp[None, :] - sp[:, None]  # D[i, j] = sp[j] - sp[i]
    trans = []
    for k in range(K - 1):
        sgp = seg_prior[k]
        if sgp[0] == "flat":
            T = np.zeros_like(D)
        elif sgp[0] == "ratio":
            # segment as a fraction of the arc from a fixed distal anchor (pt0) to the proximal joint of the segment
            _, mu, sd, a0 = sgp
            L = np.maximum(sp[None, :] - a0, 1e-3) * np.ones_like(D)
            T = w_seg * (_seg_logprior(D / L, mu, sd) - np.log(L))
        else:
            mu, sd = sgp[:2]
            scale = sgp[2] if len(sgp) > 2 else 1.0
            T = w_seg * (_seg_logprior(D / scale, mu, sd) - np.log(scale))  # density per % BL
        T[D < min_gap_pct] = -np.inf
        trans.append(T)
    # max-product
    V = U[0].copy()
    back = []
    for k in range(K - 1):
        M = V[:, None] + trans[k]
        b = np.argmax(M, 0)
        V = M[b, np.arange(n)] + U[k + 1]
        back.append(b)
    idx = [int(np.argmax(V))]
    best = float(V[idx[0]])
    for b in reversed(back):
        idx.append(int(b[idx[-1]]))
    idx = idx[::-1]
    # sum-product
    A = [U[0].copy()]
    for k in range(K - 1):
        A.append(logsumexp(A[-1][:, None] + trans[k], axis=0) + U[k + 1])
    logZ = float(logsumexp(A[-1]))
    B = [np.zeros(n)]
    for k in range(K - 2, -1, -1):
        B.append(logsumexp(trans[k] + (U[k + 1] + B[-1])[None, :], axis=1))
    B = B[::-1]
    marg = np.stack([np.exp(A[k] + B[k] - logZ) for k in range(K)])
    return np.array(idx), logZ, best, marg


# ------------------------------------------------------------------------------------------------- model + placement
def sklearn_versions_of(path):
    """scikit-learn versions recorded in a pickled model file (set of strings; empty when it records none).
    Read from the pickle stream without unpickling (scikit-learn estimators store `_sklearn_version`)."""
    import pickletools

    found, take = set(), False
    with open(path, "rb") as fh:
        for op, arg, _ in pickletools.genops(fh):
            if isinstance(arg, str):
                if take:
                    found.add(arg)
                take = arg == "_sklearn_version"
    return found


def check_sklearn(path):
    """Readable error before a model file is unpickled with another scikit-learn than the one that wrote it
    (estimators pickled by one version may fail to load or predict differently under another). Setting the
    environment variable ANTSCAN_JOINTS_SKLEARN_ANY=1 turns the mismatch into a warning."""
    want = sklearn_versions_of(path)
    try:
        import sklearn
    except ImportError:
        raise ImportError(
            f"the joint classifier {os.path.basename(path)} needs scikit-learn "
            f"({' / '.join(sorted(want)) or 'any version'}; pip install scikit-learn)"
        ) from None
    have = sklearn.__version__
    if want and want != {have}:
        msg = (
            f"the joint classifier {os.path.basename(path)} was written with scikit-learn {' / '.join(sorted(want))}, "
            f"but scikit-learn {have} is installed: install that version (pip install "
            f"scikit-learn=={min(want)}) to reproduce the published joint products, or set "
            f"ANTSCAN_JOINTS_SKLEARN_ANY=1 to load it anyway"
        )
        if os.environ.get("ANTSCAN_JOINTS_SKLEARN_ANY") != "1":
            raise RuntimeError(msg)
        import warnings

        warnings.warn(msg, stacklevel=2)
    return want, have


MODEL_FILE = os.path.join(HERE, "along_model.pkl")
# sha256 of the shipped classifier: a replaced file is refused before it is unpickled
MODEL_SHA256 = "2c48fb0e9881f6817bb9cfdcc8e675b303863977e6be13acec7401fee1b27081"


def file_sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ModelUnpickler(pickle.Unpickler):
    """Unpickler for the classifier file: only scikit-learn and numpy classes are resolved (the shipped file asks for
    nothing else; '_loss' is the module name scikit-learn's compiled loss classes pickle themselves under). Any other
    global, such as os.system, is refused before it is imported."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("sklearn", "numpy") or (module == "_loss" and name.startswith("Cy")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"the joint classifier may not reference {module}.{name}")


def load_model(path=None):
    """The classifier dict. The shipped file must have its recorded sha256; every file is read with ModelUnpickler."""
    path = path or MODEL_FILE
    if path not in _MODEL:
        if os.path.abspath(path) == os.path.abspath(MODEL_FILE) and file_sha256(path) != MODEL_SHA256:
            raise RuntimeError(
                f"{path} is not the shipped joint classifier (sha256 differs from {MODEL_SHA256}); restore it from "
                "the repository"
            )
        check_sklearn(path)
        with open(path, "rb") as fh:
            _MODEL[path] = ModelUnpickler(fh).load()
    return _MODEL[path]


def unaries(model, fam, F):
    """{joint: log-likelihood ratio log(p / base rate) (n,)} from the model's classifiers."""
    out = {}
    for j in JOINTS[fam]:
        jj = "base" if (j == "ma" and model["cfg"].get("ma_base") and "base" in model["clf"]) else j
        p = model["clf"][jj].predict_proba(F)[:, 1]
        out[j] = np.log(np.clip(p, 1e-5, 1.0)) - np.log(model["base_rate"][jj])
    return out


def start_features(prof, core_arc_um, limb):
    """Limb-level features for the 'thick tip' decision (the centreline tip is a knee or a trochanter rather than a
    tarsus): tip radius, tip radius over the limb's 90th-percentile radius, arc to the body, pair."""
    BL = prof["BL"]
    sp = 100.0 * prof["s"] / BL
    rr = 100.0 * np.maximum(prof["r_eq"], 0) / BL
    tipm = (sp >= 0.5) & (sp <= 3.0)
    tip_r = float(np.median(rr[tipm])) if tipm.sum() else float(rr[:5].mean())
    body = 100.0 * core_arc_um / BL if core_arc_um is not None else np.nan
    if not np.isfinite(body):
        F, nm = features(prof, None)
        body = float(np.nanmin([F[0, nm.index("to_open_pct")], F[0, nm.index("to_thick_pct")], sp[-1]]))
    lim = sp <= max(min(body, 70.0), 5.0)
    rmax = float(np.percentile(rr[lim], 90)) if lim.sum() > 3 else tip_r
    pk = pair_key(limb)
    return np.array(
        [
            np.log(tip_r + 0.05),
            np.log((tip_r + 0.05) / (rmax + 0.05)),
            np.log(max(body, 0.5)),
            float(pk == "1"),
            float(pk == "3"),
        ]
    )


def solve(model, limb, prof, F, core_arc_um=None, pins=None):
    """Run the DP for every start hypothesis; returns the best hypothesis, all hypotheses, their probabilities, the
    arc grid (% BL) and the unaries."""
    fam = family(limb)
    js = JOINTS[fam]
    sp = 100.0 * prof["s"] / prof["BL"]
    U = unaries(model, fam, F)
    pins = pins or {}
    for j, a_pct in pins.items():
        U[j] = -0.5 * ((sp - a_pct) / PIN_SD_PCT) ** 2
    cfg = model["cfg"]
    pk = pair_key(limb)
    # core pseudo-joint: the last joint lies a learned arc before the trunk entry
    cp = model["core_prior"].get(fam)
    core_term = np.zeros_like(sp)
    if core_arc_um is not None and cp is not None and cfg.get("w_core", 0) > 0:
        d = 100.0 * core_arc_um / prof["BL"] - sp
        core_term = cfg["w_core"] * (-0.5 * ((d - cp[0]) / cp[1]) ** 2)
    p_thick = None
    if fam == "leg" and model.get("thick_clf") is not None:
        p_thick = float(model["thick_clf"].predict_proba(start_features(prof, core_arc_um, limb)[None])[0, 1])
    hyps = []
    for k0 in STARTS[fam]:
        if k0 > 0 and not cfg.get("allow_absent", True):
            continue
        if any(js.index(j) < k0 or (js.index(j) == k0 and k0 > 0 and pins[j] > 3.0) for j in pins):
            continue  # a pinned joint must be present, and cannot be the tip joint of a cut chain when it lies inside
        sub = js[k0:]
        Uk = np.stack([U[j] for j in sub]) * cfg.get("w_unary", 1.0)
        if k0 > 0:
            # the first present joint sits at the tip: its tip-offset prior is added to its classifier score
            mu, sd = model["tip_offset"].get((fam, sub[0]), (0.8, 0.8))
            Uk[0] = -0.5 * ((sp - mu) / sd) ** 2 + cfg.get("w_tipjoint", 1.0) * Uk[0]
        Uk[-1] = Uk[-1] + core_term
        # segment priors: as fractions of the limb's arc from the tip to the trunk entry when the whole limb is on
        # the centreline and core_arc_um is known (much tighter across genera than % BL), else in % BL
        Ls = 100.0 * core_arc_um / prof["BL"] if core_arc_um is not None else None
        use_frac = k0 == 0 and Ls is not None and Ls > 5.0 and cfg.get("frac_priors", True) and model.get("seg_frac")
        segp = []
        for a, b in zip(sub[:-1], sub[1:]):
            key = f"{a}-{b}"
            if (
                cfg.get("ta_ratio")
                and fam == "leg"
                and k0 == 0
                and key in ("pt-ta", "ta-ti")
                and (pk, "basi") in model.get("ratio", {})
            ):
                if key == "pt-ta":
                    segp.append(("flat",))
                else:
                    a0 = float(sp[int(np.argmax(U["pt"][sp <= 3.0]))]) if (sp <= 3.0).any() else 0.0
                    segp.append(("ratio",) + tuple(model["ratio"][(pk, "basi")]) + (a0,))
            elif use_frac and (fam, pk, key) in model["seg_frac"]:
                segp.append(tuple(model["seg_frac"][(fam, pk, key)]) + (Ls,))
            else:
                segp.append(model["seg"].get((fam, pk, key), model["seg"].get((fam, "*", key))))
        if len(sub) == 1:
            idx = np.array([int(np.argmax(Uk[0]))])
            logZ = float(logsumexp(Uk[0]))
            best = float(Uk[0].max())
            marg = np.exp(Uk - logZ)
        else:
            idx, logZ, best, marg = chain_dp(Uk, segp, sp, w_seg=cfg.get("w_seg", 1.0))
        if p_thick is not None:
            grp = [k for k in STARTS[fam] if (k in THICK_STARTS) == (k0 in THICK_STARTS)]
            within = model["start_prior"].get((fam, k0), 1e-3) / sum(
                model["start_prior"].get((fam, k), 1e-3) for k in grp
            )
            prior = np.log(max(p_thick if k0 in THICK_STARTS else 1.0 - p_thick, 1e-6) * within)
        else:
            prior = np.log(model["start_prior"].get((fam, k0), 1e-3))
        hyps.append(
            {
                "k0": k0,
                "joints": sub,
                "idx": idx,
                "logZ": logZ,
                "best": best,
                "marg": marg,
                "score": cfg.get("w_evidence", 1.0) * logZ + prior,
            }
        )
    hyps = [h for h in hyps if np.isfinite(h["score"]) and np.all(np.isfinite(h["marg"]))]
    if not hyps:
        raise ValueError("no valid joint placement on this centreline (too short for the chain)")
    sc = np.array([h["score"] for h in hyps])
    pr = np.exp(sc - logsumexp(sc))
    hb = hyps[int(np.argmax(sc))]
    return hb, hyps, pr, sp, U


ALL_J = ["pt", "ta", "ti", "fe", "tr", "co", "an_3", "an_2", "an_1", "ma"]


def calib_features(st, j):
    """Per-joint features for the confidence calibrator (fitted in training)."""
    oh = [float(j == q) for q in ALL_J]
    return np.array(
        [
            np.log(st["mass"] + 1e-3),
            np.log(st["sd"] + 0.05),
            np.log(st["p_feat"] + 1e-4),
            np.log(st["p_start"] + 1e-4),
            st["gap_frac"],
            float(st["k0"] > 0),
            float(st["is_tipjoint"]),
        ]
        + oh
    )


def joint_stats(model, limb, prof, hb, pr, hyps, sp, U):
    """{joint: stats} for the chosen hypothesis (None for joints of absent parts): arc, sample index, posterior mass
    within +-conf_win_pct, posterior sd, feature probability at the MAP, start probability, confidence."""
    fam = family(limb)
    js = JOINTS[fam]
    p_start = float(pr[[h["k0"] for h in hyps].index(hb["k0"])])
    win = model["cfg"].get("conf_win_pct", 1.0)
    gap_frac = float(np.mean(prof["gap"]))
    out = {j: None for j in js[: hb["k0"]]}
    for k, j in enumerate(hb["joints"]):
        i = int(hb["idx"][k])
        m = hb["marg"][k]
        mass = float(m[np.abs(sp - sp[i]) <= win].sum())
        mean = float((m * sp).sum())
        sd = float(np.sqrt(max((m * (sp - mean) ** 2).sum(), 0.0)))
        jj = "base" if (j == "ma" and model["cfg"].get("ma_base") and "base" in model["clf"]) else j
        p_feat = float(np.exp(U[j][i]) * model["base_rate"][jj])
        st = {
            "arc": float(sp[i]),
            "i": i,
            "mass": mass,
            "sd": sd,
            "p_feat": p_feat,
            "p_start": p_start,
            "gap_frac": gap_frac,
            "k0": hb["k0"],
            "is_tipjoint": k == 0,
        }
        cal = model.get("calib")
        if cal is not None:
            st["conf"] = float(cal.predict_proba(calib_features(st, j)[None])[0, 1])
        else:
            st["conf"] = mass * p_start
        out[j] = st
    return out


def place(
    limb_name, X, s, r, mesh, BL, core_arc_um, model=None, prof=None, return_debug=False, limb_status=None, pins=None
):
    """Joints on one limb centreline; see the module docstring. Returns {joint: {xyz, conf, provenance, ...}}.
    limb_status 'partial' (PATHS could not resolve the whole limb; its tip may not be the limb's tip): every
    confidence is halved and the joints carry limb_partial=True."""
    fam = family(limb_name)
    model = model if isinstance(model, dict) else load_model(model)
    if prof is None:
        prof = G.prepare(X, s, mesh, BL)
    F, _ = features(prof, core_arc_um, limb_name)
    pin_arc = {}
    for j, p in (pins or {}).items():
        a_, _d = G.project_arc(np.asarray(p, float), prof["X"], prof["s"])
        pin_arc[j] = 100.0 * float(a_) / BL
    try:
        hb, hyps, pr, sp, U = solve(model, limb_name, prof, F, core_arc_um, pins=pin_arc)
    except ValueError as e:
        return {j: {"xyz": None, "conf": 0.0, "provenance": "null", "reason": str(e)} for j in JOINTS[fam]}
    stats = joint_stats(model, limb_name, prof, hb, pr, hyps, sp, U)
    js = JOINTS[fam]
    out = {}
    for k, j in enumerate(js):
        st = stats[j]
        if st is None:
            p_start = float(pr[[h["k0"] for h in hyps].index(hb["k0"])])
            out[j] = {
                "xyz": None,
                "conf": round(p_start, 3),
                "provenance": "null",
                "reason": f"absent: the centreline tip is {js[hb['k0']]} (folded, hidden or missing distal part)",
            }
            continue
        thr = model["feat_thr"].get(j, 0.3)
        prov = "scan_feature" if st["p_feat"] >= thr else "proportion"
        if st["is_tipjoint"] and hb["k0"] == 0 and j in ("pt", "an_3"):
            prov = "scan_feature"  # the tip of the centreline on the mesh surface
        if st["is_tipjoint"] and hb["k0"] > 0:
            prov = "scan_feature"  # the centreline tip (knee / trochanter / tarsomere end at the tip)
        if j in pin_arc:
            prov = "scan_feature_s7_pin"
        out[j] = {
            "xyz": prof["X"][st["i"]].tolist(),
            "conf": round(st["conf"], 3),
            "provenance": prov,
            "arc_pct": round(st["arc"], 3),
            "posterior_mass_pm1pct": round(st["mass"], 3),
            "posterior_sd_pct": round(st["sd"], 3),
            "p_feature": round(st["p_feat"], 3),
            "start_prob": round(st["p_start"], 3),
            "start_joint": js[hb["k0"]],
        }
    if limb_status == "partial":
        for v in out.values():
            v["conf"] = round(0.5 * v["conf"], 3)
            v["limb_partial"] = True
    if return_debug:
        return out, {"prof": prof, "F": F, "hyps": hyps, "pr": pr, "U": U, "best": hb, "stats": stats}
    return out
