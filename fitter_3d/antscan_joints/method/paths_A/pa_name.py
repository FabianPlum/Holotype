"""PATHS-A limb naming: fitted, interpretable scores + constrained assignment.

Legs (P paths): score(i -> slot) = legness log-odds(i) [logistic regression on path features]
                                 + slot log-likelihood ratio [Gaussians of the coxa-end / root position along the
                                   mesosoma, t_cx / t_root, per slot 1 / 2 / 3, against the pooled leg distribution]
                                 + side log-likelihood ratio [Gaussians of the root / coxa-end lateral offset, y_root /
                                   y_cx, per side, against the 50/50 mixture].
Head limbs (H paths): multinomial logistic regression antenna / mandible / other + side log-likelihood ratio (root
lateral offset h_l from the head core centre along the trunk's left axis).
Assignment: Hungarian on -score with "unnamed" columns (score 0), so a candidate is only named when its evidence is
positive; candidates that are the same limb (overlapping free parts) are never named twice (the lower-scoring one is
removed and the assignment repeated).

Priors are fitted by fit() from a labelled candidate table (the candidates of a PATHS-A run labelled against the annotations); for the dev
numbers each specimen is named with priors fitted on the other 20 specimens (leave-one-specimen-out).
"""

import json

import numpy as np

LEG_FEATS = ["reach_pct", "own_pct", "free_pct", "tip_trunk_pct", "z_cx_pct", "r_tip10_pct", "r_max_pct"]
HEAD_FEATS = [
    "free_pct",
    "own_pct",
    "reach_pct",
    "tip_trunk_pct",
    "r_med_pct",
    "r_tip10_pct",
    "r_max_pct",
    "h_a",
    "h_d",
]
LEGS = ["L1", "L2", "L3", "R1", "R2", "R3"]
SD_FLOOR = {"t_root": 0.06, "t_cx": 0.06, "u_root": 0.04, "u_cx": 0.04, "y_root_pct": 0.6, "y_cx_pct": 0.8, "h_l": 1.0}
SLOT_FEATS = ("t_root", "t_cx", "u_root", "u_cx")


def _robust(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    sd = max(iqr / 1.349, float(np.std(v)) * 0.5)
    return med, sd


def _logreg(X, y, C=0.5, multi=False):
    from sklearn.linear_model import LogisticRegression

    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit((X - mu) / sd, y)
    return {
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": m.coef_.tolist(),
        "intercept": m.intercept_.tolist(),
        "classes": [str(c) for c in m.classes_],
    }


def _feat_matrix(rows, keys):
    X = np.array([[float(r.get(k, np.nan)) if r.get(k) is not None else np.nan for k in keys] for r in rows], float)
    # impute missing with column medians
    med = np.nanmedian(X, axis=0)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    return X


def fit(table, exclude=()):
    """table: {name: [candidate rows with 'label' and features]} -> priors dict."""
    P = [dict(r, _spec=n) for n, rows in table.items() if n not in exclude for r in rows if r["fam"] == "P"]
    H = [r for n, rows in table.items() if n not in exclude for r in rows if r["fam"] == "H"]
    pri = {
        "leg_feats": LEG_FEATS,
        "head_feats": HEAD_FEATS,
        "excluded": sorted(exclude),
        "slot_feats": list(SLOT_FEATS),
    }
    y = np.array([1 if r["label"] in LEGS else 0 for r in P])
    pri["legness"] = _logreg(_feat_matrix(P, LEG_FEATS), y)
    legs = [r for r in P if r["label"] in LEGS]
    pri["slot"] = {}
    for k in ("1", "2", "3"):
        s = [r for r in legs if r["label"][1] == k]
        pri["slot"][k] = {f: _robust([r[f] for r in s if r.get(f) is not None]) for f in SLOT_FEATS}
    pri["slot_all"] = {f: _robust([r[f] for r in legs if r.get(f) is not None]) for f in SLOT_FEATS}
    by = {}
    for r in legs:
        by.setdefault((r["_spec"], r["label"]), r)
    gaps = {}
    pairs = {"t_root": [], "t_cx": []}
    specs = sorted({k[0] for k in by})
    for sp in specs:
        for sd_ in "LR":
            for a_, b_ in ((1, 2), (2, 3), (1, 3)):
                ra, rb = by.get((sp, f"{sd_}{a_}")), by.get((sp, f"{sd_}{b_}"))
                if ra and rb:
                    for g in ("t_root", "t_cx", "u_root", "u_cx"):
                        if ra.get(g) is not None and rb.get(g) is not None:
                            gaps.setdefault(f"{a_}{b_}_{g}", []).append(ra[g] - rb[g])
        for k in "123":
            ra, rb = by.get((sp, "L" + k)), by.get((sp, "R" + k))
            if ra and rb:
                for g in ("t_root", "t_cx", "u_root", "u_cx"):
                    if ra.get(g) is not None and rb.get(g) is not None:
                        pairs.setdefault(g, []).append(ra[g] - rb[g])
    pri["gap"] = {k: _robust(v) for k, v in gaps.items() if len(v) >= 3}
    pri["pair"] = {g: _robust(v) for g, v in pairs.items() if len(v) >= 3}
    pri["side"] = {}
    for sd_ in ("L", "R"):
        s = [r for r in legs if r["label"][0] == sd_]
        pri["side"][sd_] = {f: _robust([r[f] for r in s]) for f in ("y_root_pct", "y_cx_pct")}
    Hg = [r for r in H if head_gate(r)]
    lab = np.array(
        ["ant" if r["label"] in ("AL", "AR") else ("man" if r["label"] in ("ML", "MR") else "oth") for r in Hg]
    )
    pri["head"] = _logreg(_feat_matrix(Hg, HEAD_FEATS), lab, multi=True)
    pri["head_gate"] = {
        "h_a_min": HEAD_GATE_HA,
        "man_hd_max": MAN_GATE_HD,
        "man_rmed_min": MAN_GATE_RMED,
        "ant_free_min": ANT_GATE_FREE,
    }
    hs = {}
    for sd_, labs in (("L", ("AL", "ML")), ("R", ("AR", "MR"))):
        hs[sd_] = _robust([r["h_l"] for r in H if r["label"] in labs])
    pri["head_side"] = hs
    pri["n_train"] = {"P": len(P), "H": len(H), "legs": len(legs)}
    return pri


# Anatomical gates for head limbs (% BL): antennal sockets and mandible bases sit in the front of the head (root not
# behind the head centre: paths rooted at the back of the head come from leg contacts or the neck); mandibles are
# ventral and broad (palps are thin); antennae are long.
HEAD_GATE_HA = -3.0
MAN_GATE_HD = 2.0
MAN_GATE_RMED = 0.45
ANT_GATE_FREE = 12.0


def head_gate(f):
    return f.get("h_a") is not None and f["h_a"] > HEAD_GATE_HA


def _lognorm(x, m, s, floor):
    s = max(s, floor)
    return -0.5 * ((x - m) / s) ** 2 - np.log(s)


def _lr_decision(model, X):
    Z = (X - np.array(model["mu"])) / np.array(model["sd"])
    W = np.array(model["coef"])
    b = np.array(model["intercept"])
    return Z @ W.T + b


def leg_scores(feats, pri):
    """-> legness log-odds (n,), slot LLR (n, 3), side LLR (n, 2) [L, R]."""
    X = _feat_matrix(feats, pri["leg_feats"])
    lo = _lr_decision(pri["legness"], X)[:, 0]
    n = len(feats)
    slot = np.zeros((n, 3))
    side = np.zeros((n, 2))
    for i, f in enumerate(feats):
        for kk, k in enumerate(("1", "2", "3")):
            v = 0.0
            for g in ("t_root", "t_cx"):
                m, s = pri["slot"][k][g]
                ma, sa = pri["slot_all"][g]
                v += _lognorm(f[g], m, s, SD_FLOOR[g]) - _lognorm(f[g], ma, sa, SD_FLOOR[g])
            slot[i, kk] = v
        ll = {}
        for sd_ in ("L", "R"):
            v = 0.0
            for g in ("y_root_pct", "y_cx_pct"):
                m, s = pri["side"][sd_][g]
                v += _lognorm(f[g], m, s, SD_FLOOR[g])
            ll[sd_] = v
        mx = max(ll.values())
        lse = mx + np.log(0.5 * np.exp(ll["L"] - mx) + 0.5 * np.exp(ll["R"] - mx))
        side[i] = [ll["L"] - lse, ll["R"] - lse]
    return lo, slot, side


def head_scores(feats, pri):
    """-> antenna log-odds vs other, mandible log-odds vs other (n,), side LLR (n, 2)."""
    X = _feat_matrix(feats, pri["head_feats"])
    dec = _lr_decision(pri["head"], X)
    cls = pri["head"]["classes"]
    if dec.shape[1] == 1:  # binary fallback
        dec = np.c_[np.zeros(len(X)), dec]
        cls = cls[:2]
    ci = {c: i for i, c in enumerate(cls)}
    oth = dec[:, ci["oth"]] if "oth" in ci else np.zeros(len(X))
    ant = dec[:, ci["ant"]] - oth if "ant" in ci else np.full(len(X), -9.0)
    man = dec[:, ci["man"]] - oth if "man" in ci else np.full(len(X), -9.0)
    for i, f in enumerate(feats):
        if not head_gate(f):
            ant[i], man[i] = -9.0, -9.0
            continue
        if f.get("free_pct", 0) < ANT_GATE_FREE:
            ant[i] = -9.0
        if f.get("h_d", 0) > MAN_GATE_HD or f.get("r_med_pct", 0) < MAN_GATE_RMED:
            man[i] = -9.0
    side = np.zeros((len(feats), 2))
    for i, f in enumerate(feats):
        ll = {sd_: _lognorm(f["h_l"], *pri["head_side"][sd_], SD_FLOOR["h_l"]) for sd_ in ("L", "R")}
        mx = max(ll.values())
        lse = mx + np.log(0.5 * np.exp(ll["L"] - mx) + 0.5 * np.exp(ll["R"] - mx))
        side[i] = [ll["L"] - lse, ll["R"] - lse]
    return ant, man, side


def hungarian(score, conflicts, names):
    """score (n, m) for m named slots; unnamed allowed (0). conflicts: set of (i, j) never both named.
    Returns {slot name: candidate index}."""
    from scipy.optimize import linear_sum_assignment

    n, m = score.shape
    active = list(range(n))
    while True:
        if not active:
            return {}
        S = score[active]
        big = np.c_[S, np.zeros((len(active), len(active)))]
        r, c = linear_sum_assignment(-big)
        res = {names[cc]: active[rr] for rr, cc in zip(r, c) if cc < m and big[rr, cc] > 0}
        chosen = sorted(res.items(), key=lambda kv: -score[kv[1], names.index(kv[0])])
        bad = None
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                i, j = chosen[a][1], chosen[b][1]
                if (i, j) in conflicts or (j, i) in conflicts:
                    bad = j  # the lower-scoring one
                    break
            if bad is not None:
                break
        if bad is None:
            return res
        active.remove(bad)


def load_priors(path):
    return json.load(open(path))


def _logt(x, m, s, nu=3.0):
    """Student-t log density (up to a constant), heavy-tailed so one odd specimen cannot dominate."""
    z = (x - m) / s
    return -0.5 * (nu + 1) * np.log1p(z * z / nu) - np.log(s)


SLOT_CLIP = 4.0


def slot_terms(f, pri, scale_k=1.5, clip=None):
    clip = SLOT_CLIP if clip is None else clip
    """Weak absolute slot evidence: Student-t LLR of t_root / t_cx per slot vs the pooled leg distribution, capped."""
    out = []
    for k in ("1", "2", "3"):
        v = 0.0
        for g in pri.get("slot_feats", ("t_root", "t_cx")):
            if f.get(g) is None or g not in pri["slot"][k]:
                continue
            m, s = pri["slot"][k][g]
            ma, sa = pri["slot_all"][g]
            v += _logt(f[g], m, scale_k * max(s, SD_FLOOR[g])) - _logt(f[g], ma, scale_k * max(sa, SD_FLOOR[g]))
        out.append(float(np.clip(v, -clip, clip)))
    return out


def side_terms(f, pri, scale_k=1.5):
    ll = {}
    for sd_ in ("L", "R"):
        v = 0.0
        for g in ("y_root_pct", "y_cx_pct"):
            m, s = pri["side"][sd_][g]
            v += _logt(f[g], m, scale_k * max(s, SD_FLOOR[g]))
        ll[sd_] = v
    mx = max(ll.values())
    lse = mx + np.log(0.5 * np.exp(ll["L"] - mx) + 0.5 * np.exp(ll["R"] - mx))
    return [float(np.clip(ll["L"] - lse, -6, 0.7)), float(np.clip(ll["R"] - lse, -6, 0.7))]


GAP_FLOOR = {"t_root": 0.05, "t_cx": 0.06, "u_root": 0.04, "u_cx": 0.04}
PAIR_FLOOR = {"t_root": 0.03, "t_cx": 0.06, "u_root": 0.03, "u_cx": 0.04}
NAME_PENALTY = 1.5


def gap_term(fa, fb, a_, b_, pri):
    v = []
    for g in ("t_root", "t_cx", "u_root", "u_cx"):
        st = pri.get("gap", {}).get(f"{a_}{b_}_{g}")
        if st is None or fa.get(g) is None or fb.get(g) is None:
            continue
        v.append(_logt(fa[g] - fb[g], st[0], max(st[1], GAP_FLOOR[g])) - np.log(1 / 1.2))
    return float(np.clip(np.mean(v), -4.0, 2.0)) if v else 0.0


def pair_term(fl, fr, pri):
    v = []
    for g in ("t_root", "t_cx", "u_root", "u_cx"):
        st = pri.get("pair", {}).get(g)
        if st is None or fl.get(g) is None or fr.get(g) is None:
            continue
        v.append(_logt(fl[g] - fr[g], st[0], max(st[1], PAIR_FLOOR[g])) - np.log(1 / 0.8))
    return float(np.clip(np.mean(v), -4.0, 2.5)) if v else 0.0


def assign_legs_enum(feats, lo, pri, conflicts, max_cands=9, name_penalty=NAME_PENALTY, slot_clip=None):
    """Enumerate side assignments (<= 3 legs per side, slot order fore -> hind) and pick the best total score:
    sum of legness log-odds + side LLR + weak absolute slot LLR, an order penalty when the coxa ends along the
    mesosoma (t_cx) do not run fore -> hind, and a pair term (same slot on both sides -> similar t_cx).
    feats: list of feature dicts (P candidates); lo: legness log-odds; conflicts: set of (i, j) never both named."""
    import itertools

    n = len(feats)
    sl = [slot_terms(f, pri, clip=slot_clip) for f in feats]
    sd = [side_terms(f, pri) for f in feats]
    cand = sorted([i for i in range(n) if lo[i] > -2.5], key=lambda i: -lo[i])[:max_cands]
    best, best_sc = {}, 0.0
    side_opts = {}
    for s_i, s_nm in enumerate(("L", "R")):
        opts = [((), (), 0.0)]
        for k in (1, 2, 3):
            for comb in itertools.combinations(cand, k):
                if any((a, b) in conflicts or (b, a) in conflicts for a, b in itertools.combinations(comb, 2)):
                    continue
                for slots in itertools.combinations((1, 2, 3), k):
                    # order: assign the slots in fore -> hind order to the candidates sorted by t_cx descending
                    cs = sorted(comb, key=lambda i: -(feats[i]["t_cx"] + feats[i]["t_root"]))
                    sc = sum(lo[i] - name_penalty + sd[i][s_i] + sl[i][sl_ - 1] for i, sl_ in zip(cs, slots))
                    for (ia, sa), (ib, sb) in zip(zip(cs, slots), list(zip(cs, slots))[1:]):
                        sc += gap_term(feats[ia], feats[ib], sa, sb, pri)
                    opts.append((tuple(cs), slots, sc))
        side_opts[s_nm] = opts
    import heapq

    top = []  # (score, assignment) of the best assignments, for per-name margins
    for cL, sL, scL in side_opts["L"]:
        setL = set(cL)
        for cR, sR, scR in side_opts["R"]:
            if setL & set(cR):
                continue
            if any((a, b) in conflicts or (b, a) in conflicts for a in cL for b in cR):
                continue
            sc = scL + scR
            mL, mR = dict(zip(sL, cL)), dict(zip(sR, cR))
            for k in (1, 2, 3):
                if k in mL and k in mR:
                    sc += pair_term(feats[mL[k]], feats[mR[k]], pri)
                    sc += 0.3 if feats[mL[k]]["y_root_pct"] > feats[mR[k]]["y_root_pct"] else -0.3
            asg = tuple(sorted([(f"L{k}", i) for k, i in mL.items()] + [(f"R{k}", i) for k, i in mR.items()]))
            if len(top) < 400:
                heapq.heappush(top, (sc, asg))
            elif sc > top[0][0]:
                heapq.heapreplace(top, (sc, asg))
            if sc > best_sc:
                best_sc = sc
                best = {f"L{k}": i for k, i in mL.items()}
                best.update({f"R{k}": i for k, i in mR.items()})
    # margin of each chosen name: best total minus the best total of an assignment without that (name, candidate);
    # the empty assignment (score 0) is always an alternative
    margins = {}
    srt = sorted(top, reverse=True)
    for nm, i in best.items():
        alt = max([sc for sc, asg in srt if (nm, i) not in asg] + [0.0])
        margins[nm] = float(best_sc - alt)
    return best, best_sc, sl, sd, margins


def hungarian_margins(score, conflicts, names, res):
    """Per chosen (name -> row): total of the best assignment minus the best total with that pairing forbidden."""

    def total(S):
        r = hungarian(S, conflicts, names)
        return sum(S[i, names.index(nm)] for nm, i in r.items())

    base = total(score)
    out = {}
    for nm, i in res.items():
        S2 = score.copy()
        S2[i, names.index(nm)] = -1e6
        out[nm] = float(base - total(S2))
    return out
