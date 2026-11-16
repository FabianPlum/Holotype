"""Combine the limb products of several PATHS approaches into one limb product (round 6).

Per limb name, every approach that has a polyline is a candidate. Two candidates AGREE when their free parts (tip ->
core entry) lie on each other: the share of samples (every 0.5 % BL) of one free part within TOL % BL of the other
polyline, taken in both directions; agreement = the smaller share. The selected candidate is the one with the largest
support (sum of agreements with the other approaches' candidates of the same name); ties go to the approach order
given (the order of the measured dev numbers). After the per-name choice, two names whose selected free parts are the
same physical limb (agreement >= DUP_AGREE) cannot both stay: the one with the lower support (then lower approach
rank) falls back to its next candidate that is not a duplicate, else to no_path.

Confidence of the merged limb = combination of the support (how many approaches put this name on this limb) and the
selected approach's own confidence (see `merged_confidence`). Disagreement is recorded as a flag.

No truth, no specimen names, no oracle: inputs are only the approaches' limb products.
"""

import json
import os

import numpy as np

LEGS = [f"{s}{i}" for s in "LR" for i in (1, 2, 3)]
LIMBS = LEGS + ["AL", "AR", "ML", "MR"]
TOL_PCT = 1.5  # agreement tolerance (% BL); the scorer's joint tolerance, hand-set, sensitivity reported
STEP_PCT = 0.5
DUP_AGREE = 0.6  # two names on the same physical limb
AGREE_SAME = 0.5  # another approach's candidate counts as the same limb (joint-level agreement)


def arclen(X):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])


def dist_poly(P, X):
    P, X = np.atleast_2d(np.asarray(P, float)), np.asarray(X, float)
    if len(X) == 1:
        return np.linalg.norm(P - X[0], axis=1)
    a, ab = X[:-1], X[1:] - X[:-1]
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.empty(len(P))
    for i0 in range(0, len(P), 256):
        Q = P[i0 : i0 + 256]
        u = np.clip(((Q[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0.0, 1.0)
        d = np.linalg.norm(Q[:, None, :] - (a[None] + u[..., None] * ab[None]), axis=2)
        out[i0 : i0 + 256] = d.min(1)
    return out


def free_samples(X, s, core, step):
    s = arclen(X) if s is None else np.asarray(s, float)
    smax = float(s[-1]) if not core else min(float(core), float(s[-1]))
    t = np.linspace(0.0, smax, max(5, int(np.ceil(smax / step)) + 1))
    return np.stack([np.interp(t, s, X[:, k]) for k in range(3)], 1)


def load_product(d, name):
    fj, fz = os.path.join(d, name, "limbs.json"), os.path.join(d, name, "limbs.npz")
    if not (os.path.isfile(fj) and os.path.isfile(fz)):
        return None
    meta = json.load(open(fj))
    z = np.load(fz)
    limbs = {}
    for ch in LIMBS:
        m = (meta.get("limbs") or {}).get(ch) or {}
        if m.get("status") in ("path", "partial") and ch + "_X" in z.files:
            X = np.asarray(z[ch + "_X"], float)
            if X.ndim == 2 and len(X) >= 2 and np.isfinite(X).all():
                s = np.asarray(z[ch + "_s"], float) if ch + "_s" in z.files else arclen(X)
                r = np.asarray(z[ch + "_r"], float) if ch + "_r" in z.files else np.full(len(X), np.nan)
                limbs[ch] = {"X": X, "s": s, "r": r, "meta": m}
    T = {k: np.asarray(z[k]) for k in ("T_X", "T_s", "T_r", "T_part") if k in z.files}
    return {"meta": meta, "limbs": limbs, "T": T}


def agreement(a, b, BL, tol_pct=TOL_PCT):
    """Symmetric agreement of two limb candidates (dicts with X, s, meta.core_arc_um): min of the two coverage shares."""
    step, tol = STEP_PCT * BL / 100.0, tol_pct * BL / 100.0
    Sa = free_samples(a["X"], a["s"], a["meta"].get("core_arc_um"), step)
    Sb = free_samples(b["X"], b["s"], b["meta"].get("core_arc_um"), step)
    ca = float((dist_poly(Sa, b["X"]) <= tol).mean())
    cb = float((dist_poly(Sb, a["X"]) <= tol).mean())
    tip = float(np.linalg.norm(a["X"][0] - b["X"][0]) * 100.0 / BL)
    return {"a_on_b": ca, "b_on_a": cb, "agree": min(ca, cb), "tip_dist_pct": tip}


def merged_confidence(support, n_avail, own_conf):
    """support: sum of agreements with the other approaches (0..n_avail-1); own_conf: the selected approach's
    confidence. Agreement of independent methods dominates; with no support the own confidence is capped low."""
    if n_avail <= 1:
        return 0.5 * own_conf
    frac = support / (n_avail - 1)
    return (
        float(np.clip(0.25 + 0.65 * frac + 0.10 * own_conf, 0.0, 1.0)) if frac > 0 else float(min(0.3, 0.3 * own_conf))
    )


def combine(products, order, BL, tol_pct=TOL_PCT, dup_agree=DUP_AGREE):
    """products: {approach: load_product(...)}; order: approach priority. Returns (limbs dict, arrays, report)."""
    avail = [a for a in order if products.get(a) is not None]
    cand = {ch: [(a, products[a]["limbs"][ch]) for a in avail if ch in products[a]["limbs"]] for ch in LIMBS}
    pair = {}
    choice, report = {}, {}
    for ch in LIMBS:
        cs = cand[ch]
        sup = []
        for i, (ai, ci) in enumerate(cs):
            tot, det = 0.0, {}
            for j, (aj, cj) in enumerate(cs):
                if i == j:
                    continue
                key = (ch, ai, aj)
                if key not in pair:
                    g = agreement(ci, cj, BL, tol_pct)
                    pair[key] = g
                    pair[(ch, aj, ai)] = {
                        "a_on_b": g["b_on_a"],
                        "b_on_a": g["a_on_b"],
                        "agree": g["agree"],
                        "tip_dist_pct": g["tip_dist_pct"],
                    }
                det[aj] = round(pair[key]["agree"], 3)
                tot += pair[key]["agree"]
            sup.append((tot, -order.index(ai), i, det))
        sup.sort(reverse=True)
        report[ch] = {
            "candidates": [cs[i][0] for _, _, i, _ in sup],
            "support": [round(t, 3) for t, _, _, _ in sup],
            "agreements": {cs[i][0]: d for _, _, i, d in sup},
            "n_avail": len(avail),
        }
        choice[ch] = [(cs[i][0], cs[i][1], t) for t, _, i, _ in sup]
    # duplicates across names (two names on one physical limb)
    sel = {ch: 0 for ch in LIMBS if choice[ch]}
    for _ in range(20):
        changed = False
        names = [ch for ch in sel if sel[ch] is not None]
        for i, c1 in enumerate(names):
            for c2 in names[i + 1 :]:
                if sel.get(c1) is None or sel.get(c2) is None:
                    continue
                a1, l1, t1 = choice[c1][sel[c1]]
                a2, l2, t2 = choice[c2][sel[c2]]
                g = agreement(l1, l2, BL, tol_pct)
                if max(g["a_on_b"], g["b_on_a"]) < dup_agree:
                    continue
                # loser: lower support, then lower priority of its approach
                k1, k2 = (t1, -order.index(a1)), (t2, -order.index(a2))
                lose = c2 if k1 >= k2 else c1
                report[lose].setdefault("dropped_as_duplicate_of", []).append(
                    {
                        "other": c1 if lose == c2 else c2,
                        "approach": choice[lose][sel[lose]][0],
                        "overlap": round(max(g["a_on_b"], g["b_on_a"]), 3),
                    }
                )
                nxt = sel[lose] + 1
                sel[lose] = nxt if nxt < len(choice[lose]) else None
                changed = True
        if not changed:
            break
    limbs, arrays = {}, {}
    for ch in LIMBS:
        k = sel.get(ch)
        if k is None:
            limbs[ch] = {"status": "no_path", "confidence": 0.0, "source": None, "evidence": {"combine": report[ch]}}
            continue
        a, c, t = choice[ch][k]
        m = dict(c["meta"])
        own = float(m.get("confidence") or 0.0)
        conf = merged_confidence(t, len(avail), own)
        flags = []
        if len(choice[ch]) > 1 and t < 0.5 * (len(avail) - 1):
            flags.append("approaches_disagree")
        if k > 0:
            flags.append("fallback_after_duplicate")
        limbs[ch] = {
            "status": m.get("status"),
            "confidence": round(conf, 3),
            "source": a,
            "source_confidence": own,
            "support": round(t, 3),
            "core_arc_um": m.get("core_arc_um"),
            "arc_um": float(c["s"][-1]),
            "n_points": int(len(c["X"])),
            "tip_scan": c["X"][0].tolist(),
            "root_scan": c["X"][-1].tolist(),
            "missing": m.get("missing") or (m.get("assign") or {}).get("missing") or [],
            "flags": flags,
            "evidence": {"combine": report[ch], "source_evidence": m.get("evidence")},
        }
        arrays[ch + "_X"], arrays[ch + "_s"], arrays[ch + "_r"] = c["X"], c["s"], c["r"]
        # the other approaches' candidates of this name that lie on the same limb (for joint-level agreement)
        report.setdefault("__agreeing__", {})[ch] = [
            (a2, c2, round(pair[(ch, a, a2)]["agree"], 3))
            for a2, c2, _ in choice[ch]
            if a2 != a and pair.get((ch, a, a2), {}).get("agree", 0.0) >= AGREE_SAME
        ]
    return limbs, arrays, report
