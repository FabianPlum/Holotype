"""S7 ENSEMBLE: combine the S6 products of one specimen made under several grid states (stage.py) into one product
with the S6 contract (joint product + combined limb product).

Inputs per state k (in a fixed, intrinsic state order; the first is the canonical s0 when present):
  <state dir>/trunk/<name>/trunk.json, <state dir>/limbs<sub>/<name>/limbs.{json,npz}, <state dir>/proposals<sub>/<name>/
  <name>_joints.json (sub = '' in the shipped method).
Steps:
  1 head end    each state's final anatomical X; the states are split by the sign of X . X_ref; the larger group is kept
                (tie: the larger sum of TRUNK head-end confidences); the others are dropped (flag).
  2 trunk       per trunk joint (b_t, b_a_1..5, b_h, head_front, gaster_tip) the medoid over the kept states; the trunk
                state kT = the state whose trunk joints lie closest to the medoids (sum over joints) gives BL, frame,
                waist_segments and the trunk arrays (T_*).
  3 limbs       per limb name, S6's combine.combine() with the STATES as the candidates (each state's combined limb):
                agreement of two polylines = smaller of the two shares of their free parts within 1.5 % BL of the other;
                the candidate with the largest support (sum of agreements with the other states) is selected, ties to the
                state order; names that end up on one physical limb are resolved as in S6 (lower support falls back).
                State support fraction f = support / (K - 1) (f = 1 for K = 1).
                Limb confidence = own x (LIMB_BASE + (1 - LIMB_BASE) x f), own = the median S6 limb confidence of the
                states that agree (>= 0.5) with the selected polyline (incl. itself).
  4 joints      per joint of a limb: the placements of the agreeing states (template-fallback placements excluded); if at
                least half of them place the joint, position = medoid of the placements, else null (the most common null
                reason). spread = largest distance of a placement to the medoid (% BL); spread_med = median distance.
                Joint confidence = median S6 confidence of the placements x (LIMB_BASE + (1 - LIMB_BASE) x f)
                x placed share x g(spread),  g(x) = 1 / (1 + (x / SPREAD_SCALE)^2)  (SPREAD_SCALE chosen
                leave-one-specimen-out on dev, declared).
  5 fallback    limbs with no path in the ensemble: round-3 joints (S6 rule, copied) unless on another named centreline.
  6 frame       S6 anatomical_frame() on the trunk state's trunk and the ensemble joints; frame check, L/R check.
With one state the output equals that state's S6 product (positions, confidences, statuses).
Nothing reads truth, oracle limbs, development-set geometry, specimen names or per-specimen constants.

  python ensemble.py --names-file F --work W --states s0,s1,s2,s3 --out W/ens_s0123 [--s3-dir D]
"""

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "assemble"))
sys.dont_write_bytecode = True
import numpy as np

import combine as CB
import assemble as AS  # the round-6 assemble module: constants and frame helpers

VERSION = "S7 ensemble 2026-11-11"
LIMB_BASE = 0.5  # limb / joint confidence factor at zero state support (hand-set)
SPREAD_SCALE = 2.0  # % BL; g(spread) = 1 / (1 + (spread / SPREAD_SCALE)^2); chosen leave-one-specimen-out on the development set (20 of 21 folds)
TRUNK_KEYS = ["b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5", "b_h", "head_front", "gaster_tip"]
LIMB_FLAGS_FROM_STATE = ("head_end_low_confidence", "head_end_from_round3_prior", "waist_segments_unknown")


def medoid(Q):
    Q = np.asarray(Q, float)
    if len(Q) == 1:
        return 0, np.zeros(1)
    D = np.linalg.norm(Q[:, None] - Q[None], axis=2)
    m = int(np.argmin(D.sum(1)))
    return m, D[m]


def g_spread(x, scale=None):
    scale = SPREAD_SCALE if scale is None else scale
    return 1.0 / (1.0 + (x / scale) ** 2) if scale > 0 else 1.0


def load_state(sdir, name, sub):
    st = {"dir": sdir}
    fj = os.path.join(sdir, "trunk", name, "trunk.json")
    st["trunk"] = json.load(open(fj)) if os.path.isfile(fj) else None
    if st["trunk"] is not None and st["trunk"].get("status") != "ok":
        st["trunk"] = None
    fp = os.path.join(sdir, "proposals" + sub, name, f"{name}_joints.json")
    st["prod"] = json.load(open(fp)) if os.path.isfile(fp) else None
    if st["prod"] is not None and st["prod"].get("status") != "ok":
        st["prod"] = None
    st["limbs"] = CB.load_product(os.path.join(sdir, "limbs" + sub), name) if st["prod"] is not None else None
    return st


def head_x(trunk):
    fr = trunk.get("anatomical_frame_final") or trunk["frame"]
    return np.asarray(fr["rotation_scan_to_canonical"], float)[0]


def ensemble(
    name, states, mesh_path=None, s3_dir=None, sub="", fallback=True, limb_base=None, spread_scale=None, pick="support"
):
    """states: list of (key, state_dir) in the fixed order. Returns (joint product, limb json, limb arrays)."""
    t0 = time.time()
    limb_base = LIMB_BASE if limb_base is None else limb_base
    S = {k: load_state(d, name, sub) for k, d in states}
    order = [k for k, _ in states]
    ok = [k for k in order if S[k]["trunk"] is not None and S[k]["prod"] is not None and S[k]["limbs"] is not None]
    info = {"states": order, "states_ok": list(ok), "sub": sub}
    flags = []
    if not ok:
        # no state has a trunk: S6 then takes every joint from round 3; the first state's product is that product
        for k in order:
            if S[k]["prod"] is not None:
                P = json.loads(json.dumps(S[k]["prod"]))
                P.setdefault("completeness", {}).setdefault("flags", []).append(
                    f"ensemble: no state with a trunk; product of state {k}"
                )
                P["ensemble"] = info
                return P, None, None
        raise RuntimeError("no state produced a product")
    if len(ok) < len(order):
        flags.append(f"ensemble_states_failed({','.join(k for k in order if k not in ok)})")
    # ---- 1 head end
    xr = head_x(S[ok[0]]["trunk"])
    grp = {True: [], False: []}
    for k in ok:
        grp[bool(head_x(S[k]["trunk"]) @ xr >= 0)].append(k)
    hc = {
        g: sum(float((S[k]["trunk"].get("confidence") or {}).get("head_end") or 0.0) for k in v) for g, v in grp.items()
    }
    keep = grp[True] if (len(grp[True]), hc[True]) >= (len(grp[False]), hc[False]) else grp[False]
    keep = [k for k in ok if k in keep]
    if len(keep) < len(ok):
        flags.append(
            f"ensemble_head_end_disagrees(kept {','.join(keep)}; dropped {','.join(k for k in ok if k not in keep)})"
        )
    info.update({"kept": keep, "head_votes": {g: v for g, v in grp.items()}})
    K = len(keep)
    # ---- 2 trunk
    BLs = {k: float(S[k]["prod"]["body_length_um"]) for k in keep}
    BLm = float(np.median(list(BLs.values())))
    tmed, tsp, tdev = {}, {}, collections.defaultdict(float)
    for j in TRUNK_KEYS:
        pts = [(k, S[k]["prod"]["joints_scan"].get(j)) for k in keep]
        pts = [(k, np.asarray(v, float)) for k, v in pts if v is not None]
        if not pts:
            continue
        m, d = medoid([p for _, p in pts])
        tmed[j] = (pts[m][0], pts[m][1], [k for k, _ in pts])
        tsp[j] = float(d.max()) * 100.0 / BLm
        for k, p in pts:
            tdev[k] += float(np.linalg.norm(p - pts[m][1]))
    kT = min(keep, key=lambda k: (round(tdev.get(k, 0.0), 6), keep.index(k)))
    trunk, PT = S[kT]["trunk"], S[kT]["prod"]
    BL = float(PT["body_length_um"])
    info.update(
        {"trunk_state": kT, "trunk_spread_pct": {j: round(v, 3) for j, v in tsp.items()}, "BL_um_by_state": BLs}
    )
    joints = {k: None for k in AS.J_NAMES}
    conf = {k: 0.0 for k in AS.J_NAMES}
    prov = {k: "null" for k in AS.J_NAMES}
    reason, jdet, parts = {}, {}, {}
    for w in AS.WINGS:
        reason[w] = "workers have no wings"
    extra = {}
    for j, (km, p, ks) in tmed.items():
        cs = [float(S[k]["prod"]["confidence"].get(j, 0.0)) for k in ks]
        c = float(np.median(cs)) * (len(ks) / K) * g_spread(tsp[j], spread_scale)
        if j in ("head_front", "gaster_tip"):
            extra[j] = p.tolist()
        else:
            joints[j] = p.tolist()
        conf[j] = round(c, 3)
        prov[j] = S[km]["prod"]["provenance"].get(j, "scan_feature")
        jdet[j] = {
            "ens": {
                "n_states": K,
                "n_placed": len(ks),
                "picked_state": km,
                "spread_pct": round(tsp[j], 3),
                "conf_states_median": round(float(np.median(cs)), 3),
            }
        }
    for j in ("b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5", "b_h"):
        if joints.get(j) is None:
            reason[j] = (PT.get("null_reasons") or {}).get(j, "not found by TRUNK in the kept states")
    for f in (PT.get("completeness") or {}).get("flags", []):
        if f.startswith(LIMB_FLAGS_FROM_STATE):
            flags.append(f"{f} [state {kT}]")
    # ---- 3 limbs (states as the candidates of S6's combine)
    prods = {k: S[k]["limbs"] for k in keep}
    limbs, arrays, crep = CB.combine(prods, keep, BL)
    agreeing = crep.get("__agreeing__", {})
    z = np.load(os.path.join(S[kT]["dir"], "trunk", name, "trunk.npz"))
    for k in ("T_X", "T_s", "T_r", "T_part"):
        if k in z.files:
            arrays[k] = z[k]
    limb_conf = {}
    for ch in AS.LIMBS:
        L = limbs[ch]
        if L["status"] == "no_path":
            parts[ch] = {
                "status": "no_path",
                "confidence": 0.0,
                "states_with_path": [k for k in keep if ch in S[k]["limbs"]["limbs"]],
            }
            continue
        ksel = L["source"]
        clus = [ksel] + [a2 for a2, _, _ in agreeing.get(ch, [])]
        if pick == "longest" and len(clus) > 1:
            # VARIANT (tested on dev, not the default): the polyline of the agreeing state with the longest free arc
            # (a geodesic shortcut past a folded knee is shorter than the limb centreline)
            def free_arc(k):
                c = S[k]["limbs"]["limbs"][ch]
                return float(c["meta"].get("core_arc_um") or c["s"][-1])

            kl = max(clus, key=lambda k: (round(free_arc(k), 3), -clus.index(k)))
            if kl != ksel:
                c = S[kl]["limbs"]["limbs"][ch]
                arrays[ch + "_X"], arrays[ch + "_s"], arrays[ch + "_r"] = c["X"], c["s"], c["r"]
                L.update(
                    {
                        "status": c["meta"].get("status"),
                        "core_arc_um": c["meta"].get("core_arc_um"),
                        "arc_um": float(c["s"][-1]),
                        "picked_longest_from": kl,
                    }
                )
                ksel = kl
        frac = 1.0 if K <= 1 else float(L["support"]) / (K - 1)
        lfac = limb_base + (1.0 - limb_base) * frac
        own = []
        for k in clus:
            mk = S[k]["limbs"]["meta"]["limbs"].get(ch) or {}
            own.append(float(mk.get("confidence") or 0.0))
        lc = float(np.median(own)) * lfac
        msel = S[ksel]["limbs"]["meta"]["limbs"][ch]
        L.update(
            {
                "confidence": round(lc, 3),
                "source": f"{ksel}:{msel.get('source')}",
                "state": ksel,
                "source_approach": msel.get("source"),
                "state_support": L["support"],
                "state_frac": round(frac, 3),
                "states_agreeing": clus,
                "s6_confidence_states": [round(x, 3) for x in own],
                "missing": msel.get("missing") or L.get("missing") or [],
            }
        )
        L["flags"] = ["states_disagree" if f == "approaches_disagree" else f for f in L["flags"]] + [
            f"s6:{f}" for f in (msel.get("flags") or [])
        ]
        limb_conf[ch] = lc
        nulls = {}
        for j in AS.CHAIN_J[ch]:
            nm = AS.omni(ch, j)
            pl, nl = [], []
            for k in clus:
                P = S[k]["prod"]
                v = P["joints_scan"].get(nm)
                pv = P["provenance"].get(nm) or ""
                if "template" in pv:
                    continue
                if v is None:
                    nl.append((k, (P.get("null_reasons") or {}).get(nm, "not found on the centreline")))
                else:
                    pl.append(
                        (
                            k,
                            np.asarray(v, float),
                            float(P["confidence"].get(nm, 0.0)),
                            pv,
                            (P.get("joint_detail") or {}).get(nm) or {},
                        )
                    )
            if not pl or len(pl) < len(nl):
                prov[nm] = "null"
                rs = collections.Counter(r for _, r in nl)
                reason[nm] = (
                    rs.most_common(1)[0][0] if rs else "not found on the centreline"
                ) + f" (ensemble: placed in {len(pl)} of {len(pl) + len(nl)} agreeing states)"
                nulls[j] = reason[nm]
                jdet[nm] = {"ens": {"n_states": K, "n_cluster": len(clus), "n_placed": len(pl), "n_null": len(nl)}}
                continue
            m, d = medoid([p for _, p, _, _, _ in pl])
            sp = float(d.max()) * 100.0 / BL
            spm = float(np.median(d[np.arange(len(d)) != m])) * 100.0 / BL if len(d) > 1 else 0.0
            cs = float(np.median([c for _, _, c, _, _ in pl]))
            share = len(pl) / len(clus)
            c = cs * lfac * share * g_spread(sp, spread_scale)
            km = pl[m][0]
            joints[nm] = pl[m][1].tolist()
            conf[nm] = round(c, 3)
            prov[nm] = pl[m][3]
            det = dict(pl[m][4])
            det["ens"] = {
                "n_states": K,
                "n_cluster": len(clus),
                "n_placed": len(pl),
                "n_null": len(nl),
                "picked_state": km,
                "spread_pct": round(sp, 3),
                "spread_med_pct": round(spm, 3),
                "conf_states_median": round(cs, 3),
                "limb_state_frac": round(frac, 3),
                "placed_share": round(share, 3),
            }
            jdet[nm] = det
        parts[ch] = {
            "status": L["status"],
            "source": L["source"],
            "state_support": L["support"],
            "state_frac": round(frac, 3),
            "states_agreeing": clus,
            "confidence": round(lc, 3),
            "flags": L["flags"],
            "missing_declared": L.get("missing"),
            "nulls": nulls,
        }
        if L["status"] == "partial":
            flags.append(f"{ch}_partial(missing: {', '.join(map(str, L.get('missing') or [])) or 'unspecified'})")
        if "states_disagree" in L["flags"]:
            flags.append(f"{ch}_states_disagree")
        if "approaches_disagree" in (msel.get("flags") or []):
            flags.append(f"{ch}_approaches_disagree" + (f" [state {ksel}]" if K > 1 else ""))
    # ---- 5 template fallback (S6 rule, copied from assemble.py)
    s3 = AS.load_s3(s3_dir, name)
    for ch in AS.LIMBS:
        if limbs[ch]["status"] != "no_path":
            continue
        if not fallback or s3 is None:
            for j in AS.CHAIN_J[ch]:
                reason[AS.omni(ch, j)] = "limb has no path (no template fallback available)"
            flags.append(f"{ch}_no_path")
            continue
        sj, sc = s3.get("joints_scan") or {}, s3.get("confidence") or {}
        cand = {j: sj.get(AS.omni(ch, j)) for j in AS.CHAIN_J[ch]}
        dpts = [np.asarray(cand[j], float) for j in AS.DISTAL[ch] if cand.get(j) is not None]
        on_other = None
        if dpts:
            for c2 in AS.LIMBS:
                if c2 == ch or limbs[c2]["status"] == "no_path":
                    continue
                dd = CB.dist_poly(np.array(dpts), arrays[c2 + "_X"]) * 100.0 / BL
                if np.mean(dd <= AS.FALLBACK_ON_LIMB_PCT) >= 0.5:
                    on_other = c2
                    break
        if not any(v is not None for v in cand.values()):
            for j in AS.CHAIN_J[ch]:
                reason[AS.omni(ch, j)] = "limb has no path; round 3 has no joint either"
            flags.append(f"{ch}_no_path")
            parts[ch]["fallback"] = "round 3 null"
            continue
        if on_other:
            for j in AS.CHAIN_J[ch]:
                reason[AS.omni(ch, j)] = f"limb has no path; the round-3 joints lie on the centreline named {on_other}"
            flags.append(f"{ch}_no_path(template fallback rejected: on {on_other})")
            parts[ch]["fallback"] = f"rejected (on {on_other})"
            continue
        for j, v in cand.items():
            nm = AS.omni(ch, j)
            if v is None:
                reason[nm] = "limb has no path; round 3 has no joint"
                continue
            joints[nm] = list(map(float, v))
            conf[nm] = round(min(AS.FALLBACK_CONF_CAP, 0.3 * float(sc.get(nm, 0.0))), 3)
            prov[nm] = "template_fallback (round 3)"
        flags.append(f"{ch}_template_fallback")
        parts[ch]["fallback"] = "round 3"
    # ---- 6 frame
    R, org, finfo = AS.anatomical_frame(trunk, joints, limb_conf)
    if (finfo.get("check_angle_to_output_deg") or 0) > AS.FRAME_CHECK_DEG:
        flags.append(f"frame_check_disagrees({finfo['check_angle_to_output_deg']:.0f} deg: roll uncertain)")
    if finfo.get("lr_consistent") is False:
        n_state_lr = sum(
            1
            for k in keep
            if any(
                f.startswith("left_right_names_disagree")
                for f in (S[k]["prod"].get("completeness") or {}).get("flags", [])
            )
        )
        if n_state_lr * 2 < K:  # most states did not apply the factor already
            flags.append("left_right_names_disagree_with_frame (limb confidences x 0.5)")
            for k in list(conf):
                if k not in AS.WINGS and k[:2] in ("l_", "an", "ma") and "template" not in prov.get(k, ""):
                    conf[k] = round(conf[k] * AS.LR_INCONSISTENT_FACTOR, 3)
        else:
            flags.append("left_right_names_disagree_with_frame (factor already applied in the states)")
    joints.update(extra)
    if s3 is not None and (s3.get("frame") or {}).get("rotation") is not None:
        R3 = np.asarray(s3["frame"]["rotation"], float)
        finfo["angle_to_round3_deg"] = AS.rot_angle(R, R3)
        finfo["axis_angles_to_round3_deg"] = AS.axis_angles(R, R3)
        if finfo["axis_angles_to_round3_deg"]["X"] > 90:
            flags.append("head_end_disagrees_with_round3_frame")
    all_names = AS.J_NAMES + ["head_front", "gaster_tip"]
    js = {k: joints.get(k) for k in all_names}
    jc = {k: (None if v is None else (R @ (np.asarray(v) - org)).tolist()) for k, v in js.items()}
    missing = sorted(k for k in AS.J_NAMES if k not in AS.WINGS and js.get(k) is None)
    info["seconds"] = round(time.time() - t0, 2)
    out = {
        "name": name,
        "units": "micrometre",
        "method": f"{VERSION}: S6 (TRUNK -> PATHS A+C+B -> ALONG) under grid states {','.join(order)}{sub}; "
        "state-level limb vote (S6 combine over states), joint medoid over agreeing states, spread "
        "confidence; template fallback (round 3) for limbs without a path",
        "frame": {"rotation": R.tolist(), "origin_scan": org.tolist(), "info": finfo},
        "joints_scan": js,
        "joints_canonical": jc,
        "confidence": {k: conf.get(k, 0.0) for k in all_names},
        "provenance": {k: prov.get(k, "null") for k in all_names},
        "null_reasons": {k: v for k, v in reason.items() if js.get(k) is None},
        "joint_detail": jdet,
        "completeness": {"missing_joints": missing, "complete": len(missing) == 0, "flags": flags, "parts": parts},
        "body_length_um": BL,
        "waist_segments": trunk.get("waist_segments"),
        "ensemble": info,
        "status": "ok",
        "error": None,
        "seconds": info["seconds"],
    }
    lj = {
        "name": name,
        "units": "um",
        "version": VERSION,
        "source": f"ensemble over states {','.join(keep)}{sub}",
        "frame": trunk["frame"],
        "BL_um": BL,
        "limbs": limbs,
        "params": {"tol_pct": CB.TOL_PCT, "dup_agree": CB.DUP_AGREE, "limb_base": limb_base},
    }
    return out, lj, arrays


def write(out_root, name, out, lj, arrays):
    d = os.path.join(out_root, "proposals", name)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{name}_joints.json"), "w"), indent=1, default=float)
    if lj is not None:
        d = os.path.join(out_root, "limbs", name)
        os.makedirs(d, exist_ok=True)
        json.dump(lj, open(os.path.join(d, "limbs.json"), "w"), indent=1, default=float)
        np.savez_compressed(os.path.join(d, "limbs.npz"), **arrays)


def state_dirs(work, keys):
    return [(k, os.path.join(work, "states", k)) for k in keys]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default=None)
    ap.add_argument("--names-file", default=None)
    ap.add_argument("--work", required=True, help="run_s7.py work dir (states under <work>/states/<key>)")
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub", default="")
    ap.add_argument("--s3-dir", default=None)
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--limb-base", type=float, default=None)
    ap.add_argument("--spread-scale", type=float, default=None)
    ap.add_argument(
        "--pick",
        default="support",
        choices=["support", "longest"],
        help="polyline of a limb: the S6 combine choice (default) or the longest agreeing one (variant)",
    )
    a = ap.parse_args()
    names = a.names.split(",") if a.names else [x.strip() for x in open(a.names_file) if x.strip()]
    st = state_dirs(a.work, a.states.split(","))
    nf = 0
    for n in names:
        try:
            out, lj, arr = ensemble(
                n,
                st,
                s3_dir=a.s3_dir,
                sub=a.sub,
                fallback=not a.no_fallback,
                limb_base=a.limb_base,
                spread_scale=a.spread_scale,
                pick=a.pick,
            )
        except Exception as e:
            import traceback

            out, lj, arr = (
                {
                    "name": n,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(),
                    "joints_scan": {},
                },
                None,
                None,
            )
            nf += 1
        write(a.out, n, out, lj, arr)
    print(f"ensemble {a.states}{a.sub}: {len(names)} specimens, {nf} failed -> {a.out}")


if __name__ == "__main__":
    main()
