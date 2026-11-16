"""Round 6, assemble: last stage for one specimen: trunk product + PATHS limb products -> combined limb product ->
ALONG joints -> anatomical frame -> template fallback for limbs without a path -> joint product (the joint product
contract + head_front, gaster_tip, provenance).

  python assemble.py NAME --mesh M --work W [--s3-dir D] [--loso] [--order A,C,B] [--no-fallback]
Reads  W/trunk/NAME/trunk.{json,npz}, W/paths_<X>/NAME/limbs.{json,npz} (+ stage_done.json)
Writes W/limbs/NAME/limbs.{json,npz} (combined limb product, round-6 limb layout)
       W/proposals/NAME/NAME_joints.json
Nothing here reads truth, oracle limbs, development-set geometry or specimen-specific constants; --loso only selects ALONG's model
fitted without NAME (dev evaluation); the round-3 output (--s3-dir) is read for the template fallback and for the
frame comparison only.
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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "along"))
import numpy as np

import combine as CB

VERSION = "S6 assemble 2026-11-11 (sign-canonical coarse frame)"
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
WINGS = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
LEGS = CB.LEGS
LIMBS = CB.LIMBS
CHAIN_J = {
    **{c: ["pt", "ta", "ti", "fe", "tr", "co"] for c in LEGS},
    "AL": ["an_3", "an_2", "an_1"],
    "AR": ["an_3", "an_2", "an_1"],
    "ML": ["ma"],
    "MR": ["ma"],
}
DISTAL = {
    **{c: ["fe", "ti", "ta", "pt"] for c in LEGS},
    "AL": ["an_2", "an_3"],
    "AR": ["an_2", "an_3"],
    "ML": ["ma"],
    "MR": ["ma"],
}
DEFAULT_ORDER = ["A", "C", "B"]  # path accuracy on the development set with TRUNK trunks: A 142, C 139, B 126 ok
FALLBACK_CONF_CAP = 0.15
FRAME_CHECK_DEG = 30.0  # coxa-attachment frame check vs output frame (hand-set)
LR_INCONSISTENT_FACTOR = 0.5  # hand-set; no dev specimen triggers it
TEMPLATE_CHORD_PITCH_DEG = 5.26  # OmniAnt25 template: angle of b_a_1 -> b_t above +X (template joints)
JOINT_AGREE_PCT = 2.0  # joint placements on the approaches' centrelines agree within this (% BL)
JOINT_DISAGREE_FACTOR = 0.6  # confidence factor when they do not (hand-set; chosen among 2 / 3 % BL on dev)
FALLBACK_ON_LIMB_PCT = 1.5  # a fallback limb whose distal joints lie on another named limb's centreline is dropped
HEAD_LOW = 0.6  # TRUNK head-end confidence below this: flag; limb joints x the head factor (22:50 rule)
HEAD_FACTOR_BASE = 0.6  # head factor = 0.6 + 0.4 x head-end confidence (TRUNK's own factor for trunk joints)
HEAD_OVERRIDE_FACTOR = 0.5  # all joints when the head end came from the round-3 prior (hand-set, = LR factor)


def omni(ch, j):
    if ch in LEGS:
        return f"l_{ch[1]}_{j}_{ch[0].lower()}"
    if ch in ("AL", "AR"):
        return f"{j}_{ch[1].lower()}"
    return f"ma_{ch[1].lower()}"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rot_angle(Ra, Rb):
    c = (np.trace(np.asarray(Ra) @ np.asarray(Rb).T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def axis_angles(Ra, Rb):
    Ra, Rb = np.asarray(Ra), np.asarray(Rb)
    return {ax: float(np.degrees(np.arccos(np.clip(Ra[i] @ Rb[i], -1, 1)))) for i, ax in enumerate("XYZ")}


def load_s3(s3_dir, name):
    if not s3_dir:
        return None
    f = os.path.join(s3_dir, name, f"{name}_joints.json")
    if not os.path.isfile(f):
        return None
    d = json.load(open(f))
    return d if d.get("status", "ok") != "failed" else None


def stage_ok(work, stage, name):
    f = os.path.join(work, stage, name, "stage_done.json")
    if not os.path.isfile(f):
        return False, "stage not run"
    r = json.load(open(f))
    return r.get("status") == "ok", r.get("error")


# ------------------------------------------------------------------------------------------------ frame from anatomy
def tilt_template(R):
    """Template convention: in the OmniAnt25 template the b_a_1 -> b_t chord points TEMPLATE_CHORD_PITCH_DEG above +X
    (towards +Z), so the canonical X is the chord tilted down by that angle about Y (a property of the template, not
    fitted to data; on synthetic instances the chord lies 4.7 deg above the GT X in the median)."""
    th = np.radians(TEMPLATE_CHORD_PITCH_DEG)
    return np.stack([np.cos(th) * R[0] - np.sin(th) * R[2], R[1], np.sin(th) * R[0] + np.cos(th) * R[2]])


def anatomical_frame(trunk, joints, limb_conf):
    """Output frame = TRUNK's fine-pass anatomical frame (X = b_a_1 -> neck; roll from the appendage attachments, the
    head flexion and the plane of the trunk path), tilted by the template chord pitch. Frame CHECK from anatomy:
    X = b_a_1 -> neck, Z = away from the coxa attachments (minus the mean direction from the trunk axis to the
    co and tr joints of the named legs with limb confidence >= 0.3), Y = Z x X; its angle to the output frame is
    reported and a disagreement > FRAME_CHECK_DEG is flagged. Left/right check: the left limbs' coxae / antennal
    sockets must lie on +Y of their right partners in the output frame."""
    tj = trunk["trunk_joints_scan"]
    fr_t = trunk.get("anatomical_frame_final") or trunk["frame"]
    Rt = np.asarray(fr_t["rotation_scan_to_canonical"], float)
    R = tilt_template(Rt)
    if tj.get("neck") is not None and tj.get("b_a_1") is not None:
        nk, b1 = np.asarray(tj["neck"], float), np.asarray(tj["b_a_1"], float)
        ex, org = unit(nk - b1), 0.5 * (nk + b1)
    else:  # damaged trunk (e.g. no waist / gaster found): TRUNK's frame axis and origin
        b1 = np.asarray(fr_t["origin_scan"], float)
        ex, org = Rt[0].copy(), b1.copy()
    info = {
        "source": "TRUNK anatomical_frame_final, tilted by the template chord pitch",
        "template_chord_pitch_deg": TEMPLATE_CHORD_PITCH_DEG,
    }
    pts = []
    for ch in LEGS:
        if limb_conf.get(ch, 0) < 0.3:
            continue
        for j in ("co", "tr"):
            p = joints.get(omni(ch, j))
            if p is not None:
                pts.append((ch, np.asarray(p, float)))
    info["check_n_leg_points"] = len(pts)
    if len({c for c, _ in pts}) >= 2:
        d = np.zeros(3)
        for _, p in pts:
            v = p - b1
            v = v - (v @ ex) * ex
            d += unit(v)
        ez = -unit(d - (d @ ex) * ex)
        ey = unit(np.cross(ez, ex))
        Rc = tilt_template(np.stack([ex, ey, np.cross(ex, ey)]))
        info["check_frame_rotation"] = Rc.tolist()
        info["check_angle_to_output_deg"] = rot_angle(Rc, R)
        info["check_axis_angles_deg"] = axis_angles(Rc, R)
    votes = []
    for i in (1, 2, 3):
        a, b = joints.get(f"l_{i}_co_l"), joints.get(f"l_{i}_co_r")
        if a is not None and b is not None:
            votes.append(float(np.sign((np.asarray(a) - np.asarray(b)) @ R[1])))
    a, b = joints.get("an_1_l"), joints.get("an_1_r")
    if a is not None and b is not None:
        votes.append(float(np.sign((np.asarray(a) - np.asarray(b)) @ R[1])))
    info["lr_votes"] = votes
    info["lr_consistent"] = None if not votes else bool(sum(votes) > 0)
    info["angle_to_trunk_frame_deg"] = rot_angle(R, Rt)
    return R, org, info


# ------------------------------------------------------------------------------------------------ assembly
def assemble(name, mesh_path, work, s3_dir=None, loso=False, order=None, fallback=True, tag="", joint_pick="medoid"):
    t0 = time.time()
    order = list(order or DEFAULT_ORDER)
    s3 = load_s3(s3_dir, name)
    flags, parts = [], {}
    joints = {k: None for k in J_NAMES}
    conf = {k: 0.0 for k in J_NAMES}
    prov = {k: "null" for k in J_NAMES}
    reason, jdet = {}, {}
    for w in WINGS:
        reason[w] = "workers have no wings"
    ok_t, err_t = stage_ok(work, "trunk", name)
    trunk = None
    if ok_t:
        trunk = json.load(open(os.path.join(work, "trunk", name, "trunk.json")))
        if trunk.get("status") != "ok":
            ok_t, err_t, trunk = False, trunk.get("error"), None
    if trunk is None:
        # no trunk: whole specimen from the template fallback (if any), low confidence
        if s3 is None:
            raise RuntimeError(f"trunk stage failed ({err_t}) and no round-3 output to fall back on")
        flags.append(f"trunk_failed({err_t}): all joints from the round-3 template fallback")
        for k in J_NAMES:
            v = (s3.get("joints_scan") or {}).get(k)
            if v is not None and k not in WINGS:
                joints[k], prov[k] = list(map(float, v)), "template_fallback"
                conf[k] = round(min(FALLBACK_CONF_CAP, 0.3 * float((s3.get("confidence") or {}).get(k, 0.0))), 3)
        R = np.asarray(s3["frame"]["rotation"], float)
        org = np.asarray(s3["frame"]["origin_scan"], float)
        finfo = {"source": "round-3 frame (trunk failed)"}
        BL = None
    else:
        BL = float(trunk["BL_um"])
        tj = trunk["trunk_joints_scan"]
        tc = trunk.get("confidence") or {}
        for k_out, k_in in (
            ("b_t", "neck"),
            ("b_h", "neck"),
            ("b_a_1", "b_a_1"),
            ("b_a_2", "b_a_2"),
            ("b_a_3", "b_a_3"),
            ("b_a_4", "b_a_4"),
            ("b_a_5", "b_a_5"),
        ):
            if tj.get(k_in) is not None:
                joints[k_out], conf[k_out] = list(map(float, tj[k_in])), float(tc.get(k_in, 0.0))
                prov[k_out] = "scan_feature (trunk: %s)" % (trunk.get("joint_rules") or {}).get(k_in, "gaster landmark")
        extra = {}
        for k in ("head_front", "gaster_tip"):
            if tj.get(k) is not None:
                extra[k] = list(map(float, tj[k]))
                conf[k] = float(tc.get(k, 0.0))
                prov[k] = "scan_feature (trunk path end on the surface)"
        if tc.get("head_end") is not None and float(tc["head_end"]) < HEAD_LOW:  # a confidence of 0.0 is low
            flags.append(f"head_end_low_confidence({tc.get('head_end')})")
        ov = (trunk.get("evidence") or {}).get("head_end_override")
        if ov:
            flags.append(f"head_end_from_round3_prior(anatomy confidence {ov.get('anatomy_head_end_confidence')})")
        if trunk.get("waist_segments") is None:
            flags.append("waist_segments_unknown (b_a_2/b_a_3 by proportion)")
        # ---- limbs: combine the approaches
        prods, used = {}, []
        for a in order:
            ok, err = stage_ok(work, f"paths_{a}", name)
            if ok:
                p = CB.load_product(os.path.join(work, f"paths_{a}"), name)
                if p is not None:
                    prods[a] = p
                    used.append(a)
                    continue
            flags.append(f"paths_{a}_failed({err})")
        limbs, arrays, crep = CB.combine(prods, order, BL)
        agreeing = crep.get("__agreeing__", {})
        if "T_X" in (np.load(os.path.join(work, "trunk", name, "trunk.npz")).files):
            z = np.load(os.path.join(work, "trunk", name, "trunk.npz"))
            for k in ("T_X", "T_s", "T_r", "T_part"):
                if k in z.files:
                    arrays[k] = z[k]
        ld = os.path.join(work, "limbs" + tag, name)
        os.makedirs(ld, exist_ok=True)
        json.dump(
            {
                "name": name,
                "units": "um",
                "version": VERSION,
                "source": "combined " + "+".join(used),
                "frame": trunk["frame"],
                "BL_um": BL,
                "limbs": limbs,
                "params": {"order": order, "tol_pct": CB.TOL_PCT, "dup_agree": CB.DUP_AGREE},
            },
            open(os.path.join(ld, "limbs.json"), "w"),
            indent=1,
            default=float,
        )
        np.savez_compressed(os.path.join(ld, "limbs.npz"), **arrays)
        # ---- ALONG
        import along as AL
        import geom_profile as GP

        mpath = (
            os.path.join(HERE, os.pardir, "along", "loso_models", f"{name}.pkl")
            if loso
            else os.path.join(HERE, os.pardir, "along", "along_model.pkl")
        )
        if loso and not os.path.isfile(mpath):
            raise FileNotFoundError(f"no LOSO ALONG model for {name}")
        model = AL.load_model(mpath)
        tm = GP.as_trimesh(mesh_path)
        limb_conf = {}
        for ch in LIMBS:
            L = limbs[ch]
            if L["status"] == "no_path":
                parts[ch] = {"status": "no_path", "confidence": 0.0}
                continue
            X, s, r = arrays[ch + "_X"], arrays[ch + "_s"], arrays[ch + "_r"]
            res = AL.place(ch, X, s, r, tm, BL, L.get("core_arc_um"), model=model, limb_status=L["status"])
            # the same joints on the other approaches' centrelines of this limb (joint-level agreement)
            alts = []
            for a2, c2, ag in agreeing.get(ch, []):
                try:
                    r2 = AL.place(
                        ch,
                        c2["X"],
                        c2["s"],
                        c2["r"],
                        tm,
                        BL,
                        c2["meta"].get("core_arc_um"),
                        model=model,
                        limb_status=c2["meta"].get("status"),
                    )
                    alts.append((a2, ag, r2))
                except Exception as e:
                    flags.append(f"{ch}_along_on_{a2}_failed({type(e).__name__})")
            lc = float(L["confidence"])
            limb_conf[ch] = lc
            nulls = {}
            for j, v in res.items():
                nm = omni(ch, j)
                others = [
                    (a2, np.asarray(r2[j]["xyz"], float), r2[j]["conf"])
                    for a2, _, r2 in alts
                    if r2.get(j, {}).get("xyz") is not None
                ]
                n_null_alt = sum(1 for _, _, r2 in alts if r2.get(j, {}).get("xyz") is None)
                if v.get("xyz") is None:
                    prov[nm] = "null"
                    reason[nm] = v.get("reason") or "not found on the centreline"
                    nulls[j] = reason[nm]
                    jdet[nm] = {"along_conf": 0.0, "limb_conf": lc, "n_alt": len(alts), "n_alt_placed": len(others)}
                    continue
                p0 = np.asarray(v["xyz"], float)
                pick, src = p0, L["source"]
                if joint_pick == "medoid" and len(others) >= 2:
                    P = [(L["source"], p0)] + [(a2, q) for a2, q, _ in others]
                    cost = [sum(np.linalg.norm(q - q2) for _, q2 in P) for _, q in P]
                    k = int(np.argmin(cost))
                    pick, src = P[k][1], P[k][0]
                d = [float(np.linalg.norm(q - pick)) * 100.0 / BL for _, q, _ in others] + (
                    [float(np.linalg.norm(p0 - pick)) * 100.0 / BL] if src != L["source"] else []
                )
                joints[nm] = list(map(float, pick))
                # confidence = ALONG's joint confidence x the combined limb confidence, x JOINT_DISAGREE_FACTOR when
                # the same joint placed on another approach's centreline of this limb lies > JOINT_AGREE_PCT away
                fac = JOINT_DISAGREE_FACTOR if (d and max(d) > JOINT_AGREE_PCT) else 1.0
                conf[nm] = round(float(v["conf"]) * lc * fac, 3)
                jdet[nm] = {
                    "along_conf": round(float(v["conf"]), 3),
                    "limb_conf": lc,
                    "n_alt": len(alts),
                    "n_alt_placed": len(others),
                    "n_alt_null": n_null_alt,
                    "picked_from": src,
                    "spread_pct": round(max(d), 3) if d else None,
                    "n_within_1.5pct": int(sum(x <= 1.5 for x in d)) if d else 0,
                    "limb_status": L["status"],
                    "provenance_along": v.get("provenance"),
                    "start_joint": v.get("start_joint"),
                    "posterior_sd_pct": v.get("posterior_sd_pct"),
                }
                prov[nm] = (
                    f"scan_feature (along: {v.get('provenance')})"
                    if v.get("provenance") == "scan_feature"
                    else f"proportion (along: {v.get('provenance')})"
                )
            parts[ch] = {
                "status": L["status"],
                "source": L["source"],
                "support": L["support"],
                "confidence": lc,
                "flags": L["flags"],
                "missing_declared": L.get("missing"),
                "nulls": nulls,
            }
            if L["status"] == "partial":
                flags.append(f"{ch}_partial(missing: {', '.join(map(str, L.get('missing') or [])) or 'unspecified'})")
            if "approaches_disagree" in L["flags"]:
                flags.append(f"{ch}_approaches_disagree")
        # ---- template fallback for limbs without a path
        for ch in LIMBS:
            if limbs[ch]["status"] != "no_path":
                continue
            if not fallback or s3 is None:
                for j in CHAIN_J[ch]:
                    reason[omni(ch, j)] = "limb has no path (no template fallback available)"
                flags.append(f"{ch}_no_path")
                continue
            sj = s3.get("joints_scan") or {}
            sc = s3.get("confidence") or {}
            cand = {j: sj.get(omni(ch, j)) for j in CHAIN_J[ch]}
            dpts = [np.asarray(cand[j], float) for j in DISTAL[ch] if cand.get(j) is not None]
            on_other = None
            if dpts:
                for c2 in LIMBS:
                    if c2 == ch or limbs[c2]["status"] == "no_path":
                        continue
                    d = CB.dist_poly(np.array(dpts), arrays[c2 + "_X"]) * 100.0 / BL
                    if np.mean(d <= FALLBACK_ON_LIMB_PCT) >= 0.5:
                        on_other = c2
                        break
            if not any(v is not None for v in cand.values()):
                for j in CHAIN_J[ch]:
                    reason[omni(ch, j)] = "limb has no path; round 3 has no joint either"
                flags.append(f"{ch}_no_path")
                parts[ch]["fallback"] = "round 3 null"
                continue
            if on_other:
                for j in CHAIN_J[ch]:
                    reason[omni(ch, j)] = f"limb has no path; the round-3 joints lie on the centreline named {on_other}"
                flags.append(f"{ch}_no_path(template fallback rejected: on {on_other})")
                parts[ch]["fallback"] = f"rejected (on {on_other})"
                continue
            for j, v in cand.items():
                nm = omni(ch, j)
                if v is None:
                    reason[nm] = "limb has no path; round 3 has no joint"
                    continue
                joints[nm] = list(map(float, v))
                conf[nm] = round(min(FALLBACK_CONF_CAP, 0.3 * float(sc.get(nm, 0.0))), 3)
                prov[nm] = "template_fallback (round 3)"
            flags.append(f"{ch}_template_fallback")
            parts[ch]["fallback"] = "round 3"
        R, org, finfo = anatomical_frame(trunk, joints, limb_conf)
        if (trunk.get("evidence") or {}).get("pca_sign_canonical"):
            finfo["coarse_pca_signs"] = trunk["evidence"]["pca_sign_canonical"]
        if (finfo.get("check_angle_to_output_deg") or 0) > FRAME_CHECK_DEG:
            flags.append(f"frame_check_disagrees({finfo['check_angle_to_output_deg']:.0f} deg: roll uncertain)")
        hc = tc.get("head_end")
        if ov:
            # the head end was taken from the round-3 frame against TRUNK's (weak) anatomy vote: every joint x 0.5
            for k in list(conf):
                if k not in WINGS and "template" not in prov.get(k, ""):
                    conf[k] = round(conf[k] * HEAD_OVERRIDE_FACTOR, 3)
        if hc is not None and hc < HEAD_LOW:
            # an uncertain head end can swap fore / hind and left / right: limb joints get TRUNK's head factor
            hf = HEAD_FACTOR_BASE + (1 - HEAD_FACTOR_BASE) * float(hc)
            for k in list(conf):
                if k not in WINGS and k[:2] in ("l_", "an", "ma") and "template" not in prov.get(k, ""):
                    conf[k] = round(conf[k] * hf, 3)
        if finfo.get("lr_consistent") is False:
            # the left limbs do not lie on the left of the right ones in the anatomical frame: the names, the dorsal sign
            # or the head end are wrong somewhere -> every limb joint loses half its confidence
            flags.append("left_right_names_disagree_with_frame (limb confidences x 0.5)")
            for k in list(conf):
                if k not in WINGS and k[:2] in ("l_", "an", "ma") and "template" not in prov.get(k, ""):
                    conf[k] = round(conf[k] * LR_INCONSISTENT_FACTOR, 3)
        joints.update({k: v for k, v in extra.items()})
    # frame comparisons
    if s3 is not None and (s3.get("frame") or {}).get("rotation") is not None:
        R3 = np.asarray(s3["frame"]["rotation"], float)
        finfo["angle_to_round3_deg"] = rot_angle(R, R3)
        finfo["axis_angles_to_round3_deg"] = axis_angles(R, R3)
        if finfo["axis_angles_to_round3_deg"]["X"] > 90:
            flags.append("head_end_disagrees_with_round3_frame")
    all_names = J_NAMES + ["head_front", "gaster_tip"]
    js = {k: joints.get(k) for k in all_names}
    jc = {k: (None if v is None else (R @ (np.asarray(v) - org)).tolist()) for k, v in js.items()}
    missing = sorted(k for k in J_NAMES if k not in WINGS and js.get(k) is None)
    out = {
        "name": name,
        "units": "micrometre",
        "method": f"{VERSION}: TRUNK -> PATHS {'+'.join(order)} (agreement "
        "combine) -> ALONG; template fallback (round 3) for limbs without a path",
        "frame": {"rotation": R.tolist(), "origin_scan": org.tolist(), "info": finfo},
        "joints_scan": js,
        "joints_canonical": jc,
        "confidence": {k: conf.get(k, 0.0) for k in all_names},
        "provenance": {k: prov.get(k, "null") for k in all_names},
        "null_reasons": {k: v for k, v in reason.items() if js.get(k) is None},
        "joint_detail": jdet,
        "completeness": {"missing_joints": missing, "complete": len(missing) == 0, "flags": flags, "parts": parts},
        "body_length_um": BL,
        "waist_segments": None if trunk is None else trunk.get("waist_segments"),
        "status": "ok",
        "error": None,
        "seconds": round(time.time() - t0, 2),
    }
    return out


def write(out, work, name, tag=""):
    d = os.path.join(work, "proposals" + tag, name)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{name}_joints.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--s3-dir", default=None)
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--order", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output dirs limbs<tag>/, proposals<tag>/")
    ap.add_argument("--joint-pick", default="medoid", choices=["primary", "medoid"])
    a = ap.parse_args()
    t0 = time.time()
    try:
        out = assemble(
            a.name, a.mesh, a.work, a.s3_dir, a.loso, a.order.split(","), not a.no_fallback, a.tag, a.joint_pick
        )
    except Exception as e:
        out = {
            "name": a.name,
            "units": "micrometre",
            "method": VERSION,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
            "joints_scan": {k: None for k in J_NAMES},
            "seconds": round(time.time() - t0, 2),
        }
    write(out, a.work, a.name, a.tag)
    print(json.dumps({"name": a.name, "status": out["status"], "error": out.get("error"), "seconds": out["seconds"]}))
    sys.exit(0 if out["status"] == "ok" else 3)


if __name__ == "__main__":
    main()
