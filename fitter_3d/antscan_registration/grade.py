"""The specimen class: ONE answer to "can a consumer take this specimen without looking at it?".

  do_not_use    no usable registration: the registration failed or its root was placed by a fallback, the joint product
                is not ok, the head end is doubtful and the scan does not confirm it, the mesosoma is not registered
                on the scan (STATUS's rule, rs.status.decide_specimen), or a processing error
  model_grade   fit for a shape model: every condition below holds (all truth-free)
  partial       everything else: use the measures by their flags (morphometrics_measurable.csv); not for a shape model

Model-grade conditions (K = constants):
  M1 trunk          every trunk part (head, mesosoma, waist, gaster) measurable: its joint is usable, it lies on the
                    scan, and (mesosoma, gaster) the mesh reaches its scan surface (rs.status.decide_parts)
  M2 limbs          at most K['MAX_SEG_FILLED'] of the 34 leg / antenna segments are filled in the unposed bundle:
                    not measurable, or measurable but filled by the bundle's length gate (5 = at most one whole leg)
  M3 trunk fill     no trunk vertex filled in the unposed bundle
  M4 edges          no abnormal stretched edge in the unposed, filled mesh: longer than 8 x the population median of
                    the same edge AND longer than 10 % of the template body length (the production audit's rule). Edges
                    above 8 x the median and 1.5 % of the body length are counted and reported, not gating
  M5 antennae       no registered (measurable) scape or funiculus is crumpled: zigzag (total turning minus net bend of
                    its medial curve) at most K['ZIGZAG_DEG']
  M7 frame          the joint product does not doubt left / right (frame check) -- per-side parts could be exchanged
  M8 native         the registration's own status is not 'failed' and its native parameters reproduce the mesh
A condition that could NOT be evaluated (its check raised an error, the unposed bundle was not built) counts as not
met: a specimen is model grade only when every condition was checked and holds.

M6, head roll (is the scan's head surface more mirror-symmetric about a plane rolled away from the registered head's
mid-plane?), is measured and reported (`head_roll`) and does NOT gate by default (K['ROLL_GATE'] False): on real scans
none of its flags was confirmed by eye by any of three readers, and two of four lay at the end of its search range.
With K['ROLL_GATE'] true it gates as before (best roll beyond K['ROLL_DEG'] with contrast above K['ROLL_CONTRAST'],
and not at the end of the search range).
"""

K_GRADE = dict(
    MAX_SEG_FILLED=5,
    # antennae: set on DEVELOPMENT registrations only
    ZIGZAG_DEG=60.0,
    ROLL_GATE=False,
    ROLL_DEG=15.0,
    ROLL_CONTRAST=1.25,
    ROOT_FALLBACK_IS_DNU=True,
)
CLASSES = ("model_grade", "partial", "do_not_use")
LABEL = dict(
    M1_trunk="trunk parts",
    M2_limbs="limb segments",
    M3_trunk_fill="trunk fill of the unposed mesh",
    M4_edges="stretched edges of the unposed mesh",
    M5_antennae="antenna shape",
    M6_head_roll="head roll",
    M7_frame="left / right frame check",
    M8_native="native parameters",
)


def scorer_status(cls):
    return {"model_grade": "ok", "complete": "ok", "partial": "review", "do_not_use": "failed"}[cls]


def filled_segments(seg, fillqa, seg_bones):
    """segments filled in the unposed bundle: not measurable, or one of their bones filled by the length gate"""
    gated = set((fillqa or {}).get("gated_bones") or [])
    out = []
    for name, v in seg.items():
        if v[0] not in (0, 1, 2) or (seg_bones and gated.intersection(seg_bones.get(name, ()))):
            out.append(name)
    return out


def decide(spec, seg, part_code, fillqa, checks, reg_info, K=None, seg_bones=None):
    """spec: rs.status specimen dict (specimen_class complete / partial / do_not_use, counts, head_end, reasons);
    seg: {segment: [code, reasons]}; fillqa: fill.make_bundle QA row of this specimen (or None: no bundle); checks:
    dict(tubes, head_roll) (or None: the checks failed); reg_info: dict(status, reasons) of the registration;
    seg_bones: {segment: [bone names]}. Returns the grade dict."""
    K = dict(K_GRADE, **(K or {}))
    cond, why = {}, []
    base = spec["specimen_class"]
    dnu = base == "do_not_use"
    rs_reasons = list(reg_info.get("reasons") or [])
    root_fb = [r for r in rs_reasons if "root placed by" in r]
    if K["ROOT_FALLBACK_IS_DNU"] and root_fb and not dnu:
        dnu = True
        why.append(
            "registration root placed by a fallback (fewer than three usable root-rigid joints): " + "; ".join(root_fb)
        )
    if reg_info.get("status") == "failed" and not dnu:
        dnu = True
        why.append("registration failed")
    n_unmeas = int(spec["n_limb_segments"] - spec["n_measurable"])
    filled = filled_segments(seg, fillqa, seg_bones)
    n_fill = len(filled)
    cond["M1_trunk"] = not spec["trunk_not_measurable"]
    cond["M2_limbs"] = n_fill <= K["MAX_SEG_FILLED"]
    cond["M3_trunk_fill"] = None if fillqa is None else fillqa["n_trunk_filled"] == 0
    cond["M4_edges"] = None if fillqa is None else fillqa["n_abnormal_gt10"] == 0
    zz, zname = None, None
    tubes = (checks or {}).get("tubes")
    if tubes is None:
        cond["M5_antennae"] = None  # the check did not run
    else:
        meas = {"scape": "an_1", "funiculus": "an_2"}
        n_ant = 0
        for name, q in tubes.items():
            kind, side = name.rsplit("_", 1)
            if kind not in meas or seg.get(f"{meas[kind]}_{side}", [4])[0] not in (0, 1, 2):
                continue
            n_ant += 1
            if q.get("zigzag_excess_deg") is None:
                continue
            if zz is None or q["zigzag_excess_deg"] > zz:
                zz, zname = q["zigzag_excess_deg"], name
        if K["ZIGZAG_DEG"] is None or n_ant == 0:
            cond["M5_antennae"] = True  # switched off, or no measurable antenna tube to judge
        else:
            cond["M5_antennae"] = None if zz is None else zz <= K["ZIGZAG_DEG"]
    hr = (checks or {}).get("head_roll") or {}
    roll_flag = None
    if hr.get("best_deg") is not None:
        roll_flag = bool(
            abs(hr["best_deg"]) > K["ROLL_DEG"]
            and (hr.get("contrast") or 0) > K["ROLL_CONTRAST"]
            and not hr.get("at_range_edge")
        )
    if K["ROLL_GATE"]:
        cond["M6_head_roll"] = None if roll_flag is None else not roll_flag
    cond["M7_frame"] = not spec["head_end"].get("roll_doubt")
    cond["M8_native"] = reg_info.get("status") in ("ok", "review") and (
        reg_info.get("forward_err_um") is None or reg_info["forward_err_um"] < 1.0
    )
    text = dict(
        M1_trunk="trunk part not measurable: " + ",".join(spec["trunk_not_measurable"]),
        M2_limbs=f"{n_fill} of {spec['n_limb_segments']} limb segments filled in the unposed mesh ({n_unmeas} not "
        f"measurable, {n_fill - n_unmeas} by the length gate)",
        M3_trunk_fill=f"{0 if fillqa is None else fillqa['n_trunk_filled']} trunk vertices filled in the unposed mesh",
        M4_edges=f"{0 if fillqa is None else fillqa['n_abnormal_gt10']} abnormal stretched edges (8 x the population "
        "median, longer than 10 % of the body length) in the unposed mesh",
        M5_antennae=f"crumpled antenna ({zname}: zigzag {0 if zz is None else zz:.0f} deg)",
        M6_head_roll=f"head may be rolled about its long axis (scan symmetry best at {hr.get('best_deg')} deg, "
        f"contrast {hr.get('contrast')})",
        M7_frame="left / right naming doubted by the joint product's frame check",
        M8_native="registration status or native parameters not usable",
    )
    failed = [k for k, v in cond.items() if v is not True]
    unchecked = [k for k, v in cond.items() if v is None]
    if dnu:
        cls = "do_not_use"
        why = list(spec["reasons"]) + why if base == "do_not_use" else why
    elif not failed:
        cls = "model_grade"
        why = ["every model-grade condition holds"]
    else:
        cls = "partial"
        why = [(f"{LABEL[k]}: could not be checked" if k in unchecked else text[k]) for k in failed]
    return dict(
        specimen_class=cls,
        reasons=why,
        conditions=cond,
        failed=failed,
        unchecked=unchecked,
        n_segments_filled=n_fill,
        n_segments_not_measurable=n_unmeas,
        segments_filled_by_gate=sorted(set(filled) - {n for n, v in seg.items() if v[0] not in (0, 1, 2)}),
        antenna_zigzag_max_deg=zz,
        antenna_zigzag_tube=zname,
        head_roll=(
            dict(
                {k: hr.get(k) for k in ("cost0", "best_deg", "cost_best", "contrast", "at_range_edge")}, flag=roll_flag
            )
            if hr
            else None
        ),
        status_class=base,
        complete=bool(base == "complete"),
        K=K,
    )
