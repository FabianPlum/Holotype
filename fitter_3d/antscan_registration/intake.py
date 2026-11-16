"""Joint intake: which joints of a joint product are usable by the registration.

Every part of the pipeline (GEOM, RECOVER, STATUS, the SKEL-REG modules) holds the rule "a joint with confidence
>= 0.3 is usable" as a constant, and GEOM uses the confidence as the joint's weight. The intake therefore writes an
intake product in the same contract whose `confidence` already is the value the registration should use; the input
values stay under `confidence_input`, the rule and every joint it touched under `intake`. Positions are never touched.

Single-state products (round 6; round 7 with one state): unchanged, cut 0.3.

Products of the state ensemble (round 7 K4): the ensemble multiplies the median confidence of the states by
g(x) = 1 / (1 + (x / SPREAD_SCALE)^2), x = the LARGEST distance of a state's placement from the picked one (% of body
length). One deviating state out of four lowers a joint the other three agree on (the picked position is their medoid
and does not move). The MAJORITY spread is the distance within which a strict majority of the placing states lie; a
two-against-two split keeps its low confidence under it.
  trunk joints  (neck, waist, gaster, head front; `majority='trunk'`, the default): the confidence is recomputed with
                the majority spread from the per-state products (`joint_states`: <dir>/<state>/final/proposals/<name>/
                <name>_joints.json). No later step can bring a trunk joint back, and without the neck a specimen has no
                head. Without the per-state products the trunk joints keep the ensemble's value. Per-state products
                are used only when they reproduce the product's picked positions (`states_match`).
  limb joints   keep the ensemble's value and the cut 0.3: the recovery step brings the right ones back through its
                independent check. Raising them here as well (`majority='all'`, from `ens.spread_med_pct`) or lowering
                the cut offers more joints directly, but on the development set as many of the additional measurable
                segments are wrong as right.
The cut is 0.3 for every product (`cut='auto'`); another cut can be given and is applied by promotion to / demotion
below the constant 0.3.

Ties (`intake.ties`): joints whose position was picked between equally many states (an even number of placed states
with the majority spread above TIE_SPREAD_PCT, or a limb whose agreeing cluster is exactly half of the states).
`tie_policy`:
  'far' (default)  a limb joint at or above the cut is used directly only when a strict majority of the states that
                   placed it lie within TIE_FAR_PCT of the picked position (per-state products needed; the same
                   "majority spread" as for the trunk joints, here over ALL placing states, also those of the other
                   limb cluster). Otherwise the picked position is one of two candidates more than TIE_FAR_PCT apart
                   and sub-voxel noise chose between them: the joint is demoted (`intake.tie_far`) and only the
                   recovery step's independent check can bring it back. TIE_FAR_PCT is the project's error threshold
                   (4 % of body length): two candidates farther apart cannot both be right. Ties closer together are
                   used as they come (on the development set they are as accurate as other joints). Without per-state
                   products the rule cannot be applied and the intake block says so.
  'use'            every tie is left to the cut;
  'recover'        every tie is demoted.
"""

import copy
import json
import os

import numpy as np

from . import USABLE_CONF

USABLE = USABLE_CONF  # the constant every consumer holds
CUT = 0.30
SPREAD_SCALE = 2.0  # the ensemble's constant (g_spread of the joint method's ensemble; reproduced on its products)
LIMB_BASE = 0.5  # the ensemble's limb factor base (limbs.json params.limb_base)
TIE_SPREAD_PCT = 1.0
TIE_FAR_PCT = 4.0  # % of body length: the project's threshold for a wrong joint
DEMOTED = 0.299
REQUIRED_KEYS = ("joints_scan", "confidence", "frame", "body_length_um")


def g_spread(x, scale=SPREAD_SCALE):
    return 1.0 / (1.0 + (float(x) / scale) ** 2)


def product_type(d):
    st = (d.get("ensemble") or {}).get("states") or []
    return f"ensemble{len(st)}" if len(st) >= 2 else "single"


STATES_MATCH_MIN = 0.95  # share of an ensemble product's placed joints that must be one state's own position


def states_match(d, states):
    """Share of the product's joints with an `ens` detail whose picked position is exactly one state's position. The
    ensemble picks a medoid, so with the per-state products it was built from the share is 1 (6448 of
    6448 joints); a folder of other products is told apart without truth."""
    js = d.get("joints_scan") or {}
    BL = float(d.get("body_length_um") or 0.0) or 1.0
    hit = n = 0
    for j, det in (d.get("joint_detail") or {}).items():
        x = js.get(j)
        if x is None or not (det or {}).get("ens"):
            continue
        P = [s.get("joints_scan", {}).get(j) for s in states]
        P = np.array([p for p in P if p is not None and np.all(np.isfinite(p))], float)
        n += 1
        if len(P) and np.linalg.norm(P - np.asarray(x, float), axis=1).min() / BL < 1e-5:
            hit += 1
    return hit / n if n else 0.0


def load_states(joint_states, name, d):
    """per-state products of an ensemble product, or None. joint_states: the joint method's `states` folder
    (<state>/final/proposals/<name>/<name>_joints.json), or a list of candidate folders (the first that holds every
    kept state of this product AND reproduces its picked positions is used: `states_match`)."""
    cands = [joint_states] if isinstance(joint_states, str) else list(joint_states or [])
    ens = d.get("ensemble") or {}
    kept = ens.get("kept") or ens.get("states") or []
    for sd in cands:
        if not sd or not kept or not os.path.isdir(str(sd)):
            continue
        out = []
        for k in kept:
            p = f"{sd}/{k}/final/proposals/{name}/{name}_joints.json"
            try:
                out.append(json.load(open(p)))
            except Exception:
                out = None
                break
        if out and states_match(d, out) >= STATES_MATCH_MIN:
            return out
    return None


def check_product(d):
    """None, or the reason a joint product cannot be used at all (a missing block of the contract)."""
    if not isinstance(d, dict):
        return "joint product is not a json object"
    miss = [k for k in REQUIRED_KEYS if not d.get(k)]
    if miss:
        return "joint product without " + ", ".join(miss)
    fr = d["frame"]
    if not isinstance(fr, dict) or "rotation" not in fr or "origin_scan" not in fr:
        return "joint product without frame.rotation / frame.origin_scan"
    try:
        if not float(d["body_length_um"]) > 0:
            return "joint product with a body length of 0"
    except (TypeError, ValueError):
        return "joint product with an unreadable body length"
    return None


def parse_cut(cut):
    """the confidence cut as a number in (0, 1]; 'auto' / None = CUT. Raises ValueError otherwise."""
    if cut in (None, "auto"):
        return CUT
    try:
        c = float(cut)
    except (TypeError, ValueError):
        raise ValueError(f"confidence cut must be 'auto' or a number in (0, 1], not {cut!r}") from None
    if not (0.0 < c <= 1.0):
        raise ValueError(f"confidence cut must lie in (0, 1], not {c}")
    return c


def state_spread(d, states, j):
    """majority spread of joint j over ALL states that placed it (% of body length): the distance from the picked
    position within which a strict majority of the placing states lie; None with fewer than 3 placements."""
    x = (d.get("joints_scan") or {}).get(j)
    BL = float(d.get("body_length_um") or 0.0)
    if x is None or BL <= 0 or not states:
        return None
    P = [s.get("joints_scan", {}).get(j) for s in states]
    P = np.array([p for p in P if p is not None and np.all(np.isfinite(p))], float)
    if len(P) < 3:
        return None
    dk = np.sort(np.linalg.norm(P - np.asarray(x, float), axis=1)) / BL * 100.0
    return float(dk[len(P) // 2])


def majority_confidence(d, states=None):
    """{joint: dict(confidence, input, spread_max, spread_majority, source)} for the joints whose confidence the
    majority rule raises. Formula of the ensemble (verified on the shipped products to its rounding):
    limb c = conf_states_median * (LIMB_BASE + (1 - LIMB_BASE) * limb_state_frac) * placed_share * g(spread),
    trunk c = conf_states_median * n_placed / n_states * g(spread)."""
    out = {}
    js, cf = d.get("joints_scan") or {}, d.get("confidence") or {}
    BL = float(d.get("body_length_um") or 0.0)
    for j, det in (d.get("joint_detail") or {}).items():
        e = (det or {}).get("ens") or {}
        x, c, sp = js.get(j), cf.get(j), e.get("spread_pct")
        if x is None or c is None or sp is None or e.get("conf_states_median") is None:
            continue
        spm, src = None, None
        if "limb_state_frac" in e:
            base = (
                e["conf_states_median"]
                * (LIMB_BASE + (1.0 - LIMB_BASE) * float(e.get("limb_state_frac") or 0.0))
                * float(e.get("placed_share") or 0.0)
            )
            if (e.get("n_placed") or 0) >= 3 and e.get("spread_med_pct") is not None:
                spm, src = float(e["spread_med_pct"]), "ens.spread_med_pct"
        else:
            base = e["conf_states_median"] * float(e.get("n_placed") or 0) / max(float(e.get("n_states") or 1), 1.0)
            if states is not None and BL > 0:
                P = [s.get("joints_scan", {}).get(j) for s in states]
                P = np.array([p for p in P if p is not None and np.all(np.isfinite(p))], float)
                if len(P) >= 3:
                    spm, src = state_spread(d, states, j), "per-state products"
        if spm is None or spm >= float(sp):
            continue
        new = round(min(1.0, base * g_spread(spm)), 3)
        if new > float(c) + 1e-9:
            out[j] = dict(
                confidence=new, input=float(c), spread_max=float(sp), spread_majority=round(spm, 3), source=src
            )
    return out


def ties(d, spread_pct=TIE_SPREAD_PCT):
    """{joint: 'joint' | 'limb' | 'joint+limb'} from the product's own ens block (no per-state files needed)."""
    out = {}
    js = d.get("joints_scan") or {}
    for j, det in (d.get("joint_detail") or {}).items():
        e = (det or {}).get("ens") or {}
        npl, ncl, nst = e.get("n_placed"), e.get("n_cluster"), e.get("n_states")
        if not npl or js.get(j) is None:
            continue
        a = npl % 2 == 0 and float(e.get("spread_med_pct") or 0.0) > spread_pct
        b = bool(nst) and nst >= 2 and ncl is not None and 2 * ncl == nst
        if a or b:
            out[j] = "+".join(k for k, v in (("joint", a), ("limb", b)) if v)
    return out


def intake(d, cut="auto", tie_policy="far", states=None, majority="trunk"):
    """joint product dict -> (intake product dict, intake block). The input dict is not modified.
    states: per-state products (load_states) of an ensemble product, or None.
    majority: 'trunk' (default) | 'all' (limb joints too) | 'none' (the product's confidences as given).
    tie_policy: 'far' (default) | 'use' | 'recover' (module docstring)."""
    majority = {True: "all", False: "none", None: "none"}.get(majority, majority)
    if tie_policy not in ("far", "use", "recover"):
        raise ValueError(f"tie_policy must be far, use or recover, not {tie_policy!r}")
    c0 = parse_cut(cut)
    out = copy.deepcopy(d)
    conf_in = dict(d.get("confidence") or {})
    conf = dict(conf_in)
    ptype = product_type(d)
    maj = majority_confidence(d, states) if (majority != "none" and ptype.startswith("ensemble")) else {}
    if majority == "trunk":
        maj = {j: q for j, q in maj.items() if q["source"] == "per-state products"}
    for j, q in maj.items():
        conf[j] = q["confidence"]
    tie = ties(d)
    promoted, demoted, tie_demoted = [], [], []
    for j, c in list(conf.items()):
        if c is None:
            continue
        c = float(c)
        if c0 <= c < USABLE:
            conf[j] = USABLE
            promoted.append(j)
        elif USABLE <= c < c0:
            conf[j] = DEMOTED
            demoted.append(j)
    if tie_policy == "recover":
        for j in tie:
            if conf.get(j) is not None and conf[j] >= USABLE:
                conf[j] = DEMOTED
                tie_demoted.append(j)
    tie_far, far_rule = {}, None
    if tie_policy == "far" and ptype.startswith("ensemble"):
        if states is None:
            far_rule = "not available (no per-state products that reproduce this product): ties used as they come"
        else:
            far_rule = f"limb joints without a majority of the placing states within {TIE_FAR_PCT:g} % BL are demoted"
            for j, det in (d.get("joint_detail") or {}).items():
                e = (det or {}).get("ens") or {}
                if "limb_state_frac" not in e or conf.get(j) is None or conf[j] < USABLE:
                    continue
                sp = state_spread(d, states, j)
                if sp is not None and sp > TIE_FAR_PCT:
                    tie_far[j] = dict(spread_majority=round(sp, 3), confidence=float(conf[j]))
                    conf[j] = DEMOTED
    js = out.get("joints_scan") or {}
    raised = sorted(j for j, q in maj.items() if q["input"] < USABLE <= (conf.get(j) or 0.0))
    out["confidence"] = conf
    out["confidence_input"] = conf_in
    block = dict(
        version="intake-1.3",
        product_type=ptype,
        cut=c0,
        usable_constant=USABLE,
        confidence_rule=(
            {
                "trunk": "majority spread for trunk joints, limb joints as given",
                "all": "majority spread for trunk and limb joints",
            }[majority]
            if (majority != "none" and ptype.startswith("ensemble"))
            else "as given"
        ),
        trunk_majority=(
            "per-state products"
            if states is not None
            else "not available (no per-state products that reproduce this product): trunk joints as given"
        )
        if (majority != "none" and ptype.startswith("ensemble"))
        else None,
        majority=dict(sorted(maj.items())),
        raised_to_usable=raised,
        tie_policy=tie_policy,
        promoted=sorted(promoted),
        demoted=sorted(demoted),
        ties=dict(sorted(tie.items())),
        tie_demoted=sorted(tie_demoted),
        tie_far_rule=far_rule,
        tie_far=dict(sorted(tie_far.items())),
        n_usable=sum(1 for j, c in conf.items() if c is not None and c >= USABLE and js.get(j) is not None),
        n_usable_input=sum(1 for j, c in conf_in.items() if c is not None and c >= USABLE and js.get(j) is not None),
    )
    out["intake"] = block
    return out, block
