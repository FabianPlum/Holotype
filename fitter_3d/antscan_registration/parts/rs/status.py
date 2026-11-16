"""ONE status definition for an AntScan registration (any route), at vertex, bone, limb-segment, chain and specimen
level. Evidence per limb segment is COVER's (round 2, parts/cover04.py); the decision rules start from
COVER's `decide` (copied below as `decide_segments`, constants passed in as a dict so they can be re-estimated
leave-one-specimen-out) and add: route claims, joint-product recovered joints (codes 2 / 7, part RECOVER),
trunk / head / mandible rules with a scan check, chain level, a specimen class with a head-end gate.

    from fitter_3d.antscan_registration.parts.rs.status import evidence_one, decide_one
    ev = evidence_one(name, envelope_stl, joints_json, limbs_dir, reg)      # expensive part (tube tracing), cacheable
    st = decide_one(ev, K=DEFAULT_K, pop=None, agree=None)                  # cheap; returns every level

Codes (vertex = code of its dominant bone; bone; limb segment):
  0 joint_supported  1 surface_verified  2 recovered  3 review  4 unregistered  5 imputed_mirror  6 imputed_prior
  7 recovered_estimated.   measurable = 0, 1, 2;  registered = 0, 1, 2, 3, 7 (3 and 7 flagged);  5 / 6 only appear in
  the UC1 bundle (filled vertices), never on the posed registration.
Specimen class: complete / partial / do_not_use (definition in decide_specimen).
"""

import json
import time

import numpy as np

from fitter_3d.antscan_registration.parts import cover04 as CV  # COVER cover-0.4

from . import VERSION

STATUS_NAMES = list(CV.STATUS_NAMES)
MEASURABLE = (0, 1, 2)
REGISTERED = (0, 1, 2, 3, 7)
SEG_ORDER = None  # list of the 34 limb segment names in chain order (filled by seg_names())
TRUNK_PARTS = ("b_h", "b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5")
CORE_PARTS = ("b_t",)  # mesosoma: required for anything but do_not_use. The gaster base (dropped
# 2026-11-13) and the head (dropped 2026-11-14) are not: a gaster base placed by
# proportion or a head off the scan does not make the legs unusable (on the one
# development specimen with a lost head all 30 measurable limb segments are right);
# they block 'complete' and the model grade only
COVER_PARTS = {"mesosoma": ("b_t",), "gaster": ("b_a_3", "b_a_4", "b_a_5")}  # parts the coverage rule applies to
HEAD_FLAGS = (
    "head_end_low_confidence",
    "head_end_from_round3_prior",
    "left_right_names_disagree_with_frame",
    "head_end_disagrees_with_round3_frame",
    "ensemble_head_end_disagrees",
)  # round 7: the states of the ensemble disagreed on the head end
ROLL_FLAGS = ("frame_check_disagrees",)  # roll / left-right naming doubt (head end not affected): reported, not gating

# constants of the decision (all relative). Those marked DEV were read off development truth by an earlier round;
# they were re-estimated leave-one-specimen-out for this part.
DEFAULT_K = dict(
    LIMB_TRUST=0.6,  # DEV (COVER): named centreline counts as evidence only with conf >= this, no approaches_disagree
    ON_MIN=CV.ON_MIN,
    OTHER_MAX=CV.OTHER_MAX,
    SEVERE_OWN=CV.SEVERE_OWN,
    JTOL_BL=CV.JTOL_BL,
    TIP_TOL_BL=0.04,  # DEV (leave-one-specimen-out on development truth, 2026-11-13: chosen 0.04 on 20 of 21
    # held-out folds, 0.03 on 1; COVER had 0.02): an end segment's tip marker within 4 % BL of the
    # observed centreline tip (the medial path ends inside the tarsus, the tip marker is a surface vertex)
    ASYM_MAX=CV.ASYM_MAX,
    ASYM_LOOSE=CV.ASYM_LOOSE,
    Q_GAP=CV.Q_GAP,
    TRUNK_DSCAN_BL=0.02,  # trunk / head part: median distance of its vertices to the scan above this share of BL -> review
    TRUNK_UNCOVERED=0.20,  # mesosoma / gaster: share of the scan surface nearest to the part that lies more than
    # TRUNK_COVER_TOL_BL from the registered mesh; above it -> review (the mesh lies on the scan
    # but does not reach all of the part: a truncated, wedge-shaped or collapsed gaster, a
    # mesosoma whose dorsum is not reached). Cut and tolerance are the visual review's
    # (2026-11-14, set after looking at 40 + 46 production overlays); None = off
    TRUNK_COVER_TOL_BL=0.015,
    MANDIBLE_RULE="joint",  # "joint" (COVER: usable own joint) | "review" (SURFACE: always review)
    POP_Z=None,  # population plausibility flag demotes measurable -> review when |robust z| > POP_Z (None: flag only)
    AGREE_BL=None,  # two-registration disagreement demotes measurable -> review above this share of BL (None: flag only)
    HEAD_CHECK="antenna",  # scan check that can confirm a doubtful head end
    KINK_DEG=None,  # round-2 SURFACE kink rule (DEV: 60 deg); None = off
    RMIN_REL=None,  # round-2 SURFACE constriction rule (DEV: 0.5); None = off
)


def tpl():
    return CV.tpl()


def seg_names():
    global SEG_ORDER
    if SEG_ORDER is None:
        SEG_ORDER = [sg["name"] for ch, segs in CV.chains().items() for sg in segs]
    return SEG_ORDER


def seg_type(n):
    return CV.seg_type(n)


# ======================================================================================================================
# evidence (expensive; COVER's segment_evidence + scan distances + route claims + joint-product facts)
# ======================================================================================================================
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def evidence_one(name, envelope, jfile, limbs_dir, reg):
    """All evidence for one registration. `reg` = rs.io.read_registration(...) dict. Returns a json-able dict."""
    from scipy.spatial import cKDTree

    from fitter_3d.antscan_registration.parts.skelreg.scan import load_scan

    t0 = time.time()
    t = tpl()
    jp = CV.load_joint_product(jfile)
    limbs = CV.load_limbs(limbs_dir) if limbs_dir else {}
    ok = np.isfinite(jp["P"]).all(1)
    size = float(np.linalg.norm(jp["P"][ok].max(0) - jp["P"][ok].min(0)))
    scan = load_scan(envelope, size, assume_outer=True)
    V = np.asarray(reg["V"], np.float64)
    BL = jp["BL"]
    M = CV.markers(V)
    ver = CV.tube_verification(scan, M)
    ev = CV.segment_evidence(V, M, jp, limbs, ver, BL, reg_v=reg.get("reg_v"), V_pre=reg.get("V_pre"))
    # scan distance of every template vertex (nearest scan vertex; envelope <= 100k vertices)
    sv = np.asarray(scan.V if hasattr(scan, "V") else scan.v, float)
    dv = cKDTree(sv).query(V, workers=1)[0] / BL
    rps = reg.get("route_part_status")
    d = jp["d"]
    prov = d.get("provenance") or {}
    for n, e in ev.items():
        sg = [s for s in CV.chains()[e["chain"]] if s["name"] == n][0]
        m = np.isin(t.dom, sg["own"])
        e["d_scan_med_bl"] = float(np.median(dv[m])) if m.any() else None
        e["obs_share"] = float((dv[m] <= 0.005).mean()) if m.any() else None
        lim = limbs.get(CV.LIMB_OF[e["chain"]])
        e["lim_conf"] = None if lim is None else float(lim["conf"])
        e["lim_status"] = None if lim is None else lim["status"]
        e["lim_disagree"] = None if lim is None else bool("approaches_disagree" in lim["flags"])
        e["route_part_status"] = None if rps is None else rps[e["bone"]]
        e["prov_b_full"] = str(prov.get(t.names[e["bone"]]) or "null")
        e["prov_c_full"] = str(prov.get(t.names[e["child"]]) or "null")
    # trunk / head / mandible parts: own joint, scan distance
    parts = {}
    for j, pn in enumerate(t.names):
        if pn.startswith(("l_", "an_", "w_")):
            continue
        m = t.dom == j
        parts[pn] = dict(
            usable=bool(np.isfinite(jp["P"][j]).all()),
            conf=float(jp["call"][j]),
            prov=str(prov.get(pn) or "null"),
            d_scan_med_bl=float(np.median(dv[m])) if m.any() else None,
            obs_share=float((dv[m] <= 0.005).mean()) if m.any() else None,
            route_part_status=None if rps is None else rps[j],
            nv=int(m.sum()),
        )
    # head end / gaster volumes of the registered mesh (head-end check)
    vols = {}
    for key, bones in (("head", ("b_h",)), ("gaster", ("b_a_3", "b_a_4", "b_a_5")), ("mesosoma", ("b_t",))):
        vols[key] = _part_volume(V, t, bones)
    flags = [str(f) for f in (d.get("completeness") or {}).get("flags", [])]
    rec = recovered_joints(d)
    info = (d.get("frame") or {}).get("info") or {}
    if not isinstance(info, dict):  # some joint files (oracle / synthetic writers) store frame.info as text
        info = {}
    out = dict(
        name=name,
        version=VERSION,
        BL=BL,
        jfile=jfile,
        limbs_dir=limbs_dir,
        envelope=envelope,
        route_status=reg.get("route_status"),
        src=reg.get("src"),
        src_format=reg.get("src_format"),
        jp_status=str(d.get("status")),
        jp_flags=flags,
        lr_consistent=info.get("lr_consistent"),
        frame_check_deg=info.get("check_angle_to_output_deg"),
        waist_segments=d.get("waist_segments"),
        limbs={k: dict(conf=v["conf"], status=v["status"], flags=v["flags"]) for k, v in limbs.items()},
        segments=ev,
        parts=parts,
        volumes=vols,
        markers=M.tolist(),
        recovered_joints=rec,
        route_reg_share=None if reg.get("reg_v") is None else float(np.mean(reg["reg_v"])),
        seconds_evidence=time.time() - t0,
    )
    out = _jsonable(out)
    add_kink(out, limbs_dir)
    return out


def _polyline_at(X, s, q):
    """points of polyline X (arc s, increasing) at arcs q (copied from the round-2 SURFACE registration)"""
    q = np.clip(q, s[0], s[-1])
    return np.stack([np.interp(q, s, X[:, k]) for k in range(3)], 1)


def _tangent_at(X, s, q, h):
    """unit tangent at arcs q, central difference over +-h (copied from the round-2 SURFACE registration)"""
    a = _polyline_at(X, s, q + h) - _polyline_at(X, s, q - h)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def kink_on_centreline(lim, A, B):
    """round-2 SURFACE's segment diagnostics (copied) on the named centreline between the
    projections of the registration's two end markers A, B: largest bend angle (deg) and smallest radius relative to
    the segment's median radius inside the inner 60 % of the segment. A sharp bend or a constriction inside a segment
    means an articulation lies inside it (a joint put down the neighbouring segment)."""
    X, s, r = lim["X"], lim["s"], lim["r"]
    if len(X) < 4:
        return None, None
    _, ia = lim["kd"].query(A)
    _, ib = lim["kd"].query(B)
    s0, s1 = sorted([float(s[ia]), float(s[ib])])
    if s1 - s0 < 1e-9:
        return None, None
    rr = np.where(np.isfinite(r), r, np.nanmedian(r) if np.isfinite(r).any() else 0.0)
    q = np.linspace(s0 + 0.2 * (s1 - s0), s1 - 0.2 * (s1 - s0), 25)
    rq = np.interp(q, s, rr)
    win = max(1.5 * float(np.median(rq)), 0.05 * (s1 - s0))
    t0 = _tangent_at(X, s, q - win, 0.5 * win)
    t1 = _tangent_at(X, s, q + win, 0.5 * win)
    ang = np.degrees(np.arccos(np.clip((t0 * t1).sum(1), -1, 1)))
    r_all = np.interp(np.linspace(s0, s1, 41), s, rr)
    return float(ang.max()), float(rq.min() / max(np.median(r_all), 1e-12))


def add_kink(E, limbs_dir):
    """adds kink_deg / r_min_rel to every limb segment of an evidence dict (cheap; also for cached evidence)."""
    limbs = CV.load_limbs(limbs_dir) if limbs_dir else {}
    M = np.asarray(E["markers"])
    for n, e in E["segments"].items():
        lim = limbs.get(CV.LIMB_OF[e["chain"]])
        k, r = kink_on_centreline(lim, M[e["bone"]], M[e["child"]]) if lim is not None else (None, None)
        e["kink_deg"], e["r_min_rel"] = k, r
    E["has_kink"] = True
    return E


def _part_volume(V, t, bones):
    """volume of the faces owned by the bones (dominant bone of all three vertices) closed by a fan to their centroid."""
    ids = [t.idx[b] for b in bones]
    fm = np.isin(t.dom[t.F], ids).all(1)
    if not fm.any():
        return None
    F = t.F[fm]
    c = V[np.unique(F)].mean(0)
    a, b, cc = V[F[:, 0]] - c, V[F[:, 1]] - c, V[F[:, 2]] - c
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, cc)).sum()) / 6.0)


# ======================================================================================================================
# decision: limb segments (COVER's decide, copied, constants from K)
# ======================================================================================================================
def _trusted(lim, K):
    return lim is not None and lim["conf"] >= K["LIMB_TRUST"] and "approaches_disagree" not in (lim.get("flags") or [])


def decide_segments(E, K):
    """COVER cover-0.4 `decide` (copied; only change: constants and the 'trusted' test come from K, and the limb
    record is the json form; joint support is recomputed with K['JTOL_BL'] from the stored marker-joint distances).
    Returns {segment: [code, reasons]}."""
    J = K["JTOL_BL"]
    ev = {
        n: dict(
            e,
            jsup=bool(
                e["usable_b"]
                and e["usable_c"]
                and e["dmark_b"] is not None
                and e["dmark_c"] is not None
                and e["dmark_b"] <= J
                and e["dmark_c"] <= J
            ),
        )
        for n, e in E["segments"].items()
    }
    limbs = E["limbs"]
    CH = CV.chains()
    out = {}
    for ch, segs in CH.items():
        lim = limbs.get(CV.LIMB_OF[ch])
        prev_reg = True
        for sg in segs:
            n = sg["name"]
            e = ev[n]
            why = []
            dup_js = [d for d in e["dup_with"] if ev[d]["jsup"]]
            if e["jsup"]:
                if dup_js:
                    why.append("duplicate with joint-supported " + ",".join(sorted(dup_js)))
                if e["tube"] == "fail":
                    why.append("no tube / limb material along the segment")
                if e["on_own"] is not None and e["on_own"] < K["SEVERE_OWN"] and e["on_other"] > K["OTHER_MAX"]:
                    why.append(f"its surface lies on another limb ({','.join(sorted(e['on_other_who']))})")
                code = 3 if why else 0
                if not why:
                    why = ["joint-supported"]
            else:
                if e["dup_with"]:
                    why.append("duplicate with " + ",".join(sorted(e["dup_with"])))
                if e["tube"] == "duplicate":
                    why.append("duplicate tube claim")
                if e["on_other"] > K["OTHER_MAX"]:
                    why.append(f"on another limb ({','.join(sorted(e['on_other_who']))})")
                if lim is None:
                    why.append("no named centreline")
                elif not _trusted(lim, K):
                    why.append(
                        f"its named centreline is not trusted (conf {lim['conf']:.2f}"
                        + (", approaches disagree)" if "approaches_disagree" in lim["flags"] else ")")
                    )
                elif e["on_own"] is None:
                    why.append("beyond the end of its partial centreline")
                elif e["on_own"] < K["ON_MIN"]:
                    why.append(f"off its own centreline ({e['on_own']:.2f})")
                if sg["end"] and lim is not None and lim["status"] == "partial":
                    why.append("tip not observed (partial centreline)")
                if lim is not None and e["own_defined"] < 0.5:
                    why.append(f"only {e['own_defined']:.2f} of it on the defined centreline")
                if not e["chart_ok"]:
                    why.append("no self-consistent tube" if e["tube"] != "fail" else "no tube / limb material")
                if why:
                    code = 4
                else:
                    code = 1 if prev_reg else 3
                    why = (
                        ["surface-verified (traced tube on its own centreline)"]
                        if prev_reg
                        else ["surface-verified but detached (proximal segment not registered)"]
                    )
                    if code == 1 and sg is segs[0]:
                        okp = (
                            e["dmark_any_b"] is not None
                            and e["dmark_any_b"] <= K["JTOL_BL"]
                            and (e["prov_b"] == "scan_feature" or e["usable_b"])
                        )
                        if not okp:
                            code, why = (
                                3,
                                [
                                    "on its own limb, but its base (coxa base / antennal socket) is not an "
                                    "observed feature"
                                ],
                            )
                    if code == 1:
                        if sg["end"]:
                            okd = bool(e.get("tip_observed")) and (e.get("tip_dist") or 9) <= K["TIP_TOL_BL"]
                            if not okd:
                                code, why = 3, ["on its own limb, but its tip is not at the observed centreline tip"]
                        else:
                            okd = (
                                e["dmark_any_c"] is not None
                                and e["dmark_any_c"] <= K["JTOL_BL"]
                                and (e["prov_c"] == "scan_feature" or e["usable_c"])
                            )
                            if not okd:
                                code, why = (
                                    3,
                                    ["on its own limb, but its distal articulation is not an observed feature"],
                                )
            out[n] = [code, why]
            prev_reg = code in (0, 1, 2)
    # pass 2: left / right lengths
    for ch, segs in CH.items():
        pc = CV.PARTNER[ch]
        for sg, sp in zip(segs, CH[pc]):
            n, m = sg["name"], sp["name"]
            a = ev[n]["asym"]
            amax = K["ASYM_LOOSE"] if seg_type(n) in ("co", "ta") else K["ASYM_MAX"]
            if a <= amax:
                continue
            cn, cm = out[n][0], out[m][0]
            msg = f"left/right length {np.exp(a):.2f}x"
            if cn == 1 and cm == 0:
                out[n] = [4, [msg + " vs joint-supported partner"]]
            elif cn == 0 and cm == 0 and n < m:
                qn = min(ev[n]["conf_b"], ev[n]["conf_c"])
                qm = min(ev[m]["conf_b"], ev[m]["conf_c"])
                if qn < qm - K["Q_GAP"]:
                    _flag_weak(out, ev, CH[ch], n, msg + f" (weaker joints {qn:.2f} vs {qm:.2f})")
                elif qm < qn - K["Q_GAP"]:
                    _flag_weak(out, ev, CH[pc], m, msg + f" (weaker joints {qm:.2f} vs {qn:.2f})")
                else:
                    out[n] = [3, [msg + " (undecided side)"]]
                    out[m] = [3, [msg + " (undecided side)"]]
    # round-2 SURFACE kink / constriction rule (off by default; K['KINK_DEG'] / K['RMIN_REL'] set it), for the segment
    # types SURFACE applied it to (trochanter+femur, tibia, basitarsus, scape)
    if K.get("KINK_DEG") is not None or K.get("RMIN_REL") is not None:
        for n, e in ev.items():
            if out[n][0] not in MEASURABLE or seg_type(n) not in ("tr", "fe", "ti", "an_1"):
                continue
            w = []
            if K.get("KINK_DEG") is not None and e.get("kink_deg") is not None and e["kink_deg"] > K["KINK_DEG"]:
                w.append(f"its centreline bends by {e['kink_deg']:.0f} deg inside the segment")
            if K.get("RMIN_REL") is not None and e.get("r_min_rel") is not None and e["r_min_rel"] < K["RMIN_REL"]:
                w.append(f"its centreline narrows to {e['r_min_rel']:.2f} of its radius inside the segment")
            if w:
                out[n] = [3, w + out[n][1]]
    _detach(out, ev)
    E["_jsup"] = {n: e["jsup"] for n, e in ev.items()}
    return out


def _flag_weak(out, ev, segs, n, msg):
    out[n] = [3, [msg]]
    e = ev[n]
    weak = e["bone"] if e["conf_b"] <= e["conf_c"] else e["child"]
    for sg in segs:
        if sg["name"] != n and (sg["bone"] == weak or sg["child"] == weak) and out[sg["name"]][0] == 0:
            out[sg["name"]] = [3, [f"shares the weaker joint with {n}: " + msg]]


def _detach(out, ev):
    for ch, segs in CV.chains().items():
        prev = True
        for sg in segs:
            n = sg["name"]
            if out[n][0] == 1 and not prev:
                out[n] = [3, ["surface-verified but detached (proximal segment not registered)"]]
            prev = out[n][0] in (0, 1, 2) or (out[n][0] == 3 and ev[n]["jsup"])


def recovered_joints(jdict):
    """{joint: {kind, verified_by}} from an augmented joint product (part RECOVER, recover-0.3 contract: provenance
    'recovered (<kind>; verified: ...)' and a `recovery.joints` block)."""
    rec = ((jdict.get("recovery") or {}).get("joints")) or {}
    out = {j: dict(kind=v.get("kind"), verified_by=list(v.get("verified_by") or [])) for j, v in rec.items()}
    for j, pv in (jdict.get("provenance") or {}).items():
        if str(pv).startswith("recovered") and j not in out:
            out[j] = dict(
                kind=str(pv)[11:].split(";")[0],
                verified_by=[x.strip() for x in str(pv).split("verified:")[-1].rstrip(")").split(",")]
                if "verified:" in str(pv)
                else [],
            )
    return out


def recovered_is_observed(r):
    """a recovered joint is an OBSERVED feature when its acceptance includes a scan feature at the joint (radius
    minimum / step, fold, traced tip: RECOVER's 'scan feature'); joints pinned only by a partner or population length,
    a socket prediction or a second placement are ESTIMATED (their segment length would be partly circular)."""
    return "scan feature" in (r.get("verified_by") or [])


def apply_recovered(out, E, K):
    """Joints the part RECOVER added to the joint product: a segment that is joint-supported (code 0 after the
    rules) and has at least one recovered end joint is 2 (recovered: every recovered end joint is an observed scan
    feature) or 7 (recovered_estimated: flagged, not measurable)."""
    rec = E.get("recovered_joints") or {}
    if not rec:
        return
    t = tpl()
    for n, (c, why) in list(out.items()):
        e = E["segments"][n]
        ends = [t.names[e["bone"]], t.names[e["child"]]]
        rj = [j for j in ends if j in rec]
        if c != 0 or not rj:
            continue
        obs = all(recovered_is_observed(rec[j]) for j in rj)
        desc = "; ".join(f"{j}: {rec[j]['kind']}, verified by {', '.join(rec[j]['verified_by'])}" for j in rj)
        out[n] = [
            2 if obs else 7,
            [
                (
                    "recovered (scan feature at every recovered joint): "
                    if obs
                    else "recovered_estimated (a recovered joint pinned by lengths / sockets only): "
                )
                + desc
            ],
        ]


def apply_route_claims(out, E, K):
    """The route itself says it did not register the segment (route part status 'unregistered' / 'failed'): a
    joint-supported code is kept only with tube evidence; otherwise the route claim wins (code 4)."""
    ev = E["segments"]
    for n, (c, why) in list(out.items()):
        rp = ev[n].get("route_part_status")
        if rp in ("unregistered", "failed") and c in (0, 1):
            if not (ev[n]["chart_ok"] and c == 0):
                out[n] = [4, [f"route reports the part {rp}"] + why]


def apply_extra_flags(out, E, K, pop=None, agree=None):
    """Batch-level signals: population plausibility (robust z of log segment length / BL, leave-one-out) and the
    disagreement between two independent registrations. Flags are always recorded; they demote a measurable segment
    to review only when K['POP_Z'] / K['AGREE_BL'] is set."""
    flags = {}
    for n, (c, why) in list(out.items()):
        f = []
        if pop is not None and pop.get(n) is not None and np.isfinite(pop[n]):
            if K["POP_Z"] is not None and abs(pop[n]) > K["POP_Z"]:
                f.append(f"segment proportion is a population outlier (robust z {pop[n]:+.1f})")
        if agree is not None and agree.get(n) is not None and np.isfinite(agree[n]):
            if K["AGREE_BL"] is not None and agree[n] > K["AGREE_BL"]:
                f.append(f"two registrations disagree ({agree[n] * 100:.1f} % BL)")
        if f:
            flags[n] = f
            if c in MEASURABLE:
                out[n] = [3, f + why]
    return flags


# ======================================================================================================================
# bones, vertices, chains, specimen
# ======================================================================================================================
def decide_parts(seg, E, K):
    """Per-bone code + reasons for all 55 bones. Limb bones follow their segment (leaf pt / an_3 with it); trunk and
    head: 0 when the own joint is usable (conf >= 0.3), the part lies on the scan (median vertex distance <=
    TRUNK_DSCAN_BL) and, for mesosoma and gaster, the scan surface nearest to the part is reached by the mesh
    (`scan_uncovered` <= TRUNK_UNCOVERED, when that evidence is there), else 3; mandibles: K['MANDIBLE_RULE']; wings
    (no vertices): 0. Returns code, reasons and the set of parts that are 3 for the coverage rule alone."""
    t = tpl()
    code = np.zeros(t.nJ, np.int8)
    why = {}
    cover_only = set()
    cover_parts = {b for bs in COVER_PARTS.values() for b in bs}
    for ch, segs in CV.chains().items():
        for sg in segs:
            c, w = seg[sg["name"]]
            for b in sg["own"]:
                code[b], why[t.names[b]] = c, w
    for pn, p in E["parts"].items():
        j = t.idx[pn]
        w = []
        if not p["usable"]:
            w.append("no usable joint")
        if p["d_scan_med_bl"] is not None and p["d_scan_med_bl"] > K["TRUNK_DSCAN_BL"]:
            w.append(f"part off the scan (median {p['d_scan_med_bl'] * 100:.1f} % BL)")
        if p.get("route_part_status") in ("unregistered", "failed"):
            w.append(f"route reports the part {p['route_part_status']}")
        if pn.startswith("ma_"):
            if K["MANDIBLE_RULE"] == "review":
                w.append("mandible: template cannot reach long mandibles (always review)")
        unc = p.get("scan_uncovered")
        if (
            pn in cover_parts
            and unc is not None
            and K.get("TRUNK_UNCOVERED") is not None
            and unc > K["TRUNK_UNCOVERED"]
        ):
            if not w:
                cover_only.add(pn)
            w.append(
                f"scan surface of the part not reached by the mesh ({unc * 100:.0f} % of it more than "
                f"{K['TRUNK_COVER_TOL_BL'] * 100:.1f} % BL away)"
            )
        if p.get("route_part_status") in ("unregistered", "failed") and not p["usable"]:
            code[j] = 4
        else:
            code[j] = 3 if w else 0
        why[pn] = w or ["joint-supported"]
    for j, pn in enumerate(t.names):
        if pn.startswith("w_"):
            code[j], why[pn] = 0, ["wing (no vertices)"]
    return code, why, cover_only


def chain_level(seg):
    out = {}
    for ch, segs in CV.chains().items():
        cs = [seg[s["name"]][0] for s in segs]
        meas = [c in MEASURABLE for c in cs]
        if all(meas):
            lev = "measurable"
        elif any(meas):
            lev = "partly_measurable"
        elif any(c in (3, 7) for c in cs):
            lev = "flagged"
        else:
            lev = "unregistered"
        # measurable proximal prefix (how far down the chain lengths can be read)
        k = 0
        while k < len(cs) and cs[k] in MEASURABLE:
            k += 1
        out[ch] = dict(level=lev, codes=cs, n_measurable=int(sum(meas)), measurable_prefix=k, n=len(cs))
    return out


def head_end_check(E, seg, K):
    """Scan check of the head end (used only when the joint product doubts it). 'antenna': confirmed when BOTH antenna
    chains are measurable over both segments with a self-consistent traced tube on the REGISTERED markers: the scan
    carries two antenna tubes starting at the sockets of the end the registration calls the head. (v0.1 also required
    the registered head to be smaller than the gaster; dropped: 3 of 8 doubtful production specimens have a larger
    head, at least one of them with a visibly correct head end.) The head / gaster volumes are reported."""
    ev = E["segments"]
    ant_ok = []
    for side in "lr":
        ok = all(seg[f"{a}_{side}"][0] in MEASURABLE and ev[f"{a}_{side}"]["chart_ok"] for a in ("an_1", "an_2"))
        ant_ok.append(ok)
    vh, vg = (E.get("volumes") or {}).get("head"), (E.get("volumes") or {}).get("gaster")
    res = dict(
        antenna_chains_verified=int(sum(ant_ok)),
        head_volume=vh,
        gaster_volume=vg,
        head_to_gaster_volume=(vh / vg) if (vh and vg) else None,
    )
    res["confirmed"] = bool(all(ant_ok)) if K["HEAD_CHECK"] == "antenna" else False
    return res


def cls_dnu(why):
    return any(not w.startswith("left/right naming") for w in why)


def decide_specimen(E, seg, code, chains, K, cover_only=()):
    """Specimen class (consumer selection without reading part lists):
    do_not_use   no usable registration / joint product not ok / head end doubtful and not confirmed by the scan
                 check / the mesosoma (b_t) not registered on the scan (flagged by the coverage rule alone does
                 not count: `cover_only`)
    complete     not do_not_use, every trunk part (b_h b_t b_a_1..b_a_5) measurable, all 34 leg and antenna
                 segments measurable (mandibles are reported, not required: a limit of the template)
    partial      everything else: use only measures whose parts are measurable (status per measure in the csv)."""
    t = tpl()
    why = []
    flags = E.get("jp_flags") or []
    doubt = [f for f in flags if f.startswith(HEAD_FLAGS)]
    if E.get("lr_consistent") is False and not any(f.startswith("left_right_names") for f in doubt):
        doubt.append("left_right_names_disagree_with_frame (lr_consistent false)")
    roll = [f for f in flags if f.startswith(ROLL_FLAGS)]
    he = dict(doubtful=bool(doubt), reasons=doubt, roll_doubt=roll)
    if doubt:
        he.update(head_end_check(E, seg, K))
        he["state"] = "doubtful_confirmed_by_scan" if he["confirmed"] else "doubtful"
    else:
        he["state"] = "ok"
    nseg = len(seg)
    nmeas = sum(1 for v in seg.values() if v[0] in MEASURABLE)
    nrev = sum(1 for v in seg.values() if v[0] in (3, 7))
    nunr = sum(1 for v in seg.values() if v[0] in (4, 5, 6))
    core_bad = [p for p in CORE_PARTS if code[t.idx[p]] not in MEASURABLE and p not in cover_only]
    trunk_bad = [p for p in TRUNK_PARTS if code[t.idx[p]] not in MEASURABLE]
    rs = str(E.get("route_status") or "ok")
    if rs == "failed":
        why.append("route reports the registration failed")
    if E.get("jp_status") not in ("ok", None, "None"):
        why.append(f"joint product status {E.get('jp_status')}")
    if he["state"] == "doubtful":
        why.append("head end doubtful and not confirmed by the scan check: " + "; ".join(doubt))
    if core_bad:
        why.append("core part not measurable: " + ",".join(core_bad))
    if roll and not why:
        why.append(
            "left/right naming may be swapped (frame check: " + "; ".join(roll) + "): per-side values may be exchanged"
        )
    if cls_dnu(why):
        cls = "do_not_use"
    elif not trunk_bad and nmeas == nseg:
        cls = "complete"
        why = why + ["every trunk part and all 34 limb segments measurable"]
    else:
        cls = "partial"
        if trunk_bad:
            why.append("trunk part not measurable: " + ",".join(trunk_bad))
        why.append(f"{nmeas} of {nseg} limb segments measurable ({nrev} review, {nunr} unregistered)")
    mand = [code[t.idx[m]] in MEASURABLE for m in ("ma_l", "ma_r")]
    return dict(
        specimen_class=cls,
        reasons=why,
        head_end=he,
        n_limb_segments=nseg,
        n_measurable=nmeas,
        n_review=nrev,
        n_unregistered=nunr,
        trunk_not_measurable=trunk_bad,
        mandibles_measurable=int(sum(mand)),
        chains_measurable=int(sum(1 for c in chains.values() if c["level"] == "measurable")),
        legs_measurable=int(sum(1 for k, c in chains.items() if k.startswith("l_") and c["level"] == "measurable")),
        antennae_measurable=int(
            sum(1 for k, c in chains.items() if k.startswith("an_") and c["level"] == "measurable")
        ),
    )


def decide_one(E, K=None, pop=None, agree=None):
    """Every level from the evidence. Returns dict: segments {name: [code, reasons]}, part_code (55,), part_reasons,
    vertex_status (10235,), part_status (55,) ok/review/unregistered, chains, specimen, extra_flags."""
    K = dict(DEFAULT_K, **(K or {}))
    seg = decide_segments(E, K)
    apply_route_claims(seg, E, K)
    apply_recovered(seg, E, K)
    extra = apply_extra_flags(seg, E, K, pop=pop, agree=agree)
    _detach(seg, {n: {"jsup": v} for n, v in E["_jsup"].items()})
    code, why, cover_only = decide_parts(seg, E, K)
    t = tpl()
    vst = code[t.dom].astype(np.int8)
    pst = np.array(["ok" if c in MEASURABLE else ("review" if c in (3, 7) else "unregistered") for c in code])
    ch = chain_level(seg)
    sp = decide_specimen(E, seg, code, ch, K, cover_only)
    return dict(
        segments=seg,
        part_code=code,
        part_reasons=why,
        vertex_status=vst,
        part_status=pst,
        chains=ch,
        specimen=sp,
        extra_flags=extra,
        K=K,
    )


def save_evidence(E, path):
    json.dump(E, open(path, "w"))


def load_evidence(path):
    return json.load(open(path))
