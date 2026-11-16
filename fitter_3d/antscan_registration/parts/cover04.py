"""COVER: per-part and per-vertex STATUS, RECOVERY and IMPUTATION for an AntScan registration of the OmniAnt25
template, from scan evidence only. Route-agnostic: it takes a joint product plus a registration made by any route
(fitter_3d stage output collected to the scan frame, SKEL-REG, ...) and returns statuses, recovered parts and imputed
parts.

    from fitter_3d.antscan_registration.parts.cover04 import run_one
    res = run_one(name, envelope_stl, joints_json, limbs_dir, V_reg, V_pre=None, reg_v=None, opts=None)

Inputs (all in the SCAN frame, micrometres):
  joints_json  round-6 joint product (joint-product contract + the round-6 extras): joints_scan, confidence, frame, body_length_um
  limbs_dir    <dir>/limbs.json + limbs.npz of the same product (named limb centrelines L1..R3, AL, AR; tip first)
  V_reg        (10235, 3) registered template vertices (the route's final mesh), template vertex order
  V_pre        optional (10235, 3): the route's mesh before its free-deformation stage (evidence of how far the free
               stage moved a part; not required)
  reg_v        optional (10235,) bool: what the route itself claims as registered (SKEL-REG: vertex_status != 2)

Nothing here reads truth, oracle limbs, names or taxonomy; no AI or network calls; numpy / scipy / trimesh
(embree) only, CPU.

Status codes (per template vertex, int8; per part = per bone, string):
  0 joint_supported   both joints of the segment are usable input joints (conf >= CONF_MIN), the registration's
                      virtual markers sit on them, and the scan does not contradict (tube or limb-like material)
  1 surface_verified  not joint-supported, but the registered segment is a traced scan tube that runs along its OWN
                      named limb centreline and no other chain claims it
  2 recovered         re-placed here from scan evidence (low-confidence joints, the named centreline + partner
                      lengths) and the new placement passed the same verification
  3 review            registered by the route and kept, but some evidence contradicts it (flagged)
  4 unregistered      no support; vertices left where the route put them (no claim)
  5 imputed_mirror    the scan lacks the part (or it could not be verified): placed as the mirror image of the
                      registered bilateral partner, attached at the own proximal joint; EXCLUDED from every measurement
  6 imputed_prior     as 5, without a usable partner: the route's own placement (model prior) is kept; EXCLUDED
  7 recovered_estimated  re-placed on its own named limb and verified like 2, but its distal joint position along
                      the limb is estimated (proportions / partner lengths), not an observed articulation: registered
                      for correspondence and a shape model, flagged, NOT measurable
Measurable = codes 0, 1, 2; registered = 0, 1, 2, 3, 7 (3 and 7 flagged: the scorer counts them as flagged).
"""

import json
import os

import numpy as np
from scipy.spatial import cKDTree

from fitter_3d.antscan_registration.parts import partconf

VERSION = "cover-0.4"
STATUS_NAMES = [
    "joint_supported",
    "surface_verified",
    "recovered",
    "review",
    "unregistered",
    "imputed_mirror",
    "imputed_prior",
    "recovered_estimated",
]
REGISTERED = (0, 1, 2, 3, 7)
MEASURABLE = (0, 1, 2)
CONF_MIN = partconf.USABLE_CONF
LEG_SEGS = [("co", "tr"), ("tr", "fe"), ("fe", "ti"), ("ti", "ta"), ("ta", "pt")]
ANT_SEGS = [("an_1", "an_2"), ("an_2", "an_3")]
LIMB_OF = {**{f"l_{k}_{s}": ("L" if s == "l" else "R") + k for k in "123" for s in "lr"}, "an_l": "AL", "an_r": "AR"}
PARTNER = {
    **{f"l_{k}_l": f"l_{k}_r" for k in "123"},
    **{f"l_{k}_r": f"l_{k}_l" for k in "123"},
    "an_l": "an_r",
    "an_r": "an_l",
}

# ---- constants (all relative; hand-set) -----------------------------------------------------
JTOL_BL = 0.015  # a marker 'sits on' its input joint within 1.5 % BL
NB = 8  # axial bins of a registered segment's medial curve
ON_TOL_R = 2.0  # a medial sample is on a centreline within this many local radii ...
ON_TOL_BL = 0.012  # ... or this share of BL, whichever is larger
ON_MIN = 0.6  # share of a segment's medial samples that must lie on its own centreline (where it is defined)
OTHER_MAX = 0.5  # share of samples nearer another named limb's centreline (and on it) above which the segment is
# 'on a neighbour'
DUP_MIN = 0.5  # two chains' medial curves within tolerance over this share of the shorter segment -> duplicate
ASYM_MAX = 0.3  # |log(L / L_partner)| above which a segment length is implausible (left / right) ...
ASYM_LOOSE = 0.5  # ... for the coxa and the rest of the tarsus (tip placement), whose chords are short / noisy
Q_GAP = 0.15  # joint-supported pair with mismatched lengths: the side whose weaker joint confidence is lower by
# this much is flagged, otherwise both
CONTACT_OWN = 0.8  # a recovered segment whose tube is only 'contact' (limb-like material, no clean trace) is accepted
# when this share of it lies on its own named centreline
LOWCONF_MIN = 0.15  # recovery candidates from the product's own joints only down to this confidence (DECLARED:
# chosen from development-truth bands, not leave-one-out)
LIMB_TRUST = 0.6  # a named centreline is evidence only with limb confidence >= 0.6 and without approaches_disagree
TIP_TOL_BL = 0.02  # an end segment's tip marker within 2 % BL of the observed centreline tip
CHORD_ON_MIN = 0.6  # recovery: share of the chord between two candidate joints within tolerance of the named centreline
SEVERE_OWN = 0.3  # a joint-supported segment is contradicted only when its surface is clearly elsewhere
BEYOND_BL = 0.02  # a sample whose nearest centreline point is the path tip and farther than this is 'beyond the path'


# ======================================================================================================================
# template / inputs
# ======================================================================================================================
_T = {}


def tpl():
    if "t" not in _T:
        from fitter_3d.antscan_registration.parts.skelreg.model import template

        t = template("medial")
        t.marker_regressor()
        _T["t"] = t
        _T["M"] = t.marker_regressor()
    return _T["t"]


def markers(V):
    tpl()
    return _T["M"] @ V


def chains():
    """chain -> list of segment dicts (bone, child, names, owner bones whose vertices form the segment)."""
    t = tpl()
    out = {}
    for side in "lr":
        for k in "123":
            ch = f"l_{k}_{side}"
            segs = []
            for a, b in LEG_SEGS:
                na, nb = f"l_{k}_{a}_{side}", f"l_{k}_{b}_{side}"
                own = [t.idx[na]] + ([t.idx[nb]] if b == "pt" else [])
                segs.append(dict(name=na, bone=t.idx[na], child=t.idx[nb], own=own, end=(b == "pt")))
            out[ch] = segs
        ch = f"an_{side}"
        segs = []
        for a, b in ANT_SEGS:
            na, nb = f"{a}_{side}", f"{b}_{side}"
            own = [t.idx[na]] + ([t.idx[nb]] if b == "an_3" else [])
            segs.append(dict(name=na, bone=t.idx[na], child=t.idx[nb], own=own, end=(b == "an_3")))
        out[ch] = segs
    return out


def load_joint_product(jfile):
    """-> dict P (nJ,3) usable joints (nan otherwise), Pall (every placed joint, also conf < CONF_MIN), conf, BL, frame."""
    from fitter_3d.antscan_registration.parts.skelreg import sources

    t = tpl()
    jt = sources.load(os.path.basename(jfile).replace("_joints.json", ""), "s6fix", t, path=jfile)
    d = json.load(open(jfile))
    js, cf = d.get("joints_scan", {}) or {}, d.get("confidence", {}) or {}
    Pall = np.full((t.nJ, 3), np.nan)
    call = np.zeros(t.nJ)
    for i, n in enumerate(t.names):
        v = js.get(n)
        if v is not None:
            Pall[i] = np.asarray(v, float)
            c = cf.get(n, 1.0)
            call[i] = 1.0 if c is None else float(c)
    for i in (t.idx["b_a_4"], t.idx["b_a_5"], t.idx["b_t"], t.idx["b_h"]):
        if np.isfinite(jt["P"][i]).all():
            Pall[i], call[i] = jt["P"][i], max(call[i], jt["conf"][i])
    R = np.asarray(d["frame"]["rotation"], float)
    o = np.asarray(d["frame"]["origin_scan"], float)
    return dict(
        P=jt["P"],
        conf=jt["conf"],
        Pall=Pall,
        call=call,
        BL=float(d.get("body_length_um") or np.nan),
        R=R,
        origin=o,
        d=d,
        path=jfile,
    )


def load_limbs(limbs_dir):
    """named centrelines (scan frame, tip first) with radius profile; status / confidence per limb."""
    out = {}
    jf, nf = f"{limbs_dir}/limbs.json", f"{limbs_dir}/limbs.npz"
    if not (os.path.isfile(jf) and os.path.isfile(nf)):
        return out
    L = json.load(open(jf))["limbs"]
    z = np.load(nf)
    for k, v in L.items():
        if f"{k}_X" in z.files and v.get("status") in ("path", "partial") and len(z[f"{k}_X"]) >= 2:
            X = np.asarray(z[f"{k}_X"], float)
            r = np.asarray(z[f"{k}_r"], float) if f"{k}_r" in z.files else np.full(len(X), np.nan)
            s = (
                np.asarray(z[f"{k}_s"], float)
                if f"{k}_s" in z.files
                else np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
            )
            core = v.get("core_arc_um")
            conf = float(v.get("confidence") or 0.0)
            flags = list(v.get("flags") or [])
            # round 7 writes the same fact as 's6:approaches_disagree' (the chosen state's approaches disagree) and
            # 'states_disagree' (the states of the ensemble disagree on the limb): both count as approaches_disagree
            if "approaches_disagree" not in flags and any(
                "approaches_disagree" in str(f) or str(f) == "states_disagree" for f in flags
            ):
                flags.append("approaches_disagree")
            out[k] = dict(
                X=X,
                r=r,
                s=s,
                status=v["status"],
                conf=conf,
                flags=flags,
                trusted=bool(conf >= LIMB_TRUST and "approaches_disagree" not in flags),
                core_arc=float(core) if core is not None else float(s[-1]),
                kd=cKDTree(X),
            )
    return out


# ======================================================================================================================
# geometry helpers
# ======================================================================================================================
def medial_curve(V, own_mask, A, B, nb=NB):
    """Medial samples of a registered segment: centroids of its own vertices in nb axial bins along A -> B, with the
    bin radius (median perpendicular distance). Returns C (k,3), r (k,), t (k,) (bin centres in chord fractions)."""
    X = V[own_mask]
    L = float(np.linalg.norm(B - A))
    if len(X) < 3 or L <= 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    u = (B - A) / L
    ax = (X - A) @ u / L
    C, r, tt = [], [], []
    edges = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (ax >= edges[i]) & (ax < edges[i + 1]) if i < nb - 1 else (ax >= edges[i]) & (ax <= 1.0 + 1e-9)
        if m.sum() >= 3:
            c = X[m].mean(0)
            q = X[m] - c
            r.append(float(np.median(np.linalg.norm(q - (q @ u)[:, None] * u, axis=1))))
            C.append(c)
            tt.append(0.5 * (edges[i] + edges[i + 1]))
    return np.asarray(C).reshape(-1, 3), np.asarray(r), np.asarray(tt)


def seg_points(A, B, n=9):
    f = np.linspace(0.1, 0.9, n)[:, None]
    return A[None] + f * (B - A)[None]


def near_polyline(lim, Q):
    """distance of Q to the polyline (vertex kd-tree on a densely sampled centreline), index of the nearest point."""
    d, i = lim["kd"].query(Q)
    return d, i


# ======================================================================================================================
# evidence
# ======================================================================================================================
def tube_verification(scan, P_joints):
    """SKEL-REG's stage A' (tube tracing + verify_limbs) run on an arbitrary joint set (here: the registration's own
    virtual markers, or candidate joints). Returns {segment name: verdict dict}."""
    from fitter_3d.antscan_registration.parts.skelreg import tubes, verify
    from fitter_3d.antscan_registration.parts.skelreg.bones import BoneFit

    t = tpl()
    P = P_joints.copy()
    for n in ("w_1_r", "w_2_r", "w_1_l", "w_2_l"):
        P[t.idx[n]] = np.nan
    bf = BoneFit(t, P).solve(lookahead=True)
    tc = tubes.template_charts("medial")
    st = tubes.trace_scan_tubes(scan, t, tc, bf, twist=False)
    ver, _ = verify.verify_limbs(scan, t, tc, st, bf)
    qa = {s.name: s.qa for s in st if s.resolved}
    for k, v in ver.items():
        v["tube_qa"] = qa.get(k, {})
    return ver


def segment_evidence(V, M, jp, limbs, ver, BL, reg_v=None, V_pre=None):
    """Per chain and segment: joint support, tube verdict, own / other centreline agreement, duplicate, asymmetry."""
    t = tpl()
    CH = chains()
    P, _conf = jp["P"], jp["conf"]
    ev = {}
    curves = {}
    for ch, segs in CH.items():
        own_lim = limbs.get(LIMB_OF[ch])
        for sg in segs:
            b, c = sg["bone"], sg["child"]
            A, B = M[b], M[c]
            mask = np.isin(t.dom, sg["own"])
            C, r, tt = medial_curve(V, mask, A, B)
            if len(C) == 0:
                C, r = seg_points(A, B, NB), np.full(NB, 0.004 * BL)
            curves[sg["name"]] = (C, np.maximum(r, 0.002 * BL), ch)
            e = dict(chain=ch, bone=b, child=c, length=float(np.linalg.norm(B - A)))
            # joint support
            ok_b, ok_c = bool(np.isfinite(P[b]).all()), bool(np.isfinite(P[c]).all())
            db = float(np.linalg.norm(M[b] - P[b]) / BL) if ok_b else None
            dc = float(np.linalg.norm(M[c] - P[c]) / BL) if ok_c else None
            e.update(
                conf_b=float(jp["call"][b]),
                conf_c=float(jp["call"][c]),
                usable_b=ok_b,
                usable_c=ok_c,
                dmark_b=db,
                dmark_c=dc,
            )
            e["jsup"] = bool(ok_b and ok_c and db <= JTOL_BL and dc <= JTOL_BL)
            prov = jp["d"].get("provenance") or {}
            e["prov_c"] = str(prov.get(t.names[c]) or "null").split(" ")[0]
            e["prov_b"] = str(prov.get(t.names[b]) or "null").split(" ")[0]
            e["dmark_any_b"] = (
                float(np.linalg.norm(M[b] - jp["Pall"][b]) / BL) if np.isfinite(jp["Pall"][b]).all() else None
            )
            e["dmark_any_c"] = (
                float(np.linalg.norm(M[c] - jp["Pall"][c]) / BL) if np.isfinite(jp["Pall"][c]).all() else None
            )
            if sg["end"] and own_lim is not None:
                e["tip_dist"] = float(np.linalg.norm(M[c] - own_lim["X"][0]) / BL)
                e["tip_observed"] = own_lim["status"] == "path"
            # tube verdict on the registration's own markers
            v = ver.get(sg["name"])
            e["tube"] = v["verdict"] if v else "unresolved"
            e["chart_ok"] = bool(v and v["chart_ok"])
            e["chord_support"] = float(v["chord_support"]) if v else None
            e["vertex_support"] = float(v["vertex_support"]) if v else None
            e["inner_step"] = (v or {}).get("tube_qa", {}).get("inner_step")
            # own named centreline
            tol = np.maximum(ON_TOL_BL * BL, ON_TOL_R * r if len(r) == len(C) else ON_TOL_BL * BL)
            if own_lim is not None:
                d_own, i_own = near_polyline(own_lim, C)
                rr = own_lim["r"][i_own]
                tol_o = np.maximum(tol, ON_TOL_R * np.where(np.isfinite(rr), rr, 0))
                beyond = (i_own == 0) & (d_own > BEYOND_BL * BL) & (own_lim["status"] == "partial")
                defined = ~beyond
                e["own_defined"] = float(defined.mean())
                e["on_own"] = float((d_own[defined] <= tol_o[defined]).mean()) if defined.any() else None
                e["d_own_med_bl"] = float(np.median(d_own) / BL)
            else:
                d_own = np.full(len(C), np.inf)
                e["own_defined"], e["on_own"], e["d_own_med_bl"] = 0.0, None, None
                tol_o = tol
            # other named limbs' centrelines (legs and antennae; mandibles too: a leg lying on a mandible)
            other = np.zeros(len(C), bool)
            who = {}
            for k, lim in limbs.items():
                if k == LIMB_OF[ch]:
                    continue
                d_o, i_o = near_polyline(lim, C)
                ro = lim["r"][i_o]
                tl = np.maximum(tol, ON_TOL_R * np.where(np.isfinite(ro), ro, 0))
                hit = (d_o <= tl) & (d_o < 0.5 * d_own)
                if hit.any():
                    who[k] = float(hit.mean())
                other |= hit
            e["on_other"] = float(other.mean())
            e["on_other_who"] = who
            # free-deformation magnitude of the part (route's own stages), in local radii
            if V_pre is not None:
                dm = np.linalg.norm(V[mask] - V_pre[mask], axis=1)
                e["moved_free_r"] = float(np.median(dm) / max(np.median(r) if len(r) else 1e-9, 0.002 * BL))
            if reg_v is not None:
                e["route_reg"] = float(reg_v[mask].mean()) if mask.any() else 0.0
            ev[sg["name"]] = e
    # duplicates between chains (medial curves of two registered segments of different chains on the same material)
    names = list(curves)
    trees = {n: cKDTree(curves[n][0]) for n in names}
    for n in names:
        ev[n]["dup_with"] = {}
    for i, a in enumerate(names):
        Ca, ra, cha = curves[a]
        for b_ in names[i + 1 :]:
            Cb, rb, chb = curves[b_]
            if cha == chb:
                continue
            da, _ = trees[b_].query(Ca)
            db, _ = trees[a].query(Cb)
            sa = float(np.mean(da <= np.maximum(ra, ON_TOL_BL * BL * 0.5)))
            sb = float(np.mean(db <= np.maximum(rb, ON_TOL_BL * BL * 0.5)))
            s = max(sa, sb) if min(len(Ca), len(Cb)) > 0 else 0.0
            if s >= DUP_MIN:
                ev[a]["dup_with"][b_] = s
                ev[b_]["dup_with"][a] = s
    # left / right segment lengths
    for ch, segs in CH.items():
        pc = PARTNER[ch]
        for sg, sp in zip(segs, CH[pc]):
            L1, L2 = ev[sg["name"]]["length"], ev[sp["name"]]["length"]
            ev[sg["name"]]["asym"] = float(abs(np.log(max(L1, 1e-9) / max(L2, 1e-9))))
    return ev


# ======================================================================================================================
# status
# ======================================================================================================================
def seg_type(name):
    p = name.split("_")
    return p[2] if p[0] == "l" else f"an_{p[1]}"


# ======================================================================================================================
# per-vertex / per-part assembly
# ======================================================================================================================


# ======================================================================================================================
# re-posing a chain (recovery and imputation): per-bone similarity, blended with the skinning weights
# ======================================================================================================================


# ======================================================================================================================
# recovery candidates
# ======================================================================================================================


# ======================================================================================================================
# recovery and imputation of chain tails
# ======================================================================================================================
TRUNK_KEEP = ("b_t", "b_h", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5")


# ======================================================================================================================
# one specimen
# ======================================================================================================================
