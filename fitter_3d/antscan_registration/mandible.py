"""Mandible bases the joint method holds below the cut: the same independent check RECOVER applies to a coxa base.

A mandible has one joint (its base, ma_l / ma_r); the registration can use it only when it is usable (>= 0.3), and
then aims the mandible at the tip of its limb centreline (ML / MR). A base below the cut used to leave the mandible
unregistered. Here the product's own position is accepted (never moved) when it lies where evidence that did not
use it puts the socket:

  mirror        within SOCK_MIRROR_BL of the mirror image (sagittal plane of the product frame) of the partner base;
                the partner may be usable, or itself a base below the cut (two bases the joint method placed
                separately, each on its own centreline, that are mirror images of each other), or
  registration  within SOCK_REG_BL of the first registration's virtual marker of that mandible (the head of the first
                registration is placed by the neck, the antennal sockets, the head front and the usable mandible
                bases only, so the marker does not depend on the joint under test), and
  identity      with a registration: not nearer to the OTHER mandible's registered marker than to its own by more
                than IDENT_MARGIN_BL,
  centreline    the limb product holds a centreline for this mandible (without it the registration has no tip to
                aim at and the joint would change nothing).

The constants are RECOVER's coxa-socket constants (limbrec.recover.DEFAULTS), unchanged. An accepted base gets
confidence 0.30 (0.35 with both kinds of socket evidence), provenance `recovered (lowconf; verified: ...)` and an
entry in `recovery.joints`, exactly like a recovered limb joint. The status step reads `verified_by`: a mandible whose
base has BOTH kinds of evidence is `recovered` (measurable), with one kind `recovered_estimated` (registered, flagged).
"""

import json
import os

import numpy as np

from . import USABLE_CONF, setup

USABLE = USABLE_CONF
SIDES = (("ma_l", "ma_r", "ML"), ("ma_r", "ma_l", "MR"))


def _pos(d, n):
    v = (d.get("joints_scan") or {}).get(n)
    if v is None or not np.all(np.isfinite(np.asarray(v, float))):
        return None
    return np.asarray(v, float)


def _usable(d, n):
    c = (d.get("confidence") or {}).get(n)
    miss = (d.get("completeness") or {}).get("missing_joints") or []
    return _pos(d, n) is not None and c is not None and float(c) >= USABLE and n not in miss


def _mirror(d, x):
    fr = d.get("frame") or {}
    R = np.asarray(fr.get("rotation", np.eye(3)), float)
    o = np.asarray(fr.get("origin_scan", np.zeros(3)), float)
    c = (np.asarray(x, float) - o) @ R.T
    c[1] = -c[1]
    return c @ R + o


def check(d, limbs_dir, reg_markers=None, names_idx=None):
    """d: joint product (after RECOVER); limbs_dir: <limbs>/<name>; reg_markers: (55, 3) virtual markers of the first
    registration (scan frame, um) or None; names_idx: {joint name: row}. Returns (d, log): d is modified in place for
    accepted bases."""
    setup()
    from fitter_3d.antscan_registration.parts.limbrec.recover import (
        DEFAULTS as RD,
    )  # the coxa-socket tolerances of the recovery step, not a copy

    SOCK_MIRROR_BL, SOCK_REG_BL, IDENT_MARGIN_BL = RD["sock_mirror_bl"], RD["sock_reg_bl"], RD["ident_margin_bl"]
    BL = float(d.get("body_length_um") or 0.0)
    log = {}
    if BL <= 0:
        return d, log
    meta = {}
    lj = os.path.join(limbs_dir or "", "limbs.json")
    if limbs_dir and os.path.isfile(lj):
        meta = json.load(open(lj)).get("limbs", {})
    usable0 = {n: _usable(d, n) for n in ("ma_l", "ma_r")}  # as the input holds them (not what this check accepts)
    miss0 = list((d.get("completeness") or {}).get("missing_joints") or [])
    for own, other, limb in SIDES:
        x = _pos(d, own)
        if x is None or usable0[own]:
            continue
        ev = dict(input_confidence=(d.get("confidence") or {}).get(own))
        why = []
        q = meta.get(limb) or {}
        ev["centreline"] = q.get("status")
        if q.get("status") in (None, "no_path"):
            why.append("no centreline for this mandible")
        ver = []
        xo = _pos(d, other)
        if xo is not None and other not in miss0:
            ev["mirror_pct"] = float(np.linalg.norm(x - _mirror(d, xo)) / BL * 100)
            ev["mirror_partner"] = "usable" if usable0[other] else "below the cut"
            if ev["mirror_pct"] <= 100 * SOCK_MIRROR_BL:
                ver.append("socket (mirror)" if usable0[other] else "socket (mirror of the partner base below the cut)")
        if reg_markers is not None:
            m_own, m_oth = reg_markers[names_idx[own]], reg_markers[names_idx[other]]
            ev["registration_pct"] = float(np.linalg.norm(x - m_own) / BL * 100)
            ev["registration_other_pct"] = float(np.linalg.norm(x - m_oth) / BL * 100)
            if ev["registration_pct"] <= 100 * SOCK_REG_BL:
                ver.append("socket (registration)")
            if ev["registration_pct"] > ev["registration_other_pct"] + 100 * IDENT_MARGIN_BL:
                why.append("nearer to the other mandible's registered socket")
        if not any(v in ("socket (mirror)", "socket (registration)") for v in ver):
            why.append("no socket evidence within tolerance (a usable partner's mirror image or the registration)")
        ev["verified_by"] = ver
        if why:
            log[own] = dict(recovered=False, reasons=why, evidence=ev)
            continue
        conf = min(0.30 + 0.05 * (len(ver) - 1), 0.35)
        prov = f"recovered (lowconf; verified: {', '.join(ver)})"
        d["confidence"][own] = conf
        d.setdefault("provenance", {})[own] = prov
        comp = d.setdefault("completeness", {})
        if own in (comp.get("missing_joints") or []):
            comp["missing_joints"] = [m for m in comp["missing_joints"] if m != own]
        rb = d.setdefault("recovery", {})
        rb.setdefault("joints", {})[own] = dict(
            kind="lowconf",
            confidence=conf,
            input_confidence=ev["input_confidence"],
            input_position=[float(v) for v in x],
            moved_pct_bl=0.0,
            verified_by=ver,
            evidence=ev,
            by="mandible_check",
        )
        rb["n_recovered"] = len(rb["joints"])
        log[own] = dict(recovered=True, confidence=conf, provenance=prov, evidence=ev)
    return d, log
