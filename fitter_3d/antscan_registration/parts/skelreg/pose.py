"""Joints -> similarity and bone rotations (closed form).

The FK-IK with the limit-aware twist search follows fitter_3d/joint_init.py (umeyama, min_rotation,
twist_for_limits, fk_ik); re-implemented here in numpy/scipy so that SKEL-REG does not import fitter_3d.
"""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from fitter_3d.antscan_registration.parts import partconf

from .model import WINGS

ROOT_RIGID = ["b_t", "b_a_1", "b_h"] + [f"l_{i}_co_{s}" for i in (1, 2, 3) for s in "rl"]
CONF_MIN = partconf.USABLE_CONF  # joints with a confidence below it, or null, are unresolved


def load_joints(src, tpl, conf_min=CONF_MIN):
    """Joint file in the build-off contract -> dict(P (nJ,3) nan = unresolved, conf, missing names, frame).
    Only `joints_scan`, `confidence`, `completeness.missing_joints` and `frame` are read (never `fit`)."""
    d = json.load(open(src)) if isinstance(src, str) else src
    P = np.full((tpl.nJ, 3), np.nan)
    conf = np.zeros(tpl.nJ)
    js, cf = d.get("joints_scan", {}) or {}, d.get("confidence", {}) or {}
    for i, n in enumerate(tpl.names):
        v = js.get(n)
        if v is None or n in WINGS:
            continue
        c = cf.get(n, 1.0)
        c = 1.0 if c is None else float(c)
        conf[i] = c
        if c >= conf_min and np.all(np.isfinite(v)):
            P[i] = np.asarray(v, dtype=np.float64)
    missing = [
        m for m in (d.get("completeness", {}) or {}).get("missing_joints", []) if m in tpl.idx and m not in WINGS
    ]
    for m in missing:
        P[tpl.idx[m]] = np.nan
    fr = d.get("frame") or {}
    R = np.asarray(fr["rotation"], dtype=np.float64) if fr.get("rotation") is not None else None
    o = np.asarray(fr["origin_scan"], dtype=np.float64) if fr.get("origin_scan") is not None else None
    return {
        "P": P,
        "conf": conf,
        "missing": missing,
        "frame_R": R,
        "frame_origin": o,
        "status": d.get("status"),
        "method": d.get("method"),
        "version": d.get("version"),
    }


def unresolved_bones(tpl, P):
    """Bones (segments joint -> child) that cannot be registered: the bone of an unresolved joint, and a chain bone
    whose only child joint is unresolved (its far end is unknown). b_t / b_h keep their segments. A bone distal of a
    gap is registered when its own two joints are given (annotated truth often lacks a coxa base while the rest of the
    leg is there). With the damage lists of the synthetic sets (every distal joint listed) this equals the earlier rule
    'everything distal of an unresolved joint' and the prototype's `excluded_bones`."""
    bad = np.zeros(tpl.nJ, bool)
    ok = np.isfinite(P).all(1)
    wing = np.array([n in WINGS for n in tpl.names])
    for j in range(1, tpl.nJ):
        if wing[j]:
            continue
        if not ok[j]:
            bad[j] = True
            p = tpl.parents[j]
            if p > 0 and len(tpl.children[p]) == 1:
                bad[p] = True
    bad[wing] = False
    return bad


def umeyama(src, dst, w=None, with_scale=True):
    w = np.ones(len(src)) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    S, D = src - mu_s, dst - mu_d
    H = (S * w[:, None]).T @ D
    U, sv, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    s = float((sv * [1, 1, d]).sum() / (w * (S * S).sum(1)).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def min_rotation(u, v):
    u = u / (np.linalg.norm(u) + 1e-30)
    v = v / (np.linalg.norm(v) + 1e-30)
    c = float(np.clip(u @ v, -1, 1))
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(u, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(u, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    return Rotation.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def kabsch_uncentred(U, V, w=None):
    w = np.ones(len(U)) if w is None else np.asarray(w)
    H = (U * w[:, None]).T @ V
    A, _, Bt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Bt.T @ A.T))
    return Bt.T @ np.diag([1, 1, d]) @ A.T


_PHIS = np.linspace(-np.pi, np.pi, 145)
_PHIS = _PHIS[np.argsort(np.abs(_PHIS), kind="stable")]


def limit_violation(aa, lo, hi):
    return np.maximum(aa - hi, 0).sum(-1) + np.maximum(lo - aa, 0).sum(-1)


def similarity_root(tpl, P, J=None):
    """Similarity (s, R, t) of the template rest pivots onto the scan joints of the root-rigid set."""
    J = tpl.J0 if J is None else J
    ids = [tpl.idx[n] for n in ROOT_RIGID if np.isfinite(P[tpl.idx[n]]).all()]
    return umeyama(J[ids], P[ids])
