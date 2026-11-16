"""Per-specimen inputs of the geometric registration core, all in the fitter's NORMALISED TARGET FRAME.
(developed from the earlier surface registration; the specimen no longer needs a fitter_3d
Stage_1 result; the frame comes from the joint product and the envelope; mandible tips from the limb product)

Frames (facts read from the stage .npz and the joint product):
  scan frame (micrometres)      envelope STL, joints_scan, limb polylines
  canonical frame (micrometres) x_can = (x_scan - origin_scan) @ R^T, R = frame.rotation (rows = canonical axes)
  normalised target frame       x_n = (x_can - target_centre) / target_scale      <- stage .npz verts / pivots live here
so x_scan = (x_n * target_scale + target_centre) @ R + origin_scan.
"""

import json
import os

import numpy as np

from . import geom
from .model import LIMB_CHAINS, template


class Frame:
    def __init__(self, R, origin, centre, scale):
        self.R, self.o = np.asarray(R, float), np.asarray(origin, float)
        self.c, self.s = np.asarray(centre, float), float(scale)

    def to_norm(self, Xs):
        return ((np.asarray(Xs, float) - self.o) @ self.R.T - self.c) / self.s

    def to_scan(self, Xn):
        return (np.asarray(Xn, float) * self.s + self.c) @ self.R + self.o

    def dir_to_norm(self, D):
        return np.asarray(D, float) @ self.R.T

    def dir_to_scan(self, D):
        return np.asarray(D, float) @ self.R


def joint_targets(jfile, frame, tpl=None, conf_min=0.3, kind="medial"):
    """Targets (55, 3) in the normalised frame (nan = unusable) and weights (55,), with the rules of the proposal's
    fitter_3d.joint_init.load_proposal (same result): null, wings, completeness.missing_joints,
    confidence < conf_min -> unusable; kind 'medial': b_t dropped, b_a_5 := gaster_tip, b_a_4 on b_a_3 -> tip."""
    tpl = tpl or template()
    d = json.load(open(jfile))
    js, cf = d.get("joints_scan") or {}, d.get("confidence") or {}
    missing = [m for m in (d.get("completeness") or {}).get("missing_joints", []) if m in tpl.idx]
    P = np.full((tpl.nJ, 3), np.nan)
    w = np.zeros(tpl.nJ)
    for j, n in enumerate(tpl.names):
        x = js.get(n)
        if x is None or n.startswith("w_") or n in missing or (kind == "medial" and n == "b_t"):
            continue
        if not np.all(np.isfinite(np.asarray(x, float))):  # a non-finite joint is no joint
            continue
        c = cf.get(n, 1.0)
        c = float(np.clip(1.0 if c is None else c, 0.0, 1.0))
        if c < conf_min:
            continue
        P[j] = frame.to_norm(np.asarray(x, float))
        w[j] = c
    if kind == "medial" and js.get("gaster_tip") is not None:
        i3, i4, i5 = (tpl.idx[k] for k in ("b_a_3", "b_a_4", "b_a_5"))
        ct = float(cf.get("gaster_tip") or 0.0)
        if ct >= conf_min and np.isfinite(P[i3]).all():
            tip = frame.to_norm(np.asarray(js["gaster_tip"], float))
            f = tpl.gaster_fraction
            P[i5], P[i4] = tip, P[i3] + f * (tip - P[i3])
            w[i4] = w[i5] = max(w[i3], conf_min)
            missing = [m for m in missing if m not in ("b_a_4", "b_a_5")]
    return P, w, missing, d


def bone_support(tpl, w, frozen=None):
    """bone j (the segment(s) leaving joint j) is joint-supported when joint j and at least one child joint have a
    usable target (leaf: its own target and its parent's); root: neck and b_a_1. frozen (55,) bool overrides."""
    ok = w > 0
    sup = np.zeros(tpl.nJ, bool)
    for j in range(tpl.nJ):
        ch = tpl.children[j]
        if j == 0:
            sup[j] = ok[tpl.idx["b_h"]] and ok[tpl.idx["b_a_1"]]
        elif ch:
            sup[j] = ok[j] and any(ok[c] for c in ch)
        else:
            sup[j] = ok[j] and ok[tpl.parents[j]]
    if frozen is not None:
        sup &= ~frozen
    return sup


class Limbs:
    """Limb product of the joint method in the normalised frame: per limb a polyline ROOT -> TIP (reversed from the file), arc from the
    root, radius, and the arcs of the limb's usable joints."""

    def __init__(self, limb_dir, frame, tpl, P, w):
        self.limbs = {}
        jf, zf = f"{limb_dir}/limbs.json", f"{limb_dir}/limbs.npz"
        if not (os.path.isfile(jf) and os.path.isfile(zf)):
            return
        meta = json.load(open(jf))["limbs"]
        z = np.load(zf)
        for limb, chain in LIMB_CHAINS.items():
            if limb not in meta or meta[limb].get("status") == "no_path" or f"{limb}_X" not in z.files:
                continue
            X = frame.to_norm(z[f"{limb}_X"])[::-1].copy()
            r = np.maximum(z[f"{limb}_r"][::-1].astype(float), 0.0) / frame.s
            if len(X) < 8:
                continue
            s = geom.polyline_arclen(X)
            s_file = z[f"{limb}_s"].astype(float)
            core = meta[limb].get("core_arc_um")
            core_s = (
                float(s[-1] - (core / frame.s)) if core is not None else 0.0
            )  # arc (from the root) of the trunk entry
            js = [tpl.idx[n] for n in chain][::-1]  # proximal first
            arcs, dist = np.full(len(js), np.nan), np.full(len(js), np.nan)
            for a, j in enumerate(js):
                if w[j] > 0:
                    d = np.linalg.norm(X - P[j], axis=1)
                    k = int(d.argmin())
                    arcs[a], dist[a] = s[k], d[k]
            self.limbs[limb] = dict(
                X=X,
                s=s,
                r=r,
                joints=js,
                arcs=arcs,
                dist=dist,
                core_s=max(core_s, 0.0),
                status=meta[limb].get("status"),
                conf=meta[limb].get("confidence"),
                file_len=float(s_file[-1]) if len(s_file) else 0.0,
            )
        # pseudo limb "G": the stretch of the trunk path between the gaster base (b_a_3) and the gaster apex
        i3, i4, i5 = (tpl.idx[k] for k in ("b_a_3", "b_a_4", "b_a_5"))
        if "T_X" in z.files and w[i3] > 0 and w[i5] > 0 and len(z["T_X"]) >= 8:
            TX = frame.to_norm(z["T_X"])
            Tr = np.maximum(z["T_r"].astype(float), 0.0) / frame.s
            k3 = int(np.linalg.norm(TX - P[i3], axis=1).argmin())
            k5 = int(np.linalg.norm(TX - P[i5], axis=1).argmin())
            if abs(k5 - k3) >= 8:
                sl = slice(k3, k5 + 1) if k3 < k5 else slice(k5, k3 + 1)
                X, r = TX[sl], Tr[sl]
                if k3 > k5:
                    X, r = X[::-1].copy(), r[::-1].copy()
                s = geom.polyline_arclen(X)
                d4 = np.linalg.norm(X - P[i4], axis=1) if w[i4] > 0 else None
                arcs = np.array([0.0, s[int(d4.argmin())] if d4 is not None else np.nan, s[-1]])
                dist = np.array(
                    [
                        np.linalg.norm(X[0] - P[i3]),
                        d4.min() if d4 is not None else np.nan,
                        np.linalg.norm(X[-1] - P[i5]),
                    ]
                )
                self.limbs["G"] = dict(
                    X=X,
                    s=s,
                    r=r,
                    joints=[i3, i4, i5],
                    arcs=arcs,
                    dist=dist,
                    core_s=0.0,
                    status="trunk_path",
                    conf=None,
                    file_len=float(s[-1]),
                )


def frame_from_product(jd, Vs_scan):
    """Frame of a specimen from the joint product alone: canonical frame = the product's frame (rotation rows =
    canonical axes in scan coordinates, origin_scan), normalisation = what fitter_3d's load_meshes(return_norm=True)
    does with the canonical-frame envelope (centre = vertex mean, scale = largest |coordinate| after centring), so
    target_centre / target_scale mean the same as in the fitter's own runs (fitter_3d/utils.py load_meshes, read
    2026-11-13). A product without a frame gets the identity (scan frame = canonical frame)."""
    fr = jd.get("frame") or {}
    R = np.asarray(fr.get("rotation", np.eye(3)), float)
    o = np.asarray(fr.get("origin_scan", np.zeros(3)), float)
    Xc = (np.asarray(Vs_scan, float) - o) @ R.T
    c = Xc.mean(0)
    sc = float(np.abs(Xc - c).max())
    return Frame(R, o, c, sc)


def mandible_tips(limb_dir, frame, tpl, min_len_um=0.0):
    """{bone index: dict(tip (3,) normalised frame, conf, status, len)} from the limb product's mandible centrelines
    (ML / MR, stored tip first). Only the TIP is used: a mandible has one joint, its limb centreline gives
    the second point that fixes its direction and length."""
    out = {}
    jf, zf = f"{limb_dir}/limbs.json", f"{limb_dir}/limbs.npz"
    if not (limb_dir and os.path.isfile(jf) and os.path.isfile(zf)):
        return out
    meta = json.load(open(jf)).get("limbs", {})
    z = np.load(zf)
    for limb, bone in (("ML", "ma_l"), ("MR", "ma_r")):
        q = meta.get(limb) or {}
        if q.get("status") in (None, "no_path") or f"{limb}_X" not in z.files or len(z[f"{limb}_X"]) < 4:
            continue
        X = z[f"{limb}_X"]
        out[tpl.idx[bone]] = dict(
            tip=frame.to_norm(X[0]),
            conf=float(q.get("confidence") or 0.0),
            status=q.get("status"),
            flags=list(q.get("flags") or []),
            core_arc=q.get("core_arc_um"),
            X=frame.to_norm(X),
            r=np.maximum(z[f"{limb}_r"].astype(float), 0.0) / frame.s,
        )
    return out


class Specimen:
    """Everything the registration needs for one specimen (normalised target frame). No truth, no names used."""

    def __init__(
        self,
        name,
        env_path,
        jfile,
        limb_dir=None,
        conf_min=0.3,
        kind="medial",
        scan=None,
        tpl=None,
        frame=None,
        scan_norm=None,
    ):
        """env_path: envelope STL in the scan frame (micrometres), or scan=(Vs, Fs) in the SCAN frame, or (inside
        fitter_3d, where load_meshes has already normalised the target) scan_norm=(Vs, Fs) in the normalised target
        frame together with its frame=Frame(...)."""
        tpl = tpl or template()
        self.name, self.tpl, self.kind, self.jfile = name, tpl, kind, jfile
        jd = json.load(open(jfile))
        if scan_norm is not None:
            assert frame is not None, "scan_norm needs its frame"
            Vs, Fs = frame.to_scan(np.asarray(scan_norm[0], np.float64)), np.asarray(scan_norm[1], np.int64)
        elif scan is None:
            Vs, Fs = geom.load_mesh(env_path)
        else:
            Vs, Fs = np.asarray(scan[0], np.float64), np.asarray(scan[1], np.int64)
        if frame is None:
            frame = frame_from_product(jd, Vs)
        self.frame = frame
        self.Vs = frame.to_norm(Vs) if scan_norm is None else np.asarray(scan_norm[0], np.float64)
        self.Fs = Fs
        self.Ns = geom.vertex_normals(self.Vs, Fs)
        self.As = geom.vertex_areas(self.Vs, Fs)
        self.P, self.w, self.missing, _ = joint_targets(jfile, frame, tpl, conf_min, kind)
        hf = (jd.get("joints_scan") or {}).get("head_front")
        cf = (jd.get("confidence") or {}).get("head_front")
        self.head_front = (
            frame.to_norm(np.asarray(hf, float))
            if (kind == "medial" and hf is not None and (cf is None or cf >= conf_min))
            else None
        )
        bl = jd.get("body_length_um")
        self.BL_um = float(bl) if bl else None
        self.BL = self.BL_um / frame.s if self.BL_um else None
        self.flags = list((jd.get("completeness") or {}).get("flags") or [])
        self.product_status = jd.get("status")
        self.limbs = Limbs(limb_dir, frame, tpl, self.P, self.w) if limb_dir else None
        self.mand = mandible_tips(limb_dir, frame, tpl) if limb_dir else {}
        self.frozen = np.zeros(tpl.nJ, bool)
        self.support = None  # set from the closed-form start (native.Start.support)
        self.um_per_model_unit = None
