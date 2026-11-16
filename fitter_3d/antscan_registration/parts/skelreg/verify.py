"""Scan-surface verification of every limb segment BEFORE it is allowed to place template surface (stage A').

A joint proposal is not trusted blindly: an early generation of the joint method put only 57 % of real limb chains on the right limb. A limb
segment (tube bone: coxa, trochanter+femur, tibia, basitarsus, tarsus, scape, funiculus) is

  ok        its tube was traced on the scan and is self-consistent (tubes.trace_scan_tubes: enough centred sections,
            profile agrees with the sections, enough visible wall, girth comparable with the specimen's other limbs)
            and no other limb's tube claims the same scan surface;
  contact   the tube could not be traced, but the segment lies on limb-like material: at least CHORD_MIN of the joint
            chord's samples are inside the scan solid at most DEPTH_R expected limb radii below the surface (a limb
            lying against the body) or just outside it (pivots on the wall), OR at least CHORD_MIN of the segment's
            own template vertices, placed from the joints alone (stage B), have scan surface within VERT_R expected
            radii (curved thin segments whose chord leaves the solid). The segment is placed from the joints alone;
  fail      neither: the chord runs through air or deep through a body. The segment and everything distal of it is
            UNREGISTERED (placed by continuation, vertex status 2, no morphometrics);
  duplicate two limbs' tubes claim the same scan surface (>= DUP_MIN of the smaller claim): the chain whose proximal
            segments are worse supported is unregistered from that segment on, the other is marked for review.

Trunk bones are not verified here (their joints are checked against the scan by the diagnosis, skelreg/diagnose.py).
"""

import numpy as np

from . import tubes
from .refine import inside

CHORD_MIN = 0.5  # share of chord samples / segment vertices that must be supported
DEPTH_R = 2.5  # a supported inside sample is at most this many expected limb radii below the surface
NEAR_R = 0.75  # a supported outside sample is at most this many expected radii (>= NEAR_MIN x size) off the surface
NEAR_MIN = 0.004
VERT_R = 1.5  # stage-B vertices of the segment count as supported within this many expected radii of the surface
DUP_MIN = 0.5
CURVE_R = 1.0  # duplicate: the two medial curves are closer than this many radii over half of the shorter tube
FRACS = np.linspace(0.15, 0.85, 9)


def chain_of(name):
    """l_2_fe_r -> 'l_2_r'; an_1_l -> 'an_l'; else None (trunk)."""
    p = name.split("_")
    if p[0] == "l":
        return f"l_{p[1]}_{p[3]}"
    if p[0] == "an":
        return f"an_{p[2]}"
    return None


def chord_support(scan, A, B, r_exp):
    X = A[None] + FRACS[:, None] * (B - A)[None]
    ins = inside(scan, X)
    d, _ = scan.kd.query(X)
    sup = np.where(ins, d <= DEPTH_R * r_exp, d <= max(NEAR_R * r_exp, NEAR_MIN * scan.size))
    return float(sup.mean()), float(ins.mean()), float(np.median(d) / max(r_exp, 1e-12))


def vertex_support(scan, VB, r_exp):
    if not len(VB):
        return 0.0
    d, _ = scan.kd.query(VB)
    return float((d <= max(VERT_R * r_exp, NEAR_MIN * scan.size)).mean())


def verify_limbs(scan, tpl, tc, st, bf, girth=1.0):
    """Verdict per resolved limb tube. Returns dict name -> dict(verdict, chart_ok, chord_support, inside_share,
    depth_r, reason) and the list of joints to drop (child joints of failed segments)."""
    out = {}
    claims = {}
    tub = {}
    VB = bf.vertices()
    for k, (d, s) in enumerate(zip(tc, st)):
        ch = chain_of(s.name)
        if ch is None or not s.resolved:
            continue
        r_exp = float(np.median(d["rguess"])) * bf.s * girth
        sup, ins, dep = chord_support(scan, bf.P[s.bone], bf.P[s.child], r_exp)
        chart_ok = bool(s.qa.get("use", False))
        vsup = sup if chart_ok else vertex_support(scan, VB[d["own"]], r_exp)
        v = "ok" if chart_ok else ("contact" if max(sup, vsup) >= CHORD_MIN else "fail")
        why = "" if chart_ok else (s.qa.get("note", "") or "tube not traced self-consistently")
        out[s.name] = dict(
            verdict=v,
            chart_ok=chart_ok,
            chord_support=sup,
            vertex_support=vsup,
            inside_share=ins,
            depth_r=dep,
            chain=ch,
            reason=why,
        )
        if chart_ok:
            ids = tubes.tube_claims(scan, s)[0]
            claims[s.name] = set(ids.tolist())
            tub[s.name] = s.tube
    # duplicates: two different chains on the same scan surface
    names = list(claims)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if out[a]["chain"] == out[b]["chain"] or not claims[a] or not claims[b]:
                continue
            ov = len(claims[a] & claims[b]) / min(len(claims[a]), len(claims[b]))
            if ov >= DUP_MIN and _same_curve(tub[a], tub[b]):
                # the chain with the weaker proximal support loses
                sa = _chain_support(out, out[a]["chain"])
                sb = _chain_support(out, out[b]["chain"])
                lose, keep = (a, b) if sa < sb else (b, a)
                out[lose].update(verdict="duplicate", reason=f"same scan surface as {keep} (overlap {ov:.2f})")
                if out[keep]["verdict"] == "ok":
                    out[keep].update(verdict="ok", review=f"shares scan surface with {lose}")
    drop = []
    for d, s in zip(tc, st):
        r = out.get(s.name)
        if r and r["verdict"] in ("fail", "duplicate"):
            drop.append(s.child)
    return out, drop


def _same_curve(ta, tb):
    """Two traced tubes run along the SAME physical limb: the medial curve of the shorter one lies within CURVE_R of
    its own radius of the other's curve for most of its length. (Shared surface claims alone also arise between two
    limbs that lie against each other: 25 false duplicates among coxae and femora of oomA with oracle joints.)"""
    from scipy.spatial import cKDTree

    from .charts import N_EXT, N_IN

    a, b = (ta, tb) if ta.length <= tb.length else (tb, ta)
    Ca, Cb = a.C[N_EXT : N_EXT + N_IN], b.C[N_EXT : N_EXT + N_IN]
    Cbd = np.concatenate([Cb, 0.5 * (Cb[1:] + Cb[:-1])])
    d, _ = cKDTree(Cbd).query(Ca)
    return bool(np.mean(d < CURVE_R * np.maximum(a.rmean, 1e-9)) >= 0.5)


def _chain_support(out, chain):
    v = [
        max(r["chord_support"], r["vertex_support"]) + (1.0 if r["chart_ok"] else 0.0)
        for r in out.values()
        if r["chain"] == chain
    ]
    return float(np.mean(v)) if v else 0.0
