"""Tubes of one specimen: template charts (computed once), scan tubes traced from the joints, chart transfer.

    tc = template_charts()                       # per tube: template Tube, reference normal, vertex chart coordinates
    st = trace_scan_tubes(scan, tpl, tc, bf)     # per tube: scan Tube or None (unresolved), twist, QA numbers
    X  = transfer(tc[k], st[k], ...)             # scan positions of the template vertices charted by tube k

Chart transfer of a template vertex with chart coordinates (t, phi, rho, ax) of tube k:
    x = C_s(t) + ax * (L_s / L_t) * T_s(t) + rho * ratio * dir_s(t, phi + psi)
    ratio = R_s(t, phi + psi) / R_t(t, phi) for vertices on the wall (rho <= R_t), fading to the ratio of the section
            means for vertices far from the wall (rho >= 2 R_t; these belong to other parts and only carry a small
            skinning weight of this bone), clamped to [0.5, 2] x the section-mean ratio.
t is pinned at both joints (t = 0, 1 are the projections of the joints onto the medial curves; end tubes: t = 1 at
the tip), so nothing slides along the tube. psi is the twist from the profile correlation (0 when not evident).
"""

import functools
import hashlib
import warnings

import numpy as np

from . import charts
from .charts import N_CENT, N_IN, trace_tube, tube_defs
from .model import template
from .scan import Scan
from .sets import cache_file

W_EPS = 1e-6
MIN_OK = 0.30  # share of centring sections that must be centred for a chart to be used (5 of 17)
MIN_VALID = 0.30  # share of the polar profile that must see the tube's own wall
PROF_TOL = 2.0  # accepted ratio between the profile radius of the final curve and the centred sections' radius
RR_OUT = 2.5  # accepted ratio between a tube's girth ratio and that of the same segment on the other limbs
GIRTH_FROM_VALID = True  # girth ratio per section from the directions that see the tube's own wall (C12)
AX_SUP = 2.0  # the chart fades to the bone's affine map over this many template end radii beyond a joint ...
RHO_SUP = 4.0  # ... and between 2 and this many template wall radii away from the curve
WALL_CLAMP = (0.8, 1.25)  # relative radius of a tube's own vertices in the chart transfer
_WALL = {"on": True}


GUESS_SCALES = (1.0, 3.0, 1.0 / 3.0)  # radius-guess multipliers tried in this order
TAU = np.linspace(0.08, 0.92, N_CENT)  # the centring stations of charts.trace_tube (chord fractions)
TT = np.linspace(0, 1, N_IN)


def _ref_normal(u):
    return np.array([0.0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([0.0, 1.0, 0])


def _radius_guess(V, A, B):
    """Radius about the centroid of the part's vertices in 8 axial bins (the joint chord can lie outside a curved
    tube, so distances to the chord are no radius)."""
    L = np.linalg.norm(B - A)
    u = (B - A) / L
    rel = V - A
    ax = rel @ u
    rb = []
    for lo in np.linspace(0, 1, 9)[:-1]:
        m = (ax >= lo * L) & (ax < (lo + 0.125) * L)
        if m.sum() >= 3:
            q = V[m] - V[m].mean(0)
            rb.append(np.median(np.linalg.norm(q - (q @ u)[:, None] * u, axis=1)))
    if rb:
        return float(np.median(rb))
    lat = np.linalg.norm(rel - ax[:, None] * u, axis=1)
    return float(np.median(lat) / 1.41)


@functools.lru_cache(maxsize=2)
def template_charts(kind=None):
    """List of dicts, one per tube (charts.tube_defs order): bone, child, end, merged, name, tube (template Tube, model
    units), ref (reference normal, model frame), vid / w (vertices with merged skinning weight > 0 and the weight),
    t, phi, rho, ax (their chart coordinates), rT (wall radius of the template at their t, phi), rguess (radius at
    the centring stations), own (vertices whose dominant merged bone is this tube)."""
    tpl = template(kind)
    # the outer-face mask of the template is cached on disk: without the cache every worker process would run the
    # winding-number test through torch. The file name carries a hash of the rest mesh, so another model (even one
    # with the same faces) never reuses the mask of the first one.
    key = hashlib.md5(np.ascontiguousarray(tpl.V0).tobytes() + np.ascontiguousarray(tpl.F).tobytes()).hexdigest()[:16]
    surf = Scan(tpl.V0.copy(), tpl.F.copy(), tpl.rest_body_length, cache=cache_file(f"template_outer_{key}.npz"))
    own = charts.owners(tpl)
    Wm = charts.merged_weights(tpl)
    dom_m = own[tpl.dom]
    out = []
    for d in tube_defs(tpl):
        d = dict(d)
        j, c = d["bone"], d["child"]
        A, B = tpl.J0[j], tpl.J0[c]
        u = (B - A) / np.linalg.norm(B - A)
        ownv = np.where(dom_m == j)[0]
        r0 = _radius_guess(tpl.V0[ownv], A, B)
        ref = _ref_normal(u)
        tb = trace_tube(surf, A, B, r0, ref, end=d["end"])
        vid = np.where(Wm[:, j] > W_EPS)[0]
        t, phi, rho, ax = tb.coords(tpl.V0[vid])
        d.update(
            tube=tb,
            ref=ref,
            vid=vid,
            w=Wm[vid, j],
            t=t,
            phi=phi,
            rho=rho,
            ax=ax,
            rT=tb.radius(t, phi),
            rguess=np.interp(TAU, TT, tb.rmean),
            own=ownv,
            r0=r0,
        )
        d["sup"] = support(d)
        d["own_mask"] = np.isin(vid, ownv)
        out.append(d)
    return out


def support(d):
    """Share of the chart in the position of each template vertex of tube `d`: 1 on the tube's own surface, 0 far
    beyond the joints (AX_SUP end radii; not beyond the tip of an end tube) or far outside the wall (RHO_SUP wall
    radii), where the vertex belongs to a neighbouring part and the bone's affine map takes over."""
    tT = d["tube"]
    t, rho, ax, rT = d["t"], d["rho"], d["ax"], d["rT"]
    r0, r1 = max(float(tT.rmean[0]), 1e-12), max(float(tT.rmean[-1]), 1e-12)
    over0 = (np.clip(-t, 0, None) * tT.length + np.clip(-ax, 0, None)) / r0
    over1 = (np.clip(t - 1, 0, None) * tT.length + np.clip(ax, 0, None)) / r1
    if d["end"]:
        over1 = np.zeros_like(over1)
    sup_ax = np.clip(1.0 - np.maximum(over0, over1) / AX_SUP, 0, 1)
    sup_rho = np.clip((RHO_SUP - rho / np.maximum(rT, 1e-12)) / (RHO_SUP - 2.0), 0, 1)
    return sup_ax * sup_rho


class ScanTube:
    """Result of tracing one tube on the scan."""

    def __init__(self, d):
        self.name, self.bone, self.child, self.end = d["name"], d["bone"], d["child"], d["end"]
        self.tube = None
        self.resolved = False  # both joints present and the bone not unresolved
        self.traced = False  # the medial curve was centred on the scan
        self.use = False  # the chart of this tube is used (traced and plausible); else bone-wise fallback
        self.psi = 0.0
        self.psi_evidence = 0.0
        self.m = None  # girth ratio per station (girth_profile) and its tube-wide value
        self.mg = float("nan")
        self.qa = {}


def trace_scan_tubes(scan, tpl, tc, bf, twist=True):
    """Trace every resolved tube on the scan from the joints. `bf`: solved bones.BoneFit (scale, bone rotations)."""
    out = []
    for d in tc:
        st = ScanTube(d)
        j, c = d["bone"], d["child"]
        out.append(st)
        if bf.unres[j] or not bf.ok[j] or not bf.ok[c]:
            continue
        st.resolved = True
        A, B = bf.P[j], bf.P[c]
        ref = bf.GR[j] @ d["ref"]
        # the template's relative girths are not the scan's (real scapes are 5-7 times thicker than the template's,
        # real tibiae 2-3 times): when the chart at the scaled template radius is not self-consistent, the radius
        # guess is retried at GUESS_SCALES and the first self-consistent trace is taken
        mid = (TT >= 0.1) & (TT <= 0.9)
        best = None
        for gs in GUESS_SCALES:
            tb = trace_tube(scan, A, B, d["rguess"] * bf.s * gs, ref, end=d["end"])
            traced = bool(tb.traced)
            prof = (
                float(np.median(tb.rmean[mid]))
                if tb.rmean is not None and np.isfinite(tb.rmean[mid]).any()
                else float("nan")
            )
            cons = prof / tb.rad_ok if traced and tb.rad_ok > 0 else float("nan")
            use = bool(
                traced
                and tb.ok_share >= MIN_OK
                and tb.valid.mean() >= MIN_VALID
                and np.isfinite(cons)
                and 1.0 / PROF_TOL < cons < PROF_TOL
            )
            if best is None or use or (not best[1] and tb.ok_share > best[0].ok_share):
                best = (tb, use, cons, gs)
            if use:
                break
        tb, st.use, cons, gs_used = best
        st.tube = tb
        st.traced = bool(tb.traced)
        tT = d["tube"]
        if st.traced and twist:
            st.psi, st.psi_evidence = charts.twist_from_profiles(tT, tb)
        st.m, st.mg = girth_profile(tT, tb, st.psi)
        chord = float(np.linalg.norm(B - A))
        st.qa = dict(
            traced=st.traced,
            use=st.use,
            guess_scale=float(gs_used),
            profile_vs_sections=float(cons),
            ok_share=float(tb.ok_share),
            length=float(tb.length),
            chord=chord,
            length_ratio=float(tb.length / (tT.length * bf.s)),
            valid_share=float(tb.valid.mean()),
            radius_ratio=float(np.median(tb.rmean / (tT.rmean * bf.s))),
            psi_deg=float(np.rad2deg(st.psi)),
            psi_evidence=float(st.psi_evidence),
            tip_found=tb.tip_found,
            note=tb.note.strip(),
        )
        st.qa["inner_step"] = inner_step(tb)
    _girth_outliers(out)
    return out


def inner_step(tb, lo=0.25, hi=0.75, half=2):
    """Largest radius step INSIDE a tube: max over stations in [lo, hi] of |log(r[i + half] / r[i - half])| of the
    3-point smoothed section radius (window = 4 of 32 station steps = 12.5 % of the length). A segment between two
    real articulations has a smooth profile; a joint that was put down the neighbouring segment leaves the real
    articulation (thick tibia apex -> thin basitarsus) as a step inside the tube. None when the tube is not traced."""
    if tb is None or not tb.traced or tb.rmean is None:
        return None
    r = np.convolve(np.maximum(tb.rmean, 1e-9), np.ones(3) / 3, mode="same")
    r[0], r[-1] = tb.rmean[0], tb.rmean[-1]
    i = np.where((TT >= lo) & (TT <= hi))[0]
    return float(np.max(np.abs(np.log(r[i + half] / r[i - half]))))


def girth_profile(tT, tS, psi=0.0):
    """m (N_IN,): ratio between the scan's and the template's wall radius per station, the median over the directions
    in which the scan shows this tube's own wall; stations with little visible wall (a contact) lean on the tube-wide
    ratio mg. Hidden wall is continued with this profile: the template's section shape at the measured scale."""
    sh = int(round(psi / (2 * np.pi) * charts.K_ANG))
    Rt = np.roll(tT.Rf, sh, axis=1)
    mid = (TT >= 0.1) & (TT <= 0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = np.where(tS.valid, tS.R / np.maximum(Rt, 1e-12), np.nan)
        m = np.nanmedian(ratio, axis=1)
        mg = np.nanmedian(ratio[mid]) if np.isfinite(ratio[mid]).any() else np.nanmedian(ratio)
    if not np.isfinite(mg):
        mg = float(np.median(tS.rmean / tT.rmean))
    wt = np.clip((tS.valid.mean(1) - 0.25) / 0.5, 0, 1)
    m = np.where(np.isfinite(m), wt * m + (1 - wt) * mg, mg)
    return m, float(mg)


def _segment_type(name):
    """l_2_fe_r -> fe; an_1_l -> an_1; b_a_1 -> b_a_1 (trunk tubes have no serial partner)."""
    p = name.split("_")
    if p[0] == "l":
        return p[2]
    if p[0] == "an":
        return "an_" + p[1]
    return name


ASPECT_MAX = 0.6  # a tube without a comparable partner is rejected when its radius exceeds this share of its length


def _girth_outliers(st):
    """A tube whose girth (relative to the template's) is far from that of the same segment type on the specimen's
    other limbs was traced inside a neighbouring body (a femur lying on the gaster): its chart is not used. The
    reference is the median over the other used tubes of the same segment type (the contralateral scape for a scape:
    template girths are not comparable ACROSS segment types, the template's scape is 5-7 times thinner than real
    ones). A tube without any partner of its type is only checked for being a tube at all (radius / length)."""
    used = [s for s in st if s.use and np.isfinite(s.qa.get("radius_ratio", np.nan))]
    for s in used:
        same = [o.qa["radius_ratio"] for o in used if o is not s and _segment_type(o.name) == _segment_type(s.name)]
        bad = False
        if same:
            r = s.qa["radius_ratio"] / max(float(np.median(same)), 1e-9)
            s.qa["girth_vs_limbs"] = float(r)
            bad = r > RR_OUT or r < 1.0 / RR_OUT
        else:
            asp = float(np.median(s.tube.rmean)) / max(float(s.tube.length), 1e-9)
            s.qa["aspect"] = asp
            bad = asp > ASPECT_MAX
        if bad:
            s.use = False
            s.qa["use"] = False
            s.qa["note"] = (s.qa.get("note", "") + " girth outlier against the other limbs").strip()


# ---- stage A: scan surface claimed by the tubes ---------------------------------------------------------------------
WALL_TOL = 0.35  # a scan vertex is on the wall of a tube when |rho - R(t, phi)| < WALL_TOL * R(t, phi)
T_EXT = 0.12  # claims reach this far beyond the cuts (share of the tube length); the smaller cost wins
CLAIM_MAX = 1.0  # largest accepted claim cost


def tube_claims(scan, s):
    """Scan vertices on the wall of traced tube `s`: ids, cost (wall distance in local radii + distance beyond the
    cuts in local radii), t, phi."""
    tb = s.tube
    ks = np.arange(charts.N_EXT, charts.N_EXT + N_IN)
    P = tb.C[ks]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # stations without any valid direction
        rmax = np.nanmax(np.where(tb.valid, tb.R, np.nan), axis=1) if tb.valid.any() else tb.rmean
    rmax = np.where(np.isfinite(rmax), rmax, tb.rmean)
    step = tb.length / (N_IN - 1.0)
    kd = scan.kd
    cand = set()
    for k in range(N_IN):
        cand.update(kd.query_ball_point(P[k], 1.5 * rmax[k] + step))
    if not cand:
        z = np.zeros(0)
        return np.zeros(0, int), z, z, z
    ids = scan._kd_ids[np.fromiter(cand, int)]
    X = scan.V[ids]
    t, phi, rho, ax = tb.coords(X)
    R = tb.radius(t, phi)
    ok = tb.valid_at(t, phi)
    C, dirv = tb.direction(t, phi)
    facing = (scan.vn[ids] * dirv).sum(1) > 0.0
    over = np.maximum(0, np.maximum(-t, t - 1.0))
    if s.end:  # a tip closes the tube: nothing lies beyond t = 1
        over = np.maximum(0, -t)
    rloc = np.maximum(R, 1e-9)
    cost = np.abs(rho - R) / (WALL_TOL * rloc) * 0.5 + over * tb.length / rloc * 0.5
    m = ok & facing & (np.abs(rho - R) < WALL_TOL * rloc) & (over < T_EXT) & (cost < CLAIM_MAX)
    return ids[m], cost[m], t[m], phi[m]
