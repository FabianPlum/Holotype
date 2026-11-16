"""Limb tubes from the limb product of the joint method: (1) which scan surface belongs to which limb / segment (tube claims), (2) the
closed-form 'tube warp' that carries the template limb's medial polyline (ring centroids of the posed template) onto
the scan limb's centreline, arc fraction to arc fraction between the joints, with one radial scale per station.

Facts read from the limb products of the development set: limb polylines are stored tip first with `s` = arc from the tip and
`r` = distance of the centreline to the scan surface (can be slightly negative at the tip: clamped at 0), spacing about
3.5 um; usable joints lie 0.1-0.5 r from their limb centreline (median); between consecutive joints the centreline
leaves the straight chord by a median of 2.6 r (tarsus), 3.8 r (funiculus), 2.8 r (scape), 1.3 r (femur), 0.5 r (tibia):
straight bones cannot follow real limbs, the surface stage has to bend them.
"""

import numpy as np
from scipy.spatial import cKDTree

from . import geom
from .model import LIMB_CHAINS

LIMB_IDS = {k: i for i, k in enumerate(LIMB_CHAINS)}


def usable_joints(L, BL, tol_r=2.5, tol_bl=0.006):
    """joints of a limb that can anchor the warp: usable target, close to the polyline, arcs increasing distally."""
    arcs, dist = L["arcs"], L["dist"]
    r_at = np.interp(np.nan_to_num(arcs), L["s"], L["r"])
    ok = np.isfinite(arcs) & (dist <= np.maximum(tol_r * r_at, tol_bl * BL))
    last = -np.inf
    for a in range(len(arcs)):
        if ok[a]:
            if arcs[a] <= last + 1e-9:
                ok[a] = False
            else:
                last = arcs[a]
    return ok


def claims(spec, alpha=1.9, cos_min=0.25, rmin_bl=0.0015, margin_bl=0.001):
    """Per scan vertex: limb id (-1 none), bone index (-1 = segment unknown), relative radial distance. A vertex is
    claimed by the limb whose centreline it is closest to in units of the local centreline radius, when it lies within
    alpha radii, its normal points away from the centreline, and the centreline point is outside the trunk."""
    n = len(spec.Vs)
    lab_limb = np.full(n, -1, np.int64)
    lab_bone = np.full(n, -1, np.int64)
    best = np.full(n, np.inf)
    if spec.limbs is None:
        return lab_limb, lab_bone, best
    BL = spec.BL
    for limb, L in spec.limbs.limbs.items():
        if limb == "G":  # the gaster's trunk path is used for the warp only: its radius is so large that a
            continue  # claim would take the surface of legs folded against the gaster
        X, s, r = L["X"], L["s"], L["r"]
        d, k = cKDTree(X).query(spec.Vs, distance_upper_bound=alpha * max(r.max(), rmin_bl * BL) + margin_bl * BL)
        near = np.isfinite(d)
        if not near.any():
            continue
        ids = np.where(near)[0]
        kk, dd = k[ids], d[ids]
        rr = np.maximum(r[kk], rmin_bl * BL)
        rad = spec.Vs[ids] - X[kk]
        cosr = (rad * spec.Ns[ids]).sum(1) / np.maximum(dd, 1e-12)
        rho = dd / rr
        okc = (dd <= alpha * rr + margin_bl * BL) & (cosr >= cos_min) & (s[kk] >= L["core_s"])
        okc &= rho < best[ids]
        ids, kk, rho = ids[okc], kk[okc], rho[okc]
        best[ids] = rho
        lab_limb[ids] = LIMB_IDS[limb]
        ok = usable_joints(L, BL)
        arcs, js = L["arcs"], L["joints"]
        bone = np.full(len(ids), -1, np.int64)
        sa = s[kk]
        ua = [a for a in range(len(js)) if ok[a]]
        for q in range(len(ua) - 1):
            a0, a1 = ua[q], ua[q + 1]
            m = (sa >= arcs[a0]) & (sa < arcs[a1])
            if a1 == a0 + 1:
                bone[m] = js[a0]
            # joints in between are unusable: the segment is unknown (-1)
        if ua:
            if ua[0] == 0:  # proximal of the first joint (coxa base), outside the trunk
                bone[sa < arcs[0]] = js[0]
            if ua[-1] == len(js) - 1:  # beyond the tip joint
                bone[sa >= arcs[ua[-1]]] = js[-1]
        lab_bone[ids] = bone
    return lab_limb, lab_bone, best


def _station_frames(M):
    """unit tangents of a station polyline (n, 3)"""
    T = np.zeros_like(M)
    T[1:-1] = M[2:] - M[:-2]
    T[0], T[-1] = M[1] - M[0], M[-1] - M[-2]
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)


def _tangent_at(X, s, q, h):
    """unit tangent of polyline (X, arc s) at arcs q, central difference over +-h"""
    a = geom.polyline_at(X, s, q + h) - geom.polyline_at(X, s, q - h)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _rot_between_batch(U, W):
    """smallest rotations taking unit rows U to unit rows W, (n, 3, 3)"""
    k = np.cross(U, W)
    c = np.clip((U * W).sum(1), -1.0, 1.0)
    K = np.zeros((len(U), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    return np.eye(3)[None] + K + (K @ K) / np.maximum(1.0 + c, 1e-6)[:, None, None]


def tube_warp(spec, V, per_bone=4, radial=True, sig_lo=0.4, sig_hi=8.0, V_limb=None, warp_coxa=True):
    """Closed-form limb warp (a generalised cylinder map). Returns V_warped, ring targets [(w (nV,), target (3,), limb,
    scan radius)], info.
    Per limb with a limb-product polyline and at least one segment with two usable joints: the posed template limb has a medial
    polyline through its ring centroids M_k (stations: the joints' marker rings and up to `per_bone` rings per bone).
    Station k of a usable segment corresponds to the scan centreline point at the same arc fraction between the two
    joints (arc a_k) and gets a radial scale sigma_k = scan centreline radius / inner radius of the template ring
    (median-filtered along the limb; coxa and joint rings take their neighbours' value / 1). A vertex v with foot point
    m(u) on the template polyline (own segment only) and offset rho = v - m(u) goes to
        X(a(u)) + R(u) [rho_par + sigma(u) rho_perp],
    X the scan centreline at the interpolated arc, R(u) the smallest rotation from the template tangent to the scan
    tangent. Segments without two usable joints follow the nearest usable station rigidly. The result is blended with
    the vertex's summed skinning weight on the limb's bones.
    V_limb: optional second posed vertex set that supplies the LIMB geometry (stations, offsets); used to take the limb
    cross-sections from the mean template (betas = 0) instead of the fitted shape coefficients, which flatten thin
    limbs."""
    tpl = spec.tpl
    BL = spec.BL
    out = V.copy()
    Vblend = V
    if V_limb is not None:
        V = V_limb
    targets, info = [], {}
    if spec.limbs is None:
        return out, targets, info
    st_all = tpl.limb_stations(per_bone)
    for limb, L in spec.limbs.limbs.items():
        js = L["joints"]
        ok = usable_joints(L, BL)
        segs = [a for a in range(len(js) - 1) if ok[a] and ok[a + 1] and spec.support[js[a]]]
        # option: the coxa is a blob that sits against the trunk; its centreline turns into the trunk
        # core, and bending the coxa along it flips faces. warp_coxa=False leaves leg segment 0 alone.
        skip0 = (not warp_coxa) and limb[0] in "LR"
        if skip0:
            segs = [a for a in segs if a > 0]
        if not segs:
            info[limb] = dict(warped=False, why="no segment with two usable joints")
            continue
        is_g = limb == "G"
        if is_g:
            # sanity gate for the gaster's trunk path: it must run inside the scan
            if getattr(spec, "_kd", None) is None:
                spec._kd = cKDTree(spec.Vs)
            Qp = geom.polyline_at(
                L["X"], L["s"], L["arcs"][0] + np.linspace(0.1, 0.9, 17) * (L["arcs"][-1] - L["arcs"][0])
            )
            dq, iq = spec._kd.query(Qp)
            inside = ((Qp - spec.Vs[iq]) * spec.Ns[iq]).sum(1) < 0
            if inside.mean() < 0.9:
                info[limb] = dict(warped=False, why="trunk path leaves the scan", inside_share=float(inside.mean()))
                continue
            # ... and the bend is applied only to a CURLED gaster: the straight line from gaster base to apex comes
            # closer to the surface (or leaves the scan) than a quarter of the trunk path's median depth. On the 21
            # development workers that is 3 of 19 (base-apex line 0.8, 1.3 and -3.4 % BL deep; the other 16 are 2.2
            # - 4.5 % BL deep, where bending along the trunk path made the fit worse on 3 and better on none).
            fr_ = np.linspace(0.15, 0.85, 15)
            p3, p5 = spec.P[js[0]], spec.P[js[-1]]
            Qc = p3[None] + fr_[:, None] * (p5 - p3)[None]
            dc, ic = spec._kd.query(Qc)
            depth_c = np.where(((Qc - spec.Vs[ic]) * spec.Ns[ic]).sum(1) < 0, dc, -dc)
            depth_p = np.where(inside, dq, -dq)
            if not (depth_c.min() < 0.25 * np.median(depth_p)):
                info[limb] = dict(
                    warped=False,
                    why="base-apex line runs inside the gaster: not curled, no bend",
                    chord_min_depth_bl=float(depth_c.min() / BL),
                    path_median_depth_bl=float(np.median(depth_p) / BL),
                )
                continue
            # The gaster is BENT only: its base-apex line goes onto the trunk path, girth untouched. The trunk path
            # between gaster base and apex is not the gaster's medial axis (both ends lie ventrally on a globular
            # gaster), so its radius must not be compared with the template's ring radii (that shrank two globular
            # development gasters to 0.4-0.6), and the template's ring centroids must not be put on it.
            V = Vblend
        S = st_all[limb]["stations"]
        nS = len(S)
        Wst = np.stack([s_["w"] for s_ in S])
        M = Wst @ V  # template station centroids (posed)
        if is_g:  # stations on the base-apex line (see above)
            u_ = M[-1] - M[0]
            u_ = u_ / max(np.linalg.norm(u_), 1e-12)
            M = M[0][None] + ((M - M[0]) @ u_)[:, None] * u_[None]
        seg_of = np.array([s_["seg"] for s_ in S])
        jst = {s_["joint"]: k for k, s_ in enumerate(S) if s_["joint"] >= 0}
        arc_t = geom.polyline_arclen(M)
        D = _station_frames(M)
        arc = np.full(nS, np.nan)
        for a in segs:
            k0, k1 = jst[js[a]], jst[js[a + 1]]
            tau = (arc_t[k0 : k1 + 1] - arc_t[k0]) / max(arc_t[k1] - arc_t[k0], 1e-12)
            arc[k0 : k1 + 1] = L["arcs"][a] + tau * (L["arcs"][a + 1] - L["arcs"][a])
        has = np.isfinite(arc)
        hid = np.where(has)[0]
        X = np.full((nS, 3), np.nan)
        X[hid] = geom.polyline_at(L["X"], L["s"], arc[hid])
        rs = np.full(nS, np.nan)
        rs[hid] = np.interp(arc[hid], L["s"], L["r"])
        sig = np.full(nS, np.nan)
        for k in hid:
            if radial and S[k]["joint"] < 0 and (seg_of[k] > 0 or limb[0] in "AG"):
                ids = np.where(S[k]["w"] > 0)[0]
                rel = V[ids] - M[k]
                perp = rel - (rel @ D[k])[:, None] * D[k]
                rt = np.percentile(np.linalg.norm(perp, axis=1), 25) if len(ids) >= 3 else np.nan
                if np.isfinite(rt) and rt > 1e-9 and rs[k] > 1e-9:
                    sig[k] = np.clip(rs[k] / rt, sig_lo, sig_hi)
        if not radial or is_g:
            sig[hid] = 1.0
        coxa = np.array([seg_of[k] == 0 and limb[0] in "LR" for k in range(nS)])
        sig[coxa & has] = 1.0
        for k in hid:  # joint rings / failed rings: neighbours' median
            if not np.isfinite(sig[k]):
                nb = [sig[j] for j in hid if 0 < abs(j - k) <= 2 and np.isfinite(sig[j]) and not coxa[j]]
                sig[k] = float(np.median(nb)) if nb else 1.0
        sg = sig.copy()
        for q, k in enumerate(hid):  # median of 3 along the limb
            if coxa[k]:
                continue
            nb = [sig[j] for j in hid[max(q - 1, 0) : q + 2] if not coxa[j]]
            sg[k] = float(np.median(nb))
        sig = sg
        h = 0.5 * float(np.median(np.diff(arc[hid]))) if len(hid) > 1 else 0.01 * BL
        Dscan = np.zeros((nS, 3))
        Dscan[hid] = _tangent_at(L["X"], L["s"], arc[hid], max(h, 1e-6))
        # ---- vertices
        chain_w = tpl.W[:, js[1:] if skip0 else js].sum(1)
        vid = np.where(chain_w > 0.02)[0]
        pos = {j: a for a, j in enumerate(js)}
        seg_v = np.array([min(pos[tpl.dom[i]], len(js) - 2) if tpl.dom[i] in pos else 0 for i in vid])
        newv = V[vid].copy()
        usable_seg = set(segs)
        for a in range(len(js) - 1):
            m = seg_v == a
            if not m.any() or (skip0 and a == 0):
                continue
            P = V[vid[m]]
            if a not in usable_seg:
                # rigid continuation with the map of the nearest station that has a target
                kj = jst[js[a]] if abs(jst[js[a]] - hid).min() <= abs(jst[js[a + 1]] - hid).min() else jst[js[a + 1]]
                j = hid[np.argmin(np.abs(hid - kj))]
                R = geom.rot_between(D[j], Dscan[j])
                newv[m] = (P - M[j]) @ R.T + X[j]
                continue
            ks = np.arange(jst[js[a]], jst[js[a + 1]] + 1)
            Mk = M[ks]
            best = np.full(m.sum(), np.inf)
            qi = np.zeros(m.sum(), int)
            fr = np.zeros(m.sum())
            for q in range(len(ks) - 1):
                p0, u = Mk[q], Mk[q + 1] - Mk[q]
                f = np.clip(((P - p0) @ u) / max(u @ u, 1e-18), 0, 1)
                dd = np.linalg.norm(P - (p0 + f[:, None] * u), axis=1)
                b = dd < best
                best[b], qi[b], fr[b] = dd[b], q, f[b]
            ka, kb = ks[qi], ks[qi + 1]
            foot = (1 - fr)[:, None] * M[ka] + fr[:, None] * M[kb]
            dt = (1 - fr)[:, None] * D[ka] + fr[:, None] * D[kb]
            dt /= np.maximum(np.linalg.norm(dt, axis=1, keepdims=True), 1e-12)
            av = (1 - fr) * arc[ka] + fr * arc[kb]
            Xv = geom.polyline_at(L["X"], L["s"], av)
            Ds = _tangent_at(L["X"], L["s"], av, max(h, 1e-6))
            sv = (1 - fr) * sig[ka] + fr * sig[kb]
            rho = P - foot
            par = (rho * dt).sum(1, keepdims=True) * dt
            y = par + sv[:, None] * (rho - par)
            newv[m] = Xv + np.einsum("nab,nb->na", _rot_between_batch(dt, Ds), y)
        out[vid] = Vblend[vid] + chain_w[vid, None] * (newv - Vblend[vid])
        if not is_g:  # ring centroids are medial; the gaster's path is not
            for k in hid:
                targets.append((S[k]["w"], X[k], limb, float(rs[k])))
        if is_g and V_limb is not None:
            V = V_limb
        # segment diagnostics from the centreline alone: a sharp bend or a constriction INSIDE a segment (not at its
        # ends) means an articulation lies inside it, i.e. a joint was put down the neighbouring segment
        seg_diag = {}
        for a in segs:
            s0, s1 = L["arcs"][a], L["arcs"][a + 1]
            if s1 - s0 < 1e-9:
                continue
            q = np.linspace(s0 + 0.2 * (s1 - s0), s1 - 0.2 * (s1 - s0), 25)
            rq = np.interp(q, L["s"], L["r"])
            win = max(1.5 * float(np.median(rq)), 0.05 * (s1 - s0))
            t0 = _tangent_at(L["X"], L["s"], q - win, 0.5 * win)
            t1 = _tangent_at(L["X"], L["s"], q + win, 0.5 * win)
            ang = np.degrees(np.arccos(np.clip((t0 * t1).sum(1), -1, 1)))
            r_all = np.interp(np.linspace(s0, s1, 41), L["s"], L["r"])
            seg_diag[tpl.names[js[a]]] = dict(
                kink_deg=float(ang.max()),
                r_min_rel=float(rq.min() / max(np.median(r_all), 1e-12)),
                len_bl=float((s1 - s0) / BL),
                r_med_bl=float(np.median(r_all) / BL),
            )
        info[limb] = dict(
            warped=True,
            segments=[tpl.names[js[a]] for a in segs],
            n_stations=int(has.sum()),
            seg_diag=seg_diag,
            sigma=[float(x) for x in sig[hid]],
            r_scan_bl=[float(x / BL) for x in rs[hid]],
            move_bl=float(np.median(np.linalg.norm(X[hid] - M[hid], axis=1)) / BL),
        )
    return out, targets, info
