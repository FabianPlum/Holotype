"""Stages A and C for tubular parts: centred medial curves, polar radius profiles and the chart transfer.

A tube is the part between two joints (or between a joint and the tip of the part). On the template and on the scan
the same procedure produces
  * a centred medial curve (ray-centring of cross-sections, started from the chord between the two joints; the
    joints only pin the axial coordinate: t = 0 at the projection of the first joint onto the curve, t = 1 at the
    projection of the second joint, or at the geometric tip for end tubes),
  * a rotation-minimising frame along the curve, anchored at mid length,
  * the polar radius profile r(t, phi) (first exit of a ray from the curve; directions that run into another part are
    detected as "long" and marked invalid = hidden wall).
A template vertex has chart coordinates (t, phi, relative radius) and is transferred to the point with the same
coordinates on the scan. The lateral offset of a joint from the medial curve is never used (joint files differ in
whether joints sit on the surface or on the medial axis).
"""

import warnings

import numpy as np

from . import geom

K_ANG = 64  # directions of the polar profile
N_IN = 33  # stations for t in [0, 1]
N_EXT = 8  # extension stations on each side (t in [-0.25, 0) and (1, 1.25])
N_CENT = 17  # stations used for centring
K_CEN = 32  # directions used for centring
JUMP_C = 1.8  # centring: neighbouring rays whose lengths differ by more than this factor are not on the same wall
JUMP_P = 1.5  # profile (K_ANG directions): same, finer angular step (an ellipse of aspect 8 stays below it)
MIN_ARC = 0.25  # an arc of wall must cover this share of the directions to count as this tube's wall
MIN_FIT = 0.40  # share of valid directions needed to centre a section
SMOOTH_R = 2.0  # kernel width of the medial-curve regression, in local radii
R_TOL = 6.0  # accepted ratio between the measured radius and the radius guess (scaled template), both ways
RUN_SLOPE = 1.0  # two section centres belong to the same curve when their lateral distance is below
RUN_R = 1.0  # RUN_SLOPE x their axial distance + RUN_R x the smaller of their radii
TIP_BACK = 0.15  # a geometric tip is accepted at most this share of the chord before the leaf joint
R_CAP = 4.0  # a wall direction longer than this many times the lower-quartile radius of its section (running
# median over +-2 stations) is another part behind a contact, not this tube's wall


def tube_defs(tpl):
    """Tubes of the template: bone = owner (first joint), child = joint that gives the direction, end = the tube ends
    in a tip (the child is a leaf joint), merged = bones whose vertices are charted by this tube as well."""
    ix = tpl.idx
    out = []
    for side in "rl":
        for leg in (1, 2, 3):
            for a, b in (("co", "tr"), ("tr", "fe"), ("fe", "ti"), ("ti", "ta")):
                out.append(dict(bone=ix[f"l_{leg}_{a}_{side}"], child=ix[f"l_{leg}_{b}_{side}"], end=False, merged=[]))
            out.append(
                dict(
                    bone=ix[f"l_{leg}_ta_{side}"],
                    child=ix[f"l_{leg}_pt_{side}"],
                    end=True,
                    merged=[ix[f"l_{leg}_pt_{side}"]],
                )
            )
        out.append(dict(bone=ix[f"an_1_{side}"], child=ix[f"an_2_{side}"], end=False, merged=[]))
        out.append(dict(bone=ix[f"an_2_{side}"], child=ix[f"an_3_{side}"], end=True, merged=[ix[f"an_3_{side}"]]))
    out.append(dict(bone=ix["b_a_1"], child=ix["b_a_2"], end=False, merged=[]))
    out.append(dict(bone=ix["b_a_2"], child=ix["b_a_3"], end=False, merged=[]))
    out.append(dict(bone=ix["b_a_3"], child=ix["b_a_5"], end=True, merged=[ix["b_a_4"], ix["b_a_5"]]))
    for d in out:
        d["name"] = tpl.names[d["bone"]]
    return out


def owners(tpl):
    """owner[j]: the bone whose part bone j belongs to (leaf bones are merged into their tube)."""
    own = np.arange(tpl.nJ)
    for d in tube_defs(tpl):
        for m in d["merged"]:
            own[m] = d["bone"]
    return own


def merged_weights(tpl):
    own = owners(tpl)
    Wm = np.zeros_like(tpl.W)
    for j in range(tpl.nJ):
        Wm[:, own[j]] += tpl.W[:, j]
    return Wm


def _perp_basis(T):
    a = np.where(np.abs(T[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    U = np.cross(T, a)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)
    return U, V


def _snap_inside(surf, O, r0):
    """Move points that are not inside the solid to the middle of the solid behind the closest surface point."""
    _, k = surf.kd.query(O)
    vid = surf._kd_ids[k]
    q, n = surf.V[vid], surf.vn[vid]
    d, f, c = surf.first_hit(q - 1e-3 * r0[:, None] * n, -n)
    half = np.where(np.isfinite(d) & (c > 0), d / 2, r0)
    half = np.minimum(half, 3 * r0)
    return q - half[:, None] * n


def valid_arc(d, ex, jump):
    """Which rays of a section (rows of d: lengths, ex: the ray leaves the solid there) see the tube's own wall.
    Neighbouring directions are linked when both exist and their lengths differ by less than the factor `jump`; the
    links cut the circle into arcs. The wall is the arc with the smallest median length among the arcs that cover at
    least MIN_ARC of the directions (rays that run on into a touching part or into the next part form their own,
    longer arc); without such an arc the section has no valid direction. Smooth non-circular sections stay whole."""
    n, K = d.shape
    out = np.zeros((n, K), bool)
    dn, exn = np.roll(d, -1, axis=1), np.roll(ex, -1, axis=1)
    with np.errstate(all="ignore"):
        link = ex & exn & (np.maximum(d, dn) < jump * np.minimum(d, dn))  # link between direction k and k + 1
    for i in range(n):
        if not ex[i].any():
            continue
        if link[i].all():
            out[i] = True
            continue
        brk = np.where(~link[i])[0]
        order = (brk[-1] + 1 + np.arange(K)) % K  # once around, starting behind a cut
        lab = np.concatenate([[0], np.cumsum(~link[i][order[:-1]])])
        best, best_med = None, np.inf
        for a in range(lab[-1] + 1):
            idx = order[lab == a]
            idx = idx[ex[i, idx]]
            if len(idx) >= MIN_ARC * K:
                med = np.median(d[i, idx])
                if med < best_med:
                    best, best_med = idx, med
        if best is not None:
            out[i, best] = True
    return out


def centre_sections(surf, O, T, r0, iters=6, K=K_CEN):
    """Centre the points O in the cross-sections perpendicular to T: rays in the section plane, the valid wall arc
    (valid_arc), a circle fit to the wall points (works on partial arcs; for a symmetric section the centre is its
    fixed point), move, repeat. Points outside the solid are first moved inside. Returns centres, radius (mean valid
    ray from the final centre), ok flag, number of valid rays."""
    n = len(O)
    U, V = _perp_basis(T)
    ang = np.arange(K) * 2 * np.pi / K
    ca, sa = np.cos(ang), np.sin(ang)
    D = ca[None, :, None] * U[:, None, :] + sa[None, :, None] * V[:, None, :]
    O = O.copy()
    r0 = np.broadcast_to(np.asarray(r0, dtype=np.float64), (n,)).copy()
    rad = np.full(n, np.nan)
    good = np.zeros(n, bool)
    nval = np.zeros(n, int)
    for it in range(iters):
        d, f, c = surf.first_hit(np.repeat(O, K, 0), D.reshape(-1, 3))
        d = d.reshape(n, K)
        ex = (c.reshape(n, K) > 0) & np.isfinite(d)
        inside = ex.sum(1) >= 0.7 * K
        out = ~inside
        good[out] = False
        if out.any() and it < iters - 1:
            O[out] = _snap_inside(surf, O[out], r0[out])
        ii = np.where(inside)[0]
        if not len(ii):
            continue
        val = valid_arc(np.where(ex[ii], d[ii], np.inf), ex[ii], JUMP_C)
        for m, i in enumerate(ii):
            v = val[m]
            nval[i] = v.sum()
            if v.sum() < MIN_FIT * K:
                good[i] = False
                continue
            x, y = d[i, v] * ca[v], d[i, v] * sa[v]
            A = np.stack([2 * x, 2 * y, np.ones(len(x))], 1)
            sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
            e = sol[:2]
            lim = np.median(d[i, v])
            ne = np.linalg.norm(e)
            if ne > lim:  # never move further than one radius per iteration
                e = e * (lim / ne)
            if it < iters - 1:
                O[i] += e[0] * U[i] + e[1] * V[i]
            rad[i] = d[i, v].mean()
            good[i] = ne < 0.25 * lim or it < iters - 1
    return O, rad, good, nval


class Tube:
    """Traced tube. Stations k = 0 .. N_EXT + N_IN + N_EXT - 1 at t = (k - N_EXT) / (N_IN - 1)."""

    n = N_IN + 2 * N_EXT
    t = (np.arange(N_IN + 2 * N_EXT) - N_EXT) / (N_IN - 1.0)

    def __init__(self):
        self.C = self.T = self.N = self.B = None
        self.R = self.valid = self.Rf = None
        self.length = 0.0  # arc length between t = 0 and t = 1
        self.traced = False  # the medial curve was found on the surface (else: fallback curve)
        self.ok_share = 0.0  # share of the centring stations that were centred
        self.tip_found = None
        self.rmean = None
        self.note = ""

    # -- geometry -------------------------------------------------------------------------------------------
    def set_curve(self, C, ref_normal):
        self.C = C
        T, N, B = geom.rmf(C, ref_normal)
        self.T, self.N, self.B = T, N, B
        self.anchor(ref_normal)

    def anchor(self, ref):
        """Rotate the frame about the curve so that N at mid length is the projection of `ref`."""
        k = self.n // 2
        d = np.arctan2(ref @ self.B[k], ref @ self.N[k])
        self.rotate(d)

    def rotate(self, d):
        c, s = np.cos(d), np.sin(d)
        N = c * self.N + s * self.B
        B = -s * self.N + c * self.B
        self.N, self.B = N, B

    def frame_at(self, t):
        x = np.clip((np.asarray(t) * (N_IN - 1.0) + N_EXT), 0, self.n - 1 - 1e-9)
        k = np.floor(x).astype(int)
        f = (x - k)[:, None]
        C = (1 - f) * self.C[k] + f * self.C[k + 1]
        N = (1 - f) * self.N[k] + f * self.N[k + 1]
        B = (1 - f) * self.B[k] + f * self.B[k + 1]
        T = (1 - f) * self.T[k] + f * self.T[k + 1]
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        return C, T, N, B

    def direction(self, t, phi):
        C, T, N, B = self.frame_at(t)
        return C, np.cos(phi)[:, None] * N + np.sin(phi)[:, None] * B

    def coords(self, X):
        """Chart coordinates of points: t (clipped to the station range), phi, rho (distance from the curve),
        ax (signed axial overshoot beyond the station range, 0 inside)."""
        A, Bp = self.C[:-1], self.C[1:]
        seg = Bp - A
        L2 = (seg * seg).sum(1)
        best = np.full(len(X), np.inf)
        tk = np.zeros(len(X))
        for k in range(len(A)):
            w = np.clip(((X - A[k]) @ seg[k]) / L2[k], 0, 1)
            d = np.linalg.norm(X - (A[k] + w[:, None] * seg[k]), axis=1)
            upd = d < best
            best[upd] = d[upd]
            tk[upd] = (k + w[upd] - N_EXT) / (N_IN - 1.0)
        C, T, N, B = self.frame_at(tk)
        dv = X - C
        ax = (dv * T).sum(1)
        dp = dv - ax[:, None] * T
        phi = np.arctan2((dp * B).sum(1), (dp * N).sum(1))
        return tk, phi, np.linalg.norm(dp, axis=1), ax

    def radius(self, t, phi, filled=True):
        """Bilinear lookup of the polar profile (stations of t in [0,1] only; t is clipped)."""
        G = self.Rf if filled else self.R
        x = np.clip(np.asarray(t), 0, 1) * (N_IN - 1.0)
        k = np.minimum(np.floor(x).astype(int), N_IN - 2)
        f = x - k
        y = (np.asarray(phi) % (2 * np.pi)) / (2 * np.pi) * K_ANG
        a = np.floor(y).astype(int) % K_ANG
        g = y - np.floor(y)
        a1 = (a + 1) % K_ANG
        return (1 - f) * (1 - g) * G[k, a] + (1 - f) * g * G[k, a1] + f * (1 - g) * G[k + 1, a] + f * g * G[k + 1, a1]

    def valid_weight(self, t, phi):
        """Bilinear lookup of the validity of the profile (1: this tube's own wall is seen in that direction)."""
        G, self.Rf = self.Rf, self.valid.astype(np.float64)
        try:
            return self.radius(t, phi)
        finally:
            self.Rf = G

    def valid_at(self, t, phi):
        x = np.clip(np.asarray(t), 0, 1) * (N_IN - 1.0)
        k = np.clip(np.rint(x).astype(int), 0, N_IN - 1)
        a = np.rint((np.asarray(phi) % (2 * np.pi)) / (2 * np.pi) * K_ANG).astype(int) % K_ANG
        return self.valid[k, a]

    # -- perimeter parametrisation of the sections -------------------------------------------------------------
    def perimeter_table(self):
        """U (N_IN, K_ANG + 1): cumulative perimeter fraction of every section at the K_ANG grid angles (0 at phi = 0,
        1 at 2 pi), from the filled profile averaged over +-1 station."""
        if getattr(self, "_U", None) is None:
            R = self.Rf.copy()
            R[1:-1] = (self.Rf[:-2] + self.Rf[1:-1] + self.Rf[2:]) / 3.0
            ang = np.arange(K_ANG + 1) * 2 * np.pi / K_ANG
            Rc = np.concatenate([R, R[:, :1]], axis=1)
            x, y = Rc * np.cos(ang), Rc * np.sin(ang)
            ds = np.sqrt(np.diff(x, axis=1) ** 2 + np.diff(y, axis=1) ** 2) + 1e-12
            U = np.concatenate([np.zeros((len(R), 1)), np.cumsum(ds, axis=1)], axis=1)
            self._U = U / U[:, -1:]
        return self._U

    def angle_to_u(self, t, phi):
        """Perimeter fraction (0..1) of direction phi on the section at t (nearest station)."""
        U = self.perimeter_table()
        k = np.clip(np.rint(np.clip(np.asarray(t), 0, 1) * (N_IN - 1.0)).astype(int), 0, N_IN - 1)
        y = (np.asarray(phi) % (2 * np.pi)) / (2 * np.pi) * K_ANG
        a = np.minimum(np.floor(y).astype(int), K_ANG - 1)
        g = y - a
        return (1 - g) * U[k, a] + g * U[k, a + 1]

    def u_to_angle(self, t, u):
        """Direction with perimeter fraction u on the section at t (nearest station)."""
        U = self.perimeter_table()
        k = np.clip(np.rint(np.clip(np.asarray(t), 0, 1) * (N_IN - 1.0)).astype(int), 0, N_IN - 1)
        u = np.clip(np.asarray(u), 0, 1 - 1e-9)
        out = np.zeros(len(u))
        for kk in np.unique(k):
            m = k == kk
            a = np.clip(np.searchsorted(U[kk], u[m], side="right") - 1, 0, K_ANG - 1)
            g = (u[m] - U[kk, a]) / np.maximum(U[kk, a + 1] - U[kk, a], 1e-12)
            out[m] = (a + g) * 2 * np.pi / K_ANG
        return out

    # -- the polar profile ----------------------------------------------------------------------------------
    def measure(self, surf, r_exp=None):
        """Polar profile. `r_exp` (N_IN,): radius of the tube as seen by its centred sections, interpolated along t
        (None: unknown); together with the section's own thin side it bounds what can be this tube's wall."""
        ks = np.arange(N_EXT, N_EXT + N_IN)
        ang = np.arange(K_ANG) * 2 * np.pi / K_ANG
        D = np.cos(ang)[None, :, None] * self.N[ks][:, None, :] + np.sin(ang)[None, :, None] * self.B[ks][:, None, :]
        d, f, c = surf.first_hit(np.repeat(self.C[ks], K_ANG, 0), D.reshape(-1, 3))
        d = d.reshape(N_IN, K_ANG)
        ex = (c.reshape(N_IN, K_ANG) > 0) & np.isfinite(d)
        valid = valid_arc(np.where(ex, d, np.inf), ex, JUMP_P)
        R = np.where(valid, d, np.nan)
        if R_CAP > 0:
            # rays that run on into a neighbouring part through a contact (a folded knee) grow smoothly from one
            # direction to the next, so the arc test keeps them: cap them against the section's own thin side
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q25 = np.nanpercentile(R, 25, axis=1)
                q25s = np.array([np.nanmedian(q25[max(0, i - 2) : i + 3]) for i in range(N_IN)])
                if np.isfinite(q25s).any():
                    q25s = np.where(np.isfinite(q25s), q25s, np.nanmedian(q25s))
                    if r_exp is not None:
                        # a section that lies inside another part sees only long rays, its thin side included: the
                        # radius the centred sections saw (interpolated along the tube) bounds it as well
                        q25s = np.minimum(q25s, np.asarray(r_exp, dtype=np.float64))
                    valid = valid & (d <= R_CAP * q25s[:, None])
                    R = np.where(valid, d, np.nan)
        # sections without a valid ray give NaN on purpose (filled below): their 'All-NaN slice' warnings are silenced
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q = np.nanmedian(R, axis=1)  # section radius, the last fallback of the filling
        if np.isfinite(q).any():
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q = np.where(np.isfinite(q), q, np.nanmedian(q))
        self.R = R
        self.valid = valid
        self.Rf = _fill(self.R, valid, q)
        self.rmean = self.Rf.mean(1)
        return self


def _fill(R, valid, q):
    """Fill invalid profile entries: along phi (circular, linear) when the section has enough valid directions, else
    from the nearest valid station at the same angle, else the robust section radius."""
    n, K = R.shape
    out = R.copy()
    for i in range(n):
        v = valid[i]
        if v.sum() >= K // 3 and not v.all():
            idx = np.where(v)[0]
            xi = np.concatenate([idx - K, idx, idx + K])
            yi = np.tile(R[i, idx], 3)
            bad = np.where(~v)[0]
            out[i, bad] = np.interp(bad, xi, yi)
    still = ~np.isfinite(out)
    if still.any():
        for a in range(K):
            col = out[:, a]
            g = np.where(np.isfinite(col))[0]
            b = np.where(~np.isfinite(col))[0]
            if len(g) and len(b):
                near = g[np.argmin(np.abs(b[:, None] - g[None, :]), axis=1)]
                out[b, a] = col[near]
    still = ~np.isfinite(out)
    if still.any():
        fallback = np.where(np.isfinite(q), q, np.nanmedian(q) if np.isfinite(q).any() else 1.0)
        out[still] = np.broadcast_to(fallback[:, None], out.shape)[still]
    return out


def _polyline_param(P):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return s


def _project_arc(P, s, X):
    """Arc-length position of the closest point of X on the polyline P extended linearly at both ends."""
    best, sb = np.inf, 0.0
    for k in range(len(P) - 1):
        seg = P[k + 1] - P[k]
        L = np.linalg.norm(seg)
        if L < 1e-12:
            continue
        w = ((X - P[k]) @ seg) / (L * L)
        lo = -np.inf if k == 0 else 0.0
        hi = np.inf if k == len(P) - 2 else 1.0
        w = min(max(w, lo), hi)
        d = np.linalg.norm(X - (P[k] + w * seg))
        if d < best:
            best, sb = d, s[k] + w * L
    return sb


def _eval_arc(P, s, x):
    """Points on the polyline (extended linearly) at arc-length positions x."""
    x = np.asarray(x, dtype=np.float64)
    k = np.clip(np.searchsorted(s, x, side="right") - 1, 0, len(P) - 2)
    seg = P[k + 1] - P[k]
    L = np.maximum(s[k + 1] - s[k], 1e-12)
    w = (x - s[k]) / L
    return P[k] + w[:, None] * seg


def _local_lines(C, rad, ok, x, xq, u, spacing):
    """Medial curve from centred sections by local linear regression along the chord coordinate.
    C: section centres, rad: their radii, ok: accepted sections, x: chord coordinate of the centres, xq: chord
    coordinates at which the curve is wanted. The kernel width is SMOOTH_R local radii (a medial curve carries no
    information below that scale), at least 0.75 station spacings: a short fat part gets a straight axis, a long thin
    one keeps its bends. Sections that were not accepted are filled by the regression (extrapolation is linear).
    Returns points and unit tangents."""
    idx = np.where(ok)[0]
    r_at = np.interp(xq, x[idx], rad[idx])
    P = np.zeros((len(xq), 3))
    T = np.zeros((len(xq), 3))
    for k in range(len(xq)):
        sig = max(SMOOTH_R * r_at[k], 0.75 * spacing)
        w = np.exp(-0.5 * ((x[idx] - xq[k]) / sig) ** 2) + 1e-12
        w /= w.sum()
        xb = w @ x[idx]
        cb = w @ C[idx]
        sxx = w @ (x[idx] - xb) ** 2
        if sxx > (0.25 * spacing) ** 2:
            b = ((w * (x[idx] - xb)) @ (C[idx] - cb)) / sxx
        else:  # one section within reach: direction of the chord
            b = u
        P[k] = cb + b * (xq[k] - xb)
        T[k] = b / np.linalg.norm(b)
    return P, T


def _consistent_run(ok, ax, latv, rad):
    """Keep the centred sections that form ONE curve. Sections snapped into a neighbouring body (a coxa section that
    ends up in the thorax) are far from the others and wide. Sections are linked when
    their centres are compatible with one curve (RUN_SLOPE, RUN_R); the seed is the linked run with the smallest
    median radius among the runs of at least three sections (else the longest run); other sections join while they
    link to the nearest kept section."""
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return ok

    def linked(i, j):
        return np.linalg.norm(latv[i] - latv[j]) < RUN_SLOPE * abs(ax[i] - ax[j]) + RUN_R * min(rad[i], rad[j])

    runs, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if linked(a, b):
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    if len(runs) == 1:
        return ok
    big = [r for r in runs if len(r) >= 3]
    seed = min(big, key=lambda r: np.median(rad[r])) if big else max(runs, key=len)
    keep = set(seed)
    changed = True
    while changed:
        changed = False
        for i in idx:
            if i in keep:
                continue
            kk = np.array(sorted(keep))
            j = kk[np.argmin(np.abs(ax[kk] - ax[i]))]
            if linked(i, j):
                keep.add(i)
                changed = True
    out = np.zeros_like(ok)
    out[list(keep)] = True
    return out


def trace_tube(surf, A, B, r0, ref_normal, end=False, passes=3):
    """Trace the tube between joints A and B on `surf`. r0: radius guess (scaled template radius).
    Returns a Tube (curve, frame, profile). If the curve cannot be centred, tube.traced is False and the curve is the
    chord (the caller replaces it by the bone-wise posed template curve)."""
    tb = Tube()
    L0 = np.linalg.norm(B - A)
    u = (B - A) / L0
    tau = np.linspace(0.08, 0.92, N_CENT)
    O = A[None] + tau[:, None] * L0 * u[None]
    T = np.tile(u, (N_CENT, 1))
    r0s = np.broadcast_to(np.asarray(r0, dtype=np.float64), (N_CENT,)).copy()  # scalar or one guess per station
    ok = np.zeros(N_CENT, bool)
    rad = np.full(N_CENT, np.nan)
    for p in range(passes):
        O2, rad, good, nval = centre_sections(surf, O, T, r0s)
        ax = (O2 - A) @ u
        with np.errstate(invalid="ignore"):
            ok = good & (rad < R_TOL * r0s) & (rad > r0s / R_TOL) & (np.abs(ax - tau * L0) < 0.2 * L0)
        lat = np.linalg.norm((O2 - A) - ax[:, None] * u, axis=1)
        ok &= lat < np.maximum(4 * np.nan_to_num(rad, nan=0.0), 0.35 * L0)
        ok = _consistent_run(ok, ax, (O2 - A) - ax[:, None] * u, rad)
        if ok.sum() < 3:
            break
        np.where(ok)[0]
        O, T = _local_lines(O2, rad, ok, (O2 - A) @ u, tau * L0, u, L0 / (N_CENT - 1))
    tb.ok_share = float(ok.mean())
    tb.rad_ok = float(np.median(rad[ok])) if ok.any() else float("nan")  # radius seen by the centred sections
    if ok.sum() < 3:
        tb.traced = False
        P = A[None] + np.linspace(0, 1, N_CENT)[:, None] * (B - A)[None]
        s = _polyline_param(P)
        sA, sB = 0.0, s[-1]
        tb.note = "not centred"
    else:
        tb.traced = True
        P = O
        s = _polyline_param(P)
        sA = _project_arc(P, s, A)
        sB = _project_arc(P, s, B)
        if end:
            # geometric tip: where the curve leaves the solid beyond the last centred station
            k1 = np.where(ok)[0][-1]
            d, f, c = surf.first_hit(P[k1][None], T[k1][None])
            rl = np.nanmedian(rad[ok][-3:])
            if (
                np.isfinite(d[0])
                and c[0] > 0
                and (s[k1] + d[0]) < sB + max(4 * rl, 0.4 * L0)
                and (s[k1] + d[0]) > sB - TIP_BACK * L0
            ):
                sB = s[k1] + d[0]
                tb.tip_found = True
            else:
                tb.tip_found = False
                tb.note = "tip not found"
        if sB - sA < 0.3 * L0:
            sA, sB = _project_arc(P, s, A), _project_arc(P, s, A) + L0
            tb.note += " pins degenerate"
    tb.length = float(sB - sA)
    C = _eval_arc(P, s, sA + Tube.t * (sB - sA))
    tb.set_curve(C, ref_normal)
    r_exp = None
    if tb.traced and np.isfinite(rad[ok]).all():
        tcen = (s[ok] - sA) / max(sB - sA, 1e-12)  # the centred sections along t
        o = np.argsort(tcen)
        r_exp = np.interp(np.linspace(0, 1, N_IN), tcen[o], rad[ok][o])
    tb.measure(surf, r_exp=r_exp)
    return tb


def twist_from_profiles(tT, tS, sigma_rel=0.05, sigma_prior=np.deg2rad(20.0), n_eff=20.0, t_lo=0.15, t_hi=0.85):
    """Angle psi (rad) by which the scan profile is rotated against the template profile, from the correlation of the
    normalised polar profiles, combined with a zero-mean prior (the frame of the scan tube was anchored with the
    joint-based bone rotation). Returns psi, evidence (cost contrast of the profile term)."""
    tt = np.linspace(0, 1, N_IN)
    ks = np.where((tt >= t_lo) & (tt <= t_hi))[0]
    qT = tT.Rf[ks] / tT.Rf[ks].mean(1, keepdims=True) - 1
    vS = tS.valid[ks]
    RS = tS.Rf[ks]
    qS = RS / RS.mean(1, keepdims=True) - 1
    row_ok = vS.mean(1) > 0.6
    if row_ok.sum() < 3:
        return 0.0, 0.0
    qT, qS, vS = qT[row_ok], qS[row_ok], vS[row_ok]
    shifts = np.arange(-K_ANG // 2, K_ANG // 2)
    psi = shifts * 2 * np.pi / K_ANG
    msd = np.zeros(len(shifts))
    for i, k in enumerate(shifts):
        a = np.roll(qS, -k, axis=1)  # scan value at phi + psi
        v = np.roll(vS, -k, axis=1)
        msd[i] = ((a - qT) ** 2 * v).sum() / max(v.sum(), 1)
    cost = n_eff * msd / sigma_rel**2 + (psi / sigma_prior) ** 2
    i = int(np.argmin(cost))
    best = psi[i]
    if 0 < i < len(cost) - 1:
        c0, c1, c2 = cost[i - 1], cost[i], cost[i + 1]
        den = c0 - 2 * c1 + c2
        if den > 1e-12:
            best += 0.5 * (c0 - c2) / den * (2 * np.pi / K_ANG)
    evid = float(n_eff * (msd.max() - msd.min()) / sigma_rel**2)
    return float(best), evid
