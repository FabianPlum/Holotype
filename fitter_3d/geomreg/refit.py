"""Native parameters of a registered mesh (replaces an earlier Adam refit on the GPU).

After the surface stage every template vertex has a registered position, so fitter_3d's parameters can be fitted to it
with DENSE correspondences and without an optimiser: block coordinate descent on the exact blended least-squares
objective  sum_v m_v | sum_j W_vj (A_j x_v + b_j) - y_v |^2,  one bone at a time, each block in closed form:

    bone j:  A_j = s G_j diag(S_j)   G_j rotation, S_j three scales along the MODEL axes (what log_beta_scales offers;
             smal_model/batch_lbs.py), found by alternating Kabsch / per-axis least squares on the bone's weighted
             residual targets;  b_j free (the pivot offset betas_trans makes every pivot position reachable)
    root:    rotation only (its log_beta_scales entry must stay 0: it enters every child inverted); the global scale s
             is kept from the start
    betas:   optional (use_betas): with the bone maps fixed the mesh is linear in the shape coefficients; ridge least
             squares with the model's shape covariance as prior

deform_verts is then what the parameters cannot express. The pipeline reports how much of the registered shape each
part carries (info['native']).
"""

import numpy as np

from . import native


def _fit_rot_scale(X, Y, w, G0, S0, free_scale=3, iters=12, s_clip=(0.2, 5.0)):
    """min sum w |G diag(S) x + b - y|^2 over rotation G, scales S (free_scale: 3 per axis, 1 isotropic, 0 fixed), b.
    X, Y (n,3), w (n,). Returns G, S, b."""
    sw = w.sum()
    if sw <= 0:
        return G0, S0, np.zeros(3)
    mx, my = (w[:, None] * X).sum(0) / sw, (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    G, S = G0.copy(), S0.copy()
    for _ in range(iters):
        # rotation given scales (Kabsch)
        H = (Yc * w[:, None]).T @ (Xc * S)
        U, _, Vt = np.linalg.svd(H)
        D = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            D[2, 2] = -1.0
        G = U @ D @ Vt
        if free_scale == 0:
            break
        Yl = Yc @ G  # targets in the bone's rest axes
        num = (w[:, None] * Yl * Xc).sum(0)
        den = (w[:, None] * Xc * Xc).sum(0)
        if free_scale == 3:
            Sn = np.where(den > 1e-18, num / np.maximum(den, 1e-18), S)
        else:
            Sn = np.full(3, num.sum() / max(den.sum(), 1e-18))
        Sn = np.clip(Sn, *s_clip)
        if np.abs(Sn - S).max() < 1e-7:
            S = Sn
            break
        S = Sn
    return G, S, my - G @ (S * mx)


def maps_to_native(tpl, s, G, S, b, betas=None):
    """Per-bone maps x -> s G_j diag(S_j) x + b_j (target frame, applied to the shaped rest template) -> fitter_3d
    parameters. S[0] must be 1."""
    nJ, par = tpl.nJ, tpl.parents
    be = np.zeros(tpl.nB) if betas is None else np.asarray(betas, float)
    J = tpl.Jreg @ (tpl.V0 + tpl.shapedirs @ be)
    M = np.einsum("jab,jb->jab", G, S)  # G diag(S)
    trans = b[0] + s * (M[0] @ J[0]) - s * J[0]  # root: s R (x - J0) + s J0 + trans
    X = np.zeros((nJ, 3))
    X[0] = J[0]
    theta = np.zeros((nJ, 3))
    theta[0] = native._rotvec(G[0])
    bt = np.zeros((nJ, 3))
    for j in range(1, nJ):
        p = par[j]
        theta[j] = native._rotvec(G[p].T @ G[j])
        X[j] = (b[j] - trans) / s + M[j] @ J[j]
        off = (G[p].T @ (X[j] - X[p])) / S[p] - (J[j] - J[p])
        bt[j] = off * native.YFLIP
    lbs = np.log(S)
    lbs[0] = 0.0
    return dict(
        global_rot=theta[0],
        joint_rot=theta[1:],
        betas=be,
        log_beta_scales=lbs,
        trans=trans,
        betas_trans=bt,
        log_scale=np.array([np.log(s)]),
        rotation_mask=np.ones((nJ - 1, 3)),
    )


def native_refit(
    tpl, V, sup_v, p0, BL=1.0, sweeps=6, min_full=20, min_any=4, use_betas=False, beta_prior=1.0, aniso=True
):
    """V (nV,3) registered vertices (normalised target frame); sup_v (nV,) bool vertices of joint-supported bones (the
    others are fitted rigidly: their shape is the template's by construction); p0: start parameters (native.to_native).
    Returns (parameters dict without deform_verts, info)."""
    nJ = tpl.nJ
    W = tpl.W
    s = float(np.exp(np.asarray(p0["log_scale"]).reshape(-1)[0]))
    A0, b0 = native.linear_maps(tpl, p0)  # target-frame maps of the start (include s)
    G = np.zeros((nJ, 3, 3))
    S = np.ones((nJ, 3))
    for j in range(nJ):
        U, sv, Vt = np.linalg.svd(A0[j] / s)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        G[j] = R
        S[j] = np.diag(R.T @ (A0[j] / s))  # start: isotropic, so R^T A / s = sig I
    b = b0.copy()
    betas = np.asarray(p0["betas"], float).copy()
    cnt = np.bincount(tpl.dom, minlength=nJ)
    sup_b = np.zeros(nJ, bool)
    for j in range(nJ):
        m = tpl.dom == j
        sup_b[j] = bool(m.any() and sup_v[m].mean() > 0.5)
    nz = [np.where(W[:, j] > 1e-4)[0] for j in range(nJ)]
    X0 = tpl.V0 + tpl.shapedirs @ betas

    def blend(Xr):
        A = s * np.einsum("jab,jb->jab", G, S)
        return np.einsum("vj,jab,vb->va", W, A, Xr) + W @ b, A

    Vl, A = blend(X0)
    e0 = float(np.sqrt(((Vl - V) ** 2).sum(1)).mean() / BL)
    hist = [e0]
    for sweep in range(sweeps):
        for j in range(nJ):
            ids = nz[j]
            if cnt[j] < min_any or len(ids) < min_any:
                continue
            wj = W[ids, j]
            Aj = s * G[j] * S[j][None, :]
            own = X0[ids] @ Aj.T + b[j]
            r = V[ids] - (Vl[ids] - wj[:, None] * own)  # what bone j has to deliver, times w_vj
            Y = r / wj[:, None]
            ww = wj**2
            if j == 0:
                fs = 0
            elif not sup_b[j]:
                fs = 0  # imputed part: rigid, the start's scale
            elif cnt[j] >= min_full and aniso:
                fs = 3
            else:
                fs = 1
            Gj, Sj, bj = _fit_rot_scale(s * X0[ids], Y, ww, G[j], S[j], free_scale=fs)
            G[j], S[j], b[j] = Gj, Sj, bj
            new = X0[ids] @ (s * Gj * Sj[None, :]).T + bj
            Vl[ids] += wj[:, None] * (new - own)
        if use_betas:
            A = s * np.einsum("jab,jb->jab", G, S)
            WA = np.einsum("vj,jab->vab", W, A)  # (nV,3,3)
            Dm = np.einsum("vab,vbk->vak", WA, tpl.shapedirs)  # (nV,3,nB)
            # limb tubes are left out of the shape fit: their girth comes from the scan's centreline radius (tube
            # warp), several times the template's, and must not be explained by shape coefficients
            m = sup_v.astype(float) * (1.0 - W[:, tpl.tube_bone].sum(1))
            r = (V - (np.einsum("vab,vb->va", WA, tpl.V0) + W @ b)) * m[:, None]
            Dw = Dm * m[:, None, None]
            H = np.einsum("vak,val->kl", Dw, Dw)
            g = np.einsum("vak,va->k", Dw, r)
            # ridge: data noise 0.5 % BL per vertex, vertex residuals strongly correlated (about 1 independent
            # datum per 100 vertices), prior = the model's shape covariance
            prior = (
                beta_prior
                * np.linalg.inv(tpl.shape_cov + 1e-6 * np.eye(tpl.nB))
                * ((0.005 * BL) ** 2 * m.sum() / 100.0)
            )
            betas = np.linalg.solve(H + prior, g)
            X0 = tpl.V0 + tpl.shapedirs @ betas
            Vl, A = blend(X0)
        hist.append(float(np.sqrt(((Vl - V) ** 2).sum(1)).mean() / BL))
    p = maps_to_native(tpl, s, G, S, b, betas)
    chk, _ = native.forward_np(tpl, p, deform=False)
    info = dict(
        mean_dist_start_bl=e0,
        mean_dist_refit_bl=hist[-1],
        hist=hist,
        sweeps=sweeps,
        use_betas=bool(use_betas),
        native_vs_maps_max=float(np.abs(chk - blend(X0)[0]).max()),
        scale_aniso_med=float(np.median(S.max(1) / S.min(1))),
        betas_abs_max=float(np.abs(betas).max()),
    )
    return p, info
