"""Closed-form articulated start in fitter_3d's NATIVE parameters (geometric registration core).

From a joint product (joint targets P, weights w) to the parameters of fitter_3d's own forward pass
    verts = exp(log_scale) * LBS(betas, [global_rot, joint_rot * rotation_mask], log_beta_scales, betas_trans) + trans
            + deform_verts
without any optimisation: no GPU, no iterations, deterministic. It replaces Stage 0 / Stage 1 of the joint-informed
schedule (or initialises them). `forward_np` below is a float64 re-implementation of that forward pass
(smal_model/smal_torch.py SMAL.__call__ + batch_lbs.batch_global_rigid_transformation, read 2026-11-13);
A check showed that the fitter's own forward of the written parameters reproduces the closed-form posed template.

The kinematics the model offers (facts read from smal_model/batch_lbs.py):
    bone j maps x -> M_j (x - J_j) + X_j,  M_j = G_j S_j,  G_j = R_0 R_.. R_j (pure rotation), S_j = diag(exp(lbs_j))
    X_j = X_p + M_p (J_j - J_p + betas_trans_j * (1, -1, 1)),   X_0 = J_0,   root scale must stay 0 (its entry is
    applied inverted to every child: S_parent^-1 R S_child)
so every bone can have ANY rotation, any scale along the model axes and any pivot position, independently of its
parent. An axial stretch along a bone that is not parallel to a model axis is NOT representable; the start therefore
uses ONE isotropic scale per bone (the bone's length ratio), and what the surface stage adds to it (limb girth, bends,
surface detail) goes to deform_verts or, after `refit`, partly into three scales per bone.

Per bone the start is  T_j(x) = s * sig_j * G_j (x - ref0_j) + a_j   (normalised target frame)
    ref0_j  rest reference point of joint j: its virtual marker (joint_kind 'medial') or its pivot ('pivot')
    a_j     where that point goes: the joint target when it is usable, else the continuation of the parent's map
    G_j     root / head: (robust) similarity of their landmark joints; chain bone: smallest rotation from the parent's
            frame onto the direction to the child's target; one-joint part with a tip target (mandible + its limb
            centreline): the same with the tip; leaf: the parent's
    sig_j   length ratio (clipped), isotropic
Bones WITHOUT usable joints are never extrapolated: they keep the REST articulation relative to their parent (joint
rotation 0) or take the MIRRORED local rotation and the scale of their bilateral partner when that one is
joint-supported, at the specimen's size, attached at their joint. A bone whose own joint is unusable but whose child
joint is usable BRIDGES from the continuation point to the child's target. Such bones are reported as unsupported.
The root never raises for lack of joints (the fitter's `_similarity` needs 3 root-rigid joints and raised on 9 of 30
thin-limb synthetic specimens): with fewer it falls back to the trunk chord + the frame prior, then to the
frame prior + body length; only a specimen without any usable trunk or coxa joint fails, as a status.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from . import geom

ROOT_RIGID = ("b_h", "b_a_1", "l_1_co_r", "l_1_co_l", "l_2_co_r", "l_2_co_l", "l_3_co_r", "l_3_co_l")
PARAMS = ("global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts", "betas_trans", "log_scale")
YFLIP = np.array([1.0, -1.0, 1.0])  # batch_lbs: betas_trans * (1, -1, 1)
SMIRROR = np.diag([1.0, -1.0, 1.0])  # the template's symmetry plane is y = 0 (read from the model file)

OPTS = dict(
    scale_clip=(0.25, 4.0),
    head_clip=(0.4, 2.5),
    irls_sigma=0.03,
    irls_min_points=5,
    mirror=True,
    collinear_tol=0.08,
    twist="limits",
)

# twist candidates about a chain bone (its two joints do not fix the spin about its own axis), smallest first
PSI = np.deg2rad(np.arange(-180, 180, 5.0))
PSI = PSI[np.argsort(np.abs(PSI), kind="stable")]


def limit_violation(aa, lo, hi):
    return np.maximum(aa - hi, 0).sum(-1) + np.maximum(lo - aa, 0).sum(-1)


def twist_by_limits(Gp, Rm, u, lim_j, look=None):
    """Spin of a chain bone about its own axis (logic of SKEL-REG's bones.BoneFit._twist_cost_limits, re-implemented):
    among G = Rot(u, psi) Rm, the candidate whose local rotation Gp^T G violates the bone's joint limits least, plus
    (look-ahead) the smallest violation its chain child can reach when that child is aimed at ITS target: the bend
    plane at the next joint (a knee is a hinge) fixes the spin of this bone. Ties -> smallest |psi|.
    lim_j (3,2); look = (rest direction of the child bone b_c (3,), unit target direction of the child uc (3,),
    limits of the child (3,2)) or None. Returns G (3,3), psi."""
    u = u / (np.linalg.norm(u) + 1e-30)
    Rs = Rotation.from_rotvec(u[None] * PSI[:, None]).as_matrix() @ Rm  # (K,3,3)
    aa = Rotation.from_matrix(np.einsum("ab,kbc->kac", Gp.T, Rs)).as_rotvec()
    cost = limit_violation(aa, lim_j[:, 0][None], lim_j[:, 1][None])
    if look is not None:
        bc, uc, lim_c = look
        uc = uc / (np.linalg.norm(uc) + 1e-30)
        sub = PSI[::3]
        tw = Rotation.from_rotvec(uc[None] * sub[:, None]).as_matrix()  # (S,3,3)
        la = np.zeros(len(PSI))
        for k in range(len(PSI)):
            Rc = geom.rot_between(Rs[k] @ bc, uc) @ Rs[k]
            aac = Rotation.from_matrix(np.einsum("ab,sbc->sac", Rs[k].T, tw @ Rc)).as_rotvec()
            la[k] = limit_violation(aac, lim_c[:, 0][None], lim_c[:, 1][None]).min()
        cost = cost + la
    k = int(np.argmin(cost + 1e-6 * np.abs(PSI)))
    return Rs[k], float(PSI[k])


class StartError(RuntimeError):
    """the specimen has no usable trunk / coxa joint at all: reported as status 'failed' by the pipeline"""


# ------------------------------------------------------------------------------------------------ small helpers
def _rotvec(R):
    return Rotation.from_matrix(R).as_rotvec()


def _rot(rv):
    return Rotation.from_rotvec(np.asarray(rv, float)).as_matrix()


def similarity(X, Y, w=None, with_scale=True):
    """weighted Umeyama: Y ~ s R X + t (rows = points). Returns s, R, t."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    mx, my = w @ X, w @ Y
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd((Yc * w[:, None]).T @ Xc)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / max(float((w * (Xc**2).sum(1)).sum()), 1e-30)) if with_scale else 1.0
    return s, R, my - s * R @ mx


def robust_similarity(X, Y, w, sigma, min_points=5, iters=6):
    """similarity with Geman-McClure reweighting (width sigma, same units as Y) when there are enough points for an
    outlier to be told from the rest; plain weighted Umeyama otherwise. Returns s, R, t, residuals."""
    w = np.asarray(w, float)
    ww = w.copy()
    s, R, t = similarity(X, Y, ww)
    if len(X) >= min_points and sigma > 0:
        for _ in range(iters):
            r = np.linalg.norm(Y - (s * X @ R.T + t), axis=1)
            ww = w * (sigma**2 / (r**2 + sigma**2)) ** 2
            s, R, t = similarity(X, Y, ww)
    return s, R, t, np.linalg.norm(Y - (s * X @ R.T + t), axis=1)


def _spread(X, w):
    """second / first singular value of the weighted centred points (0 = collinear)"""
    w = np.asarray(w, float) / np.sum(w)
    Xc = (X - w @ X) * np.sqrt(w)[:, None]
    sv = np.linalg.svd(Xc, compute_uv=False)
    return float(sv[1] / max(sv[0], 1e-30)) if len(sv) > 1 else 0.0


def chord_body_length(tpl, mk=None, head_front=None):
    """body length by the joint product's definition (sum of the trunk chords gaster tip -> b_a_3 -> b_a_2 -> b_a_1 ->
    neck -> head front) of a marker set (default: the rest template)."""
    mk = tpl.mk0 if mk is None else mk
    hf = tpl.head_front_row() @ tpl.V0 if head_front is None else head_front
    ix = tpl.idx
    seq = [mk[ix["b_a_5"]], mk[ix["b_a_3"]], mk[ix["b_a_2"]], mk[ix["b_a_1"]], mk[ix["b_h"]], hf]
    return float(sum(np.linalg.norm(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)))


def seg_class(name):
    """segment class of a bone name for scale imputation (same class = comparable proportions)"""
    if name.startswith("l_"):
        return "leg_" + name.split("_")[2]
    if name.startswith("an_"):
        return name[:4]
    if name.startswith("ma_"):
        return "ma"
    return name


# ------------------------------------------------------------------------------------------------ the start
class Start:
    """Result of closed_form_start: per-bone similarity maps in the normalised target frame."""

    def __init__(self, tpl, kind):
        nJ = tpl.nJ
        self.tpl, self.kind = tpl, kind
        self.s = 1.0
        self.G = np.tile(np.eye(3), (nJ, 1, 1))
        self.sig = np.ones(nJ)
        self.a = np.zeros((nJ, 3))
        self.ref0 = (tpl.mk0 if kind == "medial" else tpl.J0).copy()
        self.ref0[0] = tpl.J0[0]
        self.how = [""] * nJ  # root*, similarity, aim, chain, bridge, tip, leaf, rest, mirror, wing
        self.support = np.zeros(nJ, bool)  # placed by its own usable joints
        self.pinned = np.zeros(nJ, bool)  # reference point on its own usable joint target
        self.clipped = np.zeros(nJ, bool)
        self.psi = np.zeros(nJ)  # spin chosen about the bone axis (rad), chain bones
        self.info = {}

    def bone_map(self, j):
        A = self.s * self.sig[j] * self.G[j]
        return A, self.a[j] - A @ self.ref0[j]

    def apply(self, j, X):
        A, b = self.bone_map(j)
        return np.asarray(X, float) @ A.T + b

    def maps(self):
        A = self.s * self.sig[:, None, None] * self.G
        b = self.a - np.einsum("jab,jb->ja", A, self.ref0)
        return A, b

    def vertices(self, V0=None):
        """linear blend of the bone maps = the closed-form posed template (normalised target frame)"""
        tpl = self.tpl
        V0 = tpl.V0 if V0 is None else V0
        A, b = self.maps()
        return np.einsum("vj,jab,vb->va", tpl.W, A, V0) + tpl.W @ b


def _root(tpl, st, P, ok, w, R_prior, BL, o):
    ix = tpl.idx
    ref0 = st.ref0
    ids = [ix[n] for n in ROOT_RIGID if ok[ix[n]]]
    i_n, i_a = ix["b_h"], ix["b_a_1"]
    Rp = np.eye(3) if R_prior is None else np.asarray(R_prior, float)
    if len(ids) >= 3 and _spread(ref0[ids], w[ids]) > o["collinear_tol"]:
        sig = o["irls_sigma"] * (BL if BL else 1.0)
        s, R, t, res = robust_similarity(ref0[ids], P[ids], w[ids], sig, o["irls_min_points"])
        how, resid = "root_similarity", {tpl.names[i]: float(r) for i, r in zip(ids, res)}
    elif ok[i_n] and ok[i_a]:
        u0, u = ref0[i_a] - ref0[i_n], P[i_a] - P[i_n]
        s = float(np.linalg.norm(u) / np.linalg.norm(u0))
        R = geom.rot_between(Rp @ u0, u) @ Rp
        t = P[i_n] - s * R @ ref0[i_n]
        how, resid = "root_chord+frame_prior", {}
    elif ids and BL:
        j = i_n if ok[i_n] else (i_a if ok[i_a] else ids[0])
        s = float(BL / chord_body_length(tpl))
        R = Rp
        t = P[j] - s * R @ ref0[j]
        how, resid = "root_frame_prior+body_length", {}
    else:
        raise StartError("no usable trunk or coxa joint: the root cannot be placed")
    st.s = s
    st.G[0], st.sig[0] = R, 1.0
    st.a[0] = s * R @ ref0[0] + t
    st.how[0] = how
    st.support[0] = bool(ok[i_n] and ok[i_a])
    st.info["root"] = dict(how=how, n_joints=len(ids), residual=resid, scale=s)


def closed_form_start(tpl, P, w, kind="medial", R_prior=None, BL=None, head_front=None, tips=None, opts=None):
    """tpl: geomreg.model.Template; P (nJ,3) joint targets in the normalised target frame (nan = none); w (nJ,) weights
    (0 = unusable); kind 'medial' | 'pivot'; R_prior: rotation model axes -> target frame given by the joint product's
    frame (identity when the target frame is the product's canonical frame); BL: body length in target units (the
    joint product's chord definition); head_front (3,) target or None; tips {bone index: (3,) tip target} for one-joint
    parts with a tip (mandibles). Returns a Start. Raises StartError when the root cannot be placed."""
    o = dict(OPTS)
    o.update(opts or {})
    nJ = tpl.nJ
    par = tpl.parents
    assert all(par[j] < j for j in range(1, nJ)), "bones must be ordered parents first"
    P = np.asarray(P, float)
    w = np.asarray(w, float)
    ok = (w > 0) & np.isfinite(P).all(1)
    wing = np.array([n.startswith("w_") for n in tpl.names])
    ok &= ~wing
    tips = tips or {}
    tip_rows = tpl.tip_rows()
    tip0 = {j: tip_rows[j] @ tpl.V0 for j in tip_rows}
    hf0 = tpl.head_front_row() @ tpl.V0
    i_h = tpl.idx["b_h"]
    st = Start(tpl, kind)
    ref0 = st.ref0
    _root(tpl, st, P, ok, w, R_prior, BL, o)
    s = st.s
    kids_of = [[c for c in tpl.children[j] if not wing[c]] for j in range(nJ)]
    cls = [seg_class(n) for n in tpl.names]
    data_Rloc, data_sig = {}, {}

    def solve(impute):
        for j in range(1, nJ):
            p = par[j]
            Gp = st.G[p]
            kids = kids_of[j]
            st.clipped[j] = False
            st.pinned[j] = bool(ok[j])
            aj = P[j] if ok[j] else st.apply(p, ref0[j])
            st.a[j] = aj
            if wing[j]:
                st.G[j], st.sig[j], st.how[j], st.support[j] = Gp, st.sig[p], "wing", False
                continue
            if j == i_h:  # head: landmark similarity
                ids = [j] + [c for c in kids if ok[c]]
                X = [ref0[i] for i in ids]
                Y = [P[i] if ok[i] else None for i in ids]
                ww = [w[i] for i in ids]
                if not ok[j]:
                    X, Y, ww = X[1:], Y[1:], ww[1:]
                if head_front is not None:
                    X.append(hf0), Y.append(np.asarray(head_front, float)), ww.append(max(w[j], 0.3))
                X, Y, ww = np.array(X).reshape(-1, 3), np.array(Y, float).reshape(-1, 3), np.array(ww, float)
                if ok[j] and len(X) >= 3 and _spread(X, ww) > o["collinear_tol"]:
                    sh, Rh, th, res = robust_similarity(
                        X, Y, ww, o["irls_sigma"] * (BL if BL else 1.0), o["irls_min_points"]
                    )
                    sg = sh / s
                    st.clipped[j] = not (o["head_clip"][0] <= sg <= o["head_clip"][1])
                    sg = float(np.clip(sg, *o["head_clip"]))
                    st.G[j], st.sig[j] = Rh, sg
                    st.a[j] = (s * sg * Rh @ ref0[j] + th) if not st.clipped[j] else aj
                    st.how[j], st.support[j] = "similarity", True
                    st.info["head"] = dict(how="similarity", n=len(X), scale=sg, residual=[float(r) for r in res])
                elif ok[j] and len(X) >= 2:
                    k = int(np.argmax(ww[1:])) + 1
                    u0, u = X[k] - X[0], Y[k] - aj
                    sg = float(np.linalg.norm(u) / (s * np.linalg.norm(u0)))
                    st.clipped[j] = not (0.5 <= sg <= 2.0)
                    st.G[j], st.sig[j] = geom.rot_between(Gp @ u0, u) @ Gp, float(np.clip(sg, 0.5, 2.0))
                    st.how[j], st.support[j] = "aim", True
                    st.info["head"] = dict(how="aim", n=len(X), scale=st.sig[j])
                else:
                    st.G[j], st.sig[j], st.how[j], st.support[j] = Gp, 1.0, "rest", False
                    st.info["head"] = dict(how="rest")
                data_Rloc[j], data_sig[j] = Gp.T @ st.G[j], st.sig[j]
                continue
            # the point this bone is aimed at: its chain child's target, or the tip target of a one-joint part
            c = kids[0] if len(kids) == 1 else -1
            tgt, r1, how = None, None, ""
            if c >= 0 and ok[c]:
                tgt, r1, how = P[c], ref0[c], ("chain" if ok[j] else "bridge")
            elif not kids and j in tips and tips[j] is not None and ok[j]:
                tgt, r1, how = np.asarray(tips[j], float), tip0[j], "tip"
            if tgt is not None:
                u0, u = r1 - ref0[j], tgt - aj
                lu0, lu = np.linalg.norm(u0), np.linalg.norm(u)
                if lu0 > 1e-12 and lu > 1e-12:
                    lam = lu / (s * lu0)
                    st.clipped[j] = not (o["scale_clip"][0] <= lam <= o["scale_clip"][1])
                    Rm = geom.rot_between(Gp @ u0, u) @ Gp
                    # (not for a one-joint part aimed at a tip: on the development scans the limit-aware spin put one
                    # mandible on its neighbour that the smallest rotation from the head's frame had right)
                    if o["twist"] == "limits" and tpl.limits is not None and how == "chain":
                        look = None
                        g = kids_of[c][0] if (c >= 0 and len(kids_of[c]) == 1) else -1
                        if g >= 0 and ok[g] and ok[c]:
                            look = (
                                (ref0[g] - ref0[c]) / max(np.linalg.norm(ref0[g] - ref0[c]), 1e-30),
                                P[g] - P[c],
                                tpl.limits[c],
                            )
                        Rm, st.psi[j] = twist_by_limits(Gp, Rm, u, tpl.limits[j], look)
                    st.G[j] = Rm
                    st.sig[j] = float(np.clip(lam, *o["scale_clip"]))
                    st.how[j] = how
                    st.support[j] = how in ("chain", "tip") and not st.clipped[j]
                    if st.support[j]:
                        data_Rloc[j], data_sig[j] = Gp.T @ st.G[j], st.sig[j]
                    continue
            if not kids and j not in tip0:  # leaf tip bone: the parent's map
                st.G[j], st.sig[j], st.how[j] = Gp, st.sig[p], "leaf"
                st.support[j] = bool(ok[j] and st.support[p])
                continue
            # no usable joints for this bone: rest articulation, or the mirror of a joint-supported partner
            m = tpl.mirror_bone[j]
            if impute and o["mirror"] and m != j and m in data_Rloc:
                st.G[j] = Gp @ (SMIRROR @ data_Rloc[m] @ SMIRROR)
                st.sig[j], st.how[j] = data_sig[m], "mirror"
            else:
                same = [data_sig[q] for q in data_sig if cls[q] == cls[j]] if impute else []
                st.G[j] = Gp
                st.sig[j] = float(np.median(same)) if same else float(st.sig[p])
                st.how[j] = "rest"
            # a one-joint part (mandible) without a tip target that is pinned at its own usable joint on a supported
            # parent takes part in the surface stage (as in the earlier surface registration; it stays 'review': one joint only)
            st.support[j] = bool(not kids and j in tip0 and ok[j] and st.support[p])
            if st.support[j]:
                st.how[j] = "pinned_" + st.how[j]

    solve(impute=False)  # pass 1: which bones are placed by data, and their local rotations / scales
    solve(impute=True)  # pass 2: imputation of the others (mirror partner, class median), children re-derived
    solve(impute=True)  # pass 3: partners that were themselves re-derived in pass 2 (bridges below an imputed bone)
    st.info["counts"] = {k: int(sum(1 for h in st.how if h == k)) for k in sorted(set(st.how))}
    st.info["clipped"] = [tpl.names[j] for j in range(nJ) if st.clipped[j]]
    return st


# ------------------------------------------------------------------------------------------------ native parameters
def to_native(st, betas=None):
    """Start -> fitter_3d parameters (float64 arrays without the batch dimension) such that the fitter's forward with
    deform_verts = 0 gives st.vertices(). betas must be 0 / None for the identity to hold exactly (the start is built
    on the mean template); with betas the pivots J(beta) are used and the identity holds for the shaped template."""
    tpl = st.tpl
    nJ, par = tpl.nJ, tpl.parents
    b = np.zeros(tpl.nB) if betas is None else np.asarray(betas, float)
    V = tpl.V0 + tpl.shapedirs @ b
    J = tpl.Jreg @ V
    s = st.s
    trans = st.a[0] - s * st.G[0] @ (st.ref0[0] - J[0]) - s * J[0]
    X = np.zeros((nJ, 3))
    X[0] = J[0]
    theta = np.zeros((nJ, 3))
    theta[0] = _rotvec(st.G[0])
    lbs = np.zeros((nJ, 3))
    bt = np.zeros((nJ, 3))
    for j in range(1, nJ):
        p = par[j]
        theta[j] = _rotvec(st.G[p].T @ st.G[j])
        lbs[j] = np.log(st.sig[j])
        X[j] = (st.a[j] - trans) / s - st.sig[j] * st.G[j] @ (st.ref0[j] - J[j])
        off = st.G[p].T @ (X[j] - X[p]) / st.sig[p] - (J[j] - J[p])
        bt[j] = off * YFLIP
    return dict(
        global_rot=theta[0],
        joint_rot=theta[1:],
        betas=b,
        log_beta_scales=lbs,
        trans=trans,
        betas_trans=bt,
        log_scale=np.array([np.log(s)]),
        rotation_mask=np.ones((nJ - 1, 3)),
    )


def forward_np(tpl, p, deform=True, return_maps=False, weights=None):
    """float64 re-implementation of fitter_3d's forward for ONE specimen (parameter arrays without batch dimension).
    weights: (nV, nJ) skinning weights; default tpl.W, the rows renormalised to sum 1 as in every harness. With
    tpl.W_raw (the rows as the model file stores them) this is SMAL3DFitter.forward itself: the fitter blends with
    the stored rows, and the two differ on the vertices whose row does not sum to 1. Returns verts (nV,3), pivots
    (nJ,3)."""
    nJ, par = tpl.nJ, tpl.parents
    V = tpl.V0 + tpl.shapedirs @ np.asarray(p["betas"], float)
    J = tpl.Jreg @ V
    mask = np.asarray(p.get("rotation_mask", np.ones((nJ - 1, 3))), float)
    theta = np.concatenate([np.asarray(p["global_rot"], float)[None], np.asarray(p["joint_rot"], float) * mask])
    Rs = Rotation.from_rotvec(theta).as_matrix()
    S = np.exp(np.asarray(p["log_beta_scales"], float))
    off = np.asarray(p["betas_trans"], float) * YFLIP
    M = np.zeros((nJ, 3, 3))
    X = np.zeros((nJ, 3))
    M[0], X[0] = Rs[0], J[0]
    for i in range(1, nJ):
        q = par[i]
        M[i] = M[q] @ np.diag(1.0 / S[q]) @ Rs[i] @ np.diag(S[i])
        X[i] = X[q] + M[q] @ (J[i] - J[q] + off[i])
    b = X - np.einsum("jab,jb->ja", M, J)
    s = float(np.exp(np.asarray(p["log_scale"], float).reshape(-1)[0]))
    tr = np.asarray(p["trans"], float)
    W = tpl.W if weights is None else weights
    verts = s * (np.einsum("vj,jab,vb->va", W, M, V) + W @ b) + tr
    if deform and p.get("deform_verts") is not None:
        verts = verts + np.asarray(p["deform_verts"], float)
    if return_maps:
        return verts, s * X + tr, (s * M, s * b + tr)
    return verts, s * X + tr


def linear_maps(tpl, p):
    """per-bone affine maps (A (nJ,3,3), b (nJ,3)) of the native parameters p in the target frame: bone j maps the
    SHAPED rest template x -> A_j x + b_j (before deform_verts)"""
    return forward_np(tpl, p, deform=False, return_maps=True)[2]
