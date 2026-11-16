"""Joint-informed registration helpers for fitter_3d (AntScan): proposals -> normalised targets, completeness masks,
closed-form initialisation (design doc docs/antscan_registration_from_joints.md, 3.C, 3.D1-D4).

Validated on 40 synthetic specimens:
  D1 weighted Umeyama on the root-rigid joints (b_t, b_a_1, b_h, six coxae) -> rotation, first scale;
  D2 robust Gauss-Newton on (log sigma, betas) over pose-invariant distances (bones + pairs inside the root- and
     head-rigid sets), Geman-McClure IRLS, shape prior; sigma is the global scale;
  D3 forward-kinematic IK: smallest rotation onto each proposed child direction; if that violates joint_limits, the
     smallest twist about the bone axis that satisfies them; then clipping.
D1 alone is NOT enough when Stage 1 has no per-joint log_beta_scales (measured: p90 pivot error 2.8 % body length
vs 0.82 % with D3).

Proposal contract: <name>_joints.json with joints_canonical (micrometres, canonical frame of the processed mesh, or
null), confidence, completeness.missing_joints.

joint_kind (prepare_batch): 'pivot' (default) compares the model's pivots (bone heads) with the proposals; 'medial'
(round-6 proposals and the annotated truth: articulation points on the scan's medial axis) compares VIRTUAL MARKERS, a
fixed linear map of the mesh vertices (marker_regressor: centroid of the vertex ring at each articulation, tips for leaf
joints, gaster apex), as the registration pipeline does (fitter_3d/geomreg/model.py). Measured on the template
the two differ by 1.7-3.2 % body length at neck / coxae / waist and 4.1 % at the mandibles; and because markers are
vertex functions the joint term also constrains the deformation stages (pivots do not move with deform_verts).
"""

import json
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT_RIGID = ["b_t", "b_a_1", "b_h", "l_1_co_r", "l_1_co_l", "l_2_co_r", "l_2_co_l", "l_3_co_r", "l_3_co_l"]
HEAD_RIGID = ["b_h", "ma_r", "ma_l", "an_1_r", "an_1_l"]
TRUNK = ["b_t", "b_h", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]


class Model:
    """Static model geometry (float64) from a SMAL3DFitter: rest pivots J(beta) = J0 + Jdirs @ beta, tree, limits."""

    def __init__(self, fitter, joint_names):
        sm = fitter.smal_model
        self.names = list(joint_names)
        self.nj = len(self.names)
        par = np.asarray(sm.parents).astype(int)
        par[0] = -1
        self.parents = par
        self.children = [[c for c in range(self.nj) if par[c] == j] for j in range(self.nj)]
        jreg = sm.J_regressor.detach().cpu().numpy().astype(np.float64).T  # (J, V)
        v = sm.v_template.detach().cpu().numpy().astype(np.float64)
        nb = sm.shapedirs.shape[0]
        sd3 = sm.shapedirs.detach().cpu().numpy().astype(np.float64).reshape(nb, -1, 3).transpose(1, 2, 0)  # (V, 3, nb)
        self.J0 = jreg @ v
        self.Jdirs = np.einsum("jv,vck->jck", jreg, sd3)
        self.nb = nb
        prec = fitter.betas_prec.detach().cpu().numpy().astype(np.float64)
        self.shape_sd = np.sqrt(np.diag(np.linalg.inv(prec @ prec.T)))
        self.limits = np.zeros((self.nj, 3, 2))
        self.limits[1:, :, 0] = fitter.min_limits.detach().cpu().numpy()
        self.limits[1:, :, 1] = fitter.max_limits.detach().cpu().numpy()
        self.dom = sm.weights.argmax(1).cpu().numpy()
        self.verts = v
        self.body_length_template = body_length(v, self.J0, self.dom, self.names)

    def rest_pivots(self, beta):
        return self.J0 + self.Jdirs @ np.asarray(beta, dtype=np.float64)


def load_proposal(path, joint_names, norm, conf_min=0.0, kind="pivot", gaster_fraction=None):
    """-> (P (J,3) in the normalised target frame, nan for null; w (J,) weights; missing joint names).
    norm = {"centre", "scale"} from load_meshes(return_norm=True) on the canonical-frame mesh.
    conf_min: proposals below this confidence get weight 0 (0 = the previous behaviour, weight = confidence).
    kind 'medial': b_t is dropped (round 6 sets b_t := b_h := neck; the root pivot is not a medial point) and, when the
    file has a gaster_tip, b_a_5 := gaster_tip and b_a_4 on b_a_3 -> tip at the template's gaster_fraction (the
    gaster as one rigid unit, SKEL-REG's convention)."""
    d = json.load(open(path))
    js, cf = d.get("joints_canonical") or {}, d.get("confidence") or {}
    missing = [m for m in (d.get("completeness") or {}).get("missing_joints", []) if m in joint_names]
    P = np.full((len(joint_names), 3), np.nan)
    w = np.zeros(len(joint_names))
    for j, n in enumerate(joint_names):
        x = js.get(n)
        if x is None or n.startswith("w_") or n in missing or (kind == "medial" and n == "b_t"):
            continue
        c = cf.get(n, 1.0)
        c = float(np.clip(1.0 if c is None else c, 0.0, 1.0))
        if c < conf_min:
            continue
        P[j] = (np.asarray(x, dtype=np.float64) - norm["centre"]) / norm["scale"]
        w[j] = c
    if kind == "medial" and js.get("gaster_tip") is not None and gaster_fraction is not None:
        i3, i4, i5 = (joint_names.index(k) for k in ("b_a_3", "b_a_4", "b_a_5"))
        ct = float(cf.get("gaster_tip") or 0.0)
        if ct >= conf_min and np.isfinite(P[i3]).all():
            tip = (np.asarray(js["gaster_tip"], dtype=np.float64) - norm["centre"]) / norm["scale"]
            P[i5], P[i4] = tip, P[i3] + gaster_fraction * (tip - P[i3])
            w[i4] = w[i5] = max(w[i3], conf_min)
            missing = [m for m in missing if m not in ("b_a_4", "b_a_5")]
    return P, w, missing


def marker_regressor(fitter, joint_names):
    """(J, V) float64 virtual-marker regressor and the gaster fraction (port of SKEL-REG's Template.marker_regressor):
    centroid of the vertex ring at each articulation (vertices whose summed skinning weight of the joint's subtree lies
    in 0.25-0.75, area weighted); l_*_pt and an_3 = the limb tip vertex, b_a_5 = mean of the five extreme gaster
    vertices, b_a_4 on b_a_3 -> b_a_5 at its own ring's rest fraction, b_t = b_h (neck); wings keep the J_regressor."""
    sm = fitter.smal_model
    V0 = sm.v_template.detach().cpu().numpy().astype(np.float64)
    F = sm.faces.detach().cpu().numpy().astype(np.int64)
    W = sm.weights.detach().cpu().numpy().astype(np.float64)
    W = W / W.sum(1, keepdims=True)
    Jreg = sm.J_regressor.detach().cpu().numpy().astype(np.float64).T
    names = list(joint_names)
    idx = {n: i for i, n in enumerate(names)}
    par = np.asarray(sm.parents).astype(int)
    par[0] = -1
    nJ, nV = len(names), len(V0)
    children = [[c for c in range(nJ) if par[c] == j] for j in range(nJ)]
    dom = W.argmax(1)
    J0 = Jreg @ V0

    def desc(j):
        out, st = [], [j]
        while st:
            k = st.pop()
            out.append(k)
            st += children[k]
        return out

    tri = V0[F]
    fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    va = np.zeros(nV)
    for k in range(3):
        np.add.at(va, F[:, k], fa / 3)

    def ring(j):
        dsum = W[:, desc(j)].sum(1)
        for lo, hi in ((0.25, 0.75), (0.1, 0.9), (0.02, 0.98)):
            m = (dsum > lo) & (dsum < hi)
            if m.sum() >= 3:
                r = va * m * (1 - np.abs(2 * dsum - 1) + 0.05)
                return r / r.sum()
        return None

    M = Jreg.copy()
    for j, n in enumerate(names):
        if n.startswith("w_") or j == 0:
            continue
        if not children[j] and (n.startswith("an_3") or "_pt_" in n):
            u = J0[j] - J0[par[j]]
            ids = np.where(np.isin(dom, [j, par[j]]))[0]
            M[j] = 0.0
            M[j, ids[np.argmax(V0[ids] @ u)]] = 1.0
            continue
        if n in ("b_a_4", "b_a_5"):
            continue
        r = ring(j)
        if r is not None:
            M[j] = r
    i3, i4, i5 = idx["b_a_3"], idx["b_a_4"], idx["b_a_5"]
    u = J0[i5] - J0[i3]
    ids = np.where(np.isin(dom, [i3, i4, i5]))[0]
    M[i5] = 0.0
    M[i5, ids[np.argsort(V0[ids] @ u)[-5:]]] = 0.2
    a, b = M[i3] @ V0, M[i5] @ V0
    r4 = ring(i4)
    c4 = (r4 @ V0) if r4 is not None else J0[i4]
    f = float(np.clip(((c4 - a) @ (b - a)) / ((b - a) @ (b - a)), 0.2, 0.8))
    M[i4] = (1 - f) * M[i3] + f * M[i5]
    M[0] = M[idx["b_h"]]
    return M, f


def marker_model(fitter, joint_names, Mreg):
    """Model whose rest reference points J(beta) are the virtual markers (linear in beta like the pivots); row 0 stays
    the true root pivot, the centre of the global rotation (closed_form_init's trans formula needs it)."""
    m = Model(fitter, joint_names)
    sm = fitter.smal_model
    jreg = sm.J_regressor.detach().cpu().numpy().astype(np.float64).T
    nb = sm.shapedirs.shape[0]
    sd3 = sm.shapedirs.detach().cpu().numpy().astype(np.float64).reshape(nb, -1, 3).transpose(1, 2, 0)
    Mr = np.asarray(Mreg, dtype=np.float64).copy()
    Mr[0] = jreg[0]
    m.J0 = Mr @ m.verts
    m.Jdirs = np.einsum("jv,vck->jck", Mr, sd3)
    return m


def excluded_bones(missing, parents):
    """Joint indices whose segments are freed for a broken-off part: each missing joint, everything distal to it, and
    the segment ending at it when its parent is a single-child chain joint (reproduces the synthetic DAMAGE lists)."""
    children = [[c for c in range(len(parents)) if parents[c] == j and c != j] for j in range(len(parents))]
    out = set()
    for j in missing:
        out.add(j)
        p = int(parents[j])
        if p > 0 and len(children[p]) == 1:
            out.add(p)
        stack = list(children[j])
        while stack:
            c = stack.pop()
            out.add(c)
            stack += children[c]
    out.discard(0)
    return sorted(out)


def body_length(verts, joints, dominant, joint_names):
    """Body length as in custom_processing/omniant_synthetic.py: extent of the head/mesosoma/waist/gaster vertices
    along the b_a_4 -> b_h axis of the regressed joints."""
    trunk = [joint_names.index(k) for k in TRUNK]
    u = joints[joint_names.index("b_h")] - joints[joint_names.index("b_a_4")]
    u = u / np.linalg.norm(u)
    p = verts[np.isin(dominant, trunk)] @ u
    return float(p.max() - p.min())


# ------------------------------------------------------------------------------------------ closed form (D1-D3)
def umeyama(src, dst, w=None, with_scale=True):
    """dst ~= s R src + t (weighted). Returns s, R, t."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd((xd * w[:, None]).T @ xs)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt)) or 1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / float((w * (xs**2).sum(1)).sum())) if with_scale else 1.0
    return s, R, md - s * R @ ms


def _rot(aa):
    return Rotation.from_rotvec(np.asarray(aa, dtype=np.float64)).as_matrix()


def _log(R):
    return Rotation.from_matrix(R).as_rotvec()


def _min_rotation(u, v):
    u = u / (np.linalg.norm(u) + 1e-30)
    v = v / (np.linalg.norm(v) + 1e-30)
    ax = np.cross(u, v)
    sn, cs = np.linalg.norm(ax), float(np.dot(u, v))
    if sn < 1e-9:
        if cs > 0:
            return np.eye(3)
        p = np.cross(u, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(u, [0, 1.0, 0])
        return _rot(np.pi * p / np.linalg.norm(p))
    return _rot(ax / sn * np.arctan2(sn, cs))


def _kabsch_uncentred(U, V, w):
    H = (np.asarray(V) * np.asarray(w)[:, None]).T @ np.asarray(U)
    Us, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Us) * np.linalg.det(Vt)) or 1.0
    return Us @ D @ Vt


def _shape_pairs(m, ok):
    pairs = set()
    for c in range(1, m.nj):
        p = m.parents[c]
        if m.names[c].startswith("w_") or (m.names[p], m.names[c]) == ("b_t", "b_h"):
            continue
        if ok[p] and ok[c]:
            pairs.add((p, c))
    for grp in (ROOT_RIGID, HEAD_RIGID):
        ids = [m.names.index(n) for n in grp if ok[m.names.index(n)]]
        for i in range(len(ids)):
            for k in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[k]))
                if {m.names[a], m.names[b]} != {"b_t", "b_h"}:
                    pairs.add((a, b))
    return sorted(pairs)


def _fit_shape(P, w, m, s0, L, n_iter=8, noise=0.01, robust_c=0.02, prior=1.0):
    """D2: (log sigma, beta) from pose-invariant distances, IRLS Geman-McClure (width robust_c * L), shape prior."""
    ok = np.isfinite(P).all(-1) & (w > 0)
    pairs = _shape_pairs(m, ok)
    if len(pairs) < 3:
        return np.log(s0), np.zeros(m.nb)
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    wp = np.sqrt(w[a] * w[b])
    Lobs = np.linalg.norm(P[b] - P[a], axis=1)
    dJd = m.Jdirs[b] - m.Jdirs[a]
    x = np.zeros(1 + m.nb)
    x[0] = np.log(s0)
    sd = m.shape_sd
    for _ in range(n_iter):
        sig, beta = np.exp(x[0]), x[1:]
        J = m.rest_pivots(beta)
        d = J[b] - J[a]
        ln = np.linalg.norm(d, axis=1) + 1e-12
        r = (sig * ln - Lobs) / (noise * L)
        ww = wp * (robust_c**2 / ((r * noise) ** 2 + robust_c**2)) ** 2
        Jac = np.zeros((len(pairs), 1 + m.nb))
        Jac[:, 0] = sig * ln / (noise * L)
        Jac[:, 1:] = sig * np.einsum("mc,mck->mk", d / ln[:, None], dJd) / (noise * L)
        Jp = np.zeros((m.nb, 1 + m.nb))
        Jp[:, 1:] = np.diag(np.sqrt(prior) / sd)
        A = Jac.T @ (Jac * ww[:, None]) + Jp.T @ Jp + 1e-6 * np.eye(1 + m.nb)
        g = Jac.T @ (r * ww) + Jp.T @ (np.sqrt(prior) * beta / sd)
        step = np.linalg.solve(A, -g)
        x = x + step
        if np.abs(step).max() < 1e-6:
            break
    return x[0], x[1:]


def _similarity(P, w, m, beta, scale=None):
    ids = [m.names.index(n) for n in ROOT_RIGID]
    ok = [i for i in ids if np.isfinite(P[i]).all() and w[i] > 0]
    if len(ok) < 3:
        raise ValueError(f"only {len(ok)} root-rigid joints proposed")
    J = m.rest_pivots(beta)
    s, R, t = umeyama(J[ok], P[ok], w[ok])
    if scale is not None:
        ww = w[ok] / w[ok].sum()
        t, s = ww @ P[ok] - scale * R @ (ww @ J[ok]), scale
    return s, R, t


_PHIS = np.linspace(-np.pi, np.pi, 145)
_PHIS = _PHIS[np.argsort(np.abs(_PHIS), kind="stable")]


def _violation(aa, lo, hi):
    return np.maximum(aa - hi, 0).sum(-1) + np.maximum(lo - aa, 0).sum(-1)


def _twist_for_limits(Rmin, u, lo, hi):
    u = u / (np.linalg.norm(u) + 1e-30)
    aa = (Rotation.from_matrix(Rmin) * Rotation.from_rotvec(u[None] * _PHIS[:, None])).as_rotvec()
    return aa[int(np.argmin(_violation(aa, lo[None], hi[None]) + 1e-9 * np.abs(_PHIS)))]


def _fk_ik(Pm, w, m, beta, R_root):
    """D3 in SMAL coordinates (targets already mapped by (P - trans) / s). Returns theta (J, 3), root row = global."""
    J = m.rest_pivots(beta)
    GR = [None] * m.nj
    X = np.zeros((m.nj, 3))
    theta = np.zeros((m.nj, 3))
    GR[0], X[0], theta[0] = R_root, J[0], _log(R_root)
    okP = np.isfinite(Pm).all(-1) & (w > 0)
    for j in range(1, m.nj):
        p = m.parents[j]
        X[j] = GR[p] @ (J[j] - J[p]) + X[p]
        kids = [c for c in m.children[j] if okP[c] and not m.names[c].startswith("w_")]
        Rj = np.eye(3)
        if kids:
            U = np.array([J[c] - J[j] for c in kids])
            V = np.array([GR[p].T @ (Pm[c] - X[j]) for c in kids])
            lo, hi = m.limits[j, :, 0], m.limits[j, :, 1]
            if len(kids) == 1:
                aa = _log(_min_rotation(U[0], V[0]))
                if _violation(aa, lo, hi) > 0:
                    aa = _twist_for_limits(_rot(aa), U[0], lo, hi)
            else:
                aa = _log(_kabsch_uncentred(U, V, w[kids]))
            aa = np.clip(aa, lo, hi)
            Rj = _rot(aa)
            theta[j] = aa
        GR[j] = GR[p] @ Rj
    return theta


def closed_form_init(P, w, m, do_shape=True):
    """P (J,3) proposals in the normalised target frame (nan = none), w (J,) weights ->
    dict(global_rot, joint_rot, betas, log_scale, trans) for fitter.pivots = exp(log_scale) * J_transformed + trans."""
    beta0 = np.zeros(m.nb)
    s0, _, _ = _similarity(P, w, m, beta0)
    if do_shape:
        log_sig, beta = _fit_shape(P, w, m, s0, s0 * m.body_length_template)
        s = float(np.exp(log_sig))
    else:
        beta = beta0
        s, _, _ = _similarity(P, w, m, beta)
    _, R, t = _similarity(P, w, m, beta, scale=s)
    J = m.rest_pivots(beta)
    trans = t + s * (R @ J[0]) - s * J[0]  # the global rotation acts about the root pivot
    theta = _fk_ik((P - trans) / s, w, m, beta, R)
    return {
        "global_rot": theta[0],
        "joint_rot": theta[1:],
        "betas": beta,
        "log_scale": float(np.log(s)),
        "trans": trans,
    }


# ------------------------------------------------------------------------------------------ batch preparation
def prepare_batch(
    fitter,
    mesh_names,
    norms,
    joints_dir,
    device,
    joint_sigma=0.02,
    do_shape=True,
    joint_kind="pivot",
    conf_min=0.0,
    native_start=False,
    limbs_dir=None,
):
    """Everything a Stage needs for the joint-informed schedule (fitter_3d/antscan_joints_cfg.yaml), for meshes loaded
    with load_meshes(return_norm=True): proposals in the normalised frame, closed-form initialisation of the fitter
    (in place), rotation_mask and chamfer source mask for reported missing parts, body length for the loss
    normalisation. Options (defaults = previous behaviour): joint_kind 'medial' (marker reference points, see the
    module docstring), conf_min,
    native_start (closed-form articulated start in native parameters, native_start_init below; with it the batch never
    raises for lack of root-rigid joints; limbs_dir then supplies the mandible tips)."""
    import config

    names = [str(n) for n in config.joint_names]
    Mreg, gfrac = marker_regressor(fitter, names) if joint_kind == "medial" else (None, None)
    m = marker_model(fitter, names, Mreg) if joint_kind == "medial" else Model(fitter, names)
    B = len(mesh_names)
    P, W, excl = np.zeros((B, m.nj, 3)), np.zeros((B, m.nj)), []
    inits, frames = [], []
    for b, mn in enumerate(mesh_names):
        stem = os.path.splitext(os.path.basename(mn))[0]
        stem = stem[: -len("_processed")] if stem.endswith("_processed") else stem
        jf = os.path.join(joints_dir, f"{stem}_joints.json")
        P[b], W[b], missing = load_proposal(
            jf, names, norms[b], conf_min=conf_min, kind=joint_kind, gaster_fraction=gfrac
        )
        excl.append(excluded_bones([names.index(n) for n in missing], m.parents))
        if native_start:
            inits.append(native_start_init(jf, norms[b], joint_kind, conf_min, stem, limbs_dir))
        else:
            inits.append(closed_form_init(P[b], W[b], m, do_shape=do_shape))
        prod = json.load(open(jf))
        if joint_kind == "pivot" and b == 0 and ("method" in prod or "ensemble" in prod):
            import warnings

            warnings.warn(
                f"{os.path.basename(jf)} looks like a product of the joint method (fitter_3d/antscan_joints), whose "
                "joints are medial articulation points, but --joint_kind is 'pivot': pass --joint_kind medial",
                stacklevel=2,
            )
        fr = prod.get("frame") or {}
        frames.append(
            (
                np.asarray(fr.get("rotation", np.eye(3)), dtype=np.float64),
                np.asarray(fr.get("origin_scan", np.zeros(3)), dtype=np.float64),
            )
        )
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    keep = np.ones((B, len(m.dom)), dtype=bool)
    with torch.no_grad():
        for k in ("global_rot", "joint_rot", "betas", "trans"):
            getattr(fitter, k).copy_(t([ini[k] for ini in inits]))
        fitter.log_scale.copy_(t([[ini["log_scale"]] for ini in inits]))
        if native_start:  # per-bone isotropic scales and pivot offsets of the closed-form start
            fitter.log_beta_scales.copy_(t([ini["log_beta_scales"] for ini in inits]))
            fitter.betas_trans.copy_(t([ini["betas_trans"] for ini in inits]))
        for b, ex in enumerate(excl):
            for j in ex:  # the segments of a missing part keep the rest pose and are not matched to the scan
                fitter.rotation_mask[b, j - 1] = 0.0
                if not native_start:  # (the native start already left them in the rest / mirrored articulation)
                    fitter.joint_rot[b, j - 1] = 0.0
            keep[b] = ~np.isin(m.dom, ex)
        verts, joints = fitter(return_joints=True)
    L = [body_length(verts[b].cpu().numpy(), joints[b].cpu().numpy(), m.dom, names) for b in range(B)]
    out = {
        "target_joints": t(np.nan_to_num(P)),
        "joint_weights": t(W),
        "body_length": t(L),
        "joint_sigma": joint_sigma,
        "src_keep": torch.tensor(keep, device=device),
        "target_norm": {
            "target_centre": np.stack([n["centre"] for n in norms]),
            "target_scale": np.array([n["scale"] for n in norms]),
            "target_units": "micrometre",
            # x_scan = (x_norm * target_scale + target_centre) @ target_frame_rotation + target_frame_origin
            "target_frame_rotation": np.stack([f[0] for f in frames]),
            "target_frame_origin": np.stack([f[1] for f in frames]),
        },
    }
    if Mreg is not None:
        out["joint_ref_regressor"] = t(Mreg)
    return out


def native_start_init(jfile, norm, joint_kind, conf_min, stem, limbs_dir=None):
    """Closed-form articulated start in native parameters (fitter_3d/geomreg/native.py) for one specimen, in the frame
    of load_meshes(return_norm=True). Returns the keys of closed_form_init plus log_beta_scales (J,3) and betas_trans
    (J,3). Unlike closed_form_init it does not need three root-rigid joints: with fewer it uses the trunk chord and the
    proposals' frame (the canonical frame the meshes are in), then the frame and the body length."""
    import config
    from fitter_3d.geomreg import model as g_model
    from fitter_3d.geomreg import native as g_native
    from fitter_3d.geomreg import spec as g_spec

    g_model.set_default_model(config.SMAL_FILE)
    tpl = g_model.template()
    d = json.load(open(jfile))
    fr = d.get("frame") or {}
    frame = g_spec.Frame(
        np.asarray(fr.get("rotation", np.eye(3)), float),
        np.asarray(fr.get("origin_scan", np.zeros(3)), float),
        norm["centre"],
        norm["scale"],
    )
    P, w, _, _ = g_spec.joint_targets(jfile, frame, tpl, conf_min, joint_kind)
    hf, cf = (d.get("joints_scan") or {}).get("head_front"), (d.get("confidence") or {}).get("head_front")
    head_front = (
        frame.to_norm(np.asarray(hf, float))
        if (joint_kind == "medial" and hf is not None and (cf is None or cf >= conf_min))
        else None
    )
    tips = {}
    if limbs_dir:
        tips = {
            j: q["tip"] for j, q in g_spec.mandible_tips(os.path.join(limbs_dir, stem), frame, tpl).items() if w[j] > 0
        }
    BL = float(d["body_length_um"]) / frame.s if d.get("body_length_um") else None
    st = g_native.closed_form_start(
        tpl, P, w, kind=joint_kind, R_prior=np.eye(3), BL=BL, head_front=head_front, tips=tips
    )
    p = g_native.to_native(st)
    return {
        "global_rot": p["global_rot"],
        "joint_rot": p["joint_rot"],
        "betas": p["betas"],
        "log_scale": float(p["log_scale"][0]),
        "trans": p["trans"],
        "log_beta_scales": p["log_beta_scales"],
        "betas_trans": p["betas_trans"],
    }
