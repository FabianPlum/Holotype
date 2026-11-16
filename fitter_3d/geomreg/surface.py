"""The surface stage of the geometric registration core: closed-form native start -> scan
envelope, vertex level. Developed from the earlier surface registration.

Order of work per specimen (all in the fitter's normalised target frame, lengths relative to the body length BL):
  S  start          native.closed_form_start: every usable joint's reference point on its target, bones without usable
                    joints in the rest articulation / mirrored from their partner (no optimisation; replaces the
                    fitter's Stage 0 / 1 and the earlier bone snap)
     girth          closed form: girth of the trunk parts from the scan's inscribed radius (snap.girth_from_scan)
  T  tube warp      closed form: limb rings onto the limb centrelines, radial scale per ring (tubes.tube_warp) [needs limbs]
  L  scan labels    tube claims from the centrelines + nearest compatible template vertex (scan_codes)
  C  coarse stage   one affine map per joint-supported bone, blended with the skinning weights; robust part-aware
                    point-to-plane in both directions, joint landmarks / marker term, ring (centreline) term, joint
                    continuity, ridge. Bones without usable joints get NO map of their own and do not share a
                    neighbour's affine map (that made needles in an earlier version): they follow the neighbouring supported bone
                    RIGIDLY, pinned at the joint they share with it
  F  fine stage     free vertices (steps scaled by the local edge length); same data terms; regularisers RELATIVE to
                    the coarse result: edge length, dihedral angles, tangential anchor; anti-fold barrier
  status            vertex: 0 observed (compatible scan surface at the vertex), 1 hidden (none: contact / covered /
                    absent), 2 unsupported (bone without usable joints: placed by the start's imputation only);
                    bone: ok / review / unregistered
Every optimised term is a plain PyTorch function of the vertex tensor (loss_* below). The stage runs on the CPU by
default (nearest neighbours by scipy cKDTree, fixed seeds: the same input gives the same output); `device='cuda'`
uses pytorch3d's knn and is NOT bit-reproducible.
"""

import contextlib
import time

import numpy as np
import torch

from . import snap, tubes
from .model import LIMB_CHAINS

_TREES = {}


@contextlib.contextmanager
def single_thread(enabled=True):
    """One thread for torch and, when threadpoolctl is installed, for every BLAS / OpenMP pool of the process, for the
    duration of the block; the previous limits are restored. This is what makes the CPU path bit-reproducible: with
    two threads the thread pools split the same sums differently from run to run on a loaded machine."""
    if not enabled:
        yield
        return
    n_prev = torch.get_num_threads()
    torch.set_num_threads(1)
    with contextlib.ExitStack() as stack:
        try:
            from threadpoolctl import threadpool_limits

            stack.enter_context(threadpool_limits(limits=1))
        except Exception:  # threadpoolctl missing: torch's own limit still applies
            pass
        try:
            yield
        finally:
            torch.set_num_threads(n_prev)


class _KNN:
    def __init__(self, dists, idx):
        self.dists, self.idx = dists, idx


def knn_points(X, Y, K=1):
    """K nearest neighbours of X (1, n, 3) in Y (1, m, 3): squared distances and indices, shapes (1, n, K), like
    pytorch3d.ops.knn_points. CPU: scipy cKDTree (exact, deterministic); GPU: pytorch3d."""
    if X.device.type != "cpu":
        from pytorch3d.ops import knn_points as _k

        return _k(X, Y, K=K)
    from scipy.spatial import cKDTree

    x = X[0].detach().numpy().astype(np.float64)
    key = (Y.data_ptr(), tuple(Y.shape))
    if _TREES.get("key") == key:  # the scan points do not change during a registration: one tree
        tree = _TREES["tree"]
    else:
        tree = cKDTree(Y[0].detach().numpy().astype(np.float64))
        if Y.shape[1] > 30000:
            _TREES.update(key=key, tree=tree, keep=Y)
    K = min(K, tree.n)
    d, i = tree.query(x, k=K, workers=2)  # two query threads (the 2-thread budget); the result does not depend on it
    if K == 1:
        d, i = d[:, None], i[:, None]
    return _KNN(torch.as_tensor(d.astype(np.float32) ** 2)[None], torch.as_tensor(i.astype(np.int64))[None])


OPTS = dict(
    # labels / correspondences
    use_limbs=True,
    claim_alpha=1.9,
    claim_cos=0.25,
    ndot=0.3,
    knn=12,
    label_gate=0.02,
    label_every=1,
    gate_r=2.5,
    gate_min=0.003,
    girth=True,
    tube=True,
    gaster_tube=True,
    tube_radial=True,
    tube_coxa=False,
    per_bone=4,
    rigid_unsupported=True,
    depenetrate=True,
    unfold=True,
    # coarse stage
    coarse=True,
    c_opt="adam",
    c_outer=24,
    c_inner=10,
    c_lr=0.004,
    c_gate=(0.03, 0.015),
    c_ridge=0.02,
    c_ridge_tube=0.2,
    c_cont=30.0,
    c_p2p=0.1,
    c_min_verts=30,
    c_land=3.0,
    c_merge_gaster=False,
    c_kappa=(0.6, 0.15),
    f_kappa=(0.2, 0.08),
    c_betas="auto",
    c_beta_prior=0.05,
    c_beta_lr=0.02,
    c_beta_outer=14,
    c_beta_accept=0.72,
    c_beta_limbs=True,
    gate_grow_below=0.35,
    gate_boost_max=4.0,
    # fine stage
    fine=True,
    f_opt="adam",
    f_outer=40,
    f_inner=10,
    f_lr=0.02,
    f_gate=(0.015, 0.008),
    w_edge=3.0,
    w_bend=1.0,
    w_tan=0.05,
    w_fold=50.0,
    fold_cos=0.3,
    fold_area=0.3,
    f_p2p=0.0,
    precond=True,
    w_repel=0.0,
    repel_knn=24,
    repel_r=2.0,
    repel_depth=0.3,
    repel_margin=0.0,
    # shared
    w_data=1.0,
    w_s2t=1.0,
    w_marker=3.0,
    w_ring=1.0,
    sigma_joint=0.02,
    data_scale=(0.001, 0.005),
    robust_min=0.0015,
    n_scan=40000,
    obs_gate=0.006,
    seed=0,
    cpu_threads=1,
    review_kink=60.0,
    review_rmin=0.5,
    review_obs=0.5,
    review_shift=0.01,
    measure_cap=0.02,
)


def check_opts(opts):
    """Raise ValueError for keys of a registration options dict that nothing reads (a typo would otherwise be ignored
    silently): the keys of OPTS, plus 'start' whose keys are those of geomreg.native.OPTS."""
    from . import native

    opts = opts or {}
    bad = sorted(set(opts) - set(OPTS) - {"start"})
    start = opts.get("start") or {}
    if not isinstance(start, dict):
        raise ValueError("'start' must be a dict of geomreg.native.OPTS")
    bad += [f"start.{k}" for k in sorted(set(start) - set(native.OPTS))]
    if bad:
        raise ValueError(f"unknown option(s) {bad}; known: {sorted(OPTS)} and start: {sorted(native.OPTS)}")


# ----------------------------------------------------------------------------------------------- tensor helpers
def fnormals(V, F):
    n = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1)
    a = n.norm(dim=1)
    return n / a.clamp(min=1e-20)[:, None], 0.5 * a


def vnormals(V, F):
    n = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1)
    out = torch.zeros_like(V)
    for k in range(3):
        out.index_add_(0, F[:, k], n)
    return out / out.norm(dim=1, keepdim=True).clamp(min=1e-20)


def dihedral(V, E, EO):
    """signed dihedral angle per edge (0 = flat), from the two opposite vertices"""
    a, b = V[E[:, 0]], V[E[:, 1]]
    c, d = V[EO[:, 0]], V[EO[:, 1]]
    e = b - a
    n1 = torch.cross(e, c - a, dim=1)
    n2 = torch.cross(d - a, e, dim=1)
    en = e / e.norm(dim=1, keepdim=True).clamp(min=1e-20)
    n1 = n1 / n1.norm(dim=1, keepdim=True).clamp(min=1e-20)
    n2 = n2 / n2.norm(dim=1, keepdim=True).clamp(min=1e-20)
    cos = (n1 * n2).sum(1).clamp(-1, 1)
    sin = (torch.cross(n1, n2, dim=1) * en).sum(1)
    return torch.atan2(sin, cos)


class Tpl:
    """template tensors on a device"""

    def __init__(self, tpl, device):
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        self.np = tpl
        self.F, self.E, self.EO = t(tpl.F, torch.long), t(tpl.E, torch.long), t(tpl.EO, torch.long)
        self.dom, self.W = t(tpl.dom, torch.long), t(tpl.W)
        K = int((tpl.W > 0).sum(1).max())  # non-zero skinning weights per vertex (4 in OmniAnt25)
        kb = np.argsort(-tpl.W, axis=1)[:, :K]
        self.kb, self.kw = t(kb, torch.long), t(np.take_along_axis(tpl.W, kb, 1))
        self.M = t(tpl.marker_regressor())
        self.adj = t(tpl.adj, torch.bool)
        self.adj2 = t(tpl.adj2, torch.bool)
        limb_of_bone = np.full(tpl.nJ, -1)
        for li, (limb, chain) in enumerate(LIMB_CHAINS.items()):
            for n in chain:
                limb_of_bone[tpl.idx[n]] = li
        self.limb_of_bone = limb_of_bone
        nJ, nL = tpl.nJ, len(LIMB_CHAINS)
        # compatibility table: scan label code x bone. codes: 0..nJ-1 bone, nJ..nJ+nL-1 limb with unknown segment,
        # nJ+nL unlabelled (compatible with every bone), nJ+nL+1 unresolved (compatible with none)
        T = np.zeros((nJ + nL + 2, nJ), bool)
        T[:nJ] = tpl.adj
        for li in range(nL):
            T[nJ + li] = limb_of_bone == li
        T[nJ + nL] = True
        self.compat = t(T, torch.bool)
        self.code_unl, self.code_unres = nJ + nL, nJ + nL + 1
        self.nJ, self.nL, self.nV = nJ, nL, tpl.nV


# ----------------------------------------------------------------------------------------------- correspondences
def scan_codes(spec, T, V, N, o, sup_v, dev, claim=None, gate_v=None):
    """Label code per scan vertex. Tube claims first; the rest takes the bone of the nearest template vertex with an
    agreeing normal inside the label gate; nothing compatible -> unlabelled. Scan surface of unsupported bones ->
    unresolved (no template part may be fitted to it)."""
    Ps, Ns = spec.Ps_t, spec.Ns_t
    nS = len(Ps)
    code = torch.full((nS,), T.code_unl, dtype=torch.long, device=dev)
    kn = knn_points(Ps[None], V[None], K=o["knn"])
    d, idx = kn.dists[0].sqrt(), kn.idx[0]
    lg = (
        o["label_gate"] * spec.BL
        if gate_v is None
        else torch.maximum(gate_v[idx], torch.tensor(o["label_gate"] * spec.BL, device=dev))
    )
    ok = (d < lg) & ((Ns[:, None, :] * N[idx]).sum(-1) > o["ndot"])
    has = ok.any(1)
    first = ok.float().argmax(1)
    near = idx[torch.arange(nS, device=dev), first]
    code[has] = T.dom[near[has]]
    unres = has & ~sup_v[near]
    if claim is not None:
        lab_limb, lab_bone = claim
        cl = lab_limb >= 0
        cb = cl & (lab_bone >= 0)
        code[cl] = T.nJ + lab_limb[cl]
        code[cb] = lab_bone[cb]
        unres = unres & ~cl
        sup_b = torch.as_tensor(spec.support, device=dev)
        unres |= cb & ~sup_b[lab_bone.clamp(min=0)]
    code[unres] = T.code_unres
    return code


def correspond(spec, T, V, N, code, gate_v, o, dev, sub=None):
    """template -> scan and scan -> template pairs: nearest candidate among the K nearest with agreeing normals, a
    compatible label and inside the vertex's gate. Returns dict of index / validity tensors."""
    Ps, Ns = spec.Ps_t, spec.Ns_t
    K = o["knn"]
    ar = torch.arange(T.nV, device=dev)
    kn = knn_points(V[None], Ps[None], K=K)
    d, idx = kn.dists[0].sqrt(), kn.idx[0]
    ok = (d < gate_v[:, None]) & ((N[:, None, :] * Ns[idx]).sum(-1) > o["ndot"]) & T.compat[code[idx], T.dom[:, None]]
    ok &= spec.sup_v[:, None]
    has1 = ok.any(1)
    f1 = ok.float().argmax(1)
    s1 = idx[ar, f1]
    # scan -> template
    sid = sub if sub is not None else torch.arange(len(Ps), device=dev)
    kn2 = knn_points(Ps[sid][None], V[None], K=K)
    d2, idx2 = kn2.dists[0].sqrt(), kn2.idx[0]
    ok2 = (
        (d2 < gate_v[idx2])
        & ((Ns[sid][:, None, :] * N[idx2]).sum(-1) > o["ndot"])
        & T.compat[code[sid]][torch.arange(len(sid), device=dev)[:, None], T.dom[idx2]]
    )
    ok2 &= spec.sup_v[idx2]
    has2 = ok2.any(1)
    f2 = ok2.float().argmax(1)
    v2 = idx2[torch.arange(len(sid), device=dev), f2]
    return dict(i1=ar[has1], s1=s1[has1], has1=has1, s2=sid[has2], v2=v2[has2], has2_share=float(has2.float().mean()))


def sched(ab, f):
    """geometric schedule from ab[0] (f = 0) to ab[1] (f = 1); 0 stays 0"""
    a, b = float(ab[0]), float(ab[1])
    return a if (a <= 0 or b <= 0) else a * (b / a) ** f


def robust_w(r, floor):
    s = torch.clamp(1.4826 * r.abs().median(), min=floor) * 2.0
    return 1.0 / (1.0 + (r / s) ** 2)


# ----------------------------------------------------------------------------------------------- loss terms
def loss_data(V, spec, c, ds, o, p2p=0.0):
    """point-to-plane, both directions; residuals in units of the per-vertex data scale ds; robust weights are fixed
    between correspondence updates (stored in c)"""
    Ps, Ns = spec.Ps_t, spec.Ns_t
    out = {}
    if len(c["i1"]):
        dv = V[c["i1"]] - Ps[c["s1"]]
        r = (dv * Ns[c["s1"]]).sum(1) / ds[c["i1"]]
        e = c["w1"] * r**2
        if p2p > 0:
            e = e + p2p * c["w1"] * (dv**2).sum(1) / ds[c["i1"]] ** 2
        out["t2s"] = e.sum() / max(len(c["i1"]), 1)
    if len(c["s2"]):
        dv = V[c["v2"]] - Ps[c["s2"]]
        r = (dv * Ns[c["s2"]]).sum(1) / ds[c["v2"]]
        e = c["w2"] * c["a2"] * r**2
        if p2p > 0:
            e = e + p2p * c["w2"] * c["a2"] * (dv**2).sum(1) / ds[c["v2"]] ** 2
        out["s2t"] = o["w_s2t"] * e.sum() / c["a2"].sum().clamp(min=1e-12)
    return out


def loss_points(V, Wrows, targets, wts, BL, sigma):
    """Geman-McClure distance of fixed linear vertex combinations (virtual markers, limb rings) to target points"""
    if Wrows is None or len(targets) == 0:
        return None
    ref = Wrows @ V
    d2 = ((ref - targets) ** 2).sum(-1) / BL**2
    rho = d2 / (d2 + sigma**2)
    return (wts * rho).sum() / wts.sum().clamp(min=1e-8)


def loss_rel_edge(V, T, len_ref):
    e = (V[T.E[:, 0]] - V[T.E[:, 1]]).norm(dim=1)
    return (((e - len_ref) / len_ref) ** 2).mean()


def loss_bend(V, T, dih_ref):
    d = dihedral(V, T.E, T.EO) - dih_ref
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return (d**2).mean()


def loss_tan(V, Vref, Nref, ell):
    d = V - Vref
    dt = d - (d * Nref).sum(1, keepdim=True) * Nref
    return ((dt**2).sum(1) / ell**2).mean()


def loss_fold(V, T, fn_ref, fa_ref, cos_min, area_min):
    fn, fa = fnormals(V, T.F)
    c = (fn * fn_ref).sum(1)
    return (torch.relu(cos_min - c) ** 2).mean() + (torch.relu(area_min - fa / fa_ref) ** 2).mean()


def repel_pairs(V, N, T, sup_v, ell, o):
    """(i, k): template vertex i lies behind the tangent plane of its nearest vertex k of a part that is neither its
    own nor joined to it by a joint (siblings such as two coxae count as foreign), closer than repel_r local edge lengths: i is inside (or
    touching) a foreign part. Used by loss_repel; rebuilt at every correspondence update."""
    kn = knn_points(V[None], V[None], K=o["repel_knn"])
    d, idx = kn.dists[0].sqrt(), kn.idx[0]
    dom = T.dom
    foreign = ~T.adj[dom[:, None], dom[idx]]
    dn = ((V[:, None, :] - V[idx]) * N[idx]).sum(-1)
    # clearly behind the foreign tangent plane AND the two surfaces face each other (a vertex that has sunk into a
    # foreign part has an outward normal pointing further in); grazing pairs are left alone: with a nearest VERTEX
    # instead of a nearest surface point the plane test is unreliable closer than one edge length (first attempt:
    # every pair within 0.2 edge lengths in front of the plane was pushed, and the crossings rose 30-fold)
    ok = foreign & (dn < -o["repel_depth"] * ell[idx]) & (d < o["repel_r"] * torch.maximum(ell[:, None], ell[idx]))
    ok &= (N[:, None, :] * N[idx]).sum(-1) < -0.2
    ok &= sup_v[:, None] | sup_v[idx]
    has = ok.any(1)
    first = ok.float().argmax(1)
    i = torch.nonzero(has)[:, 0]
    return i, idx[i, first[i]]


def loss_repel(V, N, pairs, ell, margin):
    """one-sided: vertex i is pushed out of the foreign part along the foreign vertex's normal until it is `margin`
    local edge lengths in front of its tangent plane; the foreign vertex gets the opposite push"""
    i, k = pairs
    if len(i) == 0:
        return V.sum() * 0.0
    dn = ((V[i] - V[k]) * N[k]).sum(1) / ell[k]
    return (torch.relu(margin - dn) ** 2).sum() / V.shape[0]


# ----------------------------------------------------------------------------------------------- the stage
def _gate(spec, T, V, g_bl, o, dev, unit=None, kappa=None):
    """per-vertex gate. Limb vertices: g_bl of the body length but not more than gate_r local radii (distance to the
    limb's ring polyline, or the scan centreline radius where the limb product gives one). Other vertices: the larger
    of g_bl x BL and kappa[unit] x the unit's RMS radius (unit = (nV,) unit index, kappa (nU,)): a head that starts
    half as wide as the scan's lies several % BL from its surface, and a fixed gate never finds it."""
    g = torch.full((T.nV,), g_bl * spec.BL, device=dev)
    rl = spec.rloc_t
    limb = torch.isfinite(rl) if rl is not None else torch.zeros(T.nV, dtype=torch.bool, device=dev)
    if unit is not None and kappa is not None:
        nU = int(kappa.shape[0])
        cnt = torch.zeros(nU, device=dev).index_add_(0, unit, torch.ones(T.nV, device=dev)).clamp(min=1)
        cen = torch.zeros(nU, 3, device=dev).index_add_(0, unit, V) / cnt[:, None]
        rho = torch.sqrt(torch.zeros(nU, device=dev).index_add_(0, unit, ((V - cen[unit]) ** 2).sum(1)) / cnt)
        g = torch.where(limb, g, torch.maximum(g, (kappa * rho)[unit]))
    if rl is not None:
        lim = torch.clamp(o["gate_r"] * rl, min=o["gate_min"] * spec.BL)
        g = torch.where(limb, torch.minimum(g, lim), g)
    return g


def local_radius(spec, V, targets):
    """(nV,) local limb radius for limb vertices (nan elsewhere): distance of the vertex to its limb's station
    polyline, not smaller than the scan centreline radius at the nearest station that has one."""
    tpl = spec.tpl
    rl = np.full(tpl.nV, np.nan)
    st_all = tpl.limb_stations(spec.per_bone)
    rscan = {}
    for w, X, limb, r in targets:
        rscan.setdefault(limb, []).append((X, r))
    for limb, chain in LIMB_CHAINS.items():
        if limb == "G":  # the gaster keeps the part-size gates of the trunk parts
            continue
        S = st_all[limb]["stations"]
        M = np.stack([s_["w"] for s_ in S]) @ V
        js = [tpl.idx[n] for n in chain]
        vid = np.where(np.isin(tpl.dom, js))[0]
        P = V[vid]
        best = np.full(len(vid), np.inf)
        for q in range(len(M) - 1):
            p0, u = M[q], M[q + 1] - M[q]
            f = np.clip(((P - p0) @ u) / max(u @ u, 1e-18), 0, 1)
            best = np.minimum(best, np.linalg.norm(P - (p0 + f[:, None] * u), axis=1))
        if limb in rscan:
            Xr = np.stack([x for x, _ in rscan[limb]])
            rr = np.array([r for _, r in rscan[limb]])
            k = np.linalg.norm(P[:, None] - Xr[None], axis=2).argmin(1)
            best = np.maximum(best, rr[k])
        rl[vid] = best
    return rl


def part_status_from(tpl, support, diag, tube_info, o, how=None):
    """Per-bone status and the reasons for 'review'.
    'unregistered' = bone without usable joints (placed by continuation only).
    'review', three rules:
      (1) femur / tibia / basitarsus / scape whose scan centreline bends by more than review_kink degrees or narrows
          below review_rmin of its median radius INSIDE the segment: an articulation inside a segment, i.e. a joint was
          put down the neighbouring segment (thresholds fixed before looking; not for the coxa, whose
          centreline turns into the trunk, nor the tarsus rest / funiculus, which have articulations inside);
      (2) any part with at least 20 vertices of which fewer than review_obs found scan surface;
      (3) a LIMB part that the fine stage had to move by more than review_shift x BL (median);
      (4) every mandible (one joint only: see the end of this function).
    (2) and (3) come from the thin-limb synthetic set with exact truth (precision 87 % / 78 %, together
    about a quarter of the wrong parts). (3) is not applied to trunk parts: on real workers the waist nodes need
    1-4 % BL of free deformation because the template has no node shapes (a limit of the template), which says nothing about
    the registration being wrong. Leaf tips share their parent's status."""
    part_status = np.array(["ok"] * tpl.nJ, dtype=object)
    part_status[~np.asarray(support, bool)] = "unregistered"
    review = {}
    for limb, d in (tube_info or {}).items():
        for bone, q in (d.get("seg_diag") or {}).items():
            if not (bone.startswith("an_1") or ("_tr_" in bone or "_fe_" in bone or "_ti_" in bone)):
                continue
            j = tpl.idx[bone]
            if part_status[j] == "unregistered":
                continue
            why = {}
            if q["kink_deg"] > o["review_kink"]:
                why["kink_deg"] = float(q["kink_deg"])
            if q["r_min_rel"] < o["review_rmin"]:
                why["r_min_rel"] = float(q["r_min_rel"])
            if why:
                part_status[j] = "review"
                review[bone] = why
    cntv = np.bincount(tpl.dom, minlength=tpl.nJ)
    # rule (3) for the limb TUBES only. The coxa is a blob against the trunk that the tube warp no
    # longer pre-places (tube_coxa False); like the trunk parts it legitimately needs free deformation
    limb_bone = np.array([n.startswith(("l_", "an_")) and "_co_" not in n for n in tpl.names])
    for j in range(tpl.nJ):
        if part_status[j] == "unregistered" or cntv[j] < 20:
            continue
        why = {}
        if np.isfinite(diag["obs_share"][j]) and diag["obs_share"][j] < o["review_obs"]:
            why["observed_share"] = float(diag["obs_share"][j])
        if limb_bone[j] and np.isfinite(diag["shift_F"][j]) and diag["shift_F"][j] > o["review_shift"]:
            why["fine_shift_bl"] = float(diag["shift_F"][j])
        if why:
            part_status[j] = "review"
            review.setdefault(tpl.names[j], {}).update(why)
    for j in range(tpl.nJ):
        if (
            not tpl.children[j]
            and tpl.parents[j] >= 0
            and part_status[j] == "ok"
            and part_status[tpl.parents[j]] == "review"
        ):
            part_status[j] = "review"
    # (4) structural: a mandible has ONE joint (its base), so neither its length nor its direction is held by joints;
    # it follows the head's map and the surface terms only (trap-jaw mandibles end up on the head: the shared
    # benchmark found 2 such mandibles on the 21 development workers, with nothing in the stage's signals to tell)
    # a mandible that the start aimed at the tip of its own limb centreline has TWO points and is
    # not flagged by this rule
    for j in range(tpl.nJ):
        if tpl.names[j].startswith("ma_") and part_status[j] == "ok" and not (how is not None and how[j] == "tip"):
            part_status[j] = "review"
            review.setdefault(tpl.names[j], {})["single_joint_part"] = 1.0
    return part_status, review


FLAG_PREFIXES = (
    "left_right_names_disagree",
    "head_end_low_confidence",
    "head_end_from_round3_prior",
    "frame_check_disagrees",
    "head_end_disagrees",
)
TRUNK_BONES = ("b_t", "b_h", "b_a_1", "b_a_2", "b_a_3")


def specimen_status(tpl, part_status, start, flags=(), max_limb_share=0.25):
    """The geometric core's own specimen status with its reasons (the pipeline's status step defines the final one).
    'review' when: a TRUNK part (mesosoma, head, waist, gaster) is in review or has no usable joints; the root was
    placed by a fallback (fewer than three usable root-rigid joints); the joint product itself doubts its frame, head
    end or left / right names; or more than max_limb_share of the 34 limb segments (5 per leg, 2 per antenna) are not
    'ok' (in review or unregistered). A single flagged limb segment does not put the specimen in review (an earlier
    version put 13 of 21 development specimens in review, mostly from one noisy segment rule), and unregistered
    segments count (that version reported 'ok' for specimens with half their legs unregistered). Mandibles are reported per part only."""
    reasons = []
    ps = np.asarray(part_status).astype(str)
    tr_rv = [n for n in TRUNK_BONES if ps[tpl.idx[n]] == "review"]
    if tr_rv:
        reasons.append("trunk parts in review: " + ",".join(tr_rv))
    tr = [n for n in TRUNK_BONES if not start.support[tpl.idx[n]]]
    if tr:
        reasons.append("trunk parts without usable joints: " + ",".join(tr))
    if start.how[0] != "root_similarity":
        reasons.append("root placed by " + start.how[0])
    for f in flags:
        if str(f).startswith(FLAG_PREFIXES):
            reasons.append("joint product: " + str(f))
    segs = [f"l_{k}_{q}_{sd}" for k in "123" for q in ("co", "tr", "fe", "ti", "ta") for sd in "lr"] + [
        f"an_{k}_{sd}" for k in "12" for sd in "lr"
    ]
    bad = [n for n in segs if ps[tpl.idx[n]] != "ok"]
    if len(bad) > max_limb_share * len(segs):
        nrev = sum(1 for n in bad if ps[tpl.idx[n]] == "review")
        reasons.append(
            f"{len(bad)} of {len(segs)} limb segments not ok ({nrev} in review, {len(bad) - nrev} unregistered)"
        )
    return ("review" if reasons else "ok"), reasons


def register(spec, start, opts=None, device="cpu", log=None):
    """spec: geomreg.spec.Specimen; start: geomreg.native.Start (closed-form articulated start of the same specimen).
    Returns dict(V (nV,3) final vertices in the normalised frame, V_S (start), V_B (after girth), V_T, V_C, status,
    part_status, scan_code, info).

    On the CPU the optimised stages run with ONE torch thread (option cpu_threads, default 1): with two threads two
    runs of the same input differ (vertex distance median 0.02-0.09 um, single vertices up to 35 um), with one thread
    they are bit-identical; the thread count the caller had is restored."""
    n_cpu = int((opts or {}).get("cpu_threads", OPTS["cpu_threads"]))
    with single_thread(str(device) == "cpu" and n_cpu == 1):
        return _register(spec, start, opts, device, log)


def _register(spec, start, opts=None, device="cpu", log=None):
    o = dict(OPTS)
    o.update(opts or {})
    t0 = time.time()
    dev = device
    tpl = spec.tpl
    T = Tpl(tpl, dev)
    tt = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
    BL = spec.BL
    info = {"opts": {k: v for k, v in o.items()}, "BL_norm": BL}
    spec.per_bone = o["per_bone"]
    spec.support = np.asarray(start.support, bool).copy()
    # ---- scan tensors (all envelope vertices; s2t uses an area-weighted subsample per correspondence update)
    spec.Ps_t, spec.Ns_t, spec.As_t = tt(spec.Vs), tt(spec.Ns), tt(spec.As)
    rng = np.random.default_rng(o["seed"])
    sup_v_np = spec.support[tpl.dom]
    spec.sup_v = tt(sup_v_np, torch.bool)
    # ---- S: the closed-form start (native parameters), then the girth of the trunk parts from the scan
    Mnp = tpl.marker_regressor()
    VS = start.vertices()
    As_, bs_ = start.maps()
    VB = VS
    if o["girth"]:
        from scipy.spatial import cKDTree

        As_, bs_, ginfo = snap.girth_from_scan(
            tpl,
            VS,
            As_,
            bs_,
            spec.P,
            spec.w,
            spec.support,
            cKDTree(spec.Vs),
            head_front=getattr(spec, "head_front", None),
        )
        VB = snap.blend(tpl, tpl.V0, As_, bs_)
        info["girth"] = ginfo
    ref_now = np.einsum("jab,jb->ja", As_, start.ref0) + bs_  # where every bone's own map puts its reference point
    # ---- T: tube warp
    targets, tinfo = [], {}
    VT = VB
    if spec.limbs is not None and not o["gaster_tube"] and "G" in spec.limbs.limbs:  # ablation: gaster as a blob
        spec.limbs.limbs = {k: v for k, v in spec.limbs.limbs.items() if k != "G"}
    if o["tube"] and o["use_limbs"] and spec.limbs is not None:
        VT, targets, tinfo = tubes.tube_warp(
            spec, VB, per_bone=o["per_bone"], radial=o["tube_radial"], warp_coxa=o["tube_coxa"]
        )
    elif o["use_limbs"] and spec.limbs is not None:
        _, targets, tinfo = tubes.tube_warp(
            spec, VB, per_bone=o["per_bone"], radial=False, warp_coxa=o["tube_coxa"]
        )  # ring targets only
    info["tube"] = tinfo
    claim = None
    if o["use_limbs"] and spec.limbs is not None:
        ll, lb, _ = tubes.claims(spec, alpha=o["claim_alpha"], cos_min=o["claim_cos"])
        claim = (tt(ll, torch.long), tt(lb, torch.long))
        info["claimed_share"] = float((ll >= 0).mean())
    rl = local_radius(spec, VT, targets)
    spec.rloc_t = tt(rl)
    # ---- point terms: virtual markers (medial joints only) and limb rings
    if spec.kind == "medial":
        jm = spec.w > 0
        Mrows, Mtar, Mw = T.M[tt(jm, torch.bool)], tt(spec.P[jm]), tt(spec.w[jm])
    else:
        Mrows, Mtar, Mw = None, [], None
    if targets:
        Rrows, Rtar = tt(np.stack([x[0] for x in targets])), tt(np.stack([x[1] for x in targets]))
        Rw = torch.ones(len(targets), device=dev)
    else:
        Rrows, Rtar, Rw = None, [], None

    def point_terms(V):
        out = {}
        lm = loss_points(V, Mrows, Mtar, Mw, BL, o["sigma_joint"])
        if lm is not None:
            out["marker"] = o["w_marker"] * lm
        lr_ = loss_points(V, Rrows, Rtar, Rw, BL, o["sigma_joint"])
        if lr_ is not None:
            out["ring"] = o["w_ring"] * lr_
        return out

    def data_scale(V):
        el = (V[T.E[:, 0]] - V[T.E[:, 1]]).norm(dim=1)
        s = torch.zeros(T.nV, device=dev).index_add_(0, T.E[:, 0], el).index_add_(0, T.E[:, 1], el)
        c = (
            torch.zeros(T.nV, device=dev)
            .index_add_(0, T.E[:, 0], torch.ones_like(el))
            .index_add_(0, T.E[:, 1], torch.ones_like(el))
        )
        ell = s / c
        return ell, torch.clamp(ell, min=o["data_scale"][0] * BL, max=o["data_scale"][1] * BL)

    def update_corr(V, g_bl, code=None, relabel=True, kap=None):
        """kap = (kappa schedule value, boost tensor per unit): unit-relative gates with a per-unit boost that grows
        while less than half of a unit's supported vertices find scan surface, and decays when most do"""
        with torch.no_grad():
            N = vnormals(V, T.F)
            kappa = None if kap is None else kap[0] * kap[1]
            gate_v = _gate(spec, T, V, g_bl, o, dev, unit_t if kap is not None else None, kappa)
            if code is None or relabel:
                code = scan_codes(spec, T, V, N, o, spec.sup_v, dev, claim, gate_v)
            nS = len(spec.Ps_t)
            sub = None
            if o["n_scan"] and nS > o["n_scan"]:
                p = spec.As / spec.As.sum()
                sub = tt(rng.choice(nS, o["n_scan"], replace=False, p=p), torch.long)
            c = correspond(spec, T, V, N, code, gate_v, o, dev, sub)
            Ps, Ns = spec.Ps_t, spec.Ns_t
            floor = o["robust_min"] * BL
            r1 = ((V[c["i1"]] - Ps[c["s1"]]) * Ns[c["s1"]]).sum(1)
            c["w1"] = robust_w(r1, floor) if len(r1) else r1
            r2 = ((V[c["v2"]] - Ps[c["s2"]]) * Ns[c["s2"]]).sum(1)
            c["w2"] = robust_w(r2, floor) if len(r2) else r2
            c["a2"] = torch.ones_like(r2)
            c["med1"] = float(r1.abs().median() / BL) if len(r1) else float("nan")
            c["share1"] = float(c["has1"].float().sum() / spec.sup_v.float().sum().clamp(min=1))
            if kap is not None:
                nU = int(kap[1].shape[0])
                tot = torch.zeros(nU, device=dev).index_add_(0, unit_t, spec.sup_v.float())
                hit = torch.zeros(nU, device=dev).index_add_(0, unit_t, (c["has1"] & spec.sup_v).float())
                share = hit / tot.clamp(min=1)
                grow = (share < o["gate_grow_below"]) & (tot > 0)
                kap[1][grow] = torch.clamp(kap[1][grow] * 1.5, max=o["gate_boost_max"])
                shrink = share > 0.8
                kap[1][shrink] = torch.clamp(kap[1][shrink] / 1.2, min=1.0)
                c["boost_max"] = float(kap[1].max())
        return c, code

    # coarse units: bones that share one affine map (set below); until then every bone is its own unit
    unit_np = tpl.dom.copy()
    unit_t = tt(unit_np, torch.long)
    boost = torch.ones(tpl.nJ, device=dev)

    hist = []
    state = {"code": None, "corr": None, "ds": None}
    VTt = tt(VT)

    def run_opt(params, terms, refresh, n_outer, n_inner, lr, mode, stage):
        """n_outer correspondence updates (refresh(f), f = 0..1), n_inner optimiser steps each. mode 'adam' (one Adam
        over the whole stage) or 'lbfgs' (a fresh strong-Wolfe L-BFGS per update: the objective changes with the pairs)."""
        opt = torch.optim.Adam(params, lr=lr) if mode == "adam" else None
        for k in range(n_outer):
            refresh(k / max(n_outer - 1, 1), k)
            if mode == "adam":
                for _ in range(n_inner):
                    opt.zero_grad()
                    L = terms()
                    tot = sum(L.values())
                    tot.backward()
                    opt.step()
            else:
                lb = torch.optim.LBFGS(
                    params,
                    lr=1.0,
                    max_iter=n_inner,
                    history_size=12,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-9,
                    tolerance_change=1e-12,
                )
                last = {}

                def closure():
                    lb.zero_grad()
                    L = terms()
                    tot = sum(L.values())
                    tot.backward()
                    last["L"] = {q: float(v) for q, v in L.items()}
                    return tot

                lb.step(closure)
                L = last["L"]
            if k % max(n_outer // 6, 1) == 0 or k == n_outer - 1:
                Lf = {q: float(v) for q, v in L.items()}
                hist.append(
                    dict(
                        stage=stage,
                        k=k,
                        total=float(sum(Lf.values())),
                        med1=state["corr"]["med1"],
                        share1=state["corr"]["share1"],
                        **Lf,
                    )
                )

    # ---- C: coarse stage, one affine map per bone
    VC = VTt
    if o["coarse"]:
        nJ = T.nJ
        cnt = np.bincount(tpl.dom, minlength=nJ)
        eff = np.arange(nJ)
        merged = {tpl.idx["b_a_4"], tpl.idx["b_a_5"]} if o["c_merge_gaster"] else set()  # the gaster is one unit
        own = np.array(
            [(cnt[j] >= o["c_min_verts"] and spec.support[j] and j not in merged) or j == 0 for j in range(nJ)]
        )
        eff[:] = -1
        eff[own] = np.where(own)[0]
        # bones without a map of their own (unsupported, too few vertices) share the map of the nearest bone that has
        # one: breadth first over the kinematic tree, parents before children at equal distance
        attach = np.full(nJ, -1)  # joint through which a bone without a map hangs on the bone that has one
        frontier = list(np.where(own)[0])
        while frontier:
            nxt = []
            for m in frontier:
                for j in ([tpl.parents[m]] if tpl.parents[m] >= 0 else []) + list(tpl.children[m]):
                    if eff[j] < 0:
                        eff[j] = eff[m]
                        attach[j] = (j if tpl.parents[j] == m else m) if own[m] else attach[m]
                        nxt.append(j)
            frontier = nxt
        eff[eff < 0] = 0
        # bones WITHOUT usable joints (placed by the start's imputation) must not share a neighbour's affine map: an
        # anisotropic map extrapolated over a whole unsupported limb made needles in an earlier version. After the optimisation
        # they follow the bone they hang on rigidly (rotation = polar part of its map), pinned at the shared joint.
        rigid_j = [
            j
            for j in range(nJ)
            if o["rigid_unsupported"] and not own[j] and not spec.support[j] and attach[j] >= 0 and cnt[j] > 0
        ]
        eff_t = tt(eff, torch.long)
        unit_np = eff[tpl.dom]
        unit_t = tt(unit_np, torch.long)
        cen = torch.zeros(nJ, 3, device=dev)
        for j in range(nJ):
            m = tpl.dom == j
            if m.any():
                cen[j] = VTt[tt(m, torch.bool)].mean(0)
        Bm = torch.zeros(nJ, 3, 3, device=dev, requires_grad=True)
        tr = torch.zeros(nJ, 3, device=dev, requires_grad=True)  # translations in units of BL
        eye = torch.eye(3, device=dev)
        ridge_w = torch.where(
            tt(tpl.tube_bone, torch.bool),
            torch.tensor(o["c_ridge_tube"], device=dev),
            torch.tensor(o["c_ridge"], device=dev),
        )
        # joint continuity: ring vertices of joint j under the maps of j and of its parent
        cont = [(j, int(tpl.parents[j])) for j in range(1, nJ) if eff[j] == j and cnt[j] > 0]
        cj = tt([a_ for a_, _ in cont], torch.long)
        cp = eff_t[tt([b_ for _, b_ in cont], torch.long)]
        ring_w = T.M[cj]  # (n, nV)
        ring_nz = (ring_w > 0).any(0)
        ring_w, VTr = ring_w[:, ring_nz], VTt[ring_nz]
        # joint landmarks: the reference point of every usable joint, under the map of its own bone and of its parent,
        # stays on (goes to) its target
        lm_b, lm_p, lm_t = [], [], []
        if ref_now is not None and o["c_land"] > 0:
            for j in range(nJ):
                if spec.w[j] > 0:
                    for bmap in {int(eff[j]), int(eff[tpl.parents[j]]) if tpl.parents[j] >= 0 else int(eff[j])}:
                        lm_b.append(bmap), lm_p.append(ref_now[j]), lm_t.append(spec.P[j])
        lm_b, lm_p, lm_t = tt(lm_b, torch.long), tt(np.array(lm_p).reshape(-1, 3)), tt(np.array(lm_t).reshape(-1, 3))

        # shape coefficients (option c_betas): the model's shape space moves the NON-limb vertices under the bone maps
        # of the start (limb cross-sections stay the mean template's: fitted coefficients flatten thin limbs). z = whitened coefficients (betas = Lc z, Lc Lc^T = shape covariance), prior |z|^2.
        use_b = o["c_betas"] in (True, "auto") and tpl.nB > 0
        zb = torch.zeros(tpl.nB, device=dev, requires_grad=True)
        state["beta_on"] = False
        if use_b:
            WA = np.einsum("vj,jab->vab", tpl.W, As_)
            keepw = np.ones(tpl.nV) if o["c_beta_limbs"] else 1.0 - tpl.W[:, tpl.tube_bone].sum(1)
            Lc = np.linalg.cholesky(tpl.shape_cov + 1e-8 * np.eye(tpl.nB))
            Dz = tt(np.einsum("vab,vbk->vak", WA, tpl.shapedirs @ Lc) * keepw[:, None, None])  # (nV, 3, nB)

        def base_c():
            return VTt + torch.einsum("vak,k->va", Dz, zb) if (use_b and state["beta_on"]) else VTt

        def apply_c():
            A = eye + Bm[eff_t]  # (nJ, 3, 3)
            ce, tv = cen[eff_t], BL * tr[eff_t]
            cv = ce[T.kb]  # (nV, K, 3): only the K bones of a vertex
            X = torch.einsum("vkab,vkb->vka", A[T.kb], base_c()[:, None, :] - cv) + cv + tv[T.kb]
            return (T.kw[:, :, None] * X).sum(1), A

        def refresh_c(f, k):
            with torch.no_grad():
                Vc = apply_c()[0]
            g_bl = sched(o["c_gate"], f)
            kap = (sched(o["c_kappa"], f), boost)
            state["corr"], state["code"] = update_corr(
                Vc, g_bl, state["code"], relabel=(k % o["label_every"] == 0), kap=kap
            )
            state["ds"] = data_scale(Vc)[1]

        def terms_c():
            V, A = apply_c()
            L = loss_data(V, spec, state["corr"], state["ds"], o, p2p=o["c_p2p"])
            L = {q: o["w_data"] * v for q, v in L.items()}
            L.update(point_terms(V))
            L["ridge"] = (ridge_w * (Bm**2).sum(dim=(1, 2))).mean()
            if use_b and state["beta_on"]:
                L["beta"] = o["c_beta_prior"] * (zb**2).mean()
            if len(lm_b):
                x = torch.einsum("nab,nb->na", eye + Bm[lm_b], lm_p - cen[lm_b]) + cen[lm_b] + BL * tr[lm_b]
                d2 = ((x - lm_t) ** 2).sum(1) / BL**2
                L["land"] = o["c_land"] * (d2 / (d2 + o["sigma_joint"] ** 2)).mean()
            if len(cont):
                xj = (
                    torch.einsum("nab,nvb->nva", A[cj], VTr[None] - cen[cj][:, None])
                    + cen[cj][:, None]
                    + BL * tr[cj][:, None]
                )
                xp = (
                    torch.einsum("nab,nvb->nva", eye + Bm[cp], VTr[None] - cen[cp][:, None])
                    + cen[cp][:, None]
                    + BL * tr[cp][:, None]
                )
                L["cont"] = o["c_cont"] * (ring_w * ((xj - xp) ** 2).sum(-1)).sum(1).mean() / BL**2 * 1e2
            return L

        def measure_c():
            """median |point-to-plane residual| / BL and matched share of the supported template vertices at the final
            gates (correspondences recomputed, the adaptive gate boosts untouched)"""
            with torch.no_grad():
                Vc = apply_c()[0]
                c_, _ = update_corr(
                    Vc,
                    sched(o["c_gate"], 1.0),
                    state["code"],
                    relabel=True,
                    kap=(sched(o["c_kappa"], 1.0), boost.clone()),
                )
            return float(c_["med1"]), float(c_["share1"])

        run_opt([Bm, tr], terms_c, refresh_c, o["c_outer"], o["c_inner"], o["c_lr"], o["c_opt"], "C")
        if use_b:
            # second phase: bone maps + shape coefficients, from the first phase's solution, at the final gates.
            # 'auto': the shape space is kept only when it EXPLAINS the scan, i.e. the median residual of the coarse
            # stage drops below c_beta_accept x its value without the shape space (truth-free; in-model synthetic
            # shapes drop to 0.34-0.61, out-of-model ones to 0.46-0.78, real workers to 0.71-0.97)
            m0, s0 = measure_c()
            keep = (Bm.detach().clone(), tr.detach().clone())
            state["beta_on"] = True
            run_opt(
                [{"params": [Bm, tr]}, {"params": [zb], "lr": o["c_beta_lr"]}],
                terms_c,
                lambda f, k: refresh_c(1.0, k),
                o["c_beta_outer"],
                o["c_inner"],
                o["c_lr"],
                o["c_opt"],
                "Cb",
            )
            m1, s1 = measure_c()
            accept = True if o["c_betas"] is True else bool(m1 < o["c_beta_accept"] * m0)
            info["coarse_shape"] = dict(
                mode=str(o["c_betas"]),
                resid_without=m0,
                resid_with=m1,
                ratio=m1 / max(m0, 1e-30),
                share_without=s0,
                share_with=s1,
                accepted=accept,
                z_rms=float((zb.detach() ** 2).mean().sqrt()),
            )
            if not accept:
                with torch.no_grad():
                    Bm.copy_(keep[0])
                    tr.copy_(keep[1])
                    zb.zero_()
                state["beta_on"] = False
                refresh_c(1.0, 0)  # correspondences of the kept solution
        with torch.no_grad():
            A_all = (eye + Bm[eff_t]).detach().clone()  # (nJ, 3, 3) map of every bone: x -> A (x - c) + c + t
            c_all, t_all = cen[eff_t].clone(), (BL * tr[eff_t]).detach().clone()
            b_all = c_all + t_all - torch.einsum("jab,jb->ja", A_all, c_all)
            if rigid_j:
                q_all = tt(Mnp @ np.asarray(VT, np.float64))  # joint points on the mesh before the stage
                for j in rigid_j:
                    m = int(eff[j])
                    U_, _, Vh_ = torch.linalg.svd(A_all[m])
                    R_ = U_ @ Vh_
                    if torch.linalg.det(R_) < 0:
                        U_[:, -1] = -U_[:, -1]
                        R_ = U_ @ Vh_
                    q = q_all[int(attach[j])]
                    A_all[j] = R_
                    b_all[j] = (A_all[m] @ q + b_all[m]) - R_ @ q
            X = torch.einsum("vkab,vb->vka", A_all[T.kb], base_c()) + b_all[T.kb]
            VC = (T.kw[:, :, None] * X).sum(1)
            if use_b:
                info["coarse_betas"] = (Lc @ zb.detach().cpu().numpy().astype(np.float64)).tolist()
                info["coarse_z_rms"] = float((zb.detach() ** 2).mean().sqrt())
                info["shape_space_used"] = bool(state["beta_on"])
        info["rigid_unsupported_bones"] = [tpl.names[j] for j in rigid_j]
        info["coarse_B_max"] = float(Bm.detach().abs().amax())
        info["gate_boost_after_C"] = {tpl.names[j]: float(boost[j]) for j in range(nJ) if float(boost[j]) > 1.01}
    # ---- F: fine stage
    VF = VC
    if o["fine"]:
        with torch.no_grad():
            len_ref = (VC[T.E[:, 0]] - VC[T.E[:, 1]]).norm(dim=1).clamp(min=1e-9)
            dih_ref = dihedral(VC, T.E, T.EO)
            fn_ref, fa_ref = fnormals(VC, T.F)
            fa_ref = fa_ref.clamp(min=1e-14)
            N_ref = vnormals(VC, T.F)
            ell, ds = data_scale(VC)
            state["ds"] = ds
        U = torch.zeros(T.nV, 3, device=dev, requires_grad=True)
        pc = ell[:, None] if o["precond"] else torch.full((T.nV, 1), float(ell.median()), device=dev)

        def refresh_f(f, k):
            with torch.no_grad():
                Vc = VC + pc * U
            g_bl = sched(o["f_gate"], f)
            kap = (sched(o["f_kappa"], f), boost)
            state["corr"], state["code"] = update_corr(
                Vc, g_bl, state["code"], relabel=(k % o["label_every"] == 0), kap=kap
            )
            if o["w_repel"] > 0:
                with torch.no_grad():
                    state["Nrep"] = vnormals(Vc, T.F)
                    state["rep"] = repel_pairs(Vc, state["Nrep"], T, spec.sup_v, ell, o)

        def terms_f():
            V = VC + pc * U
            L = loss_data(V, spec, state["corr"], state["ds"], o, p2p=o["f_p2p"])
            L = {q: o["w_data"] * v for q, v in L.items()}
            L.update(point_terms(V))
            if o["w_edge"] > 0:
                L["edge"] = o["w_edge"] * loss_rel_edge(V, T, len_ref)
            if o["w_bend"] > 0:
                L["bend"] = o["w_bend"] * loss_bend(V, T, dih_ref)
            if o["w_tan"] > 0:
                L["tan"] = o["w_tan"] * loss_tan(V, VC, N_ref, ell)
            if o["w_fold"] > 0:
                L["fold"] = o["w_fold"] * loss_fold(V, T, fn_ref, fa_ref, o["fold_cos"], o["fold_area"])
            if o["w_repel"] > 0:
                L["repel"] = o["w_repel"] * loss_repel(V, state["Nrep"], state["rep"], ell, o["repel_margin"])
            return L

        run_opt([U], terms_f, refresh_f, o["f_outer"], o["f_inner"], o["f_lr"], o["f_opt"], "F")
        VF = (VC + pc * U).detach()
    code = state["code"]
    # ---- R: repair (closed form, options): crossings between parts not joined by a joint, local fold-overs
    if o["depenetrate"] or o["unfold"]:
        from . import repair

        Vr = VF.detach().cpu().numpy().astype(np.float64)
        if o["depenetrate"]:
            Vr, info["depenetrate"] = repair.depenetrate(tpl, Vr, sup_v_np)
        if o["unfold"]:
            Vr, info["unfold"] = repair.unfold(tpl, Vr, sup_v_np)
        info["repair_move_max_bl"] = float(np.linalg.norm(Vr - VF.detach().cpu().numpy(), axis=1).max() / BL)
        VF = tt(Vr)
    # ---- S: status
    with torch.no_grad():
        N = vnormals(VF, T.F)
        gate_v = _gate(spec, T, VF, o["obs_gate"], o, dev)
        code = scan_codes(spec, T, VF, N, o, spec.sup_v, dev, claim)
        c = correspond(spec, T, VF, N, code, gate_v, o, dev, None)
        status = np.where(sup_v_np, np.where(c["has1"].cpu().numpy(), 0, 1), 2).astype(np.int8)
        info["explained_scan_share"] = c["has2_share"]
        # how well this result explains the scan, with a FIXED gate (independent of the adaptive gates): mean capped
        # point-to-plane distance of the supported template vertices (no pair = the cap) plus the same for the scan
        # vertices that are not unresolved, in units of BL
        cap = o["measure_cap"] * BL
        gm = torch.full((T.nV,), cap, device=dev)
        cm = correspond(spec, T, VF, N, code, gm, o, dev, None)
        r1 = ((VF[cm["i1"]] - spec.Ps_t[cm["s1"]]) * spec.Ns_t[cm["s1"]]).sum(1).abs().clamp(max=cap)
        nsup = float(spec.sup_v.float().sum().clamp(min=1))
        e1 = float((r1.sum() + (nsup - len(r1)) * cap) / nsup / BL)
        r2 = ((VF[cm["v2"]] - spec.Ps_t[cm["s2"]]) * spec.Ns_t[cm["s2"]]).sum(1).abs().clamp(max=cap)
        res_ok = code != T.code_unres
        a_all = spec.As_t[res_ok].sum().clamp(min=1e-12)
        a_hit = spec.As_t[cm["s2"]]
        e2 = float(((r2 * a_hit).sum() + (a_all - a_hit.sum()).clamp(min=0) * cap) / a_all / BL)
        info["fit_measure"] = dict(
            t2s=e1, s2t=e2, total=e1 + e2, matched_t=float(len(r1) / nsup), matched_s=float(a_hit.sum() / a_all)
        )
    # ---- per-bone diagnostics (signals for a self-check; all relative to BL)
    with torch.no_grad():
        VFn, VCn = VF.cpu().numpy().astype(np.float64), VC.cpu().numpy().astype(np.float64)
        mk = Mnp @ VFn
        diag = dict(
            marker_dev=np.full(tpl.nJ, np.nan),
            obs_share=np.full(tpl.nJ, np.nan),
            shift_C=np.full(tpl.nJ, np.nan),
            shift_F=np.full(tpl.nJ, np.nan),
            resid=np.full(tpl.nJ, np.nan),
        )
        if spec.kind == "medial":
            okj = spec.w > 0
            diag["marker_dev"][okj] = np.linalg.norm(mk[okj] - spec.P[okj], axis=1) / BL
        r_all = np.full(tpl.nV, np.nan)
        i1 = c["i1"].cpu().numpy()
        r_all[i1] = ((VF[c["i1"]] - spec.Ps_t[c["s1"]]) * spec.Ns_t[c["s1"]]).sum(1).abs().cpu().numpy() / BL
        dC = np.linalg.norm(VCn - np.asarray(VT), axis=1) / BL
        dF = np.linalg.norm(VFn - VCn, axis=1) / BL
        for j in range(tpl.nJ):
            m = tpl.dom == j
            if m.sum() >= 3 and spec.support[j]:
                diag["obs_share"][j] = float((status[m] == 0).mean())
                diag["shift_C"][j], diag["shift_F"][j] = float(np.median(dC[m])), float(np.median(dF[m]))
                if np.isfinite(r_all[m]).any():
                    diag["resid"][j] = float(np.nanmedian(r_all[m]))
    part_status, review = part_status_from(tpl, spec.support, diag, tinfo, o, how=start.how)
    info["review"] = review
    info["hist"] = hist
    info["seconds"] = time.time() - t0
    info["status_counts"] = np.bincount(status, minlength=3).tolist()
    return dict(
        V=VF.cpu().numpy().astype(np.float64),
        V_S=np.asarray(VS, np.float64),
        V_B=np.asarray(VB, np.float64),
        V_T=np.asarray(VT, np.float64),
        V_C=VC.cpu().numpy().astype(np.float64),
        part_status=part_status.astype(str),
        bone_diag=diag,
        status=status,
        scan_code=code.cpu().numpy().astype(np.int16),
        info=info,
    )
