"""Truth-free checks of a registered mesh that the segment status does not make (model-grade conditions).

  tube_shape   per registered limb tube (scape, funiculus; any segment): the registered tube's medial curve from the
               TEMPLATE's own rings (vertices binned by their axial position on the template bone, so that every ring
               is the same vertex set in every specimen): arc / chord (tortuosity), total turning, net bend, and the
               "zigzag" = total turning - net bend (a smooth arc has 0), ring flatness. A crumpled or zigzag antenna
               has a large zigzag; a curled funiculus does not.
  head_roll    is the scan's head surface mirror-symmetric about the registered head's mid-plane? The scan points
               labelled head by the registration are reflected across planes that contain the registered head's long
               axis, rotated about that axis by theta; the symmetry cost is the median distance of a reflected point to
               the head points, in head widths. Reported: the cost at 0, the best theta and its cost. A head registered
               rolled about its long axis shows a clearly lower cost at a theta away from 0.
  trunk_cover  does the registered mesh reach the scan surface of each trunk part? The scan surface is sampled at its
               face centroids (weighted by face area); every sample goes to the nearest CLAIMED vertex of the mesh
               (every vertex that is not unregistered) and to that vertex's part group. Per group: the area share of
               its samples farther than a tolerance from that vertex = scan surface the part does not reach. A trunk
               part can lie on the scan (small mesh-to-scan distance) and still miss a third of its scan surface (a
               truncated or wedge-shaped gaster, a mesosoma whose dorsum is not reached); the status rule uses the
               mesosoma and gaster values. The head and waist values are reported only: scan surface of unregistered
               mandibles and antennae counts towards the head, and the template's waist does not reach high nodes.
               Scan surface of limbs that the registration left unregistered counts towards the mesosoma or the
               gaster in this value (the nearest registered vertex is a trunk vertex): on a specimen with several
               unregistered legs a trunk part can be called uncovered for its legs. A second value,
               `uncovered_off_limbs`, leaves out the samples inside the tube of a limb centreline of the limb product
               (the limb proper, up to where the centreline enters the trunk; its radius x LIMB_TUBE + LIMB_SLACK_BL
               of body length). It is REPORTED, not used by the rule: against exact truth (synthetic scans whose
               joints and limb products come from the joint method) the plain value flags 19 trunk parts of which 17
               are wrong by more than 2 % of body length, the second value only 10 (9 wrong), because wrong limb
               products hide trunk surface.
All work in the scan frame on the posed registration; no truth, no names.
"""

import numpy as np

NRING = 6  # template rings per tube (thin limbs have 60-70 vertices: 6 rings of 10 or more)
_S = {}


def _rings(t, bones, a0, b0):
    """template rings of the vertices dominated by `bones`, along the template axis a0 -> b0: list of index arrays."""
    ids = np.where(np.isin(t.dom, bones))[0]
    if len(ids) < 3 * NRING:
        return None
    u = (b0 - a0) / max(np.linalg.norm(b0 - a0), 1e-30)
    s = (t.V0[ids] - a0) @ u
    q = np.quantile(s, np.linspace(0, 1, NRING + 1))
    out = []
    for k in range(NRING):
        m = (s >= q[k]) & ((s <= q[k + 1]) if k == NRING - 1 else (s < q[k + 1]))
        out.append(ids[m])
    return out if all(len(r) >= 3 for r in out) else None


def tubes_static(t):
    """{tube name: dict(rings, r0)} for the scapes and funiculi and every leg segment (template rings and radii)."""
    key = id(t)
    if key in _S:
        return _S[key]
    J = t.J0 if hasattr(t, "J0") else t.Jreg @ t.V0
    idx = t.idx
    defs = {}
    for s in "lr":
        defs[f"scape_{s}"] = ([idx[f"an_1_{s}"]], idx[f"an_1_{s}"], idx[f"an_2_{s}"])
        defs[f"funiculus_{s}"] = ([idx[f"an_2_{s}"], idx[f"an_3_{s}"]], idx[f"an_2_{s}"], idx[f"an_3_{s}"])
        for k in "123":
            for a, b in (("tr", "fe"), ("fe", "ti"), ("ti", "ta")):
                defs[f"leg{k}{s}_{a}"] = ([idx[f"l_{k}_{a}_{s}"]], idx[f"l_{k}_{a}_{s}"], idx[f"l_{k}_{b}_{s}"])
            defs[f"leg{k}{s}_ta"] = (
                [idx[f"l_{k}_ta_{s}"], idx[f"l_{k}_pt_{s}"]],
                idx[f"l_{k}_ta_{s}"],
                idx[f"l_{k}_pt_{s}"],
            )
    out = {}
    for name, (bones, ja, jb) in defs.items():
        a0, b0 = J[ja], J[jb]
        if len(bones) > 1:  # the last tube of a chain (funiculus, rest of the tarsus): axis to its farthest vertex
            ids = np.where(np.isin(t.dom, bones))[0]
            if len(ids):
                b0 = t.V0[ids[np.argmax(np.linalg.norm(t.V0[ids] - a0, axis=1))]]
        if np.linalg.norm(b0 - a0) < 1e-9:
            continue
        rg = _rings(t, bones, a0, b0)
        if rg is None:
            continue
        C0 = np.stack([t.V0[r].mean(0) for r in rg])
        r0 = np.array([np.linalg.norm(t.V0[r] - c, axis=1).mean() for r, c in zip(rg, C0)])
        out[name] = dict(rings=rg, r0=r0, bones=bones)
    _S[key] = out
    return out


def _curve(V, rings):
    C = np.stack([V[r].mean(0) for r in rings])
    d = np.diff(C, axis=0)
    L = np.linalg.norm(d, axis=1)
    chord = np.linalg.norm(C[-1] - C[0])
    if chord < 1e-9 or (L < 1e-12).any():
        return None
    u = d / L[:, None]
    ang = np.degrees(np.arccos(np.clip((u[1:] * u[:-1]).sum(1), -1, 1)))
    net = float(np.degrees(np.arccos(np.clip(u[0] @ u[-1], -1, 1))))
    # roundness of the cross-sections: per ring, the vertices projected onto the plane across the local direction of the
    # curve; sqrt(smaller / larger eigenvalue) of their 2-D scatter (1 = round, 0 = a flat ribbon); median over rings
    rnd = []
    for k, r in enumerate(rings):
        tk = C[min(k + 1, len(C) - 1)] - C[max(k - 1, 0)]
        tk = tk / max(np.linalg.norm(tk), 1e-30)
        X = V[r] - C[k]
        X = X - np.outer(X @ tk, tk)
        ev = np.linalg.eigvalsh(X.T @ X)  # ascending; the smallest belongs to the removed direction
        rnd.append(float(np.sqrt(max(ev[1], 0.0) / max(ev[2], 1e-30))))
    return dict(
        tort=float(L.sum() / chord),
        turn_deg=float(ang.sum()),
        net_deg=net,
        zigzag_deg=float(ang.sum() - net),
        max_step_deg=float(ang.max()),
        round=float(np.median(rnd)),
    )


def tube_shape(t, V, names=None):
    """{tube: dict(tort, turn_deg, net_deg, zigzag_deg, zigzag_excess_deg, max_step_deg, round, round_rel)} for the
    registered mesh V (any frame). zigzag_excess = zigzag - the template's own zigzag of the same rings (ring centroids
    of a coarse mesh are not exactly on a line); round_rel = roundness of the cross-sections relative to the
    template's (a tube collapsed to a ribbon is far below 1)."""
    TS = tubes_static(t)
    res = {}
    for name, q in TS.items():
        if names is not None and name not in names:
            continue
        if "base" not in q:
            q["base"] = _curve(t.V0, q["rings"])
        c = _curve(V, q["rings"])
        if c is None or q["base"] is None:
            res[name] = dict(
                tort=None,
                turn_deg=None,
                net_deg=None,
                zigzag_deg=None,
                zigzag_excess_deg=None,
                max_step_deg=None,
                round=None,
                round_rel=None,
            )
            continue
        c["zigzag_excess_deg"] = c["zigzag_deg"] - q["base"]["zigzag_deg"]
        c["round_rel"] = c["round"] / max(q["base"]["round"], 1e-9)  # relative to the template's own cross-sections
        res[name] = c
    return res


TRUNK_GROUPS = {
    "head": ("b_h",),
    "mesosoma": ("b_t",),
    "waist": ("b_a_1", "b_a_2"),
    "gaster": ("b_a_3", "b_a_4", "b_a_5"),
}


LIMB_TUBE = 1.5  # a scan sample within LIMB_TUBE x the centreline's radius + LIMB_SLACK_BL x BL of a limb centreline
LIMB_SLACK_BL = 0.005  # is limb surface


def trunk_cover(t, V, vertex_status, scan_V, scan_F, BL, tol_bl=0.015, limbs=None):
    """see module docstring. vertex_status: the status step's codes (4 = unregistered); limbs: {limb: dict(X (n, 3)
    centreline tip first, r (n,) radius[, s (n,) arc from the tip, core_arc: arc at which the centreline enters the
    trunk])} of the limb product, or None. Returns {group: dict(uncovered (area of the group's far samples / area of
    all of the group's samples, or None), uncovered_off_limbs (the same without the far samples inside a limb tube;
    equal to uncovered without a limb product), area_share (of the whole scan surface), limb_share (share of the
    group's samples inside a limb tube))} + 'all'."""
    from scipy.spatial import cKDTree

    P = scan_V[scan_F].mean(1)
    a = 0.5 * np.linalg.norm(
        np.cross(scan_V[scan_F[:, 1]] - scan_V[scan_F[:, 0]], scan_V[scan_F[:, 2]] - scan_V[scan_F[:, 0]]), axis=1
    )
    tot = float(a.sum())
    idx = np.where(np.asarray(vertex_status) != 4)[0]
    out = {}
    if len(idx) == 0 or tot <= 0:
        for g in TRUNK_GROUPS:
            out[g] = dict(uncovered=None, uncovered_off_limbs=None, area_share=0.0, limb_share=0.0)
        out["all"] = dict(uncovered=None, uncovered_off_limbs=None, area_share=1.0, limb_share=0.0)
        return out
    d, i = cKDTree(V[idx]).query(P, workers=1)
    bone = t.dom[idx[i]]
    far = d > tol_bl * BL
    on_limb = np.zeros(len(P), bool)
    X, R = [], []
    for q in (limbs or {}).values():
        x, r = np.asarray(q["X"], float), np.nan_to_num(np.asarray(q["r"], float), nan=0.0)
        if len(x) < 2:
            continue
        # the limb proper only: a centreline is stored tip first and runs on INSIDE the trunk beyond its core arc
        # (there its radius is the body's, and its tube would hide trunk surface)
        if q.get("s") is not None and q.get("core_arc") is not None:
            k = np.asarray(q["s"], float) <= float(q["core_arc"])
            x, r = x[k], r[k]
        if len(x):
            X.append(x)
            R.append(np.maximum(r, 0.0))
    if X:
        R = np.concatenate(R)
        dc, ic = cKDTree(np.concatenate(X)).query(P, workers=1)
        on_limb = dc <= LIMB_TUBE * R[ic] + LIMB_SLACK_BL * BL

    def share(m):
        ar = float(a[m].sum())
        k = m & ~on_limb
        return dict(
            uncovered=float(a[m & far].sum() / ar) if ar > 0 else None,
            uncovered_off_limbs=float(a[k & far].sum() / ar) if ar > 0 else None,
            area_share=ar / tot,
            limb_share=float(a[m & on_limb].sum() / ar) if ar > 0 else 0.0,
        )

    for g, bones in TRUNK_GROUPS.items():
        out[g] = share(np.isin(bone, [t.idx[b] for b in bones]))
    out["all"] = share(np.ones(len(P), bool))
    return out


def head_frame(t, V):
    """registered head: mid-plane point c, lateral axis n (mirror vertex pairs), long axis a (neck -> head front,
    made perpendicular to n), width w (extent of the head vertices along n)."""
    ih = t.idx["b_h"]
    hv = np.where(t.dom == ih)[0]
    mv = t.mirror_v[hv]
    keep = np.isin(mv, hv) & (t.V0[hv, 1] > 1e-6)
    a, b = hv[keep], mv[keep]
    n = (V[a] - V[b]).mean(0)
    n /= max(np.linalg.norm(n), 1e-30)
    c = 0.5 * (V[a] + V[b]).mean(0)
    X = V[hv]
    # long axis: first principal direction of the head vertices perpendicular to n, signed from the neck side
    Y = X - X.mean(0)
    Y = Y - np.outer(Y @ n, n)
    ax = np.linalg.svd(Y, full_matrices=False)[2][0]
    neck = (t.Jreg @ V)[ih]
    if (X.mean(0) - neck) @ ax < 0:
        ax = -ax
    w = float(np.ptp(X @ n))
    return c, n, ax, w, hv


def head_roll(t, V, scan_V, BL, thetas=None, n_sub=3000, seed=0):
    """see module docstring. Returns dict(cost0, best_deg, cost_best, contrast, n_points, width_bl)."""
    from scipy.spatial import cKDTree

    c, n, ax, w, hv = head_frame(t, V)
    d, i = cKDTree(V).query(scan_V, workers=1)
    m = (t.dom[i] == t.idx["b_h"]) & (d <= 0.02 * BL)
    P = scan_V[m]
    if len(P) < 200 or w <= 0:
        return dict(cost0=None, best_deg=None, cost_best=None, contrast=None, n_points=int(len(P)), width_bl=w / BL)
    tree = cKDTree(P)
    rng = np.random.default_rng(seed)
    Q = P[rng.choice(len(P), min(n_sub, len(P)), replace=False)]
    thetas = np.arange(-60.0, 60.1, 3.0) if thetas is None else np.asarray(thetas, float)
    k = np.cross(ax, n)
    cost = []
    for th in thetas:
        r = np.radians(th)
        nt = np.cos(r) * n + np.sin(r) * k  # n rotated about the long axis
        R = Q - 2.0 * np.outer((Q - c) @ nt, nt)
        cost.append(float(np.median(tree.query(R, workers=1)[0]) / w))
    cost = np.array(cost)
    i0 = int(np.argmin(np.abs(thetas)))
    ib = int(np.argmin(cost))
    return dict(
        at_range_edge=bool(ib in (0, len(thetas) - 1)),  # the minimum lies at the end of the search: not an estimate
        cost0=float(cost[i0]),
        best_deg=float(thetas[ib]),
        cost_best=float(cost[ib]),
        contrast=float(cost[i0] / max(cost[ib], 1e-9)),
        n_points=int(len(P)),
        width_bl=float(w / BL),
        curve=[round(float(x), 5) for x in cost],
    )
