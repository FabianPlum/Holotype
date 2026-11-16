"""Bone snap: closed-form, bone-wise correction of the posed template so that the reference point of every usable joint
(virtual marker for medial joints, pivot for pivot joints) lies on its target. The skeleton stages leave a share of the
joints far from their targets (Geman-McClure joint term: no pull beyond ~2 % BL; measured: knee p90 10 % BL,
and 10 % of the tibia vertices > 6 % BL off on the synthetic set with ORACLE joints), and nothing after
them can bring a limb back from that distance.

Per bone j (the segment(s) leaving joint j), an affine map of the CURRENT posed mesh:
  chain bone, joint j and its one usable child c:  x -> P_j + R [g (x - r_j)_perp + lam (x - r_j)_par]
        r = current reference points, P = targets, R = smallest rotation (r_c - r_j) -> (P_c - P_j),
        lam = |P_c - P_j| / |r_c - r_j|; g = 1 for limb tubes (axial stretch only: limb girth is measured from the
        scan by the tube warp / the surface terms), g = lam for every other bone (waist, gaster, coxa, a head with
        one usable child): a blob that is twice as long starts twice as thick (isotropic); without that the
        gaster of one development specimen started as a sausage 3 x thinner than the scan's and never met its surface
  bone with several usable children (head, mesosoma): similarity (Umeyama) of {r_j, r_children} onto the targets
  bone without usable joints: continued from the nearest bone that has a map (parent or child side): the full map of
        a parent that was scaled as a whole (a head's mandibles grow with the head), else rigidly, pinned at the
        shared joint
blended with the skinning weights. Maps are absolute (targets are absolute), so no composition along the chain.
"""

import numpy as np

from . import geom


def umeyama(X, Y, with_scale=True):
    """similarity Y ~ s R X + t (rows = points)"""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / max((Xc**2).sum() / len(X), 1e-30)) if with_scale else 1.0
    return s, R, my - s * R @ mx


def bone_maps(tpl, ref, P, w, support, max_scale=(0.25, 4.0)):
    """A (nJ,3,3), b (nJ,3): x -> A x + b per bone; info: bones snapped / by continuation"""
    nJ = tpl.nJ
    A = np.tile(np.eye(3), (nJ, 1, 1))
    b = np.zeros((nJ, 3))
    done = np.zeros(nJ, bool)
    kind = [""] * nJ
    ok = w > 0
    for j in range(nJ):  # parents come before children in this model
        kids = [c for c in tpl.children[j] if ok[c]]
        pts = ([j] if ok[j] else []) + kids
        if len(pts) >= 3 and (j == 0 or support[j]):
            X, Y = ref[pts], P[pts]
            if np.linalg.matrix_rank(X - X.mean(0), tol=1e-9) >= 2:
                s, R, t = umeyama(X, Y)
                s = float(np.clip(s, *max_scale))
                A[j], b[j] = s * R, Y.mean(0) - s * R @ X.mean(0)
                done[j], kind[j] = True, "similarity"
                continue
        if ok[j] and len(kids) >= 1 and support[j]:
            c = kids[0]
            u, v = ref[c] - ref[j], P[c] - P[j]
            lu, lv = np.linalg.norm(u), np.linalg.norm(v)
            if lu > 1e-9 and lv > 1e-9:
                R = geom.rot_between(u / lu, v / lv)
                lam = float(np.clip(lv / lu, *max_scale))
                Ppar = np.outer(u / lu, u / lu)
                g = 1.0 if tpl.tube_bone[j] else lam
                A[j] = R @ (g * (np.eye(3) - Ppar) + lam * Ppar)
                b[j] = P[j] - A[j] @ ref[j]
                done[j], kind[j] = True, "chain"
                continue
        kind[j] = "pending"
    # bones without a map of their own continue RIGIDLY from the nearest bone that has one (parent or child side,
    # breadth first over the kinematic tree), pinned at the joint they share with it: x -> R_m (x - r_c) + m(r_c).
    # (Taking the parent's map instead tore the knee ring of a leg whose tibia was usable and whose femur was not.)
    res = done.copy()
    iso = np.array([done[j] and (kind[j] == "similarity" or not tpl.tube_bone[j]) for j in range(nJ)])
    frontier = [j for j in range(nJ) if done[j]]
    while frontier:
        nxt = []
        for m in frontier:
            nb = ([tpl.parents[m]] if tpl.parents[m] >= 0 else []) + list(tpl.children[m])
            for j in nb:
                if res[j]:
                    continue
                if tpl.parents[j] == m and iso[m]:
                    # child of a part that was scaled as a whole (mesosoma, head, gaster ...): it grows with it
                    A[j], b[j], iso[j] = A[m], b[m], True
                    res[j], kind[j] = True, "inherited"
                    nxt.append(j)
                    continue
                c = j if tpl.parents[j] == m else m  # the joint between bone j and bone m
                U, _, Vt = np.linalg.svd(A[m])
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                A[j] = R
                b[j] = (A[m] @ ref[c] + b[m]) - R @ ref[c]
                res[j], kind[j] = True, "continued"
                nxt.append(j)
        frontier = nxt
    return A, b, dict(snapped=int(done.sum()), kinds=kind)


def bone_snap(tpl, V, ref, P, w, support):
    A, b, info = bone_maps(tpl, ref, P, w, support)
    X = np.einsum("jab,vb->vja", A, V) + b[None]  # (nV, nJ, 3)
    out = (tpl.W[:, :, None] * X).sum(1)
    ok = w > 0
    info["ref_err_before"] = float(np.median(np.linalg.norm(ref[ok] - P[ok], axis=1))) if ok.any() else float("nan")
    return out, A, b, info


def girth_from_scan(tpl, V, A, b, P, w, support, scan_kd, head_front=None, n_t=15, clip=(0.4, 2.5)):
    """Closed-form girth of the non-limb parts from the scan itself (no optimisation, no labels).
    For a TRUNK part with an axis between two usable joint targets (waist segments, gaster base -> tip, mesosoma neck ->
    petiole, head neck -> head front when the joint file gives it) the INSCRIBED radius about that axis is measured on
    both sides: the largest distance from a point of the axis (middle 70 %) to the nearest scan vertex, and to the
    nearest vertex of the part in the snapped template V. Their ratio g scales the part's bone map across the axis,
    x -> a0 + [g (I - u u^T) + u u^T] (x - a0). The inscribed radius is taken at the thickest station, so a contact on
    one side of the part does not change it. Returns A, b (updated copies) and {bone name: g}."""
    from scipy.spatial import cKDTree

    A, b = A.copy(), b.copy()
    ok = w > 0
    idx = tpl.idx
    units = []  # (bones sharing the scale, axis start, axis end)
    for j in range(tpl.nJ):
        if not support[j] or not tpl.names[j].startswith("b_"):
            continue  # trunk parts only: coxae lie packed under the mesosoma and their axes end near its surface
        n = tpl.names[j]
        if n == "b_t":
            if ok[idx["b_h"]] and ok[idx["b_a_1"]]:
                units.append(([j], P[idx["b_h"]], P[idx["b_a_1"]]))
        elif n == "b_h":
            if head_front is not None and ok[j]:
                units.append(([j], P[j], np.asarray(head_front, float)))
        elif n == "b_a_3":
            tip = idx["b_a_5"]
            if ok[j] and ok[tip]:
                units.append(([j, idx["b_a_4"], tip], P[j], P[tip]))
        elif n in ("b_a_4", "b_a_5"):
            continue
        else:
            kids = [c for c in tpl.children[j] if ok[c]]
            if ok[j] and len(kids) == 1:
                units.append(([j], P[j], P[kids[0]]))
    out = {}
    ts = np.linspace(0.15, 0.85, n_t)
    for bones, a0, a1 in units:
        L = np.linalg.norm(a1 - a0)
        vm = np.isin(tpl.dom, bones)
        if L < 1e-9 or vm.sum() < 10:
            continue
        X = a0[None] + ts[:, None] * (a1 - a0)[None]
        r_scan = float(scan_kd.query(X)[0].max())
        r_tpl = float(cKDTree(V[vm]).query(X)[0].max())
        if r_tpl < 1e-9 or r_scan < 1e-9:
            continue
        g = float(np.clip(r_scan / r_tpl, *clip))
        u = (a1 - a0) / L
        G = g * (np.eye(3) - np.outer(u, u)) + np.outer(u, u)
        for j in bones:
            A[j], b[j] = G @ A[j], a0 + G @ (b[j] - a0)
        out[tpl.names[bones[0]]] = g
    return A, b, out


def blend(tpl, V, A, b):
    return (tpl.W[:, :, None] * (np.einsum("jab,vb->vja", A, V) + b[None])).sum(1)
