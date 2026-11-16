"""Mesh repair after the surface stage: closed-form / combinatorial passes on the registered
template mesh, no optimiser, deterministic.

  depenetrate   crossings between parts that are neither the same nor joined by a joint (two coxae, a coxa and the
                petiole, the two mandibles): where the scan shows one fused surface the registration has no data for
                the contact, and two parts placed independently can run through each other. Vertices of one part that
                lie behind the surface of the other are moved onto it (each side half the way, a small margin outside),
                the displacement is smoothed over the part's neighbourhood. Exact triangle-triangle tests decide where
                (same test as the shared benchmark's, re-implemented here), closest points on triangles say how far.
  unfold        faces whose normal points against the mean normal of their edge neighbours (local fold-overs) are
                relaxed: their vertices move tangentially towards the centroid of their one-ring; a step is kept only
                where it reduces the folded faces around the vertex.
Both passes move hidden or badly conditioned vertices by a fraction of an edge length; they are applied to registered
parts only and report what they did.
"""

import numpy as np
from scipy.spatial import cKDTree


def _face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-30)[:, None], 0.5 * a


def _seg_tri(P0, P1, A, B, C, eps=1e-12):
    """segments P0->P1 against triangles ABC (rows): True where the segment crosses the triangle"""
    d = P1 - P0
    e1, e2 = B - A, C - A
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > eps * np.maximum(
        np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d, axis=1), 1e-300
    )
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = P0 - A
    u = f * (s * h).sum(1)
    q = np.cross(s, e1)
    v = f * (d * q).sum(1)
    t = f * (e2 * q).sum(1)
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)


def part_tables(tpl):
    """parts = bones with leaf tips merged into their parent segment; adjacency = same part or joined by a joint"""
    if getattr(tpl, "_rep_parts", None) is not None:
        return tpl._rep_parts
    own = np.arange(tpl.nJ)
    cnt = np.bincount(tpl.dom, minlength=tpl.nJ)
    for j in range(tpl.nJ):
        if not tpl.children[j] and tpl.parents[j] >= 0 and cnt[j] < 30:  # leaf tips (pt, an_3, b_a_5)
            own[j] = tpl.parents[j]
    for j in range(tpl.nJ):
        while own[own[j]] != own[j]:
            own[j] = own[own[j]]
    adj = np.zeros((tpl.nJ, tpl.nJ), bool)
    for j in range(tpl.nJ):
        adj[own[j], own[j]] = True
        p = tpl.parents[j]
        if p >= 0:
            adj[own[j], own[p]] = adj[own[p], own[j]] = True
    pv = own[tpl.dom]
    fp = pv[tpl.F]
    face_part = np.where(fp[:, 1] == fp[:, 2], fp[:, 1], fp[:, 0])
    tpl._rep_parts = (own, adj, pv, face_part)
    return tpl._rep_parts


def crossing_pairs(tpl, V, face_ok):
    """(ia, ib) face index pairs of non-adjacent parts whose triangles intersect"""
    own, adj, pv, face_part = part_tables(tpl)
    F = tpl.F
    T = V[F]
    c = T.mean(1)
    rho = np.linalg.norm(T - c[:, None], axis=2).max(1)
    idx = np.where(face_ok & np.isfinite(c).all(1))[0]
    if len(idx) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    tree = cKDTree(c[idx])
    pr = tree.query_pairs(float(2 * np.percentile(rho[idx], 99.5)), output_type="ndarray")
    if len(pr) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    ia, ib = idx[pr[:, 0]], idx[pr[:, 1]]
    m = np.linalg.norm(c[ia] - c[ib], axis=1) <= rho[ia] + rho[ib]
    ia, ib = ia[m], ib[m]
    m = ~adj[face_part[ia], face_part[ib]]
    ia, ib = ia[m], ib[m]
    sh = (F[ia][:, :, None] == F[ib][:, None, :]).any((1, 2))
    ia, ib = ia[~sh], ib[~sh]
    if len(ia) == 0:
        return ia, ib
    TA, TB = T[ia], T[ib]
    hit = np.zeros(len(ia), bool)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        hit |= _seg_tri(TA[:, i], TA[:, j], TB[:, 0], TB[:, 1], TB[:, 2])
        hit |= _seg_tri(TB[:, i], TB[:, j], TA[:, 0], TA[:, 1], TA[:, 2])
    return ia[hit], ib[hit]


def _closest_on_tris(P, T):
    """closest point of each P[i] on triangle T[i] (rows). Ericson, Real-Time Collision Detection 5.1.5, vectorised."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac, ap = b - a, c - a, P - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = P - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = P - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    den = np.where(np.abs(va + vb + vc) > 1e-300, va + vb + vc, 1.0)
    v, w = vb / den, vc / den
    Q = a + ab * v[:, None] + ac * w[:, None]  # interior
    inside = np.ones(len(P), bool)

    def setm(m, val):
        nonlocal Q, inside
        Q = np.where(m[:, None], val, Q)
        inside &= ~m

    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    setm(m, a + ab * (d1 / np.where(np.abs(d1 - d3) > 1e-300, d1 - d3, 1.0))[:, None])
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    setm(m, a + ac * (d2 / np.where(np.abs(d2 - d6) > 1e-300, d2 - d6, 1.0))[:, None])
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    ww = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-300, (d4 - d3) + (d5 - d6), 1.0)
    setm(m, b + (c - b) * ww[:, None])
    setm((d1 <= 0) & (d2 <= 0), a)
    setm((d3 >= 0) & (d4 <= d3), b)
    setm((d6 >= 0) & (d5 <= d6), c)
    return Q, inside


def _edge_scale(tpl, V):
    """per-vertex length scale for the caps: the mean incident edge length of the mesh, but not more than 1.5 x the
    rest template's at the mesh's overall size (edges stretched across a tear must not loosen the caps)"""
    el = np.linalg.norm(V[tpl.E[:, 0]] - V[tpl.E[:, 1]], axis=1)
    el0 = np.linalg.norm(tpl.V0[tpl.E[:, 0]] - tpl.V0[tpl.E[:, 1]], axis=1)
    s = float(np.median(el / np.maximum(el0, 1e-30)))
    ell = np.zeros(tpl.nV)
    np.add.at(ell, tpl.E[:, 0], el)
    np.add.at(ell, tpl.E[:, 1], el)
    return np.minimum(ell / np.maximum(tpl.deg, 1), 1.5 * s * tpl.vscale0)


def depenetrate(tpl, V, reg_v, iters=6, margin=0.15, reach=2.0, smooth=2, k_near=8, step_cap=0.5, total_cap=1.5):
    """V (nV,3) -> (V', info). Only registered vertices move. margin / reach / caps in units of the local edge length
    of the INPUT mesh: a vertex moves at most step_cap per iteration and total_cap in all. The iterate with the fewest
    crossing pairs is returned; when no iterate has fewer than the input, the input is returned unchanged."""
    own, adj, pv, face_part = part_tables(tpl)
    F = tpl.F
    V0 = np.asarray(V, float).copy()
    V = V0.copy()
    ell0 = _edge_scale(tpl, V0)
    face_ok = reg_v[F].all(1)
    hist = []
    nbr = _vertex_neighbours(tpl)
    best = (None, None)
    for it in range(iters):
        ia, ib = crossing_pairs(tpl, V, face_ok)
        hist.append(int(len(ia)))
        if best[0] is None or len(ia) < best[0]:
            best = (len(ia), V.copy())
        if len(ia) == 0:
            break
        fn, fa = _face_normals(V, F)
        ell = ell0
        disp = np.zeros_like(V)
        wsum = np.zeros(tpl.nV)
        pa, pb = face_part[ia], face_part[ib]
        for a, b in sorted(set(zip(np.minimum(pa, pb).tolist(), np.maximum(pa, pb).tolist()))):
            sel = ((pa == a) & (pb == b)) | ((pa == b) & (pb == a))
            fa_x = np.unique(np.concatenate([ia[sel], ib[sel]]))
            for A, B in ((a, b), (b, a)):
                seed = np.unique(F[fa_x[face_part[fa_x] == A]].ravel())
                if len(seed) == 0:
                    continue
                va = np.unique(np.concatenate([seed] + [nbr[i] for i in seed]))  # crossing faces + one ring
                va = va[(pv[va] == A) & reg_v[va]]
                fb = np.where((face_part == B) & (fa > 0))[0]
                if len(va) == 0 or len(fb) == 0:
                    continue
                cb = V[F[fb]].mean(1)
                kk = min(k_near, len(fb))
                _, nn = cKDTree(cb).query(V[va], k=kk)
                nn = nn.reshape(len(va), kk)
                best_d = np.full(len(va), np.inf)
                best_q = np.zeros((len(va), 3))
                best_n = np.zeros((len(va), 3))
                best_in = np.zeros(len(va), bool)
                for k in range(kk):
                    tri = fb[nn[:, k]]
                    Q, ins = _closest_on_tris(V[va], V[F[tri]])
                    d = np.linalg.norm(V[va] - Q, axis=1)
                    up = d < best_d
                    best_d[up], best_q[up], best_n[up], best_in[up] = d[up], Q[up], fn[tri][up], ins[up]
                sd = ((V[va] - best_q) * best_n).sum(1)
                lim = np.maximum(ell[va], 1e-12)
                # behind B's surface, close to it, and the closest point is a real foot point on a face
                inside = (sd < margin * lim) & (best_d < reach * lim) & best_in
                if not inside.any():
                    continue
                tgt = best_q[inside] + margin * lim[inside, None] * best_n[inside]
                ids = va[inside]
                np.add.at(disp, ids, 0.5 * (tgt - V[ids]))
                np.add.at(wsum, ids, 1.0)
        mv = wsum > 0
        if not mv.any():
            break
        disp[mv] /= wsum[mv, None]
        has = mv.copy()
        for _ in range(smooth):  # spread the push ring by ring, halving (same part)
            acc = np.zeros_like(V)
            cnt = np.zeros(tpl.nV)
            for i in np.where(has)[0]:
                for j in nbr[i]:
                    if not has[j] and pv[j] == pv[i] and reg_v[j]:
                        acc[j] += disp[i]
                        cnt[j] += 1
            new = cnt > 0
            disp[new] = 0.5 * acc[new] / cnt[new, None]
            has |= new
        dn = np.linalg.norm(disp, axis=1)
        sc = np.minimum(1.0, step_cap * ell0 / np.maximum(dn, 1e-30))
        Vn = V + disp * sc[:, None]
        d = Vn - V0  # total cap relative to the input mesh
        dn = np.linalg.norm(d, axis=1)
        sc = np.minimum(1.0, total_cap * ell0 / np.maximum(dn, 1e-30))
        V = V0 + d * sc[:, None]
    ia, ib = crossing_pairs(tpl, V, face_ok)
    hist.append(int(len(ia)))
    if len(ia) < best[0]:
        best = (len(ia), V)
    out = best[1] if best[0] < hist[0] else V0
    mv = np.linalg.norm(out - V0, axis=1)
    return out, dict(
        crossing_pairs=hist,
        kept=int(best[0]) if best[0] < hist[0] else int(hist[0]),
        moved_vertices=int((mv > 0).sum()),
        max_move_in_edge_lengths=float((mv / np.maximum(ell0, 1e-30)).max()),
    )


def _vertex_neighbours(tpl):
    if getattr(tpl, "_nbr", None) is None:
        nb = [[] for _ in range(tpl.nV)]
        for a, b in tpl.E:
            nb[a].append(int(b))
            nb[b].append(int(a))
        tpl._nbr = [np.array(x, int) for x in nb]
    return tpl._nbr


def folded_faces(tpl, V):
    """faces whose normal points against the summed normals of their edge neighbours (the benchmark's 'folded')"""
    n = np.cross(V[tpl.F[:, 1]] - V[tpl.F[:, 0]], V[tpl.F[:, 2]] - V[tpl.F[:, 0]])
    nb = np.zeros_like(n)
    np.add.at(nb, tpl.EF[:, 0], n[tpl.EF[:, 1]])
    np.add.at(nb, tpl.EF[:, 1], n[tpl.EF[:, 0]])
    return (n * nb).sum(1) < 0


def unfold(tpl, V, reg_v, iters=8, step=0.5, total_cap=1.0):
    """TANGENTIAL relaxation of the vertices of folded faces: a step towards the centroid of the one-ring with its
    component along the vertex normal (area-weighted normals of the incident faces that are not folded) removed, so the
    surface is kept to first order. A vertex keeps its step only when the number of folded faces among its incident
    faces does not grow, and it never moves more than total_cap local edge lengths (of the input mesh) in all."""
    V0 = np.asarray(V, float).copy()
    V = V0.copy()
    ell0 = _edge_scale(tpl, V0)
    nbr = _vertex_neighbours(tpl)
    if getattr(tpl, "_vfaces", None) is None:
        vf = [[] for _ in range(tpl.nV)]
        for f, tri in enumerate(tpl.F):
            for v in tri:
                vf[v].append(f)
        tpl._vfaces = [np.array(x, int) for x in vf]
    vfaces = tpl._vfaces
    hist = []
    for it in range(iters):
        fo = folded_faces(tpl, V)
        hist.append(int(fo.sum()))
        ids = np.unique(tpl.F[fo].ravel())
        ids = ids[reg_v[ids]]
        if len(ids) == 0:
            break
        Vn = V.copy()
        nrm = np.cross(V[tpl.F[:, 1]] - V[tpl.F[:, 0]], V[tpl.F[:, 2]] - V[tpl.F[:, 0]])  # area-weighted face normals
        for i in ids:
            d = step * (V[nbr[i]].mean(0) - V[i])
            fi = vfaces[i]
            good = fi[~fo[fi]]
            n = nrm[good if len(good) else fi].sum(0)
            ln = np.linalg.norm(n)
            if ln > 1e-30:
                n = n / ln
                d = d - (d @ n) * n
            t = V[i] + d - V0[i]
            lt = np.linalg.norm(t)
            if lt > total_cap * ell0[i]:
                t = t * (total_cap * ell0[i] / lt)
            Vn[i] = V0[i] + t
        fo2 = folded_faces(tpl, Vn)
        for i in ids:  # reject steps that make it worse locally
            if fo2[vfaces[i]].sum() > fo[vfaces[i]].sum():
                Vn[i] = V[i]
        V = Vn
    hist.append(int(folded_faces(tpl, V).sum()))
    mv = np.linalg.norm(V - V0, axis=1)
    return V, dict(
        folded=hist,
        moved_vertices=int((mv > 0).sum()),
        max_move_in_edge_lengths=float((mv / np.maximum(ell0, 1e-30)).max()),
    )
