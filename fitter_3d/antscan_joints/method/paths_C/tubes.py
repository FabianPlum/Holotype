"""Tube decomposition of the non-trunk solid (approach C).

The appendage solid (occupancy minus the opened trunk core) is cut into tube segments by a watershed of the distance
transform; adjacent segments whose saddle is shallow and spans a full cross-section are merged (plateau noise along a
tube), so that segments end at constrictions (articulations) and at contacts. Every pair of adjacent segments shares an
interface ("neck"). For each side of each interface the local tube axis (principal direction of the segment's voxels
near the interface) gives
  align   |normal . axis|: ~1 for a cross-section of the tube (articulation / continuation), ~0 for a side contact
  offset  distance of the interface centroid from the axis line, in units of the local radius: ~0 when the axis runs
          through the neck (articulation), ~1 when the neck sits on the tube's flank (contact)
All arrays are voxel arrays of one grid (axis-aligned in some working frame; nothing here depends on the axes).
"""

import numpy as np
from scipy import ndimage as ndi


def watershed_segments(D, mask, smooth=0.7):
    from skimage.segmentation import watershed

    Ds = ndi.gaussian_filter(D, smooth) if smooth > 0 else D
    mx = ndi.maximum_filter(Ds, size=3)
    peaks = mask & (Ds >= mx) & (D >= 1.0)
    mk, nm = ndi.label(peaks, structure=np.ones((3, 3, 3)))
    lab = watershed(-Ds, mk, mask=mask)
    return lab.astype(np.int32), nm


def interfaces(lab, D):
    """6-neighbour faces between different positive labels. Returns dict with per-pair arrays: pairs (k, 2) a < b,
    n_faces, vec_area (sum of unit face vectors from a to b), centroid (voxel coords), neck (max of min(D_p, D_q))."""
    recs = []
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax] = slice(0, -1)
        sl1[ax] = slice(1, None)
        a = lab[tuple(sl0)]
        b = lab[tuple(sl1)]
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            continue
        p = np.argwhere(m)
        la, lb = a[m], b[m]
        da = D[tuple(sl0)][m]
        db = D[tuple(sl1)][m]
        sign = np.where(la < lb, 1.0, -1.0)
        mid = p.astype(np.float64)
        mid[:, ax] += 0.5
        recs.append((np.minimum(la, lb), np.maximum(la, lb), ax, sign, mid, np.minimum(da, db), p))
    if not recs:
        return None
    A = np.concatenate([r[0] for r in recs]).astype(np.int64)
    B = np.concatenate([r[1] for r in recs]).astype(np.int64)
    AX = np.concatenate([np.full(len(r[0]), r[2]) for r in recs])
    S = np.concatenate([r[3] for r in recs])
    MID = np.concatenate([r[4] for r in recs])
    NK = np.concatenate([r[5] for r in recs])
    P0 = np.concatenate([r[6] for r in recs])
    M = int(lab.max()) + 1
    key = A * M + B
    uk, inv = np.unique(key, return_inverse=True)
    k = len(uk)
    nf = np.bincount(inv, minlength=k)
    va = np.zeros((k, 3))
    for ax in range(3):
        sel = AX == ax
        va[:, ax] = np.bincount(inv[sel], weights=S[sel], minlength=k)
    cen = np.stack([np.bincount(inv, weights=MID[:, d], minlength=k) for d in range(3)], 1) / nf[:, None]
    neck = np.zeros(k)
    np.maximum.at(neck, inv, NK)
    pairs = np.stack([uk // M, uk % M], 1)
    return {
        "pairs": pairs,
        "n_faces": nf,
        "vec_area": va,
        "centroid": cen,
        "neck": neck,
        "face_inv": inv,
        "face_p0": P0,
        "face_ax": AX,
    }


def merge_shallow(lab, D, I, ratio=0.9, fill=0.5):
    """Union-find merge across interfaces with a shallow saddle (neck >= ratio x the smaller of the two segments'
    maximal radius) that span a full cross-section (faces >= fill x pi neck^2). Returns the relabelled array."""
    nl = int(lab.max()) + 1
    dmax = np.zeros(nl)
    np.maximum.at(dmax, lab.ravel(), D.ravel())
    parent = np.arange(nl)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    a, b = I["pairs"][:, 0], I["pairs"][:, 1]
    ok = (I["neck"] >= ratio * np.minimum(dmax[a], dmax[b])) & (I["n_faces"] >= fill * np.pi * I["neck"] ** 2)
    for k in np.argsort(-I["neck"]):
        if ok[k]:
            ra, rb = find(a[k]), find(b[k])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(x) for x in range(nl)])
    _, newid = np.unique(roots, return_inverse=True)
    return newid.astype(np.int32)[lab]


def segment_voxels(lab, skip=()):
    """Flat voxel indices per positive label (labels in skip are left empty): (flat index array, starts)."""
    flat = lab.ravel()
    nz = np.flatnonzero(flat > 0)
    if skip:
        nz = nz[~np.isin(flat[nz], list(skip))]
    order = nz[np.argsort(flat[nz], kind="stable")]
    counts = np.bincount(flat[nz], minlength=int(flat.max()) + 1)
    starts = np.r_[0, np.cumsum(counts)]
    return order, starts


def interface_features(lab, D, I, rho_fac=3.0, rho_min=4.0, skip=()):
    """Per interface and side: local radius, axis, elongation, align and offset (see module doc). Labels in skip (the
    trunk parts) get no features (nan)."""
    order, starts = segment_voxels(lab, skip)
    shape = lab.shape
    Df = D.ravel()
    k = len(I["pairs"])
    F = {
        "align": np.full((k, 2), np.nan),
        "offset": np.full((k, 2), np.nan),
        "elong": np.zeros((k, 2)),
        "rad": np.zeros((k, 2)),
        "axis": np.zeros((k, 2, 3)),
        "size": np.zeros((k, 2), int),
    }
    va = I["vec_area"]
    nrm = va / np.maximum(np.linalg.norm(va, axis=1, keepdims=True), 1e-9)
    F["normal"] = nrm
    by_seg = {}
    for q in range(k):
        for side in (0, 1):
            by_seg.setdefault(int(I["pairs"][q, side]), []).append((q, side))
    for l, lst in by_seg.items():
        if l in skip:
            continue
        ii = order[starts[l] : starts[l + 1]]
        P = np.stack(np.unravel_index(ii, shape), 1).astype(np.float64)
        dP = Df[ii]
        for q, side in lst:
            c = I["centroid"][q]
            sel0 = np.all(np.abs(P - c) <= max(rho_min, rho_fac * (3.0 * I["neck"][q] + 3)) + 1, axis=1)
            dd = np.full(len(P), np.inf)
            dd[sel0] = np.linalg.norm(P[sel0] - c, axis=1)
            near = dd <= max(rho_min, 2.0 * I["neck"][q] + 2)
            r = float(dP[near].max()) if near.any() else float(I["neck"][q])
            rho = max(rho_min, rho_fac * r)
            sel = dd <= rho
            F["rad"][q, side] = r
            F["size"][q, side] = len(P)
            if sel.sum() < 6:
                continue
            Q = P[sel]
            m = Q.mean(0)
            w, U = np.linalg.eigh((Q - m).T @ (Q - m) / len(Q))
            ax = U[:, 2]
            F["axis"][q, side] = ax
            F["elong"][q, side] = float(np.sqrt(w[2] / max(w[1], 1e-9)))
            F["align"][q, side] = abs(float(ax @ nrm[q]))
            v = c - m
            F["offset"][q, side] = float(np.linalg.norm(v - (v @ ax) * ax)) / max(r, 1.0)
    return F
