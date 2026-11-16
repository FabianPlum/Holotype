"""PATHS-A contact detection and cutting on voxel medial paths.

A geodesic medial path (tip -> core, voxel staircase) is tested at each of its radius minima ("necks"). Around a neck
centre c with local tube radius r_loc (largest distance-transform value on the path within a window), the solid is
split by a ball of radius rho = BALL_K * r_loc; the connected parts of the solid in the shell rho <= |x - c| <= 2 rho
are the BRANCHES leaving the neck, and a branch is LONG when it reaches the outer shell (short spurs, hairs and setae do
not). The path enters the neck through one branch (E) and leaves through another (X).

  articulation / plain tube   2 long branches (E and X), whatever the bend between them
  contact crossing            >= 3 long branches and the path does not go straight through: X is not the branch most
                              opposite to E (or E not the one most opposite to X), i.e. the path turns from one tube
                              into another one at a junction where its own tube continues; and the neck is narrow
                              against both sides (r_neck <= RATIO_MAX * min(r_E, r_X)).

A detected contact is CUT: the long branches are grouped into the two tubes ({E, the branch most opposite to E} and
{X, the branch most opposite to X}; any other branch joins the group of its most opposite branch), every solid voxel
of the ball + shell is assigned to a group by a watershed of -D from the branch voxels, and the voxels on the interface
between the two groups become impassable (a wall of 1-2 voxels). Paths are then recomputed.
"""

import numpy as np
from scipy import ndimage

BALL_K = 1.3  # ball radius / local tube radius
RATIO_MAX = 0.55  # neck radius / min(entry tube radius, exit tube radius) for a cut
MIN_RHO = 3.0  # voxels
WIN_K = 3.0  # window (in local radii) for the tube radius on either side of the neck


def _local_box(shape, c, R):
    lo = np.maximum(c - R, 0)
    hi = np.minimum(c + R + 1, np.array(shape))
    return lo, hi


def side_radius(Dp, i, n_side, forward):
    """Typical tube radius on one side of path index i: 80th percentile of D over n_side path voxels."""
    if forward:
        seg = Dp[i + 1 : i + 1 + n_side]
    else:
        seg = Dp[max(0, i - n_side) : i]
    if len(seg) == 0:
        return 0.0
    return float(np.percentile(seg, 80))


def junction(occ, D, p, i, Dp=None, ball_k=BALL_K, trunk=None):
    """Branch analysis of the solid around path voxel p[i]. Returns a dict (see module docstring) or None when the
    test is not applicable (ball reaches the trunk core, path does not leave the ball on both sides)."""
    if Dp is None:
        Dp = D[tuple(p.T)]
    c = p[i]
    r_neck = float(Dp[i])
    nwin = int(max(4, WIN_K * max(float(np.median(Dp[max(0, i - 8) : i + 9])), 2.0)))
    rE = side_radius(Dp, i, nwin, forward=False)
    rX = side_radius(Dp, i, nwin, forward=True)
    r_loc = max(rE, rX, r_neck, 1.0)
    rho = max(MIN_RHO, ball_k * r_loc)
    R2 = 2.0 * rho
    Ri = int(np.ceil(R2)) + 1
    lo, hi = _local_box(occ.shape, c, Ri)
    sub = occ[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    gi = np.indices(sub.shape).reshape(3, -1).T + lo
    dist = np.linalg.norm(gi - c, axis=1).reshape(sub.shape)
    res = {
        "i": int(i),
        "c": c.tolist(),
        "r_neck": r_neck,
        "r_E": rE,
        "r_X": rX,
        "rho": rho,
        "ratio": r_neck / max(1e-6, min(rE, rX)) if min(rE, rX) > 0 else 9.0,
    }
    if trunk is not None:
        tsub = trunk[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        if (tsub & (dist <= R2)).any():
            res["skip"] = "trunk_in_ball"
            return res
    outer = sub & (dist >= rho) & (dist <= R2)
    lab, n = ndimage.label(outer, structure=np.ones((3, 3, 3)))
    if n == 0:
        res["skip"] = "no_branches"
        return res
    reach = ndimage.maximum(dist, lab, index=np.arange(1, n + 1))
    long_ids = [k + 1 for k in range(n) if reach[k] >= R2 - 1.5]
    # entry / exit branch: the path's first voxel beyond rho on either side (still inside the shell)
    dp = np.linalg.norm(p - c, axis=1)

    def side_comp(rng):
        for j in rng:
            if dp[j] >= rho + 0.5:
                if dp[j] > R2:
                    return None
                q = p[j] - lo
                if np.all(q >= 0) and np.all(q < np.array(sub.shape)):
                    return int(lab[tuple(q)]) or None
                return None
        return None

    E = side_comp(range(i - 1, -1, -1))
    X = side_comp(range(i + 1, len(p)))
    res.update({"n_branches": int(n), "n_long": len(long_ids), "E": E, "X": X})
    if E is None or X is None:
        res["skip"] = "path_not_leaving_ball"
        return res
    if E == X:
        res["skip"] = "entry_equals_exit"
        return res
    shell = (dist >= rho) & (dist <= rho + 2.0)
    dirs = {}
    for k in set(long_ids) | {E, X}:
        m = (lab == k) & shell
        if not m.any():
            m = lab == k
        v = (np.argwhere(m) + lo).mean(0) - c
        dirs[k] = v / max(np.linalg.norm(v), 1e-9)
    res["turn_deg"] = float(np.degrees(np.arccos(np.clip(-dirs[E] @ dirs[X], -1, 1))))
    longs = [k for k in long_ids]
    for k in (E, X):
        if k not in longs:
            longs.append(k)

    def most_opposite(k):
        others = [m for m in longs if m != k]
        if not others:
            return None
        return min(others, key=lambda m: float(dirs[k] @ dirs[m]))

    oE, oX = most_opposite(E), most_opposite(X)
    res.update({"opp_E": oE, "opp_X": oX, "long_ids": longs, "straight": bool(oE == X and oX == E)})
    res["_box"] = (lo, hi, lab, dist, longs, dirs, E, X, oE, oX)
    return res


def is_contact(j, ratio_max=RATIO_MAX):
    if j is None or j.get("skip"):
        return False
    return bool(len(j["long_ids"]) >= 3 and j["n_long"] >= 3 and not j["straight"] and j["ratio"] <= ratio_max)


def cut_wall(occ, D, j, cut):
    """Add the interface wall between the two tube groups of junction j to the boolean array `cut` (in place)."""
    from skimage.segmentation import watershed

    lo, hi, lab, dist, longs, dirs, E, X, oE, oX = j["_box"]
    group = {E: 1, X: 2}
    if oE is not None and oE != X:
        group.setdefault(oE, 1)
    if oX is not None and oX != E:
        group.setdefault(oX, 2)
    for k in longs:
        if k in group:
            continue
        others = [m for m in longs if m != k]
        m = min(others, key=lambda q: float(dirs[k] @ dirs[q]))
        group[k] = group.get(m, 1)
    sub = occ[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    rho = j["rho"]
    R2 = 2.0 * rho
    mask = sub & (dist <= R2)
    markers = np.zeros(sub.shape, np.int32)
    for k, g in group.items():
        markers[(lab == k) & (dist >= rho + 1.0)] = g
    if not ((markers == 1).any() and (markers == 2).any()):
        return 0
    Dl = D[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    ws = watershed(-Dl, markers=markers, mask=mask, connectivity=1)
    a = ws == 1
    b = ws == 2
    wall = (a & ndimage.binary_dilation(b, structure=ndimage.generate_binary_structure(3, 1))) | (
        b & ndimage.binary_dilation(a, structure=ndimage.generate_binary_structure(3, 1))
    )
    wall &= dist < R2 - 0.5
    cs = cut[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    cs |= wall
    return int(wall.sum())


def path_minima(Dp, lo_i, hi_i, sigma=1.0):
    """Indices of local minima of the (lightly smoothed) radius profile Dp within [lo_i, hi_i)."""
    r = ndimage.gaussian_filter1d(Dp.astype(float), sigma, mode="nearest") if sigma > 0 else Dp.astype(float)
    out = []
    for k in range(max(1, lo_i), min(len(r) - 1, hi_i)):
        if r[k] <= r[k - 1] and r[k] < r[k + 1]:
            out.append(k)
        elif r[k] < r[k - 1] and r[k] <= r[k + 1]:
            out.append(k)
    # snap to the raw minimum within +-2
    snapped = []
    for k in out:
        a, b = max(0, k - 2), min(len(Dp), k + 3)
        snapped.append(int(a + np.argmin(Dp[a:b])))
    return sorted(set(snapped))


def _path_run(p, dp, i, step, R2):
    """Contiguous run of path indices from i (exclusive) in direction step while the path stays within R2 of c."""
    out = []
    j = i + step
    while 0 <= j < len(p) and dp[j] <= R2:
        out.append(j)
        j += step
    return out


def junction2(occ, D, p, i, Dp=None, ball_k=BALL_K, trunk=None, need_box=False):
    """Region-based neck analysis (v2). The solid within 2 rho of the neck is split by a watershed of -D seeded with the
    path's own voxels before (region A) and after (region B) the neck. n_A / n_B = long branches of each region (parts
    beyond rho that reach the outer shell): an articulation joins two tube ENDS (1 + 1), a contact enters or leaves a
    tube through its SIDE (a region with >= 2 long branches). `loop`: the two sides stay connected in the box when a
    small ball around the neck is removed (another connection exists nearby)."""
    from skimage.segmentation import watershed

    if Dp is None:
        Dp = D[tuple(p.T)]
    c = p[i]
    r_neck = float(Dp[i])
    nwin = int(max(4, WIN_K * max(float(np.median(Dp[max(0, i - 8) : i + 9])), 2.0)))
    rE = side_radius(Dp, i, nwin, forward=False)
    rX = side_radius(Dp, i, nwin, forward=True)
    r_loc = max(rE, rX, r_neck, 1.0)
    rho = max(MIN_RHO, ball_k * r_loc)
    R2 = 2.0 * rho
    res = {
        "i": int(i),
        "c": c.tolist(),
        "r_neck": r_neck,
        "r_E": rE,
        "r_X": rX,
        "rho": rho,
        "ratio": r_neck / max(1e-6, min(rE, rX)) if min(rE, rX) > 0 else 9.0,
    }
    Ri = int(np.ceil(R2)) + 1
    lo, hi = _local_box(occ.shape, c, Ri)
    sub = occ[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    gi = np.indices(sub.shape).reshape(3, -1).T + lo
    dist = np.linalg.norm(gi - c, axis=1).reshape(sub.shape)
    if trunk is not None:
        tsub = trunk[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        res["trunk_in_box"] = int((tsub & (dist <= R2)).sum())
    dp = np.linalg.norm(p - c, axis=1)
    bef = [j for j in _path_run(p, dp, i, -1, R2) if dp[j] >= r_neck + 1.5]
    aft = [j for j in _path_run(p, dp, i, +1, R2) if dp[j] >= r_neck + 1.5]
    if not bef or not aft:
        res["skip"] = "path_not_leaving_neck"
        return res
    mask = sub & (dist <= R2)
    markers = np.zeros(sub.shape, np.int32)
    qb = p[bef] - lo
    qa = p[aft] - lo
    markers[tuple(qb.T)] = 1
    markers[tuple(qa.T)] = 2
    markers[~mask] = 0
    Dl = D[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    ws = watershed(-Dl, markers=markers, mask=mask, connectivity=3)  # 26-connected, as the min-cost paths
    st = np.ones((3, 3, 3))
    nb = {}
    for g in (1, 2):
        lab, n = ndimage.label((ws == g) & (dist >= rho), structure=st)
        if n == 0:
            nb[g] = 0
            continue
        reach = ndimage.maximum(dist, lab, index=np.arange(1, n + 1))
        nb[g] = int(sum(r >= R2 - 1.5 for r in reach))
    # other solid in the box not reached from the path (separate parts)
    res["unassigned_long"] = 0
    rest = mask & (ws == 0)
    if rest.any():
        lab, n = ndimage.label(rest & (dist >= rho), structure=st)
        if n:
            reach = ndimage.maximum(dist, lab, index=np.arange(1, n + 1))
            res["unassigned_long"] = int(sum(r >= R2 - 1.5 for r in reach))
    # local loop: sides connected without the neck ball
    lab, n = ndimage.label(mask & (dist > r_neck + 1.5), structure=st)
    lb = set(lab[tuple(qb.T)].tolist()) - {0}
    la = set(lab[tuple(qa.T)].tolist()) - {0}
    res["loop"] = bool(lb & la)
    # interface size between the regions (voxels of A touching B)
    a, b = ws == 1, ws == 2
    s6 = ndimage.generate_binary_structure(3, 1)
    iface = a & ndimage.binary_dilation(b, structure=s6)
    res["iface_vox"] = int(iface.sum())
    res["iface_rel"] = float(iface.sum()) / max(1.0, np.pi * max(1.0, min(rE, rX)) ** 2)
    # turn: entry and exit directions at the neck
    ue = p[bef[-1]] - c
    ux = p[aft[-1]] - c
    ue = ue / max(np.linalg.norm(ue), 1e-9)
    ux = ux / max(np.linalg.norm(ux), 1e-9)
    res["turn_deg"] = float(np.degrees(np.arccos(np.clip(-ue @ ux, -1, 1))))
    res["n_A"], res["n_B"] = nb[1], nb[2]
    if need_box:
        res["_box2"] = (lo, hi, ws, dist)
    return res


def cut_wall2(j, cut):
    """Wall between the two watershed regions of junction2 (in place on `cut`); returns the number of wall voxels."""
    lo, hi, ws, dist = j["_box2"]
    s26 = np.ones((3, 3, 3), bool)  # 26-tight: the min-cost paths move to all 26 neighbours
    a, b = ws == 1, ws == 2
    wall = (a & ndimage.binary_dilation(b, structure=s26)) | (b & ndimage.binary_dilation(a, structure=s26))
    cs = cut[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    n_new = int((wall & ~cs).sum())
    cs |= wall
    return n_new


def continuation(D, p, i, side, r_tube, r_neck, win=(1.5, 4.0), probe=(0.0, 1.5), frac=0.5):
    """Does the tube on one side of the neck (side -1 = entry, +1 = exit) continue straight PAST the neck?
    The tube axis is a line fitted to the path voxels at arc [win[0], win[1]] x r_tube from the neck on that side; the
    solid is probed along that line from the neck's projection (+probe[0] r) to +probe[1] r beyond it. Returns
    (continues: bool, length beyond the neck's projection with D >= frac * r_tube, in units of r_tube)."""
    c = p[i].astype(float)
    if side < 0:
        seg = p[:i][::-1]
    else:
        seg = p[i + 1 :]
    if len(seg) < 3 or r_tube <= 0:
        return False, 0.0
    a = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.vstack([c, seg]), axis=0), axis=1))][1:]
    m = (a >= win[0] * r_tube + r_neck) & (a <= win[1] * r_tube + r_neck)
    if m.sum() < 3:
        m = (a >= 0.5 * r_tube + r_neck) & (a <= win[1] * r_tube + r_neck)
    if m.sum() < 3:
        return False, 0.0
    P = seg[m].astype(float)
    cen = P.mean(0)
    _, _, vt = np.linalg.svd(P - cen, full_matrices=False)
    t = vt[0]
    if (c - cen) @ t < 0:
        t = -t
    s_c = (c - cen) @ t
    shape = np.array(D.shape)
    step = 0.5
    s = s_c + probe[0] * r_tube
    L = 0.0
    while s <= s_c + probe[1] * r_tube + 1e-9:
        q = np.round(cen + s * t).astype(int)
        if np.any(q < 0) or np.any(q >= shape) or D[tuple(q)] < frac * r_tube:
            break
        L = s - s_c
        s += step
    return bool(L >= probe[1] * r_tube - step), float(L / r_tube)
