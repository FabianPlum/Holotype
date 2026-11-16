"""Fit of an articulated limb chain to a centreline polyline with ABSOLUTE expected segment lengths.

What the scan-only joint method cannot do: its along-limb placement assumes that a named centreline is the whole
limb and places joints by proportions of the path. Here the chain is the model's: rigid straight segments (coxa,
femur, tibia, basitarsus, scape) of expected length (the bilateral partner's, else the population prediction from the
specimen's own supported segments), attached at the socket, joints at articulation features (a fold or bend of the
centreline, a radius minimum, the tip). A dynamic programme over the polyline nodes places the joints the product
lacks; joints the product has with confidence >= 0.3 are fixed anchors and are never moved. Trailing joints stay
unplaced when the polyline is too short for them (a truncated or folded path), so a path that ends at the knee is
not mistaken for a whole leg.
"""

import numpy as np

STEP_BL = 0.0025  # node spacing, share of BL
LAM_FEAT = 2.0  # weight of the articulation feature (in units of 0.5 z^2 of the length term)
LAM_STRAIGHT = 4.0  # weight of the straightness of a rigid segment: (arc / chord - RIGID_SLACK) / 0.1
RIGID_SLACK = 1.06
Z_FIT = 3.0  # a placed joint is kept only when its segments are within this many sd of the expectation
ANCHOR_OFF_BL = 0.02  # an anchor farther than this from the polyline is "off the polyline" (no order constraint)
SOCKET_SD_BL = 0.015  # sd of the socket prior on the root joint, share of BL
FEAT_ON = 0.5  # feature score from which a joint counts as "at an articulation feature"
FLEX_SD = 1.0  # sd of the log chord of a flexible segment (tarsus rest, funiculus): a curled tarsus or
# funiculus has a shorter chord than a straight one, on either side of the body, and the
# product's tip can be a truncated path's end: the chord only breaks ties in the fit


def resample(X, step):
    X = np.asarray(X, float)
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    keep = np.concatenate([[True], d > 1e-9])
    X = X[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    if s[-1] < step:
        return X[[0, -1]], np.array([0.0, s[-1]])
    a = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - a[-1] > 0.3 * step:
        a = np.append(a, s[-1])
    else:
        a[-1] = s[-1]
    Q = np.stack([np.interp(a, s, X[:, k]) for k in range(3)], 1)
    return Q, a


def smooth(Q, w=2):
    if len(Q) < 2 * w + 3:
        return Q
    k = np.ones(2 * w + 1) / (2 * w + 1)
    S = np.stack([np.convolve(np.pad(Q[:, c], w, mode="edge"), k, mode="valid") for c in range(3)], 1)
    S[0], S[-1] = Q[0], Q[-1]
    return S


def features(Q, a, r, BL):
    """bend angle (deg) and radius-minimum score in [0, 1] per node."""
    m = len(Q)
    bend = np.zeros(m)
    win = np.maximum(0.012 * BL, 1.5 * r)
    for i in range(m):
        lo = np.searchsorted(a, a[i] - win[i])
        hi = min(np.searchsorted(a, a[i] + win[i]), m - 1)
        if lo >= i or hi <= i:
            continue
        u, v = Q[i] - Q[lo], Q[hi] - Q[i]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        bend[i] = np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1)))
    rs = np.convolve(np.pad(r, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    rmin = np.zeros(m)
    w2 = 0.03 * BL
    for i in range(1, m - 1):
        lo = np.searchsorted(a, a[i] - w2)
        hi = min(np.searchsorted(a, a[i] + w2), m - 1)
        if lo >= i or hi <= i:
            continue
        if rs[i] <= rs[max(i - 1, 0)] and rs[i] <= rs[min(i + 1, m - 1)]:
            depth = (min(rs[lo:i].max(), rs[i + 1 : hi + 1].max()) - rs[i]) / max(rs[i], 1e-9)
            rmin[i] = float(np.clip(depth / 0.25, 0, 1))
    # spread a minimum over +-1 node so that the DP can land next to it
    rmin = np.maximum(rmin, 0.8 * np.maximum(np.roll(rmin, 1), np.roll(rmin, -1)))
    rmin[0] = rmin[-1] = 0.0
    bscore = np.clip((bend - 25.0) / 35.0, 0, 1)
    return bend, rmin, np.maximum(bscore, rmin)


def fit_chain(Q, a, r, BL, joints, segs, socket=None, tip_is_end=True):
    """Place the free joints of one chain on a polyline.

    Q, a, r   polyline nodes ROOT FIRST, arc (um), radius (um)
    joints    list (proximal first) of dict(name, fixed: xyz or None)
    segs      list (len(joints) - 1) of dict(mu (log um) or None, sd, rigid bool)
    socket    xyz of the predicted socket (prior on a free root joint) or None
    tip_is_end  the last node is a real limb tip (the chain's tip joint may only sit there)
    Returns list of per-joint dict (None for unplaced / fixed): node, xyz, arc, bend_deg, feat, z_prox, z_dist,
    at_feature, estimated.
    """
    n, m = len(joints), len(Q)
    bend, rmin, feat = features(Q, a, r, BL)
    INF = 1e18
    fixed_node = [None] * n
    for k, j in enumerate(joints):
        if j["fixed"] is not None:
            d = np.linalg.norm(Q - np.asarray(j["fixed"])[None], axis=1)
            i = int(np.argmin(d))
            fixed_node[k] = i if d[i] <= ANCHOR_OFF_BL * BL else -1
    free = [k for k in range(n) if joints[k]["fixed"] is None]
    if not free:
        return [None] * n

    def run(K_end):
        """DP over joints 0..K_end (inclusive). Returns (total cost, node per joint)."""
        cost_prev, back = None, []
        states_prev = None
        for k in range(K_end + 1):
            j = joints[k]
            if j["fixed"] is not None:
                pos = np.asarray(j["fixed"], float)[None]
                nodes = np.array([fixed_node[k]])
                un = np.zeros(1)
            else:
                pos, nodes = Q, np.arange(m)
                un = -LAM_FEAT * feat.copy()
                last_joint = k == n - 1
                if last_joint:  # the tip joint sits at the end of a real tip path
                    un = np.full(m, INF)
                    if tip_is_end:
                        un[m - 1] = -LAM_FEAT
                if k == 0 and socket is not None:
                    un = un + 0.5 * (np.linalg.norm(Q - np.asarray(socket)[None], axis=1) / (SOCKET_SD_BL * BL)) ** 2
            if cost_prev is None:
                cost, bk = un.copy(), np.full(len(un), -1)
            else:
                sg = segs[k - 1]
                ppos, pnodes = states_prev
                ch = np.linalg.norm(pos[None, :, :] - ppos[:, None, :], axis=2)  # (prev, cur)
                T = np.zeros_like(ch)
                if sg["mu"] is not None:
                    dv = np.log(np.maximum(ch, 1e-6)) - sg["mu"]
                    if sg["rigid"]:
                        T += 0.5 * (dv / sg["sd"]) ** 2
                    else:  # a flexible segment's chord varies with its curl
                        T += 0.5 * (dv / max(sg["sd"], FLEX_SD)) ** 2
                on_p, on_c = pnodes >= 0, nodes >= 0
                both = on_p[:, None] & on_c[None, :]
                arc = a[np.maximum(nodes, 0)][None, :] - a[np.maximum(pnodes, 0)][:, None]
                T = np.where(both & (arc <= 0.5 * STEP_BL * BL), INF, T)  # order along the polyline
                if sg["rigid"]:
                    ratio = np.where(both, arc / np.maximum(ch, 1e-6), 1.0)
                    T += np.where(both, LAM_STRAIGHT * np.maximum(ratio - RIGID_SLACK, 0) / 0.1, 0.0)
                tot = cost_prev[:, None] + T
                bk = np.argmin(tot, axis=0)
                cost = tot[bk, np.arange(tot.shape[1])] + un
            back.append(bk)
            cost_prev, states_prev = cost, (pos, nodes)
        i = int(np.argmin(cost_prev))
        total = float(cost_prev[i])
        path = [0] * (K_end + 1)
        for k in range(K_end, -1, -1):
            path[k] = i
            i = int(back[k][i]) if k > 0 else -1
        return total, path

    def describe(K_end, path):
        X = []
        for k in range(K_end + 1):
            X.append(np.asarray(joints[k]["fixed"], float) if joints[k]["fixed"] is not None else Q[path[k]])
        out = [None] * n
        zs = []
        for k in range(K_end):
            sg = segs[k]
            ch = float(np.linalg.norm(X[k + 1] - X[k]))
            if sg["mu"] is None:
                zs.append(None)
            else:
                dv = float(np.log(max(ch, 1e-6)) - sg["mu"])
                zs.append(abs(dv) / (sg["sd"] if sg["rigid"] else max(sg["sd"], FLEX_SD)))
        for k in range(K_end + 1):
            if joints[k]["fixed"] is not None:
                continue
            i = path[k]
            zp = zs[k - 1] if k > 0 else None
            zd = zs[k] if k < K_end else None
            is_tip = k == n - 1 and i == m - 1 and tip_is_end
            out[k] = dict(
                node=int(i),
                xyz=Q[i].copy(),
                arc=float(a[i]),
                bend_deg=float(bend[i]),
                rmin=float(rmin[i]),
                feat=float(1.0 if is_tip else feat[i]),
                z_prox=zp,
                z_dist=zd,
                at_feature=bool(is_tip or feat[i] >= FEAT_ON),
                is_tip=bool(is_tip),
            )
        return out, zs

    best = [None] * n
    last_free = max(free)
    # When a usable joint lies distal of the last free joint, the fit runs up to it (the free joints are constrained
    # from both sides). Otherwise the chain may end early: the longest prefix whose placed segments are all plausible.
    k_stop = next((k for k in range(last_free + 1, n) if joints[k]["fixed"] is not None), None)
    ends = [k_stop] if k_stop is not None else list(range(n - 1, min(free) - 1, -1))
    for K_end in ends:
        total, path = run(K_end)
        if total >= INF / 2:
            continue
        out, zs = describe(K_end, path)
        ok = all(
            z is None or z <= Z_FIT
            for kz, z in enumerate(zs)
            if joints[kz]["fixed"] is None or joints[kz + 1]["fixed"] is None
        )
        if ok:
            best = out
            for k in range(n):
                if best[k] is not None:
                    best[k]["prefix_end"] = K_end
                    best[k]["cost"] = total
            break
    return best


def extend_tail(Q, r_tip, pool, BL, radius_fn):
    """Continuations of a root-first polyline beyond its tip, from the path stages' candidate polylines (the pieces the
    joint method named otherwise or dropped): a candidate that passes the tip and leads away from the polyline over at
    least 3 % BL of limb-like (thin) material. Returns a list of (extended polyline, source tag), longest first."""
    from scipy.spatial import cKDTree

    tip = Q[-1]
    kd = cKDTree(Q)
    eps_ext = max(0.015 * BL, 3.0 * r_tip)
    eps_same = max(0.0075 * BL, 1.5 * r_tip)
    out = []
    for c in pool:
        Y = c["X"]
        d = np.linalg.norm(Y - tip[None], axis=1)
        mi = int(np.argmin(d))
        if d[mi] > eps_ext:
            continue
        for piece in (Y[mi::-1], Y[mi:]):
            if len(piece) < 5:
                continue
            dq = kd.query(piece)[0]
            seg = np.linalg.norm(np.diff(piece, axis=0), axis=1)
            free = dq[1:] > eps_same
            free_len = float(seg[free].sum())
            if free_len < 0.03 * BL:
                continue
            first = int(np.argmax(free)) if free.any() else len(piece)
            back = float(seg[:first].sum())  # distance travelled along the polyline before leaving it
            if back > 0.03 * BL:
                continue
            Z = piece[first:]
            rz = radius_fn(Z[::3])
            if np.median(rz) > max(2.5 * r_tip, 0.012 * BL):  # not limb-like (runs inside a body)
                continue
            out.append((np.concatenate([Q, Z]), f"{c['src']}:{c['key']}", free_len, Z[-1]))
    out.sort(key=lambda t: -t[2])
    keep = []
    for X, tag, fl, tp in out:  # one continuation per physical tail (tip position)
        if all(np.linalg.norm(tp - k[3]) > 0.015 * BL for k in keep):
            keep.append((X, tag, fl, tp))
    return [(k[0], k[1], k[2]) for k in keep[:3]]
