"""Morphometrics as ONE module: the 24 measures of the shape-model work (the list is `columns()`; each is defined
where it is computed in `measure`, and in the documentation page) on the registered mesh, the status of rs.status and
the UC1 rest mesh. One row per specimen, micrometres (um, um2, um3). Measures and helpers come from the earlier
skeleton registration's morphometrics and were changed only where the definitions say otherwise (CHANGES below).

    from fitter_3d.antscan_registration.parts.rs.morpho import measure
    row, flags = measure(V_scan_um, part_code, obs, uc1, waist_segments, specimen_class)

Flags per measure (same keys as row): 0 valid (every required part measurable, codes 0-2; observed-share rule
met), 1 flagged (a required part in review, codes 3 / 7; or observed share 0.2-0.5 for surface measures; or a marker
ring / part holds UC1 fallback vertices), 2 NaN (a required part unregistered, codes 4-6; observed share < 0.2;
< 6 vertices; section not closed; for rest-mesh volumes / areas: any UC1 fallback vertex in the part), 3 the
specimen is do_not_use (value kept, never to be used). Values are computed whenever the geometry exists; a consumer
selects by flag (rs.run writes a measurable-only csv with every flag != 0 set to NaN).

CHANGES against the table / SKEL-REG (each one stated):
  - 'ok' / 'review' / 'failed|unregistered' of the table = codes 0-2 / 3, 7 / 4-6 of rs.status.
  - observed share: share of the part's vertices within 0.5 % BL of the scan surface (point-to-surface, dense
    scan samples), the route-agnostic replacement of SKEL-REG's vertex_status 0 (which only SKEL-REG writes).
  - segment cross-section (PROPOSED in the table): implemented on the UC1 rest mesh x s, plane perpendicular to the
    rest marker axis at 50 % (also 25 / 75 % in the json detail only); NaN when the cut of the segment's own faces is
    not one closed loop.
  - leg tarsus_rest and funiculus need the leaf part (pt / an_3), which rs.status gives the code of its segment.
  - specimen class do_not_use gives flag 3 on every measure (the table: NaN when the specimen failed).
"""

import numpy as np
from scipy.spatial import cKDTree

from . import VERSION  # noqa: F401

LEG_SEG = [
    ("coxa", "co", "tr"),
    ("femur", "tr", "fe"),
    ("tibia", "fe", "ti"),
    ("basitarsus", "ti", "ta"),
    ("tarsus_rest", "ta", "pt"),
]
ANT_SEG = [("scape", "an_1", "an_2"), ("funiculus", "an_2", "an_3")]
MEAS = (0, 1, 2)
FLAGGED = (3, 7)
_ST = {}


def _static():
    if not _ST:
        from fitter_3d.antscan_registration.parts.skelreg import charts
        from fitter_3d.antscan_registration.parts.skelreg.model import template

        t = template("medial")
        V0 = t.V0
        d, mir = cKDTree(V0).query(V0 * np.array([1, -1, 1]))
        _ST.update(
            t=t,
            mirror=mir,
            pair_ok=(d < 0.01 * t.rest_body_length) & (V0[:, 1] > 1e-6),
            M=t.marker_regressor(),
            owners=charts.owners(t),
        )
        M = _ST["M"]
        _ST["ring"] = [np.where(M[j] != 0)[0] for j in range(t.nJ)]
    return _ST


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _lateral(V, ids, st):
    m = np.zeros(len(V), bool)
    m[ids] = True
    sel = np.where(m & st["pair_ok"])[0]
    if len(sel) < 3:
        return None
    return _unit((V[sel] - V[st["mirror"][sel]]).mean(0))


def _extent(X, u):
    p = X @ u
    return float(p.max() - p.min())


def _vol_area(V, F):
    if not len(F):
        return np.nan, np.nan
    tri = V[F]
    c = tri.reshape(-1, 3).mean(0)
    a, b, d = tri[:, 0] - c, tri[:, 1] - c, tri[:, 2] - c
    vol = float(abs(np.einsum("ij,ij->i", a, np.cross(b, d)).sum()) / 6.0)
    area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    return vol, area


def observed_vertices(V, scan_V, scan_F, BL, tol=0.005):
    """vertex has scan surface within tol * BL (scan vertices + face centroids + edge midpoints as samples)."""
    Fv = scan_V[scan_F]
    S = np.concatenate(
        [scan_V, Fv.mean(1), 0.5 * (Fv[:, 0] + Fv[:, 1]), 0.5 * (Fv[:, 1] + Fv[:, 2]), 0.5 * (Fv[:, 2] + Fv[:, 0])]
    )
    d = cKDTree(S).query(V, workers=1)[0]
    return d <= tol * BL, d / BL


def _section_area(X, F, a, u):
    """area of the closed cut of faces F of mesh X by the plane through a with normal u; NaN if not one closed loop."""
    s = (X - a) @ u
    sf = s[F]
    cross = (sf.min(1) < 0) & (sf.max(1) > 0)
    if cross.sum() < 3:
        return np.nan
    segs = []
    for f in F[cross]:
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            si, sj = s[f[i]], s[f[j]]
            if (si < 0) != (sj < 0):
                w = si / (si - sj)
                pts.append(X[f[i]] + w * (X[f[j]] - X[f[i]]))
        if len(pts) == 2:
            segs.append(pts)
    if len(segs) < 3:
        return np.nan
    P = np.array([p for sg in segs for p in sg])
    # link segment ends into loops (ends coincide up to rounding)
    key = np.round(P / (1e-6 + np.abs(P).max()) * 1e7).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 2)
    adj = {}
    for k, (p, q) in enumerate(inv):
        adj.setdefault(p, []).append((q, k))
        adj.setdefault(q, []).append((p, k))
    if any(len(v) != 2 for v in adj.values()):
        return np.nan
    used = np.zeros(len(inv), bool)
    loops = []
    for start in adj:
        if all(used[k] for _, k in adj[start]):
            continue
        loop, cur, prev_k = [start], start, None
        while True:
            nxt = [(q, k) for q, k in adj[cur] if not used[k] and k != prev_k]
            if not nxt:
                break
            q, k = nxt[0]
            used[k] = True
            if q == start:
                break
            loop.append(q)
            cur, prev_k = q, k
        loops.append(loop)
    if len(loops) != 1:
        return np.nan
    pts = {}
    for k, (p, q) in enumerate(inv):
        pts[p], pts[q] = P[2 * k], P[2 * k + 1]
    L = np.array([pts[i] for i in loops[0]])
    c = L.mean(0)
    return float(0.5 * abs(np.cross(L - c, np.roll(L, -1, 0) - c) @ u).sum())


def measure(V, part_code, obs, uc1, waist_segments=None, specimen_class="partial", xsec=True):
    """V (10235,3) posed, scan frame, um; part_code (55,) rs.status codes; obs (10235,) bool observed vertices;
    uc1 = parts.uc1_unpose.unpose(V, valid) result (V_rest_um, fallback, s). Returns (row, flags, detail)."""
    st = _static()
    t = st["t"]
    ix, dom = t.idx, t.dom
    code = np.asarray(part_code, int)
    Mk = st["M"] @ V
    X = np.asarray(uc1["V_rest_um"], float)
    Mr = st["M"] @ X
    fb = np.asarray(uc1["fallback"], bool)
    row, flag, detail = {}, {}, {}

    def pflag(parts):
        cs = [code[ix[p]] for p in parts]
        if any(c not in MEAS + FLAGGED for c in cs):
            return 2
        return 1 if any(c in FLAGGED for c in cs) else 0

    def put(k, v, f):
        row[k] = float(v) if v is not None and np.isfinite(v) else np.nan
        flag[k] = int(2 if not np.isfinite(row[k]) else f)

    def obs_rule(ids, f):
        if len(ids) < 6:
            return 2
        sh = float(obs[ids].mean())
        if sh < 0.2:
            return 2
        return max(f, 1 if sh < 0.5 else 0)

    def girth(a, b, ids):
        u = Mk[b] - Mk[a]
        L = np.linalg.norm(u)
        u = u / max(L, 1e-9)
        rel = V[ids] - Mk[a]
        tt = (rel @ u) / max(L, 1e-9)
        r = np.linalg.norm(rel - (rel @ u)[:, None] * u, axis=1)
        m = (tt > 0.2) & (tt < 0.8)
        return float(np.median(r[m])) if m.sum() >= 3 else np.nan

    def limb(key, na, nb, own):
        a, b = ix[na], ix[nb]
        parts = [na, nb]
        f = pflag(parts)
        ring_fb = bool(fb[st["ring"][a]].any() or fb[st["ring"][b]].any())
        put(f"{key}_len", np.linalg.norm(Mk[b] - Mk[a]), max(f, 1 if ring_fb and f < 2 else 0))
        ids = np.where(np.isin(dom, own))[0]
        g = girth(a, b, np.where(dom == a)[0])
        put(f"{key}_rad", g, obs_rule(np.where(dom == a)[0], f) if f < 2 else 2)
        if xsec:
            u = _unit(Mr[b] - Mr[a])
            fm = np.isin(dom[t.F], [a]).all(1)
            areas = [_section_area(X, t.F[fm], Mr[a] + q * (Mr[b] - Mr[a]), u) for q in (0.25, 0.5, 0.75)]
            fbp = bool(fb[ids].any())
            put(f"{key}_xsec_um2", areas[1], max(f, 1 if fbp and f < 2 else 0) if f < 2 else 2)
            detail[f"{key}_xsec_um2_q25_q50_q75"] = areas

    for k in "123":
        for s in "lr":
            for seg, a, b in LEG_SEG:
                na, nb = f"l_{k}_{a}_{s}", f"l_{k}_{b}_{s}"
                limb(f"leg{k}{s}_{seg}", na, nb, [ix[na]] + ([ix[nb]] if b == "pt" else []))
            ba, tr = f"leg{k}{s}_basitarsus_len", f"leg{k}{s}_tarsus_rest_len"
            put(f"leg{k}{s}_tarsus_len", row[ba] + row[tr], max(flag[ba], flag[tr]))
    for s in "lr":
        for seg, a, b in ANT_SEG:
            na, nb = f"{a}_{s}", f"{b}_{s}"
            limb(f"{seg}_{s}", na, nb, [ix[na]] + ([ix[nb]] if b == "an_3" else []))
    # ---- trunk chords (posed markers) ------------------------------------------------------------------------------
    head = np.where(dom == ix["b_h"])[0]
    meso = np.where(dom == ix["b_t"])[0]
    neck, pet = Mk[ix["b_h"]], Mk[ix["b_a_1"]]
    put("trunk_neck_to_petiole", np.linalg.norm(pet - neck), pflag(["b_h", "b_t", "b_a_1"]))
    put("waist1_length", np.linalg.norm(Mk[ix["b_a_2"]] - Mk[ix["b_a_1"]]), pflag(["b_a_1", "b_a_2"]))
    put("waist2_length", np.linalg.norm(Mk[ix["b_a_3"]] - Mk[ix["b_a_2"]]), pflag(["b_a_1", "b_a_2", "b_a_3"]))
    put("waist_length", np.linalg.norm(Mk[ix["b_a_3"]] - Mk[ix["b_a_1"]]), pflag(["b_a_1", "b_a_2", "b_a_3"]))
    put("gaster_length", np.linalg.norm(Mk[ix["b_a_5"]] - Mk[ix["b_a_3"]]), pflag(["b_a_3", "b_a_4"]))
    # ---- head / mesosoma extents (posed = rest for a rigid part) --------------------------------------------------
    fh = pflag(["b_h", "b_t"])
    yh = _lateral(V, head, st)
    head_front = np.nan
    if yh is not None:
        mb = V[head].mean(0)
        ah = _unit((mb - neck) - ((mb - neck) @ yh) * yh)
        ph = V[head] @ ah
        fo = obs_rule(head, fh) if fh < 2 else 2
        put("head_width", _extent(V[head], yh), fo)
        put("head_length", ph.max() - ph.min(), fo)
        put("head_height", _extent(V[head], np.cross(ah, yh)), fo)
        head_front = float(ph.max() - neck @ ah)
    ym = _lateral(V, meso, st)
    if ym is not None:
        am = _unit((neck - pet) - ((neck - pet) @ ym) * ym)
        pm = V[meso] @ am
        fm_ = pflag(["b_h", "b_t"])
        put("mesosoma_length", pm.max() - pm.min(), fm_)
        ant = pm > pm.min() + 0.6 * (pm.max() - pm.min())
        put("pronotum_width", _extent(V[meso][ant], ym), fm_)
        put("mesosoma_height", _extent(V[meso], np.cross(am, ym)), fm_)
    trunk_parts = ["b_h", "b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]
    chords = [
        np.linalg.norm(Mk[ix[b]] - Mk[ix[a]])
        for a, b in (("b_h", "b_a_1"), ("b_a_1", "b_a_2"), ("b_a_2", "b_a_3"), ("b_a_3", "b_a_5"))
    ]
    put("body_length_chords", head_front + sum(chords), pflag(trunk_parts))
    # ---- waist / gaster extents on the UC1 rest mesh x s (pose-free) -------------------------------------------------
    w1, w2 = np.where(dom == ix["b_a_1"])[0], np.where(dom == ix["b_a_2"])[0]
    gas = np.where(np.isin(dom, [ix["b_a_3"], ix["b_a_4"], ix["b_a_5"]]))[0]
    for nm, ids, a, b, parts in (
        ("waist1", w1, "b_a_1", "b_a_2", ["b_a_1", "b_a_2"]),
        ("waist2", w2, "b_a_2", "b_a_3", ["b_a_1", "b_a_2", "b_a_3"]),
        ("waist", np.concatenate([w1, w2]), "b_a_1", "b_a_3", ["b_a_1", "b_a_2", "b_a_3"]),
        ("gaster", gas, "b_a_3", "b_a_5", ["b_a_3", "b_a_4", "b_a_5"]),
    ):
        f = pflag(parts)
        y = _lateral(X, ids, st)
        if y is None or len(ids) < 6:
            put(f"{nm}_width", np.nan, 2)
            put(f"{nm}_height", np.nan, 2)
            continue
        ax = _unit((Mr[ix[b]] - Mr[ix[a]]) - ((Mr[ix[b]] - Mr[ix[a]]) @ y) * y)
        fo = obs_rule(ids, f) if f < 2 else 2
        if waist_segments not in (1, 2) and nm.startswith("waist"):
            fo = max(fo, 1) if fo < 2 else 2
        put(f"{nm}_width", _extent(X[ids], y), fo)
        put(f"{nm}_height", _extent(X[ids], np.cross(ax, y)), fo)
    for k in ("length", "width", "height"):
        if waist_segments == 2:
            put(f"petiole_{k}", row[f"waist1_{k}"], flag[f"waist1_{k}"])
            put(f"postpetiole_{k}", row[f"waist2_{k}"], flag[f"waist2_{k}"])
        else:
            fw = flag[f"waist_{k}"]
            put(f"petiole_{k}", row[f"waist_{k}"], max(fw, 1) if (waist_segments not in (1, 2) and fw < 2) else fw)
            put(f"postpetiole_{k}", np.nan, 2)
    # ---- volumes / areas of part groups on the UC1 rest mesh x s -------------------------------------------------------
    fdom = dom[t.F]
    fbone = np.where(fdom[:, 0] == fdom[:, 1], fdom[:, 0], np.where(fdom[:, 1] == fdom[:, 2], fdom[:, 1], fdom[:, 0]))
    own = st["owners"]
    for g, bl in (("head", ["b_h"]), ("mesosoma", ["b_t"]), ("waist", ["b_a_1", "b_a_2"]), ("gaster", ["b_a_3"])):
        ids_b = [ix[b] for b in bl]
        fm = np.isin(own[fbone], ids_b)
        vol, area = _vol_area(X, t.F[fm])
        parts = bl + (["b_a_4", "b_a_5"] if g == "gaster" else [])
        f = pflag(parts)
        vids = np.unique(t.F[fm])
        if f < 2:
            if fb[vids].any():
                f = 2
            else:
                f = max(f, 1 if obs[vids].mean() < 0.5 else 0)
        put(f"{g}_volume_um3", vol, f)
        put(f"{g}_area_um2", area, f)
    # ---- left / right -----------------------------------------------------------------------------------------------
    lr_vals = []
    pairs = [
        (f"leg{k}l_{s}_{q}", f"leg{k}r_{s}_{q}", f"lr_leg{k}_{s}_{q}_reldiff")
        for k in "123"
        for s in ("coxa", "femur", "tibia", "basitarsus", "tarsus_rest", "tarsus")
        for q in ("len", "rad")
        if not (s == "tarsus" and q == "rad")
    ]
    pairs += [
        (f"{s}_l_{q}", f"{s}_r_{q}", f"lr_{s}_{q}_reldiff") for s in ("scape", "funiculus") for q in ("len", "rad")
    ]
    for a, b, k in pairs:
        va, vb = row[a], row[b]
        v = 2 * (va - vb) / (va + vb) if np.isfinite(va + vb) and va + vb > 0 else np.nan
        put(k, v, max(flag[a], flag[b]))
        if k.endswith("_len_reldiff") and "_tarsus_len" not in k and flag[k] == 0:
            lr_vals.append(abs(v))
    put("lr_len_absdiff_median", float(np.median(lr_vals)) if lr_vals else np.nan, 0)
    row["lr_len_pairs"], flag["lr_len_pairs"] = float(len(lr_vals)), 0
    tr_ok = pflag(trunk_parts)
    put("um_per_model_unit", uc1["s"], 0 if tr_ok < 2 else 2)
    if specimen_class == "do_not_use":
        flag = {k: 3 for k in flag}
    return row, flag, detail


COLUMNS = None


def columns():
    """fixed column order = the keys measure() writes (computed once on the template rest mesh)."""
    global COLUMNS
    if COLUMNS is None:
        from fitter_3d.antscan_registration.parts import uc1_unpose as U

        t = _static()["t"]
        u = U.unpose(t.V0 * 1000.0)
        row, _, _ = measure(t.V0 * 1000.0, np.zeros(t.nJ, int), np.ones(t.nV, bool), u, 1, "complete", xsec=True)
        COLUMNS = list(row)
    return COLUMNS
