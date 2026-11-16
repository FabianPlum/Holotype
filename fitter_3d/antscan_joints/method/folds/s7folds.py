"""S7 FOLDS-AND-TIPS: corrections of S6 limb products along the limb (folded knees, antennal elbows, ...).

Input: an S6 work dir (trunk/<name>/trunk.{json,npz}, limbs/<name>/limbs.{json,npz}, proposals/<name>/<name>_joints.json)
and the envelope mesh. Output: the same contracts (limbs.json/npz and the joint product) in an S7 out dir, with every
changed limb / joint marked (`s7` evidence, provenance) and everything else copied unchanged.

Nothing here reads truth, oracle limbs, development-set geometry, specimen names or per-specimen constants. ALONG's LOSO models are
selected by name only for the dev evaluation (--loso), exactly as S6 does.
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s7geom as G
import s7lib as L

# ------------------------------------------------------------------------------------------------ constants
KNEE = dict(
    pitch_div=500,  # local grid pitch = BL / pitch_div
    margin_pct=10.0,  # box margin around the leg's free polyline
    tube_pct=14.0,  # passable solid within this distance of the leg polyline
    other_pct=3.0,  # material within this of another limb's polyline and nearer to it is that limb's
    root_seed_pct=2.0,  # geodesic seeds: the last 2 % BL of the free polyline (towards the trunk)
    nms_pct=3.5,  # extremities: local maxima of the root geodesic, NMS radius
    gmin_pct=5.0,  # extremities farther than this from the root (geodesic)
    tip_excl_pct=4.0,  # extremities within this of the polyline tip are tips, not knees
    dir_win_pct=3.0,  # incoming / outgoing direction windows at the departure point
    cos_in_min=0.3,  # knee geometry: the apex continues the incoming (distal arm) direction ...
    cos_out_max=-0.3,  # ... and reverses the outgoing (proximal arm) direction
)


def free_part(lj, lz, ch):
    X = np.asarray(lz[ch + "_X"], float)
    s = L.arclen(X)
    core = (lj["limbs"][ch] or {}).get("core_arc_um") or s[-1]
    return X, s, float(core)


class LegContext:
    """Local fine grid around one leg's free polyline; passable = solid minus the trunk core minus other limbs'
    material, within tube_pct of the polyline."""

    def __init__(self, ch, lj, lz, grid, tri, BL, cfg=KNEE):
        self.ch, self.BL, self.cfg = ch, BL, cfg
        X, s, core = free_part(lj, lz, ch)
        self.X, self.s, self.core = X, s, core
        free = s <= core
        self.Xf, self.sf = X[free], s[free]
        box = G.Box(tri, self.Xf, cfg["margin_pct"] / 100 * BL, BL / cfg["pitch_div"])
        self.box = box
        occ = box.occ.copy()
        idx = np.argwhere(occ)
        P = box.xyz(idx)
        q = grid.to_ijk(P)
        ok = grid.inside(q)
        tp = np.full(len(idx), -1)
        tp[ok] = grid.tpart[tuple(q[ok].T)]
        rm = tp >= 0
        self.Xd = L.resample(self.Xf, 0.25 * box.pitch)[0]
        d_own, _ = cKDTree(self.Xd).query(P, k=1)
        for c2 in L.LIMBS:
            if c2 == ch or c2 + "_X" not in lz or (lj["limbs"].get(c2) or {}).get("status") in ("no_path", None):
                continue
            X2, s2, c2core = free_part(lj, lz, c2)
            X2f = L.resample(X2[s2 <= c2core], 0.5 * box.pitch)[0]
            d2, _ = cKDTree(X2f).query(P, k=1, distance_upper_bound=cfg["other_pct"] / 100 * BL)
            rm |= np.isfinite(d2) & (d2 < d_own)
        rm |= d_own > cfg["tube_pct"] / 100 * BL
        occ[tuple(idx[rm].T)] = False
        self.passable = occ
        self._dtp = None

    @property
    def dtp(self):
        """Distance transform of the passable region (um)."""
        if self._dtp is None:
            from scipy import ndimage

            self._dtp = ndimage.distance_transform_edt(self.passable) * self.box.pitch
        return self._dtp

    def root_seeds(self):
        sd = L.arclen(self.Xd)
        cand = self.Xd[sd >= sd[-1] - self.cfg["root_seed_pct"] / 100 * self.BL][::2]
        return [v for v in self.box.snap(cand, self.passable, r_vox=3) if v is not None]


def find_knees(ctx):
    """Fold apices: extremities of the geodesic distance from the leg root inside the leg's own solid that lie in the
    leg's interior with knee geometry. Returns a list of dicts (apex, cap centre, proximal route, departure arc, ...)."""
    cfg, box, BL = ctx.cfg, ctx.box, ctx.BL
    seeds = ctx.root_seeds()
    if not seeds:
        return []
    cum, mcp = G.geodesic(ctx.passable, seeds)
    nms = max(2, int(round(cfg["nms_pct"] / 100 * BL / box.pitch)))
    tips = G.local_maxima(cum, ctx.passable, nms, cfg["gmin_pct"] / 100 * BL / box.pitch)
    out = []
    dtp = ctx.dtp
    mcp_c = None
    for t in tips:
        apex = box.xyz(t)[0]
        if np.linalg.norm(apex - ctx.X[0]) < cfg["tip_excl_pct"] / 100 * BL:
            continue
        if mcp_c is None:
            # centred routes from the root seeds (cost 1/DT^2 as a medial path), one cost field for all extremities
            from skimage.graph import MCP_Geometric

            w = 1.0 / np.maximum(dtp / box.pitch, 0.5) ** 2
            mcp_c = MCP_Geometric(np.where(ctx.passable, w, np.inf), fully_connected=True)
            cum_c, _ = mcp_c.find_costs([tuple(int(v) for v in sd) for sd in seeds])
        if not np.isfinite(cum_c[tuple(t)]):
            continue
        rt = np.asarray(mcp_c.traceback(tuple(int(v) for v in t)))[::-1]  # apex first
        R = box.xyz(rt)
        dR, aR = L.dist_poly(R, ctx.Xf)
        rR = dtp[tuple(rt.T)]
        # departure: walking from the root end towards the apex, the last route point still on the polyline
        on = dR <= np.maximum(0.6 * rR, 1.5 * box.pitch)
        if not on.any():
            continue
        k_dep = int(np.nonzero(on)[0].min())  # first (from the apex) point on the polyline
        a_dep = float(aR[k_dep])
        if a_dep < cfg["tip_excl_pct"] / 100 * BL or a_dep > ctx.core - 0.03 * BL:
            continue
        w = cfg["dir_win_pct"] / 100 * BL
        kk = lambda a: min(int(np.searchsorted(ctx.sf, a)), len(ctx.Xf) - 1)  # noqa: E731
        pd, pin, pout = ctx.Xf[kk(a_dep)], ctx.Xf[kk(a_dep - w)], ctx.Xf[kk(a_dep + w)]
        d_in, d_out = L.unit(pd - pin), L.unit(pout - pd)
        av = L.unit(apex - pd)
        cin, cout = float(av @ d_in), float(av @ d_out)
        # cap centre: along the route from the apex, the first point whose inscribed radius reaches its distance
        # from the apex (the centre of the rounded end of the fold)
        sR = L.arclen(R)
        cap_k = None
        for k in range(len(R)):
            if rR[k] >= 0.9 * np.linalg.norm(R[k] - apex) and k > 0:
                cap_k = k
                break
        if cap_k is None:
            cap_k = min(len(R) - 1, int(np.searchsorted(sR, 0.01 * BL)))
        cap = R[cap_k]
        r_cap = float(rR[cap_k])
        dcap, acap = L.dist_poly(cap[None], ctx.Xf)
        out.append(
            {
                "apex": apex,
                "cap": cap,
                "r_cap": r_cap,
                "route": R[: k_dep + 1],
                "a_dep": a_dep,
                "G_pct": float(cum[tuple(t)] * box.pitch * 100 / BL),
                "cos_in": cin,
                "cos_out": cout,
                "knee_geom": bool(cin >= cfg["cos_in_min"] and cout <= cfg["cos_out_max"]),
                "d_cap_poly": float(dcap[0]),
                "d_apex_poly": float(L.dist_poly(apex[None], ctx.Xf)[0][0]),
                "len_arm": float(sR[k_dep]),
            }
        )
    return out


# ------------------------------------------------------------------------------------------------ leg tip (claws)
TIP = dict(
    pitch_div=700,  # claws are thin: finer grid
    region_pct=12.0,  # distal part of the polyline analysed
    margin_pct=5.0,
    tube_pct=4.0,  # passable within this of the polyline
    seed_pct=10.0,  # geodesic seed: the polyline point at this arc from the tip
    nms_pct=1.0,  # candidate tips: local maxima of G, NMS radius
    near_pct=6.0,  # candidate tips within this of the current tip
    move_min_pct=1.0,  # change the tip only when the farthest candidate is this far from the current tip ...
    move_max_pct=4.0,  # ... and not farther (claws of one leg lie within ~3.5 % BL of each other)
    cur_r_pct=0.5,  # G at the current tip = max G within this radius
    dG_max_pct=1.0,  # a far larger G is not a claw of this tip (contact / other structure): keep
)  # ... and only when it is farther (G) than the current tip by more than one voxel (ties)


def tip_context(ch, lj, lz, grid, tri, BL, cfg=TIP):
    """Local fine grid around the distal part of a leg; passable = own solid within tube_pct of the polyline."""
    X, s, core = free_part(lj, lz, ch)
    sel = s <= cfg["region_pct"] / 100 * BL
    box = G.Box(tri, X[sel], cfg["margin_pct"] / 100 * BL, BL / cfg["pitch_div"])
    occ = box.occ.copy()
    idx = np.argwhere(occ)
    P = box.xyz(idx)
    q = grid.to_ijk(P)
    ok = grid.inside(q)
    tp = np.full(len(idx), -1)
    tp[ok] = grid.tpart[tuple(q[ok].T)]
    rm = tp >= 0
    Xd = L.resample(X[s <= core], 0.25 * box.pitch)[0]
    d_own, _ = cKDTree(Xd).query(P, k=1)
    for c2 in L.LIMBS:
        if c2 == ch or c2 + "_X" not in lz or (lj["limbs"].get(c2) or {}).get("status") in ("no_path", None):
            continue
        X2, s2, c2core = free_part(lj, lz, c2)
        X2f = L.resample(X2[s2 <= c2core], 0.5 * box.pitch)[0]
        d2, _ = cKDTree(X2f).query(P, k=1, distance_upper_bound=KNEE["other_pct"] / 100 * BL)
        rm |= np.isfinite(d2) & (d2 < d_own)
    rm |= d_own > cfg["tube_pct"] / 100 * BL
    occ[tuple(idx[rm].T)] = False
    return X, s, box, occ


def farthest_tip(ch, lj, lz, grid, tri, BL, cfg=TIP):
    """The leg tip as the extremity of the leg's solid farthest (geodesically, through the solid) from the body: among
    the local maxima of the geodesic distance from a point 10 % BL up the leg, near the current tip, the farthest.
    Returns None (keep) or {tip, route (new tip -> seed), seed_arc, candidates}."""
    X, s, box, passable = tip_context(ch, lj, lz, grid, tri, BL, cfg)
    a_seed = cfg["seed_pct"] / 100 * BL
    if s[-1] < 2 * a_seed:
        return None
    k_seed = min(int(np.searchsorted(s, a_seed)), len(X) - 1)
    sd = box.snap(X[k_seed][None], passable, r_vox=6)[0]
    if sd is None:
        return None
    cum, mcp = G.geodesic(passable, [sd])
    nms = max(2, int(round(cfg["nms_pct"] / 100 * BL / box.pitch)))
    tips = G.local_maxima(cum, passable, nms, 0.5 * a_seed / box.pitch)
    if len(tips) == 0:
        return None
    tipx = box.xyz(tips)
    near = np.linalg.norm(tipx - X[0], axis=1) < cfg["near_pct"] / 100 * BL
    if not near.any():
        return None
    tips, tipx = tips[near], tipx[near]
    g = cum[tuple(tips.T)] * box.pitch
    o = np.argsort(-g)
    best = int(o[0])
    # geodesic distance reached at the current tip: the largest G within cur_r of it
    q0 = box.ijk(X[0][None])[0]
    rv = max(1, int(round(cfg["cur_r_pct"] / 100 * BL / box.pitch)))
    lo_, hi_ = np.maximum(q0 - rv, 0), np.minimum(q0 + rv + 1, np.array(box.shape))
    sub = cum[lo_[0] : hi_[0], lo_[1] : hi_[1], lo_[2] : hi_[2]]
    sub = sub[np.isfinite(sub)]
    g_cur = float(sub.max() * box.pitch) if sub.size else 0.0
    cands = [
        {
            "xyz": tipx[i].tolist(),
            "G_pct": float(g[i] * 100 / BL),
            "d_current_pct": float(np.linalg.norm(tipx[i] - X[0]) * 100 / BL),
        }
        for i in o[:4]
    ]
    dG = float(g[best] - g_cur)
    info = {"candidates": cands, "G_current_pct": g_cur * 100 / BL, "dG_pct": dG * 100 / BL}
    dmove = float(np.linalg.norm(tipx[best] - X[0]))
    if (
        dmove < cfg["move_min_pct"] / 100 * BL
        or dmove > cfg["move_max_pct"] / 100 * BL
        or dG <= box.pitch
        or dG > cfg["dG_max_pct"] / 100 * BL
    ):
        return {"tip": None, **info}
    from scipy import ndimage

    dt = ndimage.distance_transform_edt(passable)
    rt = G.route(passable, sd, tips[best], dt=dt)
    if rt is None:
        return {"tip": None, "candidates": cands}
    R = box.xyz(rt)[::-1]  # new tip first ... seed
    return {"tip": tipx[best], "route": R, "seed_arc": float(s[k_seed]), **info}


# ------------------------------------------------------------------------------------------------ antennal elbow
ELBOW = dict(
    along_conf_max=0.5,  # only where ALONG's own an_2 placement is unsure (S6 joint_detail along_conf; hand-set on dev)
    bend_win_pct=2.0,  # bend measured between chords of this arc length
    tip_excl_pct=3.0,  # not within this of the tip
    core_excl_pct=2.0,  # nor within this of the trunk entry (the socket bend)
    min_bend_deg=25.0,  # a weaker strongest bend is no evidence
    peak_win_pct=1.5,  # v8: the bend must be the maximum of the unmasked profile within this arc
    near_core_pct=3.0,  # v8: within this of the trunk entry ...
    near_core_min_deg=60.0,  # ... only a sharp elbow (scape in its scrobe) is accepted
    min_funiculus_frac=0.25,  # v8: elbow at least this fraction of the free arc from the tip
    min_scape_pct=1.5,  # v8: elbow at least this arc (% BL) distal of S6's an_1
)


def elbow_pin(ch, lj, lz, prop, BL, cfg=ELBOW):
    """Antennal elbow = the strongest bend of the free centreline, used only when ALONG was unsure of an_2."""
    nm = L.omni(ch, "an_2")
    jd = (prop.get("joint_detail") or {}).get(nm) or {}
    ac = jd.get("along_conf")
    if ac is None or ac >= cfg["along_conf_max"]:
        return None
    X, s, core = free_part(lj, lz, ch)
    Xr, sr = L.resample(X, 0.0025 * BL)
    b = G.bend_profile(Xr, sr, cfg["bend_win_pct"] / 100 * BL)
    m = (sr > cfg["tip_excl_pct"] / 100 * BL) & (sr < core - cfg["core_excl_pct"] / 100 * BL)
    if not m.any():
        return None
    bb = np.where(m, b, -1.0)
    k = int(np.argmax(bb))
    if bb[k] < cfg["min_bend_deg"]:
        return None
    # a genuine peak of the unmasked bend profile (not the rising flank of the socket bend at the window edge) ...
    near = np.abs(sr - sr[k]) <= cfg["peak_win_pct"] / 100 * BL
    if b[near].max() > b[k] + 1e-6:
        return None
    # ... and close to the trunk entry (scape lying in its scrobe) only when it is a sharp elbow
    if core - sr[k] < cfg["near_core_pct"] / 100 * BL and bb[k] < cfg["near_core_min_deg"]:
        return None
    # anatomical plausibility: the funiculus (tip -> elbow) is not a stub, and the scape is not shorter than
    # min_scape_pct (arc from the elbow to S6's scape base an_1)
    if sr[k] < cfg["min_funiculus_frac"] * core:
        return None
    a1 = (prop.get("joints_scan") or {}).get(L.omni(ch, "an_1"))
    if a1 is not None:
        _, s1 = L.dist_poly(np.asarray(a1, float)[None], Xr)
        if s1[0] - sr[k] < cfg["min_scape_pct"] / 100 * BL:
            return None
    return {"xyz": Xr[k], "bend_deg": float(bb[k]), "arc_pct": float(sr[k] * 100 / BL), "along_conf_s6": ac}


# ------------------------------------------------------------------------------------------------ fold correction
FOLD = dict(
    g_min_pct=15.0,  # knee apex at least this far (geodesic) from the leg root ...
    g_max_pct=50.0,  # ... and at most this far (coxa + trochanter + femur; longest dev femur ~33 % BL)
    cos_out_max=-0.7,  # the apex lies backwards along the proximal arm (folded tibia)
    dep_frac=(0.25, 0.75),  # the fold departs from the polyline in the femur-tibia range of the free arc (from the tip)
    cos_in_min=-0.45,  # and not backwards along the distal arm
    dcap_min_pct=1.0,  # the fold's cap centre is off the polyline
    spread_min_pct=1.0,  # apply only where S6's fe is uncertain: approaches' fe spread above this, or fe by proportion
    fe_offset_pct=1.0,  # fe = the point of the centred route at this arc from the apex (chosen on dev)
)


def s6_fe_uncertain(prop, ch, cfg=FOLD):
    nm = L.omni(ch, "fe")
    jd = (prop.get("joint_detail") or {}).get(nm) or {}
    if (prop.get("joints_scan") or {}).get(nm) is None:
        return True, "fe null"
    if str((prop.get("provenance") or {}).get(nm, "")).startswith("proportion"):
        return True, "fe by proportion"
    if (jd.get("spread_pct") or 0) > cfg["spread_min_pct"]:
        return True, f"approaches' fe spread {jd.get('spread_pct')} % BL"
    if jd.get("start_joint") not in (None, "pt"):
        return True, f"S6 chain starts at {jd.get('start_joint')}"
    return False, None


def fold_correction(ctx, prop, cfg=FOLD):
    """Fold correction for one leg: (new polyline, new core arc, fe pin, evidence) or None."""
    BL = ctx.BL
    ks = [
        k
        for k in find_knees(ctx)
        if cfg["g_min_pct"] <= k["G_pct"] <= cfg["g_max_pct"]
        and k["cos_out"] <= cfg["cos_out_max"]
        and k["cos_in"] >= cfg["cos_in_min"]
        and k["d_cap_poly"] >= cfg["dcap_min_pct"] / 100 * BL
        and cfg["dep_frac"][0] <= k["a_dep"] / max(ctx.core, 1e-9) <= cfg["dep_frac"][1]
    ]
    if not ks:
        return None
    unc, why = s6_fe_uncertain(prop, ctx.ch, cfg)
    k = max(ks, key=lambda q: q["G_pct"])
    ev = {
        "apex": k["apex"].tolist(),
        "G_pct": round(k["G_pct"], 2),
        "departure_arc_pct": round(k["a_dep"] * 100 / BL, 2),
        "cos_in": round(k["cos_in"], 2),
        "cos_out": round(k["cos_out"], 2),
        "cap_off_polyline_pct": round(k["d_cap_poly"] * 100 / BL, 2),
        "s6_fe_uncertain": why,
        "n_candidates": len(ks),
    }
    if not unc:
        ev["applied"] = False
        return {"applied": False, "evidence": ev}
    R = k["route"]  # apex ... departure point (on the polyline)
    sR = L.arclen(R)
    fe = R[min(int(np.searchsorted(sR, cfg["fe_offset_pct"] / 100 * BL)), len(R) - 1)]
    X, s, core = ctx.X, ctx.s, ctx.core
    a = k["a_dep"]
    Xn = np.vstack([X[s < a], R[::-1], R[1:], X[s > a]])
    add = 2.0 * float(sR[-1])
    ev.update({"applied": True, "fe_pin": fe.tolist(), "detour_arc_pct": round(add * 100 / BL, 2)})
    return {"applied": True, "X": Xn, "core_arc_um": core + add, "pins": {"fe": fe}, "evidence": ev}


# ------------------------------------------------------------------------------------------------ distal extension
EXTEND = dict(
    agree_min=0.80,  # share of our free part within tol of the other approach's free polyline (hand-set on dev)
    tol_pct=1.5,
    ext_min_pct=5.0,  # the other polyline continues at least this far beyond our tip
    other_share_max=0.2,  # its extension may not run along another named limb's polyline
    arc_ratio_max=1.3,  # extended free arc <= this x the longest free arc of the specimen's complete legs
    pin_s6_start=False,  # keep S6's tip joint pinned (tried in v5: worse where S6's tip joint was wrong; off)
    inside_min=0.9,  # v8: >= this share of the extension's samples inside the envelope (no jumps through air)
    tip_gap_max_pct=4.0,  # v9: our tip within this of the other polyline (dev extensions 1.0-3.9; a 6.7 jump was wrong)
)


def s6_tip_not_free(prop, ch):
    """S6's ALONG decided that the leg polyline's tip is not the leg tip (chain starts at ta/ti/fe/tr) or pt is null."""
    jd = None
    for j in ("pt", "ta", "ti", "fe", "tr", "co"):
        jd = (prop.get("joint_detail") or {}).get(L.omni(ch, j))
        if jd and jd.get("start_joint"):
            break
    st = (jd or {}).get("start_joint")
    return st != "pt", st


def complete_leg_arc(lj, lz, prop):
    """Longest free arc (tip -> trunk entry) among the legs whose S6 chain starts at pt (complete legs), or None."""
    best = None
    for c in L.LEGS:
        if c + "_X" not in lz or (lj["limbs"].get(c) or {}).get("status") in ("no_path", None):
            continue
        if s6_tip_not_free(prop, c)[0]:
            continue
        _, _, core = free_part(lj, lz, c)
        best = core if best is None else max(best, core)
    return best


def inside_share(grid, P):
    """Share of points inside the trunk product's occupancy (any voxel of the 3x3x3 neighbourhood occupied)."""
    from scipy import ndimage

    if getattr(grid, "_occ_d", None) is None:
        grid._occ_d = ndimage.binary_dilation(grid.occ, iterations=1)
    q = grid.to_ijk(P)
    ok = grid.inside(q)
    v = np.zeros(len(q), bool)
    v[ok] = grid._occ_d[tuple(q[ok].T)]
    return float(v.mean()) if len(v) else 0.0


def distal_extension(ch, lj, lz, raws, BL, prop, cfg=EXTEND, grid=None):
    """Where S6's leg polyline ends before the leg tip, another PATHS approach's polyline of the same leg that contains
    ours and continues distally supplies the missing distal part. Returns None or {X, core_arc_um, evidence}."""
    notfree, st = s6_tip_not_free(prop, ch)
    if not notfree:
        return None
    X, s, core = free_part(lj, lz, ch)
    Xf = L.resample(X[s <= core], 0.005 * BL)[0]
    tol = cfg["tol_pct"] / 100 * BL
    best = None
    for a, (rj, rz) in raws.items():
        if (
            rz is None
            or ch + "_X" not in rz
            or (rj["limbs"].get(ch) or {}).get("status") in ("no_path", "failed", None)
        ):
            continue
        Y = np.asarray(rz[ch + "_X"], float)
        sy = L.arclen(Y)
        cy = (rj["limbs"][ch] or {}).get("core_arc_um") or sy[-1]
        Yf = Y[sy <= cy]
        if len(Yf) < 3:
            continue
        d, _ = L.dist_poly(Xf, Yf)
        agree = float(np.mean(d <= tol))
        d_tip, a_tip = L.dist_poly(X[:1], Yf)
        ext = float(a_tip[0])
        if agree < cfg["agree_min"] or ext < cfg["ext_min_pct"] / 100 * BL:
            continue
        if float(d_tip[0]) > cfg["tip_gap_max_pct"] / 100 * BL:  # v9: a continuation of our tip, not a jump
            continue
        Ye = Yf[L.arclen(Yf) < ext]
        Yed = L.resample(Ye, 0.005 * BL)[0] if len(Ye) > 1 else Ye
        near = {}
        for c2 in L.LIMBS:
            if c2 == ch or c2 + "_X" not in lz or (lj["limbs"].get(c2) or {}).get("status") in ("no_path", None):
                continue
            X2, s2, c2c = free_part(lj, lz, c2)
            dd, _ = L.dist_poly(Yed, X2[s2 <= c2c])
            sh = float(np.mean(dd <= tol))
            if sh > cfg["other_share_max"]:
                near[c2] = round(sh, 2)
        if near:
            continue
        if grid is not None and len(Yed) > 2:
            # (the envelope as loaded is not watertight, so mesh.contains() is unreliable; the voxel occupancy is not)
            if inside_share(grid, Yed) < cfg["inside_min"]:
                continue
        if best is None or ext > best["ext"]:
            best = {"approach": a, "agree": agree, "ext": ext, "Ye": Ye}
    if best is None:
        return None
    cmax = complete_leg_arc(lj, lz, prop)
    if cmax is not None and core + best["ext"] > cfg["arc_ratio_max"] * cmax:
        return None
    Xn = np.vstack([best["Ye"], X])
    ev = {
        "from_approach": best["approach"],
        "agree": round(best["agree"], 3),
        "extension_pct": round(best["ext"] * 100 / BL, 2),
        "s6_start_joint": st,
        "longest_complete_leg_pct": None if cmax is None else round(cmax * 100 / BL, 1),
    }
    return {"X": Xn, "core_arc_um": core + best["ext"], "evidence": ev}


# ------------------------------------------------------------------------------------------------ runner
VERSION = "S7 folds 2026-11-11 v9"
CLASSES_ALL = ("extend", "fold", "tip", "elbow")
S6_ALONG = os.path.normpath(os.path.join(HERE, os.pardir, "along"))  # ALONG and its model (method/along)


def _global_conf_factor(prop):
    """S6's specimen-level confidence factors (assemble.py): head-end override x0.5, low head-end confidence x the head
    factor, left/right inconsistency x0.5 -- reapplied to the joints S7 re-places."""
    f = 1.0
    for fl in (prop.get("completeness") or {}).get("flags", []):
        if fl.startswith("head_end_from_round3_prior"):
            f *= 0.5
        elif fl.startswith("head_end_low_confidence("):
            try:
                hc = float(fl.split("(")[1].split(")")[0])
                f *= 0.6 + 0.4 * hc
            except ValueError:
                pass
        elif fl.startswith("left_right_names_disagree"):
            f *= 0.5
    return f


def surface_r(tm, X):
    """Distance of polyline points to the mesh surface (um)."""
    import trimesh

    _, d, _ = trimesh.proximity.closest_point(tm, X)
    return d


def run(name, s6_work, mesh_path, out_dir, loso=False, classes=CLASSES_ALL, replace="corrected"):
    """Correct one specimen's S6 products. replace: 'corrected' = only the corrected joint of a limb changes (others
    keep S6's placement), 'limb' = every joint of a changed limb is re-placed by ALONG on the corrected centreline."""
    import copy
    import json
    import time

    t0 = time.time()
    sys.path.insert(0, S6_ALONG)
    import along as AL
    import geom_profile as GP

    lj, lz = L.load_limbs(os.path.join(s6_work, "limbs"), name)
    tj, tz = L.load_trunk(s6_work, name)
    prop = L.load_prop(os.path.join(s6_work, "proposals"), name)
    if lj is None or prop is None or prop.get("status") != "ok":
        raise RuntimeError("S6 products missing or failed")
    grid = L.Grid(tj, tz)
    BL = float(lj["BL_um"])
    tm = GP.as_trimesh(mesh_path)
    tri = np.asarray(tm.vertices, float)[np.asarray(tm.faces)]
    mpath = os.path.join(S6_ALONG, "loso_models", f"{name}.pkl") if loso else os.path.join(S6_ALONG, "along_model.pkl")
    model = AL.load_model(mpath)
    raws = {a: L.load_limbs(os.path.join(s6_work, f"paths_{a}"), name) for a in "ABC"}
    changes, report = {}, {}
    lzw = dict(lz)  # working copy: legs extended distally are analysed on the extended polyline
    import copy as _copy

    ljw = _copy.deepcopy(lj)
    if "extend" in classes:
        for ch in L.LEGS:
            if ch + "_X" not in lz or (lj["limbs"].get(ch) or {}).get("status") in ("no_path", None):
                continue
            try:
                de = distal_extension(ch, lj, lz, raws, BL, prop, grid=grid)
            except Exception as e:
                report.setdefault("errors", []).append(f"{ch} extend: {type(e).__name__}: {e}")
                de = None
            if de is not None:
                changes[ch] = {"extend": de}
                lzw[ch + "_X"] = de["X"]
                ljw["limbs"][ch]["core_arc_um"] = de["core_arc_um"]
    for ch in L.LIMBS:
        if ch + "_X" not in lz or (lj["limbs"].get(ch) or {}).get("status") in ("no_path", None):
            continue
        rec = changes.get(ch, {})
        # v8: a chain whose S6 polyline tip is the knee or the trochanter (S6 start fe / tr) and that was not extended
        # has its knee at the tip already; a far 'knee' found elsewhere would contradict it
        st_ = s6_tip_not_free(prop, ch)[1] if ch in L.LEGS else None
        skip_fold = st_ in ("fe", "tr") and "extend" not in rec
        if ch in L.LEGS and "fold" in classes and not skip_fold:
            try:
                ctx = LegContext(ch, ljw, lzw, grid, tri, BL)
                fc = fold_correction(ctx, prop)
                if fc is not None:
                    rec["fold"] = fc
            except Exception as e:
                report.setdefault("errors", []).append(f"{ch} fold: {type(e).__name__}: {e}")
        if ch in L.LEGS and "tip" in classes and "extend" not in rec:
            jd = (prop.get("joint_detail") or {}).get(L.omni(ch, "pt")) or {}
            if jd.get("start_joint") == "pt" and (prop.get("joints_scan") or {}).get(L.omni(ch, "pt")) is not None:
                try:
                    ft = farthest_tip(ch, lj, lz, grid, tri, BL)
                    if ft is not None:
                        rec["tip"] = ft
                except Exception as e:
                    report.setdefault("errors", []).append(f"{ch} tip: {type(e).__name__}: {e}")
        if ch in L.ANTS and "elbow" in classes:
            ep = elbow_pin(ch, lj, lz, prop, BL)
            if ep is not None:
                rec["elbow"] = ep
        if rec:
            changes[ch] = rec
    # ---- build the corrected limb product
    lj2 = copy.deepcopy(lj)
    lz2 = dict(lz)
    out = copy.deepcopy(prop)
    gf = _global_conf_factor(prop)
    R_out = np.asarray(prop["frame"]["rotation"], float)
    org = np.asarray(prop["frame"]["origin_scan"], float)
    for ch, rec in changes.items():
        X, s, core = free_part(lj, lz, ch)
        Xn, coren, pins, corrected = X, core, {}, []
        ev = {}
        if "extend" in rec:
            Xn, coren = rec["extend"]["X"], rec["extend"]["core_arc_um"]
            ev["extend"] = rec["extend"]["evidence"]
            st = rec["extend"]["evidence"].get("s6_start_joint")
            js_ = L.CHAIN_J[ch]
            # S6's own tip joint (the joint ALONG put at S6's polyline tip) is kept and pinned; the joints distal of it
            # (the part the extension supplies) are placed by ALONG; tr/co keep S6's placement
            if EXTEND["pin_s6_start"]:
                dist = js_[: js_.index(st)] if st in js_ else ["pt", "ta", "ti"]
                corrected += [j for j in dist if j in ("pt", "ta", "ti", "fe")]
                p_st = (prop.get("joints_scan") or {}).get(L.omni(ch, st)) if st in js_ else None
                if p_st is not None and not rec.get("fold", {}).get("applied"):
                    pins[st] = np.asarray(p_st, float)
            else:
                corrected += ["pt", "ta", "ti", "fe"]
        if rec.get("fold", {}).get("applied"):
            Xn, coren = rec["fold"]["X"], rec["fold"]["core_arc_um"]
            pins.update(rec["fold"]["pins"])
            if "fe" not in corrected:
                corrected.append("fe")
        if "fold" in rec:
            ev["fold"] = rec["fold"]["evidence"]
        if "tip" in rec:
            tp = rec["tip"]
            ev["tip"] = {k: v for k, v in tp.items() if k in ("candidates", "dG_pct", "G_current_pct")}
            if tp.get("tip") is not None:
                sn = L.arclen(Xn)
                a_seed = tp["seed_arc"]
                Xn = np.vstack([tp["route"], Xn[sn > a_seed]])
                coren = coren - a_seed + float(L.arclen(tp["route"])[-1])
                corrected.append("pt")
                ev["tip"]["moved_pct"] = round(float(np.linalg.norm(tp["tip"] - X[0]) * 100 / BL), 2)
        if "elbow" in rec:
            ep = rec["elbow"]
            pins["an_2"] = ep["xyz"]
            corrected.append("an_2")
            ev["elbow"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ep.items()}
        L2 = lj2["limbs"][ch]
        L2.setdefault("evidence", {})
        if isinstance(L2["evidence"], dict):
            L2["evidence"]["s7"] = ev
        L2["s7_corrected"] = corrected
        if Xn is not X:
            Xn = np.asarray(Xn, float)
            lz2[ch + "_X"] = Xn
            lz2[ch + "_s"] = L.arclen(Xn)
            lz2[ch + "_r"] = surface_r(tm, Xn)
            L2["core_arc_um"] = float(coren)
            L2["arc_um"] = float(L.arclen(Xn)[-1])
            L2["n_points"] = int(len(Xn))
            L2["tip_scan"] = Xn[0].tolist()
            if ch in L.LEGS and "fold" in rec and rec["fold"].get("applied") and L2.get("status") == "partial":
                L2.setdefault("flags", []).append("s7_fold_detour")
        report[ch] = {"corrected": corrected, "evidence": ev}
        if not corrected:
            continue
        # ---- joints on the corrected centreline
        res = AL.place(
            ch,
            lz2[ch + "_X"],
            lz2[ch + "_s"],
            lz2[ch + "_r"],
            tm,
            BL,
            L2.get("core_arc_um"),
            model=model,
            limb_status=L2.get("status"),
            pins=pins or None,
        )
        lc = float(L2.get("confidence", 1.0))
        js = (replace == "limb") and L.CHAIN_J[ch] or corrected
        for j in js:
            nm = L.omni(ch, j)
            v = res.get(j) or {}
            if v.get("xyz") is None:
                out["joints_scan"][nm] = None
                out["joints_canonical"][nm] = None
                out["confidence"][nm] = 0.0
                out["provenance"][nm] = "null"
                out.setdefault("null_reasons", {})[nm] = "S7: " + str(v.get("reason"))
                continue
            p = np.asarray(v["xyz"], float)
            out["joints_scan"][nm] = p.tolist()
            out["joints_canonical"][nm] = (R_out @ (p - org)).tolist()
            out["confidence"][nm] = round(float(v["conf"]) * lc * gf, 3)
            tag = {"fe": "fold apex", "pt": "farthest tip", "an_2": "strongest elbow bend"}.get(j, "re-placed")
            if "extend" in rec and not (j == "fe" and rec.get("fold", {}).get("applied")):
                tag = "re-placed on the distally extended centreline"
            out["provenance"][nm] = (
                f"s7 {tag} ({v.get('provenance')})"
                if j in corrected
                else f"s7 re-placed on the corrected centreline ({v.get('provenance')})"
            )
            out.setdefault("joint_detail", {}).setdefault(nm, {})["s7"] = {
                "along_conf": v.get("conf"),
                "arc_pct": v.get("arc_pct"),
                "start_joint": v.get("start_joint"),
                "s6_xyz": (prop["joints_scan"] or {}).get(nm),
            }
            if out.get("null_reasons", {}).get(nm):
                out["null_reasons"].pop(nm, None)
    out["method"] = prop.get("method", "") + f" | {VERSION}: folds/tips/elbows ({','.join(classes)}; replace={replace})"
    out["s7"] = {
        "version": VERSION,
        "classes": list(classes),
        "replace": replace,
        "changes": report,
        "seconds": round(time.time() - t0, 1),
    }
    missing = sorted(
        k
        for k, v in out["joints_scan"].items()
        if v is None and not k.startswith("w_") and k not in ("head_front", "gaster_tip")
    )
    out.setdefault("completeness", {})["missing_joints"] = missing
    ld = os.path.join(out_dir, "limbs", name)
    os.makedirs(ld, exist_ok=True)
    lj2["version"] = lj.get("version", "") + f" | {VERSION}"
    json.dump(lj2, open(os.path.join(ld, "limbs.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(ld, "limbs.npz"), **lz2)
    pd = os.path.join(out_dir, "proposals", name)
    os.makedirs(pd, exist_ok=True)
    json.dump(out, open(os.path.join(pd, f"{name}_joints.json"), "w"), indent=1, default=float)
    return out["s7"]


def main():
    import argparse
    import json
    import traceback

    ap = argparse.ArgumentParser(description="S7 folds/tips/elbows correction of S6 products")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--names-file", default=None)
    ap.add_argument("--s6-work", required=True, help="S6 work dir with trunk/, limbs/, proposals/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mesh-tmpl", required=True, help="envelope mesh path with {name}")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--classes", default=",".join(CLASSES_ALL))
    ap.add_argument("--replace", default="corrected", choices=["corrected", "limb"])
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    names = list(a.names)
    if a.names_file:
        names += [x.strip() for x in open(a.names_file) if x.strip()]
    for n in names:
        f = os.path.join(a.out, "proposals", n, f"{n}_joints.json")
        if a.skip_existing and os.path.isfile(f):
            continue
        try:
            r = run(n, a.s6_work, a.mesh_tmpl.format(name=n), a.out, a.loso, tuple(a.classes.split(",")), a.replace)
            print(
                json.dumps(
                    {
                        "name": n,
                        "status": "ok",
                        "seconds": r["seconds"],
                        "changed": {
                            k: v["corrected"]
                            for k, v in r["changes"].items()
                            if isinstance(v, dict) and v.get("corrected")
                        },
                    }
                ),
                flush=True,
            )
        except Exception as e:
            print(
                json.dumps(
                    {
                        "name": n,
                        "status": "failed",
                        "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
