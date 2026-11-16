"""UC1 unposed bundle for the Blender add-on, with an ATTACHED fill.

The add-on ignores a validity mask, so every vertex that is not measurable has to be filled before the bundle is
written. The earlier fill (rs.export.make_bundle: mirror vertex, else the bundle mean, both
in the absolute model frame) had three faults:
  1. a registered vertex at which the unposing falls back to a single bone can stick out of its neighbours;
  2. filled parts sit at the mean or mirror position and are not attached to the specimen's own registered parent:
     steps, flaps and stretched edges at the seams;
  3. no cap on the size of a filled part.
This module fills BONE BY BONE, parents first, in the rest frame (UC1: template rest directions, the specimen's own
lengths and girths, model units):
  source    the specimen's own mirror part when that is measurable (code 5), else the mean part of the shipped
            POPULATION reference (code 6). Every row of the bundle therefore depends on its own specimen only: the
            same specimen gets the same row in any batch. Only without a population reference (`population='none'`)
            the mean is taken over the bundle's own specimens (fewer than MIN_MEAN of them for a part: the template)
            and the rows then depend on the batch
  attach    the part is translated so that the centroid of its SEAM RING in the source (the vertices of the
            neighbouring ancestor part that share an edge with the part) lies on the same ring of the specimen's own,
            already final, mesh: the part hangs on the specimen's own parent joint
  size      the part keeps the size of its source (the mirror part is the specimen's own; rest meshes are in model
            units, so a mean part has the bundle's mean proportions). Scaling a mean part to the specimen's seam ring
            was tried and dropped: it lengthened thin parts by up to a half. The caps are the ratio gate
            below (a limb bone outside [1/g, g] of the mean length is never kept, whatever its source) and BRIDGE_CAP
  bridge    a filled part between its registered parent and ONE registered child part is stretched along the line
            between the two seam rings (factor capped to [1 / BRIDGE_CAP, BRIDGE_CAP]) so that it meets both
  stick-out a vertex whose offset from its ring of neighbours differs from the template's by more than
            RING_OFFSET_BL of the rest body length (in practice: single-bone fallback vertices) is put back on its
            ring (code 6); rule adopted in production
  gate      R2-SHAPE's ratio gate (a limb bone whose add-on length ratio lies outside [1/g, g] is filled with its
            descendants); the ratio is taken against the population reference's mean mesh (without a population
            reference: against the batch mean, and only for batches of two or more)
`fill='absolute'` reproduces the earlier fill (for the before / after measurement).
"""

import json
import os

import numpy as np

MIN_MEAN = 5
BRIDGE_CAP = 2.0
RING_OFFSET_BL = 0.01
MX = np.array([1.0, -1.0, 1.0])
FILL_CODES = (3, 4, 5, 6, 7)
POPULATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rest_population.json")
_S = {}
_P = {}


def population(path=None):
    """the truth-free population reference (default: the shipped one; a text file so that
    it can travel in a patch) or None ('none', or no such file): dict(mean (nV, 3), mean_n (nV,), edge_med (nE,),
    edge_n (nE,))"""
    if path == "none":
        return None
    path = path or POPULATION
    if path not in _P:
        if os.path.isfile(path):
            d = json.load(open(path))
            _P[path] = {
                k: np.asarray(d[k], float if k in ("mean", "edge_med") else int)
                for k in ("mean", "mean_n", "edge_med", "edge_n")
            }
        else:
            _P[path] = None
    return _P[path]


def static(t):
    """edge lists, vertex rings and seam rings of the template (computed once)."""
    key = id(t)
    if key in _S:
        return _S[key]
    F = t.F
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    nb = [[] for _ in range(t.nV)]
    for a, b in E:
        nb[a].append(int(b))
        nb[b].append(int(a))
    anc = [set() for _ in range(t.nJ)]
    for j in range(1, t.nJ):
        p = int(t.parents[j])
        anc[j] = anc[p] | {p}
    desc = [[j] for j in range(t.nJ)]
    for j in range(t.nJ - 1, 0, -1):
        desc[int(t.parents[j])] += desc[j]
    dom = t.dom
    verts = [np.where(dom == j)[0] for j in range(t.nJ)]
    da, db = dom[E[:, 0]], dom[E[:, 1]]
    up = [[] for _ in range(t.nJ)]  # seam ring towards the ancestors (vertices of ancestor parts next to the part)
    rim_up = [[] for _ in range(t.nJ)]  # the part's own vertices next to that ring
    down = [dict() for _ in range(t.nJ)]  # child bone -> seam ring vertices in that child's subtree
    rim_down = [dict() for _ in range(t.nJ)]  # child bone -> the part's own vertices next to that ring
    sub_of = {}
    for j in range(t.nJ):
        for c in t.children[j]:
            for q in desc[c]:
                sub_of[(j, q)] = c
    for (a, b), ja, jb in zip(E, da, db):
        if ja == jb:
            continue
        for u, v, ju, jv in ((int(a), int(b), int(ja), int(jb)), (int(b), int(a), int(jb), int(ja))):
            # u (dominant bone ju) is a neighbour of v, a vertex of bone jv
            if ju in anc[jv]:
                up[jv].append(u)
                rim_up[jv].append(v)
            elif (jv, ju) in sub_of:
                c = sub_of[(jv, ju)]
                down[jv].setdefault(c, []).append(u)
                rim_down[jv].setdefault(c, []).append(v)
    up = [np.unique(np.array(u, int)) for u in up]
    rim_up = [np.unique(np.array(u, int)) for u in rim_up]
    down = [{c: np.unique(np.array(v, int)) for c, v in d.items()} for d in down]
    rim_down = [{c: np.unique(np.array(v, int)) for c, v in d.items()} for d in rim_down]
    e0 = np.linalg.norm(t.V0[E[:, 0]] - t.V0[E[:, 1]], axis=1)
    ring0 = np.stack([t.V0[i] - t.V0[nb[i]].mean(0) for i in range(t.nV)])
    limb = np.array([n.startswith(("l_", "an_", "ma_")) for n in t.names])
    trunk = np.array([n.startswith("b_") for n in t.names])
    _S[key] = dict(
        E=E,
        nb=nb,
        anc=anc,
        desc=desc,
        verts=verts,
        up=up,
        rim_up=rim_up,
        down=down,
        rim_down=rim_down,
        e0=e0,
        ring0=ring0,
        limb=limb,
        trunk=trunk,
    )
    return _S[key]


def ring_offset_excess(t, X):
    """|offset of each vertex from the centroid of its neighbours - the template's offset| (model units)."""
    S = static(t)
    off = np.stack([X[i] - X[S["nb"][i]].mean(0) for i in range(t.nV)])
    return np.linalg.norm(off - S["ring0"], axis=1)


def _rot_between(a, b):
    a = a / max(np.linalg.norm(a), 1e-30)
    b = b / max(np.linalg.norm(b), 1e-30)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def fill_one(t, X, bone_ok, mean, mean_n, info=None):
    """X (nV, 3) UC1 rest mesh of one specimen (model units), bone_ok (nJ,) bool: the bone's own vertices are kept;
    mean (nV, 3) / mean_n (nV,): mean rest position per vertex (the population reference's, or the bundle's own) and
    the number of specimens behind it. Returns (filled mesh, kind per vertex: 0 kept / 5 mirror / 6 mean or template).

    Anchors use the part's own RIM (its vertices next to the seam ring) in the source and the seam RING of the
    specimen's own final mesh, with the template's rim-to-ring offset between them: only vertices that are valid in
    the source and final in the specimen are read."""
    S = static(t)
    out = X.copy()
    kind = np.zeros(t.nV, np.int8)
    V0 = t.V0
    for j in range(t.nJ):  # parents precede children in OmniAnt's joint order
        vj = S["verts"][j]
        if bone_ok[j] or len(vj) == 0:
            continue
        m = int(t.mirror_j[j])
        if m != j and bone_ok[m]:
            k = 5

            def pos(idx, X=X):
                return X[t.mirror_v[idx]] * MX
        elif mean_n[vj].min() >= MIN_MEAN:
            k = 6

            def pos(idx, mean=mean):
                return mean[idx]
        else:
            k = 6

            def pos(idx):
                return V0[idx]

        src = pos(vj)
        ring, rim = S["up"][j], S["rim_up"][j]
        new = src
        if len(ring) >= 3 and len(rim) >= 3:
            c_src = pos(rim).mean(0)
            c_own = out[ring].mean(0) + (
                V0[rim].mean(0) - V0[ring].mean(0)
            )  # the ring's centroid + the template's ring-to-rim offset
            new = src - c_src + c_own
            # bridge: exactly one child part, and it is kept -> meet its seam ring as well
            kids = list(S["down"][j])
            if (
                len(kids) == 1
                and bone_ok[kids[0]]
                and len(S["down"][j][kids[0]]) >= 3
                and len(S["rim_down"][j][kids[0]]) >= 3
            ):
                rd, rmd = S["down"][j][kids[0]], S["rim_down"][j][kids[0]]
                b0 = pos(rmd).mean(0) - c_src  # source: proximal rim -> distal rim
                b1 = X[rd].mean(0) + (V0[rmd].mean(0) - V0[rd].mean(0)) - c_own  # own: where the distal rim has to be
                l0, l1 = np.linalg.norm(b0), np.linalg.norm(b1)
                if l0 > 1e-9 and l1 > 1e-9:
                    lam = float(np.clip(l1 / l0, 1.0 / BRIDGE_CAP, BRIDGE_CAP))
                    u = b0 / l0
                    d = new - c_own
                    d = d + np.outer((d @ u) * (lam - 1.0), u)  # stretch along the rim-to-rim line
                    new = d @ _rot_between(u, b1 / l1).T + c_own
                    if info is not None:
                        info.setdefault("bridged", []).append(
                            dict(bone=t.names[j], stretch=float(l1 / l0), capped=bool(abs(lam - l1 / l0) > 1e-12))
                        )
        out[vj] = new
        kind[vj] = k
    return out, kind


def bone_ratios(t, V, ref=None, ref_mesh=None):
    """The add-on's per-bone length ratios as R2-SHAPE defined them, copied: raw = mean over children
    of |child - joint| relative to a reference mesh, J_regressor pivots; hierarchical = raw / product of the
    ancestors'. ref_mesh (nV, 3): the reference (the population mean); else the mean mesh of the specimens ref (n,)
    bool (default: all)."""
    J = np.einsum("jv,nvc->njc", t.Jreg, V)
    if ref_mesh is not None:
        Jm = t.Jreg @ ref_mesh
    else:
        Jm = t.Jreg @ (V.mean(0) if ref is None or not np.any(ref) else V[ref].mean(0))
    raw = np.ones(J.shape[:2])
    for j in range(1, t.nJ):
        kids = t.children[j]
        if kids:
            raw[:, j] = np.mean(
                [np.linalg.norm(J[:, c] - J[:, j], axis=1) / max(np.linalg.norm(Jm[c] - Jm[j]), 1e-6) for c in kids], 0
            )
    hier = np.ones_like(raw)
    for j in range(1, t.nJ):
        cum = np.ones(len(V))
        p = t.parents[j]
        while p > 0:
            cum = cum * hier[:, p]
            p = t.parents[p]
        hier[:, j] = raw[:, j] / cum
    return raw, hier


def _absolute_fill(t, V0, ok):
    """the earlier fill (rs.export._fill_tracked): mirror vertex when valid, else the bundle mean; absolute frame."""
    Vm = V0[:, t.mirror_v] * MX
    use_m = ~ok & ok[:, t.mirror_v]
    V = np.where(use_m[:, :, None], Vm, V0)
    ok2 = ok | use_m
    mean = (V * ok2[:, :, None]).sum(0) / np.maximum(ok2.sum(0), 1)[:, None]
    out = np.where(ok2[:, :, None], V, mean[None])
    kind = np.zeros(ok.shape, np.int8)
    kind[use_m] = 5
    kind[~ok2] = 6
    return out, kind


def edge_stats(t, V, kind, ref_len=None):
    """per specimen: seam edges (exactly one filled end) and all edges against a reference length of the same edge
    (ref_len: the batch median when given, else the template). Lengths in units of the template rest body length."""
    S = static(t)
    E = S["E"]
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    ref = S["e0"] if ref_len is None else ref_len
    ratio = L / np.maximum(ref, 1e-9)
    fa, fb = kind[E[:, 0]] > 0, kind[E[:, 1]] > 0
    seam = fa != fb
    bl = t.rest_bl
    abn = (ratio > 8.0) & (L > 0.015 * bl)
    return dict(
        n_seam=int(seam.sum()),
        seam_ratio_med=float(np.median(ratio[seam])) if seam.any() else None,
        seam_ratio_p90=float(np.percentile(ratio[seam], 90)) if seam.any() else None,
        seam_ratio_max=float(ratio[seam].max()) if seam.any() else None,
        seam_gt3=int((ratio[seam] > 3.0).sum()),
        seam_len_max_bl=float(L[seam].max() / bl) if seam.any() else None,
        n_abnormal=int(abn.sum()),
        n_abnormal_gt10=int((abn & (L > 0.10 * bl)).sum()),
        n_abnormal_seam=int((abn & seam).sum()),
        n_abnormal_filled=int((abn & fa & fb).sum()),
        n_abnormal_kept=int((abn & ~fa & ~fb).sum()),
        edge_len_max_bl=float(L.max() / bl),
        edge_ratio_max=float(ratio.max()),
    )


def make_bundle(
    items,
    out_path=None,
    fill="attached",
    fill_codes=FILL_CODES,
    ratio_gate=3.0,
    ring_offset_bl=RING_OFFSET_BL,
    version="",
    use_population=True,
    population_path=None,
):
    """items: list of dict(name, V_scan (nV, 3) um, vertex_status (nV,) codes[, in_mean bool]). Returns (bundle dict,
    per-specimen QA). Writes the add-on bundle when out_path is given (verts, labels; the other keys are ignored by the
    add-on). With the population reference (the default) every row depends on its own specimen only. Without it
    (use_population False or population_path 'none') the mean a part is filled from and the reference of the ratio
    gate are the batch's own; specimens with in_mean False (do_not_use) then do not enter them."""
    from fitter_3d.antscan_registration.parts import uc1_unpose as U

    t = U.template()
    S = static(t)
    pop = population(population_path) if use_population else None
    if pop is not None and (len(pop["mean"]) != t.nV or len(pop["edge_med"]) != len(S["E"])):
        raise ValueError("the population reference does not belong to this model (vertex or edge count differs)")
    rows = []
    for it in items:
        vs = np.asarray(it["vertex_status"], int)
        r = U.unpose(it["V_scan"], np.isin(vs, (0, 1, 2, 3, 7)), t=t)
        r["name"], r["vs"] = it["name"], vs
        rows.append(r)
    n = len(rows)
    in_mean = np.array([bool(it.get("in_mean", True)) for it in items])
    V0 = np.stack([r["V_rest"] for r in rows])
    okv = np.stack([~np.isin(r["vs"], fill_codes) for r in rows])
    fb = np.stack([r["fallback"] for r in rows])
    thr = ring_offset_bl * t.rest_bl
    stuck = np.zeros_like(okv)
    if fill == "attached" and ring_offset_bl:
        for i in range(n):
            ex = ring_offset_excess(t, V0[i])
            stuck[i] = (ex > thr) & okv[i] & fb[i]
    ok0 = okv.copy()
    gated = np.zeros_like(okv)

    def bones_of(okm):
        return np.stack(
            [[bool(okm[i, S["verts"][j]].all()) if len(S["verts"][j]) else True for j in range(t.nJ)] for i in range(n)]
        )

    def run_fill(okm):
        if fill == "absolute":
            return _absolute_fill(t, V0, okm) + ([{} for _ in range(n)],)
        if pop is not None:  # the population mean: the same for every batch
            mean, cnt = pop["mean"], pop["mean_n"]
        else:
            okk = okm & ~stuck & in_mean[:, None]
            w = okk[:, :, None].astype(float)
            cnt = okk.sum(0)
            mean = (V0 * w).sum(0) / np.maximum(cnt, 1)[:, None]
        bok = bones_of(okm)
        Vf, kind, infos = np.empty_like(V0), np.zeros(okm.shape, np.int8), []
        for i in range(n):
            info = {}
            Vf[i], kind[i] = fill_one(t, V0[i], bok[i], mean, cnt, info)
            if stuck[i].any():
                idx = np.where(stuck[i])[0]
                for v in idx:
                    Vf[i, v] = Vf[i, S["nb"][v]].mean(0) + S["ring0"][v]
                kind[i, idx] = 6
            infos.append(info)
        return Vf, kind, infos

    V, kind, infos = run_fill(okv)
    gate_ref = None if pop is None else pop["mean"]
    if ratio_gate and ratio_gate > 0 and (n > 1 or gate_ref is not None):
        for _ in range(4):
            raw, hier = bone_ratios(t, V, in_mean, gate_ref)
            bad = ((raw > ratio_gate) | (raw < 1 / ratio_gate) | (hier > ratio_gate) | (hier < 1 / ratio_gate)) & S[
                "limb"
            ][None]
            if not bad.any():
                break
            for i, j in zip(*np.where(bad)):
                msk = np.isin(t.dom, S["desc"][j])
                gated[i] |= msk & okv[i]
                okv[i] &= ~msk
            V, kind, infos = run_fill(okv)
    status = np.stack([r["vs"] for r in rows]).astype(np.int8)
    status = np.where(kind > 0, kind, status).astype(np.int8)
    E = S["E"]
    if pop is not None:
        ref, ref_name = np.asarray(pop["edge_med"], float), "population"
    elif n >= 20:
        ref, ref_name = np.median(np.linalg.norm(V[:, E[:, 0]] - V[:, E[:, 1]], axis=2), axis=0), "batch median"
    else:
        ref, ref_name = None, "template"
    qa = []
    trunk_v = S["trunk"][t.dom]
    for i, r in enumerate(rows):
        q = edge_stats(t, V[i], kind[i], ref)
        q.update(
            name=r["name"],
            n_filled=int((kind[i] > 0).sum()),
            n_mirror=int((kind[i] == 5).sum()),
            n_mean=int((kind[i] == 6).sum()),
            n_gated=int(gated[i].sum()),
            gated_bones=[t.names[j] for j in range(t.nJ) if len(S["verts"][j]) and gated[i, S["verts"][j]].any()],
            n_fallback=int(fb[i].sum()),
            n_stuck=int(stuck[i].sum()),
            n_trunk_filled=int(((kind[i] > 0) & trunk_v).sum()),
            um_per_model_unit=float(r["s"]),
            roundtrip_max_um=float(r["roundtrip_max_um"]),
            bridged=infos[i].get("bridged", []),
            edge_reference=ref_name,
        )
        qa.append(q)
    bundle = dict(
        verts=V.astype(np.float32),
        labels=np.array([r["name"] for r in rows]),
        valid=okv & ~stuck,
        measurable=ok0,
        filled=kind > 0,
        gated=gated,
        vertex_status=status,
        fallback=fb,
        stuck=stuck,
        in_mean=in_mean,
        um_per_model_unit=np.array([r["s"] for r in rows]),
        roundtrip_max_um=np.array([r["roundtrip_max_um"] for r in rows]),
        convention=U.CONVENTION,
        fill=f"{fill}: codes {list(fill_codes)} filled bone by bone (mirror part, else "
        f"{'population mean' if pop is not None else 'bundle mean'}), attached at the seam ring; bridge cap "
        f"{BRIDGE_CAP}, ring offset {ring_offset_bl} BL, ratio gate {ratio_gate} against the "
        f"{'population mean' if pop is not None else 'bundle mean'}"
        if fill == "attached"
        else f"absolute mirror_mean of codes {list(fill_codes)}; ratio_gate {ratio_gate}",
        version=version,
    )
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not np.isfinite(bundle["verts"]).all():
            raise ValueError("non-finite vertices in the unposed bundle")
        tmp = f"{out_path}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **bundle)
        with np.load(tmp, allow_pickle=True) as b:
            if not np.array_equal(b["verts"], bundle["verts"]):
                raise ValueError("the unposed bundle does not read back as written")
        os.replace(tmp, out_path)
        summ = dict(
            n=n,
            fill=str(bundle["fill"]),
            filled_median=float(np.median([q["n_filled"] for q in qa])),
            filled_max=int(max(q["n_filled"] for q in qa)),
            gated_specimens=int(sum(q["n_gated"] > 0 for q in qa)),
            stuck_vertices=int(sum(q["n_stuck"] for q in qa)),
            fallback_max=int(max(q["n_fallback"] for q in qa)),
            specimens_with_abnormal_edge=int(sum(q["n_abnormal"] > 0 for q in qa)),
            specimens_with_abnormal_edge_gt10=int(sum(q["n_abnormal_gt10"] > 0 for q in qa)),
            roundtrip_max_um=float(np.nanmax([q["roundtrip_max_um"] for q in qa])),
            rest_extent_median=float(np.median([np.ptp(v, 0).max() for v in V])),
            template_extent=float(np.ptp(t.V0, 0).max()),
            per_specimen=qa,
        )
        with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
            json.dump(summ, f, indent=1)
    return bundle, qa
