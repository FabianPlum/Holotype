"""Status, morphometrics and export of the final registrations of a batch (the STATUS part, fitter_3d/antscan_registration/parts/rs, driven here).

Passes: (1) per specimen: evidence (rs.status.evidence_one, cached in OUT/evidence with the signature of what it was
made from); (2) decision at segment / bone / vertex / chain level (rs.status.decide_one), then per specimen the
coverage of the scan by the trunk parts given that decision (antscan_registration.checks.trunk_cover), then the decision again
with it; (3) per specimen: UC1 rest mesh, morphometrics, stage row with the registration's native parameters (rs),
tube and head checks (antscan_registration.checks); (4) unposed bundle with the attached fill (antscan_registration.fill), the specimen class
with the model grade (antscan_registration.grade), csv files, the fitter_3d stage .npz, one summary table; (5) per specimen:
overlay. Every per-specimen result depends on that specimen only, except `pop_z` in the status json (a robust z of
each segment length against the other specimens of the batch: information, never part of a decision).
A failure on a specimen is a row with its error and the class do_not_use.
"""

import csv
import json
import os
import time
import traceback

import numpy as np

from . import VERSION, setup
from . import steps as ST

SUMMARY_COLS = [
    "name",
    "specimen_class",
    "class_reasons",
    "status_class",
    "registration_status",
    "n_measurable",
    "n_review",
    "n_unregistered",
    "legs_measurable",
    "antennae_measurable",
    "mandibles_measurable",
    "trunk_not_measurable",
    "uncovered_mesosoma",
    "uncovered_gaster",
    "n_segments_filled",
    "head_end",
    "product_type",
    "conf_cut",
    "n_raised_by_majority",
    "n_ties",
    "n_far_ties_demoted",
    "n_lost_joints",
    "n_recovered",
    "n_mandible_recovered",
    "second_registration",
    "registration_source",
    "lbs_energy_share",
    "deform_med_pct_bl",
    "deform_p90_pct_bl",
    "native_forward_err_um",
    "stage_weight_rows_um",
    "n_filled_vertices",
    "n_trunk_filled",
    "n_abnormal_edges",
    "n_stretched_edges",
    "seam_ratio_max",
    "antenna_zigzag_max_deg",
    "head_roll_best_deg",
    "head_roll_contrast",
    "body_length_um",
    "seconds_reg1",
    "seconds_recover",
    "seconds_reg2",
    "seconds_status",
    "error",
]


ERROR_KEYS = (
    "evidence_error",
    "decision_error",
    "cover_error",
    "post_error",
    "checks_error",
    "bundle_error",
    "overlay_error",
)


def _env_tmpl(env, names):
    if "{n}" in env or "{name}" in env:
        return env
    for n in names:
        if os.path.isfile(f"{env}/{n}/{n}_envelope.stl"):
            return env + "/{n}/{n}_envelope.stl"
        if os.path.isfile(f"{env}/{n}_envelope.stl"):
            return env + "/{n}_envelope.stl"
    return env + "/{n}/{n}_envelope.stl"


def _ev_one(job):
    """evidence of one specimen. Cached evidence is reused only when its signature (registration, joint product, limb
    product, envelope, model, pipeline version) is the one of this run; without a signature (the function called on a
    folder of other origin): when it is not older than its registration and joint product."""
    name, A, sig = job
    from fitter_3d.antscan_registration.parts.rs import run as RR

    t0 = time.time()
    if sig is not None:  # the evidence holds absolute paths: it belongs to this output folder where it is now
        sig = f"{sig}:{os.path.abspath(A['out'])}"
    p = os.path.join(A["out"], "evidence", f"{name}.json")
    sp = os.path.join(A["out"], "evidence", f"{name}.sig")
    if os.path.isfile(p) and not A.get("force"):
        if sig is not None:
            stale = ST.read_sig(sp) != sig
        else:
            reg = os.path.join(A["regs"], f"{name}.npz")
            jf = ST.find_joint_file(A["joints_dir"], name)
            newest = max([os.path.getmtime(x) for x in (reg, jf) if x and os.path.isfile(x)] or [0])
            stale = os.path.getmtime(p) < newest
        if stale:
            os.remove(p)
    n, st, err = RR._ev_one((name, A))
    if st == "ok":
        ST.write_sig(sp, sig)
    return n, st, err, time.time() - t0


def _ev_fail(job, msg):
    return job[0], "error", msg, 0.0


def _cover_one(job):
    """coverage of the scan surface by the trunk parts, given the vertex status of a first decision"""
    name, A, vs, tol = job
    try:
        from fitter_3d.antscan_registration.parts import cover04 as CV
        from fitter_3d.antscan_registration.parts import uc1_unpose as U
        from fitter_3d.antscan_registration.parts.rs import io as rio
        from fitter_3d.antscan_registration.parts.skelreg import geom

        from . import checks as CK

        jf = ST.find_joint_file(A["joints_dir"], name)
        V = rio.read_registration(A["regs"], name, "route", jfile=jf)["V"]
        Vs, Fs = geom.load_mesh(rio.envelope(A["env_tmpl"], name))
        BL = float(json.load(open(jf))["body_length_um"])
        ld = rio.limbs_dir(A["limbs_dir"], name) if A.get("limbs_dir") else None
        limbs = CV.load_limbs(ld) if ld else None
        return name, CK.trunk_cover(U.template(), V, np.asarray(vs), Vs, Fs, BL, tol, limbs), None
    except Exception as e:
        return name, None, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-500:]}"


def _cover_fail(job, msg):
    return job[0], None, msg


def _post_one(job):
    """UC1 + morphometrics + stage row (rs.run._post_one) + the tube and head checks."""
    name, A, dec_path = job
    from fitter_3d.antscan_registration.parts.rs import io as rio
    from fitter_3d.antscan_registration.parts.rs import run as RR

    t0 = time.time()
    res = RR._post_one((name, dict(A, no_overlay=True), dec_path))
    if "error" in res:
        return res
    try:
        from fitter_3d.antscan_registration.parts import uc1_unpose as U
        from fitter_3d.antscan_registration.parts.skelreg import geom

        from . import checks as CK

        t = U.template()
        D = json.load(open(dec_path))
        V = rio.read_registration(A["regs"], name, "route", jfile=D["jfile"])["V"]
        Vs, _ = geom.load_mesh(rio.envelope(A["env_tmpl"], name))
        res["checks"] = dict(tubes=CK.tube_shape(t, V), head_roll=CK.head_roll(t, V, Vs, D["BL"]))
    except Exception as e:
        res["checks"] = None
        res["checks_error"] = f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-500:]}"
    res["seconds_post"] = time.time() - t0
    return res


def _post_fail(job, msg):
    return dict(name=job[0], error=msg)


def _overlay_fail(job, msg):
    return job[0], msg


def _overlay_one(job):
    name, A, dec_path = job
    try:
        from fitter_3d.antscan_registration.parts.rs import io as rio
        from fitter_3d.antscan_registration.parts.rs import overlay as OV

        D = json.load(open(dec_path))
        jf = D["jfile"]
        V = rio.read_registration(A["regs"], name, "route", jfile=jf)["V"]
        jd = json.load(open(jf))
        spec = dict(D["specimen"])
        failed = (spec.get("grade") or {}).get("failed") or []
        if spec["specimen_class"] == "partial" and failed:
            spec["specimen_class"] = "partial (not " + ", ".join(f.split("_", 1)[1] for f in failed) + ")"
        OV.render_one(
            name,
            V,
            np.asarray(D["part_code"], int),
            D["segments"],
            jd["frame"]["rotation"],
            rio.envelope(A["env_tmpl"], name),
            rio.limbs_dir(A["limbs_dir"], name),
            spec,
            os.path.join(A["out"], "overlays", f"{name}.png"),
            D["BL"],
        )
        return name, None
    except Exception as e:
        return name, f"{type(e).__name__}: {e}"


def mandible_codes(d, E, S):
    """A mandible whose base was accepted by antscan_registration.mandible (a recovered joint): `recovered` (2, measurable) when the
    base has TWO kinds of socket evidence (the first registration's marker, and the mirror image of the partner base;
    the partner may itself be a base below the cut, see antscan_registration.mandible), else `recovered_estimated` (7: registered,
    flagged) -- the rule STATUS applies to recovered limb joints (a single prediction is an estimate). Changes d in
    place."""
    t = S.tpl()
    rec = E.get("recovered_joints") or {}
    changed = False
    for pn in ("ma_l", "ma_r"):
        j = t.idx[pn]
        if pn in rec and int(d["part_code"][j]) == 0:
            ver = list(rec[pn].get("verified_by") or [])
            c = 2 if len(ver) >= 2 else 7
            d["part_code"][j] = c
            d["part_reasons"][pn] = [
                ("recovered" if c == 2 else "recovered_estimated") + " mandible base: " + ", ".join(ver)
            ]
            changed = True
    if changed:
        code = d["part_code"]
        d["vertex_status"] = code[t.dom].astype(np.int8)
        d["part_status"] = np.array(
            ["ok" if c in S.MEASURABLE else ("review" if c in (3, 7) else "unregistered") for c in code]
        )
        d["specimen"]["mandibles_measurable"] = int(sum(int(code[t.idx[m]]) in S.MEASURABLE for m in ("ma_l", "ma_r")))
    return d


def run(
    names,
    out,
    env,
    limbs,
    jobs=1,
    force=False,
    no_overlay=False,
    no_export=False,
    K=None,
    manifest=None,
    sigs=None,
    population=None,
    model=None,
):
    """status step on OUT/registration + OUT/joints (see the module docstring). sigs: {name: signature of the final
    registration and its inputs} for the evidence cache; population: path of the population reference of the unposed
    mesh, 'none', or None for the shipped one; manifest: {name: row} that is updated in place (when None, a
    manifest.jsonl is written here)."""
    model = setup(model)
    from fitter_3d.antscan_registration.parts import cover04 as CV
    from fitter_3d.antscan_registration.parts import partconf
    from fitter_3d.antscan_registration.parts.rs import export as X
    from fitter_3d.antscan_registration.parts.rs import io as rio
    from fitter_3d.antscan_registration.parts.rs import morpho as MO
    from fitter_3d.antscan_registration.parts.rs import run as RR
    from fitter_3d.antscan_registration.parts.rs import status as S

    from . import fill as FI
    from . import grade as GR

    t00 = time.time()
    K = K or {}
    Ks = dict(S.DEFAULT_K, **{k: v for k, v in K.items() if k in S.DEFAULT_K})
    Kg = {k: v for k, v in K.items() if k in GR.K_GRADE}
    for d in ("evidence", "status", "routes", "overlays", "export"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    regs, joints_dir = f"{out}/registration", f"{out}/joints"
    sigs = sigs or {}

    def pmap(fn, jobs_, fail):
        return ST.pool_map(fn, jobs_, jobs, ST.worker_init, (model, partconf.CACHE), fail)

    A = dict(
        out=out,
        regs=regs,
        reg_format="route",
        joints_dir=joints_dir,
        limbs_dir=limbs or "",
        env_tmpl=_env_tmpl(env, names),
        native=None,
        no_overlay=True,
        no_export=no_export,
        no_xsec=False,
        force=force,
    )
    man = manifest if manifest is not None else {n: dict(name=n) for n in names}
    for n in names:  # errors of an earlier status step do not carry over
        for k in ERROR_KEYS:
            man[n].pop(k, None)
    reginfo = {}
    for n in names:
        p = f"{regs}/{n}.json"
        q = json.load(open(p)) if os.path.isfile(p) else {}
        nat = q.get("native") or {}
        reginfo[n] = dict(
            status=q.get("status", "failed" if not os.path.isfile(f"{regs}/{n}.npz") else None),
            reasons=q.get("status_reasons") or [],
            error=q.get("error"),
            lbs_energy_share=nat.get("lbs_energy_share"),
            deform_med_bl=nat.get("deform_med_bl"),
            deform_p90_bl=nat.get("deform_p90_bl"),
            forward_err_um=nat.get("forward_plus_deform_max_err_um"),
            seconds=q.get("seconds"),
        )
    # ---- pass 1: evidence -------------------------------------------------------------------------------------------
    res = pmap(_ev_one, [(n, A, sigs.get(n)) for n in names], _ev_fail)
    tsec = {}
    for n, st, err, sec in res:
        man[n]["evidence"], man[n]["evidence_error"] = st, err
        tsec[n] = sec
    EV = {}
    for n in names:
        p = os.path.join(out, "evidence", f"{n}.json")
        if man[n]["evidence"] in ("ok", "cached") and os.path.isfile(p):
            EV[n] = S.load_evidence(p)
            if not EV[n].get("has_kink"):
                S.add_kink(EV[n], EV[n]["limbs_dir"])
    # ---- pass 2: decisions (segments, bones, vertices, chains; STATUS's own specimen class) ---------------------------
    dec0, bad_ev = {}, {}
    for n, E in EV.items():
        try:
            dec0[n] = S.decide_one(E, Ks)
        except Exception as e:
            bad_ev[n] = f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-600:]}"
    for n, err in bad_ev.items():
        man[n]["decision_error"] = err
        EV.pop(n)
    # coverage of the scan by the trunk parts, given which vertices the first decision claims (never cached)
    cov = pmap(
        _cover_one,
        [(n, A, dec0[n]["vertex_status"].tolist(), Ks["TRUNK_COVER_TOL_BL"]) for n in EV],
        _cover_fail,
    )
    COV = {}
    for n, c, err in cov:
        if err:
            man[n]["cover_error"] = err
        elif c is not None:
            COV[n] = c
            for g, bones in S.COVER_PARTS.items():
                for b in bones:
                    if b in EV[n]["parts"]:
                        EV[n]["parts"][b]["scan_uncovered"] = c[g]["uncovered"]
    pop = (
        RR.population_z(EV, {n: dec0[n]["segments"] for n in EV}) if len(EV) > 8 else {n: None for n in EV}
    )  # information only
    seg_bones = {sg["name"]: [S.tpl().names[b] for b in sg["own"]] for segs in CV.chains().values() for sg in segs}
    DEC, RAW = {}, {}
    for n, E in EV.items():
        try:
            d = mandible_codes(S.decide_one(E, Ks, pop=pop.get(n)), E, S)
            RAW[n] = d
            DEC[n] = dict(
                name=n,
                version=VERSION,
                status_version=S.VERSION,
                BL=E["BL"],
                jfile=E["jfile"],
                waist_segments=E.get("waist_segments"),
                specimen=d["specimen"],
                chains=d["chains"],
                segments={k: [int(v[0]), v[1]] for k, v in d["segments"].items()},
                part_code=d["part_code"].tolist(),
                part_status=d["part_status"].tolist(),
                part_reasons=d["part_reasons"],
                vertex_status=d["vertex_status"].tolist(),
                extra_flags=d["extra_flags"],
                pop_z=pop.get(n),
                trunk_cover=COV.get(n),
                K=Ks,
                evidence_summary={
                    s: {
                        k: e.get(k)
                        for k in (
                            "jsup",
                            "conf_b",
                            "conf_c",
                            "tube",
                            "chart_ok",
                            "on_own",
                            "on_other",
                            "asym",
                            "length",
                            "lim_conf",
                        )
                    }
                    for s, e in E["segments"].items()
                },
                inputs=dict(regs=regs, envelope=E["envelope"], limbs=E["limbs_dir"]),
            )
            with open(os.path.join(out, "status", f"{n}_status.json"), "w") as f:
                json.dump(S._jsonable(DEC[n]), f)
        except Exception as e:
            man[n]["decision_error"] = f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-600:]}"
    # ---- pass 3: per specimen morphometrics, stage rows, checks -----------------------------------------------------
    post = pmap(_post_one, [(n, A, os.path.join(out, "status", f"{n}_status.json")) for n in DEC], _post_fail)
    POST = {}
    for r in post:
        n = r["name"]
        if "error" in r:
            man[n]["post_error"] = r["error"]
        else:
            POST[n] = r
            tsec[n] = tsec.get(n, 0.0) + r.get("seconds_post", 0.0)
            if r.get("checks_error"):
                man[n]["checks_error"] = r["checks_error"]
    ordered = [n for n in names if n in POST]
    # ---- pass 4: unposed bundle with the attached fill; specimen class with the model grade --------------------------
    FQ, bundle = {}, None
    if ordered:
        items = []
        for n in ordered:
            V = rio.read_registration(regs, n, "route", jfile=DEC[n]["jfile"])["V"]
            items.append(
                dict(
                    name=n,
                    V_scan=V,
                    vertex_status=DEC[n]["vertex_status"],
                    in_mean=DEC[n]["specimen"]["specimen_class"] != "do_not_use",
                )
            )
        try:
            bundle, qa = FI.make_bundle(
                items,
                None if no_export else os.path.join(out, "export", "uc1_bundle.npz"),
                version=VERSION,
                population_path=population,
            )
            FQ = {q["name"]: q for q in qa}
        except Exception as e:  # no bundle: no specimen of the batch can be model grade
            bundle, FQ = None, {}
            for n in ordered:
                man[n]["bundle_error"] = f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-700:]}"
    for n in names:
        if n not in DEC:
            continue
        sp = DEC[n]["specimen"]
        if n not in POST and sp["specimen_class"] != "do_not_use":  # no morphometrics / export row: not usable
            sp["specimen_class"] = "do_not_use"
            sp["reasons"] = list(sp.get("reasons") or []) + [
                "error: " + str(man[n].get("post_error")).split(" | ")[0][:200]
            ]
        g = GR.decide(
            sp,
            DEC[n]["segments"],
            DEC[n]["part_code"],
            FQ.get(n),
            (POST.get(n) or {}).get("checks"),
            reginfo[n],
            Kg,
            seg_bones,
        )
        sp["status_class"] = sp["specimen_class"]
        sp["specimen_class"] = g["specimen_class"]
        sp["class_reasons"] = g["reasons"]
        sp["grade"] = g
        DEC[n]["fill"] = {k: v for k, v in (FQ.get(n) or {}).items() if k != "name"}
        DEC[n]["checks"] = (POST.get(n) or {}).get("checks")
        DEC[n]["registration"] = dict(reginfo[n], stage=(POST.get(n) or {}).get("stage_extra"))
        with open(os.path.join(out, "status", f"{n}_status.json"), "w") as f:
            json.dump(S._jsonable(DEC[n]), f)
        d = RAW[n]
        V = rio.read_registration(regs, n, "route", jfile=DEC[n]["jfile"])["V"]
        np.savez_compressed(
            os.path.join(out, "status", f"{n}_status.npz"),
            V=V,
            vertex_status=d["vertex_status"],
            part_code=d["part_code"],
            part_status=d["part_status"],
            markers=np.asarray(EV[n]["markers"]),
            part_names=np.array(S.tpl().names),
            status_names=np.array(S.STATUS_NAMES),
            specimen_class=np.array(g["specimen_class"]),
        )
        extra = {}
        try:  # the registration's global scale (um per model unit): the benchmark's exact face-area definition needs it
            with np.load(os.path.join(regs, f"{n}.npz"), allow_pickle=True) as zr:
                if "um_per_model_unit" in zr.files:
                    extra["um_per_model_unit"] = np.array(float(zr["um_per_model_unit"]))
        except Exception:
            pass
        np.savez_compressed(
            os.path.join(out, "routes", f"{n}.npz"),
            V_final=V,
            V_pre=V,
            reg_v=np.isin(d["vertex_status"], S.REGISTERED),
            vertex_status=d["vertex_status"],
            part_status=d["part_status"],
            status=np.array(GR.scorer_status(g["specimen_class"])),
            seconds=np.array(float(reginfo[n].get("seconds") or 0.0)),
            jfile=np.array(DEC[n]["jfile"]),
            **extra,
        )
        man[n]["specimen_class"] = g["specimen_class"]
    # the bundle carries the class; a second file holds the model-grade specimens only (ready for the add-on)
    if bundle is not None and not no_export:
        cls_b = np.array([DEC[n]["specimen"]["specimen_class"] for n in ordered])
        bp = os.path.join(out, "export", "uc1_bundle.npz")
        np.savez_compressed(bp, **bundle, specimen_class=cls_b)
        mg = cls_b == "model_grade"
        mp = os.path.join(out, "export", "uc1_bundle_model_grade.npz")
        if mg.any():
            sub = {
                k: (v[mg] if (isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == len(ordered)) else v)
                for k, v in bundle.items()
            }
            np.savez_compressed(mp, **sub, specimen_class=cls_b[mg])
        elif os.path.isfile(mp):
            os.remove(mp)
    # ---- csv files --------------------------------------------------------------------------------------------------
    cols = MO.columns()
    for fn, kind in (
        ("morphometrics.csv", "all"),
        ("morphometrics_flags.csv", "flags"),
        ("morphometrics_measurable.csv", "measurable"),
    ):
        with open(os.path.join(out, fn), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "specimen_class"] + cols)
            for n in ordered:
                row, fl = POST[n]["row"], POST[n]["flags"]
                cls = DEC[n]["specimen"]["specimen_class"]
                if kind == "all":
                    vals = [row.get(c, np.nan) for c in cols]
                elif kind == "flags":
                    vals = [3 if cls == "do_not_use" else fl.get(c, 2) for c in cols]
                else:
                    vals = [row.get(c, np.nan) if (fl.get(c, 2) == 0 and cls != "do_not_use") else np.nan for c in cols]
                w.writerow(
                    [n, cls]
                    + [
                        (
                            ""
                            if (isinstance(v, float) and not np.isfinite(v))
                            else (f"{v:.6g}" if isinstance(v, float) else v)
                        )
                        for v in vals
                    ]
                )
    # ---- fitter_3d stage file ---------------------------------------------------------------------------------------
    exp = {}
    if not no_export and ordered:
        rows, labels, srows = [], [], []
        for n in ordered:
            sp_ = os.path.join(out, "export", "stage_rows", f"{n}.npz")
            if not os.path.isfile(sp_):
                continue
            z = np.load(sp_)
            rows.append({k: z[k] for k in z.files})
            labels.append(n)
            D = DEC[n]
            srows.append(
                dict(
                    vertex_status=D["vertex_status"],
                    part_code=D["part_code"],
                    part_status=D["part_status"],
                    specimen_class=D["specimen"]["specimen_class"],
                    params_source=POST[n].get("params_source"),
                    provenance=dict(
                        pipeline_version=VERSION,
                        status_version=S.VERSION,
                        jfile=D["jfile"],
                        registration=reginfo[n].get("status"),
                        lbs_energy_share=reginfo[n].get("lbs_energy_share"),
                    ),
                )
            )
        if rows:
            exp["stage"] = X.write_stage(
                rows, labels, srows, os.path.join(out, "export", "batch_0", "Stage_status.npz")
            )
            exp["stage"]["n"] = len(rows)
        if bundle is not None:
            exp["uc1_bundle"] = dict(
                n=len(ordered),
                filled_median=float(np.median([FQ[n]["n_filled"] for n in ordered])),
                specimens_with_abnormal_edge=int(sum(FQ[n]["n_abnormal_gt10"] > 0 for n in ordered)),
            )
    # ---- overlays ---------------------------------------------------------------------------------------------------
    if not no_overlay:
        ov = pmap(_overlay_one, [(n, A, os.path.join(out, "status", f"{n}_status.json")) for n in DEC], _overlay_fail)
        for n, err in ov:
            if err:
                man[n]["overlay_error"] = err

    # ---- the summary table ------------------------------------------------------------------------------------------
    def g(d, *ks):
        for k in ks:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        return d

    def f(v, nd=4):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.{nd}g}"
        return v

    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLS)
        for n in names:
            m = man[n]
            ri = reginfo[n]
            err = [
                str(m.get(k)).split(" | ")[0]
                for k in (
                    "evidence_error",
                    "decision_error",
                    "post_error",
                    "cover_error",
                    "checks_error",
                    "bundle_error",
                    "overlay_error",
                )
                if m.get(k)
            ]
            if ri.get("error"):
                err.insert(0, "registration: " + str(ri["error"]).split(" | ")[0])
            intake_block = {}
            jf = ST.find_joint_file(joints_dir, n)
            if jf:
                try:
                    jd = json.load(open(jf))
                    intake_block = jd.get("intake") or {}
                    rec = jd.get("recovery") or {}
                except Exception:
                    rec = {}
            else:
                rec = {}
            rj = rec.get("joints") or {}
            n_mand = sum(1 for k in rj if k.startswith("ma_"))
            lg = f"{out}/work/recover/logs/{n}_recovery.json"
            rlog = json.load(open(lg)) if os.path.isfile(lg) else {}
            n_lost = rlog.get("n_lost")
            r1 = f"{out}/work/reg1/{n}.json"
            sec1 = json.load(open(r1)).get("seconds") if os.path.isfile(r1) else None
            rsrc = (
                json.load(open(f"{regs}/{n}.json")).get("registration_source")
                if os.path.isfile(f"{regs}/{n}.json")
                else None
            )
            changed = rsrc == "reg2" if rsrc else bool(len(rj))
            sec2 = ri.get("seconds") if changed else None
            st_x = (POST.get(n) or {}).get("stage_extra") or {}
            sc_um = None
            if st_x.get("weight_rows_max") is not None and os.path.isfile(f"{out}/export/stage_rows/{n}.npz"):
                with np.load(f"{out}/export/stage_rows/{n}.npz") as zs:
                    sc_um = float(st_x["weight_rows_max"]) * float(zs["target_scale"])
            if n in DEC:
                sp = DEC[n]["specimen"]
                gr = sp["grade"]
                fq = FQ.get(n) or {}
                row = dict(
                    name=n,
                    specimen_class=sp["specimen_class"],
                    class_reasons=" | ".join(sp["class_reasons"]),
                    status_class=sp["status_class"],
                    registration_status=ri.get("status"),
                    n_measurable=sp["n_measurable"],
                    n_review=sp["n_review"],
                    n_unregistered=sp["n_unregistered"],
                    legs_measurable=sp["legs_measurable"],
                    antennae_measurable=sp["antennae_measurable"],
                    mandibles_measurable=sp["mandibles_measurable"],
                    trunk_not_measurable=",".join(sp["trunk_not_measurable"]),
                    uncovered_mesosoma=g(COV.get(n), "mesosoma", "uncovered"),
                    uncovered_gaster=g(COV.get(n), "gaster", "uncovered"),
                    n_segments_filled=gr.get("n_segments_filled"),
                    head_end=sp["head_end"]["state"],
                    n_filled_vertices=fq.get("n_filled"),
                    n_trunk_filled=fq.get("n_trunk_filled"),
                    n_abnormal_edges=fq.get("n_abnormal_gt10"),
                    n_stretched_edges=fq.get("n_abnormal"),
                    seam_ratio_max=fq.get("seam_ratio_max"),
                    antenna_zigzag_max_deg=gr.get("antenna_zigzag_max_deg"),
                    head_roll_best_deg=g(gr, "head_roll", "best_deg"),
                    head_roll_contrast=g(gr, "head_roll", "contrast"),
                    body_length_um=DEC[n]["BL"],
                )
            else:
                row = dict(
                    name=n,
                    specimen_class="do_not_use",
                    class_reasons="error: " + ("; ".join(err) or "no registration")[:300],
                    registration_status=ri.get("status") or "failed",
                )
            row.update(
                product_type=intake_block.get("product_type"),
                conf_cut=intake_block.get("cut"),
                n_raised_by_majority=len(intake_block.get("raised_to_usable") or []) if intake_block else None,
                n_ties=len(intake_block.get("ties") or {}) if intake_block else None,
                n_far_ties_demoted=len(intake_block.get("tie_far") or {}) if intake_block else None,
                n_lost_joints=n_lost,
                n_recovered=len(rj) - n_mand if rec else 0,
                n_mandible_recovered=n_mand,
                second_registration=changed,
                registration_source=rsrc,
                stage_weight_rows_um=sc_um,
                lbs_energy_share=ri.get("lbs_energy_share"),
                deform_med_pct_bl=None if ri.get("deform_med_bl") is None else 100 * ri["deform_med_bl"],
                deform_p90_pct_bl=None if ri.get("deform_p90_bl") is None else 100 * ri["deform_p90_bl"],
                native_forward_err_um=ri.get("forward_err_um"),
                seconds_reg1=sec1,
                seconds_recover=rlog.get("seconds_total"),
                seconds_reg2=sec2,
                seconds_status=tsec.get(n),
                error="; ".join(err)[:400],
            )
            w.writerow([f(row.get(c)) for c in SUMMARY_COLS])
    if manifest is None:  # called on its own: the manifest is written here (antscan_registration.run writes its own)
        with open(os.path.join(out, "manifest.jsonl"), "w") as fh:
            for n in names:
                fh.write(json.dumps(man[n], default=ST._js) + "\n")
    cls = [DEC[n]["specimen"]["specimen_class"] if n in DEC else "do_not_use" for n in names]
    summ = dict(
        version=VERSION,
        status_version=S.VERSION,
        n=len(names),
        n_evidence=len(EV),
        n_decided=len(DEC),
        n_post=len(POST),
        classes={c: cls.count(c) for c in GR.CLASSES},
        status_classes={
            c: [DEC[n]["specimen"]["status_class"] if n in DEC else "do_not_use" for n in names].count(c)
            for c in ("complete", "partial", "do_not_use")
        },
        errors={
            n: [k for k in man[n] if k.endswith("_error") and man[n][k]]
            for n in names
            if any(k.endswith("_error") and man[n][k] for k in man[n])
        },
        export=exp,
        K_status=Ks,
        K_grade=dict(GR.K_GRADE, **Kg),
        population=("none" if population == "none" else (population or FI.POPULATION)),
        bundle_error=next(
            (man[n]["bundle_error"].split(" | ")[0] for n in ordered if man[n].get("bundle_error")), None
        ),
        seconds=time.time() - t00,
    )
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(S._jsonable(summ), fh, indent=1)
    print(
        "[antscan_registration] status:",
        json.dumps({k: summ[k] for k in ("n", "n_decided", "n_post", "classes", "status_classes")}),
        "errors:",
        len(summ["errors"]),
        flush=True,
    )
    return {k: summ[k] for k in ("n", "n_decided", "n_post", "classes", "status_classes")}
