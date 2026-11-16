"""Geometric registration core, one specimen: joint product + envelope -> registered template mesh in fitter_3d's own
parameters.

    register_specimen(name, env_path, jfile, limb_dir, ...) -> dict (route arrays + log), never raises for a bad
    specimen: a failure is `status = 'failed'` with the reason.

Steps: spec.Specimen (inputs in the normalised target frame) -> native.closed_form_start (closed form, native
parameters) -> surface.register (girth, tube warp, labels, coarse, fine; CPU by default) -> native parameters of the
result (refit.native_refit when asked, else the start's parameters) with deform_verts = the residual, so that
fitter_3d's forward(parameters) + deform_verts reproduces the registered mesh.

Nothing here reads truth, oracle limbs, specimen names (the name is a file key only) or taxonomy; no network, no AI
calls; every directory is an argument.
"""

import json
import time
import traceback

import numpy as np

from fitter_3d.antscan_common import json_default as _js

from . import native
from . import spec as SP
from .model import template

VERSION = "geom-1.1 (2026-11-13)"
STAGE_KEYS = (
    "global_rot",
    "joint_rot",
    "betas",
    "log_beta_scales",
    "trans",
    "deform_verts",
    "betas_trans",
    "log_scale",
    "rotation_mask",
)


def tip_targets(sp, min_conf=0.0):
    """tip targets of the one-joint parts (mandibles): the tip of the part's limb centreline, used only when the
    part's own joint is usable (the joint's confidence already contains the limb's confidence)."""
    return {j: q["tip"] for j, q in sp.mand.items() if sp.w[j] > 0 and q["conf"] >= min_conf}


def register_specimen(
    name,
    env_path,
    jfile,
    limb_dir=None,
    kind="medial",
    conf_min=0.3,
    opts=None,
    device="cpu",
    model_pkl=None,
    refit=True,
    scan=None,
):
    """Returns (arrays dict for np.savez, info dict). arrays is None when the specimen failed (info['status'] ==
    'failed', info['error'])."""
    t0 = time.time()
    info = {
        "name": name,
        "version": VERSION,
        "kind": kind,
        "conf_min": conf_min,
        "jfile": jfile,
        "envelope": env_path,
        "limb_dir": limb_dir,
        "device": device,
    }
    try:
        from . import surface

        one = str(device) == "cpu" and int((opts or {}).get("cpu_threads", surface.OPTS["cpu_threads"])) == 1
        with surface.single_thread(one):  # CPU: one thread for torch and BLAS, so that two runs are bit-identical
            return _register_specimen(
                name, env_path, jfile, limb_dir, kind, conf_min, opts, device, model_pkl, refit, scan, info, t0
            )
    except Exception as e:  # a failure is a status, never an exception that stops a batch
        info.update(
            status="failed",
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc()[-3000:],
            seconds=time.time() - t0,
        )
        return None, info


def _register_specimen(name, env_path, jfile, limb_dir, kind, conf_min, opts, device, model_pkl, refit, scan, info, t0):
    from . import surface

    tpl = template(model_pkl)
    sp = SP.Specimen(name, env_path, jfile, limb_dir=limb_dir, conf_min=conf_min, kind=kind, scan=scan, tpl=tpl)
    info["product_flags"] = sp.flags
    info["t_load"] = time.time() - t0
    if sp.BL is None:
        raise native.StartError("the joint file has no body_length_um")
    t1 = time.time()
    sto = dict((opts or {}).get("start") or {})
    use_tips = sto.pop("tips", True)  # ablation switch: mandible tips from the limb product
    st = native.closed_form_start(
        tpl,
        sp.P,
        sp.w,
        kind=kind,
        R_prior=np.eye(3),
        BL=sp.BL,
        head_front=sp.head_front,
        tips=tip_targets(sp) if use_tips else {},
        opts=sto,
    )
    p0 = native.to_native(st)
    info["start"] = st.info
    info["t_start"] = time.time() - t1
    sopts = {k: v for k, v in (opts or {}).items() if k != "start"}
    res = surface.register(sp, st, sopts, device=device)
    info["surface"] = res["info"]
    fr = sp.frame
    Vn = res["V"]
    # ---- native parameters of the result
    p = dict(p0)
    v0_lbs, piv0 = native.forward_np(tpl, p0, deform=False)
    if refit:
        from . import refit as RF

        use_b = bool(res["info"].get("shape_space_used"))  # the shape space was used: let betas carry it
        p, rinfo = RF.native_refit(tpl, Vn, sp.support[tpl.dom], p0, BL=sp.BL, use_betas=use_b)
        info["refit"] = rinfo
    v_lbs, piv = native.forward_np(tpl, p, deform=False)
    p["deform_verts"] = Vn - v_lbs
    chk = float(np.abs(native.forward_np(tpl, p)[0] - Vn).max())
    dn = np.linalg.norm(p["deform_verts"], axis=1)
    d0 = np.linalg.norm(Vn - st.vertices(), axis=1)
    sim = st.s * (tpl.V0 @ st.G[0].T) + (st.a[0] - st.s * st.G[0] @ st.ref0[0])  # mean template, root similarity only
    dsim = np.linalg.norm(Vn - sim, axis=1)
    supv = sp.support[tpl.dom]
    info["native"] = dict(
        forward_plus_deform_max_err_norm=chk,
        forward_plus_deform_max_err_um=chk * fr.s,
        deform_med_um=float(np.median(dn) * fr.s),
        deform_p90_um=float(np.percentile(dn, 90) * fr.s),
        deform_med_bl=float(np.median(dn) / sp.BL),
        deform_p90_bl=float(np.percentile(dn, 90) / sp.BL),
        start_to_final_med_bl=float(np.median(d0) / sp.BL),
        similarity_to_final_med_bl=float(np.median(dsim) / sp.BL),
        # share of the squared displacement (root-similarity template -> registered mesh) carried by the LBS
        # parameters, over the supported vertices: 1 - |deform|^2 / |final - similarity|^2
        lbs_energy_share=float(1.0 - (dn[supv] ** 2).sum() / max((dsim[supv] ** 2).sum(), 1e-30)),
        refit=bool(refit),
    )
    part_status = res["part_status"]
    status, reasons = surface.specimen_status(tpl, part_status, st, sp.flags)
    info["status_reasons"] = reasons
    to_s = fr.to_scan
    um = float(st.s * fr.s)
    arrays = dict(
        V_final=to_s(Vn),
        V_pre=to_s(res["V_S"]),
        V_S=to_s(res["V_S"]),
        V_B=to_s(res["V_B"]),
        V_T=to_s(res["V_T"]),
        V_C=to_s(res["V_C"]),
        reg_v=res["status"] != 2,
        vertex_status=res["status"].astype(np.int8),
        part_status=np.asarray(part_status).astype(str),
        status=status,
        seconds=time.time() - t0,
        jfile=jfile,
        um_per_model_unit=um,
        scan_code=res["scan_code"],
        support=sp.support,
        start_how=np.array(st.how).astype(str),
        start_pinned=st.pinned,
        V_final_norm=Vn.astype(np.float32),
        verts=(native.forward_np(tpl, p)[0]).astype(np.float32),
        pivots=piv.astype(np.float32),
        target_centre=fr.c,
        target_scale=np.float64(fr.s),
        target_units="micrometre",
        target_frame_rotation=fr.R,
        target_frame_origin=fr.o,
        body_length_um=np.float64(sp.BL_um),
        product_flags=np.array(sp.flags, dtype=str),
        status_reasons=np.array(reasons, dtype=str),
        version=VERSION,
        **{k: np.asarray(p[k], np.float32) for k in STAGE_KEYS},
        **{f"start_{k}": np.asarray(p0[k], np.float32) for k in STAGE_KEYS if k != "deform_verts"},
        start_pivots=piv0.astype(np.float32),
        start_verts=v0_lbs.astype(np.float32),
        start_sig=st.sig.astype(np.float32),
        start_psi=st.psi.astype(np.float32),
        **{f"diag_{k}": v.astype(np.float32) for k, v in res["bone_diag"].items()},
    )
    info.update(
        status=status, seconds=time.time() - t0, um_per_model_unit=um, status_counts=res["info"]["status_counts"]
    )
    return arrays, info


def write_route(out_dir, name, arrays, info):
    import os

    os.makedirs(out_dir, exist_ok=True)
    if arrays is not None:
        tmp = f"{out_dir}/{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, f"{out_dir}/{name}.npz")
    tmp = f"{out_dir}/{name}.{os.getpid()}.tmp.json"
    with open(tmp, "w") as f:
        json.dump(info, f, indent=1, default=_js)
    os.replace(tmp, f"{out_dir}/{name}.json")


def write_stage(path, names, route_dir, tpl, stage="final"):
    """fitter_3d stage .npz (trainer.Stage.save_npz keys, batch = the given specimens that have a route file) from the
    route files: stage 'final' = the registered mesh (native parameters + deform_verts), 'start' = the closed-form
    start (deform_verts = 0). Extra keys: vertex_status (B, nV), part_status (B, nJ), status (B,). Returns the labels
    written."""
    import os

    rows, labs = [], []
    for n in names:
        f = f"{route_dir}/{n}.npz"
        if os.path.isfile(f):
            rows.append(np.load(f, allow_pickle=True))
            labs.append(n)
    if not rows:
        return []
    if stage == "start":
        out = {k: np.stack([np.asarray(z[f"start_{k}"]) for z in rows]) for k in STAGE_KEYS if k != "deform_verts"}
        out["deform_verts"] = np.zeros((len(rows), tpl.nV, 3), np.float32)
        out["verts"] = np.stack([np.asarray(z["start_verts"]) for z in rows])
        out["pivots"] = np.stack([np.asarray(z["start_pivots"]) for z in rows])
    else:
        out = {k: np.stack([np.asarray(z[k]) for z in rows]) for k in STAGE_KEYS}
        out["verts"] = np.stack([np.asarray(z["verts"]) for z in rows])
        out["pivots"] = np.stack([np.asarray(z["pivots"]) for z in rows])
    out["log_scale"] = out["log_scale"].reshape(len(rows), 1)
    out["target_centre"] = np.stack([np.asarray(z["target_centre"]) for z in rows])
    out["target_scale"] = np.array([float(z["target_scale"]) for z in rows])
    out["target_units"] = np.array("micrometre")
    out["target_frame_rotation"] = np.stack([np.asarray(z["target_frame_rotation"]) for z in rows])
    out["target_frame_origin"] = np.stack([np.asarray(z["target_frame_origin"]) for z in rows])
    out["faces"] = np.tile(tpl.F[None], (len(rows), 1, 1))
    out["labels"] = np.array([f"{n}.obj" for n in labs])
    out["vertex_status"] = np.stack([np.asarray(z["vertex_status"]) for z in rows])
    out["part_status"] = np.stack([np.asarray(z["part_status"]).astype(str) for z in rows])
    out["status"] = np.array([str(z["status"]) for z in rows])
    out["geom_version"] = np.array(VERSION)
    np.savez(path, **out)
    return labs
