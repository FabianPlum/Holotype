"""Export as ONE module: fitter_3d stage .npz (trainer.Stage.save_npz layout + status / provenance keys that the
readers ignore), the UC1 rest bundle for the Blender add-on (status-aware fill + ratio gate, R2-SHAPE's code), and
round-trip checks.

fitter_3d's forward (trainer.py SMAL3DFitter.forward): verts = exp(log_scale) * SMAL(betas, pose,
log_beta_scales, betas_trans) + trans + deform_verts, in the normalised target frame; x_file = verts * target_scale +
target_centre, x_scan = x_file @ R + origin (R / origin = target_frame_rotation / origin, else the joint product's
frame). So a registered mesh V (scan frame, um) is written exactly by
    verts' = ((V - origin) @ R^T - target_centre) / target_scale ;  deform_verts' = deform_verts + (verts' - verts)
with every native parameter unchanged (COVER's to_stage_npz rule; deform_verts carries the residual).
Without native parameters ('defaults'): the fitter's initial parameters (zero pose, betas = the model's mean betas,
log_scale 0, trans 0, rotation_mask 1) whose forward is v_shaped(mean betas) = v_template + shapedirs @ mean_betas
(config.ignore_sym = True as in every AntScan fitter_3d harness); deform_verts' = verts' - that; the target frame is
the joint product's, target_centre / target_scale = vertex mean / max |x - centre| of the envelope in that frame
(fitter_3d load_meshes normalisation). `status_params_source` says which.
"""

import json
import os
import pickle

import numpy as np

from fitter_3d.antscan_registration.parts import partconf

from . import VERSION
from .io import to_norm

PARAM_DTYPES = dict(
    global_rot=np.float32,
    joint_rot=np.float32,
    betas=np.float32,
    log_beta_scales=np.float32,
    trans=np.float32,
    deform_verts=np.float32,
    betas_trans=np.float32,
    log_scale=np.float32,
    rotation_mask=np.float32,
    verts=np.float32,
    pivots=np.float32,
    target_centre=np.float64,
    target_scale=np.float64,
    faces=np.int64,
)
SAVE_KEYS = [
    "global_rot",
    "joint_rot",
    "betas",
    "log_beta_scales",
    "trans",
    "deform_verts",
    "betas_trans",
    "log_scale",
    "rotation_mask",
    "verts",
    "pivots",
    "target_centre",
    "target_scale",
    "target_frame_rotation",
    "target_frame_origin",
    "faces",
]
_M = {}


def model():
    if not _M:
        with open(partconf.model(), "rb") as f:
            dd = pickle.load(f, encoding="latin1")
        sd = np.asarray(dd["shapedirs"], np.float64)  # (V, 3, K)
        K = sd.shape[-1]
        mb = np.asarray(dd["shape_mean_betas"], np.float64)[:K] if "shape_mean_betas" in dd else np.zeros(K)
        vt = np.asarray(dd["v_template"], np.float64)
        v_def = vt + np.einsum("vck,k->vc", sd, mb)
        Jr = np.asarray(
            dd["J_regressor"].todense() if hasattr(dd["J_regressor"], "todense") else dd["J_regressor"], np.float64
        )
        _M.update(
            v_default=v_def,
            mean_betas=mb,
            F=np.asarray(dd["f"], np.int64),
            J_default=Jr @ v_def,
            nJ=len(dd["J_names"]),
            K=K,
        )
    return _M


FORWARD_PARAMS = (
    "global_rot",
    "joint_rot",
    "betas",
    "log_beta_scales",
    "trans",
    "betas_trans",
    "log_scale",
    "rotation_mask",
)


def stage_row_from_native(native, V_scan):
    """native = rs.io.read_stage_row(...) params of one specimen; V_scan (10235,3) um. -> dict of arrays (no batch).

    A stage file must satisfy fitter_3d's own forward: SMAL3DFitter(parameters) + deform_verts = verts. The fitter
    blends with the skinning rows AS THE MODEL FILE STORES THEM; the registration (geomreg) renormalises the rows
    that do not sum to 1 (OmniAnt: 154 left-mandible vertices), so its own deform_verts does not close the fitter's
    forward on those vertices (up to millimetres). For parameters that come from a registration's route file,
    deform_verts is therefore recomputed here as the residual against the fitter's forward with the stored rows
    (geomreg.native.forward_np(weights=W_raw), float64). `weight_rows_max` (normalised units) says how much of it is
    owed to the weight rows. For a row read from a stage file of the fitter itself the stored residual is kept."""
    row = {
        k: np.asarray(native[k])
        for k in SAVE_KEYS
        if k in native and k not in ("target_frame_rotation", "target_frame_origin")
    }
    vn = to_norm(V_scan, native)
    src = f"native ({native.get('stage_file')} row {native.get('stage_row')})"
    extra = {}
    if native.get("frame_source") == "route file" and all(k in native for k in FORWARD_PARAMS):
        from fitter_3d.geomreg import native as GN
        from fitter_3d.geomreg.model import template as geom_template

        tpl = geom_template(partconf.model())
        p = {k: np.asarray(native[k], PARAM_DTYPES[k]).astype(np.float64) for k in FORWARD_PARAMS}  # as they are stored
        f_raw = GN.forward_np(tpl, p, deform=False, weights=tpl.W_raw)[0]
        f_ren = GN.forward_np(tpl, p, deform=False)[0]
        row["deform_verts"] = vn - f_raw
        extra["weight_rows_max"] = float(np.linalg.norm(f_raw - f_ren, axis=1).max())
        extra["n_weight_rows"] = int((np.linalg.norm(f_raw - f_ren, axis=1) > 1e-9).sum())
        src += "; deform_verts = residual against fitter_3d's forward with the model file's skinning rows as stored"
    else:
        row["deform_verts"] = np.asarray(native["deform_verts"], np.float64) + (
            vn - np.asarray(native["verts"], np.float64)
        )
    row["verts"] = vn
    row["target_frame_rotation"] = np.asarray(native["frame_R"], np.float64)
    row["target_frame_origin"] = np.asarray(native["frame_o"], np.float64)
    row.update({f"_{k}": v for k, v in extra.items()})
    return row, src


def stage_row_default(V_scan, jfile, envelope_V):
    m = model()
    d = json.load(open(jfile))
    R, o = np.asarray(d["frame"]["rotation"], float), np.asarray(d["frame"]["origin_scan"], float)
    Ec = (np.asarray(envelope_V, float) - o) @ R.T
    ce = Ec.mean(0)
    sc = float(np.abs(Ec - ce).max())
    native = dict(frame_R=R, frame_o=o, target_centre=ce, target_scale=sc)
    vn = to_norm(V_scan, native)
    nJ = m["nJ"]
    row = dict(
        global_rot=np.zeros(3),
        joint_rot=np.zeros((nJ - 1, 3)),
        betas=m["mean_betas"].copy(),
        log_beta_scales=np.zeros((nJ, 3)),
        trans=np.zeros(3),
        deform_verts=vn - m["v_default"],
        betas_trans=np.zeros((nJ, 3)),
        log_scale=np.zeros(1),
        rotation_mask=np.ones((nJ - 1, 3)),
        verts=vn,
        pivots=m["J_default"].copy(),
        target_centre=ce,
        target_scale=np.float64(sc),
        target_frame_rotation=R,
        target_frame_origin=o,
        faces=m["F"].copy(),
    )
    return row, "defaults (no native parameters given: zero pose, mean betas; deform_verts carries the registration)"


def write_stage(rows, labels, status_rows, out_path, stage_name="Stage_status"):
    """rows: list of dicts (stage_row_*); status_rows: list of dicts with status / provenance per specimen."""
    out = {}
    for k in SAVE_KEYS:
        if all(k in r for r in rows):
            dt = PARAM_DTYPES.get(k, np.float64)
            out[k] = np.stack([np.asarray(r[k], dt) for r in rows])
    out["target_units"] = np.array("micrometre")
    out["labels"] = np.array(labels)
    nv = rows[0]["verts"].shape[0]
    out["status_vertex"] = np.stack([np.asarray(s["vertex_status"], np.int8) for s in status_rows]).reshape(
        len(rows), nv
    )
    out["status_part_code"] = np.stack([np.asarray(s["part_code"], np.int8) for s in status_rows])
    out["status_part_status"] = np.stack([np.asarray(s["part_status"], "<U12") for s in status_rows])
    out["status_specimen_class"] = np.array([s["specimen_class"] for s in status_rows], "<U12")
    out["status_params_source"] = np.array([s["params_source"] for s in status_rows])
    out["status_names"] = np.array(
        [
            "joint_supported",
            "surface_verified",
            "recovered",
            "review",
            "unregistered",
            "imputed_mirror",
            "imputed_prior",
            "recovered_estimated",
        ]
    )
    out["status_version"] = np.array(VERSION)
    out["status_provenance"] = np.array([json.dumps(s.get("provenance", {})) for s in status_rows])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    # round trip: re-read equals written; verts' reproduce the registration in the scan frame
    z = np.load(out_path, allow_pickle=True)
    chk = {"keys": sorted(z.files), "reread_equal": all(np.array_equal(z[k], out[k]) for k in out)}
    return chk


# ======================================================================================================================
# UC1 bundle with status-aware fill and ratio gate (R2-SHAPE's bundle code, copied, + fill tracking)
# ======================================================================================================================
