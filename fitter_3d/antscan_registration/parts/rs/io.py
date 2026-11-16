"""Readers for registrations from any route, joint products and envelopes (scan frame, micrometres).

Registration formats accepted (`read_registration(path_or_dir, name)`):
  route   <dir>/<name>.npz in the input contract of the registration benchmark:
          V_final (or V, V_scan, verts) (10235,3) scan frame um; optional V_pre, reg_v, vertex_status, part_status,
          status, jfile, um_per_model_unit
  skelreg <dir>/<name>/<name>_skelreg.npz (V_scan, vertex_status 0/1/2, part names) + _skelreg.json (status, parts)
  stage   a fitter_3d stage .npz (trainer.Stage.save_npz, batch, `labels`) written with the joint options
          (target_scale / target_centre, target_frame_rotation / origin; frame else from the joint file):
          x_scan = (verts * target_scale + target_centre) @ R + origin   (regbench rb_common.adapt_f3d convention)
Returns dict(V, V_pre, reg_v, route_part_status, route_status, jfile, src_format, native) where `native` holds the
fitter_3d parameters when the source is a stage file (else None).
"""

import glob
import json
import os

import numpy as np

NV = 10235
SAVE_NPZ_PARAMS = (
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
    "faces",
)


def frame_of_joint_file(jfile):
    d = json.load(open(jfile))
    return np.asarray(d["frame"]["rotation"], float), np.asarray(d["frame"]["origin_scan"], float)


def _first(z, keys):
    for k in keys:
        if k in z.files:
            return np.asarray(z[k], np.float64)
    raise KeyError(f"none of {keys} in the file")


def stage_index(stage_path):
    """label -> (file, row) over one stage file or a run dir with batch_*/<stage>.npz."""
    out = {}
    files = (
        [stage_path] if os.path.isfile(stage_path) else sorted(glob.glob(os.path.join(stage_path, "batch_*", "*.npz")))
    )
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "labels" not in z.files or "verts" not in z.files:
            continue
        for i, lab in enumerate(z["labels"]):
            lab = str(lab)
            out[lab[:-4] if lab.endswith(".obj") else lab] = (f, i)
    return out


_STAGE_CACHE = {}


def read_stage_row(stage_path, name, jfile=None):
    key = os.path.abspath(stage_path)
    if key not in _STAGE_CACHE:
        _STAGE_CACHE[key] = stage_index(stage_path)
    idx = _STAGE_CACHE[key]
    if name not in idx:
        raise FileNotFoundError(f"{name} not in stage file(s) {stage_path}")
    f, i = idx[name]
    z = np.load(f, allow_pickle=True)
    if "target_scale" not in z.files:
        raise ValueError(f"{f}: no target_scale / target_centre: a plain fitter_3d result cannot be mapped to um")
    native = {
        k: np.asarray(z[k][i])
        for k in z.files
        if k not in ("labels", "target_units") and np.ndim(z[k]) > 0 and z[k].shape[0] == len(z["labels"])
    }
    native["target_units"] = str(z["target_units"]) if "target_units" in z.files else "micrometre"
    sc, ce = float(z["target_scale"][i]), np.asarray(z["target_centre"][i], float)
    if "target_frame_rotation" in z.files:
        R, o = np.asarray(z["target_frame_rotation"][i], float), np.asarray(z["target_frame_origin"][i], float)
        fsrc = "npz"
    elif jfile and os.path.isfile(jfile):
        R, o = frame_of_joint_file(jfile)
        fsrc = "joint_file"
    else:
        raise ValueError(f"{name}: no frame in the stage file and no joint file")
    native.update(frame_R=R, frame_o=o, frame_source=fsrc, stage_file=f, stage_row=i)
    V = (np.asarray(z["verts"][i], float) * sc + ce) @ R + o
    rm = np.asarray(z["rotation_mask"][i]) if "rotation_mask" in z.files else None
    return V, native, rm


def to_norm(V_scan, native):
    """scan frame um -> the stage file's normalised target frame (inverse of read_stage_row's map)."""
    R, o = native["frame_R"], native["frame_o"]
    return ((np.asarray(V_scan, float) - o) @ R.T - np.asarray(native["target_centre"], float)) / float(
        native["target_scale"]
    )


def read_registration(src, name, fmt="auto", jfile=None):
    """See module docstring. `src` is a directory (route / skelreg) or a stage file / run dir (fmt='stage')."""
    out = dict(
        V=None,
        V_pre=None,
        reg_v=None,
        route_part_status=None,
        route_status="ok",
        jfile=jfile,
        src_format=None,
        native=None,
        src=src,
    )
    if fmt == "auto":
        if os.path.isfile(os.path.join(src, f"{name}.npz")):
            fmt = "route"
        elif os.path.isfile(os.path.join(src, name, f"{name}_skelreg.npz")):
            fmt = "skelreg"
        elif src.endswith(".npz") or glob.glob(os.path.join(src, "batch_*", "*.npz")):
            fmt = "stage"
        else:
            raise FileNotFoundError(f"no registration of {name} under {src}")
    out["src_format"] = fmt
    if fmt == "route":
        z = np.load(os.path.join(src, f"{name}.npz"), allow_pickle=True)
        out["V"] = _first(z, ("V_final", "V", "V_scan", "verts"))
        out["V_pre"] = np.asarray(z["V_pre"], float) if "V_pre" in z.files else None
        # the registration's own codes: `geom_*` in the pipeline's registration/<name>.npz (so that they cannot be
        # mistaken for the status step's codes of the same name), plain names in a route file of any other source
        kv, kp, ks = (
            ("geom_vertex_status", "geom_part_status", "geom_status")
            if "geom_vertex_status" in z.files
            else ("vertex_status", "part_status", "status")
        )
        if "reg_v" in z.files:
            out["reg_v"] = np.asarray(z["reg_v"], bool)
        elif kv in z.files:
            out["reg_v"] = np.asarray(z[kv]) != 2
        if kp in z.files:
            out["route_part_status"] = [str(s) for s in z[kp]]
        if ks in z.files:
            out["route_status"] = str(z[ks])
        if not jfile and "jfile" in z.files and str(z["jfile"]):
            out["jfile"] = str(z["jfile"])
        # native fitter_3d parameters carried by the route file itself (GEOM contract: one specimen, no batch axis)
        if all(
            k in z.files
            for k in (
                "global_rot",
                "joint_rot",
                "deform_verts",
                "verts",
                "target_scale",
                "target_centre",
                "target_frame_rotation",
                "target_frame_origin",
            )
        ):
            nat = {k: np.asarray(z[k]) for k in SAVE_NPZ_PARAMS if k in z.files}
            nat.update(
                target_scale=float(z["target_scale"]),
                target_centre=np.asarray(z["target_centre"], float),
                frame_R=np.asarray(z["target_frame_rotation"], float),
                frame_o=np.asarray(z["target_frame_origin"], float),
                frame_source="route file",
                stage_file=os.path.join(src, f"{name}.npz"),
                stage_row=0,
                target_units=str(z["target_units"]) if "target_units" in z.files else "micrometre",
            )
            if "faces" not in nat:
                from fitter_3d.antscan_registration.parts.skelreg.model import template

                nat["faces"] = template("medial").F
            out["native"] = nat
    elif fmt == "skelreg":
        base = os.path.join(src, name, f"{name}_skelreg")
        z = np.load(base + ".npz", allow_pickle=True)
        out["V"] = np.asarray(z["V_scan"], float)
        out["reg_v"] = np.asarray(z["vertex_status"]) != 2
        if os.path.isfile(base + ".json"):
            d = json.load(open(base + ".json"))
            out["route_status"] = str(d.get("status", "ok"))
            names = [str(n) for n in z["part_names"]]
            out["route_part_status"] = [str((d.get("parts", {}).get(n) or {}).get("status", "ok")) for n in names]
    elif fmt == "stage":
        V, native, rm = read_stage_row(src, name, jfile)
        out["V"], out["native"] = V, native
        if rm is not None:
            from fitter_3d.antscan_registration.parts.skelreg.model import template

            t = template("medial")
            frozen = np.zeros(t.nJ, bool)
            frozen[1:] = np.asarray(rm).sum(1) == 0
            out["reg_v"] = ~frozen[t.dom]
    if out["route_part_status"] is not None:
        out["route_part_status"] = normalise_part_status(out["route_part_status"])
    if out["V"] is None or out["V"].shape != (NV, 3):
        raise ValueError(f"{name}: registration has shape {None if out['V'] is None else out['V'].shape}")
    if not np.isfinite(out["V"]).all():
        raise ValueError(f"{name}: non-finite vertices in the registration")
    return out


def normalise_part_status(ps):
    """Route part statuses per bone. SKEL-REG (and the benchmark's collector) write a status only for SKEL-REG's 41
    'owner' parts (skelreg.charts.owners: b_a_4 / b_a_5 belong to the gaster b_a_3, pretarsus to tarsus, an_3 to
    an_2, wings ...); the other bones read 'unregistered' by default. A bone whose
    status is 'unregistered' and that is not its own owner takes its owner's status."""
    from fitter_3d.antscan_registration.parts.skelreg import charts
    from fitter_3d.antscan_registration.parts.skelreg.model import template

    own = charts.owners(template("medial"))
    ps = [str(s) for s in ps]
    return [ps[own[j]] if (own[j] != j and ps[j] == "unregistered") else ps[j] for j in range(len(ps))]


def joint_file(joints_dir, name):
    for p in (os.path.join(joints_dir, name, f"{name}_joints.json"), os.path.join(joints_dir, f"{name}_joints.json")):
        if os.path.isfile(p):
            return p
    return None


def limbs_dir(limbs_root, name):
    p = os.path.join(limbs_root, name)
    return p if os.path.isfile(os.path.join(p, "limbs.json")) else None


def envelope(env_tmpl, name):
    return env_tmpl.format(n=name, name=name)


def read_names(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
