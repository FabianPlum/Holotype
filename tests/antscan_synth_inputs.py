"""Inputs of the pipeline made from the model itself (no scan data, no truth files): a posed, scaled template is the
'envelope'; its virtual markers are the joint product (round-6 contract, or with an `ensemble` block as round 7 K4);
polylines through the markers of every limb chain are the limb product. Extends the helper of
tests/test_fitter_3d_geomreg_pipeline.py (make_specimen) by the limb product, low-confidence joints and the ensemble block."""

import json
import os

import numpy as np

LIMB_CHAINS = {
    **{
        f"{S}{k}": [f"l_{k}_{j}_{s}" for j in ("pt", "ta", "ti", "fe", "tr", "co")]
        for k in "123"
        for S, s in (("L", "l"), ("R", "r"))
    },
    "AL": ["an_3_l", "an_2_l", "an_1_l"],
    "AR": ["an_3_r", "an_2_r", "an_1_r"],
}


def write_stl(path, V, F):
    import trimesh

    trimesh.Trimesh(vertices=V, faces=F, process=False).export(path)


def frame_of(seed):
    """a canonical frame that is not the scan frame: rotation (rows = canonical axes in scan coordinates) and origin"""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(1000 + seed)
    return Rotation.from_rotvec(rng.normal(0, 0.6, 3)).as_matrix(), rng.normal(0, 400.0, 3)


def make_specimen(root, tpl, name, seed=0, low=(), null=(), ensemble=False, pose=0.25, scale_um=2000.0, frame=None):
    """writes <root>/env/<name>/<name>_envelope.stl, <root>/joints/<name>/<name>_joints.json,
    <root>/limbs/<name>/limbs.{json,npz}; `low`: joints given confidence 0.1; `null`: joints given no position;
    `frame`: (rotation, origin) of the joint product's canonical frame (default: the scan frame itself).
    Returns (V (nV, 3) the true vertices in micrometres, joint dict)."""
    from fitter_3d.geomreg import native

    rng = np.random.default_rng(seed)
    nJ = tpl.nJ
    lo, hi = tpl.limits[1:, :, 0], tpl.limits[1:, :, 1]
    p = dict(
        global_rot=rng.normal(0, 0.3, 3),
        joint_rot=np.clip(rng.normal(0, pose, (nJ - 1, 3)), lo, hi),
        betas=np.zeros(tpl.nB),
        log_beta_scales=np.zeros((nJ, 3)),
        trans=np.array([50.0, -20.0, 10.0]),
        betas_trans=np.zeros((nJ, 3)),
        log_scale=np.array([np.log(scale_um)]),
        rotation_mask=np.ones((nJ - 1, 3)),
    )
    V, _ = native.forward_np(tpl, p)
    os.makedirs(f"{root}/env/{name}", exist_ok=True)
    write_stl(f"{root}/env/{name}/{name}_envelope.stl", V, tpl.F)
    mk = tpl.marker_regressor() @ V
    hf = tpl.head_front_row() @ V
    js, cf, prov = {}, {}, {}
    for j, n in enumerate(tpl.names):
        ok = not n.startswith("w_") and n not in null
        js[n] = mk[j].tolist() if ok else None
        cf[n] = (0.1 if n in low else 0.9) if ok else 0.0
        prov[n] = "scan_feature (synthetic)" if ok else "null"
    js["head_front"], cf["head_front"] = hf.tolist(), 0.9
    js["gaster_tip"], cf["gaster_tip"] = mk[tpl.idx["b_a_5"]].tolist(), 0.9
    d = {
        "name": name,
        "units": "micrometre",
        "method": "synthetic test input",
        "frame": {
            "rotation": (np.eye(3) if frame is None else np.asarray(frame[0])).tolist(),
            "origin_scan": ([0.0, 0.0, 0.0] if frame is None else np.asarray(frame[1]).tolist()),
            "info": {"lr_consistent": True},
        },
        "joints_scan": js,
        "joints_canonical": {
            k: (v if (v is None or frame is None) else ((np.asarray(v) - frame[1]) @ np.asarray(frame[0]).T).tolist())
            for k, v in js.items()
        },
        "confidence": cf,
        "provenance": prov,
        "null_reasons": {},
        "joint_detail": {},
        "completeness": {"missing_joints": sorted(null), "flags": [], "parts": {}},
        "body_length_um": float(native.chord_body_length(tpl, mk, hf)),
        "waist_segments": 2,
        "status": "ok",
    }
    if ensemble:
        st = ["s0", "s1", "s2", "s3"]
        d["ensemble"] = {"states": st, "states_ok": st, "kept": st, "trunk_state": "s0"}
        for n in tpl.names:
            if js.get(n) is not None and n.startswith(("l_", "an_", "ma_")):
                d["joint_detail"][n] = {
                    "ens": {
                        "n_states": 4,
                        "n_cluster": 4,
                        "n_placed": 4,
                        "n_null": 0,
                        "picked_state": "s0",
                        "spread_pct": 0.2,
                        "spread_med_pct": 0.1,
                        "conf_states_median": cf[n],
                        "limb_state_frac": 1.0,
                        "placed_share": 1.0,
                    }
                }
    os.makedirs(f"{root}/joints/{name}", exist_ok=True)
    json.dump(d, open(f"{root}/joints/{name}/{name}_joints.json", "w"))
    # limb product: polylines tip first through the chain's markers, radius from the template surface
    arrays, meta = {}, {}
    chains = dict(LIMB_CHAINS)
    for limb, bone in (("ML", "ma_l"), ("MR", "ma_r")):
        j = tpl.idx[bone]
        ids = np.where(tpl.dom == j)[0]
        tip = V[ids[np.argmax(np.linalg.norm(V[ids] - mk[j], axis=1))]]
        chains[limb] = [tip, mk[j]]
    for limb, ch in chains.items():
        P = np.array([mk[tpl.idx[n]] if isinstance(n, str) else n for n in ch])
        X = np.concatenate([np.linspace(P[i], P[i + 1], 12, endpoint=False) for i in range(len(P) - 1)] + [P[-1:]])
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
        from scipy.spatial import cKDTree

        r = cKDTree(V).query(X)[0]
        arrays[f"{limb}_X"], arrays[f"{limb}_s"], arrays[f"{limb}_r"] = X, s, r
        meta[limb] = {
            "status": "path",
            "confidence": 0.9,
            "flags": [],
            "missing": [],
            "core_arc_um": float(s[-1]),
            "arc_um": float(s[-1]),
            "n_points": len(X),
        }
    os.makedirs(f"{root}/limbs/{name}", exist_ok=True)
    np.savez(f"{root}/limbs/{name}/limbs.npz", **arrays)
    json.dump({"name": name, "units": "micrometre", "limbs": meta}, open(f"{root}/limbs/{name}/limbs.json", "w"))
    return V, d


def crash_job(job):
    """a job for fitter_3d.antscan_registration.steps.pool_map: the job named 'boom' kills its worker process (as a memory limit would)"""
    if job["name"] == "boom":
        os._exit(9)
    return dict(name=job["name"], status="ok")
