"""End-to-end test of the geometric core on a specimen made from the model itself (CPU, no scan data, no truth files):
a posed, scaled template is written as the 'envelope', its virtual markers as the joint product. It registers, the
written native parameters reproduce the mesh, and specimens that cannot be registered end as a status.

    pytest tests/test_fitter_3d_geomreg_pipeline.py        (from the repository root; CPU)
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fitter_3d.geomreg import native, pipeline
from fitter_3d.geomreg.model import template

PKL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl"
)
pytestmark = pytest.mark.skipif(not os.path.isfile(PKL), reason="model pkl not found")
FAST = {"c_outer": 6, "c_beta_outer": 2, "f_outer": 6}  # a short schedule: the tests check plumbing, not accuracy


def write_stl(path, V, F):
    import trimesh

    trimesh.Trimesh(vertices=V, faces=F, process=False).export(path)


def make_specimen(tmp, tpl, seed=0, usable=None, with_bl=True):
    rng = np.random.default_rng(seed)
    nJ = tpl.nJ
    lo, hi = tpl.limits[1:, :, 0], tpl.limits[1:, :, 1]
    p = dict(
        global_rot=rng.normal(0, 0.3, 3),
        joint_rot=np.clip(rng.normal(0, 0.25, (nJ - 1, 3)), lo, hi),
        betas=np.zeros(tpl.nB),
        log_beta_scales=np.zeros((nJ, 3)),
        trans=np.array([50.0, -20.0, 10.0]),
        betas_trans=np.zeros((nJ, 3)),
        log_scale=np.array([np.log(2000.0)]),
        rotation_mask=np.ones((nJ - 1, 3)),
    )
    V, _ = native.forward_np(tpl, p)  # micrometres, 'scan frame'
    name = f"synthetic_{seed}"
    os.makedirs(f"{tmp}/{name}", exist_ok=True)
    write_stl(f"{tmp}/{name}/{name}_envelope.stl", V, tpl.F)
    mk = tpl.marker_regressor() @ V
    hf = tpl.head_front_row() @ V
    js, cf = {}, {}
    for j, n in enumerate(tpl.names):
        ok = not n.startswith("w_") and (usable is None or n in usable)
        js[n] = mk[j].tolist() if ok else None
        cf[n] = 1.0 if ok else 0.0
    js["head_front"], cf["head_front"] = hf.tolist(), 1.0
    js["gaster_tip"], cf["gaster_tip"] = mk[tpl.idx["b_a_5"]].tolist(), 1.0
    d = {
        "name": name,
        "units": "micrometre",
        "frame": {"rotation": np.eye(3).tolist(), "origin_scan": [0.0, 0.0, 0.0]},
        "joints_scan": js,
        "confidence": cf,
        "completeness": {"missing_joints": [], "flags": []},
        "status": "ok",
    }
    if with_bl:
        d["body_length_um"] = native.chord_body_length(tpl, mk, hf)
    json.dump(d, open(f"{tmp}/{name}/{name}_joints.json", "w"))
    return name, V, d


@pytest.fixture(scope="module")
def tpl():
    return template(PKL)


def test_pipeline_registers_writes_native_parameters_and_fails_as_a_status(tpl, tmp_path):
    name, V, d = make_specimen(str(tmp_path), tpl, seed=1)
    arrays, info = pipeline.register_specimen(
        name,
        f"{tmp_path}/{name}/{name}_envelope.stl",
        f"{tmp_path}/{name}/{name}_joints.json",
        None,
        kind="medial",
        opts=FAST,
        device="cpu",
        model_pkl=PKL,
    )
    assert arrays is not None, info.get("error")
    assert info["status"] in ("ok", "review")
    err = np.linalg.norm(arrays["V_final"] - V, axis=1) / d["body_length_um"]
    assert np.median(err) < 0.02  # the template registered onto itself
    # the model's own forward of the written parameters + deform_verts is the registered mesh
    p = {k: np.asarray(arrays[k], np.float64) for k in pipeline.STAGE_KEYS}
    v, _ = native.forward_np(tpl, p)
    assert np.abs(v - arrays["V_final_norm"]).max() < 1e-4
    # the stored start is exactly native too
    p0 = {k: np.asarray(arrays[f"start_{k}"], np.float64) for k in pipeline.STAGE_KEYS if k != "deform_verts"}
    v0, _ = native.forward_np(tpl, p0)
    assert np.abs(v0 - arrays["start_verts"]).max() < 1e-4
    for k in (
        "vertex_status",
        "part_status",
        "reg_v",
        "support",
        "start_how",
        "target_centre",
        "target_scale",
        "target_frame_rotation",
    ):
        assert k in arrays
    pipeline.write_route(str(tmp_path / "out"), name, arrays, info)
    labs = pipeline.write_stage(str(tmp_path / "out" / "Stage_S_surface.npz"), [name], str(tmp_path / "out"), tpl)
    z = np.load(tmp_path / "out" / "Stage_S_surface.npz", allow_pickle=True)
    assert labs == [name] and z["verts"].shape == (1, tpl.nV, 3) and z["joint_rot"].shape == (1, tpl.nJ - 1, 3)
    # a specimen that cannot be registered is a status, never an exception
    name, V, d = make_specimen(str(tmp_path), tpl, seed=4, usable=set())  # no usable joint at all
    a, info = pipeline.register_specimen(
        name,
        f"{tmp_path}/{name}/{name}_envelope.stl",
        f"{tmp_path}/{name}/{name}_joints.json",
        None,
        opts=FAST,
        device="cpu",
        model_pkl=PKL,
    )
    assert a is None and info["status"] == "failed" and "StartError" in info["error"]
    a, info = pipeline.register_specimen(
        name,
        f"{tmp_path}/nowhere.stl",
        f"{tmp_path}/{name}/{name}_joints.json",
        None,
        opts=FAST,
        device="cpu",
        model_pkl=PKL,
    )
    assert a is None and info["status"] == "failed"
    name, V, d = make_specimen(str(tmp_path), tpl, seed=5, with_bl=False)  # joint file without a body length
    a, info = pipeline.register_specimen(
        name,
        f"{tmp_path}/{name}/{name}_envelope.stl",
        f"{tmp_path}/{name}/{name}_joints.json",
        None,
        opts=FAST,
        device="cpu",
        model_pkl=PKL,
    )
    assert a is None and info["status"] == "failed"
