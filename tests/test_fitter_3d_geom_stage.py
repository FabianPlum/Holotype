"""The geometric registration as a fitter_3d stage, through fitter_3d/optimise.py as the command line runs it
(`scheme: 'geom'`, docs/antscan_geom_stage.md), on one specimen made from the model itself (CPU, no scan data).

    pytest tests/test_fitter_3d_geom_stage.py        (from the repository root)

What it guards: the schedule runs to its end (a stage that records no loss components must not stop the loss plots
and the combine step), the stage file carries the registration and maps back to the scan frame through its two frame
keys (the specimen's canonical frame is not the scan frame), and the fitter's own forward of the written parameters
plus deform_verts gives the written verts (the skinning rows of the model are used as stored). A second test holds
the default joint path to what it does today for a part the joint product reports missing: its bones keep the rest
articulation and are masked out.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(HERE, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")
pytestmark = pytest.mark.skipif(not os.path.isfile(PKL), reason="model pkl not found")

RUN = r"""
import json, os, sys
import numpy as np
sys.dont_write_bytecode = True
pkl, mesh_dir, joints_dir, limbs_dir, yaml_src, out = sys.argv[1:7]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import config
config.set_model(pkl)
config.PLOT_RESULTS = False
config.SPLIT_TARGET_MESHES_INTO_BATCHES_OF_SIZE = 0
import torch
torch.manual_seed(0)
from fitter_3d import optimise
from fitter_3d.trainer import SMAL3DFitter
args = optimise.parser.parse_args(["--mesh_dir", mesh_dir, "--joints_dir", joints_dir, "--joint_kind", "medial",
    "--conf_min", "0.3", "--limbs_dir", limbs_dir, "--yaml_src", yaml_src, "--results_dir", out])
optimise.main(args)
z = np.load(os.path.join(out, "Stage_G_geom.npz"), allow_pickle=True)
B = len(z["labels"])
f = SMAL3DFitter(batch_size=B, device="cpu", shape_family=-1)
with torch.no_grad():
    for k in ("global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts", "betas_trans", "log_scale"):
        getattr(f, k).copy_(torch.as_tensor(np.asarray(z[k]), dtype=torch.float32))
    f.rotation_mask = torch.as_tensor(np.asarray(z["rotation_mask"]), dtype=torch.float32)
    v = f().numpy().astype(np.float64)
err = float(np.abs(v - z["verts"]).max() * float(np.max(z["target_scale"])))
json.dump({"forward_err_um": err, "keys": sorted(z.files)}, open(os.path.join(out, "check.json"), "w"))
"""


def test_geom_scheme_runs_through_optimise_and_its_stage_file_closes_the_forward(tmp_path):
    import json

    import trimesh
    import yaml
    from antscan_synth_inputs import frame_of, make_specimen

    from fitter_3d.geomreg.model import template

    tpl = template(PKL)
    root = str(tmp_path)
    Rf, of = frame_of(21)  # a canonical frame that is NOT the scan frame
    V, _ = make_specimen(root, tpl, "syn_g", seed=21, frame=(Rf, of))
    os.makedirs(f"{root}/meshes")
    # optimise.py reads the mesh in the joint product's canonical frame: x_canonical = (x_scan - origin) @ R^T
    trimesh.Trimesh(vertices=(V - of) @ Rf.T, faces=tpl.F, process=False).export(f"{root}/meshes/syn_g.obj")
    cfg = {
        "stages": {
            "Stage_G_geom": {
                "scheme": "geom",
                "refit": True,
                "refit_betas": False,
                "geom": {"c_outer": 6, "c_beta_outer": 2, "f_outer": 6},  # a short schedule: plumbing, not accuracy
            }
        },
        "args": {},
    }
    yaml.safe_dump(cfg, open(f"{root}/geom.yaml", "w"))
    out = f"{root}/out"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            RUN,
            PKL,
            f"{root}/meshes",
            f"{root}/joints/syn_g",
            f"{root}/limbs",
            f"{root}/geom.yaml",
            out,
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, (r.stdout[-800:], r.stderr[-1500:])  # the whole schedule ends, loss plots included
    chk = json.load(open(f"{out}/check.json"))
    for k in (
        "verts",
        "deform_verts",
        "joint_rot",
        "geom_vertex_status",
        "geom_part_status",
        "geom_status",
        "start_how",
    ):
        assert k in chk["keys"], k
    assert "vertex_status" not in chk["keys"]  # that name belongs to the status step of the registration pipeline
    assert chk["forward_err_um"] < 0.05  # SMAL3DFitter(parameters) + deform_verts = verts
    z = np.load(f"{out}/Stage_G_geom.npz", allow_pickle=True)
    assert str(z["geom_status"][0]) in ("ok", "review")
    # the stage file in the scan frame lies on the specimen it was registered to
    Vb = (z["verts"][0].astype(np.float64) * float(z["target_scale"][0]) + z["target_centre"][0]) @ z[
        "target_frame_rotation"
    ][0] + z["target_frame_origin"][0]
    assert np.median(np.linalg.norm(Vb - V, axis=1)) < 0.02 * np.ptp(V, 0).max()


DEFAULT_PATH = r"""
import json, os, sys
import numpy as np
sys.dont_write_bytecode = True
pkl, mesh_dir, joints_dir, out = sys.argv[1:5]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import config
config.set_model(pkl)
import torch
from fitter_3d.joint_init import prepare_batch
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.utils import load_meshes
files = sorted(os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj"))
_, meshes, norms = load_meshes(mesh_files=files, device="cpu", return_norm=True)
f = SMAL3DFitter(batch_size=len(files), device="cpu", shape_family=-1)
kw = prepare_batch(f, files, norms, joints_dir, "cpu")  # every option at its default: today's joint path
names = [str(n) for n in config.joint_names]
res = {n: [float(f.rotation_mask[0, names.index(n) - 1].abs().sum()), float(f.joint_rot[0, names.index(n) - 1].abs().sum())]
       for n in ("l_2_ta_l", "l_2_pt_l", "l_2_fe_l", "l_3_ta_r")}
res["keys"] = sorted(kw)
json.dump(res, open(out, "w"))
"""


def test_default_joint_path_keeps_a_missing_part_in_rest_and_masks_it(tmp_path):
    import json

    import trimesh
    from antscan_synth_inputs import make_specimen

    from fitter_3d.geomreg.model import template

    tpl = template(PKL)
    root = str(tmp_path)
    # the tarsus joint is reported missing while the leg tip beyond it has a position: the initialisation would
    # rotate the tarsus bone towards that tip; the joint path puts every bone of a missing part back to rest
    V, d = make_specimen(root, tpl, "syn_m", seed=22, null=("l_2_ta_l",))
    assert d["completeness"]["missing_joints"] == ["l_2_ta_l"] and d["joints_canonical"]["l_2_pt_l"] is not None
    os.makedirs(f"{root}/meshes")
    trimesh.Trimesh(vertices=V, faces=tpl.F, process=False).export(f"{root}/meshes/syn_m.obj")
    out = f"{root}/default_path.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run(
        [sys.executable, "-c", DEFAULT_PATH, PKL, f"{root}/meshes", f"{root}/joints/syn_m", out],
        cwd=HERE,
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, (r.stdout[-800:], r.stderr[-1500:])
    res = json.load(open(out))
    for n in ("l_2_ta_l", "l_2_pt_l"):  # the missing part: masked out and left in the rest articulation
        assert res[n] == [0.0, 0.0], (n, res[n])
    for n in ("l_2_fe_l", "l_3_ta_r"):  # every other bone stays free
        assert res[n][0] == 3.0, (n, res[n])
    assert "joint_ref_regressor" not in res["keys"]  # the marker joint term belongs to --joint_kind medial only
