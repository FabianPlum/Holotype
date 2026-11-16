"""Helper of the pipeline tests: a stage .npz through fitter_3d's OWN forward (SMAL3DFitter of the tree given, CPU).
Every native parameter + deform_verts is set, the forward is run, and the result is compared with the stored verts.
It runs in a process of its own because config.set_model changes global state.

    python stage_forward_helper.py TREE STAGE_NPZ MODEL_PKL OUT_JSON

Exit code 3: the fitter of this tree has no `log_scale` parameter (it predates the joint path of fitter_3d, which
added the global scale): the stage file cannot be loaded into it.
"""

import json
import os
import sys

tree, stage, model, out = sys.argv[1:5]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
tree = os.path.realpath(tree)
sys.path.insert(0, tree)
os.chdir(tree)
import numpy as np
import torch

import config

config.set_model(model)
from fitter_3d.trainer import SMAL3DFitter

torch.set_num_threads(2)
z = np.load(stage, allow_pickle=True)
B = len(z["labels"])
f = SMAL3DFitter(batch_size=B, device="cpu", shape_family=-1)
keys = ("global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts", "betas_trans", "log_scale")
missing = [k for k in keys if not hasattr(f, k)]
if missing:
    print(f"this SMAL3DFitter has no parameter {missing}")
    sys.exit(3)
with torch.no_grad():
    for k in keys:
        getattr(f, k).copy_(torch.as_tensor(np.asarray(z[k]), dtype=torch.float32))
    f.rotation_mask = torch.as_tensor(np.asarray(z["rotation_mask"]), dtype=torch.float32)
    v = f().numpy().astype(np.float64)
d = np.linalg.norm(v - z["verts"].astype(np.float64), axis=2) * np.asarray(z["target_scale"], np.float64)[:, None]
json.dump({"n": B, "max_um": float(d.max()), "median_um": float(np.median(d))}, open(out, "w"))
