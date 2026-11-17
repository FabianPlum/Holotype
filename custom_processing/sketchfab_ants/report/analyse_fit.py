"""Analysis of the fitter_3d OmniAnt fit of the tunosemi ants, for the fit report.

Outputs (in <out>):
  analysis.npz         neutral-pose shapes, PCA, per-vertex errors, regions, joint scales
  report_data.json     everything the report page charts
  skin_check_PROBE.txt reimplemented skinning vs fitter verts (must match)

Usage: python analyse_fit.py --repo <Holotype> --results <fit3d_results_OmniAnt> --meshes <processed> --log <run.log> --out <dir>
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--meshes", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
for k in ("repo", "results", "meshes", "log", "out"):
    setattr(a, k, os.path.abspath(getattr(a, k)))
os.makedirs(a.out, exist_ok=True)
os.chdir(a.repo)
sys.path.insert(0, a.repo)
import config  # noqa: E402
from smal_model.batch_lbs import batch_global_rigid_transformation, batch_rodrigues  # noqa: E402
from smal_model.smal_torch import SMAL  # noqa: E402

torch.set_grad_enabled(False)
smal = SMAL("cpu", shape_family_id=-1)
names_j = list(config.dd["J_names"])
NJ = len(names_j)
W = smal.weights  # (V, NJ)
V = W.shape[0]
faces = smal.faces.numpy().astype(np.int64)


def skin(betas, theta, log_scales, betas_trans):
    """Replicates SMAL.__call__ (smal_model/smal_torch.py) and also returns the per-vertex transforms T."""
    B = betas.shape[0]
    nb = betas.shape[1]
    v_shaped = smal.v_template + (betas @ smal.shapedirs[:nb, :]).reshape(B, -1, 3)
    if config.STATIC_JOINT_LOCATIONS:
        J = smal.J.unsqueeze(0).expand(B, -1, -1)
    else:
        J = torch.stack([v_shaped[:, :, k] @ smal.J_regressor for k in range(3)], dim=2)
    Rs = batch_rodrigues(theta.reshape(-1, 3)).reshape(B, NJ, 3, 3)
    pose_feature = (Rs[:, 1:] - torch.eye(3)).reshape(B, -1)
    v_posed = (pose_feature @ smal.posedirs).reshape(B, -1, 3) + v_shaped
    _, A = batch_global_rigid_transformation(Rs, J, smal.parents, betas_logscale=log_scales, betas_trans=betas_trans)
    T = (W.unsqueeze(0).expand(B, -1, -1) @ A.reshape(B, NJ, 16)).reshape(B, -1, 4, 4)
    verts = (T[:, :, :3, :3] @ v_posed.unsqueeze(-1))[..., 0] + T[:, :, :3, 3]
    return verts, T


def load_stage(name):
    d = np.load(os.path.join(a.results, f"{name}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


stages = ["Stage_0_init", "Stage_1_default", "Stage_2_deform_coarse", "Stage_3_deform_fine"]
S = {s: load_stage(s) for s in stages}
labels = [str(x)[:-4] for x in S["Stage_3_deform_fine"]["labels"]]
N = len(labels)


def tt(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


# ---------- reproduce posed fits (probe) + neutral pose shapes ----------
probe = []
neutral = {}
posed = {}
for s in ("Stage_1_default", "Stage_3_deform_fine"):
    d = S[s]
    theta = torch.cat([tt(d["global_rot"]).unsqueeze(1), tt(d["joint_rot"])], 1)
    v_p, T_p = skin(tt(d["betas"]), theta, tt(d["log_beta_scales"]), tt(d["betas_trans"]))
    v_p_full = v_p + tt(d["trans"]).unsqueeze(1) + tt(d["deform_verts"])
    err = (v_p_full - tt(d["verts"])).abs().max().item()
    probe.append(f"{s}: max |reimplemented posed verts - fitter verts| = {err:.2e}")
    v_0, T_0 = skin(tt(d["betas"]), torch.zeros_like(theta), tt(d["log_beta_scales"]), tt(d["betas_trans"]))
    # un-pose the posed-space deformation offsets with each vertex's skinning transform
    L_p, L_0 = T_p[:, :, :3, :3], T_0[:, :, :3, :3]
    d0 = (L_0 @ torch.linalg.solve(L_p, tt(d["deform_verts"]).unsqueeze(-1)))[..., 0]
    neutral[s] = (v_0 + d0).numpy()
    posed[s] = tt(d["verts"]).numpy()
open(os.path.join(a.out, "skin_check_PROBE.txt"), "w").write("\n".join(probe) + "\n")
print("\n".join(probe))

# ---------- regions from dominant skinning weight ----------
dom = W.argmax(1).numpy()


def region_of(jn):
    if jn.startswith("an_"):
        return "antennae"
    if jn.startswith("ma_"):
        return "mandibles"
    if jn == "b_h":
        return "head"
    if jn == "b_t":
        return "mesosoma"
    if jn in ("b_a_1", "b_a_2"):
        return "waist"
    if jn.startswith("b_a_"):
        return "gaster"
    if jn.startswith("l_"):
        return {"1": "fore legs", "2": "mid legs", "3": "hind legs"}[jn[2]]
    if jn.startswith("w_"):
        return "wings"
    return "other"


region_names = ["head", "mandibles", "antennae", "mesosoma", "waist", "gaster", "fore legs", "mid legs", "hind legs"]
vreg = np.array([region_of(names_j[j]) for j in dom])
reg_idx = np.array([region_names.index(r) if r in region_names else -1 for r in vreg])
tmpl = smal.v_template.numpy()
print("region vertex counts / x-range in template:")
for r in region_names:
    m = vreg == r
    print(f"  {r:10s} {m.sum():6d}  x {tmpl[m, 0].min():+.2f}..{tmpl[m, 0].max():+.2f}")
print("unassigned:", sorted(set(vreg) - set(region_names)), int((reg_idx < 0).sum()))
body = np.isin(vreg, ["head", "mesosoma", "waist", "gaster"])

# ---------- targets, chamfer per stage, per-vertex errors ----------
rng = np.random.default_rng(0)


def norm_target(path):
    m = trimesh.load(path, process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    v = v - v.mean(0)
    v = v / np.abs(v).max()
    return v, np.asarray(m.faces)


chamfer = {s: np.zeros(N) for s in stages}
p95 = {s: np.zeros(N) for s in stages}
fit_err = {s: np.zeros((N, V), np.float32) for s in ("Stage_1_default", "Stage_3_deform_fine")}
tgt_err = {}  # per target vertex distance to fitted surface (for heat renders)
body_len = np.zeros(N)
for i, name in enumerate(labels):
    tv, tf = norm_target(os.path.join(a.meshes, f"{name}.obj"))
    tm = trimesh.Trimesh(tv, tf, process=False)
    tp, _ = trimesh.sample.sample_surface(tm, 30000, seed=0)
    t_tree = cKDTree(np.vstack([tp, tv]))
    for s in stages:
        fv = S[s]["verts"][i].astype(np.float64)
        fm = trimesh.Trimesh(fv, faces, process=False)
        fp, _ = trimesh.sample.sample_surface(fm, 30000, seed=0)
        d_tf, _ = cKDTree(fp).query(tp)
        d_ft, _ = cKDTree(tp).query(fp)
        chamfer[s][i] = np.mean(d_tf**2) + np.mean(d_ft**2)
        p95[s][i] = np.percentile(d_tf, 95)
        if s in fit_err:
            fit_err[s][i] = t_tree.query(fv)[0]  # fitted vertex -> target surface
            fp_dense, _ = trimesh.sample.sample_surface(fm, 200000, seed=1)
            tgt_err[(s, name)] = cKDTree(np.vstack([fp_dense, fv])).query(tv)[0].astype(np.float32)
    bx = posed["Stage_3_deform_fine"][i][body, 0]
    body_len[i] = bx.max() - bx.min()
    if i % 20 == 0:
        print(f"  {i}/{N} {name}: chamfer S1 {chamfer['Stage_1_default'][i]:.5f} S3 {chamfer['Stage_3_deform_fine'][i]:.5f}")

np.savez_compressed(os.path.join(a.out, "target_errors.npz"),
                    **{f"{s}__{n}": e for (s, n), e in tgt_err.items()})

# ---------- PCA of neutral-pose shapes (Stage 3, incl. un-posed deformation) ----------
X = neutral["Stage_3_deform_fine"].astype(np.float64)
X = X - X.mean(1, keepdims=True)
csize = np.sqrt((X**2).sum(2).mean(1))
Xn = X / csize[:, None, None]
mean_shape = Xn.mean(0)
D = (Xn - mean_shape).reshape(N, -1)
U, sv, Vt = np.linalg.svd(D, full_matrices=False)
var = sv**2 / (N - 1)
ratio = var / var.sum()
scores = U * sv  # (N, K)
sigma = np.sqrt(var)
K = 20
# region share of each PC's displacement
pc_region = []
for k in range(8):
    disp = (Vt[k].reshape(V, 3) ** 2).sum(1)
    pc_region.append({r: float(disp[vreg == r].sum() / disp.sum()) for r in region_names})
recon_err = []
for k in (5, 10, 20):
    R = (scores[:, :k] @ Vt[:k]).reshape(N, V, 3) + mean_shape
    recon_err.append((k, float(np.sqrt(((R - Xn) ** 2).sum(2)).mean())))
print("PCA variance ratio (first 8):", np.round(ratio[:8], 3), "cumulative 5/10:", ratio[:5].sum(), ratio[:10].sum())

# body length in the neutral normalised shape (for scale of PC renders/traits)
nb_len = Xn[:, body, 0].max(1) - Xn[:, body, 0].min(1)


def region_extent(shape, r, ax=None):
    m = vreg == r
    p = shape[m]
    return np.linalg.norm(p.max(0) - p.min(0)) if ax is None else p[:, ax].max() - p[:, ax].min()


traits = {
    "leg span / body length": [(Xn[i][np.isin(vreg, ["fore legs", "mid legs", "hind legs"]), 1].max()
                                 - Xn[i][np.isin(vreg, ["fore legs", "mid legs", "hind legs"]), 1].min()) / nb_len[i]
                                for i in range(N)],
    "head width / body length": [region_extent(Xn[i], "head", 1) / nb_len[i] for i in range(N)],
    "gaster length / body length": [region_extent(Xn[i], "gaster", 0) / nb_len[i] for i in range(N)],
    "mesosoma length / body length": [region_extent(Xn[i], "mesosoma", 0) / nb_len[i] for i in range(N)],
    "mandible size / body length": [region_extent(Xn[i], "mandibles") / nb_len[i] for i in range(N)],
    "antenna size / body length": [region_extent(Xn[i], "antennae") / nb_len[i] for i in range(N)],
}
trait_corr = {t: [float(np.corrcoef(np.asarray(v), scores[:, k])[0, 1]) for k in range(6)] for t, v in traits.items()}

# ---------- joint scales ----------
ls = S["Stage_3_deform_fine"]["log_beta_scales"]  # (N, NJ, 3)
joint_scale = np.exp(ls.mean(2))  # isotropic summary per joint

# ---------- loss curves from the log ----------
pat = re.compile(r"STAGE = (Stage_\d_[a-z_]+?)(?:_batch_(\d+))?, TOT_LOSS = ([0-9.]+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")
curves = {}
for line in open(a.log).read().replace("\r", "\n").split("\n"):
    m = pat.search(line)
    if m:
        st, b, loss, it, tot = m.group(1), int(m.group(2) or 0), float(m.group(3)), int(m.group(4)), int(m.group(5))
        curves.setdefault(f"{b}", {}).setdefault(st, {})[it] = loss
loss_curves = {b: {st: [v[k] for k in sorted(v)] for st, v in d.items()} for b, d in curves.items()}

np.savez_compressed(
    os.path.join(a.out, "analysis.npz"), labels=np.array(labels), faces=faces,
    neutral_s3=neutral["Stage_3_deform_fine"], neutral_s1=neutral["Stage_1_default"],
    posed_s1=posed["Stage_1_default"], posed_s3=posed["Stage_3_deform_fine"],
    Xn=Xn, csize=csize, mean_shape=mean_shape, pcs=Vt[:K], sigma=sigma[:K], scores=scores[:, :K], ratio=ratio,
    fit_err_s1=fit_err["Stage_1_default"], fit_err_s3=fit_err["Stage_3_deform_fine"],
    vreg=vreg, reg_idx=reg_idx, body_len=body_len, joint_scale=joint_scale, ls=ls,
)

report = {
    "labels": labels,
    "stages": stages,
    "chamfer": {s: chamfer[s].tolist() for s in stages},
    "p95": {s: p95[s].tolist() for s in stages},
    "body_len": body_len.tolist(),
    "region_names": region_names,
    "region_err": {s: [[float(fit_err[s][i][vreg == r].mean() / body_len[i]) for r in region_names] for i in range(N)]
                   for s in fit_err},
    "pca": {"ratio": ratio[:K].tolist(), "scores": scores[:, :K].tolist(), "sigma": sigma[:K].tolist(),
            "pc_region": pc_region, "recon_err": recon_err, "trait_corr": trait_corr,
            "traits": {t: [float(x) for x in v] for t, v in traits.items()}},
    "joints": {"names": names_j, "scale": joint_scale.tolist(),
               "region": [region_of(n) for n in names_j]},
    "loss_curves": loss_curves,
}
json.dump(report, open(os.path.join(a.out, "report_data.json"), "w"))
print("done")
