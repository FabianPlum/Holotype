"""Builds Blender render jobs, 3D-viewer binaries and chart data for the OmniAnt fit report.

Needs report/analyse_fit.py to have written DATA/report_build/{analysis.npz, report_data.json, target_errors.npz}.
Writes DATA/report_build/render/{npy/, jobs_0..3.json} (rendered by render_jobs.py) and
DATA/report_site/data/{page.json, *.u16/*.f32/*.i16/*.u8}.

Usage: python build_report_assets.py DATA
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ant_taxa import display_name, subfamily_of  # noqa: E402

sys.path.insert(0, HERE)
from render_common import BLUE, CHITIN, GREY, HEAT, heat  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
RB = os.path.join(ROOT, "report_build")
RN = os.path.join(RB, "render")
SITE = os.path.join(ROOT, "report_site")
for d in (os.path.join(RN, "npy"), os.path.join(RN, "png"), os.path.join(SITE, "data"), os.path.join(SITE, "img")):
    os.makedirs(d, exist_ok=True)

A = np.load(os.path.join(RB, "analysis.npz"), allow_pickle=True)
R = json.load(open(os.path.join(RB, "report_data.json")))
TE = np.load(os.path.join(RB, "target_errors.npz"))
labels = [str(x) for x in A["labels"]]
N = len(labels)
faces = A["faces"].astype(np.int32)
V = A["Xn"].shape[1]
bl = np.array(R["body_len"])

# ---------- names and subfamilies (pinned list; same-species duplicates get the upload month) ----------
pinned = {m["key"]: m for m in json.load(open(os.path.join(os.path.dirname(HERE), "tunosemi_ant_models.json")))["models"]}
names = [display_name(pinned[k]["name"]) for k in labels]
for n in set(names):
    dup = [i for i, x in enumerate(names) if x == n]
    if len(dup) > 1:
        for i in dup:
            names[i] = f"{n} ({pinned[labels[i]]['published'][:7]} upload)"
subfam = [subfamily_of(pinned[k]["name"]) for k in labels]
assert all(subfam), [k for k, s in zip(labels, subfam) if not s]

def save(name, arr):
    p = os.path.join(RN, "npy", name)
    np.save(p, arr)
    return p


jobs = []
F = save("faces.npy", faces)
png = lambda n: os.path.join(RN, "png", n)  # noqa: E731

# ---------- 1. drawer / morphospace thumbnails: neutral shapes ----------
Xn = A["Xn"].astype(np.float32)
XN = save("neutral_all.npy", Xn)
ext_xy = float(np.max(np.abs(Xn[:, :, :2]))) * 2 * 1.02
for i in range(N):
    jobs.append(dict(verts=XN, index=i, faces=F, color=CHITIN, view="top", ortho_scale=ext_xy, centre=[0, 0, 0],
                     res=[260, 260], out=png(f"thumb_{i:03d}.png")))

# ---------- 2. PC extremes coloured by displacement from the mean ----------
mean = A["mean_shape"].astype(np.float32)
pcs, sig = A["pcs"], A["sigma"]
pc_shapes, pc_cols, pc_meta = [mean], [np.tile(np.array(GREY, np.uint8), (V, 1))], []
for k in range(4):
    disp = pcs[k].reshape(V, 3) * sig[k] * 2.5
    mag = np.linalg.norm(disp, axis=1)
    for sgn in (-1, 1):
        pc_shapes.append(mean + sgn * disp)
        pc_cols.append(heat(mag / mag.max()))
        pc_meta.append((k, sgn))
PS = save("pc_shapes.npy", np.stack(pc_shapes).astype(np.float32))
PCC = save("pc_cols.npy", np.stack(pc_cols))
allpc = np.stack(pc_shapes)
sc_top = float(np.max(np.abs(allpc[:, :, :2]))) * 2 * 1.03
sc_side = float(np.max(np.abs(allpc[:, :, 0]))) * 2 * 1.03
for j in range(len(pc_shapes)):
    tag = "mean" if j == 0 else f"pc{pc_meta[j - 1][0] + 1}_{'neg' if pc_meta[j - 1][1] < 0 else 'pos'}"
    jobs.append(dict(verts=PS, index=j, faces=F, colors=PCC, view="top", ortho_scale=sc_top, centre=[0, 0, 0],
                     res=[520, 520], out=png(f"{tag}_top.png")))
    jobs.append(dict(verts=PS, index=j, faces=F, colors=PCC, view="side", ortho_scale=sc_side, centre=[0, 0, 0],
                     res=[520, 260], out=png(f"{tag}_side.png")))

# ---------- 3. error atlas on the mean shape ----------
atlas_max = 1.5  # % body length
atlas_vals = {tag: (A[key] / bl[:, None] * 100).mean(0) for tag, key in (("s1", "fit_err_s1"), ("s3", "fit_err_s3"))}
m = save("mean_shape.npy", mean)
for tag in ("s1", "s3"):
    c = save(f"atlas_{tag}.npy", heat(atlas_vals[tag] / atlas_max))
    for view, res in (("top", [560, 560]), ("side", [560, 280]), ("persp", [640, 480])):
        jobs.append(dict(verts=m, faces=F, colors=c, view=view,
                         ortho_scale={"top": sc_top, "side": sc_side, "persp": sc_side * 0.62}[view],
                         centre=[0, 0, 0], res=res, out=png(f"atlas_{tag}_{view}.png")))

# ---------- 4. per-specimen fit tiles (fit frame, fixed camera) ----------
tile_max = 3.0  # % body length
PS3 = save("posed_s3.npy", A["posed_s3"].astype(np.float32))
for i, key in enumerate(labels):
    tm = trimesh.load(os.path.join(ROOT, "processed", f"{key}.obj"), process=False, force="mesh")
    tv = np.asarray(tm.vertices, np.float64)
    tv = tv - tv.mean(0)
    tvp = save(f"t_{i:03d}_v.npy", (tv / np.abs(tv).max()).astype(np.float32))
    tfp = save(f"t_{i:03d}_f.npy", np.asarray(tm.faces, np.int32))
    for st in ("Stage_1_default", "Stage_3_deform_fine"):
        cp = save(f"t_{i:03d}_{st[:7]}.npy", heat(TE[f"{st}__{key}"] / bl[i] * 100 / tile_max))
        jobs.append(dict(verts=tvp, faces=tfp, colors=cp, view="top", ortho_scale=2.08, centre=[0, 0, 0],
                         res=[300, 300], out=png(f"tile_{i:03d}_{st[:7]}.png")))
    jobs.append(dict(verts=PS3, index=i, faces=F, color=BLUE, view="top", ortho_scale=2.08, centre=[0, 0, 0],
                     res=[300, 300], out=png(f"tile_{i:03d}_fit.png")))

for k in range(4):
    json.dump(jobs[k::4], open(os.path.join(RN, f"jobs_{k}.json"), "w"))
print(len(jobs), "render jobs")

# ---------- viewer binaries ----------
D = os.path.join(SITE, "data")
faces.astype(np.uint16).tofile(os.path.join(D, "faces.u16"))
mean.astype(np.float32).tofile(os.path.join(D, "mean.f32"))
K = 12
pcs_k = (pcs[:K] * sig[:K, None]).astype(np.float32)  # 1-sigma displacement fields
pc_scale = np.abs(pcs_k).max(1) / 32767
(pcs_k / pc_scale[:, None]).round().astype(np.int16).tofile(os.path.join(D, "pcs.i16"))
delta = (Xn - mean[None]).reshape(N, -1)
d_scale = float(np.abs(delta).max() / 32767)
(delta / d_scale).round().astype(np.int16).tofile(os.path.join(D, "shapes.i16"))
np.clip(atlas_vals["s1"] / atlas_max * 255, 0, 255).round().astype(np.uint8).tofile(os.path.join(D, "atlas_s1.u8"))

# ---------- chart / page data ----------
scores = np.asarray(A["scores"])[:, :K] / np.asarray(sig[:K])[None]  # in sigma units
page = {
    "n": N, "V": V, "F": int(len(faces)), "K": K,
    "pc_scale": pc_scale.tolist(), "shape_scale": d_scale,
    "specimens": [dict(key=labels[i], name=names[i], subfamily=subfam[i],
                       chamfer=[R["chamfer"][s][i] for s in R["stages"]],
                       p95=[R["p95"][s][i] / bl[i] * 100 for s in R["stages"]],
                       region_err_s1=[x * 100 for x in R["region_err"]["Stage_1_default"][i]],
                       region_err_s3=[x * 100 for x in R["region_err"]["Stage_3_deform_fine"][i]],
                       pc=[round(float(x), 3) for x in scores[i]],
                       joint_scale=[round(float(x), 3) for x in R["joints"]["scale"][i]],
                       url=f"https://sketchfab.com/3d-models/{pinned[labels[i]]['uid']}")
                  for i in range(N)],
    "order_pc1": np.argsort(scores[:, 0]).tolist(),
    "stages": R["stages"],
    "region_names": R["region_names"],
    "pca": {"ratio": R["pca"]["ratio"][:K], "pc_region": R["pca"]["pc_region"][:6],
            "trait_corr": R["pca"]["trait_corr"], "recon_err": R["pca"]["recon_err"]},
    "joints": {"names": R["joints"]["names"], "region": R["joints"]["region"]},
    "loss": R["loss_curves"],
    "scales": {"tile_max": tile_max, "atlas_max": atlas_max},
    "heat": HEAT.astype(int).tolist(),
}
json.dump(page, open(os.path.join(SITE, "data", "page.json"), "w"), separators=(",", ":"))
print("viewer and page data written; subfamilies:", {s: subfam.count(s) for s in sorted(set(subfam), key=subfam.count, reverse=True)})
