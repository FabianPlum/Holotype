"""Extra render jobs: pipeline strip for a showcase specimen and perspective failure cases.
Usage: python build_extra_renders.py <tunosemi_ants dir> <showcase key> <source_shells.obj> <failure keys...>
"""
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_common import BLUE, GREY, heat  # noqa: E402

ROOT, KEY, SRC = sys.argv[1:4]
FAIL = sys.argv[4:]
RB = os.path.join(ROOT, "report_build")
RN = os.path.join(RB, "render")
A = np.load(os.path.join(RB, "analysis.npz"), allow_pickle=True)
TE = np.load(os.path.join(RB, "target_errors.npz"))
labels = [str(x) for x in A["labels"]]
faces = A["faces"].astype(np.int32)
bl = np.array(json.load(open(os.path.join(RB, "report_data.json")))["body_len"])
def save(name, arr):
    p = os.path.join(RN, "npy", name)
    np.save(p, arr)
    return p


png = lambda n: os.path.join(RN, "png", n)  # noqa: E731
F = save("faces.npy", faces)
stages = {s: np.load(os.path.join(ROOT, "fit3d_results_OmniAnt", f"{s}.npz"))["verts"]
          for s in ("Stage_0_init", "Stage_1_default", "Stage_3_deform_fine")}


def target(key):
    m = trimesh.load(os.path.join(ROOT, "processed", f"{key}.obj"), process=False, force="mesh")
    v = np.asarray(m.vertices, np.float64)
    mu, s = v.mean(0), np.abs(v - v.mean(0)).max()
    return ((v - mu) / s).astype(np.float32), np.asarray(m.faces, np.int32), mu, s


jobs = []
i = labels.index(KEY)
tv, tf, mu, scl = target(KEY)
st = json.load(open(os.path.join(ROOT, "intermediate_source_axes", f"{KEY}_stats.json")))
src = trimesh.load(SRC, process=False, force="mesh")
sv = np.asarray(src.vertices, np.float64) - np.asarray(st["bbox_centre_removed"])
sv = np.stack([-sv[:, 1], sv[:, 0], sv[:, 2]], 1)  # Rz(+90), the rotation all but one model received
sv = ((sv - mu) / scl).astype(np.float32)
cam = dict(view="persp", ortho_scale=2.2, centre=[0, 0, -0.05], res=[640, 480])
jobs.append(dict(verts=save("pipe_src_v.npy", sv), faces=save("pipe_src_f.npy", np.asarray(src.faces, np.int32)),
                 random_parts=True, out=png("pipe_1_source.png"), **cam))
TV, TF = save("pipe_tgt_v.npy", tv), save("pipe_tgt_f.npy", tf)
jobs.append(dict(verts=TV, faces=TF, color=GREY, out=png("pipe_2_target.png"), **cam))
for n, s in (("pipe_3_init", "Stage_0_init"), ("pipe_4_pose_shape", "Stage_1_default"), ("pipe_5_deform", "Stage_3_deform_fine")):
    jobs.append(dict(verts=save(f"{n}.npy", stages[s][i].astype(np.float32)), faces=F, color=BLUE, out=png(f"{n}.png"), **cam))
e = TE[f"Stage_3_deform_fine__{KEY}"] / bl[i] * 100
jobs.append(dict(verts=TV, faces=TF, colors=save("pipe_6_err.npy", heat(e / 3.0)), out=png("pipe_6_error.png"), **cam))

for k in FAIL:
    j = labels.index(k)
    fv, ff, _, _ = target(k)
    jobs.append(dict(verts=save(f"fail_{j}_fit.npy", stages["Stage_3_deform_fine"][j].astype(np.float32)), faces=F,
                     color=BLUE, out=png(f"fail_{j:03d}_fit.png"), **cam))
    e = TE[f"Stage_1_default__{k}"] / bl[j] * 100
    jobs.append(dict(verts=save(f"fail_{j}_tv.npy", fv), faces=save(f"fail_{j}_tf.npy", ff),
                     colors=save(f"fail_{j}_err.npy", heat(e / 3.0)), out=png(f"fail_{j:03d}_err_s1.png"), **cam))
json.dump(jobs, open(os.path.join(RN, "jobs_extra.json"), "w"))
print(len(jobs), "jobs; showcase index", i, "failure indices", [labels.index(k) for k in FAIL])
