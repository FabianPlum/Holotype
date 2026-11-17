"""Surface distances between registered meshes and the processed targets, for comparing registration methods.

Same metric as report/analyse_fit.py (and expected/baseline_fitter3d_metrics.csv): both surfaces are sampled
with 30,000 area-weighted points (trimesh, seed 0) in the fitter_3d frame, i.e. the target centred on its vertex
mean and divided by its largest absolute coordinate (fitter_3d.utils.load_meshes). Reported per specimen:
  chamfer  mean squared nearest distance target->fit plus fit->target
  p95      95th percentile of target->fit distances (how far the worst-covered target surface is from the fit)
The metric does not depend on the template, so any method's output can be scored against the same targets.

Inputs (one of):
  --npz FILE        fitter_3d stage file (verts (N,V,3), faces, labels = target file names)
  --mesh_dir DIR    one OBJ/PLY/STL per specimen, named like the target (<key>.obj)
--frame fitter (default): the registered vertices are already in the fitter_3d frame (true for fitter_3d output);
--frame target: they are in the processed target's own coordinates and are normalised with the target's centre/scale.

Usage:
  python custom_processing/sketchfab_ants/evaluate_registration.py --targets DATA/processed --npz RESULTS/Stage_3_deform_fine.npz \
      --out DATA/diagnostics/metrics_fitter3d.csv [--baseline_stage deform_fine]
"""
import argparse
import csv
import glob
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "expected", "baseline_fitter3d_metrics.csv")


def load_target(path):
    m = trimesh.load(path, process=False, force="mesh")
    v = np.asarray(m.vertices, np.float64)
    centre = v.mean(0)
    scale = np.abs(v - centre).max()
    return trimesh.Trimesh((v - centre) / scale, m.faces, process=False), centre, scale


def surface_metrics(target_mesh, fit_mesh, n_points=30000, seed=0):
    tp, _ = trimesh.sample.sample_surface(target_mesh, n_points, seed=seed)
    fp, _ = trimesh.sample.sample_surface(fit_mesh, n_points, seed=seed)
    d_tf, _ = cKDTree(fp).query(tp)
    d_ft, _ = cKDTree(tp).query(fp)
    return float(np.mean(d_tf**2) + np.mean(d_ft**2)), float(np.percentile(d_tf, 95))


def registered_meshes(a):
    if a.npz:
        d = np.load(a.npz, allow_pickle=True)
        faces = d["faces"][0] if d["faces"].ndim == 3 else d["faces"]
        for label, v in zip(d["labels"], d["verts"]):
            key = os.path.splitext(os.path.basename(str(label)))[0]
            yield key, np.asarray(v, np.float64), np.asarray(faces)
    else:
        for p in sorted(glob.glob(os.path.join(a.mesh_dir, "*"))):
            if os.path.splitext(p)[1].lower() in (".obj", ".ply", ".stl"):
                m = trimesh.load(p, process=False, force="mesh")
                yield os.path.splitext(os.path.basename(p))[0], np.asarray(m.vertices, np.float64), np.asarray(m.faces)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="Directory of processed target OBJs (<key>.obj)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", help="fitter_3d stage .npz")
    src.add_argument("--mesh_dir", help="Directory of registered meshes named <key>.obj/.ply/.stl")
    ap.add_argument("--frame", choices=["fitter", "target"], default="fitter")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--n_points", type=int, default=30000)
    ap.add_argument("--baseline_stage", default="deform_fine",
                    choices=["init", "pose_shape", "deform_coarse", "deform_fine"],
                    help="fitter_3d stage of the 2026-11-15 run to compare against")
    a = ap.parse_args()

    rows = []
    for key, v, f in registered_meshes(a):
        tpath = os.path.join(a.targets, f"{key}.obj")
        if not os.path.exists(tpath):
            print(f"skip {key}: no target {tpath}")
            continue
        tm, centre, scale = load_target(tpath)
        if a.frame == "target":
            v = (v - centre) / scale
        ch, p95 = surface_metrics(tm, trimesh.Trimesh(v, f, process=False), a.n_points)
        rows.append(dict(key=key, chamfer=f"{ch:.6e}", p95=f"{p95:.6f}"))
        print(f"{key}: chamfer {ch:.2e}  p95 {p95:.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "chamfer", "p95"])
        w.writeheader()
        w.writerows(rows)

    base = {r["key"]: r for r in csv.DictReader(open(BASELINE))}
    common = [r for r in rows if r["key"] in base]
    if common:
        ch = np.array([float(r["chamfer"]) for r in common])
        bch = np.array([float(base[r["key"]][f"chamfer_{a.baseline_stage}"]) for r in common])
        p = np.array([float(r["p95"]) for r in common])
        bp = np.array([float(base[r["key"]][f"p95_{a.baseline_stage}"]) for r in common])
        print(f"\n{len(common)} specimens vs fitter_3d baseline ({a.baseline_stage}):")
        print(f"  chamfer median {np.median(ch):.2e} (baseline {np.median(bch):.2e}), better on {(ch < bch).sum()}")
        print(f"  p95     median {np.median(p):.4f} (baseline {np.median(bp):.4f}), better on {(p < bp).sum()}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
