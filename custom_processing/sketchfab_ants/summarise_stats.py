"""Summarises the *_stats.json files written by prepare_sketchfab_models_for_mesh_fitting.py.
Usage: python summarise_stats.py <dir> [out_csv]
"""
import csv
import glob
import json
import os
import sys

rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*_stats.json"))):
    s = json.load(open(p))
    r, h, sh = s.get("remesh", {}), s["final_health"], s["source_health"]
    rows.append(dict(
        name=os.path.basename(p)[: -len("_stats.json")],
        source_verts=s["merged_vertices"], source_components=sh["components"],
        source_boundary=sh["boundary_edges"], source_nonmanifold=sh["non_manifold_edges"],
        voxel_rel=round(r.get("voxel_size_rel", 0), 5), attempts=len(r.get("attempts", [])),
        remeshed_verts=r.get("attempts", [{}])[-1].get("remeshed_vertices"),
        removed_area=f"{r.get('removed_area_fraction', 0):.1e}", fused=r.get("fused"),
        final_verts=s["final_vertices"], final_faces=s["final_faces"],
        components=h["components"], boundary=h["boundary_edges"], nonmanifold=h["non_manifold_edges"],
        watertight=h["watertight"], smooth_shift_edges=round(s["smoothing"]["mean_shift_in_median_edges"], 3),
        seconds=s["seconds"],
    ))
keys = list(rows[0].keys())
if len(sys.argv) > 2:
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
n = len(rows)
print(f"{n} models")
print("fused:", sum(bool(r["fused"]) for r in rows), "| single component:", sum(r["components"] == 1 for r in rows),
      "| watertight:", sum(bool(r["watertight"]) for r in rows))
print("voxel_rel used:", sorted({r["voxel_rel"] for r in rows}), "| retries:", sum(r["attempts"] > 1 for r in rows))
fv = [r["final_verts"] for r in rows]
print(f"final verts min {min(fv)} max {max(fv)}")
print("source components min/median/max:", min(r["source_components"] for r in rows),
      sorted(r["source_components"] for r in rows)[n // 2], max(r["source_components"] for r in rows))
odd = [r for r in rows if not r["fused"] or r["components"] != 1 or not r["watertight"] or r["attempts"] > 1
       or float(r["removed_area"]) > 1e-4 or r["final_verts"] < 90000]
for r in odd:
    print("CHECK", {k: r[k] for k in ("name", "voxel_rel", "attempts", "removed_area", "fused", "final_verts",
                                       "components", "boundary", "nonmanifold", "watertight")})
