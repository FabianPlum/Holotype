"""Stage 1 of the 2024 AntScan pipeline: outer envelope and vertex budget (Blender 4.2, headless).

The 2024 AntScan meshes (Biomedisa deep-learning full-body segmentation, numpy-stl 3.x headers, units in
micrometres) are watertight, consistently oriented marching-cubes surfaces of the whole body at about
1.17 scan voxels per edge (2-10M triangles). Their only internal geometry is
  (a) closed cavity shells inside the body (inward-facing, separate components) and
  (b) narrow pockets that are connected to the outside: coxal sockets, the neck, waist articulations and
      mouthparts (1-3 % of the main shell is never visible from outside).
Both are poison for mesh registration, so this stage keeps only the outer envelope.

Steps (process_mesh):
  1. Import the STL. Coordinates are never scaled: everything stays in micrometres, in the scan frame.
  2. Envelope: voxel remesh (OpenVDB) at h = envelope_voxels x scan voxel size, keep the largest loose part.
     Cavity shells become separate parts and are dropped; pockets narrower than about h are bridged.
  3. Snap back: every envelope vertex whose nearest scan surface point lies within snap_cells x h and whose
     normal agrees with that face's normal (angle < snap_max_angle) is moved onto that point. The remaining
     vertices (bridges over pockets) are relaxed with a few Laplacian steps. This removes the voxel grid's
     normal artefacts: on two test scans the final normals deviate from the scan's by a median of 3.7 deg,
     as for a direct decimation of the scan (3.6 deg).
  4. Quadric decimation (Decimate COLLAPSE) to a vertex budget that depends on the specimen's own shape,
     not its size:  N = clip(kappa^2 * 1.155 * A / V^(2/3), min_vertices, max_vertices),
     i.e. the vertex count of a uniform triangulation with edge length V^(1/3) / kappa (A = surface area,
     V = enclosed volume of the envelope). A specimen and a scaled copy of it get the same count, and leggy
     or spiny specimens (large A / V^(2/3)) get more vertices than compact ones.
  5. Write <out_dir>/<name>_envelope.stl (binary, micrometres, scan frame) and <name>_envelope.json with
     the parameters, counts and fidelity checks (see FIDELITY below).

FIDELITY (in scan voxels, from area-uniform samples):
  - output -> scan: distance from the output surface to the scan surface (added or bridged material);
  - scan -> output: distance from the scan surface to the output (removed internal surfaces and erosion),
    with the share of scan area farther than 2 voxels (the removed cavities and pockets).

Run:  blender --background --factory-startup --python-exit-code 1 -t 4 --python antscan_mesh_prep.py -- \
          <input.stl> <out_dir> [--voxel-um 2.44] [--kappa 60] [--max-vertices 100000]
If --voxel-um is not given, the specimen JSON next to the STL (field voxel_size, e.g. "2.44 µm") is used.
"""

import argparse
import json
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

DEFAULTS = {
    "envelope_voxels": 2.0,  # remesh cell size in scan voxels
    "snap_cells": 1.5,  # snap-back search radius in remesh cells
    "snap_max_angle": 60.0,  # degrees between envelope and scan normals for a vertex to be snapped
    "relax_iterations": 5,  # Laplacian steps for the unsnapped (bridge) vertices
    "kappa": 60.0,  # samples per characteristic length V^(1/3)
    "min_vertices": 20_000,
    "max_vertices": 100_000,
    "fidelity_samples": 100_000,
    "min_area_share": 0.5,  # exclude when the kept closed parts cover less of the scan area (fragmented scan)
    "min_part_area_share": 0.002,  # keep a disconnected outer part when it has at least this share of the main part's area
    "keep_external": "auto",  # auto (classify: anatomy kept, mounts/debris dropped) | none (main part only) | all (every outer part)
}


def mesh_arrays(ob):
    """Vertex coordinates (n, 3) and triangles (m, 3) of a mesh object (mesh-data coordinates)."""
    me = ob.data
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    return co.reshape(-1, 3), tri.reshape(-1, 3)


def area_volume(co, tri):
    """Surface area and enclosed (signed) volume of a closed triangle mesh."""
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return float(area), float(vol)


def median_edge_length(co, tri):
    e = np.concatenate([co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 1]], co[tri[:, 0]] - co[tri[:, 2]]])
    return float(np.median(np.linalg.norm(e, axis=1)))


def sample_surface(co, tri, n, rng):
    """n area-uniform points on a triangle mesh."""
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[pick] + u[:, None] * (b[pick] - a[pick]) + v[:, None] * (c[pick] - a[pick])


def distances(points, bvh):
    return np.array([bvh.find_nearest(Vector(p))[3] for p in points])


def boundary_loops(co, tri):
    """Boundary edges (used by one triangle) grouped into loops; returns a list of vertex-index arrays."""
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    b = ue[cnt == 1]
    if len(b) == 0:
        return []
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, c in b:
        ra, rc = find(int(a)), find(int(c))
        if ra != rc:
            parent[ra] = rc
    loops = {}
    for a, c in b:
        loops.setdefault(find(int(a)), set()).update((int(a), int(c)))
    return [np.array(sorted(s)) for s in loops.values()]


def classify_holes(co, loops, tol):
    """Split boundary loops into those lying in a bounding-box face plane (the scan volume cut the specimen:
    truncated anatomy) and all others. tol is the plane distance tolerance."""
    lo, hi = co.min(0), co.max(0)
    on_face, other = [], []
    for lp in loops:
        p = co[lp]
        faces = [
            f"{'xyz'[i]}{s}"
            for i in range(3)
            for s, ref in (("-", lo[i]), ("+", hi[i]))
            if np.all(np.abs(p[:, i] - ref) <= tol)
        ]
        rec = {
            "vertices": int(len(lp)),
            "span_um": np.round(p.max(0) - p.min(0), 1).tolist(),
            "bbox_face": faces[0] if faces else None,
        }
        (on_face if faces else other).append(rec)
    return on_face, other


def voxel_from_json(stl_path):
    j = os.path.splitext(stl_path)[0] + ".json"
    if not os.path.isfile(j):
        return None
    try:
        return float(json.load(open(j)).get("voxel_size", "").split()[0])
    except (ValueError, IndexError, AttributeError):
        return None


def select_only(ob):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def apply_modifier(ob, kind, **settings):
    m = ob.modifiers.new("M", kind)
    for k, v in settings.items():
        setattr(m, k, v)
    select_only(ob)
    bpy.ops.object.modifier_apply(modifier="M")


def keep_outer_parts(ob, min_part_area_share, mode="auto"):
    """Separate the remeshed surface into loose parts and keep the outer ones.

    The voxel remesh orients every closed shell consistently: the outer surface of a solid has outward normals
    (positive signed volume), the shell of an internal cavity faces into the cavity (negative signed volume). Only
    negative-volume shells are internal geometry; positive-volume parts are solids of the specimen that happen not to
    be connected to the main part (legs or a head detached in the scan, or joined by less than a remesh cell) and
    are kept when they cover at least min_part_area_share of the main part's area (2026-11-09: dropping them lost
    two right legs of Brachymyrmex_depilis_CASENT0709256, 13.5 % of its surface).
    Returns the joined object and {"kept_external": [...], "dropped": [...]} with faces/area/volume per part."""
    select_only(ob)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    stats = []
    for o in parts:
        co, tri = mesh_arrays(o)
        a, v = area_volume(co, tri)
        stats.append((len(o.data.polygons), a, v, o))
    stats.sort(key=lambda s: s[0], reverse=True)
    main = stats[0]
    keep, kept, dropped, flags = [main[3]], [], [], []
    for s in stats[1:]:
        rec = {"faces": s[0], "area_um2": round(s[1], 1), "volume_um3": round(s[2], 1)}
        share = s[1] / max(main[1], 1e-12)
        if s[2] <= 0 or share < min_part_area_share:
            rec["why"] = "cavity shell (negative volume)" if s[2] <= 0 else "tiny"
            dropped.append(rec)
            bpy.data.objects.remove(s[3], do_unlink=True)
            continue
        # outer part: anatomy (detached legs, antennae, a head) or a foreign object (card point, glue, soil grain)?
        # A plate (card point, mount) has its face normals concentrated along one axis: the area-weighted normal
        # tensor has one dominant eigenvalue. Legs (cylinders, even folded bundles) spread their normals over two
        # axes, blobs over three. The principal extents of the vertices then separate elongated from compact parts.
        co, tri = mesh_arrays(s[3])
        n = np.cross(co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 0]])
        a_f = 0.5 * np.linalg.norm(n, axis=1)
        n = n / np.maximum(2 * a_f[:, None], 1e-30)
        T = (n * a_f[:, None]).T @ n / max(a_f.sum(), 1e-30)
        normal_dominance = float(
            np.linalg.eigvalsh(T)[-1]
        )  # share of the normal energy along the main axis (1/3 isotropic, 1 flat)
        ext = np.sqrt(np.sort(np.linalg.eigvalsh(np.cov((co - co.mean(0)).T)))[::-1])  # principal extents, descending
        e1, e2, e3 = (float(x) for x in ext)
        if normal_dominance > 0.6:
            rec["shape"] = (
                "plate"  # card point or other mount (Carebara_affinis_CASENT0741335 carries one, 58 % of the ant's area)
            )
        elif e1 > 2.5 * e2:
            rec["shape"] = "elongated"  # leg, antenna
        else:
            rec["shape"] = "compact"  # soil grain / glue blob, or a detached head when large
        rec["normal_dominance"] = round(normal_dominance, 3)
        rec["extents_um"] = [round(e1, 1), round(e2, 1), round(e3, 1)]
        rec["area_share"] = round(share, 4)
        if mode == "none" or (
            mode == "auto" and (rec["shape"] == "plate" or (rec["shape"] == "compact" and share < 0.05))
        ):
            rec["why"] = "foreign object (" + rec["shape"] + ")"
            dropped.append(rec)
            flags.append(f"dropped_{rec['shape']}_part_{100 * share:.1f}pct")
            bpy.data.objects.remove(s[3], do_unlink=True)
            continue
        keep.append(s[3])
        kept.append(rec)
        if share >= 0.05:
            flags.append(f"kept_large_{rec['shape']}_part_{100 * share:.1f}pct")  # to be looked at in the review
    if len(keep) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in keep:
            o.select_set(True)
        bpy.context.view_layer.objects.active = main[3]
        bpy.ops.object.join()
    return main[3], {"kept_external": kept, "dropped": dropped, "main_area_um2": round(main[1], 1), "flags": flags}


def snap_to_scan(ob, bvh_scan, max_dist, max_angle_deg, relax_iterations):
    """Move envelope vertices onto the scan surface where the scan is close and faces the same way.

    Returns the share of snapped vertices. The vertices that are not snapped (bridges over pockets and
    cavity openings) are relaxed toward their neighbours' mean.
    """
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nrm = np.empty(n * 3)
    me.vertex_normals.foreach_get("vector", nrm)
    nrm = nrm.reshape(-1, 3)
    cos_max = math.cos(math.radians(max_angle_deg))
    snapped = np.zeros(n, dtype=bool)
    for i in range(n):
        loc, fn, _idx, _d = bvh_scan.find_nearest(Vector(co[i]), max_dist)
        if loc is not None and float(np.dot(nrm[i], fn)) >= cos_max:
            co[i] = loc
            snapped[i] = True
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    free = np.nonzero(~snapped)[0]
    if len(free) and relax_iterations:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        verts = [bm.verts[i] for i in free]
        for _ in range(relax_iterations):
            new = []
            for v in verts:
                nb = [e.other_vert(v).co for e in v.link_edges]
                new.append(sum(nb, Vector()) / len(nb) if nb else v.co.copy())
            for v, p in zip(verts, new):
                v.co = v.co * 0.5 + p * 0.5
        bm.to_mesh(me)
        bm.free()
        me.update()
    return float(snapped.mean())


def vertex_budget(area, volume, kappa, n_min, n_max):
    """Vertex count of a uniform triangulation of area A with edge e = V^(1/3) / kappa.

    Equilateral triangles of edge e have area sqrt(3)/4 e^2 and a closed surface has about two triangles
    per vertex, so N = A / (sqrt(3)/2 e^2) = (2/sqrt(3)) kappa^2 A / V^(2/3). Clipped to [n_min, n_max].
    """
    complexity = area / volume ** (2.0 / 3.0)  # dimensionless; a sphere has 4.84
    n = kappa**2 * (2.0 / math.sqrt(3.0)) * complexity  # = A / (sqrt(3)/2 * e^2), one vertex per two triangles
    return int(round(min(max(n, n_min), n_max))), complexity


def decimate_to_vertices(ob, n_vertices, n_max):
    """Quadric decimation to about n_vertices (2 triangles per vertex), never above n_max vertices."""
    for _ in range(3):
        co, tri = mesh_arrays(ob)
        if len(co) <= n_vertices:
            break
        apply_modifier(
            ob,
            "DECIMATE",
            decimate_type="COLLAPSE",
            use_collapse_triangulate=True,
            ratio=min(1.0, 2 * n_vertices / len(tri)),
        )
        if len(ob.data.vertices) <= n_max:
            break
        n_vertices = int(n_vertices * 0.97)
    return len(ob.data.vertices)


def process_mesh(stl_path, out_dir, voxel_um=None, **params):
    p = dict(DEFAULTS, **{k: v for k, v in params.items() if v is not None})
    t0 = time.time()
    name = os.path.splitext(os.path.basename(stl_path))[0]
    rng = np.random.default_rng(0)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=stl_path)
    scan = bpy.context.selected_objects[0]
    assert np.allclose(np.array(scan.matrix_world), np.eye(4)), "STL import applied a transform"
    co0, tri0 = mesh_arrays(scan)
    edge = median_edge_length(co0, tri0)
    voxel_meta = voxel_um if voxel_um is not None else voxel_from_json(stl_path)
    # Units: the 2024 meshes are in micrometres (median edge = 1.17 scan voxels). A few AntScan STLs are in
    # voxel units (median edge ~1.17 file units); the header does not tell, so check the edge length.
    input_units = "micrometre"
    if voxel_meta and abs(edge / 1.17 - 1.0) < 0.3 and abs(edge / 1.17 / voxel_meta - 1.0) > 0.3:
        input_units = "voxel"
        co0 = co0 * voxel_meta
        scan.data.vertices.foreach_set("co", co0.ravel())
        scan.data.update()
        edge *= voxel_meta
    voxel_edge = edge / 1.17  # 2024 meshes: median edge = 1.17 scan voxels
    # The metadata voxel is the physical resolution; a disagreement with the edge-based estimate is only flagged
    # (a coarser remesh than the scan's resolution would erase its thinnest parts).
    voxel = voxel_meta if voxel_meta else voxel_edge
    voxel_mismatch = bool(voxel_meta and abs(voxel_edge / voxel_meta - 1) > 0.3)
    # Open surfaces: holes in a bounding-box face plane mean the scan's field of view cut the specimen
    # (truncated anatomy, unusable); other holes are filled before the envelope is built.
    loops = boundary_loops(co0, tri0)
    cut, other_holes = classify_holes(co0, loops, tol=0.5 * voxel)
    os.makedirs(out_dir, exist_ok=True)
    if cut:
        info = {
            "name": name,
            "stage": "antscan_mesh_prep (envelope)",
            "status": "excluded",
            "reason": "truncated by the scan's field of view (open surface in a bounding-box face plane)",
            "holes_in_bbox_faces": cut,
            "other_holes": other_holes,
            "input": {"stl": os.path.abspath(stl_path), "triangles": int(len(tri0)), "units": input_units},
            "voxel_um": {"used": round(voxel, 4), "metadata": voxel_meta, "from_edge_length": round(voxel_edge, 4)},
        }
        with open(os.path.join(out_dir, f"{name}_envelope.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(
            f"[prep] {name}: EXCLUDED - {info['reason']}: {len(cut)} cut loop(s), largest span {max((c['span_um'] for c in cut), key=lambda s: max(s))} um",
            flush=True,
        )
        return info
    if other_holes:
        select_only(scan)
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(scan.data)
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        bmesh.update_edit_mesh(scan.data)
        bpy.ops.object.mode_set(mode="OBJECT")
        co0, tri0 = mesh_arrays(scan)
    area0, vol0 = area_volume(co0, tri0)
    bvh_scan = BVHTree.FromPolygons(co0.tolist(), tri0.tolist(), all_triangles=True)
    t1 = time.time()

    # 2. envelope (envelope_voxels <= 0 skips it: only for meshes without internal geometry, e.g. synthetic ones)
    h = p["envelope_voxels"] * voxel
    env = scan.copy()
    env.data = scan.data.copy()
    bpy.context.collection.objects.link(env)
    if h > 0:
        apply_modifier(env, "REMESH", mode="VOXEL", voxel_size=h, adaptivity=0.0)
        faces_remesh = len(env.data.polygons)
        env, part_stats = keep_outer_parts(env, p["min_part_area_share"], p["keep_external"])
    else:
        faces_remesh, part_stats = 0, {"kept_external": [], "dropped": [], "main_area_um2": 0.0, "flags": []}
    other_parts = part_stats["dropped"]
    t2 = time.time()

    # 3. snap back onto the scan surface
    snapped_share = (
        snap_to_scan(env, bvh_scan, p["snap_cells"] * h, p["snap_max_angle"], p["relax_iterations"]) if h > 0 else 1.0
    )
    co_e, tri_e = mesh_arrays(env)
    area_e, vol_e = area_volume(co_e, tri_e)
    t3 = time.time()
    # Fragmented scans (porous or shattered surfaces) remesh into thousands of shells; the largest one is then a
    # sliver of the specimen (Colobopsis_shohki_OKENT0105042: 0.6 % of the scan area). Such a specimen has no usable
    # outer surface and is excluded instead of silently producing a wrong envelope.
    area_share = area_e / max(area0, 1e-12)
    if h > 0 and area_share < p["min_area_share"]:
        info = {
            "name": name,
            "stage": "antscan_mesh_prep (envelope)",
            "status": "excluded",
            "reason": f"fragmented surface: the largest closed part covers only {100 * area_share:.1f} % of the scan area "
            f"({len(other_parts)} other parts, largest {100 * max((o['area_um2'] for o in other_parts), default=0.0) / max(area0, 1e-12):.1f} %)",
            "input": {
                "stl": os.path.abspath(stl_path),
                "triangles": int(len(tri0)),
                "area_um2": round(area0, 1),
                "signed_volume_um3": round(vol0, 1),
                "units": input_units,
            },
            "voxel_um": {"used": round(voxel, 4), "metadata": voxel_meta, "from_edge_length": round(voxel_edge, 4)},
            "envelope": {
                "remesh_cell_um": round(h, 4),
                "remesh_faces": int(faces_remesh),
                "dropped_parts": len(other_parts),
                "kept_external_parts": part_stats["kept_external"][:10],
                "largest_part_area_um2": round(area_e, 1),
            },
        }
        with open(os.path.join(out_dir, f"{name}_envelope.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"[prep] {name}: EXCLUDED - {info['reason']}", flush=True)
        return info

    # 4. vertex budget and quadric decimation
    n_target, complexity = vertex_budget(area_e, vol_e, p["kappa"], p["min_vertices"], p["max_vertices"])
    n_final = decimate_to_vertices(env, n_target, p["max_vertices"])
    co_f, tri_f = mesh_arrays(env)
    area_f, vol_f = area_volume(co_f, tri_f)
    t4 = time.time()

    # fidelity (scan voxels)
    bvh_f = BVHTree.FromPolygons(co_f.tolist(), tri_f.tolist(), all_triangles=True)
    s_out = sample_surface(co_f, tri_f, p["fidelity_samples"], rng)
    s_scan = sample_surface(co0, tri0, p["fidelity_samples"], rng)
    d_out = distances(s_out, bvh_scan) / voxel
    d_scan = distances(s_scan, bvh_f) / voxel
    t5 = time.time()

    def stats(d):
        return {
            "median": round(float(np.median(d)), 3),
            "p99": round(float(np.percentile(d, 99)), 3),
            "p99.9": round(float(np.percentile(d, 99.9)), 3),
            "max": round(float(d.max()), 3),
        }

    os.makedirs(out_dir, exist_ok=True)
    stl_out = os.path.join(out_dir, f"{name}_envelope.stl")
    select_only(env)
    bpy.ops.wm.stl_export(filepath=stl_out, export_selected_objects=True, ascii_format=False, apply_modifiers=True)
    main_area = max(area_e, 1e-12)
    info = {
        "name": name,
        "stage": "antscan_mesh_prep (envelope)",
        "status": "ok",
        "holes_filled": other_holes,
        "units": "micrometre",
        "frame": "scan (unrotated, unscaled input coordinates)",
        "input": {
            "stl": os.path.abspath(stl_path),
            "triangles": int(len(tri0)),
            "vertices": int(len(co0)),
            "area_um2": round(area0, 1),
            "signed_volume_um3": round(vol0, 1),
            "median_edge_um": round(edge, 4),
            "units": input_units,
            "converted_to_micrometre": input_units != "micrometre",
        },
        "voxel_um": {
            "used": round(voxel, 4),
            "metadata": voxel_meta,
            "from_edge_length": round(voxel_edge, 4),
            "metadata_mismatch": voxel_mismatch,
        },
        "params": p,
        "envelope": {
            "remesh_cell_um": round(h, 4),
            "remesh_faces": int(faces_remesh),
            "dropped_parts": len(other_parts),
            "dropped_parts_largest": other_parts[:5],
            "dropped_part_area_share_max": round(max((o["area_um2"] for o in other_parts), default=0.0) / main_area, 6),
            "kept_external_parts": len(part_stats["kept_external"]),
            "kept_external_parts_largest": part_stats["kept_external"][:5],
            "kept_external_area_share": round(sum(o["area_um2"] for o in part_stats["kept_external"]) / main_area, 6),
            "part_flags": part_stats["flags"],  # large kept parts and dropped foreign objects: review these specimens
            "snapped_vertex_share": round(snapped_share, 4),
            "area_um2": round(area_e, 1),
            "volume_um3": round(vol_e, 1),
            "shape_complexity_A_over_V23": round(complexity, 3),
            "characteristic_length_um": round(vol_e ** (1 / 3), 3),
        },
        "output": {
            "stl": os.path.abspath(stl_out),
            "vertex_budget": n_target,
            "vertices": int(n_final),
            "triangles": int(len(tri_f)),
            "area_um2": round(area_f, 1),
            "volume_um3": round(vol_f, 1),
        },
        "fidelity_scan_voxels": {
            "output_to_scan": stats(d_out),
            "scan_to_output": stats(d_scan),
            "scan_area_share_removed_gt2vox": round(float((d_scan > 2).mean()), 5),
        },
        "seconds": {
            "import": round(t1 - t0, 1),
            "envelope": round(t2 - t1, 1),
            "snap": round(t3 - t2, 1),
            "decimate": round(t4 - t3, 1),
            "fidelity": round(t5 - t4, 1),
        },
    }
    with open(os.path.join(out_dir, f"{name}_envelope.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"[prep] {name}: {len(tri0)} -> {len(tri_f)} triangles, {n_final} vertices (budget {n_target}, complexity {complexity:.2f}); "
        f"scan area removed (>2 vox) {100 * info['fidelity_scan_voxels']['scan_area_share_removed_gt2vox']:.2f} %, "
        f"output->scan p99 {info['fidelity_scan_voxels']['output_to_scan']['p99']} vox; {time.time() - t0:.0f}s",
        flush=True,
    )
    return info


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("out_dir")
    ap.add_argument("--voxel-um", type=float, default=None)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    a = ap.parse_args(argv)
    params = {k: getattr(a, k) for k in DEFAULTS}
    process_mesh(a.stl, a.out_dir, voxel_um=a.voxel_um, **params)


if __name__ == "__main__":
    main()
