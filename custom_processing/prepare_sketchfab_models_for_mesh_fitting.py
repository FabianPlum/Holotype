"""
Prepares downloaded Sketchfab models (e.g. tunosemi's ant models) for mesh fitting.

These models are hand-built surface meshes, often assembled from one closed shell per body
segment (antennomeres, leg segments, ...). The shells touch or interpenetrate, so the raw mesh
has dozens of components, internal surfaces where shells overlap, and some non-manifold edges.
Internal surfaces bias the chamfer loss of fitter_3d (surface samples inside the body that the
template can never reach), so the shells are fused into one watertight surface by a voxel remesh
before decimation. Keeping only the largest raw component (as the AntScan pipeline does) would
drop most of the ant.

Pipeline (process_model):
    1. Import the model (.glb/.gltf/.fbx/.obj/.stl/.ply) into an empty scene.
    2. Join all mesh objects into one with their world transforms applied. Sketchfab's glTF
       export splits meshes into chunks of at most 65k vertices.
    3. Merge the vertices that the chunk and UV-seam splits duplicated (merge distance far below
       the mesh resolution), delete loose geometry and face islands below min_faces.
    4. Voxel remesh to fuse the shells into one watertight, manifold surface without internal
       geometry (fuse_shells): the voxel size starts small (thin claws and antennomeres survive)
       and grows until the largest component holds >= 99.9 % of the surface area.
    5. Keep only that largest component (find_largest_component from
       prepare_antscan_data_for_mesh_fitting); the removed area fraction is reported.
    6. Decimate to max_vertices (decimate_mesh).
    7. Very light, volume-preserving Laplacian smoothing to even out decimation artefacts.
    8. Clean-up: dissolve degenerate faces, delete loose geometry, make normals point outwards.
    9. Centre the mesh on its bounding-box centre. The source axes are kept: orientation to the
       fitting template (head +X, dorsal +Z, left +Y) is done afterwards by
       orient_meshes_to_template.py, which needs the template model and pytorch3d.
   10. Export a triangulated OBJ (<input name>.obj) and <input name>_stats.json with mesh-health
       statistics (components, boundary and non-manifold edges) of the exported mesh.

Usage:
    blender --background --python prepare_sketchfab_models_for_mesh_fitting.py -- \
        <input_model> <output_dir> [--max_vertices 100000] [--voxel_size_rel 0.0007] \
        [--max_voxel_size_rel 0.003] [--smooth_lambda 0.3] [--smooth_iterations 2] [--debug_dir <dir>]

    --debug_dir writes <name>_preremesh_PROBE.obj (the cleaned mesh before remeshing) for
    comparisons of the remeshed and decimated result against the source surface.

Batch (one Blender process per model):
    python custom_processing/batch_process_models.py --input_dir <downloads> --output_dir <out> \
        --blender_path <blender> --script_path custom_processing/prepare_sketchfab_models_for_mesh_fitting.py \
        --extensions .glb
"""

import json
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_antscan_data_for_mesh_fitting import (  # noqa: E402
    _vertex_components,
    decimate_mesh,
    export_mesh_to_obj,
    filter_small_components,
    find_largest_component,
)

IMPORTERS = {
    ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".fbx": lambda p: bpy.ops.import_scene.fbx(filepath=p),
    ".obj": lambda p: bpy.ops.wm.obj_import(filepath=p),
    ".stl": lambda p: bpy.ops.wm.stl_import(filepath=p),
    ".ply": lambda p: bpy.ops.wm.ply_import(filepath=p),
}


def import_model(path):
    """Imports a model file into an empty scene and returns its mesh objects."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMPORTERS:
        raise ValueError(f"Unsupported model format {ext} ({path})")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    IMPORTERS[ext](path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects in {path}")
    return meshes


def join_meshes(meshes):
    """
    Joins mesh objects into a single object whose vertices are in world coordinates.

    Parents (e.g. the glTF importer's axis-conversion empty) are cleared with their transform
    kept, then all transforms are applied, so the joined mesh has an identity matrix_world.
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    # remove everything else (empties, cameras, lights)
    for other in list(bpy.context.scene.objects):
        if other is not obj:
            bpy.data.objects.remove(other, do_unlink=True)
    return obj


def vertex_array(obj):
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def bbox_diagonal(obj):
    co = vertex_array(obj)
    return float(np.linalg.norm(co.max(0) - co.min(0)))


def merge_split_vertices(obj, relative_distance=1e-6):
    """Merges coincident vertices (chunk/UV-seam splits). Returns the merge distance used."""
    threshold = relative_distance * bbox_diagonal(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return threshold


def _apply_modifier(obj, modifier):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def voxel_remesh(obj, voxel_size):
    """Fuses all shells into one watertight surface (Blender/OpenVDB voxel remesh)."""
    mod = obj.modifiers.new(name="VoxelRemesh", type="REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel_size
    mod.adaptivity = 0.0
    mod.use_smooth_shade = False
    _apply_modifier(obj, mod)
    return len(obj.data.vertices)


def component_area_fractions(obj):
    """Surface-area fraction of every vertex-connected component, largest first."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    components = _vertex_components(bm)
    component_of = {}
    for index, component in enumerate(components):
        for v in component:
            component_of[v.index] = index
    areas = np.zeros(len(components))
    for face in bm.faces:
        areas[component_of[face.verts[0].index]] += face.calc_area()
    bm.free()
    total = areas.sum()
    return np.sort(areas)[::-1] / total if total > 0 else areas


def fuse_shells(obj, diag, voxel_size_rel, max_voxel_size_rel, growth=1.4, min_main_area_fraction=0.999):
    """
    Voxel-remeshes with the smallest voxel size (starting at voxel_size_rel x diag, growing by
    `growth` up to max_voxel_size_rel x diag) for which the largest component holds at least
    min_main_area_fraction of the surface area, i.e. all shells fused and only specks (bits of
    claws or spines thinner than a voxel) are separate. Then keeps only the largest component.

    Small voxels keep thin structures (pretarsal claws, antennomeres) but bridge only small gaps
    between shells, hence the search from small to large.

    Returns:
        dict: voxel size used, attempts, area fraction outside the main component (removed),
        and whether the area criterion was met.
    """
    source = obj.data.copy()
    attempts = []
    rel = voxel_size_rel
    while True:
        n_verts = voxel_remesh(obj, rel * diag)
        fractions = component_area_fractions(obj)
        main = float(fractions[0]) if len(fractions) else 0.0
        attempts.append({"voxel_size_rel": rel, "remeshed_vertices": n_verts,
                         "components": int(len(fractions)), "main_area_fraction": main})
        print(f"voxel {rel:.5f} x diag: {n_verts} verts, {len(fractions)} components, main {main:.5f}")
        if main >= min_main_area_fraction or rel * growth > max_voxel_size_rel * (1 + 1e-9):
            break
        old = obj.data
        obj.data = source.copy()  # retry from the source surface
        bpy.data.meshes.remove(old)
        rel *= growth
    bpy.data.meshes.remove(source)
    find_largest_component(obj)
    return {
        "voxel_size_rel": rel,
        "voxel_size": rel * diag,
        "attempts": attempts,
        "removed_area_fraction": 1.0 - main,
        "fused": main >= min_main_area_fraction,
    }


def light_smooth(obj, lambda_factor=0.1, iterations=1):
    """
    Very light Laplacian smoothing with volume preservation.

    Returns the mean and maximum vertex displacement, so the intensity can be checked against
    the edge length.
    """
    before = vertex_array(obj)
    if iterations > 0 and lambda_factor > 0:
        mod = obj.modifiers.new(name="LightSmooth", type="LAPLACIANSMOOTH")
        mod.iterations = iterations
        mod.lambda_factor = lambda_factor
        mod.lambda_border = 0.0
        mod.use_volume_preserve = True
        mod.use_normalized = True
        _apply_modifier(obj, mod)
    shift = np.linalg.norm(vertex_array(obj) - before, axis=1)
    return float(shift.mean()), float(shift.max())


def clean_mesh(obj, degenerate_distance):
    """Dissolves degenerate faces, deletes loose geometry and makes the normals point outwards."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_degenerate(threshold=degenerate_distance)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def mesh_health(obj):
    """Component count, boundary/non-manifold edge counts and median edge length."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    components = _vertex_components(bm)
    sizes = sorted((len(c) for c in components), reverse=True)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    non_manifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    wire = sum(1 for e in bm.edges if not e.link_faces)
    lengths = np.array([e.calc_length() for e in bm.edges])
    tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    bm.free()
    return {
        "components": len(components),
        "component_vertex_counts": sizes[:10],
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "wire_edges": wire,
        "watertight": boundary == 0 and non_manifold == 0 and wire == 0,
        "median_edge_length": float(np.median(lengths)) if len(lengths) else 0.0,
        "non_triangle_faces": len(obj.data.polygons) - tris,
    }


def centre_on_bbox(obj):
    """Translates the mesh data so its bounding-box centre is at the origin; returns the old centre."""
    co = vertex_array(obj)
    centre = (co.max(0) + co.min(0)) / 2
    obj.data.transform(Matrix.Translation(Vector(tuple(float(-x) for x in centre))))
    obj.data.update()
    return centre


def export_plain_obj(obj, path):
    """Writes the current mesh (any polygons) as OBJ, for debugging comparisons."""
    co = vertex_array(obj)
    with open(path, "w") as f:
        for v in co:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for p in obj.data.polygons:
            f.write("f " + " ".join(str(i + 1) for i in p.vertices) + "\n")


def process_model(
    model_path,
    output_dir,
    max_vertices=100000,
    voxel_size_rel=0.0007,
    max_voxel_size_rel=0.003,
    smooth_lambda=0.3,
    smooth_iterations=2,
    min_faces=4,
    debug_dir=None,
):
    """
    Imports, joins, fuses, decimates, smooths, cleans and exports one model.

    Args:
        model_path (str): Model file to process.
        output_dir (str): Directory for <name>.obj and <name>_stats.json.
        max_vertices (int): Vertex budget of the exported mesh.
        voxel_size_rel (float): Smallest remesh voxel size tried, as a fraction of the
            bounding-box diagonal (see fuse_shells). 0 disables remeshing.
        max_voxel_size_rel (float): Largest remesh voxel size tried.
        smooth_lambda (float): Laplacian smoothing factor after decimation (0 disables it).
        smooth_iterations (int): Laplacian smoothing iterations.
        min_faces (int): Face islands with fewer faces are deleted before remeshing.
        debug_dir (str, optional): Directory for <name>_preremesh_PROBE.obj.

    Returns:
        dict: Processing statistics (also written to <name>_stats.json).
    """
    t0 = time.time()
    name = os.path.splitext(os.path.basename(model_path))[0]
    meshes = import_model(model_path)
    stats = {"source": model_path, "n_source_objects": len(meshes)}
    stats["source_vertices"] = int(sum(len(o.data.vertices) for o in meshes))

    obj = join_meshes(meshes)
    stats["merge_distance"] = merge_split_vertices(obj)
    stats["merged_vertices"] = len(obj.data.vertices)
    stats["small_islands_removed"] = filter_small_components(obj, min_faces=min_faces)
    stats["source_health"] = mesh_health(obj)
    diag = bbox_diagonal(obj)
    stats["bbox_diagonal"] = diag
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        export_plain_obj(obj, os.path.join(debug_dir, f"{name}_preremesh_PROBE.obj"))

    if voxel_size_rel > 0:
        stats["remesh"] = fuse_shells(obj, diag, voxel_size_rel, max_voxel_size_rel)

    stats["decimated_vertices"] = int(decimate_mesh(obj, max_vertices))
    mean_shift, max_shift = light_smooth(obj, smooth_lambda, smooth_iterations)
    clean_mesh(obj, degenerate_distance=1e-6 * diag)

    stats["bbox_centre_removed"] = [float(x) for x in centre_on_bbox(obj)]
    co = vertex_array(obj)
    stats["bbox_extent"] = [float(x) for x in (co.max(0) - co.min(0))]

    os.makedirs(output_dir, exist_ok=True)
    export_path = os.path.join(output_dir, f"{name}.obj")
    export_mesh_to_obj(obj, export_path)  # triangulates
    stats["final_vertices"] = len(obj.data.vertices)
    stats["final_faces"] = len(obj.data.polygons)
    stats["final_health"] = mesh_health(obj)
    edge = stats["final_health"]["median_edge_length"]
    stats["smoothing"] = {
        "lambda": smooth_lambda,
        "iterations": smooth_iterations,
        "mean_shift": mean_shift,
        "max_shift": max_shift,
        "mean_shift_in_median_edges": mean_shift / edge if edge else None,
    }
    stats["obj"] = export_path
    stats["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(output_dir, f"{name}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[stats] " + json.dumps(stats))
    return stats


def _pop_option(argv, flag, cast, default):
    if flag in argv:
        i = argv.index(flag)
        value = cast(argv[i + 1])
        del argv[i : i + 2]
        return value
    return default


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    kwargs = dict(
        max_vertices=_pop_option(argv, "--max_vertices", int, 100000),
        voxel_size_rel=_pop_option(argv, "--voxel_size_rel", float, 0.0007),
        max_voxel_size_rel=_pop_option(argv, "--max_voxel_size_rel", float, 0.003),
        smooth_lambda=_pop_option(argv, "--smooth_lambda", float, 0.3),
        smooth_iterations=_pop_option(argv, "--smooth_iterations", int, 2),
        debug_dir=_pop_option(argv, "--debug_dir", str, None),
    )
    if len(argv) < 2:
        print(
            "Usage: blender --background --python prepare_sketchfab_models_for_mesh_fitting.py -- "
            "<input_model> <output_dir> [--max_vertices N] [--voxel_size_rel F] [--smooth_lambda F] "
            "[--smooth_iterations N] [--debug_dir DIR]"
        )
        sys.exit(1)
    process_model(argv[0], argv[1], **kwargs)


if __name__ == "__main__":
    main()
