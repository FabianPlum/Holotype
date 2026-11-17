"""Blender (4.5) Workbench renderer for report figures, driven by a JSON job list.

Each job: {"verts": npy (V,3) float, "faces": npy (F,3) int, "colors": npy (V,3) uint8 sRGB or null,
           "color": [r,g,b] 0-255 if no per-vertex colors, "view": "top"|"side"|"persp"|"front",
           "ortho_scale": float or null (auto), "centre": [x,y,z] or null, "res": [w,h], "out": png,
           "random_parts": false (colour every loose part randomly)}
Usage: blender --background --factory-startup --python render_jobs.py -- jobs.json
"""
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

jobs = json.load(open(sys.argv[sys.argv.index("--") + 1]))
cache = {}


def arr(path):
    if path not in cache:
        cache[path] = np.load(path)
    return cache[path]


def setup_scene(res, transparent=True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.show_cavity = True
    sc.display.shading.cavity_type = "WORLD"
    sc.display.shading.show_specular_highlight = True
    sc.view_settings.view_transform = "Standard"
    sc.render.film_transparent = transparent
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.use_freestyle = False
    return sc


def build_mesh(verts, faces, colors):
    me = bpy.data.meshes.new("m")
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.ascontiguousarray(verts, dtype=np.float32).ravel())
    me.loops.add(len(faces) * 3)
    me.loops.foreach_set("vertex_index", np.ascontiguousarray(faces, dtype=np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 3, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(faces), 3, dtype=np.int32))
    me.update()
    me.validate()
    attr = me.color_attributes.new("col", "BYTE_COLOR", "POINT")
    rgba = np.ones((len(verts), 4), dtype=np.float32)
    rgba[:, :3] = colors / 255.0
    attr.data.foreach_set("color_srgb", rgba.ravel())
    me.color_attributes.active_color = attr
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new("m", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def place_camera(sc, view, centre, ortho_scale, ext):
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    d = 20.0
    cam_data.clip_start, cam_data.clip_end = 0.01, 200
    c = Vector(centre)
    if view == "persp":
        cam_data.type = "PERSP"
        cam_data.lens = 70
        direction = Vector((0.55, -0.85, 0.55)).normalized()
        cam.location = c + direction * (ortho_scale * 2.15)
        cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        return
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    if view == "top":
        cam.location, cam.rotation_euler = c + Vector((0, 0, d)), (0, 0, 0)
    elif view == "side":  # camera at -Y looking +Y: right = +X, up = +Z
        cam.location, cam.rotation_euler = c + Vector((0, -d, 0)), (math.pi / 2, 0, 0)
    elif view == "front":  # camera at +X looking -X: right = +Y... up = +Z
        cam.location, cam.rotation_euler = c + Vector((d, 0, 0)), (math.pi / 2, 0, math.pi / 2)


for job in jobs:
    sc = setup_scene(job.get("res", [600, 600]), job.get("transparent", True))
    v = arr(job["verts"])
    if v.ndim == 3:
        v = v[job["index"]]
    f = arr(job["faces"])
    if job.get("colors"):
        col = arr(job["colors"])
        if col.ndim == 3:
            col = col[job["index"]]
    else:
        col = np.tile(np.array(job.get("color", [190, 190, 196]), dtype=np.float32), (len(v), 1))
    ob = build_mesh(v, f, col.astype(np.float32))
    if job.get("random_parts"):
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
        sc.display.shading.color_type = "RANDOM"
    else:
        sc.display.shading.color_type = "VERTEX"
    lo, hi = v.min(0), v.max(0)
    centre = job.get("centre") or ((lo + hi) / 2).tolist()
    ext = hi - lo
    if job.get("ortho_scale"):
        scale = job["ortho_scale"]
    elif job["view"] == "top":
        scale = 1.06 * max(ext[0], ext[1])
    elif job["view"] == "side":
        scale = 1.06 * max(ext[0], ext[2])
    else:
        scale = 1.06 * float(max(ext))
    place_camera(sc, job["view"], centre, scale, ext)
    sc.render.filepath = job["out"]
    bpy.ops.render.render(write_still=True)
    print("rendered", job["out"], flush=True)
