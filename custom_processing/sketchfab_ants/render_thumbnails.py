"""Blender Workbench thumbnails (top view looking down -Z, side view looking along +Y) of OBJ meshes,
imported with raw axes (OBJ x,y,z = Blender x,y,z), framed on each mesh's bounding box.

Usage: blender --background --factory-startup --python render_thumbnails.py -- <obj_dir> <out_dir> [res]
Writes <out_dir>/<name>_top.png and <name>_side.png.
"""
import glob
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
obj_dir, out_dir = argv[0], argv[1]
res = int(argv[2]) if len(argv) > 2 else 480
os.makedirs(out_dir, exist_ok=True)

for path in sorted(glob.glob(os.path.join(obj_dir, "*.obj"))):
    name = os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(os.path.join(out_dir, f"{name}_side.png")):
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    centre = (lo + hi) / 2
    ext = hi - lo
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.78, 0.78, 0.8)
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.clip_start = 0.001
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    d = 10 * max(ext)
    cam_data.clip_end = 100 * max(ext)
    for view in ("top", "side"):
        if view == "top":  # image right = +X, image up = +Y
            cam.location = centre + Vector((0, 0, d))
            cam.rotation_euler = (0, 0, 0)
            cam_data.ortho_scale = 1.05 * max(ext.x, ext.y)
        else:  # camera at -Y looking +Y: image right = +X, image up = +Z
            cam.location = centre + Vector((0, -d, 0))
            cam.rotation_euler = (math.pi / 2, 0, 0)
            cam_data.ortho_scale = 1.05 * max(ext.x, ext.z)
        scene.render.filepath = os.path.join(out_dir, f"{name}_{view}.png")
        bpy.ops.render.render(write_still=True)
    print("rendered", name, flush=True)
