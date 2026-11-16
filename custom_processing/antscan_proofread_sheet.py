"""Proofreading sheet: four large views of a specimen in its canonical frame (runs inside Blender).

Views (canonical frame from <name>_frame.json: +X head, +Y left, +Z dorsal): left side, right side, dorsal (top) and
ventral (bottom), each at --res pixels, written as <out>_left.png ... and <out>_views.json. The proofreader checks
completeness: six legs with complete tarsi ending in a pretarsus with claws, two complete antennae, mandibles, head,
mesosoma, waist and gaster present; no debris, mounts, fragments or exposed internal structures.

  XDG_RUNTIME_DIR=$(mktemp -d) blender --background --factory-startup --python-exit-code 1 \
      --python antscan_proofread_sheet.py -- --mesh <envelope.stl> --frame <name>_frame.json --out PREFIX [--res 900]
"""

import argparse
import hashlib
import json
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--frame", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=900)
ap.add_argument("--margin", type=float, default=1.04)
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])

fr = json.load(open(a.frame))["frame"]
R = np.asarray(fr["rotation"], dtype=np.float64)
origin = np.asarray(fr["origin_scan"], dtype=np.float64)
VIEWS = {  # name: (direction from the animal to the camera, image-up vector), canonical frame
    "left": (Vector((0, 1, 0)), Vector((0, 0, 1))),
    "right": (Vector((0, -1, 0)), Vector((0, 0, 1))),
    "dorsal": (Vector((0, 0, 1)), Vector((1, 0, 0))),
    "ventral": (Vector((0, 0, -1)), Vector((1, 0, 0))),
}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath=a.mesh)
ob = bpy.context.selected_objects[0]
co = np.empty(len(ob.data.vertices) * 3)
ob.data.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
M = np.eye(4)
M[:3, :3] = R
M[:3, 3] = -R @ origin
ob.matrix_world = Matrix(M.tolist())
vc = (co - origin) @ R.T
lo, hi = vc.min(0), vc.max(0)
centre = Vector(((lo + hi) / 2).tolist())
size = float(np.linalg.norm(hi - lo))

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = scene.render.resolution_y = a.res
sh = scene.display.shading
sh.light, sh.single_color, sh.color_type = "STUDIO", (0.78, 0.74, 0.66), "SINGLE"
sh.show_cavity, sh.cavity_type = True, "BOTH"
scene.display.shading.background_type = "VIEWPORT"
scene.display.shading.background_color = (1, 1, 1)
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

meta = {"mesh": a.mesh, "mesh_md5": hashlib.md5(open(a.mesh, "rb").read()).hexdigest(), "frame": a.frame, "views": {}}
for name, (d, up) in VIEWS.items():
    right = up.cross(d).normalized()
    pr, pu = vc @ np.array(right), vc @ np.array(up)
    c = (
        centre
        + right * float((pr.max() + pr.min()) / 2 - np.array(centre) @ np.array(right))
        + up * float((pu.max() + pu.min()) / 2 - np.array(centre) @ np.array(up))
    )
    cam_data.ortho_scale = max(float(np.ptp(pr)), float(np.ptp(pu))) * a.margin
    cam.location = c + d * size * 2.0
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = Matrix((right, up, d)).transposed().to_quaternion()
    cam_data.clip_start, cam_data.clip_end = 0.001 * size, size * 4.0
    bpy.context.view_layer.update()
    scene.render.filepath = f"{a.out}_{name}.png"
    bpy.ops.render.render(write_still=True)
    meta["views"][name] = {"png": scene.render.filepath, "um_per_px": cam_data.ortho_scale / a.res}
json.dump(meta, open(f"{a.out}_views.json", "w"), indent=1)
print(f"RENDERED proofreading views -> {a.out}_*.png")
