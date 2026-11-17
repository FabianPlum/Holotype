"""Blender: import a Sketchfab glb, join + merge split vertices + drop specks (same steps as the prep script), export OBJ.
Usage: blender --background --python export_source_shells.py -- <repo> <glb> <out.obj>"""
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, os.path.join(argv[0], "custom_processing"))
from prepare_sketchfab_models_for_mesh_fitting import (  # noqa: E402
    export_plain_obj, filter_small_components, import_model, join_meshes, merge_split_vertices, mesh_health)

obj = join_meshes(import_model(argv[1]))
merge_split_vertices(obj)
filter_small_components(obj, min_faces=4)
print("source health", mesh_health(obj))
export_plain_obj(obj, argv[2])
