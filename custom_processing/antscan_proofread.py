"""Proofreading of AntScan workers (Stage 2p): is the specimen complete enough to fit?

Criteria (from an earlier proofreading of the 2022-generation meshes and a calibration on the 2024-generation meshes):
keep only complete workers: six legs with complete tarsi
ending in a pretarsus with claws, complete antennae, head, mesosoma, waist and gaster present; no fragmentation,
debris, mounts or exposed internal structures. Missing tarsi, pretarsi or claws lead to poorer fits.

Decisions live in <out>/<name>/<name>_proofread.json:
  {"name", "decision": "keep" | "reject", "reasons": [...], "source": "human" | "agent", "by", "time", "mesh_md5",
   "notes", "history"}
Sources, in order of authority: a human decision on the current mesh > an automatic decision (source "agent": a
decision written by an automatic reader of the proofreading sheet). Records of the production run may also carry
"legacy_proofread" (a keep taken over from the earlier proofreading of the 2022-generation meshes; on 48 calibration
specimens those keeps stayed keeps in 23 of 24, while earlier rejects became keeps in 13 of 24, so rejects were not
taken over).
Reasons used on the 2024 meshes: tarsi/claws missing, debris or mount, the ant carries pupae (brood in the
mandibles), exposed internal structure, fragmented or very noisy scan, a heavily deformed body (unsure).

  python -m custom_processing.antscan_proofread sheet  <envelope.stl> <name>_frame.json OUT_PREFIX   # -> OUT_PREFIX_proofread.png
  python -m custom_processing.antscan_proofread record <out>/<name> keep|reject [--reason R ...] [--by WHO] [--source human|agent]
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time

from custom_processing.antscan_pipeline import DEFAULT_BLENDER

HERE = os.path.dirname(os.path.abspath(__file__))
REASONS = [
    "tarsus_or_claw_missing",
    "leg_missing_or_broken",
    "antenna_incomplete",
    "body_part_missing",
    "fragmented_scan",
    "debris_or_mount",
    "carries_brood",
    "internal_structure_exposed",
    "noisy_scan",
    "deformed_body",
    "not_a_worker",
    "other",
]


def render_sheet(stl, frame_json, out_prefix, blender=DEFAULT_BLENDER, res=900, timeout=900):
    env = dict(os.environ, XDG_RUNTIME_DIR=tempfile.mkdtemp(prefix="blender_xdg_"))
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        os.path.join(HERE, "antscan_proofread_sheet.py"),
        "--",
        "--mesh",
        stl,
        "--frame",
        frame_json,
        "--out",
        out_prefix,
        "--res",
        str(res),
    ]
    log = out_prefix + "_proofread_render.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
    if rc != 0:
        raise RuntimeError(f"blender failed (rc {rc}), see {log}")
    return compose(out_prefix)


def compose(out_prefix):
    from PIL import Image, ImageDraw

    meta = json.load(open(out_prefix + "_views.json"))
    order = [["left", "right"], ["dorsal", "ventral"]]
    tiles = {k: Image.open(v["png"]).convert("RGB") for k, v in meta["views"].items()}
    w, h = next(iter(tiles.values())).size
    pad, foot = 6, 24
    sheet = Image.new("RGB", (2 * w + 3 * pad, 2 * h + 3 * pad + foot), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(order):
        for c, k in enumerate(row):
            x, y = pad + c * (w + pad), pad + r * (h + pad)
            sheet.paste(tiles[k], (x, y))
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(0, 0, 0))
            draw.text((x + 6, y + 4), f"{k}  ({meta['views'][k]['um_per_px']:.1f} um/px)", fill=(0, 0, 0))
    name = os.path.basename(meta["frame"]).replace("_frame.json", "")
    draw.text(
        (pad, 2 * h + 2 * pad + 5),
        f"{name}: keep only complete workers (6 legs with tarsi + claws, antennae, all body parts, no debris)",
        fill=(0, 0, 0),
    )
    path = out_prefix + "_proofread.png"
    sheet.save(path)
    return path


def record(spec_dir, decision, reasons=(), by="reviewer", source="human", notes="", mesh=None):
    name = os.path.basename(os.path.normpath(spec_dir))
    mesh = mesh or os.path.join(spec_dir, f"{name}_envelope.stl")
    rec = {
        "name": name,
        "decision": decision,
        "reasons": list(reasons),
        "source": source,
        "by": by,
        "notes": notes,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mesh_md5": hashlib.md5(open(mesh, "rb").read()).hexdigest() if os.path.isfile(mesh) else None,
    }
    path = os.path.join(spec_dir, f"{name}_proofread.json")
    prev = json.load(open(path)) if os.path.isfile(path) else None
    if prev:
        rec["history"] = (prev.get("history") or []) + [
            {k: prev.get(k) for k in ("decision", "reasons", "source", "by", "time", "notes")}
        ]
    json.dump(rec, open(path, "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("stl")
    s.add_argument("frame_json")
    s.add_argument("out_prefix")
    s.add_argument("--res", type=int, default=900)
    r = sub.add_parser("record")
    r.add_argument("spec_dir")
    r.add_argument("decision", choices=["keep", "reject"])
    r.add_argument("--reason", action="append", default=[], choices=REASONS)
    r.add_argument("--by", default="reviewer")
    r.add_argument("--source", default="human", choices=["human", "agent"])
    r.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.cmd == "sheet":
        print(render_sheet(a.stl, a.frame_json, a.out_prefix, res=a.res))
    else:
        print(json.dumps(record(a.spec_dir, a.decision, a.reason, a.by, a.source, a.notes)))


if __name__ == "__main__":
    main()
