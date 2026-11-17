"""Tiles <name>_top.png / <name>_side.png thumbnails into labelled contact sheets.

Usage: python contact_sheets.py <thumb_dir> <out_prefix> [per_page] [scores_csv]
Each tile: top view (head should point right, +X) above side view (dorsal up, +Z).
If scores_csv (orientation_scores.csv) is given, ambiguous orientations are labelled in red.
"""
import csv
import glob
import os
import sys

from PIL import Image, ImageDraw

thumb_dir, out_prefix = sys.argv[1], sys.argv[2]
per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
flags = {}
if len(sys.argv) > 4:
    for row in csv.DictReader(open(sys.argv[4])):
        flags[row["name"]] = (row["ambiguous"] == "True", float(row["margin"]))

names = sorted(os.path.basename(p)[: -len("_top.png")] for p in glob.glob(os.path.join(thumb_dir, "*_top.png")))
cols = 6
for page in range(0, len(names), per_page):
    chunk = names[page : page + per_page]
    tiles = []
    for n in chunk:
        top = Image.open(os.path.join(thumb_dir, f"{n}_top.png")).convert("RGB")
        side = Image.open(os.path.join(thumb_dir, f"{n}_side.png")).convert("RGB").resize((top.width, top.height // 2))
        tile = Image.new("RGB", (top.width, top.height + side.height + 22), "white")
        tile.paste(top, (0, 22))
        tile.paste(side, (0, 22 + top.height))
        d = ImageDraw.Draw(tile)
        amb, margin = flags.get(n, (False, None))
        label = n if margin is None else f"{n}  m={margin:.0%}"
        d.text((4, 5), label + ("  AMBIGUOUS" if amb else ""), fill="red" if amb else "black")
        tiles.append(tile)
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h + 30), "white")
    ImageDraw.Draw(sheet).text((8, 8), "top: looking down -Z (head should point right = +X) | side: looking along +Y (dorsal up = +Z)", fill="black")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, 30 + (i // cols) * h))
    out = f"{out_prefix}_{page // per_page:02d}.png"
    sheet.save(out)
    print("wrote", out)
