"""Assembles the publishable report folder from the renders and data.

- converts DATA/report_build/render/png/*.png to WebP in <out>/img (fit tiles: fit | target vs pose+shape |
  target vs final, side by side),
- writes the 3D-viewer binaries as base64 text (<out>/data/*.txt; artifact hosting serves no raw binaries),
- inlines DATA/report_site/data/page.json into index.src.html -> <out>/index.html.

Usage: python assemble_site.py DATA OUT
Then publish <out>/index.html with every file under <out>/img and <out>/data (two publishes of at most 255 files).
"""
import base64
import glob
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = (os.path.abspath(p) for p in sys.argv[1:3])
PNG = os.path.join(DATA, "report_build", "render", "png")
SITE = os.path.join(DATA, "report_site")
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)

n = 0
for p in sorted(glob.glob(os.path.join(PNG, "*.png"))):
    b = os.path.basename(p)[:-4]
    if b.startswith("tile_"):
        continue
    Image.open(p).save(os.path.join(OUT, "img", f"{b}.webp"), quality=82 if b.startswith("thumb_") else 86, method=6)
    n += 1
for fit in sorted(glob.glob(os.path.join(PNG, "tile_*_fit.png"))):
    i = os.path.basename(fit)[5:8]
    ims = [Image.open(os.path.join(PNG, f"tile_{i}_{s}.png")).convert("RGBA") for s in ("fit", "Stage_1", "Stage_3")]
    out = Image.new("RGBA", (900, 300), (0, 0, 0, 0))
    for k, im in enumerate(ims):
        out.alpha_composite(im, (k * 300, 0))
    out.resize((720, 240), Image.LANCZOS).save(os.path.join(OUT, "img", f"tile_{i}.webp"), quality=80, method=6)
    n += 1
print(n, "images")

for src, dst in (("faces.u16", "faces"), ("mean.f32", "mean"), ("pcs.i16", "pcs"), ("atlas_s1.u8", "atlas_s1"),
                 ("shapes.i16", "shapes")):
    raw = open(os.path.join(SITE, "data", src), "rb").read()
    open(os.path.join(OUT, "data", f"{dst}.txt"), "w").write(base64.b64encode(raw).decode())

page = open(os.path.join(SITE, "data", "page.json")).read()
assert "</script" not in page
html = open(os.path.join(HERE, "index.src.html")).read().replace("/*__DATA__*/", page)
open(os.path.join(OUT, "index.html"), "w").write(html)
print("wrote", os.path.join(OUT, "index.html"))
