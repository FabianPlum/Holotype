"""Per-specimen overlay coloured by status (CPU, skelreg.render with embree; layout adapted from COVER's
overlay sheet).

Sheet: title (name, specimen class, head end, counts); row 1 the registered template coloured by status (limb parts
saturated, trunk / head / mandibles lighter), dorsal / left / ventral views of the joint product's frame; row 2 the
scan envelope (grey) with the named limb centrelines (black dots) and the virtual markers of every limb joint,
coloured by the status of the segment ending there; legend.
"""

import os

import numpy as np

from . import VERSION  # noqa: F401

COL = np.array(
    [
        [0.20, 0.40, 0.85],
        [0.10, 0.75, 0.80],
        [0.15, 0.70, 0.20],
        [1.00, 0.60, 0.05],
        [0.85, 0.10, 0.10],
        [0.55, 0.30, 0.80],
        [0.95, 0.55, 0.75],
        [0.65, 0.85, 0.25],
    ]
)
LABELS = [
    "0 joint-supported",
    "1 surface-verified",
    "2 recovered",
    "3 review",
    "4 unregistered",
    "5 imputed (mirror)",
    "6 imputed (prior)",
    "7 recovered (estimated)",
]


def _text(text, width, h=20):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (width, h), (255, 255, 255))
    ImageDraw.Draw(im).text((6, 4), text, fill=(0, 0, 0))
    return np.asarray(im, float) / 255.0


def _legend(width):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (width, 26), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    x = 6
    for c, lab in zip(COL[:5].tolist() + [COL[7].tolist()], LABELS[:5] + [LABELS[7]]):
        dr.rectangle([x, 7, x + 12, 19], fill=tuple(int(255 * v) for v in c), outline=(0, 0, 0))
        dr.text((x + 16, 7), lab, fill=(0, 0, 0))
        x += 16 + 6 * len(lab) + 14
    return np.asarray(im, float) / 255.0


def face_colours(t, code):
    limb = np.array([n.startswith(("l_", "an_")) for n in t.names])
    vc = COL[np.clip(np.asarray(code)[t.dom], 0, 7)].copy()
    light = ~limb[t.dom]
    vc[light] = 0.55 * vc[light] + 0.45
    return vc[t.F].mean(1)


def render_one(name, V, part_code, seg, frame_R, envelope, limbs_dir, spec, out_png, BL, width=420):
    from fitter_3d.antscan_registration.parts import cover04 as CV
    from fitter_3d.antscan_registration.parts.skelreg import geom, render

    t = CV.tpl()
    Vs, Fs = geom.load_mesh(envelope)
    limbs = CV.load_limbs(limbs_dir) if limbs_dir else {}
    views = render.views_from_frame(np.asarray(frame_R, float))
    M = CV.markers(V)
    code = np.asarray(part_code, int)
    limbj, colj = [], []
    for ch, segs in CV.chains().items():
        for sg in segs:
            limbj.append(sg["child"])
            colj.append(seg[sg["name"]][0])
    cl = np.concatenate([lim["X"][::4] for lim in limbs.values()]) if limbs else np.zeros((0, 3))
    rows = [[], []]
    for vn in ("dorsal", "left", "ventral"):
        vw = views[vn]
        b = render.view_bounds([Vs, V], vw)
        rows[0].append(render.render(V, t.F, face_colours(t, code), vw, width, bounds=b))
        im = render.render(Vs, Fs, np.full((len(Fs), 3), 0.82), vw, width, bounds=b)
        if len(cl):
            im = render.dots(im, render.project(cl, vw, b, width), np.zeros(3), r=0)
        im = render.dots(im, render.project(M[limbj], vw, b, width), COL[np.clip(colj, 0, 7)], r=3)
        rows[1].append(im)
    W = sum(r.shape[1] for r in rows[0]) + 12
    he = spec["head_end"]["state"]
    head = (
        f"{name}   class: {spec['specimen_class'].upper()}   head end: {he}   limb segments measurable "
        f"{spec['n_measurable']}/{spec['n_limb_segments']} (review {spec['n_review']}, unregistered "
        f"{spec['n_unregistered']})"
    )
    sub = (
        "row 1: registered template by status (trunk/head/mandibles lighter); row 2: scan, named centrelines (black), "
        "limb joint markers coloured by the status of the segment ending there"
    )
    sheet = render.grid([[_text(head, W)], [_text(sub, W)], rows[0], rows[1], [_legend(W)]])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    render.save(out_png, sheet, max_width=1500)
    return out_png
