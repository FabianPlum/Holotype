"""Small CPU renderer (embree ray casting, orthographic) for overlays: meshes with per-face colours, several views.

img = render(V, F, face_rgb, view_R, width=480, bounds=None)      # view_R rows: image x, image y (up), towards viewer
sheet = grid([[img, ...], [...]]) ; save(path, sheet)
"""

import numpy as np
import trimesh

from . import geom


def views_from_frame(Rc=None):
    """Six standard views for a canonical frame (rows of Rc: +X head, +Y left, +Z dorsal, in scan coordinates).
    Returns dict name -> 3x3 (rows: image right, image up, towards viewer) in scan coordinates."""
    Rc = np.eye(3) if Rc is None else np.asarray(Rc)
    X, Y, Z = Rc
    return {
        "dorsal": np.stack([X, Y, Z]),
        "ventral": np.stack([X, -Y, -Z]),
        "left": np.stack([-X, Z, Y]),
        "right": np.stack([X, Z, -Y]),
        "front": np.stack([Y, Z, X]),
        "oblique": _orth(X + 0.2 * Y, Z + 0.7 * Y + 0.3 * X),
    }


def _orth(right, toward):
    t = toward / np.linalg.norm(toward)
    r = right - (right @ t) * t
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(t, r), t])


def render(V, F, face_rgb, view, width=480, bounds=None, bg=1.0, pad=0.04):
    """Orthographic image (H, W, 3) float in [0, 1]. bounds: (lo, hi) in view coordinates (x, y) to share a window
    between several renders; default: the mesh's own extent."""
    P = V @ view.T
    if bounds is None:
        lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    else:
        lo, hi = bounds
    ext = hi - lo
    lo, hi = lo - pad * ext.max(), hi + pad * ext.max()
    W = int(width)
    H = max(8, int(round(W * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
    xs = lo[0] + (np.arange(W) + 0.5) / W * (hi[0] - lo[0])
    ys = hi[1] - (np.arange(H) + 0.5) / H * (hi[1] - lo[1])
    gx, gy = np.meshgrid(xs, ys)
    z0 = P[:, 2].max() + 1.0
    O = gx.ravel()[:, None] * view[0] + gy.ravel()[:, None] * view[1] + z0 * view[2]
    D = np.repeat(-view[2][None], len(O), 0)
    mesh = trimesh.Trimesh(V, F, process=False)
    tri = geom.ray_intersector(mesh).intersects_first(O, D)
    fn, _ = geom.face_normals(V, F)
    img = np.full((H * W, 3), bg)
    hit = tri >= 0
    sh = 0.35 + 0.65 * np.abs(fn[tri[hit]] @ view[2])
    img[hit] = face_rgb[tri[hit]] * sh[:, None]
    return img.reshape(H, W, 3)


def view_bounds(Vs, view):
    P = np.concatenate([v @ view.T for v in Vs])
    return P[:, :2].min(0), P[:, :2].max(0)


def grid(rows, gap=6, bg=1.0):
    """rows: list of lists of images -> one sheet."""
    hs = [max(im.shape[0] for im in r) for r in rows]
    w = max(sum(im.shape[1] for im in r) + gap * (len(r) - 1) for r in rows)
    out = np.full((sum(hs) + gap * (len(rows) - 1), w, 3), bg)
    y = 0
    for r, h in zip(rows, hs):
        x = 0
        for im in r:
            out[y : y + im.shape[0], x : x + im.shape[1]] = im
            x += im.shape[1] + gap
        y += h + gap
    return out


def save(path, img, max_width=1400):
    from PIL import Image

    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if im.width > max_width:
        im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
    im.save(path)


def project(P, view, bounds, width, pad=0.04):
    """Pixel coordinates (x, y) of points P in an image made by `render` with the same view / bounds / width."""
    lo, hi = bounds
    ext = hi - lo
    lo, hi = lo - pad * ext.max(), hi + pad * ext.max()
    W = int(width)
    H = max(8, int(round(W * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
    Q = np.atleast_2d(P) @ view.T
    return np.stack([(Q[:, 0] - lo[0]) / (hi[0] - lo[0]) * W, (hi[1] - Q[:, 1]) / (hi[1] - lo[1]) * H], 1)


def dots(img, xy, rgb, r=3):
    """Filled discs with a dark rim at pixel positions xy (n,2); rgb one colour or (n,3)."""
    H, W = img.shape[:2]
    rgb = np.broadcast_to(np.asarray(rgb, float), (len(xy), 3))
    yy, xx = np.mgrid[-r - 1 : r + 2, -r - 1 : r + 2]
    d = np.sqrt(xx**2 + yy**2)
    for (x, y), c in zip(xy, rgb):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(x)), int(round(y))
        for m, col in ((d <= r + 1, np.zeros(3)), (d <= r, c)):
            ys, xs = yi + yy[m], xi + xx[m]
            k = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            img[ys[k], xs[k]] = col
    return img


def bone_colours(tpl):
    """One colour per bone: hue by chain (head, mandibles, antennae, mesosoma, waist, gaster, the six legs), brightness
    alternating along a chain, so that neighbouring segments and left / right limbs are told apart."""
    import colorsys

    hue = {
        "b_h": 0.12,
        "b_t": 0.58,
        "b_a_1": 0.20,
        "b_a_2": 0.24,
        "b_a_3": 0.83,
        "b_a_4": 0.83,
        "b_a_5": 0.83,
        "ma_r": 0.72,
        "ma_l": 0.76,
        "an_r": 0.33,
        "an_l": 0.42,
        "l_1_r": 0.00,
        "l_1_l": 0.05,
        "l_2_r": 0.50,
        "l_2_l": 0.55,
        "l_3_r": 0.90,
        "l_3_l": 0.96,
    }
    seg = {"co": 0, "tr": 1, "fe": 0, "ti": 1, "ta": 0, "pt": 0, "1": 0, "2": 1, "3": 1}
    out = np.full((tpl.nJ, 3), 0.6)
    for j, n in enumerate(tpl.names):
        p = n.split("_")
        if p[0] == "l":
            h, k = hue[f"l_{p[1]}_{p[3]}"], seg[p[2]]
        elif p[0] == "an":
            h, k = hue[f"an_{p[2]}"], seg[p[1]]
        elif n in hue:
            h, k = hue[n], 0
        else:
            continue
        out[j] = colorsys.hsv_to_rgb(h, 0.85 if k == 0 else 0.45, 0.80 if k == 0 else 1.0)
    return out
