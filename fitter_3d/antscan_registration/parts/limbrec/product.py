"""Reading the inputs: joint product, limb product, the path stages' candidate polylines (optional), a registration.

Facts read from the products: joint product `<dir>/<name>/<name>_joints.json` with joints_scan (scan frame, um),
confidence, provenance, completeness.missing_joints / parts, frame.rotation (rows = canonical axes in scan
coordinates; x_can = R (x_scan - origin)), body_length_um. Limb product `<dir>/<name>/limbs.{json,npz}`: `<L>_X` (n,3)
scan frame TIP FIRST, `<L>_s` arc from the tip, `<L>_r` radius. Path stages (optional, next to the limb product):
`paths_{A,B,C}/<name>/limbs.npz` with the named limbs of that approach and (A, C) every candidate `C<k>_X`.
Registration: `<dir>/<name>.npz` (route format: V_final | V | V_scan | verts, optional part_status / reg_v /
vertex_status) or `<dir>/<name>/<name>_skelreg.npz` (V_scan, vertex_status).
"""

import json
import os

import numpy as np

from .common import CHAINS, CONF_MIN


class Product:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path))
        d = self.d
        self.name = d.get("name")
        self.BL = float(d.get("body_length_um") or 0.0)
        self.js = d.get("joints_scan") or {}
        self.cf = d.get("confidence") or {}
        self.prov = d.get("provenance") or {}
        comp = d.get("completeness") or {}
        self.missing = set(comp.get("missing_joints") or [])
        self.parts = comp.get("parts") or {}
        fr = d.get("frame") or {}
        self.R = np.asarray(fr["rotation"], float) if fr.get("rotation") is not None else np.eye(3)
        self.o = np.asarray(fr["origin_scan"], float) if fr.get("origin_scan") is not None else np.zeros(3)

    def pos(self, n):
        v = self.js.get(n)
        if v is None or not np.all(np.isfinite(v)):
            return None
        return np.asarray(v, float)

    def conf(self, n):
        c = self.cf.get(n)
        return 0.0 if c is None else float(c)

    def usable(self, n):
        return self.pos(n) is not None and self.conf(n) >= CONF_MIN and n not in self.missing

    def to_can(self, x):
        return (np.asarray(x, float) - self.o) @ self.R.T

    def to_scan(self, c):
        return np.asarray(c, float) @ self.R + self.o

    def mirror(self, x):
        """mirror image across the sagittal plane of the product frame (canonical y -> -y)."""
        c = self.to_can(x)
        c[..., 1] = -c[..., 1]
        return self.to_scan(c)

    def is_fallback(self, n):
        return "template_fallback" in str(self.prov.get(n) or "")


def load_limbs(limbs_dir, name):
    """named centrelines of the combined limb product: dict limb -> dict(X root-first (n,3), r (n,), status, conf,
    flags). Empty dict when the product is missing."""
    out = {}
    pj, pz = f"{limbs_dir}/{name}/limbs.json", f"{limbs_dir}/{name}/limbs.npz"
    if not (os.path.isfile(pj) and os.path.isfile(pz)):
        return out
    meta = json.load(open(pj)).get("limbs", {})
    z = np.load(pz)
    for lb in CHAINS:
        if f"{lb}_X" not in z.files:
            continue
        X = np.asarray(z[f"{lb}_X"], float)
        if len(X) < 4:
            continue
        r = np.asarray(z[f"{lb}_r"], float) if f"{lb}_r" in z.files else None
        m = meta.get(lb, {})
        out[lb] = dict(
            X=X[::-1].copy(),
            r=None if r is None else r[::-1].copy(),
            status=m.get("status"),
            conf=m.get("confidence"),
            flags=m.get("flags") or [],
            core_arc=m.get("core_arc_um"),
        )
    return out


def load_pool(stage_root, name, limbs=None):
    """every polyline the path stages wrote for this specimen (named and unnamed candidates of A, B, C), tip first as
    stored: list of dict(src, key, X). `stage_root` is the folder that holds paths_A / paths_B / paths_C (the work
    directory of the joint method), or a list of such folders (one per state of the joint method's ensemble: they
    are read in the given order and `src` gets the folder's number); missing stages are skipped silently (the pool is
    optional evidence)."""
    if isinstance(stage_root, (list, tuple)):
        pool = []
        for k, r in enumerate(stage_root):
            for q in load_pool(r, name, limbs):
                pool.append(dict(q, src=f"{q['src']}{k}"))
        return pool
    pool = []
    if stage_root:
        for st in ("paths_A", "paths_B", "paths_C"):
            p = f"{stage_root}/{st}/{name}/limbs.npz"
            if not os.path.isfile(p):
                continue
            try:
                z = np.load(p)
            except Exception:
                continue
            for k in z.files:
                if k.endswith("_X") and not k.startswith("T_"):
                    X = np.asarray(z[k], float)
                    if len(X) >= 4 and np.isfinite(X).all():
                        pool.append(dict(src=st[-1], key=k[:-2], X=X))
    return pool


def load_registration(reg_dir, name):
    """-> dict(V (nV,3) scan frame um, part_status (55,) str or None, reg_v bool or None, path) or None."""
    if not reg_dir:
        return None
    cands = [
        f"{reg_dir}/{name}.npz",
        f"{reg_dir}/{name}/{name}.npz",
        f"{reg_dir}/{name}/{name}_skelreg.npz",
        f"{reg_dir}/{name}_skelreg.npz",
        f"{reg_dir}/{name}/{name}_cover.npz",
    ]
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue
        V = None
        for k in ("V_final", "V_out", "V_scan", "V", "verts"):
            if k in z.files:
                V = np.asarray(z[k], float)
                break
        if V is None or V.ndim != 2 or V.shape[1] != 3 or not np.isfinite(V).all():
            continue
        pst = np.asarray(z["part_status"]).astype(str) if "part_status" in z.files else None
        reg_v = None
        if "reg_v" in z.files:
            reg_v = np.asarray(z["reg_v"]).astype(bool)
        elif "vertex_status" in z.files:
            reg_v = np.asarray(z["vertex_status"]) != 2
        return dict(V=V, part_status=pst, reg_v=reg_v, path=p)
    return None
