"""AntScan joint method (round 7, four-state ensemble): from a prepared scan envelope to the joint product, the limb
product and the per-state products that fitter_3d/antscan_registration reads.

    python -m fitter_3d.antscan_joints.run --names NAMES.txt --env ENV_DIR --out OUT_DIR [--jobs N]

Layout: run.py (the command: scheduling, resume, per-job memory cap), stage.py (one stage per process), steps.py
(thin / folds / ensemble per process), method/ (the method modules as they ran in production; they use flat imports
from their own folder and are only ever run inside a stage or step process). README.md lists the data files and how
they were made. Importing this package imports nothing but the standard library.
"""

import importlib
import os

VERSION = "antscan-joints-1.0 (round 7 K4)"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "method", "along", "along_model.pkl")

STATES = ("s0", "s1", "s2", "s3")  # the four sign states of TRUNK's working frame (stage.STATES)
STAGES = ("trunk", "paths_A", "paths_B", "paths_C")  # one process each, per state
POST = ("thin", "assemble", "folds")  # one process each, per state, after the stages
STEPS = STAGES + POST
DEPS = {
    "trunk": (),
    "paths_A": ("trunk",),
    "paths_B": ("trunk",),
    "paths_C": ("trunk",),
    "thin": ("paths_A", "paths_B", "paths_C"),
    "assemble": ("thin",),
    "folds": ("assemble",),
}

# ------------------------------------------------------------------------------------------------ product contract
LEGS = tuple(f"{s}{i}" for s in "LR" for i in (1, 2, 3))
LIMBS = LEGS + ("AL", "AR", "ML", "MR")
TRUNK_JOINTS = ("b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5", "b_h")
LIMB_JOINTS = tuple(
    [f"l_{k}_{j}_{s}" for s in "rl" for k in "123" for j in ("co", "tr", "fe", "ti", "ta", "pt")]
    + ["ma_r", "ma_l", "an_1_r", "an_2_r", "an_3_r", "an_1_l", "an_2_l", "an_3_l"]
)
WINGS = ("w_1_r", "w_2_r", "w_1_l", "w_2_l")
EXTRA = ("head_front", "gaster_tip")
JOINTS = TRUNK_JOINTS + LIMB_JOINTS + WINGS + EXTRA  # the 55 template joints + head_front, gaster_tip
PRODUCT_KEYS = ("name", "status", "joints_scan", "confidence", "frame", "body_length_um", "completeness")


def check_product(d, ensemble=True):
    """List of problems of a joint product dict (empty = it fulfils the contract the registration reads):
    status ok, every joint key present (position null or 3 finite numbers; confidence in [0, 1]), frame rotation a
    proper rotation (rows = canonical axes in scan coordinates) + origin_scan, positive body length, and for an
    ensemble product the `ensemble` block (states, kept) and per-joint `joint_detail.<j>.ens`."""
    import numpy as np

    if not isinstance(d, dict):
        return ["not a json object"]
    out = [f"missing key {k}" for k in PRODUCT_KEYS if k not in d]
    if out:
        return out
    if d.get("status") != "ok":
        out.append(f"status {d.get('status')!r}: {d.get('error')}")
    js, cf = d.get("joints_scan") or {}, d.get("confidence") or {}
    for j in JOINTS:
        if j not in js:
            out.append(f"joint {j} missing")
            continue
        x = js[j]
        if x is not None and (len(x) != 3 or not np.all(np.isfinite(np.asarray(x, float)))):
            out.append(f"joint {j}: position {x!r}")
        c = cf.get(j)
        if c is None or not (0.0 <= float(c) <= 1.0):
            out.append(f"joint {j}: confidence {c!r}")
    fr = d.get("frame") or {}
    try:
        R = np.asarray(fr["rotation"], float)
        o = np.asarray(fr["origin_scan"], float)
        if R.shape != (3, 3) or o.shape != (3,):
            raise ValueError("shape")
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1.0) < 1e-6):
            out.append("frame.rotation is not a proper rotation")
    except (KeyError, TypeError, ValueError):
        out.append("frame without rotation (3x3) / origin_scan (3)")
    try:
        if not float(d["body_length_um"]) > 0:
            out.append("body length <= 0")
    except (TypeError, ValueError):
        out.append("body length unreadable")
    if ensemble:
        ens = d.get("ensemble") or {}
        if not ens.get("states") or not ens.get("kept"):
            out.append("ensemble block without states / kept")
        det = d.get("joint_detail") or {}
        n_ens = sum(1 for j in JOINTS if (det.get(j) or {}).get("ens"))
        if js.get("b_t") is not None and n_ens == 0:
            out.append("no joint_detail.<joint>.ens")
    return out


def check_limbs(lj, arrays):
    """List of problems of a limb product (limbs.json dict, dict of limbs.npz arrays): every limb has a status; a limb
    with a path has <limb>_X (n, 3), <limb>_s (n,) and <limb>_r (n,) arrays (points tip first)."""
    out = []
    limbs = (lj or {}).get("limbs") or {}
    for ch in LIMBS:
        L = limbs.get(ch)
        if L is None or "status" not in L:
            out.append(f"limb {ch} missing")
            continue
        if L["status"] == "no_path":
            continue
        X, s, r = arrays.get(f"{ch}_X"), arrays.get(f"{ch}_s"), arrays.get(f"{ch}_r")
        if X is None or s is None or r is None:
            out.append(f"limb {ch} ({L['status']}) without arrays")
        elif X.ndim != 2 or X.shape[1] != 3 or s.shape != (len(X),) or r.shape != (len(X),):
            out.append(f"limb {ch}: array shapes {X.shape} {s.shape} {r.shape}")
    return out


# ------------------------------------------------------------------------------------------------ dependencies
REQUIRED = ("numpy", "scipy", "numba", "networkx", "skimage", "trimesh", "sklearn")


def missing_dependencies():
    """Names of the packages the method needs that cannot be imported."""
    miss = []
    for m in REQUIRED:
        try:
            importlib.import_module(m)
        except ImportError:
            miss.append({"skimage": "scikit-image", "sklearn": "scikit-learn"}.get(m, m))
    return miss


def has_embree():
    """True when trimesh casts rays with embree (embreex). Without it trimesh falls back to its own ray tracer, whose
    results differ: the centreline profiles of the joint placement (method/along/geom_profile.py) then differ from the
    production run."""
    try:
        import trimesh.ray

        return bool(getattr(trimesh.ray, "has_embree", False))
    except ImportError:
        return False
