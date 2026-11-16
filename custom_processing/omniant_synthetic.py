"""Synthetic OmniAnt workers with known joints, for validating AntScan joint proposals.

Each sample is an OmniAnt_25PCs instance with
  - betas drawn from N(0, diag(shape_cov)) (times --shape-sd),
  - a pose drawn inside the joint limits (legs, mandibles: uniform within --limit-frac of the limit range;
    joints without limits (+-180 deg: antennae, waist/gaster, head, pretarsi): normal with a per-group
    standard deviation),
  - a random global rotation (uniform on SO(3)), a physical size (body length head front to gaster tip,
    uniform in --length-mm) and a random offset, so coordinates look like a scan's (micrometres),
  - optionally a removed part (the vertices dominated by the listed bones collapse onto the pivot where the
    part broke off, which keeps the surface closed), to test completeness checks.
The model's left-mandible skinning weights sum to less than 1 on ~140 vertices; the rows are renormalised.

Writes per sample <out>/<name>/<name>.stl (binary, micrometres) and <name>.json with the ground truth:
joints (regressed, the model's output joints) and pivots (kinematic joints, J_transformed) in micrometres,
the rotation/scale/offset that map model units to the scan frame, betas, pose, removed bones, and a
voxel_size field ("<x> µm", body length / 2000 by default; real 2024 scans have ~1200-2100 voxels per body
length) so the AntScan stages treat it like a scan. The posed model is subdivided (midpoints, surface unchanged)
until every edge is <= body length / 400. (Voxelizing it instead loses the template's thin tibiae, which are only
~0.4 % of body length thick.)

Run (pytorch3d env, from the repo root):
  python -m custom_processing.omniant_synthetic --out DIR --n 20 [--seed 0] [--damage none|random]
"""

import argparse
import json
import math
import os
import pickle

import numpy as np
import torch

import config
from smal_model.smal_torch import SMAL

MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl"
)

# joints whose limits are +-pi (free) get a normal draw with these standard deviations (radians)
FREE_SD = {"b_a": 0.20, "b_h": 0.20, "an_1": 0.45, "an_2": 0.45, "an_3": 0.15, "pt": 0.30, "w_": 0.0}

# removable parts: name -> bones whose dominated faces are removed; "missing" = joints that no longer exist
DAMAGE = {
    "tarsus_l3r": {"bones": ["l_3_ta_r", "l_3_pt_r"], "missing": ["l_3_pt_r"]},
    "tarsus_l1l": {"bones": ["l_1_ta_l", "l_1_pt_l"], "missing": ["l_1_pt_l"]},
    "leg_l2l_from_knee": {
        "bones": ["l_2_fe_l", "l_2_ti_l", "l_2_ta_l", "l_2_pt_l"],
        "missing": ["l_2_ti_l", "l_2_ta_l", "l_2_pt_l"],
    },
    "leg_l1r_from_trochanter": {
        "bones": ["l_1_tr_r", "l_1_fe_r", "l_1_ti_r", "l_1_ta_r", "l_1_pt_r"],
        "missing": ["l_1_fe_r", "l_1_ti_r", "l_1_ta_r", "l_1_pt_r"],
    },
    "antenna_funiculus_r": {"bones": ["an_2_r", "an_3_r"], "missing": ["an_3_r"]},
}


def load_model(device="cpu"):
    config.set_model(MODEL)
    config.ignore_sym = True
    smal = SMAL(device)
    w = smal.weights
    smal.weights = w / w.sum(dim=1, keepdim=True)  # renormalise (left-mandible defect)
    dd = pickle.load(open(MODEL, "rb"), encoding="latin1")
    return smal, dd


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def sample_pose(names, limits, rng, limit_frac):
    theta = np.zeros((len(names), 3))
    for j, n in enumerate(names):
        if j == 0:
            continue  # root: the global rotation is applied outside the model
        lo, hi = limits[j, :, 0], limits[j, :, 1]
        free = (hi - lo) > 2 * math.pi - 1e-3
        for ax in range(3):
            if free[ax]:
                sd = next((v for k, v in FREE_SD.items() if n.startswith(k) or f"_{k}_" in n), 0.2)
                theta[j, ax] = rng.normal(0, sd)
            else:
                mid = 0.5 * (lo[ax] + hi[ax])
                half = 0.5 * (hi[ax] - lo[ax]) * limit_frac
                theta[j, ax] = np.clip(rng.uniform(mid - half, mid + half), lo[ax], hi[ax])
    return theta


def remove_part(v, weights, names, bones, pivots_scan):
    """Simulate a part broken off at the pivot of bones[0]: every vertex dominated by the listed bones is
    collapsed onto that pivot. The surface stays closed (a pinched stump), so the envelope stage can turn it
    into a clean stump instead of having to close an open cut."""
    dom = weights.argmax(1)
    ids = [names.index(b) for b in bones]
    v = v.copy()
    v[np.isin(dom, ids)] = pivots_scan[names.index(bones[0])]
    return v


def write_stl(path, v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, np.stack([a, b, c], axis=1)
    with open(path, "wb") as fh:
        fh.write(b"omniant_synthetic".ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())


def scan_like_subdivision(v, f, max_edge):
    """Split edges until none is longer than max_edge (midpoints, no smoothing; the surface is unchanged).

    The 2024 AntScan meshes have uniform edges of ~1.17 scan voxels; this gives the synthetic mesh a comparably
    fine tessellation, so the envelope stage (cell = 2 voxels = 2 x median edge / 1.17) resolves its thinnest
    parts (template tibiae are ~0.4 % of body length thick).
    """
    import trimesh

    vv, ff = trimesh.remesh.subdivide_to_size(v, f, max_edge=max_edge, max_iter=12)
    return np.asarray(vv, dtype=np.float64), np.asarray(ff)


def generate(
    out,
    n,
    seed=0,
    damage="none",
    shape_sd=1.0,
    limit_frac=0.7,
    length_mm=(1.5, 12.0),
    max_edge_per_body_length=400,
    voxels_per_body_length=2000,
):
    rng = np.random.default_rng(seed)
    smal, dd = load_model()
    names = list(dd["J_names"])
    limits = np.asarray(dd["joint_limits"], dtype=np.float64)
    sd = np.sqrt(np.diag(dd["shape_cov"]))
    faces = smal.faces.cpu().numpy()
    weights = smal.weights.cpu().numpy()
    os.makedirs(out, exist_ok=True)
    rows = []
    for i in range(n):
        betas = rng.normal(size=len(sd)) * sd * shape_sd
        theta = sample_pose(names, limits, rng, limit_frac)
        with torch.no_grad():
            verts, joints, _rs, _vs = smal(
                torch.tensor(betas[None], dtype=torch.float32), torch.tensor(theta[None], dtype=torch.float32)
            )
        verts, joints = verts[0].numpy().astype(np.float64), joints[0].numpy().astype(np.float64)
        pivots = smal.J_transformed[0].numpy().astype(np.float64)
        # body length in model units: extent of the head, mesosoma, waist and gaster (no appendages) along the
        # gaster -> head axis at this pose
        body = np.isin(
            weights.argmax(1), [names.index(k) for k in ["b_t", "b_h", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]]
        )
        u = joints[names.index("b_h")] - joints[names.index("b_a_4")]
        u /= np.linalg.norm(u)
        proj = verts[body] @ u
        body_len_units = float(proj.max() - proj.min())
        length_um = rng.uniform(*length_mm) * 1000.0
        s = length_um / body_len_units
        R = random_rotation(rng)
        t = rng.uniform(1000, 8000, size=3)

        def to_scan(x):
            return (s * x) @ R.T + t

        v_s, f_s = to_scan(verts), faces
        dmg = None
        if damage == "random" and rng.random() < 0.5:
            dmg = str(rng.choice(sorted(DAMAGE)))
        elif damage in DAMAGE:
            dmg = damage
        watertight = True  # collapsing keeps the surface closed
        if dmg:
            v_s = remove_part(v_s, weights, names, DAMAGE[dmg]["bones"], to_scan(pivots))
        # finer tessellation (every edge <= body length / max_edge_per_body_length) and a declared scan resolution
        # of body length / voxels_per_body_length (real 2024 scans: ~1200-2100 voxels per body length)
        v_s, f_s = scan_like_subdivision(v_s, f_s, length_um / max_edge_per_body_length)
        voxel = length_um / voxels_per_body_length
        name = f"synthetic_{seed:03d}_{i:04d}" + (f"_{dmg}" if dmg else "")
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        write_stl(os.path.join(d, name + ".stl"), v_s, f_s)
        gt = {
            "name": name,
            "caste": "worker",
            "family_or_other": "Formicidae",
            "voxel_size": f"{voxel:.4f} µm",
            "synthetic": True,
            "units": "micrometre",
            "body_length_um": length_um,
            "model_to_scan": {
                "scale": s,
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "note": "x_scan = scale * R @ x_model + t",
            },
            "betas": betas.tolist(),
            "theta": theta.tolist(),
            "damage": dmg,
            "missing_joints": DAMAGE[dmg]["missing"] if dmg else [],
            "watertight": watertight,
            "joint_names": names,
            "joints_um": {n_: to_scan(joints[k][None])[0].tolist() for k, n_ in enumerate(names)},
            "pivots_um": {n_: to_scan(pivots[k][None])[0].tolist() for k, n_ in enumerate(names)},
        }
        with open(os.path.join(d, name + ".json"), "w") as fh:
            json.dump(gt, fh, indent=1)
        rows.append((name, round(length_um), dmg, watertight, len(f_s)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--damage", default="none", help="none | random | one of " + ", ".join(sorted(DAMAGE)))
    ap.add_argument("--shape-sd", type=float, default=1.0)
    ap.add_argument("--limit-frac", type=float, default=0.7)
    ap.add_argument("--max-edge-per-body-length", type=float, default=400)
    ap.add_argument("--voxels-per-body-length", type=float, default=2000)
    a = ap.parse_args()
    for r in generate(
        a.out,
        a.n,
        a.seed,
        a.damage,
        a.shape_sd,
        a.limit_frac,
        max_edge_per_body_length=a.max_edge_per_body_length,
        voxels_per_body_length=a.voxels_per_body_length,
    ):
        print(*r)


if __name__ == "__main__":
    main()
