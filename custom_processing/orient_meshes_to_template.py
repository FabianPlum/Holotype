"""
Rotates target meshes into the frame of the SMIL template used by fitter_3d.

fitter_3d (optimise.py) starts every fit at zero global rotation, and its usual first stage
('init_rot_lock', see fitter_3d/ants_cfg.yaml) does not optimise rotation at all, so target
meshes have to be in the template's frame already. For SMIL_OmniAnt.pkl that is head along +X,
dorsal +Z and the left side (joints *_l) along +Y.

For every OBJ in --input_dir, the 24 proper rotations that map coordinate axes onto coordinate
axes (no mirroring) are scored by the chamfer distance between surface samples of the
target and of the template v_template, both normalised exactly like fitter_3d.utils.load_meshes
(mean-centred, divided by the largest absolute coordinate; that normalisation is invariant to
these rotations). The best rotation is applied about the origin and the mesh is written to
--output_dir. Meshes whose best and second-best scores are within --ambiguity_margin of each
other are flagged in the CSV and the contact sheets for manual review.

Diagnostics (--diagnostics_dir): orientation_scores.csv and orientation_PROBE_<page>.png contact
sheets showing each oriented mesh (grey) over the template (colour) in top (XY) and side (XZ)
views.

Usage (from the repository root, pytorch3d env):
    python -m custom_processing.orient_meshes_to_template --input_dir <decimated_objs> \
        --output_dir <oriented_objs> --diagnostics_dir <diagnostics> [--smal_file 3D_model_prep/SMIL_OmniAnt.pkl]
"""

import argparse
import csv
import glob
import itertools
import os
import pickle

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree


def proper_axis_rotations():
    """The 24 rotation matrices (det +1) whose rows are signed coordinate axes."""
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                R[row, col] = s
            if np.isclose(np.linalg.det(R), 1.0):
                rots.append(R)
    return rots


def rotation_label(R):
    """Describes R by where it sends the source axes, e.g. '+X<-(-Y) +Y<-(+X) +Z<-(+Z)'."""
    names = "XYZ"
    parts = []
    for row in range(3):
        col = int(np.argmax(np.abs(R[row])))
        parts.append(f"+{names[row]}<-({'+' if R[row, col] > 0 else '-'}{names[col]})")
    return " ".join(parts)


def normalise_like_load_meshes(verts):
    """Same centring and scaling as fitter_3d.utils.load_meshes."""
    verts = verts - verts.mean(0)
    return verts / verts.abs().max()


def load_template(smal_file):
    with open(smal_file, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    verts = torch.as_tensor(np.asarray(dd["v_template"]), dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(dd["f"]).astype(np.int64))
    return verts, faces


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def score_rotations(t_verts, t_faces, template_pts, template_tree, rots, n_points, seed):
    """
    Chamfer distance to the template for each candidate rotation (lower is better).

    Same quantity as pytorch3d.loss.chamfer_distance (mean squared nearest-neighbour distance
    in both directions, summed), computed with KD-trees: pytorch3d's CPU KNN is brute force and
    far too slow for 24 candidates x 10k points per mesh.
    """
    torch.manual_seed(seed)
    mesh = Meshes(verts=[normalise_like_load_meshes(t_verts)], faces=[t_faces])
    pts = sample_points_from_meshes(mesh, n_points)[0].numpy().astype(np.float64)
    target_tree = cKDTree(pts)
    scores = []
    for R in rots:
        d_target, _ = template_tree.query(pts @ R.T, workers=-1)  # rotated target -> template
        d_template, _ = target_tree.query(template_pts @ R, workers=-1)  # template rotated by R^-1 -> target
        scores.append(np.mean(d_target**2) + np.mean(d_template**2))
    return np.array(scores)


def chamfer_crosscheck_PROBE(t_verts, t_faces, template_pts, R, n_points, seed):
    """pytorch3d chamfer_distance for one rotation, to cross-check the KD-tree scores."""
    torch.manual_seed(seed)
    mesh = Meshes(verts=[normalise_like_load_meshes(t_verts)], faces=[t_faces])
    pts = sample_points_from_meshes(mesh, n_points)[0]
    rotated = (pts @ torch.as_tensor(R.T, dtype=torch.float32))[None]
    d, _ = chamfer_distance(rotated, torch.as_tensor(template_pts, dtype=torch.float32)[None])
    return float(d)


def plot_pages(entries, template_pts, diagnostics_dir, per_page=12):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp = template_pts
    for page_idx in range(0, len(entries), per_page):
        page = entries[page_idx : page_idx + per_page]
        cols = 4
        rows = int(np.ceil(len(page) / 2))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.6), squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        for k, e in enumerate(page):
            r, c = divmod(k, 2)
            p = e["plot_pts"]
            for j, (a, b, view) in enumerate(((0, 1, "top XY"), (0, 2, "side XZ"))):
                ax = axes[r, 2 * c + j]
                ax.scatter(tp[:, a], tp[:, b], s=0.3, c="tab:orange", alpha=0.25, linewidths=0)
                ax.scatter(p[:, a], p[:, b], s=0.3, c="0.2", alpha=0.5, linewidths=0)
                ax.set_aspect("equal")
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05) if b == 1 else ax.set_ylim(-0.6, 0.6)
                ax.annotate("head +X →", (0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=6)
                flag = "  [AMBIGUOUS]" if e["ambiguous"] else ""
                title = f"{e['name'][:34]}{flag}\n{view}  margin {e['margin']:.0%}" if j == 0 else view
                ax.set_title(title, fontsize=7, color="red" if e["ambiguous"] else "black")
                ax.set_axis_on()
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(
            "grey: oriented target, orange: template v_template (both normalised as in load_meshes)", fontsize=9
        )
        fig.tight_layout()
        out = os.path.join(diagnostics_dir, f"orientation_PROBE_{page_idx // per_page:02d}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")


def main(args):
    if args.smal_file is None:
        import config

        args.smal_file = config.SMAL_FILE
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.diagnostics_dir, exist_ok=True)

    tmpl_verts, tmpl_faces = load_template(args.smal_file)
    torch.manual_seed(args.seed)
    template_mesh = Meshes(verts=[normalise_like_load_meshes(tmpl_verts)], faces=[tmpl_faces])
    template_pts = sample_points_from_meshes(template_mesh, args.n_points)[0].numpy().astype(np.float64)
    template_tree = cKDTree(template_pts)

    rots = proper_axis_rotations()
    identity_idx = next(i for i, R in enumerate(rots) if np.allclose(R, np.eye(3)))
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.obj")))
    print(f"{len(files)} meshes, {len(rots)} candidate rotations, template {args.smal_file}")

    entries = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        verts, faces, _ = load_obj(path, load_textures=False)
        faces = faces.verts_idx
        scores = score_rotations(verts, faces, template_pts, template_tree, rots, args.n_points, args.seed)
        order = np.argsort(scores)
        best, second = order[0], order[1]
        if not entries:  # once: KD-tree score vs pytorch3d chamfer_distance on identical samples
            ref = chamfer_crosscheck_PROBE(verts, faces, template_pts, rots[best], args.n_points, args.seed)
            print(f"cross-check {name}: KD-tree {scores[best]:.6f} vs pytorch3d chamfer {ref:.6f} "
                  f"(rel. diff {abs(scores[best] - ref) / ref:.1e})")
        margin = float((scores[second] - scores[best]) / scores[best])
        R = torch.as_tensor(rots[best], dtype=torch.float32)
        oriented = verts @ R.T
        write_obj(os.path.join(args.output_dir, f"{name}.obj"), oriented.numpy(), faces.numpy())

        plot_mesh = Meshes(verts=[normalise_like_load_meshes(oriented)], faces=[faces])
        torch.manual_seed(args.seed)
        e = dict(
            name=name,
            rotation=rotation_label(rots[best]),
            score_best=float(scores[best]),
            score_second=float(scores[second]),
            second_rotation=rotation_label(rots[second]),
            score_unrotated=float(scores[identity_idx]),
            margin=margin,
            ambiguous=margin < args.ambiguity_margin,
            plot_pts=sample_points_from_meshes(plot_mesh, 4000)[0].numpy(),
        )
        entries.append(e)
        print(f"{name}: {e['rotation']}  best {e['score_best']:.4f}  2nd {e['score_second']:.4f}  margin {margin:.0%}"
              + ("  AMBIGUOUS" if e["ambiguous"] else ""))

    csv_path = os.path.join(args.diagnostics_dir, "orientation_scores.csv")
    keys = [
        "name", "rotation", "score_best", "score_second", "second_rotation", "score_unrotated", "margin", "ambiguous",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for e in entries:
            w.writerow({k: e[k] for k in keys})
    print(f"wrote {csv_path}")
    plot_pages(entries, template_pts, args.diagnostics_dir)

    counts = {}
    for e in entries:
        counts[e["rotation"]] = counts.get(e["rotation"], 0) + 1
    print("rotations chosen:", counts)
    print("ambiguous:", [e["name"] for e in entries if e["ambiguous"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Directory of OBJ meshes (e.g. Blender-decimated)")
    parser.add_argument("--output_dir", required=True, help="Directory for the oriented OBJ meshes")
    parser.add_argument("--diagnostics_dir", required=True, help="Directory for scores CSV and contact sheets")
    parser.add_argument("--smal_file", default=None, help="Template model .pkl (default: config.SMAL_FILE)")
    parser.add_argument("--n_points", type=int, default=10000, help="Surface samples per mesh for scoring")
    parser.add_argument("--ambiguity_margin", type=float, default=0.15,
                        help="Flag meshes whose best two rotations differ by less than this relative margin")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
