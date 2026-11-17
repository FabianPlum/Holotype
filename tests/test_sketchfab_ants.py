"""Tests for the tunosemi Sketchfab ant pipeline (custom_processing/sketchfab_ants, orient_meshes_to_template)."""
import csv
import json
import os

import numpy as np
import pytest
import torch

from custom_processing.sketchfab_ants import ant_taxa

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "custom_processing", "sketchfab_ants")


def test_every_genus_has_one_subfamily():
    seen = {}
    for sub, genera in ant_taxa.SUBFAMILY_GENERA.items():
        for g in genera.split():
            assert g not in seen, f"{g} in {seen.get(g)} and {sub}"
            seen[g] = sub


@pytest.mark.parametrize("title,ant,name", [
    ("Myrmica_rubra", True, "Myrmica rubra"),
    ("Ection burchellii", True, "Eciton burchellii"),
    ("pheidole Caldwelli miner", True, "Pheidole caldwelli minor"),
    ("Procryptocerus cf.mayri", True, "Procryptocerus cf. mayri"),
    ("Ｍyrmoteras", True, "Myrmoteras"),
    ("Myrmarachne", False, None),  # ant-mimicking spider
    ("Vespa mandarinia", False, None),
    ("Bee Anatomical Model", False, None),
])
def test_ant_selection_and_names(title, ant, name):
    assert ant_taxa.is_ant(title) is ant
    if ant:
        assert ant_taxa.display_name(title) == name


def test_pinned_list_and_expected_files_agree():
    models = json.load(open(os.path.join(PKG, "tunosemi_ant_models.json")))["models"]
    assert len(models) == 140 and sum(m["downloadable"] for m in models) == 135
    assert all(ant_taxa.is_ant(m["name"]) and ant_taxa.subfamily_of(m["name"]) == m["subfamily"] for m in models)
    keys = {m["key"] for m in models if m["downloadable"]}
    assert len(keys) == 135
    for f in ("processing_summary.csv", "baseline_fitter3d_metrics.csv"):
        rows = list(csv.DictReader(open(os.path.join(PKG, "expected", f))))
        assert {r["key"] for r in rows} == keys, f


def test_axis_rotations_are_the_24_proper_ones():
    from custom_processing.orient_meshes_to_template import proper_axis_rotations

    rots = proper_axis_rotations()
    assert len({tuple(R.ravel()) for R in rots}) == 24
    assert all(np.isclose(np.linalg.det(R), 1) and np.allclose(R @ R.T, np.eye(3)) for R in rots)


def test_orientation_recovers_a_rotated_template():
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    from scipy.spatial import cKDTree

    from custom_processing.orient_meshes_to_template import (
        load_template, normalise_like_load_meshes, proper_axis_rotations, score_rotations)

    verts, faces = load_template(os.path.join(os.path.dirname(HERE), "3D_model_prep", "SMIL_OmniAnt.pkl"))
    torch.manual_seed(0)
    tp = sample_points_from_meshes(Meshes([normalise_like_load_meshes(verts)], [faces]), 3000)[0].numpy().astype(float)
    rots = proper_axis_rotations()
    R_src = rots[7]  # an arbitrary non-identity rotation: the "source" frame of a downloaded model
    rotated = verts @ torch.as_tensor(R_src.T, dtype=torch.float32)
    scores = score_rotations(rotated, faces, tp, cKDTree(tp), rots, 3000, seed=0)
    assert np.allclose(rots[int(np.argmin(scores))] @ R_src, np.eye(3))


def test_dense_chamfer_matches_pytorch3d():
    from pytorch3d.loss import chamfer_distance

    from custom_processing.sketchfab_ants.run_fit_cpu import dense_chamfer_distance

    torch.manual_seed(0)
    x, y = torch.randn(2, 200, 3), torch.randn(2, 300, 3)
    xa, ya, xb, yb = (t.clone().requires_grad_() for t in (x, y, x, y))
    la, _ = chamfer_distance(xa, ya)
    lb, _ = dense_chamfer_distance(xb, yb, chunk=1)
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb, rtol=1e-6)
    assert torch.allclose(xa.grad, xb.grad, atol=1e-6) and torch.allclose(ya.grad, yb.grad, atol=1e-6)
