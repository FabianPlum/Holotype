# Holotype Tests

`pytest` suite for the Holotype pipeline — the optimization fitters, the neural
single/multi-view path, the config system, augmentation, triangulation, and
animation export.

## Prerequisites

```bash
# 1. Activate the conda environment (see the root README "Installation")
conda activate pytorch3d
```

- **Run pytest from the repository root** (`pytest`, or `pytest tests/`) so the `data/` relative paths resolve. `pytest.ini` sets `testpaths = tests`, so a plain `pytest` collects only this directory; module-local `test_*.py` files elsewhere in the tree (e.g. `smal_fitter/sleap_data/`) are dev/integration scripts run explicitly by path, not auto-collected.
- The **slow** training test (`test_neural_smil_training_pipeline`) shells out to
  `python -m smal_fitter.neuralSMIL.train_smil_regressor` on the `test_textured` dataset, which must exist at
  `data/replicAnt_trials/replicAnt-x-SMIL-TEX/` (20 images + SMIL annotations).
- The fitter tests run the real optimization scripts, so they need PyTorch3D and a
  valid `config.SMAL_FILE` model (`test_fitter_3d_optimise` uses the bundled
  `fitter_3d/ATTA_BOI` mesh).

## Running tests

The whole suite collects and runs cleanly from the repository root — no `PYTHONPATH`
prefix, no `--continue-on-collection-errors`, and no per-module workarounds are needed.

The only registered marker is `slow` (see `pytest.ini`). Six tests carry it: three in `test_pipeline.py`
(`test_fitter_3d_optimise`, `test_smal_fitter_optimize_to_joints`, `test_neural_smil_training_pipeline`), one in
`test_fitter3d_joint_limits.py`, and one each in `test_antscan_joints.py` and `test_antscan_registration_pipeline.py`.

```bash
# The whole suite
pytest

# Fast tests only (skip the slow training subprocess)
pytest -m "not slow"

# A subset of modules
pytest tests/test_config_system.py tests/test_augmentation.py \
       tests/test_curriculum_sync.py tests/test_pipeline.py -m "not slow"

# A single test function
pytest tests/test_pipeline.py::test_neural_smil_config_validation -v -s
```

> Use `-s` to see `print()` output from the integration tests (they are verbose).

## Test modules

| File | Tests | Covers |
|------|------:|--------|
| `test_pipeline.py` | 4 | Integration smoke tests: fitter_3d optimise, SMAL `optimize_to_joints`, neural config validation, and a slow end-to-end neural training run (details below). |
| `test_config_system.py` | 46 | The unified JSON/dataclass config system (`smal_fitter/neuralSMIL/configs/`): loading, dataclass merging, validation, legacy-dict conversion, curriculum application, CLI override precedence, round-trip serialization. |
| `test_augmentation.py` | 15 | Multi-view augmentation pipeline: photometric augmentations preserve camera params and 2D keypoints, geometric behavior, etc. |
| `test_triangulation_consistency.py` | 12 | The differentiable triangulation consistency loss: round-trip known 3D joints → project to 2D → `_triangulate_joints_dlt` → compare. |
| `test_animation_export.py` | 5 | Animation exporter round-trip: synthetic `predicted_params` → `AnimationRecorder` → reload the `.npz` + `.json`. |
| `test_curriculum_sync.py` | 1 | Verifies the loss curriculum from a JSON config is correctly synced into the legacy `TrainingConfig`. |
| `test_antscan_joints.py` | 5 (1 slow) | AntScan joint method (`fitter_3d/antscan_joints`): the ensemble on hand-made states, resume, memory cap and time limit, the classifier file (recorded hash, restricted unpickler, scikit-learn version check); slow: the command end to end on a posed template (about 4 min, 7 GB). |
| `test_antscan_registration_pipeline.py` | 6 (1 slow) | AntScan registration pipeline (`fitter_3d/antscan_registration`) on a specimen made from the model: every output file's contract, the stage file in the scan frame and under `SMAL3DFitter`'s own forward, the unposed bundle, a bundle row that does not depend on the batch, the intake and grade rules; slow: a batch of degenerate inputs ends as statuses. |
| `test_fitter_3d_geomreg.py` | 3 | Closed-form start (pivot and medial joints) and native refit of the geometric registration core (`fitter_3d/geomreg`, numpy only). |
| `test_fitter_3d_geomreg_pipeline.py` | 1 | `fitter_3d/geomreg` end to end on a specimen made from the model: native parameters reproduce the mesh, a specimen that cannot be registered is a status. |
| `test_fitter_3d_geom_stage.py` | 2 | The geometric registration as a `fitter_3d` stage through `optimise.main` (`scheme: 'geom'`), and the default joint-informed path with a missing part. |

The AntScan tests need no scan data (their specimens are posed copies of `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`). `test_antscan_registration_pipeline.py` is skipped when `embreex` is not installed; its module fixture registers one specimen once (one is enough for the contracts, and it keeps the fixture short), which is the slowest setup of the fast suite. The fast tests of `test_antscan_joints.py` run without scikit-learn and numba; the slow one is skipped without them.

These files in this directory are **not** pytest modules (no `test_*` functions):

- `config_test.py` — a fast-testing config-override helper (forces the OmniAnt `SMPL_fit.pkl` model); imported by manual test runs, not collected by pytest.
- `validate_multiview_replicant_loader.py` — a manual **visual** validation of multi-view replicAnt camera geometry + model fit; run it directly to produce diagnostic visualizations.
- `antscan_synth_inputs.py` — builds the synthetic AntScan specimens (posed template meshes with joint and limb products) that the AntScan tests use.
- `antscan_stage_forward_helper.py` — run in a subprocess by `test_antscan_registration_pipeline.py`: checks that `SMAL3DFitter`'s forward of a stage file's parameters plus `deform_verts` gives its `verts`.

## `test_pipeline.py` in detail

| Test | Speed | What it does |
|------|-------|--------------|
| `test_fitter_3d_optimise` | medium | Runs `python -m fitter_3d.optimise --mesh_dir fitter_3d/ATTA_BOI --scheme default --lr 1e-3 --nits 10` on the bundled Atta mesh. |
| `test_smal_fitter_optimize_to_joints` | medium | Runs the SMAL `optimize_to_joints` optimization pipeline. |
| `test_neural_smil_config_validation` | fast (<1 s, **pure logic, no filesystem**) | Asserts `get_data_path('test_textured')` resolves to `data/replicAnt_trials/replicAnt-x-SMIL-TEX`, the 85/5/10 train/val/test split, that the loss-curriculum weights are a dict containing `keypoint_2d`, and that the per-epoch learning rate is positive. |
| `test_neural_smil_training_pipeline` | **slow** (`@pytest.mark.slow`) | Shells out to `python -m smal_fitter.neuralSMIL.train_smil_regressor` for **2 epochs** on `test_textured` with `--batch_size 4 --checkpoint DISABLE_CHECKPOINT_LOADING --scale_trans_mode ignore`. Writes to a temporary directory and disables checkpoint saving, so it never overwrites trained models; artifacts are cleaned up afterwards. Asserts training completes. |

## CI

`pytest -m "not slow"` is the fast-feedback signal on every change, and
`pytest -m "slow"` exercises the full training subprocess on scheduled builds
(or before a release).
