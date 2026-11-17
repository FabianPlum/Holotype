# tunosemi's Sketchfab ants: download, clean, orient and fit

This folder reproduces a registration dataset built from the ant models of Sketchfab user
[tunosemi](https://sketchfab.com/tunosemi). It makes 135 watertight ant meshes of about 100k vertices each, in the
OmniAnt frame. It then fits `SMIL_OmniAnt.pkl` to them with `fitter_3d` and scores the fit, so other registration
methods can be compared on identical targets. The fit of 15 November 2026 is the baseline in
`expected/baseline_fitter3d_metrics.csv`.

**Licence.** tunosemi's models are CC BY-NC-ND (132) or CC BY-ND (3) on Sketchfab. tunosemi agreed to their use for
Holotype; that is not permission to redistribute them. Never commit downloads or meshes derived from them, and
credit tunosemi in anything published.

## Quick start

From the repository root, in the `pytorch3d` conda environment (`environment.yml`):

```bash
# 1. a Sketchfab API token (sketchfab.com/settings/password -> "API Token")
echo YOUR_TOKEN > ~/.sketchfab_token && chmod 600 ~/.sketchfab_token     # or: export SKETCHFAB_TOKEN=...

# 2. everything up to the fitted baseline
python custom_processing/sketchfab_ants/pipeline.py all --data_dir /path/to/tunosemi_ants --blender /path/to/blender
```

`all` runs `download`, `process`, `orient`, `verify`, `fit` and `evaluate`. Each step can also be run alone, for
example `pipeline.py orient --data_dir ...`.

| step | what it does | output in `DATA` | time on the reference run |
|---|---|---|---|
| `list` | lists tunosemi's models (public API) and compares the ants with the pinned list | `metadata/` | 1 min |
| `download` | 135 downloadable ant models as glb (token needed); restartable | `raw/<key>/<key>.glb`, 2.2 GB | 15 min |
| `process` | Blender: shells fused by voxel remesh, about 100k vertices, very light smoothing | `intermediate_source_axes/`, 1.3 GB | 10 min, 6 Blender processes |
| `orient` | best of 24 axis rotations against the OmniAnt template | `processed/<key>.obj`, 0.9 GB; `diagnostics/orientation_*` | 5 to 60 min |
| `review` | renders and contact sheets for checking orientation by eye (optional) | `diagnostics/contact_sheet_*.png` | 2 min |
| `verify` | compares processing and orientation with `expected/processing_summary.csv` | printout | seconds |
| `fit` | `fitter_3d/optimise.py` with the stages of `fitter_3d/ants_cfg.yaml` | `fit3d_results_OmniAnt/Stage_*.npz`, `fit3d_OmniAnt_run.log` | 6 h on an M5 Pro CPU; much faster on CUDA |
| `evaluate` | chamfer and p95 of the fit, compared with the baseline | `diagnostics/metrics_*.csv` | 5 min |

**Requirements**
- **Blender 4.5 LTS.** The reference run used 4.5.16, so `verify` expects its remesh and decimation results;
  other versions may differ slightly.
- **About 5 GB of disk.**
- **RAM:** about 2 GB per Blender process (`--jobs`).

### The pinned model list

`tunosemi_ant_models.json` lists the 140 ant models found on 14 November 2026, 135 of them downloadable. The other
five are marked non-downloadable on Sketchfab: *Echinopla melanarctos*, *Holcoponera striatula*, *Melissotarsus*,
*Myrmecia* and *Myrmoteras*.

`ant_taxa.py` selects the ants by genus, because tunosemi also uploads beetles, wasps, spiders and other outgroups.
Ants are never chosen by tag. To include species published since, run `list` and then the other steps with
`--fresh_list`. `list` prints new models and any model tagged "ant" whose genus is missing from `ant_taxa.py`.

## What the processing does

`custom_processing/prepare_sketchfab_models_for_mesh_fitting.py`, run in Blender through `batch_process_models.py`,
does the following for each model:

1. **Join and weld.** It joins the glTF chunks and merges the vertices split at chunk and UV seams.
2. **Fuse.** tunosemi builds each body segment as its own closed shell (47 to 3,796 shells per model), and the
   shells overlap inside the body. A voxel remesh fuses them into one watertight, manifold surface. The voxel size
   starts at 0.07 % of the bounding-box diagonal, which keeps the tarsal claws. It grows ×1.4, up to 0.3 %, until
   99.9 % of the surface area is one component. Only that component is kept.
3. **Decimate and clean.** It decimates to 100k vertices and applies very light Laplacian smoothing (λ 0.3,
   2 iterations, volume-preserving). The mean vertex shift is about 0.15 edge lengths. Degenerate faces are
   removed and normals made consistent.

`custom_processing/orient_meshes_to_template.py` then rotates each mesh into the template frame: **head +X, dorsal
+Z, left side +Y**. It scores the 24 proper axis-aligned rotations by chamfer distance to `v_template`. tunosemi's
glbs come in with the head at -Y, and 134 models get the same rotation. *pheidole Caldwelli miner* was uploaded
facing the other way and gets the opposite one.

The processed meshes keep tunosemi's arbitrary modelling units, centred on the bounding-box centre. Because the
units carry no real size, nothing in this folder assumes millimetres or micrometres.

**Expected results** (`verify`):
- all 135 meshes are single-component and watertight
- 115 of them fuse at the finest voxel size; *Basiceros* and *Tatuidris* need the largest
- *Tatuidris* is the only model where specks (0.16 % of its area) were dropped without reaching the 99.9 % criterion

## Fitting and the baseline

On a CUDA machine, `fit` runs `python -m fitter_3d.optimise` directly. Without CUDA it uses `run_fit_cpu.py`. That
wrapper replaces pytorch3d's brute-force CPU chamfer with a dense one, which matches pytorch3d's value and gradients
exactly (checked at start-up). It also turns off the per-stage matplotlib plots, which take minutes and about
1.2 GB per 200k-face target. `config.SPLIT_TARGET_MESHES_INTO_BATCHES_OF_SIZE` (100) splits the 135 meshes into two
batches.

Baseline (`expected/baseline_fitter3d_metrics.csv`, fitter_3d frame: target centred and scaled so that its largest
absolute coordinate is 1):

| fitter_3d stage | chamfer, median | chamfer, worst | p95 target→fit, median | as % of body length |
|---|---|---|---|---|
| init | 6.3e-4 | 2.1e-3 | 0.039 | 2.90 % |
| pose + shape | 3.3e-4 | 1.0e-3 | 0.026 | 1.88 % |
| deform coarse | 1.0e-4 | 2.9e-4 | 0.012 | 0.89 % |
| deform fine | 0.86e-4 | 2.8e-4 | 0.012 | 0.83 % |

The hardest specimens are *Cephalotes* and *C. clypeatus* (disc-shaped heads, outside OmniAnt's shape space),
*Myrmica rubra* (the deformation crumples the head) and *Neivamyrmex pseudops* (the fitted antenna loops).

## Testing another registration method on this set

- **Input:** `DATA/processed/<key>.obj`, head +X, dorsal +Z, arbitrary units. fitter_3d itself normalises each
  target as in `fitter_3d.utils.load_meshes`.
- **Scoring:** `evaluate_registration.py` uses the same metric as the baseline and is template-agnostic, so a
  method with a different template or topology can be compared directly:

  ```bash
  # registered meshes in the processed targets' own coordinates, one <key>.obj each
  python custom_processing/sketchfab_ants/evaluate_registration.py --targets DATA/processed \
      --mesh_dir MY_RESULTS --frame target --out DATA/diagnostics/metrics_my_method.csv --baseline_stage deform_fine
  # or a fitter_3d stage file
  python custom_processing/sketchfab_ants/evaluate_registration.py --targets DATA/processed \
      --npz RESULTS/Stage_1_default.npz --out ... --baseline_stage pose_shape
  ```

- **AntScan pipeline (`feature/antscan-registration-pipeline`):** it reads `DATA/<name>/<name>.stl` plus a metadata
  `<name>.json` (`caste`, `family_or_other`, `voxel_size`, `problems_other`) and works in micrometres. tunosemi's
  units are arbitrary, so feeding this set to it needs a decision on scale and on `voxel_size`; there is no exporter
  for it yet. Caste is in the model names (queen, major, minor); everything else is a worker.

## Reference fit report

The "OmniAnt Fit Atlas" covers fit quality, failure cases and a PCA of the pose-free fitted shapes with a 3D
explorer. It is built by the scripts in `report/`, after `fit`:

```bash
python custom_processing/sketchfab_ants/report/analyse_fit.py --repo . --results DATA/fit3d_results_OmniAnt \
    --meshes DATA/processed --log DATA/fit3d_OmniAnt_run.log --out DATA/report_build
python custom_processing/sketchfab_ants/report/build_report_assets.py DATA
for k in 0 1 2 3; do blender --background --factory-startup --python custom_processing/sketchfab_ants/report/render_jobs.py -- DATA/report_build/render/jobs_$k.json; done
blender --background --factory-startup --python custom_processing/sketchfab_ants/report/export_source_shells.py -- . \
    DATA/raw/Harpegnathos_af9703ae/Harpegnathos_af9703ae.glb DATA/report_build/harpegnathos_source.obj
python custom_processing/sketchfab_ants/report/build_extra_renders.py DATA Harpegnathos_af9703ae DATA/report_build/harpegnathos_source.obj \
    Myrmica_rubra_1e5696ca Neivamyrmex_pseudops_abf2556e Cephalotes_3d1eaa8a
blender --background --factory-startup --python custom_processing/sketchfab_ants/report/render_jobs.py -- DATA/report_build/render/jobs_extra.json
python custom_processing/sketchfab_ants/report/assemble_site.py DATA SITE_OUT
```

`analyse_fit.py` un-poses each fit. It re-skins with zero joint rotations and carries the per-vertex deformation
back through each vertex's skinning transform; re-skinning reproduces fitter_3d's vertices exactly (`skin_check_PROBE.txt`). It
then runs the PCA, the per-vertex and per-region errors and the joint scales.

`report/index.src.html` contains copy specific to the 15 November run: dates, the *Harpegnathos* figures and the
names given to PC1 to PC4. Edit it before building a report for another run.

## CPU-only machines and macOS

On Apple Silicon, conda-forge has no pytorch3d for Python 3.10 with torch 2.3.1. The reference run built
pytorch3d 0.7.8 from source:

```bash
pip install torch==2.3.1 torchvision==0.18.1
CC=clang CXX=clang++ CFLAGS=-Wno-invalid-specialization CXXFLAGS=-Wno-invalid-specialization \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
```

- **OpenMP clash.** Importing numpy before torch segfaulted until torch's bundled `torch/lib/libomp.dylib` was
  replaced by a symlink to the environment's `llvm-openmp` (`$CONDA_PREFIX/lib/libomp.dylib`).
- **Sleep.** A laptop that sleeps stalls the fit; `caffeinate -i` does not prevent sleep with the lid closed.
