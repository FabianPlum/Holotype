# fitter_3d/antscan_joints: the AntScan joint method (round 7, four-state ensemble)

Places the 55 joints of the OmniAnt template (plus `head_front` and `gaster_tip`) and the limb centrelines on a
prepared AntScan envelope mesh, without a posed model. Its products are the input of
`fitter_3d/antscan_registration` (`--joints`, `--limbs`; the per-state products and the path stages are found next
to them).

```
python -m fitter_3d.antscan_joints.run --names NAMES.txt --env ENV_DIR --out OUT_DIR [--jobs N] [--mem-cap-gb G]
python -m fitter_3d.antscan_registration.run --names NAMES.txt --joints OUT_DIR/proposals --limbs OUT_DIR/limbs \
    --env ENV_DIR --out REG_DIR
```

`python -m fitter_3d.antscan_joints.run --help` describes the options, the output layout, resume and the memory cap.
What the steps and the ensemble do, the outputs, the cost per specimen and the place of the method in the whole
workflow: `docs/antscan_workflow.md` (step 4). How the method was chosen, with its evaluation:
`docs/antscan_joint_proposals_buildoff.md`.

## Layout

| path | what |
|---|---|
| `run.py` | the command: scheduling (specimen by specimen, `--jobs` processes), resume from step markers (a step that runs again removes its old outputs and makes every dependent step and the ensemble run again), per-job memory cap and time limit, `summary.tsv`, `jobs.jsonl` |
| `stage.py` | one stage (`trunk`, `paths_A`, `paths_B`, `paths_C`) of one specimen under one grid state, in its own process; the four sign states of TRUNK's working frame |
| `steps.py` | `thin`, `folds` and `ensemble`, each in its own process |
| `method/trunk` | TRUNK: voxel occupancy, coarse PCA frame, trunk core, head end, trunk joints, body length, working frame |
| `method/paths_A` | PATHS-A: geodesic limb paths with contact cutting, named with fitted priors (`priors_final/priors_all_trunk.json`) |
| `method/paths_B` | PATHS-B: curve-skeleton graph of the solid, tubes paired through junctions |
| `method/paths_C` | PATHS-C: tube decomposition with contact cuts (`contact_model.json`) |
| `method/thin` | THIN: mesh-native extension of thin leg / antenna tips, discovery of limbs no approach found |
| `method/assemble` | combination of the three approaches per limb, ALONG joint placement, anatomical frame |
| `method/along` | ALONG: joints along a limb centreline (classifier `along_model.pkl` + segment-length priors) |
| `method/folds` | FOLDS: folded knees, farthest leg tip, distal extension, antennal elbow |
| `method/ensemble` | the four states combined: state vote per limb, joint medoid over agreeing states, spread confidence |

Per specimen and state: trunk -> paths_A, paths_B, paths_C -> thin -> assemble -> folds; then the ensemble over the
states. The method modules use flat imports from their own folder (each stage or step runs in its own process with
that folder on `sys.path`); they are not meant to be imported as a package.

## Terms used in the code

| term | meaning |
|---|---|
| round N, rN | the N-th design round of the joint method during its development. Round 7 (four-state ensemble over round 6) is what ships; round 3 was an earlier template-based method whose prior and fallback are not shipped |
| S6, S7 (`s7folds.py`, `s7geom.py`, `s7lib.py`) | modules and version labels of round 6 and round 7 |
| TRUNK, PATHS-A / -B / -C, THIN, ALONG, FOLDS | the steps in the table above |
| state, sign state, s0..s3 | one of the four sign choices of TRUNK's working frame (PCA axes signed by the third moment, then one pair of axes flipped); a grid state adds a voxel-grid phase (only phase 0 ships) |
| BL | body length (micrometres); lengths in the code are often in % BL |
| product | the joint product (`proposals/<name>/<name>_joints.json`) or the limb product (`limbs/<name>/limbs.{json,npz}`) |
| medial joint | an articulation point on the scan's medial axis, as the method places it (not the model's pivot) |
| LOSO | leave-one-specimen-out evaluation on the 21 annotated specimens; its models are not shipped |
| the annotation tool | the development tool with which the joints of the 21 annotated specimens were placed and checked; TRUNK reuses some of its geometry rules (not part of this repository) |

## Data files

| file | size (bytes) | md5 | what |
|---|---:|---|---|
| `method/along/along_model.pkl` | 5,415,414 | def7addf14cac057daa056d21ff9d2ea (sha256 2c48fb0e9881f6817bb9cfdcc8e675b303863977e6be13acec7401fee1b27081) | ALONG's model (pickled dict): 11 scikit-learn `HistGradientBoostingClassifier` (one per joint type), a `StandardScaler` + `LogisticRegression` pipeline for the thick-tip start decision, another for the confidence calibration, segment-length priors (log-normal, % BL), start priors, the feature names, the configuration (`cfg`) and the names of the 21 training specimens (`train_names`). Written with scikit-learn 1.7.2 |
| `method/paths_A/priors_final/priors_all_trunk.json` | 4,877 | f65ea74a78cdbb5c10663b1ade51296f | PATHS-A naming priors (feature means / sds per limb slot), fitted on the same 21 specimens |
| `method/paths_C/contact_model.json` | 2,244 | cb1f70044829f93d70cd18c9e0f66bef | PATHS-C logistic contact model (feature standardisation + coefficients), fitted on the same 21 specimens |

All three were fitted on annotated joints of 21 public AntScan worker scans (the names are in `along_model.pkl` under
`train_names`) together with limb centrelines traced through those annotations. The joints were annotated by two
independent automatic annotators (vision-language models), who named them on rendered sheets of the scan's own medial
paths, with an automatic adjudicator for disagreements; selected chains were checked by eye. **The training code and
the annotations are not part of this
repository**, so the files cannot be rebuilt from it. They are shipped because the method cannot run without them and
because they are exactly the files that produced the joint products of the production run (1,013 specimens, 955 of
them registered); a refit would change those products. ALONG's hyper-parameters are stored
in the pickle (`cfg`); the classifiers were `HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, random_state=0)`, trained on the annotated centrelines
plus seven perturbed copies of them (smoothing, lateral offset, arc jitter, tip cut / extension, deeper / shallower
proximal end).

Loading (`method/along/along.py`, `load_model`) refuses the shipped path when the file's sha256 is not the one
recorded in `MODEL_SHA256`, and unpickles every model file with `ModelUnpickler`, which resolves only scikit-learn and
numpy classes (the shipped file asks for nothing else), so a replaced file cannot run code while it is loaded. It
also checks the scikit-learn version against the one recorded in the pickle (`check_sklearn`); another version stops
with a readable error (set `ANTSCAN_JOINTS_SKLEARN_ANY=1` to load anyway; the products may then differ). The command
runs the hash and version checks before it starts any job.

## Dependencies

Listed in `environment.yml` for this package: `scikit-learn==1.7.2` (the classifier), `numba==0.66.*` (the
voxelisers; production ran 0.66.0). `environment-ci.yml` leaves both out: only the slow end-to-end test needs them, and it skips
without them.
Used and normally present: `scipy`, `scikit-image`, `networkx` (installed with scikit-image), `trimesh`, `psutil`
(memory cap), and `embreex` (trimesh's embree ray caster: without it the centreline profiles of the joint placement
differ from the production run, and the command refuses to start unless `--allow-without-embree` is given).
Tested on Linux only. The steps link products with relative symlinks, so an output folder can be moved (Windows
allows symlinks only in developer mode).

## Provenance and differences to the production run

The method files are the code of the production run (2026-11-13, 1,013 specimens) with these changes only:
absolute paths and machine-specific cache constants removed (numba's cache is `--cache`, default `OUT_DIR/cache`);
the folder layout flattened (`method/<module>`), with the import paths adjusted; four duplicate files merged where
they were provably identical (see below); the scikit-learn version check; the round-3 prior and the
leave-one-specimen-out evaluation options are not exposed by the command (their files are not shipped); the
version labels written into the products were renamed (e.g. `S7 ensemble 2026-11-11`; the published products carry
the same labels with underscores), and FOLDS' frame note reads `TRUNK working frame`; the classifier is loaded with
the hash check and the restricted unpickler described above; the links between products are relative instead of
absolute. The command recomputes every state itself; the production run took state s0's four
stage products from the round-6 run, which gives the same joints, confidences and polylines (only the record of the
sign-canonical frame differs in s0's metadata). Comments and docstrings were edited.

Reproduction: the command was run on three production specimens (body length 1.8, 3.2 and 11.2 mm) and every file
the production run holds for them (joint product, limb product, and per state the stage, THIN, assembled and final
products: 159 files per specimen) was compared value for value: all arrays bit-identical, all JSON values equal
except wall-clock fields, install-path strings, the renamed version labels and the s0 frame record named above. Peak
memory per job was the same as in production (largest job 9.0 GB, a PATHS-C stage).

Merged duplicates (identical behaviour shown by code comparison and by the reproduction run):
`ensemble/combine.py` = `assemble/combine.py` (byte-identical); the ensemble's copy of the assemble module, of which
it used only constants and frame helpers identical to `assemble/assemble.py`; a second `geom_profile.py`
(byte-identical); the round-6 `along.py`, which equals the FOLDS copy kept here without its optional `pins` (the
assemble step never passes pins).

The command shares the names file and the lock of the output folder with the registration
(`fitter_3d/antscan_common.py`).

Development options that the command does not expose and that are kept so that the code stays the code that ran:
the `--loso` branches (`assemble.py`, `s7folds.py`, `paths_C/paths.py`; their models are not shipped), the stage
caches of PATHS-A (`stage1_cache`) and PATHS-C (`PATHS_C_REUSE_STAGE`, a pickle the user writes), and the grid phases
other than 0 (`stage.py`). They can be removed in a follow-up together with an equivalence run.

Repeated code that is kept (flagged for review; merging it would change modules that reproduce the published
products, so it was not done here): the exact parity voxeliser `_parity_counts` (`trunk`, `paths_B/vox.py`,
`paths_C/geomlib.py`; FOLDS has its own sub-box variant), `Surface` (signed-distance samples; `trunk`,
`paths_A/pa_geom.py`, `paths_C/geomlib.py`, `along/geom_profile.py`), the polyline helpers `arclen` (8 modules),
`resample` (6), `smooth`, `march_out`, `end_dir`, `dist_poly`, `unit`, the path tools `tips_from`, `mcp_paths`,
`geodesic`, `dense_path`, `dedup` (`paths_A` and `paths_C`), `load_mesh` (4), `load_trunk` (5), `rot_angle`, `omni`
and the limb / joint name tables (`assemble`, `combine`, `folds/s7lib.py`, `thin`, `ensemble`, `__init__.py`), the
thread settings at the top of several modules, and the stand-alone batch `main()` of each method module.
