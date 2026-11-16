# Geometric registration from a joint product inside fitter_3d (`scheme: 'geom'`, `--native_start`)

The registration core of the AntScan pipeline (`fitter_3d/geomreg`, see `docs/antscan_registration_pipeline.md`) can
also run inside `fitter_3d/optimise.py`: as one stage (`scheme: 'geom'`) or as the initialisation of the joint-informed
schedule (`--native_start`). Nothing here changes an existing configuration: every addition is an option that is off
by default, and with the options off every array of the existing stage files is bit-identical on the CPU (checked for
the joint path and the classic path; the joint path's stage files gain two keys,
`target_frame_rotation` and `target_frame_origin`, so that a stage file can be mapped back to the scan frame:
`x_scan = (verts * target_scale + target_centre) @ target_frame_rotation + target_frame_origin`).

Terms: a *joint product* is the per-scan output of the joint method (`<name>_joints.json`: joints on the scan's medial
axis with confidences, the canonical frame, the body length); a *limb product* is its set of limb centrelines
(`<name>/limbs.json` + `limbs.npz`). Both come from the joint method `fitter_3d/antscan_joints`
(`python -m fitter_3d.antscan_joints.run`). `--limbs_dir` takes its `JOINT_DIR/limbs` as it is. `--joints_dir` must
hold the joint products directly as `<name>_joints.json` (`joint_init.prepare_batch` reads only that layout), so link
or copy them out of `JOINT_DIR/proposals/<name>/` first, e.g.
`mkdir -p JOINTS && ln -s "$PWD"/JOINT_DIR/proposals/*/*_joints.json JOINTS/`. The meshes are `<name>.obj` (or
`<name>_processed.obj`) files of the envelopes in the joint product's canonical frame,
`x_canonical = R (x_scan - origin_scan)` with the product's `frame.rotation` and `frame.origin_scan`; no converter for
this ships with the repository. The whole workflow is described in `docs/antscan_workflow.md`.

## What is added
| piece | what |
|---|---|
| `fitter_3d/geomreg/` | the geometric core: `native.py` (closed-form articulated start in the fitter's own parameters), `surface.py` + `tubes.py` + `snap.py` (surface stage), `repair.py` (crossings, fold-overs), `refit.py` (closed-form native refit), `spec.py`, `model.py`, `geom.py` |
| `fitter_3d/geom_stage.py` | `GeomStage(Stage)`: the whole registration as ONE fitter_3d stage |
| `fitter_3d/antscan_geom_cfg.yaml` | a one-stage configuration |
| `optimise.py --joint_kind medial`, `--conf_min X` | the joint term compares virtual markers of the mesh surface with joints given on the scan's medial axis; proposals below the confidence `X` are not used. The default `--joint_kind` is `pivot`; with it, products of the joint method (which are medial) trigger a warning |
| `optimise.py --limbs_dir` | limb product of the joint method (`<name>/limbs.json` + `limbs.npz`) |
| `optimise.py --native_start` | closed-form native start as the INITIALISATION of the existing joint-informed schedule |
| `joint_init.native_start_init` | the function behind `--native_start` |
| `tests/test_fitter_3d_geomreg.py` | 3 tests of the closed-form start (pivot and medial joints) and the native refit (numpy only) |
| `tests/test_fitter_3d_geomreg_pipeline.py` | 1 end-to-end test of the core on a specimen made from the model itself (CPU): native parameters reproduce the mesh, a specimen that cannot be registered is a status |
| `tests/test_fitter_3d_geom_stage.py` | the stage through `optimise.main` as the command line runs it: the schedule ends, and `SMAL3DFitter`'s forward of the written parameters + `deform_verts` gives the written `verts` |

## The stage (`scheme: 'geom'`)
```
python fitter_3d/optimise.py --mesh_dir MESHES --joints_dir JOINTS --joint_kind medial --conf_min 0.3 \
       --limbs_dir LIMBS --yaml_src fitter_3d/antscan_geom_cfg.yaml
```
Meshes are the envelopes in the joint product's canonical frame (as for `antscan_joints_cfg.yaml`). Per specimen:
closed-form start (`geomreg.native.closed_form_start`) -> trunk girth -> limb tube warp -> one affine map per
joint-supported bone (+ the shape space when it explains the scan) -> regularised fine stage -> repair -> native
parameters refitted to the registered mesh; `deform_verts` is what they cannot express. The stage overwrites whatever
earlier stages left. It runs on the CPU by default, with one torch / BLAS thread inside the stage so that two runs give
bit-identical stage files (`geom: {cpu_threads: 0}` lifts the limit and the guarantee; `geom: {device: 'cuda'}` runs on
the GPU, about four times faster and not bit-equal to the CPU result).

yaml keys of the stage: `refit` (default true), `refit_betas` (default false), `geom` (dict of options:
`geomreg.surface.OPTS`; `start: {...}` for `geomreg.native.OPTS`; `device`), `geom_device`. An unknown key, in the
stage or inside `geom`, is an error (a typo is not ignored); the registration command's `--geom-opts` is checked the
same way.

Stage file = the usual `Stage.save_npz` keys, plus
| key | shape | content |
|---|---|---|
| `geom_vertex_status` | (B, V) int8 | 0 observed, 1 hidden (no compatible scan surface found), 2 unsupported (bone without usable joints; imputed), 3 specimen failed |
| `geom_part_status` | (B, J) str | `ok` / `review` / `unregistered` per bone |
| `geom_status` | (B,) str | `ok` / `review` / `failed` (a failed specimen keeps its incoming parameters; no exception leaves the stage) |
| `start_how` | (B, J) str | how the start placed each bone (`chain`, `tip`, `bridge`, `rest`, `mirror`, ...) |
and `<stage>_geom_log.json` (per specimen: seconds, status counts, start info, refit info, or the error). These are
the registration's OWN codes, the same that `registration/<name>.npz` of the pipeline carries. They are not the
status of the pipeline's status step (`status_vertex` in its stage file), which has other codes and is the one to use
for selecting vertices, parts and specimens.

The model must be the OmniAnt model the joint products were made for (`config.set_model(...)` or the `SMAL_FILE` of
`config.py`); `config.py`'s default model is another one.

## The start as an initialisation (`--native_start`)
```
python fitter_3d/optimise.py --mesh_dir MESHES --joints_dir JOINTS --joint_kind medial --conf_min 0.3 \
       --native_start [--limbs_dir LIMBS] --yaml_src fitter_3d/antscan_joints_cfg.yaml
```
`prepare_batch` then initialises `global_rot`, `joint_rot`, `log_scale`, `trans` AND `log_beta_scales` (one isotropic
scale per bone) and `betas_trans` (pivot offsets) from the closed-form start; betas stay 0. Unlike the similarity + IK
initialisation it does not need three root-rigid joints (that initialisation raises `ValueError` with fewer; 9 of 30
thin-limb synthetic specimens): with fewer it uses the trunk chord and the proposals' frame, then the frame and the
body length.

## What the native parameters can and cannot carry (read from `smal_model/batch_lbs.py`)
Bone j maps `x -> M_j (x - J_j) + X_j` with `M_j = (R_0 ... R_j) diag(exp(log_beta_scales_j))` and
`X_j = X_parent + M_parent (J_j - J_parent + betas_trans_j * (1, -1, 1))`. So rotation, a scale along each MODEL axis
and the pivot of every bone are free and independent of the parent; `log_beta_scales[root]` must stay 0 (it enters every
child inverted). An axial stretch along a bone that is not parallel to a model axis is not representable: the start
uses one isotropic scale per bone, the refit three scales per bone, and limb girth (the template's distal limbs are
2-8x thinner than real ones) largely stays in `deform_verts`.

## What changes in existing files
`optimise.py`: the options above and the branch that builds a `GeomStage` for `scheme: 'geom'`. `joint_init.py`:
`joint_kind` / `conf_min` in `load_proposal` and `prepare_batch`, the virtual-marker regressor, `native_start_init`,
the two frame keys. `trainer.py`: `joint_ref_regressor` (the marker joint term) and an early return of
`plot_loss_components` when no stage recorded a loss component (a schedule of closed-form stages only). The geom stage
does not touch `utils.py` (the joint path before it changed `load_meshes` so that it also returns the centre and
scale of each mesh). `rotation_mask` is 1 for every bone after the geom stage (an imputed bone may carry a mirrored rotation,
and the forward multiplies `joint_rot` by the mask); which bones are imputed is in `start_how` / `geom_part_status`.

## A defect of the model file, and how the stage file stays consistent
The left-mandible skinning-weight rows of `OmniAnt_25PCs_joint_limited.pkl` do not sum to 1 (154 rows, 134 of them by
more than 0.001, minimum 0.40). `geomreg` renormalises the rows in its own template (`geomreg/model.py`, `W`; the
stored rows are `W_raw`); `SMAL3DFitter` blends with the rows as stored. The stage therefore takes `deform_verts` as
the residual against the FITTER'S OWN forward of the written parameters, so that a stage file always satisfies
`SMAL3DFitter(parameters) + deform_verts = verts`. On the 154 vertices `deform_verts` then also carries the effect of
the defective rows (hundreds of micrometres up to millimetres on real workers). The Blender add-on copies the rows
unchanged into every model it exports. Fixing the rows in the model file removes the effect; it is not done here.
