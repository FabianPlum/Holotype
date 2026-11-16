# AntScan registration pipeline

Registers the OmniAnt model (10235 vertices, 55 joints) to the outer surface of a micro-CT scan of an ant worker,
starting from the joints that a joint method found on the scan. For every specimen it writes the registered mesh in
fitter_3d's own parameters, a status that says what was measured and what was not, a morphometrics table in
micrometres, and the files the Blender add-on and the neural pipeline read.

It is a separate fitting stream next to the optimisation-based `fitter_3d` schedule and writes the same style of
output (`trainer.Stage.save_npz` keys). The pipeline package imports nothing from `fitter_3d/optimise.py`,
`trainer.py` or `joint_init.py`. (The same registration core can also run as a stage inside `optimise.py`:
`docs/antscan_geom_stage.md`.) The whole workflow, from the downloaded meshes through the joint method to this
pipeline and the Blender add-on, with the numbers of the production run over 955 scans: `docs/antscan_workflow.md`.

## Terms used on this page

| term | meaning |
|---|---|
| envelope | the outer surface of a scan as a triangle mesh, `<name>_envelope.stl`, in the scan frame, micrometres |
| joint method | the program that finds the joints on a scan: `fitter_3d/antscan_joints` (`python -m fitter_3d.antscan_joints.run`, see its `README.md`). The pipeline only reads its files, so joint products from elsewhere work too when they follow the contract below |
| joint product | `<name>_joints.json` of the joint method: `joints_scan` (55 model joints + `head_front`, `gaster_tip`; scan frame, micrometres, or null), `confidence` (0..1), `frame` (`rotation`: rows = canonical axes in scan coordinates; `origin_scan`), `body_length_um`, `completeness` (`missing_joints`, `flags`), `status`. The joints lie on the scan's medial axis ("medial" joints), not on the model's bone heads |
| limb product | `<name>/limbs.json` + `limbs.npz` of the joint method: named limb centrelines (`L1..R3` legs, `AL`/`AR` antennae, `ML`/`MR` mandibles; points tip first, arc length, radius) with a confidence and flags per limb |
| ensemble product | a joint product that the joint method merged from several runs ("states") of itself; it has an `ensemble` block and per-joint `joint_detail.<joint>.ens`. A single-state product has neither. Both kinds are read |
| per-state products | the joint products of the single states: `<states>/<state>/final/proposals/<name>/<name>_joints.json` |
| path stages | work folders of the joint method with candidate limb paths, `paths_A/`, `paths_B/`, `paths_C/` (`<name>/limbs.npz` each); optional evidence for the recovery step |
| BL | body length of the joint product, `body_length_um`; distances "in % BL" are relative to it |
| measurable | a part whose registration the scan does not contradict (status codes 0, 1, 2 below) |
| UC1 | the unposing convention of the bundle for the Blender add-on (section "The unposed bundle") |
| GEOM, RECOVER, STATUS | the registration core (`fitter_3d/geomreg`), the recovery of limb joints and the status part of this pipeline |
| SKEL-REG, COVER, SURFACE, R2-SHAPE | names of the stand-alone modules the code grew from, still used in comments: the earlier skeleton-first registration (scan and tube modules in `parts/skelreg`), the status evidence (`parts/cover04.py`), the earlier surface registration that `fitter_3d/geomreg` grew from, and the shape-model work that defined the morphometric measures and UC1 |
| round 6, round 7 (K4); S6, S7 | generations of the joint method: single-state products and four-state ensemble products; S6 / S7 in version labels mean the same |
| round 2, round 3 | the earlier and the current generation of the registration modules |
| the benchmark | the project's registration benchmark that compared registration methods during development (not part of this repository); `routes/` keeps its input layout |
| the annotation tool | the development tool with which the 21 joint annotations of the development set were made and checked (not part of this repository) |

## Requirements

The conda environment of the repository plus two packages (both in `environment.yml` / `environment-ci.yml`): `embreex` (ray casting for trimesh; without it the command stops at once with a message) and `threadpoolctl`
(one BLAS thread inside a registration; without it the command prints a note and two CPU runs may differ in the last
bits). Linux is what the pipeline was run on; nothing in it needs symbolic links or a GPU.

Envelope, joint product and limb product of a specimen must be in the SAME scan frame and in micrometres. The
envelopes come from `custom_processing/antscan_pipeline.py` (`custom_processing/README_antscan2024.md`), the joint and
limb products from `python -m fitter_3d.antscan_joints.run`, whose output folder this command reads as
`--joints JOINT_DIR/proposals --limbs JOINT_DIR/limbs` (the per-state products and the path stages are then found
next to them).

## The one command

```bash
# from the repository root, conda env pytorch3d
python -m fitter_3d.antscan_registration.run \
    --names NAMES.txt --joints JOINT_DIR --limbs LIMB_DIR --env ENV_DIR --out OUT_DIR
```

| argument | what |
|---|---|
| `--names` | text file, one specimen name per line (a file key only; blank lines and `#` lines are skipped) |
| `--joints` | `<name>/<name>_joints.json` (or `<name>_joints.json`): joint products |
| `--limbs` | `<name>/limbs.json` + `limbs.npz`: limb products; `none` = no limb product (no tube warp, no recovery, no surface verification) |
| `--env` | folder with `<name>/<name>_envelope.stl` (or `<name>_envelope.stl`), or a path template containing `{n}` |
| `--out` | output folder. One run at a time writes to it (lock file `OUT_DIR/.lock`; `--break-lock` takes over the lock of a run that is gone) |
| `--joint-states` | ensemble products only: the folder with the per-state products. Default: a folder `states` next to `--joints` or up to two levels above it. It is used only when it reproduces the product (at least 95 % of the product's joint positions are exactly one state's position). Without it the trunk rule and the far-tie rule of the intake cannot be applied; the run prints one line saying so and the joint product in `joints/` records it |
| `--stages` | folders with the path stages (optional evidence of the recovery step); default: found next to `--limbs`; `none` = off |
| `--device` | `cpu` (default; two runs give bit-identical results) or `cuda[:N]` (about five times faster for the registrations, not bit-identical) |
| `--jobs` | worker processes: 1 to 3 on the CPU, 1 to 2 on a GPU (higher values are cut to that) |
| `--model` | model `.pkl` (default `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`) |
| `--cache` | folder for a small cache file (the template's outer surface, computed once in about half a minute; its name carries a hash of the model's rest mesh, so another `--model` never reuses it); default `OUT_DIR/work/cache` |
| `--steps` | any of `intake,reg1,recover,reg2,status` (default: all). The status step always reads the registration that exists: the second one where the recovery changed the joints and it was run, else the first |
| `--force` | recompute every step asked for |
| `--no-recover` | no recovery step and no second registration |
| `--no-mandible-check` | mandible bases below the cut are not checked (they stay unregistered) |
| `--recover-disable LIST` | switch off kinds of recovery: `lowconf` (the product's own joint below the cut, accepted where independent evidence confirms it), `fallback`, `chainfit` (a chain fit on the limb centreline), `tail` (candidate paths of the path stages) |
| `--confidence RULE` | ensemble products: `majority-trunk` (default), `majority-all`, `as-given` (see "intake") |
| `--conf-cut auto\|X` | confidence cut of the intake, a number in (0, 1] (default `auto` = 0.30) |
| `--ties far\|use\|recover` | what to do with joints the ensemble placed on a tie (default `far`, see "intake") |
| `--kind` | what the input joints are: `medial` (default) or `pivot` (model bone heads, e.g. oracle joints of synthetic data; the registration only, see the limits) |
| `--priors FILE`, `--population FILE\|none` | the two shipped data files (segment-length prior of the recovery step; population reference of the unposed mesh) can be replaced; `--population none` fills from the batch itself |
| `--geom-opts JSON` | options of the registration (`fitter_3d/geomreg/surface.py` `OPTS`, and `start` for `fitter_3d/geomreg/native.py` `OPTS`). Unknown keys are an error |
| `--K JSON` | override constants of the status and of the model grade. Unknown keys are an error. Status: `LIMB_TRUST`, `ON_MIN`, `OTHER_MAX`, `SEVERE_OWN`, `JTOL_BL`, `TIP_TOL_BL`, `ASYM_MAX`, `ASYM_LOOSE`, `Q_GAP`, `TRUNK_DSCAN_BL`, `TRUNK_UNCOVERED`, `TRUNK_COVER_TOL_BL`, `MANDIBLE_RULE`, `POP_Z`, `AGREE_BL`, `HEAD_CHECK`, `KINK_DEG`, `RMIN_REL`. Grade: `MAX_SEG_FILLED`, `ZIGZAG_DEG`, `ROLL_GATE`, `ROLL_DEG`, `ROLL_CONTRAST`, `ROOT_FALLBACK_IS_DNU`. Examples: `'{"TRUNK_UNCOVERED": null}'` (coverage rule off), `'{"ZIGZAG_DEG": null}'` (antenna condition off), `'{"ROLL_GATE": true}'` (head roll gates) |
| `--no-overlay`, `--no-export` | leave out the overlays / the export files |

**Exit code**: 0 the batch ran; 2 wrong arguments, a missing package, or `OUT_DIR` locked by another run (nothing was
done); 3 the batch ran and no specimen got through (every row carries an error).

**Failures**: a specimen that fails in any step keeps a status and an error message in `summary.csv` and
`manifest.jsonl`; the batch goes on. A worker process that dies (for example killed by a memory limit) is replaced,
and a specimen that takes its worker down twice becomes a row with an error. Run-time code makes no network calls.

**Processes**: every per-specimen step runs in worker processes of its own (also with `--jobs 1`), so the files of a
specimen do not depend on `--jobs` or on the other specimens of the batch. When the command is called from another
script (`fitter_3d.antscan_registration.run.main([...])`), that script needs the usual
`if __name__ == "__main__":` guard, because the worker processes import it again.

**Resume**: running the same command again recomputes nothing. A step's result is reused only when it was made from
the same input files (by content), options, model file and pipeline version; a signature is stored next to each
result (`*.sig`). If an input or an option changed, the step and everything after it is recomputed for that specimen.

**Environment**: the package sets nothing on import. The command sets defaults for the thread counts (two per
process: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) and
`CUBLAS_WORKSPACE_CONFIG`; values already in the environment are kept. No other environment variable is read. If a
torch CPU kernel crashes on a machine (seen with AVX-512), set `ATEN_CPU_CAPABILITY=avx2`.

## What it does

| step | what | code |
|---|---|---|
| intake | Decides which joints are usable: confidence at least 0.30. For ensemble products two rules need the per-state products. (1) The confidence of the TRUNK joints (neck, waist, gaster, head front) is recomputed with the distance within which a majority of the states lie, instead of the largest distance: one deviating state otherwise removes a joint the others agree on, and no later step can bring a trunk joint back. (2) A limb joint at or above the cut is used directly only when a majority of the states that placed it lie within 4 % BL of it; otherwise two groups of states put it in two places and noise chose one, and it is left to the recovery step (`intake.tie_far`). Limb joints below the cut are left to the recovery step. Positions are never changed | `intake.py` |
| first registration | closed-form start in fitter_3d's parameters from the usable joints, then a surface stage (trunk girth, limb tubes on the limb centrelines, one affine map per bone, a regularised fine stage), a repair pass and a closed-form refit of the native parameters | `fitter_3d/geomreg` |
| recovery | limb joints the joint method lacked or held below the cut are searched again with what the model knows (segment lengths of the partner limb or of a population prior, sockets predicted by the first registration, a chain fit on the centreline) and accepted only when evidence that did not produce the candidate confirms them. A mandible base below the cut is accepted at its own position when it lies where the first registration or the mirror image of the partner base puts the socket ("socket check"). A joint that was usable is never moved | `parts/limbrec`, `mandible.py` |
| second registration | only for specimens whose usable joints changed; it registers the whole specimen again from the augmented joints (parts with unchanged joints move by a median of 0.15 % BL) | `fitter_3d/geomreg` |
| status and export | status per vertex, bone, limb segment, chain and specimen; specimen class with a model grade; morphometrics; stage file; unposed bundle with an attached fill; overlay; summary table | `parts/rs`, `status_run.py`, `grade.py`, `fill.py`, `checks.py` |

## Output layout (`OUT_DIR`)

| file | content |
|---|---|
| `summary.csv` | THE table, one row per name (columns below) |
| `registration/<name>.npz`, `.json` | the final registration: `V_final` (10235 x 3, scan frame, micrometres); its native parameters `global_rot`, `joint_rot`, `betas`, `log_beta_scales`, `trans`, `betas_trans`, `log_scale`, `rotation_mask`, `deform_verts`, `verts`, `pivots` in the normalised target frame (`target_scale`, `target_centre`, `target_frame_rotation`, `target_frame_origin`); the same of the closed-form start (`start_*`); `um_per_model_unit`; `registration_source` (`reg1` or `reg2`); and the registration's OWN codes `geom_vertex_status` (0 observed, 1 hidden: no compatible scan surface found, 2 unsupported: bone without usable joints, placed by the template), `geom_part_status`, `geom_status`. These are not the status of the status step; use `status/` to select vertices and parts. Further keys are the registration's own records: the mesh after its intermediate stages (`V_S` start, `V_B` trunk girth, `V_T` limb tubes, `V_C` per-bone maps; `V_pre` = `V_S`), `reg_v`, `scan_code` (a part label per scan vertex), `support` and `start_how` / `start_pinned` / `start_sig` / `start_psi` (how the start placed each bone), `diag_*` (per-bone diagnostics), `body_length_um`, `product_flags`, `status_reasons`. `.json`: the log |
| `joints/<name>/<name>_joints.json` | the joint product the final registration used: the input + `intake` (rule, joints it touched, ties) + `confidence_input` + `recovery` (recovered joints with their evidence) |
| `status/<name>_status.json`, `.npz` | status at every level with reasons: `segments` {name: [code, reasons]}, `part_code` (55), `part_status`, `part_reasons`, `vertex_status` (10235), `chains`, `specimen` (class, reasons, `grade` with every condition, `head_end`), `trunk_cover`, `fill`, `checks`, `registration`. `pop_z` is a robust z of each segment length against the other specimens of the batch: information that depends on the batch and enters no decision |
| `routes/<name>.npz` | the final registration with the status step's result in the layout of the project's registration benchmark (not part of this repository): `V_final`, `reg_v` (vertex registered), `vertex_status` (codes below), `part_status` (`ok` / `review` / `unregistered` per bone), `status` (`ok` = model_grade, `review` = partial, `failed` = do_not_use), `um_per_model_unit` |
| `morphometrics.csv`, `morphometrics_flags.csv`, `morphometrics_measurable.csv` | one row per specimen (section "Morphometrics") |
| `export/batch_0/Stage_status.npz`, `export/stage_rows/<name>.npz` | fitter_3d stage file (section "The stage file") |
| `export/uc1_bundle.npz` (+ `.json`), `export/uc1_bundle_model_grade.npz` | unposed rest meshes for the Blender add-on (section "The unposed bundle"); the second file holds the model-grade specimens only |
| `overlays/<name>.png` | registered mesh coloured by status above the scan with centrelines and joint markers |
| `evidence/` | cache of the status step's evidence per specimen |
| `work/` | `intake/` (intake products), `reg1/` and `reg2/` (first and second registration as the registration core writes them, with plain key names `vertex_status`, `part_status`, `status` for its own codes), `recover/` (augmented products, logs), `cache/` |
| `run.json`, `summary.json`, `manifest.jsonl` | every invocation with its options and seconds; class counts and constants; per-specimen rows of every step, kept across invocations |

Paths inside the output files are absolute: a moved output folder can be read, but resuming it recomputes the status
evidence.

### Columns of `summary.csv`

| column | meaning |
|---|---|
| `specimen_class`, `class_reasons` | `model_grade` / `partial` / `do_not_use` and why (section "Status") |
| `status_class` | the status part's own class before the grade: `complete` (every trunk part and all 34 limb segments measurable), `partial`, `do_not_use` |
| `registration_status` | the registration's own status: `ok`, `review`, `failed` |
| `n_measurable`, `n_review`, `n_unregistered` | of the 34 leg and antenna segments |
| `legs_measurable`, `antennae_measurable`, `mandibles_measurable` | whole chains (6, 2) and mandibles (2) that are measurable |
| `trunk_not_measurable` | trunk bones that are not measurable (`b_h` head, `b_t` mesosoma, `b_a_1`, `b_a_2` waist, `b_a_3..5` gaster) |
| `uncovered_mesosoma`, `uncovered_gaster` | share of the scan surface nearest to the part that the mesh does not reach (coverage rule) |
| `n_segments_filled` | limb segments filled in the unposed bundle (not measurable, or filled by the length gate) |
| `head_end` | `ok`; `doubtful` (the joint product doubts which end is the head and the scan does not confirm it: `do_not_use`); `doubtful_confirmed_by_scan` (both antennae are measurable on the scan at the head end) |
| `product_type`, `conf_cut`, `n_raised_by_majority`, `n_ties`, `n_far_ties_demoted` | intake: `single` or `ensemble<N>`, the cut, trunk joints raised, ties recorded, limb joints left to the recovery step by the far-tie rule |
| `n_lost_joints`, `n_recovered`, `n_mandible_recovered` | limb joints below the cut or absent, of them recovered; mandible bases accepted |
| `second_registration`, `registration_source` | whether the final registration is the second one |
| `lbs_energy_share`, `deform_med_pct_bl`, `deform_p90_pct_bl` | how much the native parameters carry (share of the squared displacement from the placed template) and the length of the registration's `deform_verts`, % BL |
| `native_forward_err_um` | the registration's own check: its forward of the parameters + its `deform_verts` against the mesh |
| `stage_weight_rows_um` | largest part of the stage file's `deform_verts` that is owed to the model file's weight rows (section "The stage file") |
| `n_filled_vertices`, `n_trunk_filled`, `n_abnormal_edges`, `n_stretched_edges`, `seam_ratio_max` | unposed bundle: filled vertices, of them on the trunk, edges above 8 x the population median AND 10 % of the template body length, edges above 8 x and 1.5 %, largest seam edge relative to its reference |
| `antenna_zigzag_max_deg` | largest zigzag of a measurable scape or funiculus |
| `head_roll_best_deg`, `head_roll_contrast` | head-roll measurement (reported; does not gate by default) |
| `body_length_um` | BL of the joint product |
| `seconds_reg1`, `seconds_recover`, `seconds_reg2`, `seconds_status` | per step |
| `error` | error messages of any step |

## Status

Codes of the status step (vertex = code of its dominant bone; bone; limb segment). They are used in `status/`,
`routes/`, `status_vertex` of the stage file and `vertex_status` of the bundle:

| code | name | measurable | meaning |
|---|---|---|---|
| 0 | joint_supported | yes | both joints of the segment are usable and the scan does not contradict them. Trunk part: its joint is usable, the part lies on the scan, and (mesosoma, gaster) the mesh reaches its scan surface |
| 1 | surface_verified | yes | no joint support, but a traced scan tube on its own named centreline with observed articulations at both ends |
| 2 | recovered | yes | joint-supported with a recovered joint that was confirmed by a scan feature at the joint. Mandible: the base has two kinds of socket evidence (the first registration, and the mirror image of the partner base; the partner may itself be a base below the cut) |
| 7 | recovered_estimated | no (flagged) | as 2, but the recovered joint is pinned by lengths or one socket prediction only |
| 3 | review | no (flagged) | registered, but evidence contradicts it |
| 4 | unregistered | no | no support; the part is placed by the template's rest articulation or its mirror partner |
| 5, 6 | imputed_mirror, imputed_prior | no | unposed bundle only: filled from the specimen's mirror part / from the population mean |

**Coverage rule for mesosoma and gaster.** A trunk part can lie on the scan and still not reach all of the scan's
part (a truncated or wedge-shaped gaster, a mesosoma whose dorsum is not reached). The scan surface is sampled at its
face centroids; each sample goes to the nearest registered vertex and to that vertex's part; `uncovered` is the area
share of a part's samples farther than 1.5 % BL from that vertex. Above 0.20 the part is `review`: not measurable, so the
specimen is not model grade, the part's measures are flagged and the part is filled in the unposed bundle; the rule
alone never makes a specimen `do_not_use`. Cut and tolerance were set by eye on production overlays (see
"Evidence"). The rule sees a part that does not reach its scan surface; it does not see a part that lies on the scan
but is slid along it. Surface of limbs that were left unregistered counts towards the mesosoma or the gaster, so on a
specimen with several unregistered legs a trunk part can be flagged for its legs (such a specimen is `partial`
anyway); `trunk_cover.<part>.uncovered_off_limbs` in the status file is the value without the surface inside the
scan's limb tubes, for information. The head and waist values are reported only (`trunk_cover` in the
status file): surface of unregistered mandibles and antennae counts towards the head, and the template's waist does
not reach high nodes.

Specimen class (column `specimen_class`), for selecting specimens without looking at them:

- `do_not_use`: the registration failed or its root was placed by a fallback (fewer than three usable joints of the
  mesosoma); the joint product's `status` field is not `ok`; the head end is doubtful and the scan does not confirm
  it; the mesosoma is not registered on the scan (no usable joint, or off the scan); a processing error. All measures
  of such a specimen carry flag 3.
- `model_grade`: fit for a shape model. ALL of:
  - M1 every trunk part measurable;
  - M2 at most 5 of the 34 leg and antenna segments filled in the unposed bundle (not measurable, or measurable but
    filled by the bundle's length gate);
  - M3 no trunk vertex filled in the unposed mesh;
  - M4 no abnormal stretched edge in the unposed mesh (longer than 8 x the population median of the same edge AND
    longer than 10 % of the template body length);
  - M5 no crumpled registered antenna (zigzag of a measurable scape or funiculus at most 60 degrees: total turning of
    its medial curve minus its net bend);
  - M7 left / right naming not doubted by the joint product's frame check;
  - M8 the registration is usable (not failed; its native parameters reproduce the mesh within 1 micrometre).
  A condition that could not be evaluated (its check failed, the bundle could not be built) counts as NOT met.
  Mandibles are not part of the grade.
- `partial`: everything else, including a specimen whose head is not measurable. Use measures by their flag (or
  `morphometrics_measurable.csv`); not for a shape model.

M6, a head-roll check (is the scan's head more mirror-symmetric about a plane rolled about the head's long axis?), is
measured and reported and does not gate: on real scans none of its flags was confirmed by eye. `--K
'{"ROLL_GATE": true}'` makes it gate.

"Measurable" means "the scan does not contradict the joint product", not "verified correct": a joint that is
confidently placed on a real articulation at the wrong position is reproduced, also in a model-grade specimen.

## Morphometrics

One row per specimen that reached the status step; lengths in micrometres, areas in square and volumes in cubic
micrometres. `morphometrics.csv` holds every value that could be computed, ALSO flagged ones and those of
`do_not_use` specimens; `morphometrics_flags.csv` holds per value 0 valid, 1 flagged (a required part is in review,
or less than half of a surface measure's vertices are observed on the scan), 2 not available, 3 specimen
`do_not_use`; `morphometrics_measurable.csv` holds only values with flag 0. Use the last one unless you read the flags.

| columns | definition |
|---|---|
| `leg{1,2,3}{l,r}_{coxa,femur,tibia,basitarsus,tarsus_rest}_len` | chord between the virtual markers of the segment's two joints on the posed mesh (marker = centroid of the vertex ring at the articulation; leg tip = tip vertex). `femur` = trochanter + femur. `tarsus_len` = basitarsus + tarsus_rest |
| `scape_{l,r}_len`, `funiculus_{l,r}_len` | chord antennal socket to scape tip; scape tip to antenna tip |
| `<segment>_rad`, `<segment>_xsec_um2` | median distance of the segment's own vertices from its marker axis (middle 60 %); area of the section at mid-length |
| `trunk_neck_to_petiole`, `waist1_length`, `waist2_length`, `waist_length`, `gaster_length` | chords between the trunk markers (neck, the two waist joints, gaster base, gaster apex) |
| `body_length_chords` | head front to neck, then the trunk chords to the gaster apex |
| `head_width`, `head_length`, `head_height`; `mesosoma_length`, `pronotum_width`, `mesosoma_height` | extents of the part's vertices along its own axes on the posed mesh |
| `waist{1,2}_width/_height`, `petiole_*`, `postpetiole_*`, `gaster_width`, `gaster_height` | extents on the unposed rest mesh times `um_per_model_unit` (these parts span several bones) |
| `{head,mesosoma,waist,gaster}_volume_um3`, `_area_um2` | volume and area of the part's faces on the unposed rest mesh |
| `lr_<measure>_reldiff`, `lr_len_absdiff_median`, `lr_len_pairs` | left / right agreement: 2 (L - R) / (L + R) per paired measure; median absolute value over the limb lengths with both sides valid |
| `um_per_model_unit` | converts a model-unit length of the unposed bundle into micrometres |

Coxa length is not ready to report (see "Evidence"); no mandible measure is in the table.

## The stage file

`export/batch_0/Stage_status.npz` has every key of `trainer.Stage.save_npz`: `global_rot`, `joint_rot`, `betas`,
`log_beta_scales`, `trans`, `deform_verts`, `betas_trans`, `log_scale`, `rotation_mask`, `verts`, `pivots`, `faces`,
`labels`, `target_centre`, `target_scale`, `target_units`, `target_frame_rotation`, `target_frame_origin`. The mesh in
the scan frame is

    x_scan = (verts * target_scale + target_centre) @ target_frame_rotation + target_frame_origin

It also has `status_vertex` (B, 10235; codes above), `status_part_code` (B, 55), `status_part_status`,
`status_specimen_class`, `status_names`, `status_params_source`, `status_version`, `status_provenance`.

The native parameters are the registration's own. **`deform_verts` is the residual against fitter_3d's own forward**,
so that `SMAL3DFitter(parameters) + deform_verts = verts` holds for the file as it is (checked in the tests with the
fitter itself). This matters because of a defect of the model file: 154 skinning-weight rows of
`OmniAnt_25PCs_joint_limited.pkl` (left mandible) do not sum to 1 (134 by more than 0.001, minimum 0.40).
`SMAL3DFitter` blends with the rows as stored; the registration core and the unposing renormalise them. On those 154
vertices the stage file's `deform_verts` therefore also carries the effect of the defective rows
(`stage_weight_rows_um` in the summary: per specimen a median of 0.7 mm and up to 3.9 mm on the 103 production scans). The add-on copies the rows unchanged into every model it
exports. Fixing the rows in the model file removes the effect.

## The unposed bundle

One unposing convention for every registration (UC1: bone rotations from the registered vertices, inverse linear
blend skinning with renormalised weights, per-bone lengths and girths stay in the rest mesh, size removed by trunk
centroid size; model units). The add-on ignores a validity mask, so every vertex that is not measurable is filled
before the bundle is written: bone by bone, parents first, from the specimen's own mirror part when that is
measurable (code 5), else from the mean part of a population reference (code 6); the filled part is attached to the
specimen's own parent at the seam ring, and a part between two registered neighbours is stretched to meet both
(factor capped to [1/2, 2]). A limb bone whose length is outside [1/3, 3] of the population mean is filled with
everything below it ("length gate"). A registered vertex that sticks out of its ring of neighbours by more than 1 %
of the body length is put back on the ring.

**Every row of the bundle depends on its own specimen only**: the same specimen gets the same row in any batch. The
population reference (`data/rest_population.json`: mean rest position per vertex and median length per edge over the
measurable vertices of 78 production scans; no names, no truth) is the fill source, the reference of the length
gate and the reference of the stretched-edge condition. It was built with this pipeline from scans that are not in
the repository; `--population FILE` replaces it and `--population none` fills from the batch itself (then the rows
depend on the batch). `python -m fitter_3d.antscan_registration.build_data population --run OUT_DIR --names NAMES.txt
--out FILE` builds it from a finished run (the shipped file is reproduced byte for byte from its run).

Keys: `verts` (N, 10235, 3) float32, `labels`, `specimen_class`, `vertex_status` (status codes; 5 / 6 for filled
vertices), `valid` (kept as registered), `measurable`, `filled`, `gated` (filled by the length gate), `stuck` (put
back on its ring), `fallback` (unposed from a single bone), `in_mean`, `um_per_model_unit`, `roundtrip_max_um`,
`convention`, `fill`, `version`. Import in the add-on with "Auto Clean-up Mesh" OFF.

## Evidence

The numbers in this section come from the development of the pipeline. The production run over 955 scans with the
frozen pipeline (classes, measurable segments, thickness, self-intersections, run time) is summarised in
`docs/antscan_workflow.md`.

Terms: the *development set* is 21 scans with annotated joints. Two independent automatic annotators
(vision-language models) named the joints on rendered sheets of the scan's own medial paths, and an automatic
adjudicator settled disagreements. A check by eye of selected chains found every chain right on which both annotators
agreed, and errors in some adjudicated or disputed chains (mostly joints left out although they exist). The set was
used many times while the method was built, so every number on it is optimistic. The *production scans*
are the 103 scans of the development phase, 82 of them without annotation. Synthetic
sets are posed model meshes with exact truth: *in-model* / *out-of-model* sets with joints given exactly, a
*thin-limb* and a *scan-like* set whose joints come from the joint method itself. Two generations of the joint method
were used: single-state products ("round 6") and four-state ensemble products ("round 7 K4", the current one).

**Registration against annotated joints** (development set, 21 scans, ensemble products; shared registration
benchmark; median over specimens; "better / worse" counts specimens, paired). The pipeline against its own first
registration alone:

| measure | first registration alone | pipeline | better / worse |
|---|---|---|---|
| registered share of the mesh | 0.920 | 0.975 | 19 / 0 |
| part accuracy over the whole mesh | 0.914 | 0.934 | 16 / 0 |
| part accuracy among registered vertices | 0.955 | 0.952 | 0 / 6 |
| registered vertices on a wrong limb chain | 2.3 % | 2.6 % | 1 / 13 |
| scan to mesh, 90th percentile | 0.71 % BL | 0.51 % BL | 13 / 0 |
| flipped faces | 1.29 % | 1.29 % | 1 / 3 |
| crossing triangle pairs between parts | 1 | 4 | 0 / 10 |
| leg chains right / wrong and flagged / wrong and unflagged / unregistered (of 102) | 99 / 0 / 0 / 3 | 102 / 0 / 0 / 0 | |

More of the mesh is registered and more of it is on the right part; what is registered through recovered joints is
slightly less clean. One mandible chain lies on a neighbouring part without a flag.

**The recovery step** (same scans): of 54 limb joints with truth that the joint method lacked or held below the cut,
31 were recovered within 2 % BL, 5 within 2 to 4 %, 2 wrong (5.1 and 6.9 % BL), 16 left alone. Measurable limb
segments with truth go from 518 (5 more than 4 % BL off) without the step to 542 (7 off) with it; no segment that was
right and measurable is lost. Of 9 mandible bases below the cut, 8 were accepted (5 within 2 %, 3 within 2 to 4 %).

**Morphometrics against annotated joints** (measures called valid; median / 90th percentile of the relative error,
number of measures): femur 0.9 / 2.8 % (104), tibia 2.4 / 6.6 % (89), basitarsus 3.0 / 7.5 % (86), rest of the tarsus
2.1 / 6.3 % (84), scape 1.6 / 3.7 % (33), funiculus 1.7 / 3.5 % (35), neck to petiole 1.3 / 2.4 % (19), waist
2.0 / 6.9 % (18), gaster 1.0 / 3.3 % (16), body length 1.0 / 2.7 % (16). Coxa: 5.9 / 17.6 % (94), not ready to
report. Flagged values are far worse on distal segments (basitarsus: median 40 %, 3 values).

**Classes** (model grade / partial / do not use):

| set | single-state products ("round 6") | four-state ensemble products ("round 7 K4") |
|---|---|---|
| development, 21 | 12 / 9 / 0 | 9 / 12 / 0 |
| production, 103 | 50 / 51 / 2 | 41 / 58 / 4 |
| of them the 82 without annotation | 38 / 42 / 2 | 33 / 45 / 4 |

With ensemble products fewer specimens are model grade: their joint confidences are lower and more limb joints go
through the recovery step. Without the recovery step the 103 would be 30 / 68 / 5 (ensemble products). In a look at 30 of the 82
(drawn with a seed) the class fitted what was seen on 29; the one that did not was a `do_not_use` by the head-end
doubt whose registration looked usable.

**The trunk coverage rule.** A visual review of 46 model-grade production specimens (before the rule existed)
rejected 6 for a trunk part that did not reach its scan surface; the rule, built on that review's own screen and
cut, moves exactly these 6 (and no other of the 46) out of the model grade. Against exact truth (four synthetic sets,
296 trunk parts that the status called measurable): 19 flagged, 17 of them wrong by more than 2 % BL (15 by more
than 4 %); 277 not flagged, 68 of them wrong by more than 2 % BL in vertex terms (25 by more than 4 %): these lie on
the scan and are slid along it. Against the annotated joints of the development set the rule flags 3 more trunk
parts (ensemble products; 4 with single-state products), of which 1 (2) was wrong (its joint marker more than 4 % BL
off) and 2 (2) were right by that measure; 3 of 92 (4 of 93) trunk parts called measurable are still wrong.

**Synthetic scans with exact truth** (vertex error of the visible vertices, median over specimens; earlier skeleton
registration -> this registration): out-of-model shapes with exact joints 0.59 -> 0.30 % BL (40 specimens, a set
held out until the end and run once), in-model shapes 0.84 -> 0.20 % BL (40). When the joints come from the joint
method itself the errors of the joints go through: on a scan-like set (30) 12 % of the registered vertices of the
specimens that are not `do_not_use` are more than 4 % BL off, on a thin-limb set (30) 24 %; there the pipeline
registers more of the mesh than its first registration alone (0.76 -> 0.86 on the same 18 specimens) at a slightly
higher error (share above 4 % BL 0.18 -> 0.22, median over specimens; higher on 9, lower on 9), and the class
removes the worst 12 specimens.

**Native parameters.** They carry a median of 99.8 % of the squared displacement from the placed template (minimum
97.8 % over the 103 production scans); the registration's `deform_verts` has a median length of 0.6 % BL.

**Left / right agreement** of measurable pairs on the 103 production scans (ensemble products): femur median
difference 1.9 % (242 pairs; 2.5 % of the pairs differ by more than 10 %), tibia 1.6 % (209), basitarsus 1.2 % (196),
scape 4.2 % (59), funiculus 2.8 % (72); coxa 4.0 % (187; 23 % of the pairs above 10 %).

**The unposed bundle.** Attached fill against the earlier fill (mirror vertex or mean position in the model frame)
on the same registrations and statuses, 103 production scans (ensemble products; 93 with a fill): seam edges
relative to their reference length, median 2.00 -> 0.98; seam edges above 3 x: 5692 -> 1386 of about 17 500;
abnormal stretched edges (8 x the reference and 1.5 % BL) 1161 in 62 specimens -> 144 in 20; the largest seam ratio
is lower on 90 specimens and higher on 3.

**Reproducibility and consumers.** A fresh run of the command on two development and two production specimens per
generation of joint products gives bit-identical files to the batch runs reported here (other `--jobs`, other
batch); the status step on 20 of the 103 specimens gives the same per-specimen files and the same rows of the stage
file and of the bundle as in the full batch. `SMAL3DFitter`'s own forward of every row of the four stage files
(21 + 21 + 103 + 103 specimens) plus `deform_verts` reproduces `verts` within 0.01 micrometres. The Blender 4.2
add-on (Auto Clean-up Mesh off) imports the unposed bundle and the stage file, generates shape keys and exports a
model that passes the fitter's forward and the neural loader's checks (6 specimens each).

## What it cannot do

- **Wrong joints are reproduced.** A joint the joint method places confidently on a real articulation at the wrong
  position is registered and called measurable, also in a model-grade specimen. On the 21 development specimens 7 of
  542 measurable limb segments are more than 4 % BL off (ensemble products; two coxa bases among them); 3 of the 7
  are in one model-grade specimen. A coxa that the joints put on the pronotum is registered there.
- **The class grades what the scan can contradict, not correctness.** `model_grade` says: trunk on the scan and
  covering it, at most one leg's worth of segments filled, no stretched edge, no clearly crumpled antenna. It does not
  say that every joint is right, and it does not see a trunk part that lies on the scan but is slid along it.
- **`partial` is a wide class.** It reaches from a specimen with six filled leg segments to one of which only the
  mesosoma and the legs are registered (head, waist and gaster unregistered and filled from the population mean in
  the unposed bundle). A mesosoma that is registered but flagged by the coverage rule makes `partial`, not
  `do_not_use`. On the 103 production scans (ensemble products) one `partial` specimen has 1 of 34 limb segments and
  no trunk part but the waist measurable (13 values of its row in the morphometric table are valid), and three more
  have fewer than 12 measurable limb segments. Read `n_measurable`, `trunk_not_measurable` and `n_segments_filled`
  before using a `partial` specimen, or take the measures by their flags.
- **Head-end doubt.** A specimen whose joint product doubts which end is the head is `do_not_use` unless both
  antennae are measurable at the head end. On the production scans this rejects specimens whose head end is right
  (2 of 82 with ensemble products). No specimen with a wrong head end was available to test a looser rule.
- **Trunk joints.** No recovery exists for them. For ensemble products the neck or the first waist joint is below
  the cut on 2 of 103 production scans with the per-state products and on 4 of 103 without them; such a specimen has
  no registered head or waist and is `partial` (head) or `do_not_use` (mesosoma). With ensemble products more
  specimens lose trunk parts than with single-state products (see "Evidence").
- **Mandibles.** One joint and the tip of a centreline; long, closed or unusual mandibles are outside the template.
  A registered mandible can be a poor shape even when its base is right. No mandible measure is in the table and
  mandibles are not part of the model grade.
- **Antennae.** Almost no antennal joint is recovered (there is no usable prediction of the socket). Registered
  antennae are coarse, faceted tubes; clubs are not reproduced. The model grade catches clearly crumpled ones (zigzag
  above 60 degrees) and not the mild ones: between 43 and 50 degrees readers saw both degraded scapes and scapes that
  follow the scan. The check measures the mesh, not an error: a funiculus that really is sharply kinked reads the
  same (on an in-model synthetic set whose generator kinks the funiculus, 18 of 48 correctly registered specimens are
  `partial` for it).
- **Visible limbs left unregistered.** In a look at 40 production scans 17 had at least one clearly visible limb left
  unregistered (flagged, filled in the bundle): the joint method did not deliver its joints and the recovery step
  could not confirm them.
- **Template limits.** Waist nodes, spines, shields, clubs and curled gasters are not reproduced; petiole measures
  are unreliable; coxa length is not ready to report (median error 6 %, 90th percentile 17 % against annotation: its
  base convention differs).
- **Unregistered parts in the posed mesh** (code 4) are template parts in rest or mirrored articulation: they do not
  lie on the scan and can cross other parts. They are flagged, excluded from every measure and replaced in the
  unposed bundle, but not removed from the posed mesh of the stage file.
- **Filled parts in the unposed bundle** are the specimen's own mirror part or the population's mean part attached
  at the seam. A mean-filled tarsus is a thin line (the mean of differently curled tarsi), a mirror-filled part can
  cross the body, and stretched fans remain at some coxal sockets. Use `uc1_bundle_model_grade.npz` for a shape
  model; it still holds up to five filled limb segments per specimen.
- **Crossings.** Recovered limbs are registered less cleanly than directly supported ones: among the registered
  parts of the 103 production scans the median number of crossing triangle pairs between parts rises from 4 (first
  registration alone) to 23, and the share of registered vertices on a wrong chain rises on the development set from
  2.3 to 2.6 %.
- **Left and right.** Only the joint product's frame check guards against exchanged sides (condition M7). The
  head-roll measurement is reported and does not gate.
- **Joint kind.** The status compares the registered mesh's medial markers with the joints, which is right for the
  joint method's products. With `--kind pivot` only the registration follows the kind: the status then under-reports
  joint support.
- **The model file.** 154 weight rows of the left mandible do not sum to 1 (see "The stage file"); nothing here
  changes the file.
- **Speed and reproducibility.** CPU: 30 to 70 s per registration and specimen on one core, depending on how busy the
  machine is, and about 1 to 2 minutes per specimen in all (a specimen with recovered joints is registered twice). In
  the production run over 955 scans with `--jobs 3`: median 65 s, 90th percentile 85 s per specimen; 0.6 to 0.7 GB per
  registration worker and up to about 2 GB per recovery worker. Two runs are bit-identical, with any `--jobs` and in any batch (checked on the same machine; another CPU or
  library build can differ in the last bits). `--device cuda:N`: about 40 s per specimen in all, not bit-identical.
  Windows was not run.

## Where the constants come from

Chosen on the development set with its truth in view (not held out): the confidence cut 0.30, the trunk-only
majority rule, the tolerances of the mandible check and of the recovery step, the status tolerances (`TIP_TOL_BL` was
chosen leave-one-specimen-out), the registration's surface options. Set without truth on development registrations:
the antenna zigzag cut. Set by eye on production overlays: the coverage cut 0.20 at 1.5 % BL. Set a priori: the
far-tie distance 4 % BL (the project's threshold for a wrong joint), at most 5 filled segments, the edge rule. The
segment-length prior of the recovery step (`data/population_prod82.json`; specimens keyed by the md5 of their name)
was built from single-state ("round 6") products of the 82 production scans without annotation and is applied to
ensemble products as well; the population reference from the same 82 scans registered with ensemble products. Both
files are rebuilt byte for byte by `python -m fitter_3d.antscan_registration.build_data priors|population` (see its
help) from those inputs, which are not in the repository.

## Known duplication

The parts under `fitter_3d/antscan_registration/parts/` and `fitter_3d/geomreg/` were built and validated as separate
modules and each keeps its own template class (three), virtual-marker regressor (also one in `fitter_3d/joint_init.py`),
linear-blend-skinning forward, Rodrigues formula and mesh loader. The bit-identical results reported here rest on
these exact implementations, so they were not merged; a consolidation onto `smal_model/batch_lbs.py`,
`smal_model/smal_torch.SMAL` and `fitter_3d/utils.load_meshes` needs its own equivalence run. The same holds for
`face_normals` and `limit_violation` (three copies each), the tube modules `geomreg/tubes.py` and
`parts/skelreg/tubes.py`, and the JSON writer of `parts/limbrec/common.py` (it converts values itself instead of using
the shared `json_default`). The names file, the output-folder lock and the JSON `default=` helper are shared with the
joint method's command (`fitter_3d/antscan_common.py`).

## Tests

`pytest tests/test_antscan_registration_pipeline.py tests/test_fitter_3d_geomreg.py tests/test_fitter_3d_geomreg_pipeline.py`
(CPU, no scan data: the specimens are made from the model itself; the model file of the repository is used). The
fast suite (`-m "not slow"`) covers the contract of every output file, the stage file under `SMAL3DFitter`'s own
forward and with a frame that is not the scan frame, the unposed bundle and its independence of the batch, and the
rules of intake (including far ties) and grade (a check that could not run does not pass). Marked `slow`: a batch of
ten degenerate inputs that must end as statuses. Resume, bit-identical reruns and batch independence were checked
on production specimens (see the workflow page); they are not repeated in the suite.
