# AntScan workflow: from public micro-CT meshes to registered ant meshes

This is the entry page for the AntScan tools of the repository. They take the public micro-CT surface meshes of ant
workers from [AntScan](https://biomedisa.info/antscan/) to registered meshes of the OmniAnt model
(`3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`, 10,235 vertices, 55 joints). For every specimen they write a status
per body part, morphometrics in micrometres, a fitter_3d stage file and an unposed mesh for the Blender add-on.

The steps are listed in the order you run them. Run every command from the repository root in the conda environment.

| step | what | command | details |
|---|---|---|---|
| 0 | download the 2024-generation meshes | not in the repository yet (see step 0) | [Stage 0](../custom_processing/README_antscan2024.md#stage-0-acquisition) |
| 1 | outer envelope with a vertex budget (Blender) | `python -m custom_processing.antscan_pipeline` | [Stage 1](../custom_processing/README_antscan2024.md#stage-1-outer-envelope-with-a-shape-based-vertex-budget) |
| 2 | orientation for the sheets, proofreading sheets | `... --stage align`, `... --stage proofread_sheets` | [Stage 2a](../custom_processing/README_antscan2024.md#stage-2a-canonical-orientation), [Stage 2p](../custom_processing/README_antscan2024.md#stage-2p-proofreading) |
| 3 | proofreading decisions and the name list | `python -m custom_processing.antscan_proofread record` | [Stage 2p](../custom_processing/README_antscan2024.md#stage-2p-proofreading) |
| 4 | joint proposals | `python -m fitter_3d.antscan_joints.run` | [fitter_3d/antscan_joints/README.md](../fitter_3d/antscan_joints/README.md) |
| 5 | registration, status, morphometrics, export | `python -m fitter_3d.antscan_registration.run` | [antscan_registration_pipeline.md](antscan_registration_pipeline.md) |
| 6 | shape model | Blender add-on: "Load all unposed registered meshes" | [Outputs](#outputs-and-who-reads-them) |

## Before you start

- **Environment.** `conda env create -f environment.yml` installs what the steps need, including four packages the
  AntScan code adds:
  - `embreex`: trimesh's ray caster. Steps 4 and 5 refuse to start without it.
  - `threadpoolctl`: one BLAS thread per registration, so that two CPU runs are bit-identical.
  - `scikit-learn==1.7.2`: the joint method's classifier was saved with this version. Any other version stops with an
    error. `ANTSCAN_JOINTS_SKLEARN_ANY=1` loads it anyway, but the products may then differ.
  - `numba==0.66.*`: the joint method's voxelisers.

  The production run and the reproduction checks used numpy 1.26.4, scipy 1.14.0, scikit-image 0.24.0, trimesh 4.4.3,
  numba 0.66.0, embreex 4.4.0, threadpoolctl 3.6.0 and scikit-learn 1.7.2. Only scikit-learn and numba are pinned;
  other versions of the floating packages can change results in the last bits.
- **Blender 4.2** for step 1 and for the sheets of step 2: `--blender PATH`, else the environment variable `BLENDER`,
  else `blender` on the `PATH`.
- **Linux.** Steps 4 and 5 were run on Linux only. The joint method links its products with relative symbolic links,
  so an output folder can be moved; on Windows, symbolic links need developer mode. No GPU is needed (step 5 can use one with `--device cuda`, which is faster but not bit-reproducible).
- **Disk per specimen** (sizes in the production run): about 200 MB for the downloaded mesh, 10 MB for the envelope
  and its sheets, 11 MB for the joint products and 8 MB for the registration outputs.

## Step 0: download

The download tool used for the production run is not part of the repository yet. What it does is described under
[Stage 0](../custom_processing/README_antscan2024.md#stage-0-acquisition). In short:

- it fetches only the processed mesh of a specimen, never the TIFF volumes (about 1.5 GB each);
- it reads caste and metadata before downloading, so filtered specimens cost no disk;
- it checks the size of every binary STL.

Step 1 reads one folder per specimen, `DATA/<name>/<name>.stl` plus `<name>.json`. The metadata file needs the fields
`caste`, `family_or_other`, `voxel_size` and `problems_other`. The older `custom_processing/antscan_scraper.py`
(Selenium) writes a similar folder layout, but it clicks every download button of a specimen page. It was not used
for the production run and has not been checked against the current site.

## Step 1: outer envelope

```bash
python -m custom_processing.antscan_pipeline --in DATA --out PROC [--jobs 8] [--threads 4] [--timeout 1800] \
    [--names NAME1,NAME2] [--force] [--blender PATH] [--prep-args "--envelope-voxels 0"]
```

The default stage is `envelope`; `--names` takes a comma-separated list. The command processes only workers of the
Formicidae whose `problems_other` note does not mention truncation, "only head", a split, a detached part or damage.
For each specimen it writes `PROC/<name>/<name>_envelope.stl` (scan frame, micrometres) and a `<name>_envelope.json`
with the status, the parameters and the fidelity to the scan. Scans cut by the field of view are excluded (status
`excluded`, no mesh), and so are fragmented scans. What the stage does in detail (voxel remesh, cavity removal,
outer-part rules, snap-back, vertex budget) is in
[Stage 1](../custom_processing/README_antscan2024.md#stage-1-outer-envelope-with-a-shape-based-vertex-budget).
Every stage appends to `PROC/pipeline_manifest.jsonl` and skips finished specimens unless `--force` is given.

## Step 2: orientation and sheets

```bash
python -m custom_processing.antscan_pipeline --in DATA --out PROC --stage align             # <name>_frame.json
python -m custom_processing.antscan_pipeline --in DATA --out PROC --stage proofread_sheets  # <name>_proofread.png
```

`align` gives each envelope a canonical frame from anatomical cues (`antscan_align.py`). This frame only orients the
sheets: nothing downstream reads it. The joint method of step 4 computes the frame that every later step uses from the
trunk itself. `proofread_sheets` renders four views of every envelope (and aligns it first if needed). A
hand-written `PROC/<name>/<name>_overrides.json`, for example `{"keep_external": "none"}`, sets per-specimen options of
step 1 for the next `--stage envelope --force --names <name>`. The tools used during production to review frames by
eye are not part of this repository, because nothing downstream reads the frame.

## Step 3: proofreading decisions and the name list

One decision per specimen, stored in `PROC/<name>/<name>_proofread.json`:

```bash
python -m custom_processing.antscan_proofread record PROC/<name> keep
python -m custom_processing.antscan_proofread record PROC/<name> reject --reason tarsus_or_claw_missing [--notes TEXT]
```

`--reason` can be given more than once; `--by` names who decided; `--source human|agent` says whether a person or an
automatic reader decided. Earlier decisions are kept under `history`. Only complete workers are kept (see
[How the data were treated](#how-the-data-were-treated)). Steps 4 and 5 take a text file with one name per line, for
example the kept specimens:

```bash
python -c "import glob, json; print('\n'.join(sorted(d['name'] for d in (json.load(open(p)) for p in glob.glob('PROC/*/*_proofread.json')) if d['decision'] == 'keep')))" > NAMES.txt
```

## Step 4: joint proposals

```bash
python -m fitter_3d.antscan_joints.run --names NAMES.txt --env PROC --out JOINT_DIR [--jobs 2] [--mem-cap-gb G]
```

The joint method places the 55 template joints (plus `head_front` and `gaster_tip`) and the limb centrelines on each
envelope, without a posed model. For each specimen it runs one chain of steps under each of the four sign states of
TRUNK's working frame (s0 to s3). s0 is the PCA frame of the envelope, with each axis signed so that the area-weighted
third moment along it is positive. s1 to s3 are the same frame with one pair of axes flipped. All four states are
intrinsic to the shape.

| step | what it does |
|---|---|
| trunk | voxel occupancy, trunk medial path, head end by anatomy, trunk joints (neck, waist, gaster), body length, working frame |
| paths_A | geodesic limb paths with contact cutting, named with fitted priors |
| paths_B | curve-skeleton graph of the solid |
| paths_C | tube decomposition with a fitted contact model |
| thin | extension of thin leg and antenna tips on the mesh surface; discovery of limbs that no approach found |
| assemble | per limb, the best-supported of the three centrelines; ALONG places the joints along it (gradient-boosting classifiers and segment-length priors); anatomical frame |
| folds | folded knees, farthest leg tip, antennal elbow |

The ensemble then combines the four states:

- states whose head end disagrees with the majority are dropped;
- each trunk joint is the medoid over the kept states;
- each limb is the centreline best supported by the other states;
- each joint is the medoid of the agreeing states;
- confidence = median state confidence × (0.5 + 0.5 × state support) × placed share × 1/(1 + (spread / 2 % BL)²).

Outputs in `JOINT_DIR`:

| path | content |
|---|---|
| `proposals/<name>/<name>_joints.json` | joint product: 55 joints plus `head_front` and `gaster_tip` in the scan frame (micrometres, or null), a confidence per joint, the frame, the body length and the ensemble block |
| `limbs/<name>/limbs.json`, `limbs.npz` | limb product: named limb centrelines (legs `L1..R3`, antennae, mandibles), tip first |
| `states/<k>/...` | per-state products; step 5 reads `final/proposals` and `paths_{A,B,C}` from here by itself |
| `summary.tsv`, `jobs.jsonl` | failed steps per specimen (the joint product does not carry them); one record per job |

Each step runs in its own process with two BLAS threads, `--jobs` at a time. Every job has a time limit (`--timeout`,
3600 s for the stages, 900 s for the other steps) and a memory cap (`--mem-cap-gb`; default 32 GiB, or 80 % of the
machine's memory divided by `--jobs` when that is less). A job over either limit
is recorded as a failed step, and the ensemble then uses the other states and approaches. A rerun skips finished steps
and reruns whatever depends on an unfinished one. Failed steps stay failed unless `--retry-failed` is given. The
classifier and the two fitted parameter files cannot be rebuilt from the repository: see
[fitter_3d/antscan_joints/README.md](../fitter_3d/antscan_joints/README.md#data-files).

## Step 5: registration, status and export

```bash
python -m fitter_3d.antscan_registration.run --names NAMES.txt --joints JOINT_DIR/proposals \
    --limbs JOINT_DIR/limbs --env PROC --out OUT_DIR [--jobs 3]
```

The production run used exactly this command, with all other options at their defaults (CPU). Per specimen:

1. **Intake** decides which joints are usable (confidence cut 0.30, with majority rules for ensemble products).
2. **First registration**: a closed-form start in fitter_3d's own parameters, then a surface stage, a repair pass and
   a closed-form refit.
3. **Recovery** searches again for lost limb joints and accepts them only with independent evidence.
4. **Second registration**, for specimens whose usable joints changed.
5. **Status and export**: status per vertex, bone, limb segment and specimen; specimen class; morphometrics; stage
   file; unposed meshes; overlay.

The page [antscan_registration_pipeline.md](antscan_registration_pipeline.md) documents every option, rule, output key
and status code. Two runs on the CPU are bit-identical with any `--jobs` and in any batch, and a rerun of the same
command recomputes nothing.

## Outputs and who reads them

| file in `OUT_DIR` | content | read by |
|---|---|---|
| `summary.csv` | one row per specimen: class, reasons, counts of measurable segments, recovery, timings, errors | you, to select specimens |
| `morphometrics_measurable.csv` (also `morphometrics.csv`, `morphometrics_flags.csv`) | lengths, widths, areas and volumes in micrometres; the first file holds only values with flag 0 | analysis |
| `registration/<name>.npz` | the registered mesh `V_final` (scan frame, micrometres) and its native fitter_3d parameters | anything that needs the mesh on the scan |
| `status/<name>_status.json`, `.npz` | status of every vertex, bone, limb segment and chain, with reasons | selection of vertices and parts |
| `export/batch_0/Stage_status.npz` | a fitter_3d stage file: every key of `trainer.Stage.save_npz` plus the status. `SMAL3DFitter(parameters) + deform_verts = verts` holds for it | fitter_3d tools (`fitter_3d/read_out_fitter_stages.py`), the Blender add-on's "Direct Import SMIL Model" |
| `export/uc1_bundle_model_grade.npz` (all specimens: `uc1_bundle.npz`) | unposed rest meshes (`verts`, `labels`, status per vertex) of the model-grade specimens | the Blender add-on (below) |
| `overlays/<name>.png` | registered mesh coloured by status, drawn over the scan | a look by eye |

**Shape model in Blender** (step 6). Install the repository's add-on first: `python 3D_model_prep/build_addon.py`
writes `3D_model_prep/smil_importer.zip`; in Blender use Edit > Preferences > Add-ons > Install from Disk, pick the zip and enable
"SMIL Model Importer" (details, including its scipy / scikit-learn dependencies:
[3D_model_prep/smil_importer/README.md](../3D_model_prep/smil_importer/README.md)). Loading the meshes writes four files
next to the PKL (`smil_morph_data.csv`, `smil_morph_PC_data.csv`, `smil_morph_PC_xy.csv`, `smil_morph_PCA_stats.txt`),
so copy the PKL to a folder of your own first instead of pointing the add-on into the repository's `3D_model_prep/`.
Then, in the add-on panel ("SMIL Model Importer"):

1. Set "PKL Filepath" to your copy of `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl` and "NPZ Filepath" to
   `export/uc1_bundle_model_grade.npz`.
2. Untick "Auto Clean-up Mesh": it can merge midline vertices and break the vertex correspondence.
3. Click "Load all unposed registered meshes". This is slow: the 423 model-grade meshes of the production run took
   about an hour (about 9 s per mesh).
4. Click "Generate SMIL model from unposed meshes", then "Export SMIL Model". The production model used 25 principal
   components, "Symmetrise" unticked and "Perform separate PCAs for shape, scale, and translation" ticked. The add-on's
   defaults (20 components, "Symmetrise" ticked) give a different model.

The exported `.pkl` is a model like any other in the repository: fitter_3d uses it through `config.set_model` /
`config.SMAL_FILE`, and the neural pipeline loads it as its model file. The model built this way from the 423
model-grade specimens of the production run has the base model's keys, topology and joint order, and both fitter_3d's
forward pass and the neural pipeline's model loader accept it. Before you average the bundle, read
[Measured quality](#measured-quality-production-run).

## Run time and memory per specimen

Measured in the production run on one Linux machine:

| step | time per specimen | memory |
|---|---|---|
| 1 envelope | median 69 s, 90th percentile 198 s (Blender, 4 threads per job; 1,442 envelopes) | not recorded |
| 4 joint proposals | about 13 CPU-minutes, median (90th percentile 29, maximum 92); 1,013 specimens | the largest job of a specimen: median 5.8 GB, 90th percentile 12.5, 99th 18, maximum 29.3 GB. Memory depends on the shape, not on the size, because the voxel grids are sized in body lengths |
| 5 registration, all steps | median 65 s, 90th percentile 85 s, maximum 103 s (CPU, `--jobs 3`; 955 specimens) | 0.6 to 0.7 GB per registration worker and up to about 2 GB per recovery worker (measured on three specimens after the production run); plan for up to 2 GB per worker. One invocation over all 955 peaked at about 9 GB, because the export holds every row |

Specimens that had to be registered twice (88 % of them) account for most of the registration time. The joint
method's jobs that grew without bound in production reached 52 and 69 GB before the system stopped them, which is why
the packaged command has a memory cap per job.

## How the data were treated

- **Source.** Only the 2024-generation AntScan surface meshes are used: micrometre units, 4 to 5 million triangles,
  one per specimen, with the generation read from the STL header. The earlier 2022-generation meshes (voxel units,
  coarser) and the TIFF volumes are not used.
- **Workers only.** No queens, males, brood or outgroup specimens, selected by the caste field of the source. Caste
  labels can be wrong: one "worker" with wings was rejected in proofreading. 1,618 worker meshes were downloaded.
- **Outer envelope.** Each mesh is reduced to its outer surface: voxel remesh at 2 scan voxels, internal cavity shells
  removed, the surface snapped back onto the scan, at most 100,000 vertices (the budget follows the
  surface-to-volume ratio), normals cleaned. Foreign objects (card mounts, soil, glue) are dropped by their shape. The
  scale stays in micrometres and is never changed. Of the 1,618 meshes, 174 were excluded because the field of view
  cut the scan and 2 because the scan was fragmented: 1,442 envelopes remained. Stepped ("staircase") surfaces that come
  from the raw scan are neither smoothed nor excluded; proofreading rejected most of them.
- **Complete specimens only.** A specimen is kept only when it has six legs with complete tarsi, pretarsi and claws,
  complete antennae and all body parts, and shows no debris or mount, no carried brood, no exposed internal structure,
  and no fragmented or very noisy surface. A scan cut by the field of view is excluded even when only a leg or antenna
  tip is cut; a truncated gaster is excluded. Drying cracks, a crumpled gaster cuticle and strongly curled bodies are
  accepted. The rule has no exceptions: specimens with a single known defect are rejected too.
- **Proofreading.** Two independent automatic readers (vision-language models) judged a four-view sheet of each
  envelope. The same call twice was final; anything else went to a person (the curator), whose decision is always
  final. A blind audit then showed that the curator rejected 8 of 50 automatic keeps that a third read had confirmed
  (16 %), so the curator now reviews every automatic keep. The production run registered the 955 specimens that
  were usable when it started: 551 checked by the curator and 404 kept by automatic reading only, whose review was
  not finished. Some of those 404 will not meet the rule. They can be dropped by name afterwards, because every
  per-specimen file and every row of the stage file and the bundles depends on its own specimen only. The automatic
  reading itself is not part of the repository.
- **Frames.** The frame used downstream is the one the joint method computes from the trunk, one anatomical definition
  for all specimens. Orientations from an earlier proofreading of the 2022 meshes served only as a check of head and
  dorsal direction; on the first 103 scans, all 54 that had such an orientation agreed with it.
- **Units.** Joints, registrations and morphometrics are in micrometres in the scan frame. The unposed meshes are in
  model units with the size removed (trunk centroid size); `um_per_model_unit` converts them back.
- **Status classes** (column `specimen_class`; all rules are in the
  [registration page](antscan_registration_pipeline.md#status)):
  - `model_grade`: fit for a shape model. Every trunk part is measurable, at most 5 of the 34 leg and antenna segments
    are filled in the unposed mesh, no trunk vertex is filled, there is no abnormal stretched edge and no crumpled
    antenna, the left/right naming is not doubted, and the native parameters reproduce the mesh.
  - `partial`: everything else that registered. Use its measures by their flag.
  - `do_not_use`: the registration failed or its root was placed by a fallback, the head end is doubtful and not
    confirmed by the scan, the mesosoma is not on the scan, or a processing error occurred.

  "Measurable" means that the scan does not contradict the part, not that the part is verified correct.

## Measured quality (production run)

The registrations come from the frozen production run (pipeline version `regpipe-1.1`; four-state joint products):

- **Classes, 955 scans:** 423 model grade (44 %), 475 partial (50 %), 57 do not use (6 %). No specimen failed with a
  processing error. On 4 of them the recovery step failed, so they were registered without it.
- **Limb segments:** 82.5 % of the 32,470 leg and antenna segments are measurable (26,803), 5.1 % are in review and
  12.3 % are unregistered. A median specimen has 30 of 34 segments measurable, and 134 specimens (14 %) have all 34.
- **Accuracy:** against annotated joints of 21 development scans (annotated automatically and checked in part by eye;
  see the registration page), measurable limb segments are rarely wrong (7 of 542 more than 4 % of body length off). The median relative error of the lengths is 0.9 to 3 % for femur, tibia and
  tarsus, and 6 % for the coxa, which is not ready to report. The annotated set was used throughout development, so
  these numbers are optimistic; details are in the
  [registration page](antscan_registration_pipeline.md#evidence). The final test on an annotated set that has never
  been used is still to come.
- **Antennae and tarsi are too thin.** The template's distal limbs are much thinner than real ones, and the
  registration does not fully restore the girth. Measured against the scan on 103 scans of the development phase, the
  registered radius is short by 35 % on the scape, 20 % on the funiculus, 11 % on the basitarsus, 14 % on the rest of
  the tarsus and 3 % on the tibia. Radius and cross-section measures of these segments are biased low.
- **The unposed meshes self-intersect.** The 955 unposed meshes have a median of 729 crossing face pairs each (the
  posed registrations 445). Of all crossing pairs, 63 % arise in the unposing (neighbouring parts overlap in the rest
  pose), 26 % come from the fill of parts that are not measurable, and 11 % already cross in the posed registration.
- **Twisted tarsi and funiculi.** In the unposed rest mesh, a nearly straight tarsus or funiculus carries an arbitrary
  spin about its own axis, different in every specimen. Averaging such rows makes the mean tarsus collapse: over the
  423 model-grade rows, the radius of the mean tarsus is 0.27 of the rows' own radius (funiculus 0.81). A mean mesh
  or a PCA built directly from the bundle inherits this, including the add-on's model of step 6.

## Known limits

- **Wrong joints are reproduced.** A joint that the joint method places confidently at the wrong articulation is
  registered and called measurable, also in a model-grade specimen.
- **The class grades what the scan can contradict.** `partial` is a wide class: read `n_measurable`,
  `trunk_not_measurable` and `n_segments_filled` before using such a specimen.
- **Template limits.** Waist nodes, spines, shields, antennal clubs, long mandibles and curled gasters are not
  reproduced, and petiole measures are unreliable. No mandible measure is reported.
- **The model file.** 154 skinning-weight rows of the left mandible in `OmniAnt_25PCs_joint_limited.pkl` do not sum
  to 1 (see [the stage file](antscan_registration_pipeline.md#the-stage-file)).
- **Platform.** Tested on Linux only. The joint method links its products with relative symbolic links, so its
  output folders can be moved. Registration outputs record absolute input paths: a moved registration folder can be
  read, but resuming it recomputes the status evidence.
- **Not in the repository:** the download tool, the automatic proofreading readers, the training code and annotations
  behind the joint method's classifier, and the annotated evaluation sets.

The full list is under "What it cannot do" on the [registration page](antscan_registration_pipeline.md#what-it-cannot-do).

## Other routes inside fitter_3d

The registration core also runs inside `fitter_3d/optimise.py`, as one stage (`scheme: 'geom'`) or as the
initialisation of the joint-informed schedule (`--joints_dir`, `--joint_kind medial`, `--limbs_dir`, `--native_start`).
Both take the envelopes as `.obj` files in the joint product's canonical frame and the joint products in one flat
folder (`<name>_joints.json`); no converter for either ships with the repository. They produce no status, recovery
or unposed meshes; see [antscan_geom_stage.md](antscan_geom_stage.md). Every one of these options is off by default.
With all of them off, fitter_3d writes the same arrays as before; its stage files only gain the keys `log_scale`
(zeros), `rotation_mask` and `pivots`.

## Tests

```bash
pytest tests/test_antscan_joints.py tests/test_antscan_registration_pipeline.py tests/test_fitter_3d_geomreg.py \
       tests/test_fitter_3d_geomreg_pipeline.py tests/test_fitter_3d_geom_stage.py -m "not slow"
```

The tests use no scan data: their specimens are made from the model file itself. One row per file is in
[tests/README.md](../tests/README.md).

## Pages

| page | what |
|---|---|
| [custom_processing/README_antscan2024.md](../custom_processing/README_antscan2024.md) | steps 0 to 3 in detail: download, envelope, orientation, review and proofreading |
| [fitter_3d/antscan_joints/README.md](../fitter_3d/antscan_joints/README.md) | the joint method's code layout, data files, dependencies and provenance |
| [antscan_joint_proposals_buildoff.md](antscan_joint_proposals_buildoff.md) | how the joint method was chosen: the comparison of methods and the rounds that led to it |
| [antscan_registration_pipeline.md](antscan_registration_pipeline.md) | the registration pipeline: options, rules, outputs, status codes, evidence and limits |
| [antscan_geom_stage.md](antscan_geom_stage.md) | the registration core as a fitter_3d stage and as an initialisation |
| [antscan_registration_from_joints.md](antscan_registration_from_joints.md) | the original design of the joint-informed fitter_3d path and its first results |
