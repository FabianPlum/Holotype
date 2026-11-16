# AntScan 2024 pipeline: from biomedisa.info meshes to OmniAnt25 registration inputs

Mesh preparation for the 2024-generation AntScan micro-CT meshes of ant workers: an outer envelope with a
shape-based vertex budget (Stage 1), a canonical orientation from anatomical cues for the sheets (Stage 2a) and a
proofreading of completeness (Stage 2p). The joint proposals (Stage 2b,
`fitter_3d/antscan_joints`) and the registration (Stage 3, `fitter_3d/antscan_registration`) build on these files. The
whole workflow, in the order it is run, with run times and the measured quality: `docs/antscan_workflow.md`.

## Conventions (apply everywhere)

* **Units**: micrometres, never rescaled. Scan voxel size comes from the specimen metadata (`voxel_size`).
* **Frames**: *scan* = the STL's own coordinates; *canonical* = OmniAnt frame, +X head, +Y animal's left,
  +Z dorsal, proper rotation (det +1), origin at the body-core centroid. Every JSON that changes frame stores
  `rotation` (rows = canonical axes in scan coordinates) and `origin_scan`: `x_canonical = R @ (x_scan - origin)`.
  Two such frames exist: Stage 2a's (`<name>_frame.json`, used for the sheets only) and the joint method's (`frame`
  of the joint product, computed from the trunk), which is the one every later step uses.
* **Selection**: workers only (no queens, males, larvae, outgroups), `family_or_other == Formicidae`, and no
  `problems_other` note matching `truncat|only head|split|detached|damaged`. Scans cut by the field of view are
  excluded by Stage 1. Internal geometry (cavity shells) is removed by Stage 1: it poisons the registration.
* **No TIFF volumes** are ever downloaded (1.5 GB each; mesh ids only).
* **Names**: `<Genus>_<species>_<code>` (as on the website), one folder per specimen in every stage directory.
* **Manifests**: every batch driver appends one JSON line per specimen to `<out>/pipeline_manifest.jsonl`
  (`name, stage, status, reason/metrics`), finished specimens are skipped unless `--force`.
* **Blender**: Stage 1 and the proofreading sheets run Blender 4.2 in the background. The drivers take `--blender PATH`;
  without it they use the environment variable `BLENDER`, else `blender` on the `PATH`.

## Stage 0: acquisition

The download script is not part of this repository yet. What it does, so that a download can be reproduced:

* lists `https://biomedisa.info/antscan/?show_all=True` (2,225 specimens; 1,671 workers), reads caste/subfamily
  from the listing, the per-specimen page JSON and the DataCite records (prefix 10.35097) **before** downloading,
  so filtered specimens cost no disk; resumable, two parallel streams, a 2 s pause between requests,
* fetches only the processed mesh id (`/antscan/download/?id=<mesh id>&object=processed`), refuses anything that
  is not an `.stl` Content-Disposition, verifies the binary STL size (`84 + 50 * n_triangles`),
* writes `<out>/<name>/<name>.stl` + `<name>.json` (caste, subfamily, voxel_size, problems, source ids, md5)
  and `manifest.jsonl`.

The 2024-generation meshes are 4-5 M triangles, µm, watertight, with closed internal cavity shells. The meshes of
the earlier 2022 download are in voxel units and lower quality and are not used.

## Stage 1: outer envelope with a shape-based vertex budget

`custom_processing/antscan_mesh_prep.py` (runs inside Blender 4.2), driven by `custom_processing/antscan_pipeline.py`.

1. import, unit check (voxel-unit meshes are converted with the metadata voxel size),
2. field-of-view check: boundary loops lying in a bounding-box face plane mean the scan was cut -> status
   `excluded`, no mesh written; other holes are filled,
3. voxel remesh (OpenVDB) at 2 scan voxels -> drop the internal cavity shells (closed parts with negative signed
   volume) and tiny fragments, keep every other outer part (legs or a head not connected to the main part in the
   remesh; dropping them cost Brachymyrmex_depilis its right legs in an early version). Outer parts are classified by
   their principal extents and face normals: elongated (legs, antennae) are kept; plates (card points and other mounts:
   the Carebara major carries one with 58 % of the ant's area) and small compact blobs (soil, glue) are dropped as foreign objects;
   large compact parts are kept. Any of these cases sets `envelope.part_flags`, and the specimen goes to review.
   A scan whose kept closed parts cover less than half of its area is excluded as fragmented (`min_area_share`),
4. snap the remeshed surface back onto the scan surface (normal-gated, 1.5 cells, 60 degrees) and Laplacian-relax
   the few unsnapped bridge vertices: this removes the voxel-grid staircase normals,
5. quadric-collapse decimation to `N = clip((2/sqrt 3) kappa^2 A / V^(2/3), 20k, 100k)` vertices (kappa = 60):
   the budget follows the surface-to-volume ratio, so spiny or long-legged specimens get more vertices per
   volume, compact ones fewer,
6. fidelity in scan voxels (output->scan and scan->output distances, removed-area share) written to the JSON.

Outputs: `<out>/<name>/<name>_envelope.stl` (µm, scan frame) and `<name>_envelope.json` (status, input stats,
voxel size, parameters, envelope stats, output stats, fidelity, timings), `<name>_envelope.log`.

```
# pytorch3d env, repo root; DATA = the downloaded meshes (<name>/<name>.stl + <name>.json)
python -m custom_processing.antscan_pipeline --in DATA --out OUT \
    [--jobs 8] [--threads 4] [--names A,B] [--force] [--prep-args "--envelope-voxels 0"] [--blender PATH]
```

Validation on 51 workers: 48 processed, 3 excluded (field of view); output->scan p99 below 0.5 scan voxels; median
normal deviation 3.7 degrees (same as direct decimation, i.e. no staircase left); 40k-100k vertices. Production
(1,618 workers): 1,442 envelopes, 174 excluded for a field-of-view cut and 2 as fragmented; 41k-99k vertices (median
60k); median 69 s per specimen (90th percentile 198 s). Finished specimens are skipped on a rerun, and outputs of a
specimen that has become excluded are removed.

## Stage 2a: canonical orientation

`custom_processing/antscan_align.py` (`align(stl) -> frame JSON`, CLI below). Why not PCA: the longest axis
leaves head/gaster and dorsal/ventral undecided and long legs or a curled gaster tilt or dominate it. The module
therefore

1. rasterises the envelope into an occupancy grid (column parity; pitch = V^(1/3) / 55),
2. splits body core from appendages with a ball opening (r = 0.09 V^(1/3)); appendage components are separated
   where only stripped surface layers glue them together (1-voxel erosion, voxels handed back to the nearest
   surviving component),
3. finds the coxa attachments (interior of the core axis, densest cluster, same side), antennae/mandibles (thin
   parts attached beyond the coxa span) and leg tips,
4. votes on anterior (coxae in the anterior half; thin appendages ahead of the fore coxae, none behind the hind
   coxae; hind legs bulkier than fore legs; waist deeper than neck; core thickness profile matched to the OmniAnt
   template forward vs reversed, `antscan_align_template_profile.json`) and on dorsal (coxae on the ventral surface;
   feet below the body),
5. takes the axis through the coxa attachments (the mesosoma, OmniAnt's root) when at least three are found,
   otherwise the core's principal axis,
6. reports confidences and flags; `status: "review"` when cues disagree or margins are small. Specimens in review
   are rendered and looked at; the decision is recorded in the JSON (`review` block) rather than silently accepted.

Outputs: `<name>_frame.json` (rotation, origin, confidences, votes, flags, leg attachments in µm, core length,
grid info) and optionally `<name>_canonical.stl`.

```
python -m custom_processing.antscan_align <name>_envelope.stl OUT/<name> [--write-mesh]
python -m custom_processing.antscan_pipeline --in DATA --out OUT --stage align    # all specimens with an envelope
```

**What the frame is used for.** Stage 2a's frame orients the review and proofreading sheets; nothing downstream reads
it. The joint method (Stage 2b) computes the frame that the joint products, the registration and the exports use from
the trunk itself, one anatomical definition for every specimen. A wrong Stage 2a frame therefore costs a less readable
sheet, not a wrong registration (in the production run roughly one sheet in five was rendered in a wrong frame).

**Review.** In production, frames with `status: "review"` were looked at on eight-hypothesis sheets (head flip x four
rolls) and corrected by hand; a corrected frame carries a `review` block, which `antscan_align` keeps on re-runs while
the envelope's md5 matches. The review tool is not part of this repository, because nothing downstream reads the
frame. Per-specimen Stage 1 options (which outer parts to keep) go into a hand-written
`OUT/<name>/<name>_overrides.json`, e.g. `{"keep_external": "none"}` (values `auto`, `none`, `all`), applied on the
next `--stage envelope --force --names <name>`.

**Validation.** Synthetic numbers do not transfer to real specimens (rest-pose legs, straight gaster, no sculpture):
a skeleton-based joint method scored 40/40 frames on the synthetic set and 32/47 on real workers. Real ground truth
therefore comes from rendered frames judged by eye (47 workers; single three-view montages first, then side-by-side
hypothesis sheets, which corrected four single-view dorsal judgements on compact spiny or arched ants).
Brachymyrmex_depilis_CASENT0709256 is excluded: the raw scan has no gaster (specimen damage not noted in the
metadata). Wrong automatic cases are big-headed majors (Carebara), strongly curled bodies (Acanthostichus),
specimens with legs merged into one appendage complex (Acanthomyrmex) and raised gasters (Crematogaster). Individual
cue accuracies on the real set: template profile 37/47, leg-tip reach 34/44, coxae in the anterior half 33/47, end
volume 33/47, thin appendages at the ends 27/39, waist vs neck 28/47; whole-shape overlap with the template is
recorded but not used (it confirmed every unflagged error). On the synthetic set the same module scores only 18/40
frames: the head direction is always within 35 degrees, but the dorsal side is often wrong because the synthetic legs
are posed knees-up with tibiae thinner than the grid, so the "legs beneath the body" cues have nothing to hold on to.
The real-data numbers are the ones that matter; the synthetic set stays the benchmark for joint positions.

## Stage 2p: proofreading

`custom_processing/antscan_proofread.py`: a four-view sheet per specimen (`--stage proofread_sheets`, rendered by
`antscan_proofread_sheet.py`) and a keep / reject decision per specimen in `<out>/<name>/<name>_proofread.json`
(`record`). Only complete workers are kept: six legs with complete tarsi ending in a pretarsus with claws, complete
antennae, all body parts, no fragmentation, debris, mounts or exposed internal structures. Records of the production
run may carry `source: "legacy_proofread"`: a keep taken over from the earlier proofreading of the 2022-generation
meshes (its rejections were not taken over: on the 2024 meshes about half of them are keeps). The import tool needs
that earlier, unpublished proofreading and is not part of this repository.

```
python -m custom_processing.antscan_proofread record OUT/<name> keep|reject [--reason R ...] [--by WHO] \
    [--source human|agent] [--notes TEXT]
```

`--reason` is one of `tarsus_or_claw_missing`, `leg_missing_or_broken`, `antenna_incomplete`, `body_part_missing`,
`fragmented_scan`, `debris_or_mount`, `carries_brood`, `internal_structure_exposed`, `noisy_scan`, `deformed_body`,
`not_a_worker`, `other`; `--source agent` marks a decision of an automatic reader (the readers themselves are not in
the repository). How the production set was proofread, and how far: `docs/antscan_workflow.md`, "How the data were
treated".

## Stage 2b: joint proposals

`fitter_3d/antscan_joints` (round 7 of the joint method, four-state ensemble):

```
python -m fitter_3d.antscan_joints.run --names NAMES.txt --env OUT --out JOINT_DIR [--jobs 2] [--mem-cap-gb G]
```

`--env OUT` is the output folder of Stage 1 (`<name>/<name>_envelope.stl`). Per specimen it writes a joint product
`proposals/<name>/<name>_joints.json`: the 55 joints plus `head_front` and `gaster_tip` (`joints_scan` /
`joints_canonical` in µm or `null`), `confidence`, `completeness`, `frame`, `body_length_um`, `status` and an
`ensemble` block. It also writes a limb product `limbs/<name>/limbs.json` + `limbs.npz` (named limb centrelines). The
joints lie on the scan's medial axis, inside the body, not on its surface. The method, its options and its cost are
in `docs/antscan_workflow.md` (step 4) and `fitter_3d/antscan_joints/README.md`; how it was chosen is in
`docs/antscan_joint_proposals_buildoff.md`.

## Stage 3: registration

`docs/antscan_registration_pipeline.md`: the registration pipeline (`python -m fitter_3d.antscan_registration.run
--names NAMES.txt --joints JOINT_DIR/proposals --limbs JOINT_DIR/limbs --env OUT --out REG_DIR`) from the joint and
limb products and the envelopes to registered OmniAnt meshes with a status per part, morphometrics and the files for
the Blender add-on. The design of the joint-informed `fitter_3d` path
(`--joints_dir`) is in `docs/antscan_registration_from_joints.md`, the registration core as a `fitter_3d` stage in
`docs/antscan_geom_stage.md`.

## Validation assets

* `custom_processing/omniant_synthetic.py`: posed OmniAnt instances with known joints, pivots, betas, pose,
  scale, rotation, optional removed parts, scan-like tessellation and a declared voxel size (e.g. 40 instances,
  seed 4, limit fraction 0.4: `python -m custom_processing.omniant_synthetic --out DIR --n 40 --seed 4
  --limit-frac 0.4`; `--damage` removes parts).

## Running everything

```
# 1. acquisition                               (download script not in the repository yet, see Stage 0)
# 2. envelopes                                 python -m custom_processing.antscan_pipeline --in DATA --out OUT
# 3. orientation                               python -m custom_processing.antscan_pipeline --in DATA --out OUT --stage align
#    proofreading sheets                       ... --stage proofread_sheets
#    proofreading decisions                    python -m custom_processing.antscan_proofread record OUT/<name> keep|reject
# 4. joints                                    python -m fitter_3d.antscan_joints.run --names NAMES.txt --env OUT --out JOINT_DIR
# 5. registration                              python -m fitter_3d.antscan_registration.run --names NAMES.txt \
#                                                  --joints JOINT_DIR/proposals --limbs JOINT_DIR/limbs --env OUT --out REG_DIR
```

`docs/antscan_workflow.md` walks through these steps, including how to write `NAMES.txt` from the proofreading
decisions.

## Open items

* Scans cut by the field of view stay excluded, even when only a leg or antenna tip is cut: only complete specimens
  are used (174 of the 1,618 downloaded workers were excluded for a cut, 2 more as fragmented scans).
* Add the download script to `custom_processing/`.
* Stepped ("staircase") envelopes that come from the raw scan are not smoothed or excluded, and no flag for them is
  written; proofreading rejects most of them.
* Stage 2a: thin tibiae of the synthetic template (0.4 % body length) fall below the grid pitch, so leg-based
  cues are weaker on synthetic than on real data; real femora/tibiae are 3-6 % and resolve fine.
* Legacy script defects (not fixed here, superseded for 2024 data): `prepare_antscan_data_for_mesh_fitting.py`
  PCA alignment with det -1 half of the time (outputs rotated 180 degrees), `reduce_vertices_by_distance` is a
  no-op, batch runner without `--python-exit-code 1`.
