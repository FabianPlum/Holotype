# Registering OmniAnt25 to AntScan meshes with joint proposals (design, 2026-11-09)

The design behind the joint-informed path of `fitter_3d` (`--joints_dir`, `fitter_3d/joint_init.py`,
`fitter_3d/antscan_joints_cfg.yaml`), written before it was built; sections 5 and 6 record what the first prototype
showed. The registration that is used for the AntScan scans today grew out of it and is described in
`docs/antscan_registration_pipeline.md` (the same core as a `fitter_3d` stage: `docs/antscan_geom_stage.md`). The
mesh preparation it starts from is in `custom_processing/README_antscan2024.md`, the joint method that writes the
joint products in `fitter_3d/antscan_joints/README.md`. This page is kept as the record of the design: where it differs
from the code (file names such as `<name>_processed.obj`, the frame of the hand-over, the joint method of section 6),
the code and `docs/antscan_workflow.md` describe what the repository does today.

## 1. What the current fitter does with a mesh

`fitter_3d/optimise.py` + `fitter_3d/trainer.py` before the joint path was added (line numbers of that version):

| aspect | current behaviour | where |
|---|---|---|
| input | `.obj` only; every mesh is centred on its vertex mean and divided by its largest absolute coordinate; centre and scale are discarded | `fitter_3d/utils.py:343-348` |
| initial pose | identity global rotation ("head-on"): the target must already face +X with +Z up (README: "manually align your model") | `trainer.py:125-129` |
| size | no global scale parameter. `ants_cfg.yaml` Stage_0 (`init_rot_lock`: `trans` + `log_beta_scales`) uses the per-joint log scales for size. The root's scale is not applied to the root segment at all and enters every child transform inverted (`S_parent^-1 R S_child`, `smal_model/batch_lbs.py:128-152`), so a larger root scale shrinks the legs | `batch_lbs.py:140-152` |
| losses | chamfer (3000 target samples vs all model vertices), edge, normal consistency, Laplacian, shape-diameter ("SDF") correspondence, joint-limit hinge. No joint term | `trainer.py:401-479` |
| masks | `rotation_mask` / `global_mask` are declared but never applied in `forward` | `trainer.py:151-157, 198-275` |
| outputs | per-stage `.npz`: `global_rot, joint_rot, betas, log_beta_scales, trans, deform_verts, betas_trans, verts, faces, labels`. No target scale, no joints | `trainer.py:538-552` |

Consequences for AntScan: physical scale is lost (no morphometrics), the fit depends entirely on a correct
pre-alignment, size is fitted through a parameter with inverted semantics, and nothing stops a leg from locking
onto a neighbouring leg or a wrong side (the failure mode we saw on the 2022 data). Missing body parts cannot be
excluded from the fit.

## 2. What the AntScan pipeline will hand over (per specimen, Stage 2)

* `<name>_processed.obj`: watertight outer envelope, 20k-100k vertices, **micrometres**, canonical frame
  (+X head, +Y animal's left, +Z dorsal), origin at the body-core centroid.
* `<name>_joints.json`: all 55 OmniAnt joint names with `joints_canonical` (µm or `null`), a `confidence` per
  joint, `completeness.missing_joints`, `frame` (scan -> canonical rotation and origin), `status`.
  For the trunk chain (`b_h`, `b_t`, `b_a_1..5`) the file will also carry `part_centroids` (centre of volume of
  head, mesosoma, petiole node(s) and gaster thirds): the model's own gaster bone heads sit close to the surface,
  so centre-of-volume targets are compared with centre-of-volume model points, not with pivots (section 3.C).
* `<name>_frame.json`: orientation cues, confidences and review flags (Stage 2a).

## 3. Proposal: joint-informed registration, as an extension of `fitter_3d`

Default behaviour stays identical when no joints are given; everything below is switched on by `--joints_dir`.

### A. Keep the physical scale
`load_meshes` returns the per-mesh `centre` and `scale` it already computes; `Stage.save_npz` writes
`target_centre`, `target_scale`, `target_units="micrometre"` and the Stage-2 frame rotation. Any model-space length
times `target_scale` is micrometres; the inverse frame rotation maps results back into the scan.

### B. A real global scale
New parameter `log_scale` (one scalar per mesh) in `SMAL3DFitter.forward`:
`verts = exp(log_scale) * verts_model + trans`, same for joints and pivots. Added to the `init`, `default`, `shape`,
`pose`, `all` groups. The per-joint `log_beta_scales` then only express proportions. (Alternative with no new
parameter: rescale the target by `1/s0` from the joint initialisation and record it. The parameter is preferred
because the mesh terms refine it.)

### C. Joint loss
`w_joint` in `loss_weights`; the Stage receives `target_joints` (B x 55 x 3, normalised target frame) and
`joint_weights` (B x 55: confidence, 0 for `null`, wings and missing joints).

* limb articulations (`co, tr, fe, ti, ta, pt, an_1..3, ma`): compare the model **pivots**
  `exp(log_scale) * smal_model.J_transformed + trans` (the posed kinematic joints; `J_transformed` already includes
  `log_beta_scales` and `betas_trans`) with the proposed joints;
* trunk parts (`b_h`, `b_t`, `b_a_1..5`): compare **part centroids**: mean of the posed model vertices whose dominant
  skinning weight belongs to the part (differentiable, pose-dependent) with the proposed centre-of-volume points;
* robust penalty (Geman-McClure or Huber with width ~2 % body length) so that a wrong proposal cannot drag the fit;
* loss normalised by the body length (`b_h` to gaster tip in the target) so weights do not depend on specimen size.

### D. Initialisation from the proposals (closed form, before any optimisation)

1. **Similarity**: Umeyama/Kabsch with scale between the model rest-pose pivots and the proposed joints on the set
   that is rigid with the root segment: `b_t`, `b_a_1`, `b_h` and the six coxae (all direct children of `b_t`,
   kinematic tree `kintree_table`). Gives `global_rot`, `trans`, `log_scale`. The canonical frame makes this a
   small correction; it absorbs the residual tilt of the orientation stage.
2. **Shape from bone lengths**: rest-pose pivots are linear in `betas`
   (`J(beta) = J_regressor (v_template + shapedirs beta)`), so bone lengths `|J_child - J_parent|` measured on the
   proposals (pose-invariant, scaled by `1/s0`) give `betas` by two or three Gauss-Newton steps with the shape prior
   (`betas_prec`). This is also the first morphometric output (segment lengths in µm) and needs no mesh fit.
3. **Pose by forward-kinematic IK**: walk the tree from `b_t`; at each joint compute the smallest rotation that turns
   the current bone direction (after the parents' rotations) into the proposed child direction, express it in the
   parent frame, clip to `joint_limits`. Joints with several children (`b_t`, `b_h`) take the Kabsch rotation over
   all confident children. Twist about the bone axis stays 0 (unobservable from one direction; the mesh terms
   refine it). Joints without a proposal keep the rest pose.
4. **Completeness**: `missing_joints` -> weight 0 in C, `rotation_mask` 0 for the joints beyond the break
   (`rotation_mask` must actually be applied in `forward`: `joint_rot * rotation_mask`), and the model vertices
   dominated by the missing bones are excluded from the chamfer source set (`x_lengths` padding in
   `chamfer_distance`), so an intact model leg is not dragged onto a stump. The specimen is still fitted; only the
   missing part is free.

### E. Stage schedule (`fitter_3d/antscan_joints_cfg.yaml`)

| stage | parameters | losses | purpose |
|---|---|---|---|
| 0 skeleton | `global_rot, trans, log_scale, joint_rot` | `w_joint 1, w_limit 100` (no mesh terms) | from the closed-form init to a consistent skeleton, 200 its |
| 1 shape+pose | `default` + `log_scale` | `w_chamfer 1, w_joint 0.3, w_limit 100, w_edge 0.8, w_normal 0.02, w_laplacian 0.01` | surface fit with the skeleton as regulariser, 500 its |
| 2 SDF refine (optional) | as `ants_sdf_cfg.yaml` Stage_1B | + `w_joint 0.1` | thin-part correspondence |
| 3-4 deform coarse/fine | `deform_verts` | as today + `w_joint 0.05` | per-vertex detail without the surface drifting off the skeleton |

### F. Outputs for morphometrics
Per specimen `<name>_fit.json`: `betas`, `log_scale`, `target_scale`, final pivots and part centroids in µm in the
canonical **and** the scan frame, bone lengths in µm, body length, completeness used, final loss components.
The registered mesh is written in µm in both frames (`.obj`).

### G. Validation (repeatable, before touching real fits)
1. Synthetic set (`custom_processing/omniant_synthetic.py`, 40 instances with known `betas`, `theta`, pivots,
   scale, rotation; 17 with a removed part): after each stage measure pivot error (% body length), betas recovery,
   bone-length error (µm), chamfer, missing-part handling. Compare the joint-informed schedule with
   `ants_cfg.yaml` on the same meshes, same iteration budget.
2. Real QA set (12 specimens): overlay renders of the registered model on the envelope, and look at every image.
3. A fast regression test: the closed-form init on one synthetic specimen reproduces the GT rotation within 5
   degrees, scale within 2 % and median pivot error below 3 % body length.

### H. Effort and files
* `fitter_3d/utils.py` (load_meshes centre/scale), `fitter_3d/trainer.py` (log_scale, joint loss, masks, npz),
  `fitter_3d/joint_init.py` (new: Umeyama, bone-length betas, FK-IK), `fitter_3d/optimise.py` (`--joints_dir`,
  `--target_units`), `fitter_3d/antscan_joints_cfg.yaml`, tests under `tests/`. Roughly 400-500 lines.
* Minimal viable subset if time is short: A + C (pivots only) + D1. That alone fixes scale, size and side/limb
  correspondence; D2/D3 mostly save iterations and remove the remaining local minima in leg pose.

## 4. Decisions needed
* Whether the gaster chain follows the model convention (near-surface bone heads) or centre-of-volume targets
  (section 3.C, recommended: centroids on both sides).
* Whether `log_scale` becomes a model parameter (preferred) or the target is pre-scaled.
* Whether the SDF stage stays in the AntScan schedule (it needs per-specimen SDF files; the joint term may make it
  unnecessary; decide on the synthetic A/B).

## 5. Prototype results (2026-11-09)

A prototype that extended fitter_3d without modifying it. The numbers were re-computed independently; a visual check
of the real specimens rejected their grades.

Synthetic set (40 posed OmniAnt instances; dev = test set, so optimistic), pivot error in % body length:

| | median | p90 | bone-length error | note |
|---|---|---|---|---|
| current fitter_3d (ants_cfg.yaml, GT pre-alignment) | 2.11 | 12.26 | 11.8 % | surface fits, skeleton wrong |
| joint-informed (closed-form init + joint term), good proposals | 0.34 | 0.85 | 0.3 % | same 2400 iterations |
| same, through the patched fitter_3d (optimise.main) | 0.32 | 0.80 | 0.4 % | minimal patch of fitter_3d |

What matters, by measurement:
* the joint term on posed pivots is the patch (without it: 1.08 % from a perfect init, bones 4.4 %);
* physical-scale bookkeeping (centre/scale/frame) costs nothing and gives micrometres;
* rotation_mask + chamfer source exclusion do not change joints but stop the deform stages gluing missing parts onto the
  scan (removed parts 0.17 % BL from the scan without masks, 2.65 % with);
* closed-form init D1 + FK-IK with a limit-aware twist search (D3); freeze per-joint log_beta_scales for AntScan
  (they drift into implausible left/right asymmetries and the root scale enters children inverted);
* robust loss, part centroids, the SDF stage: no measurable gain;
* betas are weakly identified: report bone lengths and body length in micrometres, not betas;
* with noisy proposals (> ~1 % BL) a weak joint term in S1 is better; with good proposals a strong one.

Real specimens (12 QA): the fit reproduces the proposals (every joint within 0.5 % BL of its proposal), so proposal
correspondence errors (adjacent-leg swaps, a leg inside the gaster) pass into skeleton and morphometrics, and the
deform stage hides them (final surface < 2 % BL off on all 12, failures included). Visual check: 3 of 12 with all six
legs on the right scan legs. Consequences: fix leg identity in the proposals, add an
automatic leg-identity check (each chain must lie in the appendage component attached at its coxa), use pre-deform
metrics and ventral views with one colour per leg for QA, report body length in a straightened pose, publish segment
lengths only for chains that pass the identity check, and fit only proofread (complete) specimens.

The minimal patch of fitter_3d (utils.py load_meshes return_norm, trainer.py log_scale / applied rotation_mask /
pivots / joint term / chamfer source mask / param groups, joint_init.py, optimise.py --joints_dir,
antscan_joints_cfg.yaml; all defaults keep the previous behaviour) is the joint path in the repository. Runtime: batch
about 40 specimens per run (5 s per specimen batched, about 55 s alone).

## 6. First end-to-end run on complete workers (2026-11-10)

Joint proposals of an early generation of the joint method -> prototype registration (the schedule of section 5
without deformation stages, legs with confidence < 0.3 unweighted; frames from the joint method, the legacy frame where
it was unsure). 88 of the 103 curator-kept workers fitted in 96 s on one GPU (batches of 44); 15 waited for frame
review. Visual check of 12 specimens: frames 12/12 correct, trunk on the scan trunk everywhere, 45 of 62 weighted leg chains on their own scan
leg, 7 wrong; surface-only legs mostly wrong (8/10). Placement and pose are usable now.

Morphometrics are NOT usable yet (the fit reproduces the proposals: limb lengths within 0.2 %, scale within 0.3 %):
- tibia-tarsus joints sit 70-440 um down the basitarsus on 11 of 12 checked legs (tibia +42 %, tarsus -29 %, even at
  confidence 0.99). Cause, checked on the template: OmniAnt25's tibia is thickest at
  mid-length and narrowest at the ti joint, while real ant tibiae widen distally, so the fit slides the joint distally;
- the waist joints b_a_2/b_a_3 sit 8-16 % of body length in front of the scan's petiole/postpetiole constrictions at
  confidence 0.8-1.0;
- "straightened" body length (rest-pose X extent) is not straight (the rest pose raises the petiole and tilts the
  gaster): about -6.5 %, -13 % overall against the scan's own articulation-to-articulation length;
- left/right asymmetry cannot be measured (symmetric shape space, proposals are the symmetric fit's pivots).

Fixes, in order: measure segment lengths on the scan itself (medial paths + constrictions) so lengths are independent
of the model; place ti at the scan's tibia end and the waist joints at the trunk's radius minima in the proposals;
define body length as the sum of articulation-to-articulation chords; gate leg identity on femur + distal joints lying on
the scan leg from the chain's own coxa; consider an OmniAnt template revision of the tibia shape.
