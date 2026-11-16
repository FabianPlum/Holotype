# AntScan joint proposals: comparison of methods (2026-11-09 to 2026-11-10)

How the joint-proposal method for the AntScan scans was chosen. Three independent methods were built against one
common specification (input: the envelope of Stage 1 and the frame of Stage 2a; output: the joint-product contract
below) and evaluated independently (metrics and methodology; two visual ratings on real scans). The combined method
then went through several rounds; this page covers rounds 1 to 4 in detail and rounds 6 and 7 briefly (last
section). The method in the repository, `fitter_3d/antscan_joints` (round 7, four-state ensemble; layout, data files
and provenance in its `README.md`, use in `docs/antscan_workflow.md`), is not a further development of rounds 1 to 4:
round 6 was a new design that used round 3 only as an optional prior and fallback, and the production run used no
round-3 input.

Joint-product contract per specimen (`<name>_joints.json`): 55 joint names, `joints_scan` / `joints_canonical` in µm
or `null`, `confidence`, `completeness` (`missing_joints`, `complete`, `flags`), `frame`, `status`. (The products of
rounds 6 and 7 add `head_front`, `gaster_tip`, `body_length_um` and per-joint details; the registration reads that
contract, `docs/antscan_registration_pipeline.md`, "Terms".)

## Methods

| | A_skeleton | B_modelfit | C_hybrid |
|---|---|---|---|
| idea | voxel curve skeleton + junction graph; appendage paths; template-calibrated body joints; no model fit | OmniAnt25 registered to the envelope (torso stage with 16 orientation hypotheses, limb beam search); joints = fitted pivots | geometric landmarks (voxel core, mesh geodesics) + constrained OmniAnt25 fit with Hungarian region/tip correspondences |
| runtime alone | 5-13 s CPU | 28-41 s GPU | 14-18 s GPU |
| deterministic | yes | no (frame flipped in 1 of 4 reruns on Atta) | yes (0.26 % BL jitter) |

## Synthetic set (40 posed OmniAnt instances, GT known; development set = test set for all three methods)

| | A | B | C |
|---|---|---|---|
| joint error median / p90 (% body length) | 1.10 / 4.43 | 0.32 / 0.88 | 0.31 / 0.81 |
| joints within 2 % | 70.6 % | 97.7 % | 96.4 % |
| frame correct | 40/40 | 40/40 | 39/40 |
| completeness precision / recall | 0.81 / 0.93 | 0.69 / 0.71 | 1.00 / 0.75 |

## Real specimens (47 workers; frames scored against frames judged by eye)

| | A | B | C | cue module (Stage 2a) |
|---|---|---|---|---|
| head direction correct (< 30 deg) | 33/47 | 36/47 | 37/47 | 40/47 |
| dorsal correct (< 50 deg) | 40/47 | 39/47 | 36/47 | 35/47 |
| wrong frames carrying an orientation flag | 3/16 | 12/15 | 3/16 | 15/16 |

Where B's and C's frames agree, their joints still sit 7.2 % body length apart (median; 0.44 % on synthetic) and
40 % of limb chains lie on different physical limbs. Real completeness claims come from one method only in 69 of 75
cases. Visual ratings: average leg placement below 1.5 of 3 for every method; A most precise on spread legs and
antennae when its frame is right, B best on torso/waist (follows curled gasters), C most reliable frame among the
three but fitted chains leave the scan most often (bone segments on the mesh 0.72 vs 0.84/0.85).

Note: the three methods ran before the Stage 1 outer-part fix. Of the three specimens re-enveloped afterwards only
Brachymyrmex_depilis changed in anatomy (two detached right legs restored); its raw scan has no gaster at all, so it is
excluded. Carebara_affinis and Agraulomyrmex_meridionalis had dropped a card mount and a soil grain, correctly.

## Verdict

No method is usable alone, and synthetic scores do not predict real behaviour. All three evaluations rank C first
and recommend a combination:

1. **Frame**: C's multi-hypothesis fit as a candidate, cross-checked with the cue module and A/B; any disagreement
   or low margin goes to a visual frame review (a tool of the production run, not part of this repository). The
   reviewed frame is the input to everything downstream.
2. **Trunk joints** (neck, waist, gaster chain): B's torso stage or C, which agree; export part centroids for the
   registration's trunk term.
3. **Limbs and antennae**: A's skeleton branch paths (joints inside the limbs, tips followed) as correspondence
   targets in C's constrained fit; anatomical gating (legs start at coxae on the mesosoma, never enter the head;
   mandibles and antennae assigned on the head first); every bone segment must lie on the scan; no duplicate tips,
   no collapsed chains.
4. **Confidence**: cross-method agreement within 0.06 body length per joint; signed distance of joints to the
   envelope; honest nulls instead of invented chains.
5. **Completeness**: C's support test plus A's truncation candidates, confirmed only when two methods agree;
   torso-level checks (missing gaster/petiole) using the Stage 1 kept/dropped-part records.
6. **Engineering**: B's batch runner (workers, skip-existing) plus per-specimen timeout and logs; deterministic
   torch; export betas/theta/scale/rotation and the fitted mesh so registration starts from the fit.

The combination was built as a fourth method against the same evaluation, with the real frame truth and the
regenerated meshes as acceptance tests.

## Rounds of the combined method

Round 1: C backbone + A-style centrelines + gates. Round 2: runner fixes, reviewed-frame provenance (honoured only
with a matching mesh md5), anatomical gates on final pivots, leg reorder, trunk landmarks, torso-level completeness
(one-bulb test excludes the gaster-less Brachymyrmex), roll scoring with core symmetry. Held-out test (40 real workers
from unseen genera with eight-panel frame truth; synthetic set v4, seed 5):

| | round 1 | round 2 |
|---|---|---|
| real frames correct | 35/40 | 34/40 |
| wrong frames flagged | 5/5 | 6/6 |
| synthetic median / p90 (% body length) | 0.32 / 0.82 | 0.33 / 0.90 |
| synthetic misses (present joints left null) | 7 | 16 |
| completeness precision / recall | 1.00 / 0.62 | 0.96 / 0.54 |

Round 2's core fit generalises (development and held-out sets agree), but its add-ons tuned on the development set do
not (the neck-landmark move wrecked two held-out necks; some gates null correct legs); it was rejected. Remaining
real-data error classes: leg chains on antennae or on the gaster surface, adjacent-leg swaps, unresolved antennae on
complete workers. Frame review trigger: confidence < 0.55 caught all wrong held-out frames (8 false alarms of 34).
Round 3 adds rule-based fixes (assignment gating by attachment, identity check against the coxa, second assignment
pass, partial-overlap detection) and was tested on a fresh synthetic set (v5, seed 6) and 12 curator-kept workers that
had not been used before.

### Round 3: accepted

Fresh tests that nothing had been tuned on: synthetic v5 (seed 6, 40) and 12 curator-kept complete real workers.

| | round 1 | round 2 | round 3 |
|---|---|---|---|
| synthetic v5 median / p90 (% L) | 0.36 / 1.14 | 0.37 / 1.52 | 0.35 / 0.96 |
| completeness precision / recall | 0.96 / 0.48 | 0.92 / 0.48 | 1.00 / 0.80 |
| trochanter-loss legs detected | 0/4 | 0/4 | 4/4 |
| joints > 5 % L on specimens not sent to frame review | 35 | - | 17 |
| confidence AUC (errors > 5 %) | 0.89 | 0.85 | 0.98 |
| real legs correct / wrong / unresolved (72) | 34 / 33 / 4 | 39 / 21 / 12 | 42 / 15 / 14 |
| real workers with all six legs correct (12) | 1 | 2 | 3 |
| real legs with confidence >= 0.3: correct / wrong | 31 / 21 | 34 / 13 | 35 / 5 |
| real antennae correct (24) | 15 | 13 | 17 |

Conditions: frames with confidence < 0.55 go to review (round 3's frame choice is less stable; Paraponera flipped,
caught); legs with confidence < 0.3 are dropped from the joint loss.

First production run (round 3 on 103 curator-kept workers): 103/103 ok in 26 min (4 workers, one GPU); 22 frames
below confidence 0.55 (to review); 78 % of legs at confidence >= 0.3, 33 specimens with all six legs resolved at that
level; 1 missing-joint claim (false: all 103 are complete); frames agree with the transferred legacy orientation on
all 54 specimens that have a gated legacy frame.

### Round 4: rejected, round 3 stays

Fresh tests (synthetic v6, seed 7; 12 other curator-kept workers). Round 4 decoupled the frame choice from the leg
gates (ungated frame score + 20 degree limb window) and added side-shift, femur-on-head, fore-leg-under-head,
contralateral-tarsus and tarsus-restore rules.

| | round 1 | round 3 | round 4 |
|---|---|---|---|
| synthetic v6 null-aware share < 2 % L | 0.966 | 0.986 | 0.973 |
| v6 joints > 10 % L / misses | 28 / 6 | 4 / 5 | 13 / 12 |
| v6 completeness recall (precision 1.0) | 0.72 | 0.87 | 0.85 |
| real legs correct / wrong / unresolved (72) | 47 / 23 / 2 | 57 / 6 / 9 | 56 / 6 / 10 |
| real legs kept after the filter: correct / wrong | 45 / 19 | 52 / 6 | 49 / 4 |
| real antennae correct (24) | 20 | 21 | 21 |

The decoupled frame choice caused the regressions (v6 0027 and 0036 with confident swapped antennae; real Lasius
rolled 44 degrees). The fore-leg gate (2/2 correct) and tarsus restore (1/1) work and were candidates to port onto
round 3 as isolated changes, together with a within-side leg-order and left/right symmetry check (the hind leg's
branch is the longest and attaches furthest back), which would also make leg confidence reflect identity errors.

## Rounds 6 and 7: the method in the repository

Round 6 started again from the trunk. TRUNK finds the trunk path, the head end, the trunk joints and a working frame
on a voxel grid of the envelope. Three independent approaches trace the limbs: PATHS-A (geodesic paths on the
surface), PATHS-B (curve skeleton of the solid) and PATHS-C (tube decomposition). ALONG then places the joints along
the best-supported centreline of each limb, with gradient-boosting classifiers trained on annotated centrelines.
Round 7 adds a tracer for thin limb tips (THIN), a step for folded knees, leg tips and antennal elbows (FOLDS), and an
ensemble over the four sign states of TRUNK's working frame. The code is `fitter_3d/antscan_joints`.

Development set (21 annotated scans; ALONG scored leave-one-specimen-out, but the design was tuned on these scans, so
the numbers are optimistic):

| | round 6 (one state) | round 7 (four-state ensemble) |
|---|---|---|
| limb paths right, of 178 | 150 | 157 |
| limb identity right, of 182 | 161 | 160 |
| limb joints more than 3 % BL off | 37 | 28 |
| confidence AUROC (wrong against right joints) | 0.827 | 0.875 |

Round 7's 3D joint error on these scans has a median of 0.39 % BL and a 90th percentile of 1.35 % BL. The ensemble
adds no limb paths over a single state; its gain is in the confidence values. On 27 other annotated scans, which had
been used once before to score round 6, round 7 placed more limb paths right and had about half as many confident
joints far off, and both rounds had the head end and left / right right on all 27. On a synthetic set of 30 posed
template meshes, each fused into one closed surface like a scan, both rounds have a median joint error of 1.5 % BL.
The final test on annotated scans that were never used is still to come; it will be run once, on the frozen
pipeline.
