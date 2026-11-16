"""G. Self-diagnosis of a registration. Nothing here rests on the final surface distance (stage D makes it small
whatever happened before).

Per part (owner bone: head, mandibles, scape, funiculus, mesosoma, waist bones, gaster, per leg coxa / femur / tibia /
basitarsus / tarsus) the signals are
  verify           the limb verification verdict (skelreg/verify.py): fail / duplicate -> the part is `unregistered`;
                   contact -> `review` (placed from the joints alone);
  misfit_C         median distance of the part's stage-C vertices (charts and aligned blobs, BEFORE the free
                   refinement) to the scan surface, in local radii: how well the constrained model explains the scan;
  observed         share of the part's vertices with compatible scan surface (status 0);
  area_ratio       scan area labelled with the part / registered area of the part's observed faces;
  moved_D          median displacement of stage D in local radii (chart distortion);
  seam             p95 |log| of the edge-length ratio (final / stage B) over edges that cross the part's seams;
  flips            faces flipped against the stage-C shape;
  asym             |log| of the left / right length ratio of the segment (mirrored limbs must agree);
  inner_step       largest radius step INSIDE a traced segment (tubes.inner_step). Real development workers: with annotated joints
                   the 95th percentile is 0.37 (tibia), 0.22 (basitarsus), 0.61 (femur); with round-3 joints, whose
                   tibia-tarsus joint lies down the basitarsus, the tibia median is 0.67. Above STEP_REVIEW: `review`;
  joint_feature    for limb joints between two charted tubes: distance (in % of the two segments' length) between the
                   input joint and the nearest local minimum of the scan's radius profile along the two traced tubes
                   (the articulation as the scan shows it); None when the profile has no minimum near the joint.
Specimen level: leg order (coxa bases front to back along the frame's X on each side), side (left coxae on +Y of
their right partners), unexplained / unresolved scan area, limbs unregistered.
Status: part `ok` / `review` / `unregistered` / `failed`; specimen `ok` / `review` / `failed`, with reasons.
"""

import numpy as np

MISFIT_REVIEW = 0.35  # median stage-C misfit in local radii above which a part is `review`
MISFIT_FAIL = 1.0
OBS_REVIEW = 0.5  # observed share below which a part is `review`
AREA_LO, AREA_HI = 0.5, 2.0
ASYM_REVIEW = 0.25  # |log(L / R)| of a limb segment length
JF_REVIEW = 0.15  # joint vs radius-profile minimum, share of the summed length of the two segments
UNEXPL_REVIEW = (
    0.08  # unexplained share of the scan area (reported, not a status reason: it follows from unregistered parts)
)
REVIEW_SHARE = 0.25  # specimen `review` when more than this share of the registered limb segments is not `ok`
STEP_REVIEW = 0.5  # radius step inside a traced segment, |log| over 12.5 % of its length (tubes.inner_step)
# (the TEMPLATE's own limbs have radius steps above 0.5 (tibia thickest mid-length), so on template-shaped synthetic ants
# this signal flags every specimen)
TRUNK = ("b_h", "b_t", "b_a_1", "b_a_2", "b_a_3")


def joint_feature(tpl, st):
    """Per limb joint between two used tubes: signed offset of the nearest radius minimum of the scan profile from
    the joint (+ = distal), as a share of the summed length of the two segments; None without a minimum."""
    by_bone = {s.bone: s for s in st if s.use and s.tube is not None and s.tube.rmean is not None}
    out = {}
    for s in by_bone.values():
        c = s.child
        if c not in by_bone:
            continue
        d = by_bone[c]
        r = np.concatenate([s.tube.rmean, d.tube.rmean[1:]])
        x = np.concatenate(
            [np.linspace(-s.tube.length, 0, len(s.tube.rmean)), np.linspace(0, d.tube.length, len(d.tube.rmean))[1:]]
        )
        tot = s.tube.length + d.tube.length
        win = (x > -0.35 * s.tube.length) & (x < 0.35 * d.tube.length)
        rs = np.convolve(r, np.ones(3) / 3, mode="same")
        idx = np.where(win[1:-1] & (rs[1:-1] <= rs[:-2]) & (rs[1:-1] <= rs[2:]))[0] + 1
        if not len(idx):
            out[tpl.names[c]] = None
            continue
        k = idx[np.argmin(np.abs(x[idx]))]
        out[tpl.names[c]] = float(x[k] / tot)
    return out
