"""Stage D: global refinement of the chart-transferred template to one coherent surface.

Unknowns: the nV vertex positions V. Reference: V_C, the stage-C shape (charts and aligned blobs, blended).
Per iteration
  1. part-restricted correspondences in both directions (template vertex -> nearest scan vertex and scan vertex ->
     nearest template vertex; a pair is accepted when the normals agree, the scan vertex's part label is the template
     vertex's part or a neighbour of it, and the distance is inside a gate that shrinks over the iterations and never
     exceeds GATE_R local radii of the template vertex);
  2. every template vertex with pairs gets ONE target on its own normal line: y_i = V_i + delta_i n_i, where delta_i
     is the weighted mean point-to-plane offset of its pairs. The data therefore never pulls a vertex along the
     surface (no sliding); radial change is free;
  3. template vertices without pairs have no data. Those that stick out of the scan solid by less than CONT_R local
     radii are pulled back to the surface (one-sided containment); those inside are left to the regulariser;
  4. one sparse SPD system, the same for x, y, z:
        (alpha B + beta L + mu I + W) V = (alpha B + beta L + mu I) V_C + W Y
     B = Lu' Lu (bending of the DISPLACEMENT V - V_C), L the graph Laplacian (membrane of the displacement), W the
     data weights; alpha, beta go from stiff to soft;
  5. on charted tubes the axial component of the displacement is removed (the chart pins t at both joints; nothing
     in stage D may undo that).
At the end, fold-overs against V_C (flipped faces) are relaxed by smoothing the displacement around them.
"""

import numpy as np

K_NN = 8
NDOT = 0.3
ITERS = 8
GATE0, GATE1 = 0.03, 0.01  # gate in shares of the specimen size: first and last iteration
GATE_R = 2.0  # ... and at most this many local radii of the template vertex
ALPHA0, ALPHA1 = 10.0, 0.5  # bending weight of the displacement, first and last iteration
BETA0, BETA1 = 1.0, 0.05  # membrane weight of the displacement
MU = 1e-3  # anchoring to V_C
W_MAX = 3.0  # cap of the data weight of one vertex
W_CONTAIN = 0.3  # weight of the containment pull
CONT_R = 1.0  # containment acts on vertices at most this many local radii outside the scan
SIG_GATE = 0.33  # floor of the robust scale of the offsets, as a share of the current gate
RAY_R = 4.0  # normal-ray correspondences reach this many local radii ...
RAY_MAX = 0.10  # ... and at most this share of the specimen size
W_RAY = 0.5  # weight of a normal-ray target (a nearest-neighbour pair has weight ~1)
RAY_DIRS = np.array([[0.36, 0.48, 0.80], [-0.80, 0.36, 0.48], [0.48, -0.80, -0.36]])


def inside(scan, X):
    """True where X lies inside the scan solid (majority of three rays leaving through a back face)."""
    votes = np.zeros(len(X), int)
    for d in RAY_DIRS:
        dist, face, cosn = scan.first_hit(X, np.repeat((d / np.linalg.norm(d))[None], len(X), 0))
        votes += (np.isfinite(dist) & (cosn > 0)).astype(int)
    return votes >= 2
