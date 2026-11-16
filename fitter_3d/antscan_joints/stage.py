"""One stage of the joint method (trunk, paths_A, paths_B, paths_C) for one specimen under one GRID STATE, in its own
interpreter (started by run.py; the method modules use flat imports from their folder under method/).

A grid state = (sign state, phase):
  sign state s0..s3   TRUNK's coarse PCA frame made sign-canonical (each axis signed so that the area-weighted third
                      moment of the face centroids along it is positive, det +1 by flipping axis 3), then the axis pair
                      of state k flipped: s0 (+,+,+) s1 (-,-,+) s2 (-,+,-) s3 (+,-,-). All four have det +1.
  phase (a, b, c)     every voxel grid of the stage moved by (a, b, c) voxels along its own axes (corner =
                      min - (pad + phase) * pitch); phase 0 runs the unpatched voxelisers. The shipped method uses
                      phase 0 only; the other phases are kept for grid-sensitivity experiments.
Both are intrinsic to the shape, so a fixed list of states does not depend on how the scan lies.
Keys: 's0'..'s3' (phase 0); suffix 'b' = (1/2, 1/2, 1/2), 'f1' = (1/2, 1/2, 0), 'f2' = (1/2, 0, 1/2), 'f3' = (0, 1/2, 1/2),
'q' = (1/4, 1/4, 1/4); or explicit 's2@0.25,0.5,0'.

  python stage.py trunk|paths_A|paths_B|paths_C NAME --grid KEY --mesh M --work W
Writes W/<stage>/<NAME>/... and W/<stage>/<NAME>/stage_done.json ({status, seconds, error, grid}); exit code 0 when
the stage succeeded, 3 otherwise. Nothing here reads annotations or specimen names other than as file keys.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD = os.path.join(HERE, "method")
GRID_INFO = {}
STATES = {"s0": (1, 1, 1), "s1": (-1, -1, 1), "s2": (-1, 1, -1), "s3": (1, -1, -1)}
PHASES = {
    "": (0.0, 0.0, 0.0),
    "b": (0.5, 0.5, 0.5),
    "f1": (0.5, 0.5, 0.0),
    "f2": (0.5, 0.0, 0.5),
    "f3": (0.0, 0.5, 0.5),
    "q": (0.25, 0.25, 0.25),
}
STAGES = ("trunk", "paths_A", "paths_B", "paths_C")


def setup_env():
    """Thread counts and numba's compile cache for a stage process (run.py sets them explicitly for its jobs; these are
    the defaults when a stage is started by hand). Two BLAS threads per job, as in the production run."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(k, "2")
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "antscan_joints_numba"))
    sys.dont_write_bytecode = True


def parse_grid(key):
    """'s1b' -> ('s1', (0.5, 0.5, 0.5)); 's2@0.25,0.5,0' -> ('s2', (0.25, 0.5, 0.0))."""
    if "@" in key:
        s, p = key.split("@")
        ph = tuple(float(x) for x in p.split(","))
    else:
        s, ph = key[:2], PHASES.get(key[2:])
    if s not in STATES or ph is None or len(ph) != 3 or not all(0.0 <= x < 1.0 for x in ph):
        raise ValueError(f"bad grid state {key!r}")
    return s, ph


def forced_pca(orig, state):
    """trunk.pca_frame -> the sign-canonical frame with the axis pair of `state` flipped."""
    import numpy as np

    sg = np.asarray(STATES[state], float)

    def f(V, F):
        R, o = orig(V, F)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        w = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        cen = (a + b + c) / 3.0
        R = np.array(R, float, copy=True)
        skew = []
        for i in range(3):
            p = (cen - o) @ R[i]
            m3, m2 = float((w * p**3).sum()), float((w * p**2).sum())
            skew.append(abs(m3 / w.sum() / max(m2 / w.sum(), 1e-30) ** 1.5))
            if m3 < 0:
                R[i] *= -1
        if np.linalg.det(R) < 0:
            R[2] *= -1
        GRID_INFO["pca_skewness"] = [round(s, 4) for s in skew]
        return R * sg[:, None], o

    return f


def shifted_voxelize(parity_counts, phase, centre_lo):
    """The exact parity voxeliser of the stages (identical arithmetic) with the grid corner moved by -phase * h.
    centre_lo: return the centre of voxel 0 (trunk.voxelize) instead of its corner (paths_B vox / paths_C geomlib)."""
    import numpy as np

    ph = np.asarray(phase, float)

    def voxelize(v, f, h, pad=3):
        corner = v.min(0) - pad * h - ph * h
        hi = v.max(0) + pad * h
        n = np.ceil((hi - corner) / h).astype(int)
        g = (v - corner) / h - 0.5
        cnt = parity_counts(g[f].astype(np.float64), int(n[0]), int(n[1]), int(n[2]))
        occ = (np.cumsum(cnt, axis=2)[:, :, :-1] % 2) == 1
        return occ, (corner + 0.5 * h if centre_lo else corner)

    return voxelize


def run_stage(stage, name, grid, mesh, work):
    """Run one stage; returns the stage_done record (also written to W/<stage>/<name>/stage_done.json)."""
    state, phase = parse_grid(grid)
    od = os.path.join(work, stage)
    os.makedirs(os.path.join(od, name), exist_ok=True)
    sys.path.insert(0, os.path.join(METHOD, stage))
    t0 = time.time()
    shift = any(x != 0.0 for x in phase)
    rec = {
        "stage": stage,
        "name": name,
        "status": "ok",
        "error": None,
        "loso": False,
        "grid": {"key": grid, "sign_state": state, "phase_vox": list(phase)},
    }
    try:
        trunk_dir = os.path.join(work, "trunk")
        if stage == "trunk":
            import trunk

            trunk.pca_frame = forced_pca(trunk.pca_frame, state)
            if shift:
                trunk.voxelize = shifted_voxelize(trunk._parity_counts, phase, centre_lo=True)
            trunk.run(name, mesh, od, frame_prior=None)
            f = os.path.join(od, name, "trunk.json")
            d = json.load(open(f))
            d.setdefault("evidence", {})["grid_state"] = {**rec["grid"], **GRID_INFO}
            json.dump(d, open(f, "w"), indent=1, default=float)
        elif stage == "paths_A":
            import paths

            paths.run(name, trunk_dir, mesh, od, {})
            rec["priors"] = "priors_final/priors_all_trunk.json"
        elif stage == "paths_B":
            import paths

            if shift:
                import vox

                vox.voxelize = shifted_voxelize(vox._parity_counts, phase, centre_lo=False)
            paths.run(name, trunk_dir, mesh, od)
        elif stage == "paths_C":
            import paths

            if shift:
                import geomlib

                paths.voxelize = shifted_voxelize(geomlib._parity_counts, phase, centre_lo=False)
            paths.run(name, trunk_dir, mesh, od, contact_model=None)
            rec["contact_model"] = "contact_model.json"
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    rec["seconds"] = round(time.time() - t0, 2)
    json.dump(rec, open(os.path.join(od, name, "stage_done.json"), "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("name")
    ap.add_argument("--grid", default="s0")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    setup_env()
    rec = run_stage(a.stage, a.name, a.grid, a.mesh, a.work)
    print(json.dumps({k: rec[k] for k in ("stage", "name", "status", "error", "seconds")}), flush=True)
    sys.exit(0 if rec["status"] == "ok" else 3)


if __name__ == "__main__":
    main()
