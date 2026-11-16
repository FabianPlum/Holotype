"""Post-stage steps of the joint method for one specimen and one grid state, and the ensemble over the states (each call
in its own interpreter, started by run.py):

  python steps.py thin     NAME --sd SD --mesh M [--marker F]
        THIN (method/thin/thin.py run_extend): mesh-native extension of the leg / antenna polylines of every PATHS
        product + discovery of limbs no approach found. Reads SD/trunk, SD/paths_{A,B,C}; writes SD/thin/paths_{A,B,C},
        links SD/thin/trunk. An approach whose extension failed (or the whole step) falls back to the unchanged stage
        product (symlink), so a THIN failure never removes an approach.
  python steps.py folds    NAME --sd SD --src thin --dst final --mesh M [--grid frame|scan] [--marker F]
        FOLDS (method/folds/s7folds.py run): folded knees, farthest leg tip, distal extension from another approach,
        antennal elbow, on the assembled product in SD/<src> (trunk, limbs, proposals, paths_X); writes
        SD/<dst>/{limbs,proposals} and links SD/<dst>/trunk. On failure the <src> product is copied unchanged (flag
        s7_folds_failed). --grid frame (default, the shipped method; patched in memory, method files unchanged): FOLDS'
        local voxel boxes (s7geom.Box: knee and claw geodesics) are built axis-aligned in the specimen's TRUNK working
        frame (trunk.json frame, intrinsic to the shape) instead of the scan frame, whose axes depend on how the scan
        lies ('scan': the decisions of the knee / claw search change under a rigid rotation of the input).
        Lazy knee search (default; --no-lazy-knee = run it on every leg; same joints and polylines, faster): a leg's
        knee-search context (local grid + geodesics, ~20 s per leg) is built only when the round-6 femur placement is
        uncertain (s7folds.s6_fe_uncertain, unchanged); only the evidence record of non-applied knee candidates is
        missing then.
  python steps.py ensemble --names a,b | --names-file F --work W --states s0,s1,s2,s3 --sub final --out OUT
        ENSEMBLE (method/ensemble/ensemble.py): the per-state products W/states/<k>/<sub> -> one product (state vote
        per limb, joint medoid over agreeing states, spread confidence) in OUT/{proposals,limbs}.
Nothing here reads annotations or specimen names other than as file keys.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback

try:  # imported as fitter_3d.antscan_joints.steps (run.py) or run as a script next to stage.py
    from .stage import setup_env
except ImportError:
    from stage import setup_env

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "method")


def _link(src, dst):
    """dst becomes a relative symlink to src, so that the output folder can be moved (an existing link or folder at dst
    is removed first; a link is never followed)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if os.path.islink(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    os.symlink(os.path.relpath(os.path.realpath(src), os.path.realpath(os.path.dirname(dst))), dst)


def _ok(f):
    """True when the json file f exists and has status 'ok'."""
    try:
        return json.load(open(f)).get("status") == "ok"
    except Exception:
        return False


def thin_step(name, sd, mesh):
    out = os.path.join(sd, "thin")
    rec = {"step": "thin", "name": name, "status": "ok", "error": None, "fallback": []}
    t0 = time.time()
    try:
        sys.path.insert(0, os.path.join(V, "thin"))
        import thin

        s = thin.run_extend(name, sd, mesh, out)
        rec["summary"] = {
            a: {k: v for k, v in (x or {}).items() if k in ("extended", "restatus", "error", "s6_status")}
            for a, x in (s.get("approaches") or {}).items()
        }
        rec["discovery"] = {
            k: v
            for k, v in (s.get("discovery") or {}).items()
            if k in ("missing_before", "named", "inserted_into", "error")
        }
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
    # fallbacks: the trunk link, and every approach whose stage product is ok but whose extension is not
    if not os.path.lexists(os.path.join(out, "trunk", name)):
        _link(os.path.join(sd, "trunk", name), os.path.join(out, "trunk", name))
    for a in "ABC":
        src = os.path.join(sd, f"paths_{a}", name)
        dst = os.path.join(out, f"paths_{a}", name)
        if _ok(os.path.join(src, "stage_done.json")) and not _ok(os.path.join(dst, "stage_done.json")):
            _link(src, dst)
            rec["fallback"].append(a)
        elif not os.path.lexists(dst) and os.path.lexists(src):
            _link(src, dst)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def frame_box(G, R, o):
    """s7geom.Box built in the frame x_f = R (x - o): the box is axis-aligned in that frame; ijk() / xyz() take and
    return scan coordinates, so every caller is unchanged."""
    import numpy as np

    R, o = np.asarray(R, float), np.asarray(o, float)

    class FrameBox(G.Box):
        frame = {"rotation_scan_to_frame": R.tolist(), "origin_scan": o.tolist()}

        def __init__(self, tri, pts, margin, pitch, max_vox=60e6):
            super().__init__((np.asarray(tri, float) - o) @ R.T, (np.atleast_2d(pts) - o) @ R.T, margin, pitch, max_vox)

        def ijk(self, P):
            return super().ijk((np.atleast_2d(P) - o) @ R.T)

        def xyz(self, ijk):
            return super().xyz(ijk) @ R + o

    return FrameBox


def lazy_knees(F):
    """Patch s7folds (module F) so that a leg's knee-search context is only built when the round-6 femur is uncertain."""
    Orig, orig_fc = F.LegContext, F.fold_correction

    class LazyLegContext:
        def __init__(self, ch, *args, **kw):
            self.ch = ch
            self._a, self._kw, self._real = (ch,) + args, kw, None

        def __getattr__(self, k):  # only for attributes not set above: build the real context on first use
            if k.startswith("__"):
                raise AttributeError(k)
            if self.__dict__.get("_real") is None:
                self.__dict__["_real"] = Orig(*self.__dict__["_a"], **self.__dict__["_kw"])
            return getattr(self.__dict__["_real"], k)

    def fold_correction(ctx, prop, cfg=F.FOLD):
        unc, _ = F.s6_fe_uncertain(prop, ctx.ch, cfg)
        if not unc:
            return None
        return orig_fc(ctx, prop, cfg)

    F.LegContext, F.fold_correction = LazyLegContext, fold_correction


def folds_step(name, sd, src, dst, mesh, grid="frame", lazy=True):
    s, d = os.path.join(sd, src), os.path.join(sd, dst)
    rec = {"step": f"folds:{src}->{dst}", "name": name, "status": "ok", "error": None, "grid": grid, "lazy_knee": lazy}
    t0 = time.time()
    try:
        sys.path.insert(0, os.path.join(V, "folds"))
        if grid == "frame":
            import s7geom

            fr = json.load(open(os.path.join(s, "trunk", name, "trunk.json")))["frame"]
            s7geom.Box = frame_box(s7geom, fr["rotation_scan_to_canonical"], fr["origin_scan"])
        import s7folds

        if lazy:
            lazy_knees(s7folds)
        r = s7folds.run(name, s, mesh, d, loso=False)
        if grid == "frame":
            f = os.path.join(d, "proposals", name, f"{name}_joints.json")
            P = json.load(open(f))
            P.setdefault("s7", {})["folds_grid"] = "TRUNK working frame"
            P["method"] = P.get("method", "") + " [FOLDS local grids in the TRUNK working frame]"
            json.dump(P, open(f, "w"), indent=1, default=float)
        rec["changes"] = {
            k: v.get("corrected")
            for k, v in (r.get("changes") or {}).items()
            if isinstance(v, dict) and v.get("corrected")
        }
        rec["errors"] = (r.get("changes") or {}).get("errors")
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        for sub in ("limbs", "proposals"):
            a, b = os.path.join(s, sub, name), os.path.join(d, sub, name)
            if os.path.isdir(a):
                if os.path.lexists(b):
                    shutil.rmtree(b)
                shutil.copytree(a, b)
        f = os.path.join(d, "proposals", name, f"{name}_joints.json")
        if os.path.isfile(f):
            P = json.load(open(f))
            if P.get("status") == "ok":
                P.setdefault("completeness", {}).setdefault("flags", []).append(
                    f"s7_folds_failed({rec['error'][:200]})"
                )
                json.dump(P, open(f, "w"), indent=1, default=float)
    _link(os.path.join(s, "trunk", name), os.path.join(d, "trunk", name))
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def ensemble_step(names, work, states, sub, out):
    sys.path.insert(0, os.path.join(V, "ensemble"))
    import ensemble as EN

    st = [(k, os.path.normpath(os.path.join(work, "states", k, sub))) for k in states]
    res = []
    for n in names:
        t0 = time.time()
        try:
            o, lj, arr = EN.ensemble(n, st, s3_dir=None)
        except Exception as e:
            o, lj, arr = (
                {
                    "name": n,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(),
                    "joints_scan": {},
                },
                None,
                None,
            )
        EN.write(out, n, o, lj, arr)
        res.append(
            {"name": n, "status": o.get("status"), "error": o.get("error"), "seconds": round(time.time() - t0, 2)}
        )
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["thin", "folds", "ensemble"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--sd")
    ap.add_argument("--mesh")
    ap.add_argument("--src", default="thin")
    ap.add_argument("--dst", default="final")
    ap.add_argument("--names-file")
    ap.add_argument("--names")
    ap.add_argument("--work")
    ap.add_argument("--states", default="s0,s1,s2,s3")
    ap.add_argument("--sub", default="final")
    ap.add_argument("--out")
    ap.add_argument("--marker", default=None, help="write the step record here (json)")
    ap.add_argument("--grid", default="frame", choices=["scan", "frame"], help="FOLDS local grid frame (see docstring)")
    ap.add_argument(
        "--no-lazy-knee",
        dest="lazy_knee",
        action="store_false",
        help="run the knee search on every leg (default: lazy, skipped where the round-6 femur is certain)",
    )
    a = ap.parse_args()
    setup_env()
    if a.step == "ensemble":
        names = a.names.split(",") if a.names else [x.strip() for x in open(a.names_file) if x.strip()]
        res = ensemble_step(names, a.work, a.states.split(","), a.sub, a.out)
        nf = sum(1 for r in res if r["status"] != "ok")
        print(f"ensemble {a.states} sub={a.sub}: {len(names)} specimens, {nf} failed -> {a.out}", flush=True)
        if a.marker:
            json.dump(res, open(a.marker, "w"), indent=1)
        sys.exit(0)
    if a.step == "thin":
        rec = thin_step(a.name, a.sd, a.mesh)
    else:
        rec = folds_step(a.name, a.sd, a.src, a.dst, a.mesh, a.grid, a.lazy_knee)
    if a.marker:
        os.makedirs(os.path.dirname(a.marker), exist_ok=True)
        json.dump(rec, open(a.marker, "w"), indent=1, default=str)
    print(json.dumps({k: rec.get(k) for k in ("step", "name", "status", "error", "seconds")}), flush=True)
    sys.exit(0 if rec["status"] == "ok" else 3)


if __name__ == "__main__":
    main()
