"""Batch driver for the 2024 AntScan pipeline.

Stages (per specimen folder <in>/<Name>_<code>/<Name>_<code>.stl + .json):
  envelope  Blender: outer envelope, snap-back, vertex budget (custom_processing/antscan_mesh_prep.py)
            -> <out>/<name>/<name>_envelope.stl/.json
  align     canonical orientation from anatomical cues (custom_processing/antscan_align.py)
            -> <out>/<name>/<name>_frame.json (+ <name>_canonical.stl); status ok | review
  proofread_sheets  four-view proofreading sheet per specimen (aligns first if needed), custom_processing/antscan_proofread.py

Only usable specimens are processed: caste == worker, family_or_other == Formicidae, and no
problems_other note that means the scan is fragmented or truncated (EXCLUDE_PROBLEMS). Skipped specimens are
listed in the manifest with the reason.

Scans that the field of view cut (open surface in a bounding-box face plane) are excluded by the envelope stage
itself (status "excluded" in <name>_envelope.json, no mesh written).

Each Blender call runs with --python-exit-code 1 (a script error is a failure, not exit 0), a timeout and its
own log file; finished specimens are skipped unless --force. Results are appended to <out>/pipeline_manifest.jsonl.

Run (pytorch3d env, repo root):
  python -m custom_processing.antscan_pipeline --in DATA --out OUT \
      [--stage envelope|align|proofread_sheets] [--jobs 8] [--limit N] [--names A,B] [--force] [--blender PATH]
  DATA = the downloaded AntScan meshes, one <name>/<name>.stl + <name>.json per specimen. Blender: --blender, else
  the environment variable BLENDER, else `blender` on the PATH.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BLENDER = os.environ.get("BLENDER", "blender")  # Blender 4.2 executable: env var BLENDER, else on PATH
EXCLUDE_PROBLEMS = re.compile(r"truncat|only head|split|detached|damaged", re.I)


def find_specimens(root):
    out = []
    for d in sorted(os.listdir(root)):
        stl = os.path.join(root, d, d + ".stl")
        js = os.path.join(root, d, d + ".json")
        if os.path.isfile(stl):
            out.append((d, stl, js if os.path.isfile(js) else None))
    return out


def usable(meta):
    """(ok, reason) from the specimen metadata."""
    if meta is None:
        return False, "no metadata json"
    if meta.get("caste") != "worker":
        return False, f"caste {meta.get('caste')!r}"
    if meta.get("family_or_other", "Formicidae") != "Formicidae":
        return False, f"family {meta.get('family_or_other')!r}"
    if EXCLUDE_PROBLEMS.search(meta.get("problems_other", "") or ""):
        return False, f"problems {meta.get('problems_other')!r}"
    return True, ""


def run_envelope(name, stl, out_root, blender, threads, timeout, force, prep_args=()):
    d = os.path.join(out_root, name)
    done = os.path.join(d, f"{name}_envelope.json")
    if os.path.isfile(done) and not force:
        prev = json.load(open(done))
        return {
            "name": name,
            "stage": "envelope",
            "status": "skipped_existing" if prev.get("status", "ok") == "ok" else "excluded",
            "reason": prev.get("reason"),
        }
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, f"{name}_envelope.log")
    # optional per-specimen options of antscan_mesh_prep.py, written by hand, e.g. {"keep_external": "none"}
    ov_path = os.path.join(d, f"{name}_overrides.json")
    overrides = []
    if os.path.isfile(ov_path):
        for k, v in json.load(open(ov_path)).items():
            if k in ("note", "by", "time", "reason"):  # bookkeeping fields, not parameters
                continue
            overrides += ["--" + k.replace("_", "-"), str(v)]
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "-t",
        str(threads),
        "--python",
        os.path.join(HERE, "antscan_mesh_prep.py"),
        "--",
        stl,
        d,
        *prep_args,
        *overrides,
    ]
    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = tempfile.mkdtemp(prefix="blender_xdg_")  # a stalled gvfs mount can hang Blender at startup
    t0 = time.time()
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec = {
        "name": name,
        "stage": "envelope",
        "status": "ok" if rc == 0 and os.path.isfile(done) else f"failed ({rc})",
        "seconds": round(time.time() - t0, 1),
        "log": log,
    }
    info = json.load(open(done)) if rec["status"] == "ok" else None
    if info is not None and info.get("status") == "excluded":
        rec.update(status="excluded", reason=info.get("reason"))
        for suffix in (
            "_envelope.stl",
            "_canonical.stl",
            "_frame.json",
            "_review_sheet.png",
            "_review_panels.json",
        ):  # stale outputs of an earlier run
            stale = os.path.join(d, name + suffix)
            if os.path.isfile(stale):
                os.remove(stale)
    elif info is not None:
        rec.update(
            vertices=info["output"]["vertices"],
            removed_area_share=info["fidelity_scan_voxels"]["scan_area_share_removed_gt2vox"],
            output_to_scan_p99_vox=info["fidelity_scan_voxels"]["output_to_scan"]["p99"],
        )
    return rec


def run_align(name, out_root, force, write_mesh=True):
    """Stage 2a in-process (numpy/scipy only): frame JSON + canonical STL."""
    from custom_processing import antscan_align

    d = os.path.join(out_root, name)
    env_json = os.path.join(d, f"{name}_envelope.json")
    done = os.path.join(d, f"{name}_frame.json")
    if not os.path.isfile(env_json):
        return {"name": name, "stage": "align", "status": "skipped_no_envelope"}
    env = json.load(open(env_json))
    if env.get("status", "ok") != "ok":  # early envelope JSONs carry no status key
        return {"name": name, "stage": "align", "status": "excluded", "reason": env.get("reason")}
    if os.path.isfile(done) and not force:
        prev = json.load(open(done))
        return {"name": name, "stage": "align", "status": "skipped_existing", "frame_status": prev.get("status")}
    t0 = time.time()
    try:
        out, _, _ = antscan_align.align(env["output"]["stl"], d, write_mesh=write_mesh)
    except Exception as ex:
        return {
            "name": name,
            "stage": "align",
            "status": "failed",
            "error": repr(ex)[:300],
            "seconds": round(time.time() - t0, 1),
        }
    return {
        "name": name,
        "stage": "align",
        "status": "ok",
        "frame_status": out["status"],
        "confidence": out["confidence"]["overall"],
        "flags": out["flags"],
        "seconds": round(time.time() - t0, 1),
    }


def run_proofread_sheet(name, out_root, blender, force):
    """Four-view proofreading sheet (antscan_proofread.py); aligns first when the specimen has no frame yet."""
    from custom_processing import antscan_align, antscan_proofread

    d = os.path.join(out_root, name)
    env_json = os.path.join(d, f"{name}_envelope.json")
    if not os.path.isfile(env_json):
        return {"name": name, "stage": "proofread_sheets", "status": "skipped_no_envelope"}
    env = json.load(open(env_json))
    if env.get("status", "ok") != "ok":
        return {"name": name, "stage": "proofread_sheets", "status": "excluded", "reason": env.get("reason")}
    out = os.path.join(d, f"{name}")
    if os.path.isfile(out + "_proofread.png") and not force:
        return {"name": name, "stage": "proofread_sheets", "status": "skipped_existing"}
    fj = os.path.join(d, f"{name}_frame.json")
    try:
        if not os.path.isfile(fj):
            antscan_align.align(env["output"]["stl"], d, write_mesh=False)
        png = antscan_proofread.render_sheet(env["output"]["stl"], fj, out, blender=blender)
    except Exception as ex:
        return {"name": name, "stage": "proofread_sheets", "status": "failed", "error": repr(ex)[:300]}
    return {"name": name, "stage": "proofread_sheets", "status": "ok", "sheet": png}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="envelope", choices=["envelope", "align", "proofread_sheets"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4, help="Blender threads per job")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument(
        "--prep-args", default="", help="extra arguments for antscan_mesh_prep.py, e.g. '--envelope-voxels 0'"
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifest = os.path.join(a.out, "pipeline_manifest.jsonl")
    specs = find_specimens(a.inp)
    if a.names:
        keep = set(a.names.split(","))
        specs = [s for s in specs if s[0] in keep]
    todo = []
    with open(manifest, "a") as mf:
        for name, stl, js in specs:
            ok, why = usable(json.load(open(js)) if js else None)
            if not ok:
                mf.write(json.dumps({"name": name, "stage": "select", "status": "excluded", "reason": why}) + "\n")
                continue
            todo.append((name, stl))
    if a.limit:
        todo = todo[: a.limit]
    print(
        f"[pipeline] stage {a.stage}: {len(todo)} usable specimens of {len(specs)} -> {a.out} ({a.jobs} jobs)",
        flush=True,
    )
    counts = {}
    if a.stage == "align":
        from concurrent.futures import ProcessPoolExecutor as Pool  # CPU-bound numpy/scipy work
    else:
        Pool = ThreadPoolExecutor  # subprocess-bound Blender work
    with Pool(a.jobs) as ex, open(manifest, "a") as mf:
        if a.stage == "envelope":
            futs = [
                ex.submit(run_envelope, n, s, a.out, a.blender, a.threads, a.timeout, a.force, a.prep_args.split())
                for n, s in todo
            ]
        elif a.stage == "align":
            futs = [ex.submit(run_align, n, a.out, a.force) for n, s in todo]
        else:
            futs = [ex.submit(run_proofread_sheet, n, a.out, a.blender, a.force) for n, s in todo]
        for i, fu in enumerate(futs, 1):
            rec = fu.result()
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            counts[rec["status"].split(" ")[0]] = counts.get(rec["status"].split(" ")[0], 0) + 1
            if not rec["status"].startswith(("ok", "skipped", "excluded")) or i % 10 == 0 or i == len(futs):
                print(f"[pipeline] {i}/{len(futs)} {counts} last={rec['name']}:{rec['status']}", flush=True)


if __name__ == "__main__":
    main()
