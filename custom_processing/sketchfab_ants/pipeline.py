"""End-to-end pipeline for tunosemi's Sketchfab ant models: list, download, clean, orient, verify, fit, evaluate.

Run from the repository root in the `pytorch3d` conda environment:

    python custom_processing/sketchfab_ants/pipeline.py <step> --data_dir DATA [options]

Steps (each can be re-run; `all` runs download -> process -> orient -> verify -> fit -> evaluate):

    list       query Sketchfab for tunosemi's models, select the ants, compare with the pinned list
    download   download the downloadable ant models (glb) into DATA/raw/<key>/   (needs an API token)
    process    Blender: fuse shells, decimate to ~100k vertices, light smoothing -> DATA/intermediate_source_axes/
    orient     rotate every mesh into the OmniAnt frame (head +X, dorsal +Z)       -> DATA/processed/
    review     Blender renders + contact sheets of DATA/processed for checking orientation by eye
    verify     compare the processing and orientation results with expected/processing_summary.csv
    fit        fitter_3d/optimise.py with the ants_cfg.yaml stages             -> DATA/fit3d_results_OmniAnt/
    evaluate   surface metrics of the fit vs. the targets, compared with expected/baseline_fitter3d_metrics.csv

By default the pinned model list (tunosemi_ant_models.json, 140 ants, 135 downloadable, listed 2026-11-14) is
used, which reproduces the processed set of the 2026-11-15 OmniAnt fit. `--fresh_list` uses
DATA/metadata/tunosemi_ant_models.json written by `list` instead, to include species published since.
See README.md in this folder.
"""
import argparse
import csv
import glob
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from ant_taxa import is_ant  # noqa: E402

PINNED = os.path.join(HERE, "tunosemi_ant_models.json")
EXPECTED = os.path.join(HERE, "expected", "processing_summary.csv")
STAGES = {"pose_shape": "Stage_1_default", "deform_fine": "Stage_3_deform_fine"}


class Layout:
    def __init__(self, data_dir):
        self.root = os.path.abspath(data_dir)
        self.metadata = os.path.join(self.root, "metadata")
        self.raw = os.path.join(self.root, "raw")
        self.intermediate = os.path.join(self.root, "intermediate_source_axes")
        self.processed = os.path.join(self.root, "processed")
        self.diagnostics = os.path.join(self.root, "diagnostics")
        self.configs = os.path.join(self.root, "configs")
        self.results = os.path.join(self.root, "fit3d_results_OmniAnt")


def run(cmd, log=None, **kw):
    """Runs a command; with `log`, its output also goes to that file (report/analyse_fit.py reads the fit log)."""
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, check=True, **kw)
        return
    with open(log, "wb") as fh, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw) as p:
        for chunk in iter(lambda: p.stdout.read1(65536), b""):
            fh.write(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def find_blender(arg):
    candidates = [arg, os.environ.get("BLENDER"), shutil.which("blender")]
    if platform.system() == "Darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise SystemExit("Blender not found: pass --blender PATH or set BLENDER (Blender 4.5 LTS was used)")


def load_models(a, L):
    if a.fresh_list:
        path = os.path.join(L.metadata, "tunosemi_ant_models.json")
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing: run the `list` step first")
        return json.load(open(path))
    return json.load(open(PINNED))["models"]


def step_list(a, L):
    from sketchfab import list_user_models, model_key

    rows = list_user_models("tunosemi")
    ants = [dict(r, key=model_key(r["name"], r["uid"])) for r in rows if is_ant(r["name"])]
    os.makedirs(L.metadata, exist_ok=True)
    json.dump(rows, open(os.path.join(L.metadata, "tunosemi_all_models.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(ants, open(os.path.join(L.metadata, "tunosemi_ant_models.json"), "w"), indent=1, ensure_ascii=False)
    pinned = {m["uid"]: m for m in json.load(open(PINNED))["models"]}
    new = [m for m in ants if m["uid"] not in pinned]
    gone = [m for u, m in pinned.items() if u not in {x["uid"] for x in ants}]
    print(f"{len(rows)} models, {len(ants)} ants ({sum(m['downloadable'] for m in ants)} downloadable)")
    print(f"not in the pinned list: {[m['name'] for m in new]}")
    print(f"pinned but no longer listed: {[m['name'] for m in gone]}")
    unknown = sorted({r["name"].split()[0] for r in rows if "ant" in [t.lower() for t in r["tags"]] and not is_ant(r["name"])})
    if unknown:
        print(f"tagged 'ant' but genus not in ant_taxa.py (check and add if they are ants): {unknown}")


def step_download(a, L):
    from sketchfab import download_models, read_token

    counts = download_models(load_models(a, L), L.raw, read_token(a.token_file))
    print("download:", counts)


def step_process(a, L):
    run([sys.executable, os.path.join(REPO, "custom_processing", "batch_process_models.py"),
         "--input_dir", L.raw, "--output_dir", L.intermediate, "--blender_path", find_blender(a.blender),
         "--script_path", os.path.join(REPO, "custom_processing", "prepare_sketchfab_models_for_mesh_fitting.py"),
         "--extensions", ".glb", "--num_processes", a.jobs])
    run([sys.executable, os.path.join(HERE, "summarise_stats.py"), L.intermediate,
         os.path.join(L.diagnostics, "processing_stats.csv")])


def step_orient(a, L):
    run([sys.executable, "-m", "custom_processing.orient_meshes_to_template", "--input_dir", L.intermediate,
         "--output_dir", L.processed, "--diagnostics_dir", L.diagnostics, "--smal_file", a.smal_file], cwd=REPO)


def step_review(a, L):
    thumbs = os.path.join(L.diagnostics, "thumbnails")
    run([find_blender(a.blender), "--background", "--factory-startup", "--python",
         os.path.join(HERE, "render_thumbnails.py"), "--", L.processed, thumbs, 420])
    run([sys.executable, os.path.join(HERE, "contact_sheets.py"), thumbs,
         os.path.join(L.diagnostics, "contact_sheet"), 24, os.path.join(L.diagnostics, "orientation_scores.csv")])


def step_verify(a, L):
    expected = {r["key"]: r for r in csv.DictReader(open(EXPECTED))}
    scores = os.path.join(L.diagnostics, "orientation_scores.csv")
    ori = {r["name"]: r for r in csv.DictReader(open(scores))} if os.path.exists(scores) else {}
    problems, checked = [], 0
    for key, e in expected.items():
        sp = os.path.join(L.intermediate, f"{key}_stats.json")
        if not os.path.exists(sp):
            problems.append(f"{key}: not processed")
            continue
        s = json.load(open(sp))
        h = s["final_health"]
        checked += 1
        if h["components"] != 1 or not h["watertight"]:
            problems.append(f"{key}: {h['components']} components, watertight={h['watertight']}")
        dv = abs(s["final_vertices"] - int(e["final_vertices"])) / int(e["final_vertices"])
        if dv > 0.02:
            problems.append(f"{key}: {s['final_vertices']} vertices, expected {e['final_vertices']}")
        if abs(s["remesh"]["voxel_size_rel"] - float(e["voxel_size_rel"])) > 1e-6:
            problems.append(f"{key}: voxel size {s['remesh']['voxel_size_rel']:.5f}, expected {e['voxel_size_rel']}")
        if key in ori and ori[key]["rotation"] != e["rotation"]:
            problems.append(f"{key}: rotation {ori[key]['rotation']}, expected {e['rotation']}")
        if not os.path.exists(os.path.join(L.processed, f"{key}.obj")):
            problems.append(f"{key}: no oriented mesh in processed/")
    extra = sorted(os.path.basename(p)[:-11] for p in glob.glob(os.path.join(L.intermediate, "*_stats.json"))
                   if os.path.basename(p)[:-11] not in expected)
    print(f"checked {checked}/{len(expected)} expected models; {len(extra)} not in the expected list {extra[:5]}")
    print("all match" if not problems else f"{len(problems)} differences:\n  " + "\n  ".join(problems))
    if problems and not a.allow_differences:
        raise SystemExit(1)


def step_fit(a, L):
    import yaml

    cfg = yaml.safe_load(open(os.path.join(REPO, "fitter_3d", "ants_cfg.yaml")))
    cfg.setdefault("args", {})["results_dir"] = L.results  # the YAML args block overrides CLI flags
    os.makedirs(L.configs, exist_ok=True)
    cfg_path = os.path.join(L.configs, "ants_cfg_tunosemi.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    fit_args = ["--mesh_dir", L.processed, "--yaml_src", cfg_path]
    import torch

    log = os.path.join(L.root, "fit3d_OmniAnt_run.log")
    os.makedirs(L.diagnostics, exist_ok=True)
    if torch.cuda.is_available() and not a.cpu:
        run([sys.executable, "-m", "fitter_3d.optimise", *fit_args], log=log, cwd=REPO)
    else:  # pytorch3d's CPU KNN makes the stock chamfer far too slow; see run_fit_cpu.py
        run([sys.executable, os.path.join(HERE, "run_fit_cpu.py"), "--repo", REPO,
             "--probe_out", os.path.join(L.diagnostics, "dense_chamfer_equivalence_PROBE.txt"), "--", *fit_args],
            log=log)


def step_evaluate(a, L):
    for base, stage in STAGES.items():
        npz = os.path.join(L.results, f"{stage}.npz")
        if not os.path.exists(npz):
            print(f"skip {stage}: {npz} missing")
            continue
        run([sys.executable, os.path.join(HERE, "evaluate_registration.py"), "--targets", L.processed, "--npz", npz,
             "--out", os.path.join(L.diagnostics, f"metrics_{stage}.csv"), "--baseline_stage", base])


STEPS = {"list": step_list, "download": step_download, "process": step_process, "orient": step_orient,
         "review": step_review, "verify": step_verify, "fit": step_fit, "evaluate": step_evaluate}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--data_dir", required=True, help="Working folder for downloads and results (several GB)")
    ap.add_argument("--blender", default=None, help="Blender 4.5 executable (else $BLENDER, else `blender`)")
    ap.add_argument("--token_file", default=None, help="Sketchfab API token file (default ~/.sketchfab_token; "
                                                       "$SKETCHFAB_TOKEN takes precedence)")
    ap.add_argument("--jobs", type=int, default=6, help="Parallel Blender processes for `process` (~2 GB RAM each)")
    ap.add_argument("--smal_file", default="3D_model_prep/SMIL_OmniAnt.pkl", help="Template used for orientation")
    ap.add_argument("--fresh_list", action="store_true", help="Use the list written by `list` instead of the pinned one")
    ap.add_argument("--cpu", action="store_true", help="Fit on the CPU even if CUDA is available")
    ap.add_argument("--allow_differences", action="store_true", help="`verify` reports differences without failing")
    a = ap.parse_args()
    L = Layout(a.data_dir)
    steps = ["download", "process", "orient", "verify", "fit", "evaluate"] if a.step == "all" else [a.step]
    for s in steps:
        print(f"\n=== {s} ===", flush=True)
        STEPS[s](a, L)


if __name__ == "__main__":
    main()
