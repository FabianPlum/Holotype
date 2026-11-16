"""THE ONE COMMAND of the AntScan joint method (round 7, four-state ensemble): prepared scan envelopes -> joint
products, limb products and the per-state products that fitter_3d/antscan_registration reads.

    python -m fitter_3d.antscan_joints.run --names NAMES.txt --env ENV_DIR --out OUT_DIR [--jobs N] [options]

  NAMES.txt   one specimen name per line (a file key only; blank lines and '#' lines are skipped)
  ENV_DIR     <name>/<name>_envelope.stl (or <name>_envelope.stl), or a path template containing {name}: the
              prepared outer envelope (scan frame, micrometres; custom_processing/antscan_mesh_prep.py)
  OUT_DIR     output folder:
                proposals/<name>/<name>_joints.json        joint product (registration: --joints OUT_DIR/proposals)
                limbs/<name>/limbs.json, limbs.npz          limb product   (registration: --limbs OUT_DIR/limbs)
                states/<k>/{trunk,paths_A,paths_B,paths_C}/<name>/   stage products of state k (k = s0..s3)
                states/<k>/thin/..., states/<k>/final/{proposals,limbs}/<name>/   per-state products
                states/<k>/steps/, states/<k>/logs/, logs/  markers and one log per job
                jobs.jsonl (one record per job), summary.tsv (one row per specimen of the last run), batch_<time>.json
              The registration finds the per-state products (--joint-states) and the path stages (--stages) next to
              these folders by itself.
options
  --jobs N              concurrent jobs (each one process with 2 BLAS threads; default 2)
  --mem-cap-gb G        memory cap per job (resident memory of the job's process tree): a job above it is killed and
                        recorded as a failed step, like a timeout. Default: 32, or 80 % of the machine's memory divided
                        by --jobs when that is less (so that N jobs at the cap still fit). In the production run (1013
                        specimens x 4 states) the largest successful job held 29.3 GB (a PATHS-A stage of a 21 mm
                        specimen); jobs that grew without bound reached 52 and 69 GB before the system killed them.
  --timeout S           time limit per stage job (trunk, paths_*; default 3600 s); thin / assemble / folds /
                        ensemble have 900 s
  --states LIST         grid states (default s0,s1,s2,s3: the shipped four-state ensemble)
  --cache DIR           numba's compile cache (default OUT_DIR/cache)
  --allow-without-embree  run although trimesh has no embree ray caster (results then differ from the production run)
  --retry-failed        run the steps that failed in an earlier run again (default: keep them as failed)
  --break-lock          take over OUT_DIR although a lock file of another run is there

Resume: every step leaves a marker; a rerun skips finished steps. A step that has to run again (no marker: never
run or interrupted) first removes its old outputs, and every step that depends on it, including the specimen's
ensemble, runs again too, so no product is ever older than what it was made from. A step that failed (error, time
limit, memory cap) is not run again unless --retry-failed is given (then its dependents run again too).
One run at a time writes to OUT_DIR (lock file OUT_DIR/.lock). Ctrl-C / SIGTERM stops the running jobs too.
A specimen whose steps fail still gets a product when the ensemble can use the other states; summary.tsv lists the
failed steps per specimen (the joint product does not carry them).
Exit code: 0 the batch ran; 2 wrong arguments, missing package or OUT_DIR locked; 3 the batch ran and no specimen got a
product with status ok.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from fitter_3d.antscan_common import UsageError, acquire_lock, read_names, release_lock

from . import DEPS, HERE, MODEL_FILE, STAGES, STEPS, VERSION, check_limbs, check_product, has_embree
from . import missing_dependencies
from .steps import _ok

PY = sys.executable
POST_TIMEOUT = 900  # s: thin / assemble / folds / ensemble (production value)
FOLDS_GRID = "frame"  # FOLDS' local grids in the TRUNK working frame (steps.py)
EXIT_OK, EXIT_USAGE, EXIT_NOTHING = 0, 2, 3
GIB = 1024**3
MEM_CAP_GB = 32.0  # upper end of the default memory cap per job (default_mem_cap_gb)


# ------------------------------------------------------------------------------------------------ inputs
def find_envelope(env, name):
    """envelope mesh of `name` under ENV_DIR (see the module docstring), or None"""
    if "{name}" in env:
        cands = [env.format(name=name)]
    else:
        cands = [os.path.join(env, name, f"{name}_envelope.stl"), os.path.join(env, f"{name}_envelope.stl")]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


# ------------------------------------------------------------------------------------------------ markers / outputs
def sdir(work, k):
    return os.path.join(work, "states", k)


def ens_file(work, name):
    return os.path.join(work, "proposals", name, f"{name}_joints.json")


def done(work, k, step, name):
    """True when `step` of `name` in state k has finished with status ok (its marker)."""
    d = sdir(work, k)
    if step in STAGES:
        return _ok(os.path.join(d, step, name, "stage_done.json"))
    if step == "thin":
        return os.path.isfile(os.path.join(d, "steps", f"{name}_thin.json"))
    if step == "assemble":
        return _ok(os.path.join(d, "thin", "proposals", name, f"{name}_joints.json"))
    if step == "folds":
        return _ok(os.path.join(d, "final", "proposals", name, f"{name}_joints.json"))
    raise ValueError(step)


def outputs(work, k, step, name):
    """every file / folder / link that `step` writes for `name` in state k (removed before the step runs again)"""
    d = sdir(work, k)
    if step in STAGES:
        return [os.path.join(d, step, name)]
    if step == "thin":
        return [os.path.join(d, "thin", sub, name) for sub in ("trunk", "paths_A", "paths_B", "paths_C")] + [
            os.path.join(d, "thin", "thin", f"{name}.json"),
            os.path.join(d, "steps", f"{name}_thin.json"),
        ]
    if step == "assemble":
        return [os.path.join(d, "thin", sub, name) for sub in ("limbs", "proposals")]
    if step == "folds":
        return [os.path.join(d, "final", sub, name) for sub in ("limbs", "proposals", "trunk")] + [
            os.path.join(d, "steps", f"{name}_folds.json")
        ]
    raise ValueError(step)


def remove(p):
    """remove a file, a symlink (never followed) or a folder"""
    if os.path.islink(p) or os.path.isfile(p):
        os.remove(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)


def stale_plan(work, names, states, retry_failed=False):
    """(fin, ens_fin): the steps and ensembles that are finished AND up to date. A step without a marker (never run,
    or interrupted) makes every step depending on it (DEPS, transitively) and the specimen's ensemble run again. A
    step that failed (its marker says so: error, time limit, memory cap) counts as finished, since what depends on
    it was made without it; retry_failed runs it again, and its dependents with it."""
    fin = set()
    for n in names:
        for k in states:
            todo = set()
            for s in STEPS:  # STEPS is in dependency order
                st = step_state(work, k, s, n)
                if st == "missing" or (st == "failed" and retry_failed) or any(dd in todo for dd in DEPS[s]):
                    todo.add(s)
            fin |= {(n, k, s) for s in STEPS if s not in todo}
    ens_fin = set()
    for n in names:
        try:
            st = json.load(open(ens_file(work, n))).get("status")
        except (OSError, ValueError):
            st = None
        made = st == "ok" or (st is not None and not retry_failed)
        if made and all((n, k, s) in fin for k in states for s in STEPS):
            ens_fin.add(("ens", n))
    return fin, ens_fin


def cmd_of(work, k, step, name, mesh):
    d = sdir(work, k)
    marker = os.path.join(d, "steps", f"{name}_{step}.json")
    if step in STAGES:
        return [PY, os.path.join(HERE, "stage.py"), step, name, "--grid", k, "--mesh", mesh, "--work", d]
    if step == "thin":
        return [PY, os.path.join(HERE, "steps.py"), "thin", name, "--sd", d, "--mesh", mesh, "--marker", marker]
    if step == "assemble":
        script = os.path.join(HERE, "method", "assemble", "assemble.py")
        return [PY, script, name, "--mesh", mesh, "--work", os.path.join(d, "thin")]
    if step == "folds":
        return [
            PY,
            os.path.join(HERE, "steps.py"),
            "folds",
            name,
            "--sd",
            d,
            "--src",
            "thin",
            "--dst",
            "final",
            "--mesh",
            mesh,
            "--marker",
            marker,
            "--grid",
            FOLDS_GRID,
        ]
    raise ValueError(step)


def record_failure(work, k, step, name, why, dt):
    """a failed marker for a job that was killed or exited without writing its own failed record"""
    d = sdir(work, k)
    if step in STAGES:
        sd = os.path.join(d, step, name)
        os.makedirs(sd, exist_ok=True)
        rec = {"stage": step, "name": name, "status": "failed", "seconds": round(dt, 1), "error": why}
        json.dump(rec, open(os.path.join(sd, "stage_done.json"), "w"), indent=1)
    elif step == "thin":
        os.makedirs(os.path.join(d, "steps"), exist_ok=True)
        json.dump(
            {"step": "thin", "status": "failed", "error": why}, open(os.path.join(d, "steps", f"{name}_thin.json"), "w")
        )
    else:  # assemble, folds
        pd = os.path.join(d, "thin" if step == "assemble" else "final", "proposals", name)
        f = os.path.join(pd, f"{name}_joints.json")
        if why.startswith(("timeout", "memory cap")) or not os.path.isfile(f):
            os.makedirs(pd, exist_ok=True)
            json.dump({"name": name, "status": "failed", "error": f"{step} {why}", "joints_scan": {}}, open(f, "w"))


# ------------------------------------------------------------------------------------------------ jobs
RUNNING = {}  # pid -> Popen of the running jobs (stopped on Ctrl-C / SIGTERM)
RUNNING_LOCK = threading.Lock()


def job_env(cache):
    e = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        e[k] = "2"
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e["NUMBA_CACHE_DIR"] = os.path.join(cache, "numba")
    return e


def tree_rss(pid):
    """resident memory (bytes) of a process and all its descendants; 0 when it is gone"""
    try:
        import psutil

        p = psutil.Process(pid)
        tot = 0
        for q in [p] + p.children(recursive=True):
            try:
                tot += q.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return tot
    except ImportError:
        try:  # Linux without psutil: the job process only
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0
    except Exception:  # NoSuchProcess and friends
        return 0


def _kill(p):
    try:
        if os.name == "posix":
            os.killpg(p.pid, signal.SIGKILL)  # the job is the leader of its own process group
        else:
            p.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_measured(cmd, fh, timeout, cap_bytes, env, poll=0.5):
    """run one job; returns (rc | 'timeout ...' | 'memory cap ...', cpu seconds, peak RSS MB of the waited process
    (os.wait4; None where unavailable), peak RSS MB of the process tree as sampled by the watcher)"""
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=(os.name == "posix"))
    with RUNNING_LOCK:
        RUNNING[p.pid] = p
    hit, peak, stop = {}, [0], threading.Event()
    t0 = time.time()

    def watch():
        while not stop.wait(poll):
            if time.time() - t0 > timeout:
                hit["why"] = f"timeout after {timeout:g} s"
                _kill(p)
                return
            r = tree_rss(p.pid)
            peak[0] = max(peak[0], r)
            if cap_bytes and r > cap_bytes:
                hit["why"] = f"memory cap {cap_bytes / GIB:g} GiB exceeded ({r / GIB:.1f} GiB resident)"
                _kill(p)
                return

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    cpu = maxrss = None
    try:
        if hasattr(os, "wait4"):
            _, status, ru = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            cpu, maxrss = round(ru.ru_utime + ru.ru_stime, 2), round(ru.ru_maxrss / 1024.0, 1)
        else:
            p.wait()
    finally:
        stop.set()
        th.join()
        with RUNNING_LOCK:
            RUNNING.pop(p.pid, None)
    return hit.get("why", p.returncode), cpu, maxrss, round(peak[0] / 2**20, 1)


def log_job(work, r):
    try:
        with open(os.path.join(work, "jobs.jsonl"), "a") as fh:
            fh.write(json.dumps({**r, "t_end": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    except OSError:
        pass


def run_job(work, k, step, name, mesh, opt):
    d = sdir(work, k)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    for p in outputs(work, k, step, name):  # never build on a half-written or stale earlier result
        remove(p)
    log = os.path.join(d, "logs", f"{name}_{step}.log")
    tmo = opt["timeout"] if step in STAGES else POST_TIMEOUT
    t0 = time.time()
    with open(log, "w") as fh:
        rc, cpu, rss, peak = run_measured(cmd_of(work, k, step, name, mesh), fh, tmo, opt["cap"], opt["env"])
    dt = time.time() - t0
    if isinstance(rc, str):
        record_failure(work, k, step, name, rc, dt)
    elif rc != 0 and not done(work, k, step, name):
        record_failure(work, k, step, name, f"exit code {rc} (see {log})", dt)
    r = {
        "state": k,
        "step": step,
        "name": name,
        "rc": rc,
        "seconds": round(dt, 2),
        "cpu_s": cpu,
        "maxrss_mb": rss,
        "tree_peak_mb": peak,
        "ok": done(work, k, step, name),
    }
    log_job(work, r)
    return r


def run_ensemble(work, name, states, opt):
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    for p in (os.path.join(work, "proposals", name), os.path.join(work, "limbs", name)):
        remove(p)
    c = [
        PY,
        os.path.join(HERE, "steps.py"),
        "ensemble",
        "--names",
        name,
        "--work",
        work,
        "--states",
        ",".join(states),
        "--sub",
        "final",
        "--out",
        work,
    ]
    t0 = time.time()
    with open(os.path.join(work, "logs", f"ensemble_{name}.log"), "w") as fh:
        rc, cpu, rss, peak = run_measured(c, fh, POST_TIMEOUT, opt["cap"], opt["env"])
    r = {
        "state": "ens",
        "step": "ensemble",
        "name": name,
        "rc": rc,
        "seconds": round(time.time() - t0, 2),
        "cpu_s": cpu,
        "maxrss_mb": rss,
        "tree_peak_mb": peak,
        "ok": _ok(ens_file(work, name)),
    }
    log_job(work, r)
    return r


def batch(names, meshes, work, states, opt, verbose=True):
    """every step of every state of every specimen, then its ensemble; at most opt['jobs'] jobs at a time, specimen by
    specimen (products appear early). Returns the job records of this run."""
    jobs = [(n, k, s) for n in names for k in states for s in STEPS]
    fin, ens_fin = stale_plan(work, names, states, opt.get("retry_failed", False))
    ens_jobs = [("ens", n) for n in names]
    running, results = set(), []
    lock = threading.Condition()
    t_start = time.time()

    def ready():
        for n in names:
            je = ("ens", n)
            if je not in ens_fin and je not in running and all((n, k, s) in fin for k in states for s in STEPS):
                return je
            for k in states:
                for s in STEPS:
                    j = (n, k, s)
                    if j in fin or j in running:
                        continue
                    if all((n, k, dd) in fin for dd in DEPS[s]):
                        if s in STAGES[1:] and not done(work, k, "trunk", n):
                            for p in outputs(work, k, s, n):
                                remove(p)
                            fin.add(j)  # trunk failed: PATHS cannot run; thin / assemble decide
                            results.append(
                                {
                                    "state": k,
                                    "step": s,
                                    "name": n,
                                    "rc": "skipped (no trunk)",
                                    "seconds": 0,
                                    "ok": False,
                                }
                            )
                            continue
                        return j
        return None

    def worker():
        while True:
            with lock:
                j = ready()
                while j is None:
                    if len(fin) == len(jobs) and len(ens_fin) == len(ens_jobs):
                        lock.notify_all()
                        return
                    lock.wait(timeout=5)
                    j = ready()
                running.add(j)
            try:
                if j[0] == "ens":
                    r = run_ensemble(work, j[1], states, opt)
                else:
                    n, k, s = j
                    r = run_job(work, k, s, n, meshes[n], opt)
            except Exception as e:  # an error of the scheduler itself: record it, go on
                r = {
                    "state": "ens" if j[0] == "ens" else j[1],
                    "step": "ensemble" if j[0] == "ens" else j[2],
                    "name": j[1] if j[0] == "ens" else j[0],
                    "rc": f"{type(e).__name__}: {e}",
                    "seconds": 0,
                    "ok": False,
                }
            with lock:
                running.discard(j)
                (ens_fin if j[0] == "ens" else fin).add(j)
                results.append(r)
                if verbose:
                    print(
                        f"[{time.time() - t_start:7.0f}s] {r['name']:46s} {r['state']:5s} {r['step']:9s} "
                        f"{'ok' if r['ok'] else 'FAILED'} {r['seconds']:7.1f}s rc {r['rc']}",
                        flush=True,
                    )
                lock.notify_all()

    th = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, opt["jobs"]))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return results, round(time.time() - t_start, 1)


# ------------------------------------------------------------------------------------------------ summary
def step_state(work, k, step, name):
    """'ok' | 'failed' | 'missing' from the step's marker (an unreadable marker, e.g. half written when the job was
    stopped, counts as missing)"""
    if done(work, k, step, name):
        return "ok"
    d = sdir(work, k)
    f = {
        **{s: os.path.join(d, s, name, "stage_done.json") for s in STAGES},
        "thin": os.path.join(d, "steps", f"{name}_thin.json"),
        "assemble": os.path.join(d, "thin", "proposals", name, f"{name}_joints.json"),
        "folds": os.path.join(d, "final", "proposals", name, f"{name}_joints.json"),
    }[step]
    try:
        json.load(open(f))
    except (OSError, ValueError):
        return "missing"
    return "failed"


def summarise(work, names, states, no_env):
    import numpy as np

    rows = []
    for n in names:
        if n in no_env:
            rows.append(dict(name=n, status="no envelope", error="", failed_steps="", n_joints=0, contract=""))
            continue
        bad = [f"{k}:{s}" for k in states for s in STEPS if step_state(work, k, s, n) != "ok"]
        try:
            d = json.load(open(ens_file(work, n)))
        except Exception:
            d = None
        if d is None:
            rows.append(dict(name=n, status="missing", error="", failed_steps=",".join(bad), n_joints=0, contract=""))
            continue
        prob = check_product(d, ensemble=len(states) > 1) if d.get("status") == "ok" else []
        if d.get("status") == "ok":
            try:
                lj = json.load(open(os.path.join(work, "limbs", n, "limbs.json")))
                z = np.load(os.path.join(work, "limbs", n, "limbs.npz"))
                prob += check_limbs(lj, {k: z[k] for k in z.files})
            except Exception as e:
                prob.append(f"limb product unreadable: {type(e).__name__}")
        nj = sum(1 for v in (d.get("joints_scan") or {}).values() if v is not None)
        rows.append(
            dict(
                name=n,
                status=d.get("status"),
                error=(d.get("error") or "").replace("\t", " ")[:300],
                failed_steps=",".join(bad),
                n_joints=nj,
                contract="; ".join(prob)[:300],
            )
        )
    with open(os.path.join(work, "summary.tsv"), "w") as fh:
        cols = ["name", "status", "n_joints", "failed_steps", "contract", "error"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    return rows


# ------------------------------------------------------------------------------------------------ main
def default_mem_cap_gb(jobs):
    """MEM_CAP_GB, or 80 % of the machine's memory divided by the number of concurrent jobs when that is less (psutil;
    MEM_CAP_GB when the total cannot be read)"""
    try:
        import psutil

        total_gb = psutil.virtual_memory().total / GIB
    except Exception:
        return MEM_CAP_GB
    return round(min(MEM_CAP_GB, 0.8 * total_gb / max(1, jobs)), 1)


def check_environment(allow_without_embree):
    miss = missing_dependencies()
    if miss:
        raise UsageError(f"missing packages: {', '.join(miss)} (see the package README)")
    if not has_embree() and not allow_without_embree:
        raise UsageError(
            "trimesh has no embree ray caster (pip install embreex): the joint placement's ray casts would differ from "
            "the production run; pass --allow-without-embree to run anyway"
        )
    import importlib.util

    along_dir = os.path.dirname(MODEL_FILE)
    sys.path.insert(0, along_dir)
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ inside the package
    try:
        spec = importlib.util.spec_from_file_location("antscan_joints_along", os.path.join(along_dir, "along.py"))
        along = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(along)
        if along.file_sha256(MODEL_FILE) != along.MODEL_SHA256:
            raise UsageError(f"{MODEL_FILE} is not the shipped joint classifier (sha256 differs); restore it")
        try:
            want, have = along.check_sklearn(MODEL_FILE)
        except (ImportError, RuntimeError) as e:
            raise UsageError(str(e)) from None
    finally:
        sys.path.remove(along_dir)
        sys.dont_write_bytecode = dwb
    return dict(sklearn_model=sorted(want), sklearn=have, embree=has_embree())


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--mem-cap-gb", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--states", default="s0,s1,s2,s3")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--allow-without-embree", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--break-lock", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from .stage import parse_grid

    try:
        if not os.path.isfile(a.names):
            raise UsageError(f"--names: no file {a.names}")
        if "{name}" not in a.env and not os.path.isdir(a.env):
            raise UsageError(f"--env: no folder {a.env}")
        states = [s for s in a.states.split(",") if s]
        for s in states:
            try:
                parse_grid(s)
            except ValueError as e:
                raise UsageError(f"--states: {e}") from None
        if a.mem_cap_gb is None:
            a.mem_cap_gb = default_mem_cap_gb(a.jobs)
        if a.jobs < 1 or a.mem_cap_gb < 0 or a.timeout <= 0:
            raise UsageError("--jobs must be >= 1, --mem-cap-gb >= 0 (0 = no cap), --timeout > 0")
        names = read_names(a.names)
        if not names:
            raise UsageError(f"--names: no names in {a.names}")
        versions = check_environment(a.allow_without_embree)
        out = os.path.abspath(a.out)
        os.makedirs(out, exist_ok=True)
        lock = acquire_lock(out, a.break_lock)
    except UsageError as e:
        print(f"[antscan_joints] {e}", file=sys.stderr)
        return EXIT_USAGE
    cache = os.path.abspath(a.cache or os.path.join(out, "cache"))
    os.makedirs(os.path.join(cache, "numba"), exist_ok=True)

    def _stop(signum, frame):
        with RUNNING_LOCK:
            for p in list(RUNNING.values()):
                _kill(p)
        print(f"[antscan_joints] stopped by signal {signum}; running jobs killed (a rerun resumes)", file=sys.stderr)
        release_lock(lock)
        os._exit(130)

    old = {s: signal.signal(s, _stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        meshes = {n: find_envelope(a.env, n) for n in names}
        no_env = [n for n in names if meshes[n] is None]
        todo = [n for n in names if meshes[n] is not None]
        for n in no_env:
            print(f"[antscan_joints] {n}: no envelope under {a.env}", flush=True)
        opt = dict(
            jobs=a.jobs, timeout=a.timeout, cap=int(a.mem_cap_gb * GIB), env=job_env(cache), retry_failed=a.retry_failed
        )
        print(
            f"[antscan_joints] {VERSION}: {len(todo)} specimens x {len(states)} states, {a.jobs} jobs, memory cap "
            f"{a.mem_cap_gb:g} GiB per job -> {out}",
            flush=True,
        )
        results, wall = batch(todo, meshes, out, states, opt) if todo else ([], 0.0)
        rows = summarise(out, names, states, set(no_env))
        rep = dict(
            version=VERSION,
            names=names,
            out=out,
            states=states,
            jobs=a.jobs,
            mem_cap_gb=a.mem_cap_gb,
            timeout=a.timeout,
            post_timeout=POST_TIMEOUT,
            cache=cache,
            python=PY,
            wall_s=wall,
            environment=versions,
            results=results,
        )
        json.dump(rep, open(os.path.join(out, f"batch_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        nf = sum(1 for r in results if not r["ok"])
        print(
            f"[antscan_joints] done: {n_ok}/{len(names)} products ok, {nf} failed jobs, wall {wall} s; "
            f"{os.path.join(out, 'summary.tsv')}",
            flush=True,
        )
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        release_lock(lock)
    return EXIT_OK if n_ok else EXIT_NOTHING


if __name__ == "__main__":
    sys.exit(main())
