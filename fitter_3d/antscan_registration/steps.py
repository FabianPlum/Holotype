"""Per-specimen steps of the pipeline. Every function takes one picklable job dict and returns a small row dict;
none of them raises for a bad specimen (the row carries `status` and `error`). A job may carry `sig` and `sig_path`:
the signature of everything the step's result depends on, written next to the result when the step succeeded, so that
a later run can tell a finished result from a stale one (antscan_registration.run)."""

import hashlib
import json
import os
import shutil
import time
import traceback

import numpy as np

from fitter_3d.antscan_common import json_default as _js

from . import intake as IN
from . import setup

_W = {}


def _err(e, n=4):
    return f"{type(e).__name__}: {e} | {traceback.format_exc(limit=n)[-700:]}"


def jdump(o, path, indent=1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(o, f, indent=indent, default=_js)
    os.replace(tmp, path)


def find_joint_file(joints_dir, name):
    for p in (f"{joints_dir}/{name}/{name}_joints.json", f"{joints_dir}/{name}_joints.json"):
        if os.path.isfile(p):
            return p
    return None


def find_envelope(env, name):
    if "{n}" in env or "{name}" in env:
        return env.format(n=name, name=name)
    for p in (f"{env}/{name}/{name}_envelope.stl", f"{env}/{name}_envelope.stl"):
        if os.path.isfile(p):
            return p
    return f"{env}/{name}/{name}_envelope.stl"


def limb_dir(limbs, name):
    if not limbs or limbs == "none":
        return None
    p = f"{limbs}/{name}"
    return p if os.path.isfile(f"{p}/limbs.json") else None


# -------------------------------------------------------------------------------------------------------- signatures
def file_hash(path, cache=None):
    """sha1 of a file's content ('missing' when there is none); `cache` (a dict) keeps hashes within one run"""
    if not path or not os.path.isfile(path):
        return "missing"
    if cache is not None and path in cache:
        return cache[path]
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    if cache is not None:
        cache[path] = h.hexdigest()
    return h.hexdigest()


def signature(**parts):
    """sha1 of a json-able description of everything a result depends on"""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=_js).encode()).hexdigest()


def read_sig(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def write_sig(path, sig):
    if not path or not sig:
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(sig + "\n")
    os.replace(tmp, path)


def is_done(result, sig_path, sig):
    """the step's result exists and was made from the same inputs, options and code version"""
    return os.path.isfile(result) and read_sig(sig_path) == sig


# -------------------------------------------------------------------------------------------------------------- pools
def pool_map(fn, jobs, n, init=None, initargs=(), fail=None, inline=False):
    """ordered map over jobs with n worker processes (spawn). Also n == 1 runs in a worker process of its own, so that
    a result does not depend on --jobs: a step that runs inside the calling process sees whatever earlier steps left
    there (torch loaded by a registration changes the last bits of later numpy sums), a fresh worker does not.
    `inline=True` runs in the calling process (tests of single functions). A worker that dies (killed by a signal, for
    example by a memory limit) does not stop the batch: the jobs it left without a result are run again, and a job
    that takes its worker down twice becomes `fail(job, message)` -- a row with an error."""
    if not jobs:
        return []
    if inline:
        if init:
            init(*initargs)
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from multiprocessing import get_context

    if fail is None:

        def fail(job, msg):
            return dict(name=job.get("name") if isinstance(job, dict) else None, status="failed", error=msg)

    out = [None] * len(jobs)

    def run(ids, workers):
        broken = []
        with ProcessPoolExecutor(
            max_workers=max(1, min(workers, 3, len(ids))),
            mp_context=get_context("spawn"),
            initializer=init,
            initargs=initargs,
        ) as ex:
            futs = [(i, ex.submit(fn, jobs[i])) for i in ids]
            for i, f in futs:
                try:
                    out[i] = f.result()
                except BrokenProcessPool:
                    broken.append(i)
                except Exception as e:  # a job function that raised: still a row
                    out[i] = fail(jobs[i], f"{type(e).__name__}: {e}")
        return broken

    tries = dict.fromkeys(range(len(jobs)), 0)
    pending, solo = list(range(len(jobs))), []
    while pending:
        broken = run(pending, n)
        for i in broken:
            tries[i] += 1
        solo += [i for i in broken if tries[i] >= 2]
        pending = [i for i in broken if tries[i] < 2]
    for i in solo:  # alone in a pool of one: the job that dies here is the one that kills its worker
        if run([i], 1):
            out[i] = fail(jobs[i], "worker process died (killed by a signal; out of memory?)")
    return out


def worker_init(model, cache=None):
    """initialiser of every worker process: the parts and their model file"""
    _W["model"] = setup(model, cache)


def explain(err):
    """an error text a user can act on: trimesh answers a truncated or empty STL with an import error of `chardet`"""
    err = "" if err is None else str(err)
    if "chardet" in err:
        return "envelope is not a readable STL (empty or truncated file?) | " + err
    return err


# ------------------------------------------------------------------------------------------------------------ intake
def intake_one(job):
    name, out = job["name"], job["out"]
    t0 = time.time()
    try:
        jf = find_joint_file(job["joints"], name)
        if jf is None:
            return dict(name=name, status="failed", error="no joint product")
        d = json.load(open(jf))
        bad = IN.check_product(d)
        if bad:
            return dict(name=name, status="failed", error=bad)
        maj, tie_policy = job.get("majority", "trunk"), job.get("tie_policy", "far")
        need_states = maj != "none" or tie_policy == "far"
        states = IN.load_states(job.get("joint_states"), name, d) if need_states else None
        d2, block = IN.intake(d, cut=job.get("cut", "auto"), tie_policy=tie_policy, states=states, majority=maj)
        d2["intake"]["input"] = jf
        op = f"{out}/{name}/{name}_joints.json"
        jdump(d2, op, indent=None)
        return dict(
            name=name,
            status="ok",
            jfile=op,
            cut=block["cut"],
            product_type=block["product_type"],
            n_raised=len(block["raised_to_usable"]),
            n_promoted=len(block["promoted"]),
            n_demoted=len(block["demoted"]) + len(block["tie_demoted"]) + len(block["tie_far"]),
            n_ties=len(block["ties"]),
            n_tie_far=len(block["tie_far"]),
            n_usable=block["n_usable"],
            n_usable_input=block["n_usable_input"],
            states_found=None
            if not (block["product_type"].startswith("ensemble") and need_states)
            else states is not None,
            seconds=time.time() - t0,
        )
    except Exception as e:
        return dict(name=name, status="failed", error=_err(e))


# -------------------------------------------------------------------------------------------------------------- GEOM
def geom_init(device, model, cache=None):
    """pool initialiser: device 'cpu' (bit-reproducible: one thread inside a specimen) or 'cuda' / 'cuda:N'. The
    device is handed to the registration as it is; the environment of the calling process is not touched. A worker
    process of a CPU run hides the GPUs from itself (nothing of a CPU run may touch them)."""
    import multiprocessing

    worker_init(model, cache)
    dev = str(device)
    _W["device"] = dev if dev.startswith("cuda") else "cpu"
    if _W["device"] == "cpu" and multiprocessing.parent_process() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or 2))
    torch.manual_seed(0)


def geom_one(job):
    """joint product (as given: the intake or the augmented one) + envelope + limbs -> route file <out>/<name>.npz +
    log <out>/<name>.json (GEOM's contract)."""
    from fitter_3d.geomreg import pipeline

    name, out = job["name"], job["out"]
    t0 = time.time()
    try:
        jf, ev = job["jfile"], find_envelope(job["env"], name)
        if not jf or not os.path.isfile(jf) or not os.path.isfile(ev):
            info = dict(
                name=name,
                status="failed",
                seconds=0.0,
                error=f"missing input: joints {bool(jf and os.path.isfile(jf))} envelope {os.path.isfile(ev)}",
            )
            if os.path.isfile(f"{out}/{name}.npz"):
                os.remove(f"{out}/{name}.npz")
            pipeline.write_route(out, name, None, info)
            return dict(name=name, status="failed", error=info["error"], seconds=0.0)
        try:
            arrays, info = pipeline.register_specimen(
                name,
                ev,
                os.path.realpath(jf),
                limb_dir(job["limbs"], name),
                kind=job.get("kind", "medial"),
                conf_min=IN.USABLE,
                opts=job.get("opts") or {},
                device=_W.get("device", "cpu"),
                model_pkl=_W.get("model"),
                refit=True,
            )
        except Exception as e:  # an unreadable envelope, a degenerate product: a log, not a stop
            arrays, info = None, dict(name=name, status="failed", seconds=time.time() - t0, error=_err(e))
        if arrays is None:
            info["error"] = explain(info.get("error"))
            if os.path.isfile(f"{out}/{name}.npz"):
                os.remove(f"{out}/{name}.npz")
        pipeline.write_route(out, name, arrays, info)
        row = dict(name=name, status=info["status"], seconds=time.time() - t0)
        if arrays is None:
            row["error"] = info.get("error")
        else:
            nat = info["native"]
            row.update(
                deform_med_bl=nat["deform_med_bl"],
                deform_p90_bl=nat["deform_p90_bl"],
                lbs_energy_share=nat["lbs_energy_share"],
                forward_err_um=nat["forward_plus_deform_max_err_um"],
                reasons=list(info.get("status_reasons") or []),
            )
            write_sig(job.get("sig_path"), job.get("sig"))
        return row
    except Exception as e:
        return dict(name=name, status="failed", error=_err(e), seconds=time.time() - t0)


# ----------------------------------------------------------------------------------------------------------- RECOVER
def recover_init(model, priors, kinds, use_pool=True, cache=None):
    worker_init(model, cache)
    from fitter_3d.antscan_registration.parts.limbrec.priors import Population

    _W["pop"] = Population.load(priors) if priors and os.path.isfile(priors) else None
    _W["cfg"] = dict(kinds=tuple(kinds), use_pool=use_pool)


def recover_one(job):
    """intake product + limbs + envelope + first registration -> augmented product <out>/proposals/<name>/..., log."""
    from fitter_3d.antscan_registration.parts.limbrec import recover as LR

    from . import mandible as MA

    name, out = job["name"], job["out"]
    t0 = time.time()
    op = f"{out}/proposals/{name}/{name}_joints.json"
    lp = f"{out}/logs/{name}_recovery.json"
    row = dict(name=name, status="ok", n_lost=None, n_recovered=0, n_mandible=0, changed=False)
    try:
        jf = job["jfile"]
        if not jf or not os.path.isfile(jf):
            return dict(row, status="failed", error="no joint product")
        os.makedirs(os.path.dirname(op), exist_ok=True)
        reg_dir = job.get("reg") or ""
        has_reg = os.path.isfile(f"{reg_dir}/{name}.npz")
        dout, log = None, dict(name=name, status="skipped", joints={})
        if job.get("limb_recovery", True):
            dout, log = LR.recover_specimen(
                name,
                jf,
                job["limbs"],
                find_envelope(job["env"], name),
                reg_dir if has_reg else "",
                _W.get("pop"),
                stage_root=job.get("stages"),
                cfg=_W.get("cfg"),
            )
        if dout is None:
            dout = json.load(open(jf))
        row.update(
            status=log.get("status", "ok") if log.get("status") != "skipped" else "ok",
            error=log.get("error"),
            n_lost=log.get("n_lost"),
            n_recovered=int(log.get("n_recovered") or 0),
        )
        if job.get("mandible_check", True):
            mk, idx = None, None
            if has_reg:
                from fitter_3d.antscan_registration.parts.skelreg.model import template

                t = template("medial")
                z = np.load(f"{reg_dir}/{name}.npz", allow_pickle=True)
                mk, idx = t.marker_regressor() @ np.asarray(z["V_final"], float), t.idx
            dout, mlog = MA.check(dout, limb_dir(job["limbs"], name), mk, idx)
            log["mandibles"] = mlog
            row["n_mandible"] = sum(1 for v in mlog.values() if v.get("recovered"))
        row["changed"] = bool(row["n_recovered"] + row["n_mandible"])
        log["joints_changed"] = row["changed"]
        jdump(dout, op, indent=None)
        log["seconds_total"] = time.time() - t0
        jdump(log, lp)
        write_sig(job.get("sig_path"), job.get("sig"))
        return dict(row, jfile=op, seconds=time.time() - t0)
    except Exception as e:  # the input product goes on unchanged
        try:
            shutil.copyfile(job["jfile"], op)
        except Exception:
            op = job.get("jfile")
        return dict(row, status="failed", error=_err(e), jfile=op, seconds=time.time() - t0)


# ----------------------------------------------------------------------------------------------------------- publish
GEOM_KEYS = {"vertex_status": "geom_vertex_status", "part_status": "geom_part_status", "status": "geom_status"}
GEOM_VERTEX_STATUS_NAMES = ("observed", "hidden", "unsupported")


def publish_registration(src_dir, name, dst_dir, source, sig):
    """work/<reg1|reg2>/<name>.{npz,json} (GEOM's route file) -> registration/<name>.{npz,json}: the same arrays, with
    the registration's OWN codes under `geom_vertex_status`, `geom_part_status`, `geom_status` (the status step's
    codes of the same names live in status/, routes/ and the export files and mean something else), plus
    `registration_source` ('reg1' | 'reg2'). Returns 'published', 'kept' (already there, same signature) or 'none'."""
    sj, sn = f"{src_dir}/{name}.json", f"{src_dir}/{name}.npz"
    dj, dn, ds = f"{dst_dir}/{name}.json", f"{dst_dir}/{name}.npz", f"{dst_dir}/{name}.sig"
    if not os.path.isfile(sj):
        for p in (dj, dn, ds):
            if os.path.isfile(p):
                os.remove(p)
        return "none"
    tag = f"{source}:{sig}:{file_hash(sj)}"
    if read_sig(ds) == tag and os.path.isfile(dj) and (os.path.isfile(dn) or not os.path.isfile(sn)):
        return "kept"
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.isfile(sn):
        with np.load(sn, allow_pickle=True) as z:
            arrays = {GEOM_KEYS.get(k, k): z[k] for k in z.files}
        arrays["geom_vertex_status_names"] = np.array(GEOM_VERTEX_STATUS_NAMES)
        arrays["registration_source"] = np.array(source)
        tmp = f"{dn}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, dn)
    elif os.path.isfile(dn):
        os.remove(dn)
    info = json.load(open(sj))
    info["registration_source"] = source
    jdump(info, dj)
    write_sig(ds, tag)
    return "published"
