"""THE ONE COMMAND of the AntScan registration pipeline.

    python -m fitter_3d.antscan_registration.run --names NAMES.txt --joints JOINT_DIR --limbs LIMB_DIR --env ENV --out OUT_DIR [options]

  NAMES.txt   one specimen name per line (a file key only; blank lines and '#' lines are skipped)
  JOINT_DIR   <name>/<name>_joints.json (or <name>_joints.json): joint product of the joint method (round 6 or round 7)
  LIMB_DIR    <name>/limbs.json + limbs.npz: its limb product; 'none' = no limb product (no tube warp, no recovery)
  ENV         directory with <name>/<name>_envelope.stl (or <name>_envelope.stl), or a template containing {n}
  OUT_DIR     output folder (layout: docs/antscan_registration_pipeline.md)
options
  --stages DIR[,DIR..]   the joint method's work directories holding paths_A / paths_B / paths_C (optional evidence of
                         the recovery step); default: found next to LIMB_DIR (round 6: its parent; round 7: its
                         parent's states/*), 'none' = off
  --joint-states DIR     ensemble products only: the joint method's `states` folder (<state>/final/proposals/<name>/..),
                         read for the majority spread of the trunk joints and for the rule on far ties; default:
                         `states` next to JOINT_DIR or up to two levels above it; used only when it reproduces the
                         products; 'none' = off
  --confidence RULE      ensemble products: 'majority-trunk' (default: the trunk joints' confidence is recomputed with
                         the spread of the majority of the states), 'majority-all' (limb joints too), 'as-given'
  --conf-cut auto|X      confidence cut of the intake (auto = 0.30; a number in (0, 1])
  --ties far|use|recover joints the ensemble placed on a tie: 'far' (default) leaves a limb joint without a majority
                         of the states within 4 % of body length to the recovery step, 'use' leaves every tie to the
                         cut, 'recover' leaves every tie to the recovery step (fitter_3d/antscan_registration/intake.py)
  --no-recover           skip the recovery step and the second registration
  --no-mandible-check    do not put mandible bases below the cut through the socket check
  --recover-disable LIST comma list of recovery kinds to switch off (lowconf,fallback,chainfit,tail)
  --priors FILE          segment-length prior of the recovery step (default: the shipped one)
  --population FILE|none population reference of the unposed mesh (fill source, edge reference; default: the shipped one)
  --geom-opts JSON       options of the registration (geomreg.surface.OPTS); a json dict
  --K JSON               json dict overriding status / grade constants (unknown keys are an error)
  --kind medial|pivot    what the joints are (pivot: model pivots of synthetic sets; the registration only)
  --device cpu|cuda[:N]  cpu (default, bit-reproducible) or a GPU (about 5 times faster, not bit-reproducible)
  --jobs N               worker processes (1..3 on the CPU, 1..2 on a GPU)
  --model PKL            model file (default: 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl)
  --cache DIR            folder for the cache of the template's outer surface (default: OUT_DIR/work/cache)
  --steps LIST           comma list out of intake,reg1,recover,reg2,status (default: all). The status step always
                         reads the registration that exists: the second one where the recovery changed the joints and
                         it was run, else the first
  --force                recompute every step asked for
  --break-lock           take over OUT_DIR although a lock file of another run is there
  --no-overlay --no-export   leave out overlays / export files

Every per-specimen step runs in worker processes of its own, also with --jobs 1: the files of a specimen do not
depend on --jobs or on which other specimens are in the batch.
Resume: a step's result is reused only when it was made from the same input files, options, model and pipeline version
(a signature is stored next to each result); anything else is recomputed, and so is everything that follows from it.
One run at a time writes to OUT_DIR (lock file OUT_DIR/.lock).
A specimen that fails in a step keeps a status and an error in OUT_DIR/manifest.jsonl and summary.csv; the batch goes on.
Exit code: 0 the batch ran; 2 wrong arguments, missing package, or OUT_DIR locked by another run; 3 the batch ran and
NO specimen got through (every row carries an error).
"""

from . import cli_environment

if __name__ == "__main__":  # the command: thread defaults before numpy / torch are imported
    cli_environment()

import argparse
import json
import os
import shutil
import sys
import time

from fitter_3d.antscan_common import UsageError, acquire_lock, release_lock

from . import PRIORS, VERSION, cache_dir, missing_dependencies, read_names, setup
from . import intake as IN
from . import steps as ST

STEPS = ("intake", "reg1", "recover", "reg2", "status")
EXIT_OK, EXIT_USAGE, EXIT_NOTHING = 0, 2, 3


def find_stages(limbs, given):
    if given == "none" or not limbs or limbs == "none":
        return None
    if given:
        r = [g for g in given.split(",") if g]
        return r if len(r) > 1 else r[0]
    par = os.path.dirname(os.path.abspath(limbs).rstrip("/"))
    if os.path.isdir(f"{par}/paths_A") or os.path.isdir(f"{par}/paths_C"):
        return par
    sd = f"{par}/states"
    if os.path.isdir(sd):
        r = [
            f"{sd}/{s}"
            for s in sorted(os.listdir(sd))
            if os.path.isdir(f"{sd}/{s}/paths_A") or os.path.isdir(f"{sd}/{s}/paths_C")
        ]
        if r:
            return r
    return None


def find_states(joints, given):
    """candidate `states` folders of an ensemble product: the one given, else next to the joints folder or up to two
    levels above it"""
    if given == "none":
        return None
    if given:
        return [given]
    par = os.path.dirname(os.path.abspath(joints).rstrip("/"))
    cands = (f"{par}/states", f"{os.path.dirname(par)}/states", f"{os.path.dirname(os.path.dirname(par))}/states")
    return [c for c in cands if os.path.isdir(c)]


# ------------------------------------------------------------------------------------------------------------- options
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True)
    ap.add_argument("--joints", required=True)
    ap.add_argument("--limbs", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=None)
    ap.add_argument("--joint-states", default=None)
    ap.add_argument("--confidence", default="majority-trunk", choices=["majority-trunk", "majority-all", "as-given"])
    ap.add_argument("--conf-cut", default="auto")
    ap.add_argument("--ties", default="far", choices=["far", "use", "recover"])
    ap.add_argument("--no-recover", action="store_true")
    ap.add_argument("--no-mandible-check", action="store_true")
    ap.add_argument("--recover-disable", default="")
    ap.add_argument("--kind", default="medial", choices=["medial", "pivot"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--priors", default=None)
    ap.add_argument("--population", default=None)
    ap.add_argument("--geom-opts", default="{}")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--break-lock", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--K", default=None)
    return ap.parse_args(argv)


def check_options(a):
    """everything that can be wrong with the arguments, found before any work is done (UsageError)"""
    steps = [s for s in a.steps.split(",") if s]
    bad = [s for s in steps if s not in STEPS]
    if bad or not steps:
        raise UsageError(f"--steps: unknown step(s) {bad}; choose from {','.join(STEPS)}")
    try:
        IN.parse_cut(a.conf_cut)
    except ValueError as e:
        raise UsageError(f"--conf-cut: {e}") from None
    for opt, txt in (("--geom-opts", a.geom_opts), ("--K", a.K)):
        if txt:
            try:
                v = json.loads(txt)
            except ValueError as e:
                raise UsageError(f"{opt}: not json ({e})") from None
            if not isinstance(v, dict):
                raise UsageError(f"{opt}: a json dict is expected")
            if opt == "--geom-opts":
                from fitter_3d.geomreg.surface import check_opts

                try:
                    check_opts(v)
                except ValueError as e:
                    raise UsageError(f"--geom-opts: {e}") from None
    dev = str(a.device)
    if dev != "cpu" and not dev.startswith("cuda"):
        raise UsageError(f"--device: cpu or cuda[:N], not {dev!r}")
    if not os.path.isfile(a.names):
        raise UsageError(f"--names: no file {a.names}")
    read_names(a.names)  # a name that is a path is refused here, before any work
    if not os.path.isdir(a.joints):
        raise UsageError(f"--joints: no folder {a.joints}")
    if a.limbs not in (None, "none") and not os.path.isdir(a.limbs):
        raise UsageError(f"--limbs: no folder {a.limbs} (give 'none' to run without a limb product)")
    return steps


def check_K(K):
    from fitter_3d.antscan_registration.parts.rs import status as S

    from . import grade as GR

    known = set(S.DEFAULT_K) | set(GR.K_GRADE)
    unknown = sorted(set(K) - known)
    if unknown:
        raise UsageError(f"--K: unknown constant(s) {unknown}; known: {sorted(known)}")


def main(argv=None):
    a = parse(argv)
    try:
        steps = check_options(a)
        need, opt = missing_dependencies()
        if need:
            raise UsageError(
                "missing package(s): "
                + "; ".join(f"{m} ({why})" for m, why in need)
                + ". Install them (pip install "
                + " ".join(m for m, _ in need)
                + ") and run again."
            )
        for m, why in opt:
            print(f"[antscan_registration] note: package {m} is not installed: {why}", flush=True)
        out = os.path.abspath(a.out)
        os.makedirs(out, exist_ok=True)
        model = setup(a.model)
        if not os.path.isfile(model):
            raise UsageError(f"--model: no file {model}")
        K = json.loads(a.K) if a.K else {}
        check_K(K)
        lock = acquire_lock(out, a.break_lock)
    except UsageError as e:
        print(f"[antscan_registration] error: {e}", file=sys.stderr, flush=True)
        return EXIT_USAGE
    try:
        return _run(a, steps, out, model, K)
    finally:
        release_lock(lock)


def _run(a, steps, out, model, K):
    t00 = time.time()
    names = read_names(a.names)
    W = f"{out}/work"
    from fitter_3d.antscan_registration.parts import partconf

    cache = a.cache or partconf.CACHE or cache_dir(out)  # given; set by an earlier setup(); the package's default
    setup(model, cache)
    for d in (W, f"{W}/intake", f"{W}/reg1", f"{W}/reg2", f"{W}/recover/logs", f"{out}/registration", f"{out}/joints"):
        os.makedirs(d, exist_ok=True)
    cuda = str(a.device).startswith("cuda")
    jobs_n = max(1, min(a.jobs, 3))
    gjobs = min(jobs_n, 2) if cuda else jobs_n
    limbs = None if a.limbs in (None, "none") else a.limbs
    stages = find_stages(limbs, a.stages)
    geom_opts = json.loads(a.geom_opts)
    do_recover = not a.no_recover and limbs is not None
    pri = a.priors or PRIORS
    from fitter_3d.antscan_registration.parts.limbrec.recover import KINDS

    off = {k for k in a.recover_disable.split(",") if k}
    kinds = tuple(k for k in KINDS if k not in off)
    # ---- the manifest keeps the rows of earlier invocations (a later `--steps status` does not wipe the others)
    man = {n: dict(name=n) for n in names}
    mp = f"{out}/manifest.jsonl"
    if os.path.isfile(mp) and not a.force:
        for ln in open(mp):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("name") in man:
                man[r["name"]] = r
    T = {}
    H = {}  # content hashes of the input files of this run

    def fh(p):
        return ST.file_hash(p, H)

    def jf_intake(n):
        return f"{W}/intake/{n}/{n}_joints.json"

    def jf_aug(n):
        return f"{W}/recover/proposals/{n}/{n}_joints.json"

    def limb_sig(n):
        d = ST.limb_dir(limbs, n)
        return None if d is None else [fh(f"{d}/limbs.json"), fh(f"{d}/limbs.npz")]

    common = dict(version=VERSION, model=fh(model), kind=a.kind, opts=geom_opts, device="cuda" if cuda else "cpu")

    def sig_reg(n, jf):
        return ST.signature(
            step="registration", joints=fh(jf), envelope=fh(ST.find_envelope(a.env, n)), limbs=limb_sig(n), **common
        )

    def sig_rec(n):
        return ST.signature(
            step="recover",
            version=VERSION,
            reg1=sig_reg(n, jf_intake(n)),
            stages=stages,
            priors=fh(pri),
            kinds=kinds,
            mandible=not a.no_mandible_check,
        )

    def joints_changed(n):
        """True when the recovery step of this folder changed the usable joints of n (and is still valid)"""
        lg = f"{W}/recover/logs/{n}_recovery.json"
        if not (do_recover and os.path.isfile(jf_aug(n)) and os.path.isfile(lg)):
            return False
        if ST.read_sig(f"{W}/recover/logs/{n}.sig") != sig_rec(n):
            return False
        L = json.load(open(lg))
        return bool(
            int(L.get("n_recovered") or 0) + sum(1 for v in (L.get("mandibles") or {}).values() if v.get("recovered"))
        )

    # ---- intake (always recomputed: it is cheap, and its product is what every later signature is built on) -------
    if "intake" in steps:
        t0 = time.time()
        jst = find_states(a.joints, a.joint_states)
        rows = [
            ST.intake_one(
                dict(
                    name=n,
                    joints=a.joints,
                    out=f"{W}/intake",
                    cut=a.conf_cut,
                    tie_policy=a.ties,
                    joint_states=jst,
                    majority={"majority-trunk": "trunk", "majority-all": "all", "as-given": "none"}[a.confidence],
                )
            )
            for n in names
        ]
        for r in rows:
            man[r["name"]]["intake"] = r
            if r["status"] == "failed" and os.path.isfile(jf_intake(r["name"])):
                os.remove(jf_intake(r["name"]))  # no stale product of an earlier run under a failed intake
        nost = sum(1 for r in rows if r.get("states_found") is False)
        if nost:
            print(
                f"[antscan_registration] intake: {nost} of {len(rows)} ensemble products WITHOUT per-state products that reproduce them "
                f"(searched: {jst or 'nothing found'}): their trunk joints keep the ensemble's confidence and far ties "
                f"are used as they come; pass --joint-states",
                flush=True,
            )
        T["intake"] = time.time() - t0
    # ---- first registration -----------------------------------------------------------------------------------------
    if "reg1" in steps:
        t0 = time.time()
        sg = {n: sig_reg(n, jf_intake(n)) for n in names}
        todo = [n for n in names if a.force or not ST.is_done(f"{W}/reg1/{n}.json", f"{W}/reg1/{n}.sig", sg[n])]
        print(
            f"[antscan_registration] reg1: {len(todo)} of {len(names)} to do, device {a.device}, jobs {gjobs}",
            flush=True,
        )
        for n in todo:  # a result of other inputs is not left under this name
            for ext in ("sig", "npz", "json"):
                if os.path.isfile(f"{W}/reg1/{n}.{ext}"):
                    os.remove(f"{W}/reg1/{n}.{ext}")
        rows = ST.pool_map(
            ST.geom_one,
            [
                dict(
                    name=n,
                    jfile=jf_intake(n),
                    env=a.env,
                    limbs=limbs,
                    out=f"{W}/reg1",
                    kind=a.kind,
                    opts=geom_opts,
                    sig=sg[n],
                    sig_path=f"{W}/reg1/{n}.sig",
                )
                for n in todo
            ],
            gjobs,
            ST.geom_init,
            (a.device, model, cache),
        )
        for r in rows:
            man[r["name"]]["reg1"] = r
        T["reg1"] = time.time() - t0
    # ---- recovery ---------------------------------------------------------------------------------------------------
    if "recover" in steps and do_recover:
        t0 = time.time()
        sg = {n: sig_rec(n) for n in names}
        todo = [n for n in names if a.force or not ST.is_done(jf_aug(n), f"{W}/recover/logs/{n}.sig", sg[n])]
        print(f"[antscan_registration] recover: {len(todo)} of {len(names)} to do, stages {stages}", flush=True)
        for n in todo:
            for p in (f"{W}/recover/logs/{n}.sig", f"{W}/recover/logs/{n}_recovery.json", jf_aug(n)):
                if os.path.isfile(p):
                    os.remove(p)
        rows = ST.pool_map(
            ST.recover_one,
            [
                dict(
                    name=n,
                    jfile=jf_intake(n),
                    env=a.env,
                    limbs=limbs,
                    out=f"{W}/recover",
                    reg=f"{W}/reg1",
                    stages=stages,
                    mandible_check=not a.no_mandible_check,
                    sig=sg[n],
                    sig_path=f"{W}/recover/logs/{n}.sig",
                )
                for n in todo
            ],
            jobs_n,
            ST.recover_init,
            (model, pri, kinds, True, cache),
        )
        for r in rows:
            man[r["name"]]["recover"] = r
        T["recover"] = time.time() - t0
    # ---- second registration (only specimens whose usable joints changed) ---------------------------------------------
    if "reg2" in steps:
        t0 = time.time()
        changed = [n for n in names if joints_changed(n)]
        sg = {n: sig_reg(n, jf_aug(n)) for n in changed}
        todo = [n for n in changed if a.force or not ST.is_done(f"{W}/reg2/{n}.json", f"{W}/reg2/{n}.sig", sg[n])]
        print(
            f"[antscan_registration] reg2: {len(changed)} specimens with recovered joints, {len(todo)} to do",
            flush=True,
        )
        for n in todo:
            for ext in ("sig", "npz", "json"):
                if os.path.isfile(f"{W}/reg2/{n}.{ext}"):
                    os.remove(f"{W}/reg2/{n}.{ext}")
        rows = ST.pool_map(
            ST.geom_one,
            [
                dict(
                    name=n,
                    jfile=jf_aug(n),
                    env=a.env,
                    limbs=limbs,
                    out=f"{W}/reg2",
                    kind=a.kind,
                    opts=geom_opts,
                    sig=sg[n],
                    sig_path=f"{W}/reg2/{n}.sig",
                )
                for n in todo
            ],
            gjobs,
            ST.geom_init,
            (a.device, model, cache),
        )
        for r in rows:
            man[r["name"]]["reg2"] = r
        T["reg2"] = time.time() - t0
    # ---- the final registration and its joints (always brought up to date before the status step) ---------------------
    final_sig = {}
    if "reg2" in steps or "status" in steps:
        for n in names:
            ch = joints_changed(n)
            src, jsrc, which = f"{W}/reg1", jf_intake(n), "reg1"
            second = False
            if ch:
                s2 = sig_reg(n, jf_aug(n))
                if os.path.isfile(f"{W}/reg2/{n}.json") and (
                    ST.read_sig(f"{W}/reg2/{n}.sig") == s2 or not os.path.isfile(f"{W}/reg2/{n}.npz")
                ):
                    # the second registration of these joints (or its failure: then the specimen has no registration)
                    src, jsrc, which, second = f"{W}/reg2", jf_aug(n), "reg2", True
                else:
                    second = "not run: the recovered joints are not used"
            sig = sig_reg(n, jsrc)
            ST.publish_registration(src, n, f"{out}/registration", which, sig)
            jd = f"{out}/joints/{n}/{n}_joints.json"
            if os.path.isfile(jsrc):
                os.makedirs(os.path.dirname(jd), exist_ok=True)
                shutil.copyfile(jsrc, jd)
            elif os.path.isfile(jd):
                os.remove(jd)
            man[n]["joints_changed"] = ch
            man[n]["second_registration"] = second
            man[n]["registration_source"] = which
            final_sig[n] = ST.signature(registration=sig, source=which, version=VERSION)
    # ---- status, morphometrics, export ------------------------------------------------------------------------------
    summ = None
    if "status" in steps:
        t0 = time.time()
        from . import status_run as SR

        summ = SR.run(
            names,
            out,
            env=a.env,
            limbs=limbs,
            jobs=jobs_n,
            force=a.force,
            no_overlay=a.no_overlay,
            no_export=a.no_export,
            K=K,
            manifest=man,
            sigs=final_sig,
            population=a.population,
            model=model,
        )
        T["status"] = time.time() - t0
    T["total"] = time.time() - t00
    # ---- records: every invocation is kept in run.json; the manifest holds the latest row of every step ---------------
    prev, hist = {}, []
    if os.path.isfile(f"{out}/run.json"):
        try:
            old = json.load(open(f"{out}/run.json"))
            prev, hist = old.get("seconds") or {}, list(old.get("invocations") or [])
        except Exception:
            prev, hist = {}, []
    hist.append(
        dict(command=" ".join(sys.argv), steps=steps, started=time.strftime("%F %T", time.localtime(t00)), seconds=T)
    )
    ST.jdump(
        dict(
            version=VERSION,
            command=" ".join(sys.argv),
            args=vars(a),
            model=model,
            stages=stages,
            n=len(names),
            seconds={**prev, **T},
            summary=summ,
            invocations=hist,
        ),
        f"{out}/run.json",
    )
    tmp = f"{mp}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        for n in names:
            f.write(json.dumps(man[n], default=ST._js) + "\n")
    os.replace(tmp, mp)
    # ---- exit code --------------------------------------------------------------------------------------------------
    last = [s for s in STEPS if s in steps][-1]
    if last == "status":
        nok = (summ or {}).get("n_decided", 0)
    elif last in ("reg1", "reg2"):
        nok = sum(
            1 for n in names if os.path.isfile(f"{out}/registration/{n}.npz") or os.path.isfile(f"{W}/reg1/{n}.npz")
        )
    else:
        nok = sum(1 for n in names if (man[n].get(last) or {}).get("status") != "failed")
    print(
        f"[antscan_registration] done {steps} in {T['total']:.0f} s -> {out} ({nok} of {len(names)} specimens through)",
        flush=True,
    )
    return EXIT_OK if (nok > 0 or not names) else EXIT_NOTHING


if __name__ == "__main__":
    sys.exit(main())
