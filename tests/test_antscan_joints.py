"""Tests of fitter_3d/antscan_joints (the AntScan joint method), without scan data:

- the ensemble over states (steps.py ensemble, the production entry point) on hand-made per-state products: identical
  states reproduce the state, a deviating state is outvoted, the joint confidence follows the formula the
  registration's intake relies on, a state with the other head end is dropped;
- the resume plan (a missing step reruns everything that depends on it), the per-job memory cap and time limit;
- the classifier file: its recorded hash, the restricted unpickler, the scikit-learn version check;
- one end-to-end run of the command on a synthetic specimen (posed template; slow).

    pytest tests/test_antscan_joints.py            (from the repository root)
"""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fitter_3d.antscan_joints as AJ
from fitter_3d.antscan_joints import run as RUN

PKG = os.path.dirname(AJ.__file__)


# ------------------------------------------------------------------------------------------------ helpers
def rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


# per-state products of a hand-made "ant": a straight trunk, six legs, two antennae, two mandibles
CHAIN = {
    **{c: ["pt", "ta", "ti", "fe", "tr", "co"] for c in AJ.LEGS},
    "AL": ["an_3", "an_2", "an_1"],
    "AR": ["an_3", "an_2", "an_1"],
    "ML": ["ma"],
    "MR": ["ma"],
}


def omni(ch, j):
    if ch in AJ.LEGS:
        return f"l_{ch[1]}_{j}_{ch[0].lower()}"
    return f"{j}_{ch[1].lower()}" if ch in ("AL", "AR") else f"ma_{ch[1].lower()}"


def limb_points(ch, BL):
    """joint positions of a limb, tip first (canonical frame: x head, y left, z dorsal; micrometres)"""
    s = 1.0 if (ch[0] if ch in AJ.LEGS else ch[1]) == "L" else -1.0  # +y = the animal's left
    if ch in AJ.LEGS:
        x0 = {"1": 0.15, "2": 0.08, "3": 0.01}[ch[1]] * BL
        root = np.array([x0, s * 0.03 * BL, -0.03 * BL])
        steps = [(0, 0.03, -0.03), (0, 0.10, 0.04), (0, 0.10, -0.08), (0, 0.08, -0.03), (0, 0.04, -0.01)]
        P = [root]
        for d in steps:
            P.append(P[-1] + np.array([d[0], s * d[1], d[2]]) * BL)
        return np.array(P[::-1])  # pt ... co
    if ch in ("AL", "AR"):
        root = np.array([0.40 * BL, s * 0.03 * BL, 0.02 * BL])
        return np.array(
            [root + np.array([0.20, s * 0.12, 0.0]) * BL, root + np.array([0.08, s * 0.10, 0.03]) * BL, root]
        )
    root = np.array([0.46 * BL, s * 0.02 * BL, -0.02 * BL])
    return np.array([root + np.array([0.05, -s * 0.02, -0.01]) * BL, root])  # tip, ma (base)


def dense(P, n=15):
    X = np.concatenate([np.linspace(P[i], P[i + 1], n, endpoint=False) for i in range(len(P) - 1)] + [P[-1:]])
    return X, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]


def write_state(sd, name, R, o, BL=2000.0, shift=None, head_flip=False, conf=0.9):
    """trunk / limbs / proposals of one state in sd (the per-state layout states/<k>/final/... of the method).
    R: rows = canonical axes in scan coordinates, o: origin; shift: {limb: canonical offset (um)} moves that limb."""
    to_scan = lambda P: np.asarray(P) @ R + o  # noqa: E731
    shift = shift or {}
    tj = {
        "neck": [0.30 * BL, 0, 0],
        "b_a_1": [-0.05 * BL, 0, 0],
        "b_a_2": [-0.12 * BL, 0, 0],
        "b_a_3": [-0.20 * BL, 0, 0],
        "b_a_4": [-0.35 * BL, 0, -0.02 * BL],
        "b_a_5": [-0.48 * BL, 0, -0.03 * BL],
        "head_front": [0.50 * BL, 0, 0],
        "gaster_tip": [-0.50 * BL, 0, 0],
    }
    js = {k: to_scan(v).tolist() for k, v in tj.items() if k not in ("neck",)}
    js["b_t"] = js["b_h"] = to_scan(tj["neck"]).tolist()
    arrays, lmeta = {}, {}
    for ch, joints in CHAIN.items():
        P = limb_points(ch, BL) + np.asarray(shift.get(ch, (0.0, 0.0, 0.0)))
        if ch in ("ML", "MR"):
            js[omni(ch, "ma")] = to_scan(P[1]).tolist()
        else:
            for j, p in zip(joints, P):
                js[omni(ch, j)] = to_scan(p).tolist()
        X, s = dense(P)
        arrays[f"{ch}_X"], arrays[f"{ch}_s"], arrays[f"{ch}_r"] = to_scan(X), s, np.full(len(X), 0.01 * BL)
        lmeta[ch] = {
            "status": "path",
            "confidence": conf,
            "core_arc_um": float(s[-1]),
            "source": "A",
            "flags": [],
            "missing": [],
        }
    for w in AJ.WINGS:
        js[w] = None
    cf = {k: (0.0 if v is None else conf) for k, v in js.items()}
    Rt = R.copy()
    if head_flip:
        Rt[0], Rt[1] = -Rt[0], -Rt[1]
    T = to_scan(np.array([[x * BL, 0, 0] for x in np.linspace(-0.5, 0.5, 41)]))
    os.makedirs(f"{sd}/trunk/{name}", exist_ok=True)
    fr = {"rotation_scan_to_canonical": Rt.tolist(), "origin_scan": o.tolist()}
    json.dump(
        {
            "status": "ok",
            "frame": fr,
            "anatomical_frame_final": fr,
            "confidence": {"head_end": 0.9},
            "trunk_joints_scan": {"neck": js["b_t"], "b_a_1": js["b_a_1"]},
            "waist_segments": 2,
        },
        open(f"{sd}/trunk/{name}/trunk.json", "w"),
    )
    np.savez(
        f"{sd}/trunk/{name}/trunk.npz",
        T_X=T,
        T_s=np.linspace(0, BL, 41),
        T_r=np.full(41, 0.05 * BL),
        T_part=np.repeat([3, 2, 1, 0], [10, 10, 11, 10]),
    )
    os.makedirs(f"{sd}/limbs/{name}", exist_ok=True)
    np.savez(f"{sd}/limbs/{name}/limbs.npz", **arrays)
    json.dump({"name": name, "BL_um": BL, "limbs": lmeta}, open(f"{sd}/limbs/{name}/limbs.json", "w"))
    os.makedirs(f"{sd}/proposals/{name}", exist_ok=True)
    prov = {k: ("null" if v is None else "scan_feature (synthetic)") for k, v in js.items()}
    P = {
        "name": name,
        "status": "ok",
        "units": "micrometre",
        "joints_scan": js,
        "confidence": cf,
        "provenance": prov,
        "null_reasons": {w: "workers have no wings" for w in AJ.WINGS},
        "joint_detail": {},
        "completeness": {"flags": [], "missing_joints": []},
        "body_length_um": BL,
        "waist_segments": 2,
        "frame": {"rotation": R.tolist(), "origin_scan": o.tolist()},
    }
    json.dump(P, open(f"{sd}/proposals/{name}/{name}_joints.json", "w"))


def run_ensemble(work, name, states=AJ.STATES):
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(PKG, "steps.py"),
            "ensemble",
            "--names",
            name,
            "--work",
            str(work),
            "--states",
            ",".join(states),
            "--sub",
            "final",
            "--out",
            str(work),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.load(open(f"{work}/proposals/{name}/{name}_joints.json"))
    assert d["status"] == "ok", d.get("error")
    return d


# ------------------------------------------------------------------------------------------------ tests
def test_ensemble_of_states(tmp_path):
    rng = np.random.default_rng(1)
    R, o, BL, name = rot(rng), rng.normal(0, 300, 3), 2000.0, "synth_ens"
    # (1) four identical states reproduce the state's joints and confidences
    for k in AJ.STATES:
        write_state(f"{tmp_path}/a/states/{k}/final", name, R, o, BL)
    d = run_ensemble(tmp_path / "a", name)
    one = json.load(open(f"{tmp_path}/a/states/s0/final/proposals/{name}/{name}_joints.json"))
    assert d["ensemble"]["kept"] == list(AJ.STATES)
    for j, x in one["joints_scan"].items():
        if x is not None:
            assert np.allclose(d["joints_scan"][j], x), j
            assert d["confidence"][j] == pytest.approx(one["confidence"][j], abs=1e-3), j
    assert AJ.check_product(d) == []
    # (2) one state places leg L1 elsewhere: the three agreeing states win, the confidence follows the formula
    #     c = median state confidence x (LIMB_BASE + (1 - LIMB_BASE) x state support) x placed share x g(spread)
    for k in AJ.STATES:
        write_state(
            f"{tmp_path}/b/states/{k}/final", name, R, o, BL, shift={"L1": (0.0, 0.0, 0.25 * BL)} if k == "s3" else None
        )
    d = run_ensemble(tmp_path / "b", name)
    for j in CHAIN["L1"]:
        nm = omni("L1", j)
        assert np.allclose(d["joints_scan"][nm], one["joints_scan"][nm]), nm
        e = d["joint_detail"][nm]["ens"]
        assert e["n_cluster"] == 3 and e["picked_state"] != "s3"
        g = 1.0 / (1.0 + (e["spread_pct"] / 2.0) ** 2)
        want = e["conf_states_median"] * (0.5 + 0.5 * e["limb_state_frac"]) * e["placed_share"] * g
        assert d["confidence"][nm] == pytest.approx(want, abs=2e-3), nm
        assert d["confidence"][nm] < one["confidence"][nm]
    assert "s3" not in d["completeness"]["parts"]["L1"]["states_agreeing"]
    # (3) a state whose trunk points the other way is dropped
    for k in AJ.STATES:
        write_state(f"{tmp_path}/c/states/{k}/final", name, R, o, BL, head_flip=(k == "s2"))
    d = run_ensemble(tmp_path / "c", name)
    assert d["ensemble"]["kept"] == ["s0", "s1", "s3"]
    assert any(f.startswith("ensemble_head_end_disagrees") for f in d["completeness"]["flags"])


def test_resume_reruns_every_dependent_step(tmp_path):
    w, n, states = str(tmp_path), "spec", ["s0", "s1"]

    def mark(k, step, status="ok"):
        if step in AJ.STAGES:
            f = f"{w}/states/{k}/{step}/{n}/stage_done.json"
        elif step == "thin":
            f = f"{w}/states/{k}/steps/{n}_thin.json"
        else:
            f = f"{w}/states/{k}/{'thin' if step == 'assemble' else 'final'}/proposals/{n}/{n}_joints.json"
        os.makedirs(os.path.dirname(f), exist_ok=True)
        json.dump({"status": status}, open(f, "w"))

    for k in states:
        for s in AJ.STEPS:
            mark(k, s)
    os.makedirs(f"{w}/proposals/{n}", exist_ok=True)
    json.dump({"status": "ok"}, open(f"{w}/proposals/{n}/{n}_joints.json", "w"))
    fin, ens = RUN.stale_plan(w, [n], states)
    assert len(fin) == 2 * len(AJ.STEPS) and ens == {("ens", n)}
    mark("s1", "paths_B", "failed")  # paths_B of s1 failed: kept as failed, unless failed steps are retried
    assert RUN.stale_plan(w, [n], states) == (fin, ens)
    fin_r, ens_r = RUN.stale_plan(w, [n], states, retry_failed=True)
    assert {s for s in AJ.STEPS if (n, "s1", s) not in fin_r} == {"paths_B", "thin", "assemble", "folds"}
    assert ens_r == set() and all((n, "s0", s) in fin_r for s in AJ.STEPS)
    os.remove(f"{w}/states/s1/paths_C/{n}/stage_done.json")  # interrupted: it and what depends on it run again
    fin, ens = RUN.stale_plan(w, [n], states)
    assert {s for s in AJ.STEPS if (n, "s1", s) not in fin} == {"paths_C", "thin", "assemble", "folds"} and not ens
    # outputs of a step that runs again are removed first; a fallback symlink is unlinked, never followed
    src = f"{w}/states/s1/paths_A/{n}"
    os.makedirs(f"{w}/states/s1/thin/paths_A", exist_ok=True)
    os.symlink(src, f"{w}/states/s1/thin/paths_A/{n}")
    for p in RUN.outputs(w, "s1", "thin", n):
        RUN.remove(p)
    assert not os.path.lexists(f"{w}/states/s1/thin/paths_A/{n}") and os.path.isfile(f"{src}/stage_done.json")


def test_job_memory_cap_and_time_limit(tmp_path):
    pytest.importorskip("psutil")
    env = dict(os.environ)
    grow = 'import time\nx = []\nfor _ in range(60):\n    x.append(b"\\x01" * (20 * 2**20))\n    time.sleep(0.02)\ntime.sleep(5)\n'
    with open(tmp_path / "log", "w") as fh:
        rc, cpu, rss, peak = RUN.run_measured([sys.executable, "-c", grow], fh, 60, 200 * 2**20, env, poll=0.1)
    assert isinstance(rc, str) and rc.startswith("memory cap") and peak > 200
    with open(tmp_path / "log", "w") as fh:
        rc, *_ = RUN.run_measured([sys.executable, "-c", "import time; time.sleep(30)"], fh, 1, 0, env, poll=0.1)
    assert rc.startswith("timeout")
    with open(tmp_path / "log", "w") as fh:
        rc, *_ = RUN.run_measured([sys.executable, "-c", "pass"], fh, 60, 2**30, env)
    assert rc == 0
    # the runner records such a job as a failed step
    RUN.record_failure(str(tmp_path), "s0", "paths_C", "spec", "memory cap 1 GiB exceeded", 3.0)
    assert RUN.step_state(str(tmp_path), "s0", "paths_C", "spec") == "failed"


def test_classifier_file_hash_unpickler_and_version_check(tmp_path, monkeypatch):
    import importlib.util

    along_dir = os.path.join(PKG, "method", "along")
    pytest.importorskip("trimesh")
    monkeypatch.syspath_prepend(along_dir)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # no __pycache__ inside the package
    spec = importlib.util.spec_from_file_location("antscan_joints_along_test", os.path.join(along_dir, "along.py"))
    along = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(along)
    finally:
        sys.modules.pop("geom_profile", None)  # along's flat sibling import
    assert along.sklearn_versions_of(AJ.MODEL_FILE) == {"1.7.2"}
    # the shipped file is the one whose hash the loader checks, and the loader resolves no foreign global
    assert os.path.abspath(along.MODEL_FILE) == os.path.abspath(AJ.MODEL_FILE)
    assert along.file_sha256(AJ.MODEL_FILE) == along.MODEL_SHA256
    evil = tmp_path / "evil.pkl"
    evil.write_bytes(pickle.dumps(os.system))  # unpickling resolves os.system / posix.system
    with pytest.raises(pickle.UnpicklingError, match="may not reference"):
        along.ModelUnpickler(open(evil, "rb")).load()
    fake = tmp_path / "model.pkl"
    pickle.dump({"clf": {"_sklearn_version": "0.0.1"}}, open(fake, "wb"))
    assert along.sklearn_versions_of(str(fake)) == {"0.0.1"}
    monkeypatch.delenv("ANTSCAN_JOINTS_SKLEARN_ANY", raising=False)
    with pytest.raises((ImportError, RuntimeError), match="scikit-learn"):
        along.check_sklearn(str(fake))
    if importlib.util.find_spec("sklearn") is not None:
        monkeypatch.setenv("ANTSCAN_JOINTS_SKLEARN_ANY", "1")
        with pytest.warns(UserWarning, match="0.0.1"):
            along.check_sklearn(str(fake))


@pytest.mark.slow
def test_end_to_end_on_a_synthetic_specimen(tmp_path):
    """The command on a posed template (tests/antscan_synth_inputs.py; two states, so that the ensemble runs; about
    4 min with 2 jobs). On this template the PATHS-B and PATHS-C stages need more than 6 GiB and are stopped by the
    memory cap (recorded as failed steps in summary.tsv); the product then comes from PATHS-A alone, which checks the
    cap and the failure handling on the real stages. Checked: the product contract, the layout the registration
    reads, the joints against the template's own markers (median error of the usable joints < 3 % of body length),
    and that a rerun has nothing left to do."""
    for m in ("numba", "sklearn", "networkx", "skimage", "psutil"):
        pytest.importorskip(m)
    if not AJ.has_embree():
        pytest.skip("embreex not installed")
    pkl = os.path.join(ROOT, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")
    if not os.path.isfile(pkl):
        pytest.skip("model pkl not found")
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import antscan_synth_inputs as SI

    from fitter_3d.geomreg.model import template

    name = "synth_ant_0"
    SI.make_specimen(str(tmp_path), template(pkl), name, seed=0)
    truth = json.load(open(tmp_path / "joints" / name / f"{name}_joints.json"))
    (tmp_path / "names.txt").write_text(name + "\n")
    out = tmp_path / "out"
    args = ["--names", str(tmp_path / "names.txt"), "--env", str(tmp_path / "env"), "--out", str(out)]
    args += ["--states", "s0,s1", "--jobs", "2", "--mem-cap-gb", "6"]
    assert RUN.main(args) == 0
    d = json.load(open(out / "proposals" / name / f"{name}_joints.json"))
    assert AJ.check_product(d) == [], AJ.check_product(d)
    assert d["ensemble"]["states"] == ["s0", "s1"]
    z = np.load(out / "limbs" / name / "limbs.npz")
    assert AJ.check_limbs(json.load(open(out / "limbs" / name / "limbs.json")), {k: z[k] for k in z.files}) == []
    for k in ("s0", "s1"):  # the registration's per-state products and path stages
        assert (out / "states" / k / "final" / "proposals" / name / f"{name}_joints.json").is_file()
        assert (out / "states" / k / "paths_A" / name / "limbs.npz").is_file()
    BL = truth["body_length_um"]
    assert abs(d["body_length_um"] / BL - 1.0) < 0.05
    err = [
        np.linalg.norm(np.subtract(x, truth["joints_scan"][j])) / BL * 100.0
        for j, x in d["joints_scan"].items()
        if x is not None and truth["joints_scan"].get(j) is not None and d["confidence"][j] >= 0.3
    ]
    assert len(err) >= 25 and np.median(err) < 3.0, (len(err), np.median(err))
    n_jobs = len(open(out / "jobs.jsonl").read().splitlines())
    assert RUN.main(args) == 0  # resume: nothing left to do (the failed stages stay failed without --retry-failed)
    assert len(open(out / "jobs.jsonl").read().splitlines()) == n_jobs
