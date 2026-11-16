"""Tests of the registration pipeline on a specimen made from the model itself (CPU, no scan data, no truth).

    pytest tests/test_antscan_registration_pipeline.py        (from the repository root; CPU)

What they carry: the contract of every output file of the one command, the round trips of the format conversions
(stage file -> scan frame with a frame that is not the scan frame, native parameters -> mesh under fitter_3d's own
forward, the unposed bundle), a bundle row that does not depend on the batch, and the rules that decide what a
specimen is (intake of the joints, the model grade, which fails closed). One slow test runs a batch of degenerate
inputs that must end as statuses.
"""

import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitter_3d.antscan_registration as regpipe
from fitter_3d.antscan_registration import intake as IN

PKL = regpipe.DEFAULT_MODEL
pytest.importorskip("embreex", reason="the pipeline needs the package embreex (ray casting)")
pytestmark = pytest.mark.skipif(not os.path.isfile(PKL), reason="model pkl not found")
FAST = json.dumps({"c_outer": 6, "c_beta_outer": 2, "f_outer": 6})  # a short schedule: plumbing, not accuracy
STAGE_KEYS = [
    "global_rot",
    "joint_rot",
    "betas",
    "log_beta_scales",
    "trans",
    "deform_verts",
    "betas_trans",
    "log_scale",
    "rotation_mask",
    "verts",
    "pivots",
    "faces",
    "labels",
    "target_centre",
    "target_scale",
    "target_units",
    "target_frame_rotation",
    "target_frame_origin",
]


@pytest.fixture(scope="module")
def tpl(tmp_path_factory):
    regpipe.setup(PKL, cache=str(tmp_path_factory.mktemp("regpipe_cache")))
    from fitter_3d.geomreg.model import template

    return template(PKL)


def run_code(root, names, out, extra=()):
    """the one command, in this process; returns its exit code"""
    from fitter_3d.antscan_registration import run as RUN

    nf = f"{root}/names_{os.path.basename(out)}.txt"
    open(nf, "w").write("# a comment line\n" + "\n".join(names) + "\n")
    return RUN.main(
        [
            "--names",
            nf,
            "--joints",
            f"{root}/joints",
            "--limbs",
            f"{root}/limbs",
            "--env",
            f"{root}/env",
            "--out",
            out,
            "--geom-opts",
            FAST,
            "--jobs",
            "1",
            "--stages",
            "none",
            *extra,
        ]
    )


def run_cmd(root, names, out, extra=()):
    assert run_code(root, names, out, extra) == 0
    return list(csv.DictReader(open(f"{out}/summary.csv")))


@pytest.fixture(scope="module")
def batch(tpl, tmp_path_factory):
    """one specimen through the one command (one is enough and keeps the suite fast): an ensemble product with
    low-confidence joints (a mandible base, a distal leg), a joint between the cut and 0.3, and a canonical frame that
    is not the scan frame"""
    from antscan_synth_inputs import frame_of, make_specimen

    root = str(tmp_path_factory.mktemp("regpipe"))
    V = {}
    V["syn_b"], d = make_specimen(
        root, tpl, "syn_b", seed=2, low=("ma_l", "l_2_ta_l", "l_2_pt_l"), ensemble=True, frame=frame_of(2)
    )
    # one joint that three of four states agree on and the ensemble's largest-spread factor put below 0.3
    d["confidence"]["l_1_fe_r"] = round(0.9 * IN.g_spread(5.0), 3)
    d["joint_detail"]["l_1_fe_r"]["ens"].update(spread_pct=5.0, spread_med_pct=0.1)
    json.dump(d, open(f"{root}/joints/syn_b/syn_b_joints.json", "w"))
    rows = run_cmd(root, ["syn_b"], f"{root}/out")
    return dict(root=root, out=f"{root}/out", rows=rows, V=V)


def test_intake_rule():
    d = {
        "joints_scan": {"a": [0, 0, 0], "b": [1, 0, 0], "c": [2, 0, 0], "d": None, "b_h": [5, 0, 0]},
        "body_length_um": 100.0,
        "confidence": {"a": 0.9, "b": 0.27, "c": 0.1, "d": 0.0, "b_h": 0.027},
        "joint_detail": {},
    }
    out, blk = IN.intake(d)  # a single-state product is taken as it is
    assert blk["cut"] == 0.3 and out["confidence"] == d["confidence"] and not blk["promoted"] and not blk["majority"]
    g = IN.g_spread
    e = dict(
        d,
        ensemble={"states": ["s0", "s1", "s2", "s3"], "kept": ["s0", "s1", "s2", "s3"]},
        joint_detail={
            # three states agree within 0.2 % BL, one is 6 % away: the ensemble's value 0.4 * g(6) = 0.04
            "c": {
                "ens": {
                    "n_states": 4,
                    "n_cluster": 4,
                    "n_placed": 4,
                    "spread_pct": 6.0,
                    "spread_med_pct": 0.2,
                    "conf_states_median": 0.4,
                    "limb_state_frac": 1.0,
                    "placed_share": 1.0,
                }
            },
            # two against two, 3 % apart: stays low, and is a tie
            "b": {
                "ens": {
                    "n_states": 4,
                    "n_cluster": 4,
                    "n_placed": 4,
                    "spread_pct": 3.2,
                    "spread_med_pct": 3.0,
                    "conf_states_median": 0.9,
                    "limb_state_frac": 1.0,
                    "placed_share": 1.0,
                }
            },
            "b_h": {"ens": {"n_states": 4, "n_placed": 4, "spread_pct": 10.4, "conf_states_median": 0.754}},
        },
    )
    e["confidence"] = dict(d["confidence"], c=round(0.4 * g(6.0), 3), b=round(0.9 * g(3.2), 3))
    out, blk = IN.intake(e)  # default: limb joints as given (the recovery step is their way back)
    assert out["confidence"]["c"] == e["confidence"]["c"] and not blk["raised_to_usable"]
    assert out["confidence"]["b"] < 0.3 and blk["ties"] == {"b": "joint"}  # the tie is recorded and stays unusable
    out, blk = IN.intake(e, majority="all")
    assert abs(out["confidence"]["c"] - round(0.4 * g(0.2), 3)) < 1e-9 and blk["raised_to_usable"] == ["c"]
    assert out["confidence"]["b"] < 0.3  # ... a tie stays low under the majority spread as well
    assert out["confidence"]["b_h"] == 0.027 and "not available" in blk["trunk_majority"]  # no per-state products
    assert (
        out["confidence_input"] == e["confidence"] and out["joints_scan"] == e["joints_scan"]
    )  # positions never touched
    st = [
        {"joints_scan": {"b_h": [5.0, 0, 0]}},
        {"joints_scan": {"b_h": [5.1, 0, 0]}},
        {"joints_scan": {"b_h": [5.2, 0, 0]}},
        {"joints_scan": {"b_h": [15.4, 0, 0]}},
    ]
    out, blk = IN.intake(e, states=st)  # the trunk joint three states agree on (within 0.2 % BL) is usable again
    assert (
        abs(out["confidence"]["b_h"] - round(0.754 * g(0.2), 3)) < 1e-9
        and blk["majority"]["b_h"]["source"] == "per-state products"
    )
    out, blk = IN.intake(e, states=st, majority="none")
    assert out["confidence"] == e["confidence"]
    out, blk = IN.intake(e, cut=0.2)  # another cut: promotion to the constant every part holds
    assert out["confidence"]["b"] == 0.3 and blk["promoted"] == ["b"]
    out, blk = IN.intake(dict(e, confidence=dict(e["confidence"], b=0.5)), tie_policy="recover")
    assert out["confidence"]["b"] == IN.DEMOTED and blk["tie_demoted"] == ["b"]
    assert e["confidence"]["b_h"] == 0.027  # the input is not modified
    # far ties: a limb joint at 0.5 that two states place here and two states 10 % of the body length away is one of
    # two candidates chosen by noise: demoted for the recovery step; with three states here it is used
    f = dict(e, confidence=dict(e["confidence"], a=0.5))
    f["joint_detail"] = dict(e["joint_detail"], a={"ens": dict(e["joint_detail"]["c"]["ens"], n_cluster=2)})
    two = [{"joints_scan": {"a": x}} for x in ([0, 0, 0], [0.1, 0, 0], [10.0, 0, 0], [10.1, 0, 0])]
    three = [{"joints_scan": {"a": x}} for x in ([0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [10.1, 0, 0])]
    out, blk = IN.intake(f, states=two)
    assert (
        out["confidence"]["a"] == IN.DEMOTED
        and list(blk["tie_far"]) == ["a"]
        and blk["tie_far"]["a"]["spread_majority"] == 10.0
    )
    out, blk = IN.intake(f, states=three)
    assert out["confidence"]["a"] == 0.5 and blk["tie_far"] == {}
    out, blk = IN.intake(f, states=two, tie_policy="use")
    assert out["confidence"]["a"] == 0.5 and blk["tie_far_rule"] is None
    out, blk = IN.intake(f)  # no per-state products: the rule cannot be applied, and the block says so
    assert out["confidence"]["a"] == 0.5 and "not available" in blk["tie_far_rule"]
    for bad in ("0", "1.5", "abc", -0.2):
        with pytest.raises(ValueError):
            IN.parse_cut(bad)
    assert IN.parse_cut("auto") == 0.3 and IN.parse_cut("0.25") == 0.25
    assert IN.check_product({"joints_scan": {"a": [0, 0, 0]}, "confidence": {"a": 1}, "body_length_um": 5.0}) == (
        "joint product without frame"
    )


def test_one_command_writes_every_output_with_its_contract(batch, tpl):
    out, rows = batch["out"], batch["rows"]
    assert [r["name"] for r in rows] == ["syn_b"]
    for r in rows:
        n = r["name"]
        assert r["specimen_class"] in ("model_grade", "partial", "do_not_use")
        z = np.load(f"{out}/registration/{n}.npz", allow_pickle=True)
        assert z["V_final"].shape == (tpl.nV, 3) and np.isfinite(z["V_final"]).all()
        for k in (
            "global_rot",
            "joint_rot",
            "betas",
            "log_beta_scales",
            "trans",
            "deform_verts",
            "betas_trans",
            "log_scale",
            "rotation_mask",
            "geom_vertex_status",
            "geom_part_status",
            "geom_status",
            "registration_source",
            "target_scale",
            "target_frame_rotation",
        ):
            assert k in z.files, k
        # the registration's own codes never appear under the names the status step uses
        assert not {"vertex_status", "part_status", "status"} & set(z.files)
        assert set(np.unique(z["geom_vertex_status"])) <= {0, 1, 2}
        w = np.load(f"{out}/work/{z['registration_source']}/{n}.npz", allow_pickle=True)  # GEOM's own route file
        assert np.array_equal(w["V_final"], z["V_final"]) and np.array_equal(
            w["vertex_status"], z["geom_vertex_status"]
        )
        jd = json.load(open(f"{out}/joints/{n}/{n}_joints.json"))
        assert "intake" in jd and "confidence_input" in jd
        st = json.load(open(f"{out}/status/{n}_status.json"))
        assert len(st["vertex_status"]) == tpl.nV and len(st["part_code"]) == tpl.nJ and len(st["segments"]) == 34
        assert st["specimen"]["specimen_class"] == r["specimen_class"] and "grade" in st["specimen"]
        assert set(st["trunk_cover"]) == {"head", "mesosoma", "waist", "gaster", "all"}
        unc = st["trunk_cover"]["gaster"]["uncovered"]
        assert unc is not None and r["uncovered_gaster"] != ""
        # a gaster whose scan surface the mesh does not reach is never measurable, so never model grade
        assert unc <= 0.2 or ("b_a_3" in r["trunk_not_measurable"] and r["specimen_class"] != "model_grade")
        s = np.load(f"{out}/status/{n}_status.npz")
        assert s["V"].shape == (tpl.nV, 3) and s["vertex_status"].shape == (tpl.nV,)
        rt = np.load(f"{out}/routes/{n}.npz", allow_pickle=True)  # the shared benchmark's input contract
        assert (
            rt["V_final"].shape == (tpl.nV, 3)
            and rt["reg_v"].dtype == bool
            and str(rt["status"]) in ("ok", "review", "failed")
        )
        assert float(rt["um_per_model_unit"]) == float(z["um_per_model_unit"]) > 0  # the registration's global scale
        assert os.path.isfile(f"{out}/overlays/{n}.png")
    # the joints of syn_b: an ensemble product; its limb joint below 0.3 is left to the recovery step (never raised here)
    jb = json.load(open(f"{out}/joints/syn_b/syn_b_joints.json"))
    assert jb["intake"]["product_type"] == "ensemble4" and jb["intake"]["raised_to_usable"] == []
    assert jb["confidence_input"]["l_1_fe_r"] < 0.3 and jb["joints_scan"]["l_1_fe_r"] is not None
    for fn in ("morphometrics.csv", "morphometrics_flags.csv", "morphometrics_measurable.csv"):
        m = list(csv.DictReader(open(f"{out}/{fn}")))
        assert [x["name"] for x in m] == ["syn_b"] and "leg1l_femur_len" in m[0]
    for fn in ("run.json", "summary.json", "manifest.jsonl"):
        assert os.path.isfile(f"{out}/{fn}")
    # the unposed bundle for the Blender add-on
    b = np.load(f"{out}/export/uc1_bundle.npz", allow_pickle=True)
    assert b["verts"].shape == (1, tpl.nV, 3) and b["verts"].dtype == np.float32 and np.isfinite(b["verts"]).all()
    assert list(b["labels"]) == ["syn_b"]
    for k in ("valid", "measurable", "filled", "vertex_status", "fallback", "um_per_model_unit", "roundtrip_max_um"):
        assert k in b.files, k
    assert float(np.nanmax(b["roundtrip_max_um"])) < 1e-6  # UC1 unposing is exact where it does not fall back
    assert set(np.unique(b["vertex_status"][b["filled"]])) <= {5, 6}  # filled vertices are marked mirror / mean
    assert np.array_equal(b["filled"], np.isin(b["vertex_status"], (5, 6)))
    # kept vertices are the UC1 rest mesh of the registration, untouched by the fill
    from fitter_3d.antscan_registration.parts import uc1_unpose as U

    t = U.template(PKL)
    z = np.load(f"{out}/status/syn_b_status.npz")
    r = U.unpose(z["V"], np.isin(z["vertex_status"], (0, 1, 2, 3, 7)), t=t)
    keep = ~b["filled"][0] & ~b["stuck"][0]
    assert np.abs(r["V_rest"][keep] - b["verts"][0][keep]).max() < 1e-5


def test_a_bundle_row_does_not_depend_on_the_batch(batch):
    """with the population reference, the fill source, the length gate and the edge reference are the same for every
    batch: a specimen alone gets the row and the QA it gets next to another specimen"""
    from fitter_3d.antscan_registration import fill as FI

    if FI.population() is None:
        pytest.skip("no population reference shipped")
    from fitter_3d.antscan_registration.parts import uc1_unpose as U

    t = U.template(PKL)
    legs = np.array([n.startswith("l_3") for n in t.names])[t.dom]  # the hind legs: unregistered in the first specimen
    items = []
    for i in range(7):  # seven specimens of different proportions: enough for a batch mean of its own
        z = np.load(f"{batch['out']}/status/syn_b_status.npz")
        vs = z["vertex_status"].copy()
        if i == 0:
            vs[legs] = 4
        V = (z["V"] - z["V"].mean(0)) * (1.0 + 0.04 * i) + z["V"].mean(0)
        V = V + 0.02 * i * (V - V.mean(0)) * np.array([1.0, 0.0, -1.0])  # not a similarity: other rest proportions
        items.append(dict(name=f"s{i}", V_scan=V, vertex_status=vs))
    many, qa7 = FI.make_bundle(items, None)
    one, qa1 = FI.make_bundle(items[:1], None)
    assert many["filled"][0].sum() > 300 and (many["vertex_status"][0][legs] == 6).all()  # filled from the mean part
    assert np.array_equal(many["verts"][0], one["verts"][0]) and np.array_equal(
        many["vertex_status"][0], one["vertex_status"][0]
    )
    assert qa7[0] == qa1[0]
    # without the population reference the mean part is the batch's own, and the row is another one
    own, _ = FI.make_bundle(items, None, use_population=False)
    assert not np.array_equal(own["verts"][0][legs], many["verts"][0][legs])


@pytest.mark.slow  # runs the command on further specimens: not in the fast suite (pytest -m 'not slow')
def test_degenerate_inputs_end_as_statuses(tpl, tmp_path):
    from antscan_synth_inputs import make_specimen

    root = str(tmp_path)
    make_specimen(root, tpl, "good", seed=3)
    _, d = make_specimen(root, tpl, "all_null", seed=4)
    d["joints_scan"] = {k: None for k in d["joints_scan"]}
    json.dump(d, open(f"{root}/joints/all_null/all_null_joints.json", "w"))
    _, d = make_specimen(root, tpl, "nan_joints", seed=5)
    d["joints_scan"]["b_h"] = [float("nan")] * 3
    d["joints_scan"]["b_a_1"] = [float("inf"), 0, 0]
    json.dump(d, open(f"{root}/joints/nan_joints/nan_joints_joints.json", "w"))
    make_specimen(root, tpl, "no_envelope", seed=6)
    os.remove(f"{root}/env/no_envelope/no_envelope_envelope.stl")
    _, d = make_specimen(
        root, tpl, "few_joints", seed=7, null=tuple(n for n in tpl.names if n.startswith(("l_", "an_", "ma_")))
    )
    make_specimen(root, tpl, "broken_json", seed=8)
    open(f"{root}/joints/broken_json/broken_json_joints.json", "w").write("{ not json")
    _, d = make_specimen(root, tpl, "no_frame", seed=9)
    del d["frame"]
    json.dump(d, open(f"{root}/joints/no_frame/no_frame_joints.json", "w"))
    make_specimen(root, tpl, "cut_stl", seed=10)
    raw = open(f"{root}/env/cut_stl/cut_stl_envelope.stl", "rb").read()
    open(f"{root}/env/cut_stl/cut_stl_envelope.stl", "wb").write(raw[: len(raw) // 3])
    make_specimen(root, tpl, "no_limb_npz", seed=11)
    os.remove(f"{root}/limbs/no_limb_npz/limbs.npz")
    names = ["good", "all_null", "nan_joints", "no_envelope", "few_joints", "broken_json", "no_frame", "cut_stl"]
    names += ["no_limb_npz", "not_there"]
    rows = run_cmd(root, names, f"{root}/out", extra=("--no-overlay",))
    by = {r["name"]: r for r in rows}
    assert list(by) == names  # one row per name, in file order, whatever happened
    assert by["good"]["specimen_class"] in ("model_grade", "partial")
    for n in ("all_null", "no_envelope", "broken_json", "no_frame", "cut_stl", "not_there"):
        assert by[n]["specimen_class"] == "do_not_use" and by[n]["class_reasons"], n
    for n in ("nan_joints", "few_joints", "no_limb_npz"):
        assert by[n]["specimen_class"] in ("model_grade", "partial", "do_not_use"), n
    man = {json.loads(ln)["name"]: json.loads(ln) for ln in open(f"{root}/out/manifest.jsonl")}
    assert "without frame" in man["no_frame"]["intake"]["error"]  # the reason is the missing block, not a KeyError
    assert os.path.isfile(f"{root}/out/export/batch_0/Stage_status.npz")
    # a batch in which NO specimen gets through ends with exit code 3 and still writes its table
    assert run_code(root, ["all_null", "not_there"], f"{root}/out_none", extra=("--no-overlay",)) == 3
    assert len(list(csv.DictReader(open(f"{root}/out_none/summary.csv")))) == 2


def test_stage_file_round_trips_and_closes_fitter3d_own_forward(batch, tpl):
    """The stage file in the scan frame is the registration (syn_b has a frame that is not the scan frame: a
    transposed rotation or a wrong origin sign fails here); the native parameters + deform_verts give verts under
    geomreg's forward with the model file's skinning rows AS STORED, and under SMAL3DFitter itself (in a process of
    its own: config.set_model is global)."""
    from fitter_3d.geomreg import native

    out = batch["out"]
    z = np.load(f"{out}/export/batch_0/Stage_status.npz", allow_pickle=True)
    for k in STAGE_KEYS + ["status_vertex", "status_part_status", "status_specimen_class", "status_params_source"]:
        assert k in z.files, k
    B = len(z["labels"])
    assert B == 1 and z["verts"].shape == (B, tpl.nV, 3) and z["joint_rot"].shape == (B, tpl.nJ - 1, 3)
    assert z["log_beta_scales"].shape == (B, tpl.nJ, 3) and z["faces"].shape == (B, len(tpl.F), 3)
    for i, lab in enumerate(z["labels"]):
        reg = np.load(f"{out}/registration/{lab}.npz", allow_pickle=True)
        # stage file -> scan frame = the registration (float32 storage)
        Vb = (z["verts"][i].astype(np.float64) * float(z["target_scale"][i]) + z["target_centre"][i]) @ z[
            "target_frame_rotation"
        ][i] + z["target_frame_origin"][i]
        assert np.abs(Vb - reg["V_final"]).max() < 0.05  # micrometres
        # the native parameters carry the mesh: the model's forward + deform_verts = verts
        p = {
            k: np.asarray(z[k][i], np.float64)
            for k in (
                "global_rot",
                "joint_rot",
                "betas",
                "log_beta_scales",
                "trans",
                "deform_verts",
                "betas_trans",
                "log_scale",
                "rotation_mask",
            )
        }
        v, _ = native.forward_np(tpl, p, weights=tpl.W_raw)
        assert np.abs(v - z["verts"][i]).max() < 1e-5
        if np.abs(tpl.W_raw.sum(1) - 1).max() > 1e-3:  # this model has rows that do not sum to 1: they matter
            assert np.abs(native.forward_np(tpl, p)[0] - z["verts"][i]).max() > 1e-3
        assert "native" in str(z["status_params_source"][i])
        assert (
            np.linalg.norm(p["deform_verts"], axis=1).mean()
            < 0.5 * np.linalg.norm(z["verts"][i] - z["verts"][i].mean(0), axis=1).mean()
        )
    # SMAL3DFitter's own forward, in a process of its own
    import subprocess

    tree = HERE
    rep = f"{batch['root']}/stage_forward.json"
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "antscan_stage_forward_helper.py")
    r = subprocess.run(
        [sys.executable, helper, tree, f"{batch['out']}/export/batch_0/Stage_status.npz", PKL, rep],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    if r.returncode == 3:  # a fitter without the global scale (before the joint path): it cannot hold the stage file
        pytest.skip(r.stdout.strip())
    assert r.returncode == 0, r.stderr[-1500:]
    f = json.load(open(rep))
    assert f["n"] == 1 and f["max_um"] < 0.05  # the fitter's forward as it is: closed to float32 rounding


def test_grade_conditions():
    from fitter_3d.antscan_registration import grade as GR

    spec = dict(
        specimen_class="partial",
        n_limb_segments=34,
        n_measurable=31,
        trunk_not_measurable=[],
        reasons=[],
        head_end=dict(state="ok", roll_doubt=[]),
    )
    seg = {f"an_{k}_{s}": [0, []] for k in (1, 2) for s in "lr"}
    seg.update({f"x{i}": [4, []] for i in range(3)})  # three segments not measurable
    fq = dict(n_trunk_filled=0, n_abnormal=0, n_abnormal_gt10=0, gated_bones=[])
    reg = dict(status="ok", reasons=[], forward_err_um=0.0)
    ok = dict(tubes={"scape_r": dict(zigzag_excess_deg=20.0)}, head_roll=dict(best_deg=0.0, contrast=1.0))
    g = GR.decide(spec, seg, None, fq, ok, reg)
    assert g["specimen_class"] == "model_grade" and g["n_segments_filled"] == 3
    six = dict(seg, **{f"y{i}": [3, []] for i in range(3)})
    assert GR.decide(spec, six, None, fq, ok, reg)["failed"] == ["M2_limbs"]  # 6 segments filled
    # a segment that is measurable but filled by the bundle's length gate counts as filled
    sb = {"an_1_l": ["an_1_l"], "an_2_l": ["an_2_l", "an_3_l"], "an_1_r": ["an_1_r"], "an_2_r": ["an_2_r", "an_3_r"]}
    gated = dict(fq, gated_bones=["an_1_l", "an_2_l", "an_3_l", "an_2_r"])
    g2 = GR.decide(spec, seg, None, gated, ok, reg, seg_bones=sb)
    assert g2["n_segments_filled"] == 6 and g2["failed"] == ["M2_limbs"]
    assert g2["segments_filled_by_gate"] == ["an_1_l", "an_2_l", "an_2_r"]
    assert GR.decide(dict(spec, trunk_not_measurable=["b_a_1"]), seg, None, fq, ok, reg)["failed"] == ["M1_trunk"]
    assert GR.decide(spec, seg, None, dict(fq, n_abnormal=2, n_abnormal_gt10=1), ok, reg)["failed"] == ["M4_edges"]
    assert (
        GR.decide(spec, seg, None, dict(fq, n_abnormal=2), ok, reg)["specimen_class"] == "model_grade"
    )  # short stretched edges are reported only
    assert GR.decide(spec, seg, None, dict(fq, n_trunk_filled=3), ok, reg)["failed"] == ["M3_trunk_fill"]
    assert (
        GR.decide(dict(spec, specimen_class="do_not_use", reasons=["x"]), seg, None, fq, ok, reg)["specimen_class"]
        == "do_not_use"
    )
    # a condition that could not be checked is NOT met: no bundle, or the checks failed
    g = GR.decide(spec, seg, None, None, ok, reg)
    assert g["specimen_class"] == "partial" and g["unchecked"] == ["M3_trunk_fill", "M4_edges"]
    assert "could not be checked" in " ".join(g["reasons"])
    g = GR.decide(spec, seg, None, fq, None, reg)
    assert g["specimen_class"] == "partial" and g["unchecked"] == ["M5_antennae"]
    # ... while a specimen without a measurable antenna has nothing to be crumpled
    none = {k: ([4, []] if k.startswith("an_") else v) for k, v in seg.items()}
    assert GR.decide(dict(spec, n_measurable=27), none, None, fq, dict(tubes={}, head_roll={}), reg)["failed"] == [
        "M2_limbs"
    ]
    ck = dict(
        tubes={"funiculus_l": dict(zigzag_excess_deg=87.0), "scape_r": dict(zigzag_excess_deg=20.0)},
        head_roll=dict(best_deg=0.0, contrast=1.0, cost0=0.01, cost_best=0.01),
    )
    assert GR.decide(spec, seg, None, fq, ck, reg)["failed"] == ["M5_antennae"]  # a zigzag registered funiculus
    assert (
        GR.decide(spec, dict(seg, an_2_l=[4, []]), None, fq, ck, reg)["specimen_class"] == "model_grade"
    )  # ... unless it is filled anyway
    # head roll: measured and reported, gating only on request, never at the end of its search range
    ck2 = dict(tubes=ok["tubes"], head_roll=dict(best_deg=-30.0, contrast=2.0, cost0=0.02, cost_best=0.01))
    g = GR.decide(spec, seg, None, fq, ck2, reg)
    assert g["specimen_class"] == "model_grade" and g["head_roll"]["flag"] is True
    assert GR.decide(spec, seg, None, fq, ck2, reg, K=dict(ROLL_GATE=True))["failed"] == ["M6_head_roll"]
    edge = dict(ck2, head_roll=dict(ck2["head_roll"], best_deg=-60.0, at_range_edge=True))
    assert GR.decide(spec, seg, None, fq, edge, reg, K=dict(ROLL_GATE=True))["specimen_class"] == "model_grade"
    r2 = dict(reg, reasons=["root placed by root_chord+frame_prior"])
    assert GR.decide(spec, seg, None, fq, ok, r2)["specimen_class"] == "do_not_use"
    assert GR.decide(spec, seg, None, fq, ok, dict(reg, status="failed"))["specimen_class"] == "do_not_use"
