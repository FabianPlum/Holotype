"""Per-specimen workers of the status step: evidence of one specimen (cached), the batch-relative plausibility signal
of the segment lengths, and the third pass (UC1 rest mesh, morphometrics, stage row). They are driven by the
pipeline's status step; a failure on a specimen is returned as an error text, never raised."""

import json
import os
import traceback

import numpy as np

from fitter_3d.antscan_registration.parts.rs import io as rio
from fitter_3d.antscan_registration.parts.rs import status as S


def _ev_one(args):
    name, a = args
    p = os.path.join(a["out"], "evidence", f"{name}.json")
    if os.path.isfile(p) and not a.get("force"):
        return name, "cached", None
    try:
        jf = rio.joint_file(a["joints_dir"], name)
        if jf is None:
            raise FileNotFoundError("no joint product")
        reg = rio.read_registration(a["regs"], name, a["reg_format"], jfile=jf)
        env = rio.envelope(a["env_tmpl"], name)
        if not os.path.isfile(env):
            raise FileNotFoundError(f"no envelope {env}")
        E = S.evidence_one(name, env, jf, rio.limbs_dir(a["limbs_dir"], name), reg)
        tmp = p + ".tmp"
        S.save_evidence(E, tmp)
        os.replace(tmp, p)
        return name, "ok", None
    except Exception as e:  # a failure is a status, never a batch stop
        return name, "error", f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-600:]}"


def population_z(EV, dec0, min_ref=8):
    """robust z of log(segment marker chord / joint-product BL) of every limb segment against the OTHER specimens of
    the batch whose segment is measurable in the first decision (leave-one-out; flag only)."""
    names = list(EV)
    segs = S.seg_names()
    X = np.full((len(names), len(segs)), np.nan)
    ok = np.zeros_like(X, bool)
    for i, n in enumerate(names):
        E = EV[n]
        for k, s in enumerate(segs):
            L = E["segments"][s]["length"]
            if L > 0 and E["BL"]:
                X[i, k] = np.log(L / E["BL"])
            ok[i, k] = dec0[n][s][0] in S.MEASURABLE
    out = {}
    for i, n in enumerate(names):
        z = {}
        for k, s in enumerate(segs):
            ref = np.delete(X[:, k], i)[np.delete(ok[:, k], i)]
            ref = ref[np.isfinite(ref)]
            if len(ref) < min_ref or not np.isfinite(X[i, k]):
                z[s] = None
                continue
            med = np.median(ref)
            mad = 1.4826 * np.median(np.abs(ref - med))
            z[s] = float((X[i, k] - med) / max(mad, 1e-6))
        out[n] = z
    return out


def _post_one(args):
    """pass 3 for one specimen: UC1, morphometrics, stage row, overlay."""
    name, a, dec_path = args
    try:
        from fitter_3d.antscan_registration.parts import uc1_unpose as U
        from fitter_3d.antscan_registration.parts.rs import export as X
        from fitter_3d.antscan_registration.parts.rs import morpho as MO
        from fitter_3d.antscan_registration.parts.skelreg import geom

        D = json.load(open(dec_path))
        jf = rio.joint_file(a["joints_dir"], name)
        reg = rio.read_registration(a["regs"], name, a["reg_format"], jfile=jf)
        V = reg["V"]
        vs = np.asarray(D["vertex_status"], int)
        code = np.asarray(D["part_code"], int)
        env = rio.envelope(a["env_tmpl"], name)
        Vs, Fs = geom.load_mesh(env)
        BL = D["BL"]
        obs, dsc = MO.observed_vertices(V, Vs, Fs, BL)
        u = U.unpose(V, np.isin(vs, S.REGISTERED))
        row, flags, detail = MO.measure(
            V, code, obs, u, D.get("waist_segments"), D["specimen"]["specimen_class"], xsec=not a.get("no_xsec")
        )
        res = dict(
            name=name,
            row=row,
            flags=flags,
            detail=detail,
            uc1=dict(s=u["s"], fallback=int(u["fallback"].sum()), roundtrip_max_um=u["roundtrip_max_um"]),
        )
        if not a.get("no_export"):
            native = reg.get("native")
            if native is None and a.get("native"):
                try:
                    _, native, _ = rio.read_stage_row(a["native"], name, jf)
                except Exception as e:
                    native, res["native_error"] = None, str(e)
            if native is not None:
                srow, src = X.stage_row_from_native(native, V)
            else:
                srow, src = X.stage_row_default(V, jf, Vs)
            res["stage_extra"] = {k[1:]: srow.pop(k) for k in [k for k in srow if k.startswith("_")]}
            sp = os.path.join(a["out"], "export", "stage_rows", f"{name}.npz")
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            np.savez(sp, **srow)
            res["params_source"] = src
            # round trip: written verts back to the scan frame
            Vb = (
                np.asarray(srow["verts"], np.float32).astype(np.float64) * float(srow["target_scale"])
                + np.asarray(srow["target_centre"])
            ) @ np.asarray(srow["target_frame_rotation"]) + np.asarray(srow["target_frame_origin"])
            res["stage_roundtrip_um"] = float(np.abs(Vb - V).max())
        if not a.get("no_overlay"):
            from fitter_3d.antscan_registration.parts.rs import overlay as OV

            jd = json.load(open(jf))
            seg = {k: v for k, v in D["segments"].items()}
            OV.render_one(
                name,
                V,
                code,
                seg,
                jd["frame"]["rotation"],
                env,
                rio.limbs_dir(a["limbs_dir"], name),
                D["specimen"],
                os.path.join(a["out"], "overlays", f"{name}_status.png"),
                BL,
            )
        return res
    except Exception as e:
        return dict(name=name, error=f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-800:]}")
