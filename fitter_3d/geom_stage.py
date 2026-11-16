"""Geometric registration from a joint product, as a fitter_3d stage (option; nothing here runs unless a yaml stage
says `scheme: 'geom'`).

One stage replaces the joint-informed schedule's skeleton stages AND its free per-vertex deformation stages:

    closed-form articulated start in the fitter's own parameters (fitter_3d/geomreg/native.py: global_rot, joint_rot,
    log_beta_scales, betas_trans, trans, log_scale; no optimisation, no GPU; bones without usable joints keep the rest
    articulation or the mirror of their partner)
    -> trunk girth from the scan (closed form) -> limb tube warp along the limb centrelines (closed form, --limbs_dir)
    -> one affine map per joint-supported bone -> free vertices with template-relative regularisers and an anti-fold
    barrier (plain PyTorch loss terms, CPU by default) -> per-vertex and per-part status
    -> native parameters refitted to the registered mesh (closed-form block coordinate descent, three scales per bone,
    optional betas); deform_verts = what the parameters cannot express

yaml:
  Stage_G_geom:
    scheme: 'geom'
    geom: {}                 # options of geomreg.surface.OPTS / {'start': {...}} for geomreg.native.OPTS, all optional
                             # (unknown keys here and in the stage itself are an error)
    refit: true              # default true: joint_rot / log_beta_scales / betas_trans carry as much as they can
    refit_betas: false       # default false: fit the shape coefficients only when the shape space was accepted
    geom_device: 'cpu'       # default 'cpu': one thread inside the stage, two runs give bit-identical files ('cuda': about
                             # four times faster, not bit-equal to the CPU result; geom: {cpu_threads: 0} lifts the
                             # one-thread limit and the guarantee)
Needs optimise.py --joints_dir (targets, frame) and, for the limb tube warp and the mandible tips, --limbs_dir (limb
product of the joint method, <name>/limbs.json + limbs.npz). Any parameters left by earlier stages are overwritten.
Output: the usual stage .npz (Stage.save_npz) with the extra keys `geom_vertex_status` (B, V) int8 (0 observed,
1 hidden, 2 unsupported), `geom_part_status` (B, J) str (ok / review / unregistered), `geom_status` (B,) str (ok /
review / failed), `start_how` (B, J) str. A specimen that cannot be registered keeps its incoming parameters, gets
deform_verts = 0, geom_vertex_status 3 and geom_status 'failed' (no exception leaves the stage).
"""

import json
import os
import time

import numpy as np
import torch

import config
from fitter_3d.geomreg import model as g_model
from fitter_3d.geomreg import native as g_native
from fitter_3d.geomreg import refit as g_refit
from fitter_3d.geomreg import spec as g_spec
from fitter_3d.geomreg import surface as g_surface
from fitter_3d.trainer import Stage

PARAMS = ("global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "betas_trans", "log_scale")
YAML_KEYS = ("scheme", "geom", "refit", "refit_betas", "geom_device")  # what a `scheme: 'geom'` yaml stage may set


def check_stage_options(kwargs):
    """Raise ValueError for yaml keys of a geom stage that the stage does not read (a typo would be ignored)."""
    bad = sorted(set(kwargs) - set(YAML_KEYS))
    if bad:
        raise ValueError(f"yaml stage with scheme 'geom': unknown key(s) {bad}; known: {list(YAML_KEYS)}")


def stem_of(mesh_name):
    stem = os.path.splitext(os.path.basename(mesh_name))[0]
    return stem[: -len("_processed")] if stem.endswith("_processed") else stem


def joint_file(joints_dir, stem):
    p = os.path.join(joints_dir, f"{stem}_joints.json")
    return p if os.path.isfile(p) else os.path.join(joints_dir, stem, f"{stem}_joints.json")


class GeomStage(Stage):
    def __init__(
        self,
        smal_3d_fitter,
        target_meshes,
        mesh_names=(),
        name="geom",
        out_dir="static_fits_output",
        device="cuda",
        geom=None,
        refit=True,
        refit_betas=False,
        joints_dir=None,
        limbs_dir=None,
        joint_kind="pivot",
        conf_min=0.0,
        target_norm=None,
        target_joints=None,
        joint_weights=None,
        body_length=None,
        **unused,
    ):
        # Stage.__init__ is used for its bookkeeping only (reference verts, faces, loss weights); no optimiser steps
        stage_device = device
        super().__init__(
            nits=0,
            scheme="deform",
            smal_3d_fitter=smal_3d_fitter,
            target_meshes=target_meshes,
            mesh_names=list(mesh_names),
            name=name,
            out_dir=out_dir,
            device=stage_device,
            target_joints=target_joints,
            joint_weights=joint_weights,
            body_length=body_length,
            target_norm=target_norm,
        )
        if joints_dir is None or target_norm is None:
            raise ValueError("scheme 'geom' needs optimise.py --joints_dir (joint targets and the frame)")
        self.opts = dict(geom or {})
        self.geom_device = self.opts.pop("device", unused.get("geom_device", "cpu"))
        g_surface.check_opts(self.opts)  # a mistyped option is an error, not ignored
        self.do_refit, self.refit_betas = bool(refit), bool(refit_betas)
        self.joints_dir, self.limbs_dir, self.joint_kind, self.conf_min = joints_dir, limbs_dir, joint_kind, conf_min
        self.vertex_status = self.part_status = self.specimen_status = self.start_how = None
        self.geom_log = []

    def run(self, plot=False):
        g_model.set_default_model(config.SMAL_FILE)
        tpl = g_model.template()
        f = self.smal_3d_fitter
        B = f.batch_size
        tn = self.target_norm
        tv, tf = self.target_meshes.verts_list(), self.target_meshes.faces_list()
        vstat = np.full((B, tpl.nV), 3, np.int8)
        pstat = np.full((B, tpl.nJ), "unregistered", dtype="<U12")
        sstat = np.full(B, "failed", dtype="<U8")
        how = np.full((B, tpl.nJ), "", dtype="<U28")
        dev = self.geom_device if (self.geom_device == "cpu" or torch.cuda.is_available()) else "cpu"
        with torch.no_grad():
            f.deform_verts.zero_()
        Vfin = np.zeros((B, tpl.nV, 3))
        done = np.zeros(B, bool)
        for b in range(B):
            t0 = time.time()
            stem = stem_of(self.mesh_names[b])
            try:
                fr = g_spec.Frame(
                    np.asarray(tn["target_frame_rotation"][b]),
                    np.asarray(tn["target_frame_origin"][b]),
                    np.asarray(tn["target_centre"][b]),
                    float(tn["target_scale"][b]),
                )
                sp = g_spec.Specimen(
                    stem,
                    None,
                    joint_file(self.joints_dir, stem),
                    limb_dir=os.path.join(self.limbs_dir, stem) if self.limbs_dir else None,
                    conf_min=self.conf_min,
                    kind=self.joint_kind,
                    tpl=tpl,
                    frame=fr,
                    scan_norm=(tv[b].detach().cpu().numpy(), tf[b].detach().cpu().numpy()),
                )
                if sp.BL is None:
                    raise g_native.StartError("the joint file has no body_length_um")
                tips = {j: q["tip"] for j, q in sp.mand.items() if sp.w[j] > 0}
                one = dev == "cpu" and int(self.opts.get("cpu_threads", g_surface.OPTS["cpu_threads"])) == 1
                with g_surface.single_thread(one):  # CPU: bit-reproducible (one thread for torch and BLAS)
                    st = g_native.closed_form_start(
                        tpl,
                        sp.P,
                        sp.w,
                        kind=self.joint_kind,
                        R_prior=np.eye(3),
                        BL=sp.BL,
                        head_front=sp.head_front,
                        tips=tips,
                        opts=self.opts.get("start"),
                    )
                    p = g_native.to_native(st)
                    res = g_surface.register(sp, st, {k: v for k, v in self.opts.items() if k != "start"}, device=dev)
                    Vn = res["V"]
                    rinfo = None
                    if self.do_refit:
                        use_b = self.refit_betas or bool(res["info"].get("shape_space_used"))
                        p, rinfo = g_refit.native_refit(tpl, Vn, sp.support[tpl.dom], p, BL=sp.BL, use_betas=use_b)
                with torch.no_grad():
                    for k in PARAMS:
                        getattr(f, k)[b].copy_(torch.as_tensor(np.asarray(p[k]), dtype=torch.float32, device=f.device))
                    f.rotation_mask[b] = 1.0
                Vfin[b], done[b] = Vn, True
                vstat[b], pstat[b] = res["status"], res["part_status"]
                sstat[b], why = g_surface.specimen_status(tpl, res["part_status"], st, sp.flags)
                how[b] = np.array(st.how)
                self.geom_log.append(
                    dict(
                        name=stem,
                        seconds=time.time() - t0,
                        status=str(sstat[b]),
                        status_reasons=why,
                        status_counts=res["info"]["status_counts"],
                        start=st.info,
                        refit=rinfo,
                        coarse_shape=res["info"].get("coarse_shape"),
                        review=res["info"].get("review"),
                    )
                )
                print(f"[geom] {stem}: {time.time() - t0:.1f}s {sstat[b]} {res['info']['status_counts']}", flush=True)
            except Exception as e:  # a failure is a status, never an exception that stops a batch
                self.geom_log.append(
                    dict(name=stem, seconds=time.time() - t0, status="failed", error=f"{type(e).__name__}: {e}")
                )
                print(f"[geom] {stem}: FAILED {type(e).__name__}: {e}", flush=True)
        # deform_verts = registered mesh - the FITTER'S OWN forward of the written parameters, so that the stage file's
        # verts (Stage.save_npz calls the same forward) are the registered mesh whatever the fitter's skinning weights
        with torch.no_grad():
            v_lbs = f().detach()
            dv = torch.as_tensor(Vfin, dtype=torch.float32, device=f.device) - v_lbs
            dv[torch.as_tensor(~done, device=f.device)] = 0.0
            f.deform_verts.copy_(dv)
        self.vertex_status, self.part_status, self.specimen_status, self.start_how = vstat, pstat, sstat, how

    def save_npz(self, labels=None):
        super().save_npz(labels=labels)
        p = os.path.join(self.out_dir, f"{self.name}.npz")
        d = dict(np.load(p, allow_pickle=True))
        d["geom_vertex_status"], d["geom_part_status"] = self.vertex_status, self.part_status
        d["geom_status"], d["start_how"] = self.specimen_status, self.start_how
        np.savez(p, **d)
        json.dump(
            self.geom_log, open(os.path.join(self.out_dir, f"{self.name}_geom_log.json"), "w"), indent=1, default=str
        )
