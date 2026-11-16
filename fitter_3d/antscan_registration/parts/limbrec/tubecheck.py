"""Independent scan evidence for candidate joints: SKEL-REG's tube tracing and limb verification (stage A / A',
parts/skelreg) run on an arbitrary joint set. A limb segment between two joints is traced on the
ENVELOPE MESH as a tube from centred cross-sections; nothing here reads the joint method's paths, so the verdict is
independent of how a candidate joint was found.

Per segment (template bone name = the segment's proximal joint):
  verdict     ok (tube traced, self-consistent, girth like the specimen's other limbs) / contact (not traced, lies on
              limb-like material) / fail / duplicate (same scan surface as another chain's tube)
  inner_step  largest radius step inside the traced tube (an articulation INSIDE a segment = joint misplaced along it)
  joint_feature  for the segment's DISTAL joint, when both neighbouring tubes are traced: offset of the nearest radius
              minimum of the scan profile from the joint, as a share of the two segments' length (None: no minimum)
  length, chord, radius (median section radius), tip_found
"""

import numpy as np

from . import common  # noqa: F401


class TubeVerifier:
    def __init__(self, name, envelope_path, joint_path):
        from fitter_3d.antscan_registration.parts.skelreg import sources, tubes
        from fitter_3d.antscan_registration.parts.skelreg.model import template
        from fitter_3d.antscan_registration.parts.skelreg.scan import load_scan

        self.tpl = template("medial")
        self.tc = tubes.template_charts("medial")
        jt = sources.load(name, "s6", self.tpl, path=joint_path)
        if jt is None:
            raise RuntimeError("joint file not readable")
        self.P0 = jt["P"].copy()  # usable joints only (confidence rule + gaster / neck conventions)
        ok = np.isfinite(self.P0).all(1)
        if ok.sum() < 6:
            raise RuntimeError("fewer than 6 usable joints")
        self.size = float(np.linalg.norm(self.P0[ok].max(0) - self.P0[ok].min(0)))
        self.scan = load_scan(envelope_path, self.size, assume_outer=True)
        self.idx = self.tpl.idx

    def evaluate(self, extra=None, drop=None):
        """extra: dict joint name -> xyz (candidates added to the usable joints); drop: names removed.
        Returns dict(segments={bone: {...}}, P=joint array used, scale)."""
        from fitter_3d.antscan_registration.parts.skelreg import diagnose, tubes, verify
        from fitter_3d.antscan_registration.parts.skelreg.bones import BoneFit

        tpl = self.tpl
        P = self.P0.copy()
        for n, x in (extra or {}).items():
            P[self.idx[n]] = np.asarray(x, float)
        for n in drop or []:
            P[self.idx[n]] = np.nan
        bf = BoneFit(tpl, P).solve(lookahead=True)
        st = tubes.trace_scan_tubes(self.scan, tpl, self.tc, bf, twist=False)
        ver, _ = verify.verify_limbs(self.scan, tpl, self.tc, st, bf)
        jf = diagnose.joint_feature(tpl, st)
        seg = {}
        for s in st:
            if not s.resolved:
                continue
            v = ver.get(s.name, {})
            q = s.qa
            rad = (
                float(np.median(s.tube.rmean))
                if (s.tube is not None and s.tube.rmean is not None and np.isfinite(s.tube.rmean).any())
                else None
            )
            child = tpl.names[s.child]
            seg[s.name] = dict(
                verdict=v.get("verdict"),
                use=bool(q.get("use")),
                traced=bool(q.get("traced")),
                inner_step=q.get("inner_step"),
                length=q.get("length"),
                chord=q.get("chord"),
                radius=rad,
                radius_ratio=q.get("radius_ratio"),
                girth_vs_limbs=q.get("girth_vs_limbs"),
                tip_found=q.get("tip_found"),
                note=q.get("note") or v.get("reason") or "",
                chord_support=v.get("chord_support"),
                child=child,
                joint_feature=jf.get(child),
                ok_share=q.get("ok_share"),
                valid_share=q.get("valid_share"),
                rmean=(
                    None
                    if (s.tube is None or s.tube.rmean is None or not q.get("use"))
                    else [float(v) for v in s.tube.rmean]
                ),
            )
        return dict(segments=seg, P=P, scale=float(bf.s))
