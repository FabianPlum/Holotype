"""RECOVER: limb joints the scan-only joint method lacks or holds below confidence 0.3, found with what the model and
a first registration know, verified on the scan independently of how they were found. One specimen in, an AUGMENTED
joint product (same contract) and a recovery log out.

Kinds of evidence a candidate can come from (each can be switched off):
  lowconf    the product's own joint with confidence < 0.3 (position unchanged)
  fallback   the product's own joint with provenance "template_fallback" (round-3 fit), confidence < 0.3
  chainfit   the chain re-fitted on the limb's named centreline with absolute expected lengths (limbrec.chainfit)
  tail       as chainfit, on the centreline continued beyond its tip by a candidate polyline of the path stages
Verification (independent of the finding; all must hold, see `Judge.check`):
  tube       every segment between the joint and a trusted neighbour is traced on the envelope mesh as a
             self-consistent tube (SKEL-REG stage A', parts/skelreg) and is no duplicate of another chain's tube
  step       no articulation inside such a segment (radius step below the supported population's bound)
  length     the segment's length agrees with the bilateral partner's (when that one is directly supported), else with
             the population prediction from the specimen's own supported segments (leave-one-specimen-out)
  socket     a coxa base / antennal socket lies where the mirror image of its supported partner or the first
             registration's trunk puts it
  attached   every joint between it and the body is usable or recovered (a leg may start at the trochanter when the
             trochanter sits one coxa length from the predicted socket)
  along      its position ALONG the limb is pinned: partner lengths on both sides, or one length plus a scan feature
             (traced-tube radius minimum, fold / radius minimum / tip of the centreline), or two independent placements
             (the product's and the chain fit's) that agree
Nothing here reads truth, oracle limbs, specimen names (other than as file keys) or taxonomy.
"""

import time
import traceback

import numpy as np

from . import VERSION, chainfit
from .common import CHAINS, RIGID, bone_of_segment, partner, partner_joint, seg_key, seg_names
from .priors import coxa_fraction, supported_lengths
from .product import Product, load_limbs, load_pool, load_registration

KINDS = ("lowconf", "fallback", "chainfit", "tail")

DEFAULTS = dict(
    kinds=KINDS,
    agree_bl=0.015,  # product position and chain fit agree within this share of BL
    z_partner_ref=2.5,  # a partner segment is a length reference only when it is within this many sd of the
    # population prediction for the specimen (else the population is the reference)
    flex_serial_tol=0.35,  # ta / an_2 placed by the chain fit: the distal flexible chord must agree with the partner's
    z_pop=2.5,  # population length gate (sd of the conditional prediction) ...
    pop_tol_max=0.35,  # ... but never wider than this |log| (a prediction with few supported segments is vague)
    step_max={
        "coxa": 1.2,
        "femur": 1.2,
        "tibia": 0.6,
        "basitarsus": 0.6,
        "tarsus": 0.9,
        "scape": 0.8,
        "funiculus": 1.2,
    },
    jf_tol=0.05,  # traced-tube radius minimum / step within this share of the two segments' length of the joint
    max_snaps=2,  # how often a chain-fit joint may be moved onto the traced tubes' radius minimum / step
    flex_tol=0.5,  # |log| tolerance of the chord of a flexible segment (tarsus rest, funiculus): curl
    sock_mirror_bl=0.035,  # coxa base vs the mirror image of the supported partner (product frame), share of BL
    sock_reg_bl=0.03,  # coxa base vs the first registration's continuation marker (no supported partner coxa)
    ident_margin_bl=0.01,  # limb identity: another leg's registered socket may not be nearer than the own by more
    pop_identity=False,  # OFF: population check of the limb's name by the coxa's place along the mesosoma axis
    # . Tried as v1.2: it rejected right joints (dev 2, thin 17) and caught 1 wrong one,
    # because it leans on the product's own trunk joints; kept as an option only.
    sock_ant_bl=0.10,  # (not a gate: the antennal head-plane mirror distance is only logged)
    fold_deg=50.0,  # two verified neighbouring segments meeting at this angle or more pin the joint (a fold)
    snap_max=0.25,  # a chain-fit joint is moved onto the traced tubes' radius minimum when that lies within this
    # share of the two segments' length (then verified again)
    serial_joints=("ta", "an_2"),  # joints followed by a series of similar articulations: need a partner length
    relocate_types=(
        "fe",
        "ti",
        "ta",
        "pt",
        "an_2",
        "an_3",
    ),  # a product joint below 0.3 may be re-placed by the chain fit
    # only for these types (coxa base / trochanter / antennal socket: the product's own
    # position is kept or the joint is left alone)
    attach_lo=0.4,
    attach_hi=1.8,  # trochanter distance from the predicted socket in expected coxa lengths
    recover_declared_missing=True,
    use_pool=True,
    conf_base=0.30,
    conf_step=0.05,
    conf_max=0.55,
)


def _seg_type(ch, i):
    return seg_names(ch)[i]


class HeadPlane:
    """sagittal plane of the registered head from the template's mirror vertex pairs (model lateral axis = y)."""

    def __init__(self, tpl, V):
        from scipy.spatial import cKDTree

        ids = np.where(tpl.dom == tpl.idx["b_h"])[0]
        V0 = tpl.V0[ids]
        Vm = V0.copy()
        Vm[:, 1] *= -1
        d, j = cKDTree(V0).query(Vm)
        ok = (d < 1e-6) & (V0[:, 1] > 1e-4)
        a, b = ids[ok], ids[j[ok]]
        diff = V[a] - V[b]
        nrm = diff.sum(0)
        self.n = nrm / max(np.linalg.norm(nrm), 1e-12)
        self.c = 0.5 * (V[a] + V[b]).mean(0)
        self.n_pairs = int(ok.sum())

    def mirror(self, x):
        x = np.asarray(x, float)
        return x - 2.0 * ((x - self.c) @ self.n) * self.n


class Specimen:
    def __init__(self, name, joints_path, limbs_dir, env_path, reg_dir, pop, stage_root, cfg):
        from .tubecheck import TubeVerifier

        self.name, self.cfg, self.pop = name, cfg, pop
        self.prod = Product(joints_path)
        self.BL = self.prod.BL
        self.limbs = load_limbs(limbs_dir, name)
        self.pool = load_pool(stage_root, name) if (cfg["use_pool"] and "tail" in cfg["kinds"]) else []
        self.reg = load_registration(reg_dir, name)
        self.tv = TubeVerifier(name, env_path, joints_path)
        self.tpl = self.tv.tpl
        self.side, self.row = supported_lengths(self.prod)
        self.reg_markers, self.head_plane = None, None
        if self.reg is not None and len(self.reg["V"]) == self.tpl.nV:
            self.reg_markers = self.tpl.marker_regressor() @ self.reg["V"]
            try:
                self.head_plane = HeadPlane(self.tpl, self.reg["V"])
            except Exception:
                self.head_plane = None
        self._exp = {}
        self._ident = {}
        self.partner_z = {}

    # ---- what the model / population expects ------------------------------------------------------------------------
    def expected(self, ch, i):
        """expected log length of segment i of chain ch: dict(mu, sd, tol, source) or None."""
        key = (ch, i)
        if key in self._exp:
            return self._exp[key]
        out = None
        pk = (partner(ch), i)
        skey = seg_key(ch, i)
        partner_ok = pk in self.side
        pid = self.chain_identity(partner(ch))
        if partner_ok and pid is not None and not pid["ok"]:
            partner_ok = False  # a mis-named partner limb is no length reference
        if partner_ok and self.pop is not None:
            # the partner is a reference only when its own length is plausible for this specimen (a usable joint can
            # be misplaced: a wrong partner must not pull a joint to its length)
            pr = self.pop.predict(self.row, skey, without=self.name)
            if pr is not None:
                zp = abs(float(np.log(self.side[pk])) - pr[0]) / pr[1]
                self.partner_z[key] = zp
                partner_ok = zp <= self.cfg["z_partner_ref"]
        if partner_ok:
            tol = self.pop.partner_tol(skey) if self.pop is not None else 0.15
            out = dict(mu=float(np.log(self.side[pk])), sd=tol / 3.0, tol=tol, source="partner")
        elif self.pop is not None:
            pr = self.pop.predict(self.row, skey, without=self.name)
            if pr is not None:
                out = dict(
                    mu=pr[0], sd=pr[1], tol=min(self.cfg["z_pop"] * pr[1], self.cfg["pop_tol_max"]), source="population"
                )
        self._exp[key] = out
        return out

    def socket(self, ch):
        """predictions of the chain's root joint: dict source -> xyz."""
        n0 = CHAINS[ch][0]
        pn = partner_joint(n0)
        out = {}
        if self.prod.usable(pn):
            if ch[0] in "LR":
                out["mirror"] = self.prod.mirror(self.prod.pos(pn))
            elif self.head_plane is not None:
                out["mirror"] = self.head_plane.mirror(self.prod.pos(pn))
        if self.reg_markers is not None and ch[0] in "LR":
            out["registration"] = self.reg_markers[self.tpl.idx[n0]]
        return out

    def chain_identity(self, ch):
        """Population check of the limb's NAME: does the chain's coxa base (usable, else the product's position below
        0.3) sit where this leg pair's coxae sit along the mesosoma axis, and not where another pair's do? The joint
        method can name a mid leg as a hind leg (then its partner is mis-named the same way, and a registration made
        from those joints inherits the error). -> dict(t, own_z, other_z, ok) or None (not a leg / no coxa position)."""
        if ch in self._ident:
            return self._ident[ch]
        out = None
        if self.cfg.get("pop_identity") and ch[0] in "LR" and self.pop is not None:
            x = self.prod.pos(CHAINS[ch][0])
            if x is not None:
                out = self.pop.coxa_identity(ch[1], coxa_fraction(self.prod, x), without=self.name)
        self._ident[ch] = out
        return out

    def identity(self, ch, x):
        """Is a leg's base nearer to its OWN socket on the registered trunk than to another leg's socket of the same
        side? (The joint method can name a mid leg as a hind leg; its partner is then mis-named the same way, so the
        mirror check cannot see it.) -> dict(own, other, ok) in % BL, or None without a registration."""
        if self.reg_markers is None or ch[0] not in "LR":
            return None
        s = ch[0].lower()
        d = {
            k: float(
                np.linalg.norm(np.asarray(x, float) - self.reg_markers[self.tpl.idx[f"l_{k}_co_{s}"]]) / self.BL * 100
            )
            for k in "123"
        }
        own = d[ch[1]]
        other = min(v for k, v in d.items() if k != ch[1])
        return dict(own=own, other=other, ok=bool(own <= other + 100 * self.cfg["ident_margin_bl"]))

    def radius_fn(self, X):
        return self.tv.scan.kd.query(np.asarray(X, float))[0]

    # ---- candidates -------------------------------------------------------------------------------------------------
    def chain_candidates(self, ch):
        """-> (dict joint -> dict(product=..., fit=...), chain log)."""
        prod, cfg, BL = self.prod, self.cfg, self.BL
        js = CHAINS[ch]
        lost = [n for n in js if not prod.usable(n)]
        log = dict(lost=lost, polyline=None, expected={}, fit={})
        cands = {n: {} for n in lost}
        for i in range(len(js) - 1):
            e = self.expected(ch, i)
            if e is not None:
                log["expected"][_seg_type(ch, i)] = dict(
                    length_um=float(np.exp(e["mu"])), tol_log=e["tol"], source=e["source"]
                )
        for n in lost:
            x = prod.pos(n)
            if x is None:
                continue
            if n in prod.missing and not cfg["recover_declared_missing"]:
                continue
            kind = "fallback" if prod.is_fallback(n) else "lowconf"
            if kind in cfg["kinds"]:
                cands[n]["product"] = dict(
                    kind=kind, xyz=x, conf_in=prod.conf(n), prov_feature="scan_feature" in str(prod.prov.get(n) or "")
                )
        want_fit = ("chainfit" in cfg["kinds"] or "tail" in cfg["kinds"]) and ch in self.limbs
        if want_fit and not cfg["recover_declared_missing"]:
            lost = [n for n in lost if n not in prod.missing]
        if want_fit and lost:
            lb = self.limbs[ch]
            sock = self.socket(ch)
            sock_xyz = sock.get("mirror", sock.get("registration"))
            root_lost = not prod.usable(js[0])
            joints = [dict(name=n, fixed=(prod.pos(n) if prod.usable(n) else None)) for n in js]
            segs = []
            for i in range(len(js) - 1):
                e = self.expected(ch, i)
                segs.append(
                    dict(
                        mu=None if e is None else e["mu"],
                        sd=1.0 if e is None else e["sd"],
                        rigid=_seg_type(ch, i) in RIGID,
                    )
                )
            if root_lost and sock_xyz is None:
                log["fit_skipped"] = "root joint lost and no socket prediction"
            elif any(s["mu"] is None for s in segs):
                log["fit_skipped"] = "no expected length for a segment"
            else:
                r_named = self.radius_fn(lb["X"])
                polys = []
                if "chainfit" in cfg["kinds"]:
                    polys.append((lb["X"], "named", lb.get("status") == "path"))
                tail_lost = not prod.usable(js[-1])
                if tail_lost and "tail" in cfg["kinds"] and self.pool:
                    r_tip = float(np.median(r_named[-max(3, len(r_named) // 20) :]))
                    for X, tag, fl in chainfit.extend_tail(lb["X"], r_tip, self.pool, BL, self.radius_fn):
                        polys.append((X, f"named+{tag}", True))
                best, best_key = None, None
                for X, tag, tip_end in polys:
                    Q, a = chainfit.resample(X, chainfit.STEP_BL * BL)
                    if len(Q) < 6:
                        continue
                    Q = chainfit.smooth(Q)
                    r = self.radius_fn(Q)
                    fit = chainfit.fit_chain(
                        Q, a, r, BL, joints, segs, socket=sock_xyz if root_lost else None, tip_is_end=tip_end
                    )
                    placed = [f for f in fit if f is not None]
                    if not placed:
                        continue
                    key = (len(placed), -placed[0]["cost"])
                    if best_key is None or key > best_key:
                        best, best_key = (fit, tag), key
                if best is not None:
                    fit, tag = best
                    log["polyline"] = tag
                    kind = "tail" if "+" in tag else "chainfit"
                    for k, n in enumerate(js):
                        f = fit[k]
                        if f is None or n not in cands:
                            continue
                        cands[n]["fit"] = dict(
                            kind=kind,
                            xyz=f["xyz"],
                            at_feature=f["at_feature"],
                            is_tip=f["is_tip"],
                            bend_deg=f["bend_deg"],
                            rmin=f["rmin"],
                            z_prox=f["z_prox"],
                            z_dist=f["z_dist"],
                            polyline=tag,
                        )
                        log["fit"][n] = {
                            q: f[q]
                            for q in ("arc", "bend_deg", "rmin", "feat", "z_prox", "z_dist", "at_feature", "is_tip")
                        }
        for n, c in cands.items():
            if "product" in c and "fit" in c:
                d = float(np.linalg.norm(c["product"]["xyz"] - c["fit"]["xyz"]) / BL)
                c["agree"] = bool(d <= cfg["agree_bl"])
                c["product_fit_dist_pct"] = 100.0 * d
        return cands, log


class Judge:
    """the gates for one alternative of one chain, given one tube evaluation."""

    def __init__(self, sp, ch, pos, info, E):
        self.sp, self.ch, self.pos, self.info, self.E = sp, ch, pos, info, E
        self.js = CHAINS[ch]
        self.rep = {}

    def seg_checks(self, i):
        """checks of segment i (joints i, i + 1 both positioned): dict(len_ok, len_src, len_dev, tube_ok, step_ok, ...)."""
        sp, cfg = self.sp, self.sp.cfg
        a, b = self.js[i], self.js[i + 1]
        xa, xb = self.pos.get(a), self.pos.get(b)
        if xa is None or xb is None:
            return None
        L = float(np.linalg.norm(np.asarray(xb) - np.asarray(xa)))
        e = sp.expected(self.ch, i)
        out = dict(length_um=L, rigid=_seg_type(self.ch, i) in RIGID)
        if e is None:
            out.update(len_ok=False, len_src=None, len_dev=None)
        else:
            sdev = float(np.log(max(L, 1e-6)) - e["mu"])
            dev = abs(sdev)
            rigid = _seg_type(self.ch, i) in RIGID
            # a flexible segment (tarsus rest, funiculus) may be curled on either side: its chord is a loose measure
            ok_len = dev <= (e["tol"] if rigid else max(e["tol"], sp.cfg["flex_tol"]))
            out["rigid"] = rigid
            out.update(len_ok=bool(ok_len), len_src=e["source"], len_dev=dev, len_tol=e["tol"], len_signed=sdev)
        s = self.E["segments"].get(bone_of_segment(self.ch, i))
        st = _seg_type(self.ch, i)
        if s is None:
            out.update(tube_ok=False, verdict=None, step_ok=False, inner_step=None, joint_feature=None, tip_found=None)
        else:
            stp = s.get("inner_step")
            out.update(
                tube_ok=bool(s.get("verdict") == "ok"),
                verdict=s.get("verdict"),
                inner_step=stp,
                step_ok=bool(stp is None or stp <= cfg["step_max"][st]),
                joint_feature=s.get("joint_feature"),
                tip_found=s.get("tip_found"),
                radius=s.get("radius"),
                note=s.get("note"),
            )
        out["ok"] = bool(out["len_ok"] and out["tube_ok"] and out["step_ok"])
        return out

    def step_feature(self, k):
        """offset (share of the two segments' length, + = distal) of the largest radius step of the two traced tubes
        around joint k from the joint; None when there is no clear step (thick tibia -> thin basitarsus is one)."""
        sa = self.E["segments"].get(bone_of_segment(self.ch, k - 1))
        sb = self.E["segments"].get(bone_of_segment(self.ch, k)) if k < len(self.js) - 1 else None
        if not sa or not sb or sa.get("rmean") is None or sb.get("rmean") is None:
            return None
        rp, rd = np.asarray(sa["rmean"], float), np.asarray(sb["rmean"], float)
        Lp, Ld = float(sa.get("length") or 0), float(sb.get("length") or 0)
        if Lp <= 0 or Ld <= 0 or not (np.isfinite(rp).all() and np.isfinite(rd).all()):
            return None
        xx = np.concatenate([np.linspace(-Lp, 0, len(rp)), np.linspace(0, Ld, len(rd))[1:]])
        r = np.maximum(np.concatenate([rp, rd[1:]]), 1e-9)
        rs = np.convolve(np.pad(r, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        h = 2
        idx = np.arange(h, len(r) - h)
        win = idx[(xx[idx] > -0.35 * Lp) & (xx[idx] < 0.35 * Ld)]
        if not len(win):
            return None
        g = np.abs(np.log(rs[win + h] / rs[win - h]))
        i = int(np.argmax(g))
        if g[i] < 0.25:
            return None
        return float(xx[win[i]] / (Lp + Ld))

    def check(self, n, trusted):
        """-> (accept, reasons, evidence dict). trusted: set of joint names that count as verified neighbours."""
        sp, cfg, js = self.sp, self.sp.cfg, self.js
        k = js.index(n)
        x = np.asarray(self.pos[n], float)
        inf = self.info[n]
        reasons, ev = [], dict(kind=inf["kind"], agree=bool(inf.get("agree", False)))
        cid = sp.chain_identity(self.ch)
        ev["chain_identity"] = cid
        if cid is not None and not cid["ok"]:
            reasons.append(
                "limb identity: the chain's coxa base sits where another leg pair's coxae sit on the mesosoma (mis-named limb)"
            )
        prox = self.seg_checks(k - 1) if k > 0 and js[k - 1] in trusted else None
        dist_n = js[k + 1] if k + 1 < len(js) else None
        dist = self.seg_checks(k) if dist_n is not None and dist_n in trusted else None
        # the segment towards a distal candidate that is present in this evaluation (accepted or auxiliary): scan
        # evidence about THIS joint (a fold, a radius minimum between the two traced tubes) does not need the distal
        # candidate to be accepted as a joint itself
        dist_any = (
            dist
            if dist is not None
            else (self.seg_checks(k) if dist_n is not None and self.pos.get(dist_n) is not None else None)
        )
        ev["prox"], ev["dist"], ev["dist_aux"] = prox, dist, (None if dist is not None else dist_any)
        # ---- attached to the body ------------------------------------------------------------------------------------
        is_ant = self.ch[0] == "A"
        if k == 0:
            sock = sp.socket(self.ch)
            ev["socket"] = {s: float(np.linalg.norm(x - p) / sp.BL * 100) for s, p in sock.items()}
            if is_ant:
                # no usable socket prediction for antennae: the scape must agree with the PARTNER's length, and
                # the socket must not be far from the partner's mirror image in the registered head's plane
                ok = True  # the head-plane mirror distance is logged, not a gate
                ev["socket_source"] = "partner scape length"
            elif "mirror" in sock:
                ok = ev["socket"]["mirror"] <= 100 * cfg["sock_mirror_bl"]
                ev["socket_source"] = "mirror"
            elif "registration" in sock:
                ok = ev["socket"]["registration"] <= 100 * cfg["sock_reg_bl"]
                ev["socket_source"] = "registration"
            else:
                ok = False
                ev["socket_source"] = None
            ev["socket_ok"] = bool(ok)
            if not ok:
                reasons.append(
                    "socket: not where the partner's mirror image / the registered trunk puts it"
                    if ev["socket_source"]
                    else "socket: no prediction (no supported partner, no registration)"
                )
            idt = sp.identity(self.ch, x)
            ev["identity"] = idt
            if idt is not None and not idt["ok"]:
                ev["socket_ok"] = False
                reasons.append("limb identity: the base sits nearer to another leg's socket on the registered trunk")
        elif prox is None:
            attached = False
            if k == 1 and not is_ant:
                sock = sp.socket(self.ch)
                e0 = sp.expected(self.ch, 0)
                sx = sock.get("mirror", sock.get("registration"))
                if sx is not None and e0 is not None:
                    ratio = float(np.linalg.norm(x - sx) / np.exp(e0["mu"]))
                    ev["attach_ratio"] = ratio
                    attached = cfg["attach_lo"] <= ratio <= cfg["attach_hi"]
                    idt = sp.identity(self.ch, x)
                    ev["identity"] = idt
                    if idt is not None and not idt["ok"]:
                        attached = False
            ev["attached_without_coxa"] = bool(attached)
            if not attached:
                reasons.append("not attached: no verified joint between it and the body")
        if prox is not None and not prox["ok"]:
            reasons.append("proximal segment: " + _why(prox))
        dist_usable = dist_n is not None and sp.prod.usable(dist_n)
        # a usable distal neighbour must be reached by a verified RIGID segment; a flexible distal segment (tarsus
        # rest, funiculus) does not veto the joint before it (its tip may be the wrong one, its chord depends on curl)
        if dist is not None and dist_usable and not dist["ok"] and dist.get("rigid", True):
            reasons.append("distal segment: " + _why(dist))
        # ---- position along the limb ---------------------------------------------------------------------------------
        # A candidate must be pinned along its limb by evidence that did not produce it.
        #  product joints (found by the joint method: path features, proportions): the PARTNER's length on a verified
        #    neighbouring segment; or population lengths on both sides plus an independent scan feature; or one verified
        #    side plus agreement with the chain fit or a fold.
        #  chain-fit joints (found by lengths on a centreline): an independent scan feature at the joint (radius
        #    minimum of the two traced tubes, a fold between two verified segments, the traced tube's tip).
        #  joints followed by a series of similar articulations (ta, an_2) also need a partner length.
        sides = [s for s in (prox, dist) if s is not None and s["ok"]]
        # a partner length pins a joint only through a RIGID segment (the chord of a curled tarsus / funiculus is loose)
        n_partner = sum(s["len_src"] == "partner" and s.get("rigid", True) for s in sides)
        feats = []
        if prox is not None and prox.get("joint_feature") is not None and abs(prox["joint_feature"]) <= cfg["jf_tol"]:
            feats.append("tube radius minimum")
        if k == len(js) - 1 and prox is not None and prox["ok"] and prox.get("tip_found"):
            feats.append("tube tip")
        if prox is not None and dist_any is not None and prox["ok"] and dist_any["ok"]:
            u = x - np.asarray(self.pos[js[k - 1]], float)
            v = np.asarray(self.pos[dist_n], float) - x
            ang = float(
                np.degrees(np.arccos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12), -1, 1)))
            )
            ev["chord_angle_deg"] = ang
            if ang >= cfg["fold_deg"]:
                feats.append("fold")
        step_off = (
            self.step_feature(k)
            if (prox is not None and dist_any is not None and prox["ok"] and dist_any["ok"])
            else None
        )
        ev["tube_step_offset"] = step_off
        if step_off is not None and abs(step_off) <= cfg["jf_tol"] and "tube radius minimum" not in feats:
            feats.append("tube radius step")
        ev["features"], ev["n_partner"], ev["n_sides"] = feats, n_partner, len(sides)
        ev["tube_min_offset"] = None if prox is None else prox.get("joint_feature")
        if ev["tube_min_offset"] is None and step_off is not None:
            ev["tube_min_offset"] = step_off
        jtype = n.split("_")[2] if n.startswith("l_") else n[:4]
        from_product = inf.get("source") == "product"
        if k == 0:
            need_partner = is_ant
            along = (
                bool(ev.get("socket_ok"))
                and dist is not None
                and dist["ok"]
                and (not need_partner or dist["len_src"] == "partner")
            )
            if not from_product:
                along = False
                reasons.append("root joint placed by the chain fit: its socket prior is not independent evidence")
            elif dist is None:
                reasons.append("no verified distal neighbour (first segment not checked)")
            elif not dist["ok"] and not dist_usable:
                reasons.append("distal segment: " + _why(dist))
            elif need_partner and dist["len_src"] != "partner":
                reasons.append("antennal socket: the scape cannot be compared with a supported partner scape")
        else:
            if from_product:
                along = (
                    (n_partner >= 1)
                    or (len(sides) >= 2 and bool(feats))
                    or (len(sides) >= 1 and (ev["agree"] or "fold" in feats))
                )
            else:
                # the partner's length (tight) and an articulation feature of the centreline at that very place are two
                # independent sources that coincide; with population lengths (loose) a traced-tube feature is needed
                cl_feat = (
                    bool(inf.get("at_feature"))
                    and not inf.get("snapped")
                    and prox is not None
                    and prox["ok"]
                    and prox["len_src"] == "partner"
                )
                if cl_feat:
                    feats.append(
                        "tip" if inf.get("is_tip") else "centreline fold / radius minimum at the partner's length"
                    )
                along = len(sides) >= 1 and bool(feats)
            if jtype in cfg["serial_joints"] and n_partner == 0:
                along = False
                if not reasons:
                    reasons.append(
                        "serial articulation (tarsomeres / funicular segments): needs a supported partner length"
                    )
            if jtype in cfg["serial_joints"] and not from_product and along:
                # placed by the partner's length among similar articulations: the partner must agree on BOTH sides
                d2 = dist_any
                two = (
                    d2 is not None
                    and d2.get("len_src") == "partner"
                    and d2.get("tube_ok")
                    and d2.get("len_dev") is not None
                    and d2["len_dev"] <= cfg["flex_serial_tol"]
                )
                if not two:
                    along = False
                    if not reasons:
                        reasons.append(
                            "serial articulation placed by the chain fit: the distal segment does not agree with the partner's (tip wrong, truncated or curled)"
                        )
            if ev.get("attached_without_coxa") and n_partner == 0:
                # a leg that starts at the trochanter (coxa base not verified) has no verified base: its identity then
                # rests on a rigid segment that agrees with the PARTNER's, not on population lengths
                along = False
                if not reasons:
                    reasons.append(
                        "leg without a verified coxa base: needs a partner length (population lengths do not identify the limb)"
                    )
            if len(sides) == 0 and not reasons:
                reasons.append("no verified neighbouring segment")
        ev["along_ok"] = bool(along)
        if not along and not reasons:
            reasons.append(
                "position along the limb not pinned"
                + (
                    " (product joint: no partner length, no feature with two sides, no agreeing chain fit)"
                    if from_product
                    else " (chain fit: no independent scan feature at the joint)"
                )
            )
        return (not reasons), reasons, ev

    def run(self):
        """fixed point: proximal first, a joint is accepted when its checks hold with trusted = usable + accepted."""
        sp, js = self.sp, self.js
        cand = [n for n in js if n in self.info and not self.info[n].get("aux")]
        accepted = set(cand)
        rep = {}
        for _ in range(8):
            changed = False
            trusted = {n for n in js if sp.prod.usable(n)} | accepted
            for n in js:
                if n not in accepted:
                    continue
                ok, reasons, ev = self.check(n, trusted)
                rep[n] = dict(accept=ok, reasons=reasons, evidence=ev)
                if not ok:
                    accepted.discard(n)
                    trusted.discard(n)
                    changed = True
            if not changed:
                break
        self.rep = rep
        return accepted, rep


def _why(s):
    w = []
    if not s["tube_ok"]:
        w.append(f"tube {s.get('verdict')}")
    if not s["step_ok"]:
        w.append(f"radius step {s.get('inner_step'):.2f} inside")
    if not s["len_ok"]:
        w.append(
            "length "
            + (
                "without expectation"
                if s["len_src"] is None
                else f"{np.exp(s['len_dev']):.2f}x off the {s['len_src']} expectation"
            )
        )
    return ", ".join(w) or "?"


def confidence(ev, cfg):
    c = cfg["conf_base"]
    c += cfg["conf_step"] * min(ev.get("n_partner", 0), 2)
    c += cfg["conf_step"] * (1 if ev.get("features") else 0)
    c += cfg["conf_step"] * (1 if ev.get("agree") else 0)
    c += cfg["conf_step"] * (1 if ev.get("socket_source") == "mirror" and ev.get("socket_ok") else 0)
    return round(float(min(c, cfg["conf_max"])), 3)


def recover_specimen(name, joints_path, limbs_dir, env_path, reg_dir, pop, stage_root=None, cfg=None):
    """-> (augmented product dict, log dict). Never raises: a failure is a status in the log, and the product is
    returned unchanged (plus the `recovery` block saying so)."""
    import copy
    import json

    cfg = {**DEFAULTS, **(cfg or {})}
    t0 = time.time()
    log = dict(
        name=name,
        version=VERSION,
        status="ok",
        error=None,
        kinds=list(cfg["kinds"]),
        chains={},
        joints={},
        seconds=None,
    )
    try:
        d_in = json.load(open(joints_path))
    except Exception as e:
        log.update(status="failed", error=f"joint product not readable: {e}", seconds=time.time() - t0)
        return None, log
    out = copy.deepcopy(d_in)
    try:
        sp = Specimen(name, joints_path, limbs_dir, env_path, reg_dir, pop, stage_root, cfg)
        prod = sp.prod
        log["inputs"] = dict(
            joints=joints_path,
            limbs=bool(sp.limbs),
            pool_polylines=len(sp.pool),
            registration=None if sp.reg is None else sp.reg["path"],
            population_n=None if pop is None else pop.n(without=name),
        )
        cands = {}
        for ch in CHAINS:
            if all(prod.usable(n) for n in CHAINS[ch]):
                continue
            c, clog = sp.chain_candidates(ch)
            cands[ch] = c
            log["chains"][ch] = clog

        # Rounds. Start: the product's own position where it has one, else the chain fit. After each tube evaluation
        # and judgement, a rejected joint gets its next alternative: product -> chain fit (types in relocate_types,
        # when the two disagree); chain fit -> the same moved onto the traced tubes' radius minimum (once); else dropped.
        def jtype_of(n):
            return n.split("_")[2] if n.startswith("l_") else n[:4]

        def pick(ch, n, src):
            c = cands[ch][n]
            inf = dict(c[src], agree=c.get("agree", False), source=src, snapped=False)
            if src == "product" and "fit" in c and c.get("agree"):
                inf["at_feature"], inf["is_tip"] = c["fit"]["at_feature"], c["fit"]["is_tip"]
            return np.asarray(c[src]["xyz"], float), inf

        cur = {ch: {} for ch in cands}  # joint -> (xyz, info)
        for ch in cands:
            for n, c in cands[ch].items():
                src = "product" if "product" in c else ("fit" if "fit" in c else None)
                if src is not None:
                    cur[ch][n] = pick(ch, n, src)
        history = {ch: {n: [] for n in cands[ch]} for ch in cands}
        evals, accepted, reports = 0, {ch: set() for ch in cands}, {ch: {} for ch in cands}
        for rnd in range(7):
            extra = {n: v[0] for ch in cur for n, v in cur[ch].items()}
            if not extra:
                break
            E = sp.tv.evaluate(extra=extra)
            evals += 1
            nxt_changed = False
            for ch in cands:
                if not cur[ch]:
                    accepted[ch] = set()
                    continue
                pos = {n: prod.pos(n) for n in CHAINS[ch] if prod.usable(n)}
                pos.update({n: v[0] for n, v in cur[ch].items()})
                acc, rep = Judge(sp, ch, pos, {n: v[1] for n, v in cur[ch].items()}, E).run()
                accepted[ch], reports[ch] = acc, {**reports[ch], **rep}
                prox_changed = False
                for n in [m for m in CHAINS[ch] if m in cur[ch]]:
                    inf = cur[ch][n][1]
                    if inf.get("aux"):
                        continue
                    history[ch][n].append(
                        dict(
                            round=rnd,
                            source=inf["source"],
                            snapped=inf.get("snapped", False),
                            accept=n in acc,
                            reasons=(rep.get(n) or {}).get("reasons"),
                        )
                    )
                    if n in acc:
                        continue
                    c = cands[ch][n]
                    r = rep.get(n) or {}
                    if (
                        inf["source"] == "product"
                        and "fit" in c
                        and not c.get("agree")
                        and jtype_of(n) in cfg["relocate_types"]
                        and ("chainfit" in cfg["kinds"] or "tail" in cfg["kinds"])
                    ):
                        cur[ch][n] = pick(ch, n, "fit")
                        nxt_changed = prox_changed = True
                        continue
                    if prox_changed and rnd < 5:  # a joint proximal of it gets another try: so does this one
                        if (
                            inf["source"] == "product"
                            and "fit" in c
                            and not c.get("agree")
                            and jtype_of(n) in cfg["relocate_types"]
                        ):
                            cur[ch][n] = pick(ch, n, "fit")
                        continue
                    off = (r.get("evidence") or {}).get("tube_min_offset")
                    if (
                        inf["source"] == "fit"
                        and inf.get("n_snaps", 0) < cfg["max_snaps"]
                        and off is not None
                        and cfg["jf_tol"] < abs(off) <= cfg["snap_max"]
                    ):
                        js_ = CHAINS[ch]
                        k_ = js_.index(n)
                        pa = pos.get(js_[k_ - 1]) if k_ > 0 else None
                        pb = pos.get(js_[k_ + 1]) if k_ + 1 < len(js_) else None
                        if pa is not None and pb is not None:
                            x0 = cur[ch][n][0]
                            tot = np.linalg.norm(x0 - pa) + np.linalg.norm(pb - x0)
                            tgt = pb if off > 0 else pa
                            u = (np.asarray(tgt) - x0) / max(np.linalg.norm(np.asarray(tgt) - x0), 1e-9)
                            inf2 = dict(inf, snapped=True, snap_offset=float(off), n_snaps=inf.get("n_snaps", 0) + 1)
                            cur[ch][n] = (x0 + u * abs(off) * tot, inf2)
                            nxt_changed = prox_changed = True
                            continue
                    # no further alternative: the joint is not delivered. It stays in the scene as an AUXILIARY point
                    # when its proximal segment is a verified tube (so that the joint before it keeps its fold /
                    # radius-minimum evidence); otherwise it is removed.
                    pe = (r.get("evidence") or {}).get("prox")
                    if pe is not None and pe.get("tube_ok") and pe.get("step_ok") and pe.get("len_ok"):
                        cur[ch][n] = (cur[ch][n][0], dict(inf, aux=True))
                    else:
                        del cur[ch][n]
                    nxt_changed = True
            if not nxt_changed:
                break
        final_acc = {ch: {n for n in accepted[ch] if n in cur[ch] and not cur[ch][n][1].get("aux")} for ch in cands}
        final_rep = reports
        alts = {
            ch: {"X": ({n: v[0] for n, v in cur[ch].items()}, {n: v[1] for n, v in cur[ch].items()})} for ch in cands
        }
        chosen = {ch: "X" for ch in cands}
        # ---- write the augmented product and the log --------------------------------------------------------------------
        rec_block = dict(
            version=VERSION,
            input=joints_path,
            kinds=list(cfg["kinds"]),
            joints={},
            n_recovered=0,
            note="joints with provenance 'recovered (...)' were added by limbrec; every other entry is the input product's",
        )
        for ch in cands:
            tagc = chosen[ch]
            first_rep, other_rep = {}, {}
            for n in cands[ch]:
                c = cands[ch][n]
                jl = dict(
                    chain=ch,
                    rounds=history[ch].get(n),
                    input_confidence=prod.conf(n),
                    input_present=prod.pos(n) is not None,
                    declared_missing=n in prod.missing,
                    candidates={k: dict(kind=v["kind"], xyz=v["xyz"]) for k, v in c.items() if k in ("product", "fit")},
                    product_fit_dist_pct=c.get("product_fit_dist_pct"),
                    agree=c.get("agree"),
                    recovered=False,
                    reasons=None,
                    evidence=None,
                )
                acc = final_acc.get(ch, set())
                if n in acc:
                    rep = final_rep[ch][n]
                    inf = alts[ch][tagc][1][n]
                    x = np.asarray(alts[ch][tagc][0][n], float)
                    conf = confidence(rep["evidence"], cfg)
                    ver = ["tube"]
                    evd = rep["evidence"]
                    srcs = {s["len_src"] for s in (evd.get("prox"), evd.get("dist")) if s}
                    ver += [f"{s} length" for s in sorted(x_ for x_ in srcs if x_)]
                    if evd.get("features"):
                        ver.append("scan feature")
                    if evd.get("socket_ok"):
                        ver.append(f"socket ({evd.get('socket_source')})")
                    if evd.get("agree"):
                        ver.append("agrees with a second placement")
                    if inf.get("snapped"):
                        ver.append("moved onto the tube radius minimum")
                    prov = f"recovered ({inf['kind']}; verified: {', '.join(ver)})"
                    out["joints_scan"][n] = [float(v) for v in x]
                    if isinstance(out.get("joints_canonical"), dict):
                        out["joints_canonical"][n] = [float(v) for v in prod.to_can(x)]
                    out["confidence"][n] = conf
                    out.setdefault("provenance", {})[n] = prov
                    comp = out.setdefault("completeness", {})
                    if n in (comp.get("missing_joints") or []):
                        comp["missing_joints"] = [m for m in comp["missing_joints"] if m != n]
                    if isinstance(out.get("null_reasons"), dict):
                        out["null_reasons"].pop(n, None)
                    moved = None if prod.pos(n) is None else float(np.linalg.norm(x - prod.pos(n)) / sp.BL * 100)
                    rec_block["joints"][n] = dict(
                        kind=inf["kind"],
                        confidence=conf,
                        input_confidence=prod.conf(n),
                        input_position=None if prod.pos(n) is None else prod.pos(n),
                        moved_pct_bl=moved,
                        verified_by=ver,
                        evidence=evd,
                    )
                    jl.update(
                        recovered=True,
                        kind=inf["kind"],
                        confidence=conf,
                        xyz=x,
                        moved_pct_bl=moved,
                        evidence=evd,
                        provenance=prov,
                    )
                else:
                    r = (final_rep.get(ch) or {}).get(n) or first_rep.get(n) or other_rep.get(n)
                    if r is not None:
                        jl.update(reasons=r["reasons"], evidence=r["evidence"])
                    elif not c:
                        jl.update(reasons=["no candidate (no product position, no chain fit)"])
                    else:
                        jl.update(reasons=["not evaluated"])
                log["joints"][n] = jl
        rec_block["n_recovered"] = len(rec_block["joints"])
        comp = out.setdefault("completeness", {})
        if rec_block["n_recovered"]:
            comp.setdefault("flags", []).append(f"recovered_joints:{rec_block['n_recovered']}")
        out["recovery"] = rec_block
        log["n_lost"] = sum(len(c) for c in cands.values())
        log["n_recovered"] = rec_block["n_recovered"]
        log["tube_evaluations"] = evals
    except Exception as e:
        out = copy.deepcopy(d_in)
        out["recovery"] = dict(
            version=VERSION, input=joints_path, joints={}, n_recovered=0, status="failed", error=str(e)
        )
        log.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    log["seconds"] = time.time() - t0
    return out, log
