"""Joint files in the joint-product contract -> SKEL-REG's joint dict (only `load` for a joint FILE given by its
path and its helper)."""

import json
import os

import numpy as np

from .pose import CONF_MIN, load_joints

KIND = {"r3": "pivot", "s6": "medial", "s6fix": "medial", "oracle": "pivot", "production": "pivot"}


def _gaster_from_tip(jt, tpl, tip):
    i3, i4, i5 = tpl.idx["b_a_3"], tpl.idx["b_a_4"], tpl.idx["b_a_5"]
    if tip is None or not np.isfinite(jt["P"][i3]).all():
        return
    f = getattr(tpl, "gaster_fraction", None)
    if f is None:
        tpl.marker_regressor()
        f = tpl.gaster_fraction
    jt["P"][i5] = tip
    jt["P"][i4] = jt["P"][i3] + f * (np.asarray(tip) - jt["P"][i3])
    jt["conf"][i4] = jt["conf"][i5] = max(jt["conf"][i3], CONF_MIN)
    jt["missing"] = [m for m in jt["missing"] if m not in ("b_a_4", "b_a_5")]


def load(name, source, tpl, path=None):
    """Joint dict (pose.load_joints layout) + extras: head_front, gaster_tip, waist_segments, source, path, kind.
    `path` is the joint file; `source` only selects the joint kind (KIND)."""
    if path is None or not os.path.exists(path):
        return None
    d = json.load(open(path))
    extra = dict(
        source=source, path=path, kind=KIND.get(source, "pivot"), head_front=None, gaster_tip=None, waist_segments=None
    )
    jt = load_joints(d, tpl)
    js = d.get("joints_scan", {}) or {}
    if extra["kind"] == "medial":
        tip = js.get("gaster_tip")
        cf = (d.get("confidence", {}) or {}).get("gaster_tip", 1.0) or 0.0
        if tip is not None and cf >= CONF_MIN:
            _gaster_from_tip(jt, tpl, np.asarray(tip, float))
            extra["gaster_tip"] = np.asarray(tip, float)
        if js.get("head_front") is not None:
            extra["head_front"] = np.asarray(js["head_front"], float)
        ib, ih = tpl.idx["b_t"], tpl.idx["b_h"]
        if not np.isfinite(jt["P"][ih]).all() and np.isfinite(jt["P"][ib]).all():
            jt["P"][ih], jt["conf"][ih] = jt["P"][ib], jt["conf"][ib]
        if not np.isfinite(jt["P"][ib]).all() and np.isfinite(jt["P"][ih]).all():
            jt["P"][ib], jt["conf"][ib] = jt["P"][ih], jt["conf"][ih]
    ws = d.get("waist_segments")
    extra["waist_segments"] = int(ws) if ws in (1, 2) else None
    jt.update(extra)
    return jt
