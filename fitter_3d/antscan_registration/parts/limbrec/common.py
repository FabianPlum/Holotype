"""Constants, chain definitions and small helpers shared by the limbrec modules. No truth, no names, no taxonomy."""

import json
import os

import numpy as np

from fitter_3d.antscan_registration.parts import partconf

CONF_MIN = partconf.USABLE_CONF  # the consumers' rule: below this a joint is unresolved
LEGJ = ["co", "tr", "fe", "ti", "ta", "pt"]
LEG_SEG = ["coxa", "femur", "tibia", "basitarsus", "tarsus"]  # segment i joins LEGJ[i] and LEGJ[i + 1]
ANT_SEG = ["scape", "funiculus"]
RIGID = {"coxa", "femur", "tibia", "basitarsus", "scape"}  # straight rigid segments; tarsus rest / funiculus bend


def chains():
    """chain id -> list of joint names, proximal first. L1..R3 legs, AL / AR antennae (the limb product's names)."""
    out = {}
    for s in "lr":
        for k in "123":
            out[("L" if s == "l" else "R") + k] = [f"l_{k}_{j}_{s}" for j in LEGJ]
        out["A" + s.upper()] = [f"an_{i}_{s}" for i in "123"]
    return out


CHAINS = chains()
CHAIN_OF = {n: c for c, js in CHAINS.items() for n in js}


def partner(chain):
    return {"L": "R", "R": "L"}[chain[0]] + chain[1] if chain[0] in "LR" else {"AL": "AR", "AR": "AL"}[chain]


def partner_joint(n):
    return n[:-1] + ("r" if n.endswith("l") else "l")


def seg_names(chain):
    return LEG_SEG if chain[0] in "LR" else ANT_SEG


def seg_key(chain, i):
    """population key of segment i of a chain: leg{k}_{segment} or scape / funiculus (sides pooled)."""
    return f"leg{chain[1]}_{LEG_SEG[i]}" if chain[0] in "LR" else ANT_SEG[i]


def bone_of_segment(chain, i):
    """template bone that IS segment i (the bone starts at the segment's proximal joint)."""
    return CHAINS[chain][i]


def conv(x):
    if isinstance(x, dict):
        return {str(k): conv(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [conv(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return conv(x.tolist())
    return x


def jdump(o, path, indent=1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(conv(o), open(tmp, "w"), indent=indent)
    os.replace(tmp, path)
