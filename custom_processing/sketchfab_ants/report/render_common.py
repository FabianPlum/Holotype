"""Colours shared by the report's render-job builders (build_report_assets.py, build_extra_renders.py)."""
import numpy as np

# sequential orange ramp for distances (light = small, dark orange = large), sRGB
HEAT = np.array([[236, 233, 228], [246, 196, 168], [239, 138, 92], [235, 104, 52], [194, 79, 30], [107, 36, 8]], float)
BLUE = [42, 120, 214]  # fitted OmniAnt
CHITIN = [176, 128, 80]  # neutral-pose specimens (drawer, morphospace)
GREY = [196, 196, 200]  # targets


def heat(x):
    """x in [0,1] -> sRGB uint8 along HEAT."""
    x = np.clip(np.asarray(x, float), 0, 1) * (len(HEAT) - 1)
    i = np.minimum(np.floor(x).astype(int), len(HEAT) - 2)
    t = (x - i)[..., None]
    return (HEAT[i] * (1 - t) + HEAT[i + 1] * t).round().astype(np.uint8)
