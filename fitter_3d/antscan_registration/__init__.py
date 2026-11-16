"""The AntScan registration pipeline: joint product + limb product + envelope -> registered OmniAnt mesh with a status
at every level, morphometrics and export files. One command:

    python -m fitter_3d.antscan_registration.run --names N --joints J --limbs L --env E --out OUT

Documentation: docs/antscan_registration_pipeline.md. Steps: intake (which joints are usable) -> first registration
(fitter_3d.geomreg) -> recovery of lost limb joints and mandible bases (parts/limbrec, mandible.py) -> second
registration -> status, specimen class with a model grade, morphometrics, fitter_3d stage .npz with native parameters,
unposed bundle for the Blender add-on (parts/rs, fill.py, grade.py).

Importing this package changes nothing outside it: no environment variable is set and nothing is added to sys.path.
`setup()` tells the parts the model file; the command and every function that needs a part call it. The command alone
sets defaults for the thread counts (`cli_environment`). Nothing in this package reads truth, specimen or genus names
(a name is a file key) or taxonomy; no network, no AI calls.
"""

import os

VERSION = "regpipe-1.1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
PRIORS = os.path.join(DATA, "population_prod82.json")
DEFAULT_MODEL = os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")
USABLE_CONF = 0.3  # the confidence a joint needs to be usable; the parts hold the same value (checked in setup)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
REQUIRED = (("embreex", "ray casting of the scan (status, recovery, overlays)"),)
OPTIONAL = (("threadpoolctl", "without it two CPU runs may differ in the last bits"),)


def cli_environment():
    """Defaults of the COMMAND, set before numpy and torch are imported: two threads per process (the batch runs
    specimens side by side) and the cuBLAS setting that makes GPU runs repeatable. A value already in the environment
    is kept. Never called on import."""
    for v in THREAD_VARS:
        os.environ.setdefault(v, "2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def cache_dir(out):
    """folder for the cache of the template's outer surface: inside the run's work folder"""
    return os.path.join(out, "work", "cache")


def setup(model=None, cache=None):
    """Tells the parts the model file and the cache folder (None: the template's outer surface is computed in memory
    by every process that needs it). Idempotent; nothing in the environment or on sys.path is changed. Returns the
    model path."""
    from .parts import partconf

    if partconf.USABLE_CONF != USABLE_CONF:
        raise RuntimeError("the parts and the pipeline disagree on the confidence a usable joint needs")
    if cache is not None:
        partconf.CACHE = cache
    if model is not None or partconf.MODEL is None:
        set_model(model)
    return partconf.MODEL


def set_model(path=None):
    """model .pkl for every part"""
    from fitter_3d.geomreg import model as GM

    from .parts import partconf

    path = os.path.abspath(path or partconf.MODEL or DEFAULT_MODEL)
    if partconf.MODEL != path:
        partconf.MODEL = path
        GM.set_default_model(path)
    return path


def missing_dependencies():
    """(missing required, missing optional) packages as lists of (name, what it is for)"""
    import importlib.util

    def miss(pairs):
        return [(m, why) for m, why in pairs if importlib.util.find_spec(m) is None]

    return miss(REQUIRED), miss(OPTIONAL)


def read_names(path):
    """one specimen name per line (fitter_3d.antscan_common.read_names: blank and '#' lines skipped, duplicates taken
    once, a name that is a path refused)"""
    from fitter_3d.antscan_common import read_names as _read

    return _read(path)
