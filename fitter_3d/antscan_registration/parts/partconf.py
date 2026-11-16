"""Settings shared by the parts of the registration pipeline (model file, cache folder). They are set by the
pipeline's `setup()` / `set_model()`; nothing here reads the environment and importing this module changes nothing."""

MODEL = None  # path of the model .pkl (OmniAnt)
CACHE = None  # folder for the cache of the template's outer surface (None: computed in memory, not written)
USABLE_CONF = 0.3  # a joint with a confidence below this is not used by any part (the intake decides who gets it)


def model():
    if MODEL is None:
        raise RuntimeError("model path not set: call the pipeline's set_model(path) first")
    return MODEL
