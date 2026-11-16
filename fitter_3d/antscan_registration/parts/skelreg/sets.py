"""Cache folder of the SKEL-REG modules."""

from fitter_3d.antscan_registration.parts import partconf


def cache_file(name):
    """path of a cache file, or None when the pipeline has no cache folder (then nothing is cached on disk)"""
    return None if not partconf.CACHE else f"{partconf.CACHE}/{name}"
