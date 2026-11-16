"""Parts of the registration pipeline that were developed and validated as stand-alone modules: recovery of limb
joints (limbrec), status / morphometrics / export (rs), scan access, tube tracing and verification (skelreg), status
evidence (cover04), unposing to the rest pose (uc1_unpose). `partconf` holds the model file and the cache folder for
all of them (set by fitter_3d.antscan_registration.setup).

Names that occur in the comments of these modules: SKEL-REG = the earlier skeleton-first registration these scan and
tube modules come from (skelreg); COVER = the status evidence module (cover04); SURFACE = the earlier surface
registration that fitter_3d/geomreg grew from; R2-SHAPE = the shape-model work that defined the morphometric measures
and the unposing convention UC1; GEOM, RECOVER, STATUS = the registration core, the recovery of limb joints and the
status part of this pipeline; "round 6" and "round 7 (K4)" = two generations of the joint method (single-state
products and four-state ensemble products); "round 2" and "round 3" = the earlier and the current generation of
the registration modules."""
