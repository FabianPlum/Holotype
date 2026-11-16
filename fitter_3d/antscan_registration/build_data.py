"""Rebuild the two data files shipped in fitter_3d/antscan_registration/data/ (both truth-free, no names or paths):

    python -m fitter_3d.antscan_registration.build_data priors --names NAMES.txt --joints JOINT_DIR --out FILE
    python -m fitter_3d.antscan_registration.build_data population --run OUT_DIR --names NAMES.txt --out FILE

priors      population_prod82.json, the segment-length prior of the recovery step (parts/limbrec/priors.py): one row of
            log chord lengths between directly supported joints per specimen of NAMES that has a joint product
            <JOINT_DIR>/<name>/<name>_joints.json; specimens are keyed by the md5 of their name. The shipped file was
            built from single-state joint products of the 82 production scans without annotation.
population  rest_population.json, the population reference of the unposed (UC1) mesh (fill.py): from the status files
            of a finished pipeline run (<OUT_DIR>/status/<name>_status.npz: V, vertex_status, specimen_class), over
            the specimens of NAMES that are not do_not_use. Each is unposed; its vertices that are not measurable,
            that stick out of their ring, or that the length gate against the mean of THESE specimens removes are
            left out (the bundle is made without an earlier population file). Then
              mean      (nV, 3) mean rest position over the specimens in which the vertex is left in; mean_n (nV,)
              edge_med  (nE,) median length of every template edge whose two ends are left in (the template's edge
                        length where fewer than 5 specimens have it); edge_n (nE,)
            `mean` is the fill source and the reference of the length gate, `edge_med` the reference of the
            stretched-edge condition of the model grade. The shipped file was built from the run of the 82 production
            scans without annotation with four-state ensemble joint products (78 of them not do_not_use).
"""

import argparse
import hashlib
import json
import os

import numpy as np

from fitter_3d.antscan_registration import VERSION, read_names, setup


def build_priors(names, joints_dir, out):
    from fitter_3d.antscan_registration.parts.limbrec.priors import KEYS, Population

    pop = Population.build(names, joints_dir)
    pop.save(out)
    print(f"population prior: {len(pop.names)} specimens -> {out}")
    for k in KEYS:
        col = pop.A[:, KEYS.index(k)]
        print(
            f"  {k:18s} n {int(np.isfinite(col).sum()):3d}  partner tol {pop.partner_tol(k):.3f}  pairs {len(pop.asym.get(k, []))}"
        )
    return pop


def build_population(run_dir, names, out):
    from fitter_3d.antscan_registration import fill as FI
    from fitter_3d.antscan_registration.parts import uc1_unpose as U

    t = U.template()
    S = FI.static(t)
    items = []
    for n in names:
        p = f"{run_dir}/status/{n}_status.npz"
        if not os.path.isfile(p):
            continue
        z = np.load(p, allow_pickle=True)
        if str(z["specimen_class"]) == "do_not_use":
            continue
        items.append(dict(name=n, V_scan=np.asarray(z["V"], float), vertex_status=np.asarray(z["vertex_status"], int)))
    if not items:
        raise SystemExit(f"no usable status file of the names under {run_dir}/status")
    b, _ = FI.make_bundle(items, None, use_population=False)
    V = b["verts"].astype(np.float64)
    ok = b["valid"]
    n_in = ok.sum(0)
    mean = (V * ok[:, :, None]).sum(0) / np.maximum(n_in, 1)[:, None]
    mean[n_in == 0] = t.V0[n_in == 0]
    E = S["E"]
    L = np.linalg.norm(V[:, E[:, 0]] - V[:, E[:, 1]], axis=2)
    eo = ok[:, E[:, 0]] & ok[:, E[:, 1]]
    en = eo.sum(0)
    with np.errstate(all="ignore"):
        emed = np.nanmedian(np.where(eo, L, np.nan), axis=0)
    emed = np.where(en >= 5, emed, S["e0"])

    def r6(x):
        return [float(f"{v:.6g}") for v in np.asarray(x, float).ravel()]

    k = len(items)
    res = dict(
        what="truth-free population reference of the UC1 rest mesh (regpipe.fill): mean rest position per vertex and "
        "median length per template edge over measurable vertices; model units",
        n_specimens=k,
        names_sha1=hashlib.sha1("\n".join(sorted(str(x) for x in b["labels"])).encode()).hexdigest(),
        source=f"unposed registrations of {k} production scans without truth, pipeline {VERSION}",
        min_count=5,
        mean=np.asarray(r6(mean)).reshape(-1, 3).tolist(),
        mean_n=[int(v) for v in n_in],
        edge_med=r6(emed),
        edge_n=[int(v) for v in en],
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print(
        f"population reference: {k} specimens; vertices with >= 5 specimens {(n_in >= 5).sum()} of {t.nV}; "
        f"edges with >= 5 {(en >= 5).sum()} of {len(E)} -> {out}"
    )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("priors")
    p.add_argument("--names", required=True)
    p.add_argument("--joints", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("population")
    q.add_argument("--run", required=True)
    q.add_argument("--names", required=True)
    q.add_argument("--out", required=True)
    for s in (p, q):
        s.add_argument("--model", default=None, help="model file (default: the pipeline's)")
    a = ap.parse_args(argv)
    setup(a.model)
    if a.cmd == "priors":
        build_priors(read_names(a.names), a.joints, a.out)
    else:
        build_population(a.run, read_names(a.names), a.out)


if __name__ == "__main__":
    main()
