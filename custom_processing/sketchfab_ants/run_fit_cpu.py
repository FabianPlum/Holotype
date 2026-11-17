"""
Runs fitter_3d.optimise on a CPU-only machine (e.g. Apple Silicon) with a dense chamfer distance.

pytorch3d's CPU KNN (used by pytorch3d.loss.chamfer_distance) is brute force and single-threaded
(~4 s per mesh per iteration for 3000 target samples x 10229 template vertices), which makes a
multi-stage fit of >100 meshes take weeks. This wrapper replaces fitter_3d.trainer.chamfer_distance
with an equivalent implementation: nearest neighbours from chunked torch.cdist (BLAS, no grad), then
the loss on the gathered pairs, so value AND gradient are the same as pytorch3d's (its KNN backward
also only flows through the selected pairs). fitter_3d itself is not modified.

The equivalence is checked against pytorch3d on random data before fitting starts (values and
gradients); the run aborts if they differ.

Usage (pytorch3d env; --repo defaults to the checkout this file is in):
    python custom_processing/sketchfab_ants/run_fit_cpu.py [--repo REPO] -- <fitter_3d.optimise arguments>
e.g.
    python custom_processing/sketchfab_ants/run_fit_cpu.py -- --mesh_dir <oriented objs> --yaml_src <cfg.yaml>
"""

import argparse
import os
import sys
import time

import torch


def dense_chamfer_distance(x, y, x_lengths=None, y_lengths=None, chunk=8, **kwargs):
    """
    pytorch3d.loss.chamfer_distance for padded point clouds of equal size per batch element
    (no lengths, no normals, norm=2, point_reduction='mean', batch_reduction='mean', both
    directions), which is how fitter_3d.trainer.Stage.forward calls it. Returns (loss, None).
    """
    if x_lengths is not None or y_lengths is not None or kwargs:
        raise NotImplementedError(f"dense_chamfer_distance: unsupported arguments {sorted(kwargs)}")
    if hasattr(y, "verts_padded"):
        y = y.verts_padded()
    B = x.shape[0]
    idx_xy, idx_yx = [], []
    with torch.no_grad():
        for s in range(0, B, chunk):
            d = torch.cdist(x[s : s + chunk], y[s : s + chunk])  # (b, N, M)
            idx_xy.append(d.argmin(2))
            idx_yx.append(d.argmin(1))
            del d
    idx_xy = torch.cat(idx_xy)
    idx_yx = torch.cat(idx_yx)
    nn_y = torch.gather(y, 1, idx_xy[..., None].expand(-1, -1, 3))
    nn_x = torch.gather(x, 1, idx_yx[..., None].expand(-1, -1, 3))
    cham_x = ((x - nn_y) ** 2).sum(-1).mean(1)
    cham_y = ((y - nn_x) ** 2).sum(-1).mean(1)
    return (cham_x + cham_y).mean(), None


def check_equivalence_PROBE(out_path=None):
    """Compares dense_chamfer_distance with pytorch3d.loss.chamfer_distance (value and gradients)."""
    from pytorch3d.loss import chamfer_distance

    torch.manual_seed(0)
    lines = []
    ok = True
    for B, N, M in ((1, 300, 500), (3, 1000, 2000), (2, 3000, 10229)):
        x = torch.randn(B, N, 3)
        y = (torch.randn(B, M, 3) * 0.8 + 0.1)
        xa, ya = x.clone().requires_grad_(), y.clone().requires_grad_()
        xb, yb = x.clone().requires_grad_(), y.clone().requires_grad_()
        t = time.time()
        la, _ = chamfer_distance(xa, ya)
        la.backward()
        ta = time.time() - t
        t = time.time()
        lb, _ = dense_chamfer_distance(xb, yb)
        lb.backward()
        tb = time.time() - t
        rel = abs(la.item() - lb.item()) / la.item()
        gx = (xa.grad - xb.grad).abs().max().item() / xa.grad.abs().max().item()
        gy = (ya.grad - yb.grad).abs().max().item() / ya.grad.abs().max().item()
        line = (f"B={B} N={N} M={M}: pytorch3d {la.item():.7f} ({ta:.2f}s)  dense {lb.item():.7f} ({tb:.2f}s)  "
                f"rel.diff {rel:.1e}  max rel. grad diff x {gx:.1e} y {gy:.1e}")
        lines.append(line)
        print(line)
        ok &= rel < 1e-5 and gx < 1e-3 and gy < 1e-3
    lines.append("EQUIVALENT" if ok else "MISMATCH")
    if out_path:
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    return ok


def main():
    argv = sys.argv[1:]
    own, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1 :]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   help="Holotype repository root (default: this checkout)")
    p.add_argument("--probe_out", default=None, help="Write the equivalence check to this file")
    p.add_argument("--plots", action="store_true",
                   help="Keep fitter_3d's per-stage matplotlib mesh plots (config.PLOT_RESULTS). Off by default: "
                        "they draw every face of each ~200k-face target in matplotlib 3D, taking minutes and "
                        "~1.2 GB per mesh in a pool of cpu_count() workers")
    a = p.parse_args(own)

    if not check_equivalence_PROBE(a.probe_out):
        raise SystemExit("dense chamfer does not match pytorch3d.loss.chamfer_distance - not fitting")

    os.chdir(a.repo)  # config.SMAL_FILE and the default paths are relative to the repo root
    sys.path.insert(0, a.repo)
    import fitter_3d.trainer as trainer
    import fitter_3d.optimise as optimise

    import config

    trainer.chamfer_distance = dense_chamfer_distance
    config.PLOT_RESULTS = a.plots
    print(f"fitter_3d.trainer.chamfer_distance -> dense_chamfer_distance; torch threads {torch.get_num_threads()}; "
          f"PLOT_RESULTS={config.PLOT_RESULTS}")
    args = optimise.parser.parse_args(rest)
    optimise.main(args)


if __name__ == "__main__":
    main()
