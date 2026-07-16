"""
Tail-error analysis for the stress surrogate (camera-ready addition C4).

Means and medians hide the deployment-critical failures, which live in the
peak per-sample errors. This script quantifies the tail of the error
distribution and the calibrated-interval behaviour at the most highly
stressed nodes, for both the in-distribution test set (N=500) and the OOD
set (N=250).

Panels (three-panel horizontal layout):
  1. Empirical CDF of per-sample max |error| (MPa), ID vs OOD.
  2. Mean CQR coverage at the top-k most-stressed nodes (k = 1/5/10%),
     ID vs OOD, against the nominal 90% target.
  3. Per-sample max |error| vs max ensemble std, with Pearson r.

Reuses the frozen split-conformal calibration (alpha, q_hat) from
runs/cqr/cqr_primary.pt and the CQR helpers in src/uq/cqr.py. Values are in
physical MPa (the surrogate output is never normalised).

Run from the project root:
    python scripts/fig_tail_analysis.py               # Phase 1: single panel
    python scripts/fig_tail_analysis.py --panels 3    # full three-panel figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.uq.cqr import calibrated_interval  # noqa: E402

PRIMARY_ALPHA = 0.10          # nominal 90% two-sided target
NOMINAL_COVERAGE = 1.0 - PRIMARY_ALPHA
TAIL_FRACTIONS = (0.01, 0.05, 0.10)   # top-1%, top-5%, top-10% stressed nodes
ID_COLOR = "C0"
OOD_COLOR = "C3"


def load_predictions_npz(path: Path) -> dict:
    """Dict-style load of a prediction cache (matches phase2/phase3 idiom)."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def per_sample_max_abs_error(mean: np.ndarray, gt: np.ndarray,
                             offsets: np.ndarray) -> np.ndarray:
    """max_j |mean[i,j] - gt[i,j]| over the nodes j of each sample i."""
    err = np.abs(mean - gt)
    return np.array([err[offsets[i]:offsets[i + 1]].max()
                     for i in range(len(offsets) - 1)])


def per_sample_max_std(std: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """max_j std[i,j] over the nodes j of each sample i."""
    return np.array([std[offsets[i]:offsets[i + 1]].max()
                     for i in range(len(offsets) - 1)])


def tail_coverage_by_sample(mean: np.ndarray, std: np.ndarray, gt: np.ndarray,
                            offsets: np.ndarray, alpha: float, q_hat: float,
                            frac: float) -> np.ndarray:
    """Per-sample CQR coverage restricted to the top-`frac` most-stressed
    nodes (ranked by ground-truth stress magnitude within the sample)."""
    lo, hi = calibrated_interval(mean, std, alpha, q_hat)
    covered = (gt >= lo) & (gt <= hi)
    out = np.empty(len(offsets) - 1)
    for i in range(len(offsets) - 1):
        s, e = offsets[i], offsets[i + 1]
        n = e - s
        k = max(1, int(np.ceil(frac * n)))
        y = gt[s:e]
        # indices of the k largest ground-truth stresses in this sample
        top = np.argpartition(y, n - k)[n - k:]
        out[i] = covered[s:e][top].mean()
    return out


def summarize(name: str, arr: np.ndarray) -> dict:
    stats = {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "worst": float(arr.max()),
    }
    print(f"  {name:>4s} (N={stats['n']:3d}): "
          f"median={stats['median']:.3f}  p90={stats['p90']:.3f}  "
          f"p99={stats['p99']:.3f}  worst={stats['worst']:.3f}  MPa")
    return stats


def ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def draw_cdf_panel(ax, id_max: np.ndarray, ood_max: np.ndarray) -> bool:
    span = np.concatenate([id_max, ood_max])
    use_log = (span.max() / span.min()) > 10.0
    for arr, color, label in ((id_max, ID_COLOR, "ID test (N=500)"),
                              (ood_max, OOD_COLOR, "OOD (N=250)")):
        x, y = ecdf(arr)
        ax.step(x, y, where="post", color=color, label=label, lw=1.8)
    if use_log:
        ax.set_xscale("log")
    ax.set_xlabel("Per-sample max |error| [MPa]")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(a) Peak-error distribution", fontsize=9)
    return use_log


def draw_tail_coverage_panel(ax, id_cov: dict, ood_cov: dict) -> None:
    labels = [f"top-{int(f * 100)}%" for f in TAIL_FRACTIONS]
    x = np.arange(len(TAIL_FRACTIONS))
    w = 0.38
    id_vals = [id_cov[f].mean() for f in TAIL_FRACTIONS]
    ood_vals = [ood_cov[f].mean() for f in TAIL_FRACTIONS]
    ax.bar(x - w / 2, id_vals, w, color=ID_COLOR, label="ID test")
    ax.bar(x + w / 2, ood_vals, w, color=OOD_COLOR, label="OOD")
    ax.axhline(NOMINAL_COVERAGE, color="k", ls="--", lw=1.0,
               label="nominal 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean CQR coverage")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(b) Coverage at most-stressed nodes", fontsize=9)


def draw_scatter_panel(ax, id_err, id_std, ood_err, ood_std) -> tuple:
    r_id = pearson(id_err, id_std)
    r_ood = pearson(ood_err, ood_std)
    ax.scatter(id_std, id_err, s=12, color=ID_COLOR, alpha=0.55,
               label=f"ID test (r={r_id:.2f})")
    ax.scatter(ood_std, ood_err, s=12, color=OOD_COLOR, alpha=0.55,
               label=f"OOD (r={r_ood:.2f})")
    ax.set_xlabel("Per-sample max ensemble std [MPa]")
    ax.set_ylabel("Per-sample max |error| [MPa]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(c) Peak error vs peak uncertainty", fontsize=9)
    return r_id, r_ood


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/cqr")
    ap.add_argument("--panels", type=int, default=1, choices=(1, 3),
                    help="1 = Phase-1 CDF only; 3 = full three-panel figure")
    ap.add_argument("--fig-name", default="fig_tail_analysis.pdf")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cqr = torch.load(out_dir / "cqr_primary.pt", weights_only=False)
    alpha, q_hat = float(cqr["alpha"]), float(cqr["q_hat"])
    assert abs(alpha - PRIMARY_ALPHA) < 1e-9, f"alpha mismatch: {alpha}"
    print(f"Frozen calibration: alpha={alpha}  q_hat={q_hat:.6f} MPa")

    test = load_predictions_npz(out_dir / "preds_test.npz")
    ood = load_predictions_npz(out_dir / "preds_ood.npz")

    id_max = per_sample_max_abs_error(test["mean"], test["gt"],
                                      test["offsets"])
    ood_max = per_sample_max_abs_error(ood["mean"], ood["gt"], ood["offsets"])

    print("\nPhase 1 -- per-sample max |error| (MPa):")
    id_stats = summarize("ID", id_max)
    ood_stats = summarize("OOD", ood_max)

    if args.panels == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        use_log = draw_cdf_panel(ax, id_max, ood_max)
    else:
        id_std = per_sample_max_std(test["std"], test["offsets"])
        ood_std = per_sample_max_std(ood["std"], ood["offsets"])

        id_cov = {f: tail_coverage_by_sample(test["mean"], test["std"],
                                             test["gt"], test["offsets"],
                                             alpha, q_hat, f)
                  for f in TAIL_FRACTIONS}
        ood_cov = {f: tail_coverage_by_sample(ood["mean"], ood["std"],
                                              ood["gt"], ood["offsets"],
                                              alpha, q_hat, f)
                   for f in TAIL_FRACTIONS}

        print("\nPhase 2 -- mean CQR coverage at most-stressed nodes "
              f"(nominal {NOMINAL_COVERAGE:.0%}):")
        for f in TAIL_FRACTIONS:
            print(f"  top-{int(f*100):2d}%:  ID={id_cov[f].mean():.3f}   "
                  f"OOD={ood_cov[f].mean():.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        use_log = draw_cdf_panel(axes[0], id_max, ood_max)
        draw_tail_coverage_panel(axes[1], id_cov, ood_cov)
        r_id, r_ood = draw_scatter_panel(axes[2], id_max, id_std,
                                         ood_max, ood_std)
        print("\nPhase 2 -- max|error| vs max std Pearson r:  "
              f"ID={r_id:.3f}   OOD={r_ood:.3f}")

    print(f"\nCDF x-axis scale: {'log' if use_log else 'linear'} "
          f"(ID range [{id_max.min():.3f}, {id_max.max():.3f}], "
          f"OOD range [{ood_max.min():.3f}, {ood_max.max():.3f}] MPa)")

    figs = Path("paper/figures")
    figs.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = figs / args.fig_name
    fig.savefig(out)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
