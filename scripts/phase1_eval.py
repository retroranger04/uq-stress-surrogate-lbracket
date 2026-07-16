"""
Phase-1 final evaluation on the held-out test set.

Loads all 5 ensemble members + applies shared stats, computes:
- Per-member point metrics (per-node MAPE, peak MAPE, error percentiles)
- Ensemble mean prediction metrics
- Ensemble std as uncertainty, correlation with absolute error
- Calibration scatter plot (ensemble std vs abs error)
- Prediction/ground-truth/uncertainty figure for best/median/worst samples

Outputs
-------
runs/<base>/test_metrics.json
runs/<base>/member_metrics.json
paper/figures/fig_phase1_training_curves.pdf
paper/figures/fig_phase1_calibration.pdf
paper/figures/fig_phase1_example_predictions.pdf

The OOD bundle data/ood.pt is NEVER loaded here (Phase-3 only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.runtime import (
    RunCfg, apply_stats_inplace, build_model, eval_metrics, load_bundle,
    load_stats,
)


def _load_member(member_dir: Path, device: torch.device):
    blob = torch.load(str(member_dir / "best.pt"), weights_only=False,
                      map_location=device)
    cfg = RunCfg(**blob["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def _predict_all(model, items, device, batch_size=16) -> list[np.ndarray]:
    loader = DataLoader(items, batch_size=batch_size, shuffle=False)
    preds_per_graph: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr).squeeze(-1).cpu().numpy()
        ptr = batch.ptr.cpu().numpy()
        for i in range(len(ptr) - 1):
            preds_per_graph.append(pred[ptr[i]:ptr[i+1]])
    return preds_per_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base", required=True,
                    help="ensemble base dir containing seed*/ subfolders + stats.pt")
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[101, 202, 303, 404, 505])
    ap.add_argument("--figs-dir", default="paper/figures")
    ap.add_argument("--eps-mpa", type=float, default=1.0,
                    help="stabilizer for MAPE denominator [MPa]")
    args = ap.parse_args()

    base = Path(args.base)
    figs = Path(args.figs_dir); figs.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = load_stats(base / "stats.pt")
    test = load_bundle(Path(args.data_dir) / "test.pt")
    print(f"[eval] test size: {len(test)}")
    apply_stats_inplace(test, stats)
    test_loader = DataLoader(test, batch_size=16, shuffle=False)

    # Ground truth per graph
    gt_list = [d.y.squeeze(-1).numpy() for d in test]

    member_metrics = []
    member_preds = []   # list over members, each a list over graphs
    for seed in args.seeds:
        mdir = base / f"seed{seed}"
        model, cfg = _load_member(mdir, device)
        m = eval_metrics(model, test_loader, device, eps=args.eps_mpa)
        m["seed"] = seed
        member_metrics.append(m)
        print(f"[eval] seed={seed}  {m}")
        member_preds.append(_predict_all(model, test, device))
        del model
        torch.cuda.empty_cache()

    # Ensemble mean + std per node per graph
    M = len(args.seeds)
    ensemble_mean = []
    ensemble_std = []
    for g in range(len(test)):
        stack = np.stack([member_preds[m][g] for m in range(M)], axis=0)   # (M, N)
        ensemble_mean.append(stack.mean(0))
        ensemble_std.append(stack.std(0, ddof=0))

    # Ensemble point metrics
    all_abs = np.concatenate([np.abs(ensemble_mean[g] - gt_list[g])
                              for g in range(len(test))])
    all_rel = np.concatenate([
        np.abs(ensemble_mean[g] - gt_list[g]) /
        np.maximum(np.abs(gt_list[g]), args.eps_mpa)
        for g in range(len(test))])
    peak_aps = []
    for g in range(len(test)):
        yp = float(np.max(ensemble_mean[g]))
        yt = float(np.max(gt_list[g]))
        peak_aps.append(abs(yp - yt) / max(abs(yt), args.eps_mpa))
    ensemble_metrics = {
        "per_node_mape": float(all_rel.mean() * 100),
        "peak_mape": float(np.mean(peak_aps) * 100),
        "abs_err_p50": float(np.percentile(all_abs, 50)),
        "abs_err_p90": float(np.percentile(all_abs, 90)),
        "abs_err_p99": float(np.percentile(all_abs, 99)),
        "abs_err_max": float(all_abs.max()),
    }

    # Calibration: correlation of ensemble std with abs error
    all_std = np.concatenate(ensemble_std)
    pearson = float(np.corrcoef(all_std, all_abs)[0, 1])
    spearman_r = float(
        np.corrcoef(np.argsort(np.argsort(all_std)),
                    np.argsort(np.argsort(all_abs)))[0, 1])

    # Per-sample mean std vs mean abs err (coarser)
    per_sample_std = np.array([ensemble_std[g].mean() for g in range(len(test))])
    per_sample_err = np.array([np.abs(ensemble_mean[g] - gt_list[g]).mean()
                               for g in range(len(test))])
    pearson_sample = float(np.corrcoef(per_sample_std, per_sample_err)[0, 1])

    out_json = {
        "members": member_metrics,
        "ensemble": ensemble_metrics,
        "calibration": {
            "pearson_node_level_std_vs_abserr": pearson,
            "spearman_node_level_std_vs_abserr": spearman_r,
            "pearson_sample_level_std_vs_abserr": pearson_sample,
        },
    }
    (base / "test_metrics.json").write_text(json.dumps(out_json, indent=2))
    print(json.dumps(out_json, indent=2))

    # Emit paper tables
    tables_dir = Path("paper/tables"); tables_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in member_metrics:
        rows.append(
            f"Seed {m['seed']}  & {m['per_node_mape']:.2f} & {m['peak_mape']:.2f}"
            f" & {m['abs_err_p50']:.3f} & {m['abs_err_p90']:.3f}"
            f" & {m['abs_err_p99']:.2f} & {m['abs_err_max']:.2f} \\\\")
    rows.append("\\midrule")
    em = ensemble_metrics
    rows.append(
        f"Ensemble mean  & \\textbf{{{em['per_node_mape']:.2f}}} & "
        f"\\textbf{{{em['peak_mape']:.2f}}}"
        f" & {em['abs_err_p50']:.3f} & {em['abs_err_p90']:.3f}"
        f" & {em['abs_err_p99']:.2f} & {em['abs_err_max']:.2f} \\\\")
    (tables_dir / "phase1_accuracy_rows.tex").write_text("\n".join(rows) + "\n")

    calib_rows = [
        f"Pearson correlation, node-level (std vs $|$error$|$) & {pearson:.3f} \\\\",
        f"Spearman rank correlation, node-level & {spearman_r:.3f} \\\\",
        f"Pearson correlation, sample-level & {pearson_sample:.3f} \\\\",
    ]
    (tables_dir / "phase1_calibration_rows.tex").write_text("\n".join(calib_rows) + "\n")

    # ---- Figure 1: training curves for each member ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed in args.seeds:
        h = json.loads((base / f"seed{seed}" / "history.json").read_text())
        hist = h["history"]
        ep = [r["epoch"] for r in hist]
        val = [r["val"] for r in hist]
        ax.plot(ep, val, label=f"seed {seed}", linewidth=1.1)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss (Huber / MPa)")
    ax.set_title("Phase-1 Deep Ensemble — validation loss per member")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig_phase1_training_curves.pdf")
    plt.close(fig)

    # ---- Figure 2: node-level calibration scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Subsample for plotting (too many nodes)
    rng = np.random.default_rng(0)
    n = all_std.size
    idx = rng.choice(n, size=min(40_000, n), replace=False)
    axes[0].scatter(all_std[idx], all_abs[idx], s=2, alpha=0.2)
    axes[0].set_xlabel("ensemble std [MPa]")
    axes[0].set_ylabel("|pred − truth| [MPa]")
    axes[0].set_title(f"Node-level  Pearson={pearson:.3f}  "
                       f"Spearman={spearman_r:.3f}")
    lim = max(all_std[idx].max(), all_abs[idx].max())
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    axes[0].grid(alpha=0.3)
    axes[1].scatter(per_sample_std, per_sample_err, s=10)
    axes[1].set_xlabel("per-sample mean ensemble std [MPa]")
    axes[1].set_ylabel("per-sample mean |pred − truth| [MPa]")
    axes[1].set_title(f"Sample-level  Pearson={pearson_sample:.3f}")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Phase-1 ensemble uncertainty vs. prediction error")
    fig.tight_layout()
    fig.savefig(figs / "fig_phase1_calibration.pdf")
    plt.close(fig)

    # ---- Figure 3: example predictions (best / median / worst by per-sample err) ----
    order = np.argsort(per_sample_err)
    pick_idx = [order[0], order[len(order)//2], order[-1]]
    # Shared color scales across the best/median/worst rows for the error and
    # uncertainty panels: without this each panel auto-normalizes to its own
    # max, so a low-error sample renders as bright as a high-error one. A
    # single shared vmax makes the magnitudes visually comparable. Truth and
    # prediction stay on a per-sample stress scale (same physical range).
    err_vmax = max(float(np.abs(ensemble_mean[gi] - gt_list[gi]).max())
                   for gi in pick_idx)
    std_vmax = max(float(ensemble_std[gi].max()) for gi in pick_idx)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for row, gi in enumerate(pick_idx):
        # Coordinates come from the raw un-normalized pos tensor — we kept those
        # untouched (only x and edge_attr are normalized).
        pos = test[gi].pos.cpu().numpy()
        yt = gt_list[gi]
        yp = ensemble_mean[gi]
        sd = ensemble_std[gi]
        err = np.abs(yp - yt)
        vmax = max(yt.max(), yp.max())
        for ax, vals, title, vmx in zip(
            axes[row],
            [yt, yp, err, sd],
            ["ground truth σ_vm [MPa]", "ensemble mean [MPa]",
             "|error| [MPa]", "ensemble std [MPa]"],
            [vmax, vmax, err_vmax, std_vmax],
        ):
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=vals, s=4,
                            cmap=("viridis" if "std" not in title and "error" not in title else "magma"),
                            vmin=0, vmax=vmx)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=9)
        tag = ["best", "median", "worst"][row]
        axes[row, 0].set_ylabel(
            f"{tag} (idx {gi}, mean err {per_sample_err[gi]:.2f} MPa)",
            fontsize=9)
    fig.suptitle("Phase-1 test-set examples — ground truth / ensemble mean / error / uncertainty")
    fig.tight_layout()
    fig.savefig(figs / "fig_phase1_example_predictions.pdf")
    plt.close(fig)
    print(f"[eval] wrote figures -> {figs}")


if __name__ == "__main__":
    main()
