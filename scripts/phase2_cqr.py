"""
Phase-2 CQR pipeline on the frozen Phase-1 Deep Ensemble.

Steps:
  1. Load val + test bundles, apply the shared input normalization stats.
  2. Run the 5 frozen ensemble members on both splits (one forward pass per
     member per split). Cache per-graph (mean, std, gt, params) arrays to
     `runs/cqr/preds_{val,test}.npz`.
  3. For each nominal alpha in {0.20, 0.15, 0.10, 0.05}, compute Gaussian
     base quantiles from (mu, sigma), calibrate Q_hat on the validation
     pooled-node conformity scores, evaluate marginal coverage + width on
     test. Save the calibration artifacts + metrics to
     `runs/cqr/calibration.json`.
  4. Conditional coverage breakdown by (R, p, W) quartiles at the primary
     alpha = 0.10. Saved to the same JSON.
  5. Emit paper tables (`paper/tables/phase2_*_rows.tex`).
  6. Emit paper figures (`paper/figures/fig_phase2_{coverage,interval_width,
     intervals_example,comparison}.pdf`).

The test bundle is touched only in step 3-6 for the final frozen
evaluation; no hyperparameter adaptation here.
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
    RunCfg, apply_stats_inplace, build_model, load_bundle, load_stats,
)
from src.uq.cqr import (
    base_quantiles, calibrated_interval, conformal_quantile,
    conformity_scores, coverage, width, z_for_alpha,
)


def _load_member(member_dir: Path, device: torch.device):
    blob = torch.load(str(member_dir / "best.pt"), weights_only=False,
                      map_location=device)
    cfg = RunCfg(**blob["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def _predict_split(member, items, device, batch_size=16):
    loader = DataLoader(items, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = batch.to(device)
        pred = member(batch.x, batch.edge_index, batch.edge_attr
                      ).squeeze(-1).cpu().numpy()
        ptr = batch.ptr.cpu().numpy()
        for i in range(len(ptr) - 1):
            out.append(pred[ptr[i]:ptr[i+1]])
    return out


def ensemble_predict(base: Path, seeds, items, device):
    """Return (mean_list, std_list) each a list of (N_g,) float32 arrays."""
    M = len(seeds)
    per_member = []
    for s in seeds:
        m = _load_member(base / f"seed{s}", device)
        per_member.append(_predict_split(m, items, device))
        del m
        torch.cuda.empty_cache()
    G = len(items)
    means, stds = [], []
    for g in range(G):
        stack = np.stack([per_member[k][g] for k in range(M)], axis=0)
        means.append(stack.mean(0).astype(np.float32))
        stds.append(stack.std(0, ddof=0).astype(np.float32))
    return means, stds


def save_predictions_npz(path: Path, means, stds, gts, params, pos_list):
    # Flattened: store concatenated arrays with per-graph offsets.
    sizes = np.array([len(g) for g in gts], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    np.savez_compressed(
        path,
        mean=np.concatenate(means).astype(np.float32),
        std=np.concatenate(stds).astype(np.float32),
        gt=np.concatenate(gts).astype(np.float32),
        pos=np.concatenate(pos_list).astype(np.float32),
        params=np.asarray(params, dtype=np.float32),
        offsets=offsets,
    )


def load_predictions_npz(path: Path):
    z = np.load(path)
    return {
        "mean": z["mean"], "std": z["std"], "gt": z["gt"], "pos": z["pos"],
        "params": z["params"], "offsets": z["offsets"],
    }


def split_arrays(flat, offsets):
    return [flat[offsets[i]:offsets[i+1]] for i in range(len(offsets) - 1)]


def compute_all(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = load_stats(base / "stats.pt")

    splits = {"val": "val.pt", "test": "test.pt"}
    cached = {}
    for split, fname in splits.items():
        cache_path = out_dir / f"preds_{split}.npz"
        if cache_path.exists() and not args.recompute:
            print(f"[cache] {split}: using cached {cache_path}")
            cached[split] = load_predictions_npz(cache_path)
            continue
        print(f"[cache] {split}: running all 5 members through {fname}")
        items = load_bundle(Path(args.data_dir) / fname)
        gts = [d.y.squeeze(-1).numpy().astype(np.float32) for d in items]
        params = np.stack([d.params.squeeze(0).numpy() for d in items]
                          ).astype(np.float32)
        pos_list = [d.pos.numpy().astype(np.float32) for d in items]
        apply_stats_inplace(items, stats)
        means, stds = ensemble_predict(base, args.seeds, items, device)
        save_predictions_npz(cache_path, means, stds, gts, params, pos_list)
        cached[split] = load_predictions_npz(cache_path)
        print(f"[cache] {split}: wrote {cache_path}")

    # --- Calibration sweep over alphas ---
    val = cached["val"]; test = cached["test"]
    val_mu, val_sd, val_y = val["mean"], val["std"], val["gt"]
    test_mu, test_sd, test_y = test["mean"], test["std"], test["gt"]
    test_offsets = test["offsets"]

    alphas = [0.20, 0.15, 0.10, 0.05]
    cal = {}
    for a in alphas:
        q_lo_v, q_hi_v = base_quantiles(val_mu, val_sd, a)
        scores = conformity_scores(val_y, q_lo_v, q_hi_v)
        q_hat = conformal_quantile(scores, a)
        lo_t, hi_t = calibrated_interval(test_mu, test_sd, a, q_hat)
        # Uncalibrated (base Gaussian) baseline for comparison.
        base_lo_t, base_hi_t = base_quantiles(test_mu, test_sd, a)
        cal[f"alpha_{a:.2f}"] = {
            "alpha": a,
            "nominal_coverage": 1 - a,
            "z": z_for_alpha(a),
            "q_hat_MPa": q_hat,
            "empirical_coverage_test": coverage(test_y, lo_t, hi_t),
            "empirical_coverage_test_base": coverage(test_y, base_lo_t,
                                                      base_hi_t),
            "mean_width_test": float(width(lo_t, hi_t).mean()),
            "mean_width_test_base": float(width(base_lo_t, base_hi_t).mean()),
            "median_width_test": float(np.median(width(lo_t, hi_t))),
        }

    # --- Conditional coverage at primary alpha=0.10, quartiles of R/p/W ---
    primary = 0.10
    z_p = z_for_alpha(primary)
    q_lo_v, q_hi_v = base_quantiles(val_mu, val_sd, primary)
    q_hat = conformal_quantile(conformity_scores(val_y, q_lo_v, q_hi_v),
                                primary)
    lo_t, hi_t = calibrated_interval(test_mu, test_sd, primary, q_hat)

    # Per-sample coverage (fraction of nodes inside interval for each sample)
    per_sample_cov = np.array([
        ((test_y[test_offsets[i]:test_offsets[i+1]]
          >= lo_t[test_offsets[i]:test_offsets[i+1]]) &
         (test_y[test_offsets[i]:test_offsets[i+1]]
          <= hi_t[test_offsets[i]:test_offsets[i+1]])).mean()
        for i in range(len(test_offsets) - 1)
    ])
    per_sample_width = np.array([
        (hi_t[test_offsets[i]:test_offsets[i+1]]
         - lo_t[test_offsets[i]:test_offsets[i+1]]).mean()
        for i in range(len(test_offsets) - 1)
    ])
    per_sample_err = np.array([
        np.abs(test_mu[test_offsets[i]:test_offsets[i+1]]
               - test_y[test_offsets[i]:test_offsets[i+1]]).mean()
        for i in range(len(test_offsets) - 1)
    ])

    test_params = test["params"]  # (G, 3)
    param_names = ["R", "p", "W"]
    cond = {"alpha": primary}
    for j, name in enumerate(param_names):
        vals = test_params[:, j]
        edges = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0])
        rows = []
        for q in range(4):
            if q == 3:
                sel = (vals >= edges[q]) & (vals <= edges[q+1])
            else:
                sel = (vals >= edges[q]) & (vals < edges[q+1])
            if sel.sum() == 0:
                continue
            # Node-level coverage within these samples
            sel_nodes = np.concatenate([
                np.arange(test_offsets[i], test_offsets[i+1])
                for i in np.where(sel)[0]
            ])
            cov_q = coverage(test_y[sel_nodes], lo_t[sel_nodes],
                             hi_t[sel_nodes])
            rows.append({
                "quartile": q + 1,
                "range": [float(edges[q]), float(edges[q+1])],
                "n_samples": int(sel.sum()),
                "coverage": cov_q,
                "mean_width": float((hi_t[sel_nodes]
                                     - lo_t[sel_nodes]).mean()),
            })
        cond[name] = rows

    # Width vs error (informativeness)
    pearson_w_err = float(np.corrcoef(per_sample_width, per_sample_err)[0, 1])
    spearman_w_err = float(np.corrcoef(
        np.argsort(np.argsort(per_sample_width)),
        np.argsort(np.argsort(per_sample_err)))[0, 1])

    calib_json = {
        "sweep": cal,
        "conditional_coverage_primary": cond,
        "informativeness": {
            "pearson_width_vs_error_sample": pearson_w_err,
            "spearman_width_vs_error_sample": spearman_w_err,
        },
        "seeds": list(args.seeds),
        "n_cal": int(val_y.size),
        "n_test_nodes": int(test_y.size),
        "n_test_samples": int(len(test_offsets) - 1),
    }
    (out_dir / "calibration.json").write_text(json.dumps(calib_json, indent=2))
    print(json.dumps(calib_json, indent=2))

    # Save primary-alpha calibration artifact for Phase-3 consumption.
    torch.save({
        "alpha": primary,
        "q_hat": q_hat,
        "z": z_p,
        "base_quantile_rule": "gaussian_mu_pm_z_sigma",
        "ensemble_seeds": list(args.seeds),
        "stats_path": str(base / "stats.pt"),
        "paper_ref": "Romano 2019 Theorem 1 + Gopakumar 2024 cell-wise CP",
    }, str(out_dir / "cqr_primary.pt"))
    print(f"[cqr] saved primary artifact -> {out_dir / 'cqr_primary.pt'}")

    # --- Paper tables ---
    tables = Path("paper/tables"); tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in alphas:
        e = cal[f"alpha_{a:.2f}"]
        rows.append(
            f"{int(round(100*(1-a)))} & "
            f"{100*e['empirical_coverage_test_base']:.2f} & "
            f"{e['mean_width_test_base']:.2f} & "
            f"{100*e['empirical_coverage_test']:.2f} & "
            f"{e['mean_width_test']:.2f} & "
            f"{e['q_hat_MPa']:.3f} \\\\")
    (tables / "phase2_coverage_rows.tex").write_text("\n".join(rows) + "\n")

    cond_rows = []
    for name in param_names:
        for r in cond[name]:
            cond_rows.append(
                f"{name} Q{r['quartile']} & "
                f"[{r['range'][0]:.2f}, {r['range'][1]:.2f}] & "
                f"{r['n_samples']} & "
                f"{100*r['coverage']:.2f} & "
                f"{r['mean_width']:.2f} \\\\")
    (tables / "phase2_conditional_rows.tex").write_text(
        "\n".join(cond_rows) + "\n")

    # --- Figures ---
    figs = Path("paper/figures"); figs.mkdir(parents=True, exist_ok=True)

    # Figure: coverage vs nominal across alphas (CQR + uncalibrated baseline)
    nominal = [1 - a for a in alphas]
    cqr_cov = [cal[f"alpha_{a:.2f}"]["empirical_coverage_test"] for a in alphas]
    base_cov = [cal[f"alpha_{a:.2f}"]["empirical_coverage_test_base"]
                 for a in alphas]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0.75, 1.0], [0.75, 1.0], "k--", linewidth=0.8, alpha=0.5,
            label="perfect calibration")
    ax.plot(nominal, cqr_cov, "o-", color="C0", label="CQR-calibrated")
    ax.plot(nominal, base_cov, "s--", color="C3", alpha=0.75,
            label="Uncalibrated (Gaussian $\\mu \\pm z\\sigma$)")
    ax.set_xlabel("nominal coverage $1 - \\alpha$")
    ax.set_ylabel("empirical coverage on test")
    ax.set_xlim(0.77, 0.97); ax.set_ylim(0.77, 1.0)
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Phase-2 CQR coverage vs. nominal")
    fig.tight_layout(); fig.savefig(figs / "fig_phase2_coverage.pdf")
    plt.close(fig)

    # Figure: interval-width distribution at primary alpha, split by R quartile
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(per_sample_width, bins=40, color="C0", alpha=0.85)
    axes[0].axvline(per_sample_width.mean(), color="k", linestyle="--",
                    linewidth=0.8, label=f"mean {per_sample_width.mean():.2f} MPa")
    axes[0].set_xlabel("per-sample mean interval width [MPa]")
    axes[0].set_ylabel("test samples")
    axes[0].set_title(f"CQR interval width at nominal 90%")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    R_vals = test_params[:, 0]
    R_edges = np.quantile(R_vals, [0, 0.25, 0.5, 0.75, 1.0])
    for q in range(4):
        if q == 3:
            sel = (R_vals >= R_edges[q]) & (R_vals <= R_edges[q+1])
        else:
            sel = (R_vals >= R_edges[q]) & (R_vals < R_edges[q+1])
        axes[1].hist(per_sample_width[sel], bins=20, alpha=0.45,
                      label=f"R Q{q+1} [{R_edges[q]:.1f}, {R_edges[q+1]:.1f}]")
    axes[1].set_xlabel("per-sample mean interval width [MPa]")
    axes[1].set_ylabel("test samples")
    axes[1].set_title("Width by fillet-radius quartile")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("Phase-2 CQR interval widths (test set, $\\alpha = 0.10$)")
    fig.tight_layout(); fig.savefig(figs / "fig_phase2_interval_width.pdf")
    plt.close(fig)

    # Figure: example fields with lo/mean/hi/gt
    # Pick best, median, worst per-sample error (same convention as Phase 1).
    order = np.argsort(per_sample_err)
    pick_idx = [int(order[0]), int(order[len(order)//2]), int(order[-1])]
    pos_list_test = split_arrays(test["pos"].reshape(-1, 2), test_offsets)
    # Shared interval-width color scale across the best/median/worst rows so a
    # narrow interval reads as visibly cooler than a wide one; without it each
    # width panel auto-normalizes and the widths look identical. The
    # miscoverage mask is a fixed 0/1 scale and truth/mean stay per-sample.
    width_vmax = float(max((hi_t[test_offsets[gi]:test_offsets[gi+1]]
                            - lo_t[test_offsets[gi]:test_offsets[gi+1]]).max()
                           for gi in pick_idx))
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for row, gi in enumerate(pick_idx):
        s, e = test_offsets[gi], test_offsets[gi+1]
        pos = pos_list_test[gi]
        yt = test_y[s:e]; yp = test_mu[s:e]
        lo = lo_t[s:e]; hi = hi_t[s:e]
        w = hi - lo
        vmax = float(max(yt.max(), yp.max()))
        panels = [
            (yt, "ground truth $\\sigma_{vm}$ [MPa]", "viridis", 0, vmax),
            (yp, "ensemble mean [MPa]", "viridis", 0, vmax),
            (w, "interval width [MPa]", "magma", 0, width_vmax),
            (((yt < lo) | (yt > hi)).astype(float),
             "miscovered nodes (1 = outside)", "Reds", 0, 1),
        ]
        for ax, (vals, title, cmap, vmn, vmx) in zip(axes[row], panels):
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=vals, s=4, cmap=cmap,
                            vmin=vmn, vmax=vmx)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=9)
        tag = ["best", "median", "worst"][row]
        cov_i = float(((yt >= lo) & (yt <= hi)).mean())
        axes[row, 0].set_ylabel(
            f"{tag} (idx {gi}, cov {cov_i*100:.1f}%, "
            f"mean w {w.mean():.2f} MPa)", fontsize=9)
    fig.suptitle("Phase-2 CQR test examples at $\\alpha = 0.10$: "
                 "truth / mean / interval width / miscoverage mask")
    fig.tight_layout(); fig.savefig(figs / "fig_phase2_intervals_example.pdf")
    plt.close(fig)

    # Figure: comparison with raw ensemble std (Phase-1 uncertainty)
    # Sample-level: std vs abs error (Phase-1 view) side-by-side with
    # CQR width vs abs error (Phase-2 view).
    per_sample_std = np.array([
        test_sd[test_offsets[i]:test_offsets[i+1]].mean()
        for i in range(len(test_offsets) - 1)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(per_sample_std, per_sample_err, s=12, color="C3",
                    alpha=0.75)
    r1 = float(np.corrcoef(per_sample_std, per_sample_err)[0, 1])
    axes[0].set_xlabel("mean ensemble std [MPa]")
    axes[0].set_ylabel("mean $|$error$|$ [MPa]")
    axes[0].set_title(f"Phase-1: ensemble std vs. error (Pearson {r1:.3f})")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(per_sample_width, per_sample_err, s=12, color="C0",
                    alpha=0.75)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("mean CQR interval width [MPa]")
    axes[1].set_ylabel("mean $|$error$|$ [MPa]")
    axes[1].set_title(
        f"Phase-2: CQR width vs. error (Pearson {pearson_w_err:.3f})")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Phase-1 raw uncertainty vs. Phase-2 CQR-calibrated width")
    fig.tight_layout(); fig.savefig(figs / "fig_phase2_comparison.pdf")
    plt.close(fig)

    print(f"[cqr] wrote figures -> {figs}")
    print(f"[cqr] wrote tables  -> {tables}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base", default="runs/ensemble",
                    help="ensemble base dir with seed*/ subfolders + stats.pt")
    ap.add_argument("--out-dir", default="runs/cqr",
                    help="where to write predictions cache + calibration")
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[0, 101, 202, 303, 404])
    ap.add_argument("--recompute", action="store_true",
                    help="ignore cached predictions and re-run inference")
    args = ap.parse_args()
    compute_all(args)


if __name__ == "__main__":
    main()
