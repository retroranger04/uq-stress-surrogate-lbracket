"""Paper-revision ablations: vanilla split-conformal + ensemble-size sweep.

Consumes the frozen 5-member ensemble (runs/ensemble/seed{0,101,202,303,404})
and produces:
  - runs/cqr/vanilla_conformal_results.json
  - runs/ensemble/ablation_ensemble_size.json

No retraining. Runs all 5 members forward once on val + test and caches the
per-member predictions; then derives vanilla conformal + ensemble-size metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.runtime import (
    RunCfg, apply_stats_inplace, build_model, load_bundle, load_stats,
)
from src.uq.cqr import (
    base_quantiles, conformity_scores, conformal_quantile, coverage,
    width, z_for_alpha,
)

SEEDS = [0, 101, 202, 303, 404]
ALPHAS = [0.20, 0.10, 0.05]
PRIMARY_ALPHA = 0.10
ENS_BASE = ROOT / "runs" / "ensemble"
CQR_DIR = ROOT / "runs" / "cqr"


def load_member(seed: int, device):
    blob = torch.load(str(ENS_BASE / f"seed{seed}" / "best.pt"),
                      weights_only=False, map_location=device)
    cfg = RunCfg(**blob["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_split(model, items, device, batch_size=16):
    loader = DataLoader(items, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index,
                     batch.edge_attr).squeeze(-1).cpu().numpy()
        ptr = batch.ptr.cpu().numpy()
        for i in range(len(ptr) - 1):
            out.append(pred[ptr[i]:ptr[i+1]])
    return out


def per_member_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_stats(ENS_BASE / "stats.pt")
    val_items = load_bundle(ROOT / "data" / "val.pt")
    test_items = load_bundle(ROOT / "data" / "test.pt")
    apply_stats_inplace(val_items, stats)
    apply_stats_inplace(test_items, stats)
    print(f"Loaded val={len(val_items)} test={len(test_items)}")

    # Concatenate ground truth in graph-order
    val_gt_list = [g.y.squeeze(-1).cpu().numpy() for g in val_items]
    test_gt_list = [g.y.squeeze(-1).cpu().numpy() for g in test_items]
    val_sizes = np.array([len(x) for x in val_gt_list])
    test_sizes = np.array([len(x) for x in test_gt_list])
    val_offsets = np.concatenate([[0], np.cumsum(val_sizes)]).astype(np.int64)
    test_offsets = np.concatenate([[0], np.cumsum(test_sizes)]).astype(np.int64)

    per_mem_val = []
    per_mem_test = []
    for s in SEEDS:
        print(f"[seed {s}] loading + predicting ...", flush=True)
        m = load_member(s, device)
        v = np.concatenate(predict_split(m, val_items, device)).astype(np.float32)
        t = np.concatenate(predict_split(m, test_items, device)).astype(np.float32)
        per_mem_val.append(v)
        per_mem_test.append(t)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    V = np.stack(per_mem_val, axis=0)   # (M, N_val_nodes)
    T = np.stack(per_mem_test, axis=0)  # (M, N_test_nodes)
    val_gt = np.concatenate(val_gt_list).astype(np.float32)
    test_gt = np.concatenate(test_gt_list).astype(np.float32)
    return V, T, val_gt, test_gt, val_offsets, test_offsets


def vanilla_split_conformal(V, T, val_gt, test_gt):
    """Vanilla split-conformal: E_i = |ensemble_mean - y|; interval is mean +/- Q.

    Returns dict keyed by nominal level.
    """
    val_mean = V.mean(axis=0)
    test_mean = T.mean(axis=0)

    # Val nonconformity
    val_scores = np.abs(val_mean - val_gt)
    n = val_scores.size
    results = {}
    for alpha in ALPHAS:
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        Q = float(np.quantile(val_scores, q_level))
        lo = test_mean - Q
        hi = test_mean + Q
        in_int = (test_gt >= lo) & (test_gt <= hi)
        cov = float(in_int.mean())
        w = float((hi - lo).mean())
        results[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "nominal_coverage": 1 - alpha,
            "Q_vanilla_MPa": Q,
            "empirical_coverage_test": cov,
            "mean_width_test": w,
        }
    return results


def ensemble_size_ablation(V, T, val_gt, test_gt, test_offsets):
    """For M in {2,3,4,5}, compute per-node MAPE, peak MAPE, CQR coverage@90."""
    out = {}
    eps = 1.0  # MPa (consistent with eval_metrics)
    alpha = PRIMARY_ALPHA
    z = z_for_alpha(alpha)
    n_val = val_gt.size

    for M in [2, 3, 4, 5]:
        seeds = SEEDS[:M]
        Vm = V[:M]
        Tm = T[:M]
        mu_v = Vm.mean(0)
        sig_v = Vm.std(0, ddof=0)
        mu_t = Tm.mean(0)
        sig_t = Tm.std(0, ddof=0)

        # Per-node MAPE (mean over nodes of |mu - gt| / max(|gt|, eps))
        per_node_mape = float(
            (np.abs(mu_t - test_gt) / np.maximum(np.abs(test_gt), eps)).mean() * 100.0
        )

        # Peak MAPE (per-graph peak-stress error)
        peaks_pred = []
        peaks_gt = []
        for i in range(len(test_offsets) - 1):
            a, b = test_offsets[i], test_offsets[i+1]
            peaks_pred.append(mu_t[a:b].max())
            peaks_gt.append(test_gt[a:b].max())
        peaks_pred = np.asarray(peaks_pred)
        peaks_gt = np.asarray(peaks_gt)
        peak_mape = float((np.abs(peaks_pred - peaks_gt) /
                           np.maximum(np.abs(peaks_gt), eps)).mean() * 100.0)

        # CQR calibration at alpha=0.10
        q_lo_v = mu_v - z * sig_v
        q_hi_v = mu_v + z * sig_v
        E_v = conformity_scores(val_gt, q_lo_v, q_hi_v)
        Q = conformal_quantile(E_v, alpha)

        q_lo_t = mu_t - z * sig_t
        q_hi_t = mu_t + z * sig_t
        lo = q_lo_t - Q
        hi = q_hi_t + Q
        cov = float(((test_gt >= lo) & (test_gt <= hi)).mean())
        w = float((hi - lo).mean())

        # Mean ensemble std on test (sample-level mean of per-node std)
        mean_std = float(sig_t.mean())

        out[f"M_{M}"] = {
            "M": M,
            "seeds": seeds,
            "per_node_mape_pct": per_node_mape,
            "peak_mape_pct": peak_mape,
            "cqr_Q_hat_MPa": float(Q),
            "cqr_coverage_at_90": cov,
            "cqr_mean_width_MPa": w,
            "mean_ensemble_std_MPa": mean_std,
        }
    return out


def main():
    V, T, val_gt, test_gt, val_offsets, test_offsets = per_member_predictions()
    print("Predictions shape:", V.shape, T.shape)

    # Item 8: vanilla split-conformal
    vanilla = vanilla_split_conformal(V, T, val_gt, test_gt)

    # Compare against CQR from existing calibration.json
    with open(CQR_DIR / "calibration.json") as f:
        cqr_cal = json.load(f)
    cqr_sweep = cqr_cal["sweep"]
    comparison = {}
    for a in ALPHAS:
        key = f"alpha_{a:.2f}"
        cqr_key = f"alpha_{a:.2f}"
        comparison[key] = {
            "nominal_coverage": 1 - a,
            "vanilla": vanilla[key],
            "cqr": {
                "empirical_coverage_test": cqr_sweep[cqr_key]["empirical_coverage_test"],
                "mean_width_test": cqr_sweep[cqr_key]["mean_width_test"],
                "q_hat_MPa": cqr_sweep[cqr_key]["q_hat_MPa"],
            },
        }
    vanilla_out = {
        "description": "Vanilla split-conformal baseline: E=|mu-y|; interval = mu +/- Q",
        "val_calibration_n_nodes": int(val_gt.size),
        "test_n_nodes": int(test_gt.size),
        "results": vanilla,
        "comparison_with_cqr": comparison,
    }
    (CQR_DIR / "vanilla_conformal_results.json").write_text(
        json.dumps(vanilla_out, indent=2))
    print("Wrote vanilla_conformal_results.json")

    # Item 9: ensemble size ablation
    ablation = ensemble_size_ablation(V, T, val_gt, test_gt, test_offsets)
    ablation_out = {
        "description": "Ensemble-size ablation M in {2,3,4,5}. Re-uses frozen seeds; re-calibrates CQR at alpha=0.10 per M.",
        "primary_alpha": PRIMARY_ALPHA,
        "results": ablation,
    }
    (ENS_BASE / "ablation_ensemble_size.json").write_text(
        json.dumps(ablation_out, indent=2))
    print("Wrote ablation_ensemble_size.json")

    print("\n=== SUMMARY ===")
    print("\nVanilla conformal vs CQR:")
    for k, v in comparison.items():
        print(f"  {k}: nominal={v['nominal_coverage']:.2f} | "
              f"vanilla cov={v['vanilla']['empirical_coverage_test']:.4f} "
              f"w={v['vanilla']['mean_width_test']:.4f}MPa Q={v['vanilla']['Q_vanilla_MPa']:.4f} | "
              f"CQR cov={v['cqr']['empirical_coverage_test']:.4f} "
              f"w={v['cqr']['mean_width_test']:.4f}MPa")
    print("\nEnsemble-size ablation:")
    for k, v in ablation.items():
        print(f"  {k}: per-node MAPE={v['per_node_mape_pct']:.2f}%  "
              f"peak MAPE={v['peak_mape_pct']:.2f}%  "
              f"CQR cov@90={v['cqr_coverage_at_90']:.4f}  "
              f"width={v['cqr_mean_width_MPa']:.4f}MPa  "
              f"std={v['mean_ensemble_std_MPa']:.4f}MPa")


if __name__ == "__main__":
    main()
