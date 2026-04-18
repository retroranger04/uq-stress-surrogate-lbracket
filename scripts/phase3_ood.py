"""
Phase-3 OOD evaluation on the frozen Phase-1 Deep Ensemble + Phase-2 CQR layer.

Steps:
  1. Load the 250-sample OOD bundle (data/ood.pt). Never touched in Phase 1/2.
  2. Run the 5 frozen ensemble members on OOD, cache mean/std/gt/pos/params
     + per-graph direction tag to runs/cqr/preds_ood.npz.
  3. Compute ID vs OOD comparison metrics at the primary alpha = 0.10
     (per-node MAPE, peak-stress MAPE, CQR empirical coverage, CQR interval
     widths, ensemble-std distribution).
  4. Per-condition OOD breakdown by the pre-registered direction tag
     {R_low, R_high, p_low, p_high, W_low, W_high, corner}.
  5. Deferral decision rule: calibrate a threshold T on per-sample mean
     ensemble std from the ID test quantile, then report false-alarm rate on
     ID and detection rate on OOD across a sweep of thresholds.
  6. Emit paper tables (paper/tables/phase3_*_rows.tex) and four figures:
     fig_phase3_id_vs_ood_accuracy.pdf,
     fig_phase3_ood_breakdown.pdf,
     fig_phase3_ood_examples.pdf,
     fig_phase3_deferral_roc.pdf.

The OOD bundle is used only here, after the ensemble and the CQR q_hat are
frozen. No hyperparameter, threshold, or interval rule is chosen on OOD
data.
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
    base_quantiles, calibrated_interval, conformity_scores,
    conformal_quantile, coverage, width, z_for_alpha,
)


# ------------------ model loading + inference (mirror of phase2) -----------

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


def save_predictions_npz(path: Path, means, stds, gts, params, pos_list,
                         directions):
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
        directions=np.asarray(directions),
    )


def load_predictions_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    return d


# ------------------ metrics ---------------------------------------------

def per_sample_metrics(mu, y, offsets, eps=1.0):
    """Per-node MAPE and peak-stress MAPE arrays indexed by sample."""
    G = len(offsets) - 1
    pn_mape = np.zeros(G, dtype=np.float64)
    pk_mape = np.zeros(G, dtype=np.float64)
    for i in range(G):
        s, e = offsets[i], offsets[i+1]
        yi = y[s:e]; mi = mu[s:e]
        pn_mape[i] = float(np.mean(np.abs(mi - yi) / np.maximum(np.abs(yi), eps)))
        pk_mape[i] = float(abs(mi.max() - yi.max())
                           / max(abs(yi.max()), eps))
    return pn_mape, pk_mape


def per_sample_coverage_width_std_err(mu, sd, y, lo, hi, offsets):
    G = len(offsets) - 1
    cov = np.zeros(G); wid = np.zeros(G); std_ = np.zeros(G); err = np.zeros(G)
    for i in range(G):
        s, e = offsets[i], offsets[i+1]
        yi, mi = y[s:e], mu[s:e]
        lo_i, hi_i = lo[s:e], hi[s:e]
        cov[i] = float(((yi >= lo_i) & (yi <= hi_i)).mean())
        wid[i] = float((hi_i - lo_i).mean())
        std_[i] = float(sd[s:e].mean())
        err[i] = float(np.abs(mi - yi).mean())
    return cov, wid, std_, err


# ------------------ main driver -----------------------------------------

def compute_all(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = load_stats(base / "stats.pt")

    # --- Load cached ID val/test predictions (from Phase 2) ---
    val = load_predictions_npz(out_dir / "preds_val.npz")
    test = load_predictions_npz(out_dir / "preds_test.npz")

    # --- Run the ensemble on the OOD bundle (or use cached) ---
    ood_cache = out_dir / "preds_ood.npz"
    if ood_cache.exists() and not args.recompute:
        print(f"[cache] OOD: using cached {ood_cache}")
        ood = load_predictions_npz(ood_cache)
    else:
        print(f"[cache] OOD: running 5 members over data/ood.pt")
        items = load_bundle(Path(args.data_dir) / "ood.pt")
        directions = [str(d.direction) for d in items]
        gts = [d.y.squeeze(-1).numpy().astype(np.float32) for d in items]
        params = np.stack([d.params.squeeze(0).numpy() for d in items]
                          ).astype(np.float32)
        pos_list = [d.pos.numpy().astype(np.float32) for d in items]
        apply_stats_inplace(items, stats)
        means, stds = ensemble_predict(base, args.seeds, items, device)
        save_predictions_npz(ood_cache, means, stds, gts, params, pos_list,
                             directions)
        ood = load_predictions_npz(ood_cache)

    # --- CQR primary-alpha artifact (frozen from Phase 2) ---
    primary = 0.10
    val_mu, val_sd, val_y = val["mean"], val["std"], val["gt"]
    q_lo_v, q_hi_v = base_quantiles(val_mu, val_sd, primary)
    q_hat = conformal_quantile(conformity_scores(val_y, q_lo_v, q_hi_v),
                                primary)
    z_p = z_for_alpha(primary)

    # --- Intervals on ID test and OOD ---
    t_mu, t_sd, t_y, t_off = test["mean"], test["std"], test["gt"], test["offsets"]
    o_mu, o_sd, o_y, o_off = ood["mean"], ood["std"], ood["gt"], ood["offsets"]
    t_lo, t_hi = calibrated_interval(t_mu, t_sd, primary, q_hat)
    o_lo, o_hi = calibrated_interval(o_mu, o_sd, primary, q_hat)

    # Also report base (uncalibrated) Gaussian intervals for completeness.
    t_blo, t_bhi = base_quantiles(t_mu, t_sd, primary)
    o_blo, o_bhi = base_quantiles(o_mu, o_sd, primary)

    # --- Global ID vs OOD metrics ---
    id_pn, id_pk = per_sample_metrics(t_mu, t_y, t_off)
    ood_pn, ood_pk = per_sample_metrics(o_mu, o_y, o_off)

    id_cov, id_w, id_std, id_err = per_sample_coverage_width_std_err(
        t_mu, t_sd, t_y, t_lo, t_hi, t_off)
    ood_cov, ood_w, ood_std, ood_err = per_sample_coverage_width_std_err(
        o_mu, o_sd, o_y, o_lo, o_hi, o_off)

    summary = {
        "primary_alpha": primary,
        "q_hat_MPa": q_hat,
        "id_test": {
            "n_samples": int(len(t_off) - 1),
            "per_node_mape_pct": 100.0 * float(id_pn.mean()),
            "peak_stress_mape_pct": 100.0 * float(id_pk.mean()),
            "cqr_coverage": coverage(t_y, t_lo, t_hi),
            "cqr_mean_width_MPa": float((t_hi - t_lo).mean()),
            "base_coverage": coverage(t_y, t_blo, t_bhi),
            "base_mean_width_MPa": float((t_bhi - t_blo).mean()),
            "mean_ensemble_std_MPa": float(id_std.mean()),
            "median_ensemble_std_MPa": float(np.median(id_std)),
        },
        "ood": {
            "n_samples": int(len(o_off) - 1),
            "per_node_mape_pct": 100.0 * float(ood_pn.mean()),
            "peak_stress_mape_pct": 100.0 * float(ood_pk.mean()),
            "cqr_coverage": coverage(o_y, o_lo, o_hi),
            "cqr_mean_width_MPa": float((o_hi - o_lo).mean()),
            "base_coverage": coverage(o_y, o_blo, o_bhi),
            "base_mean_width_MPa": float((o_bhi - o_blo).mean()),
            "mean_ensemble_std_MPa": float(ood_std.mean()),
            "median_ensemble_std_MPa": float(np.median(ood_std)),
        },
    }

    # --- Per-condition breakdown (OOD only) ---
    directions = np.asarray(ood["directions"]).astype(str)
    cond_order = ["R_low", "R_high", "p_low", "p_high", "W_low", "W_high",
                  "corner"]
    conditions = {}
    for tag in cond_order:
        sel = np.where(directions == tag)[0]
        if sel.size == 0:
            continue
        node_idx = np.concatenate([
            np.arange(o_off[i], o_off[i+1]) for i in sel])
        conditions[tag] = {
            "n_samples": int(sel.size),
            "per_node_mape_pct": 100.0 * float(ood_pn[sel].mean()),
            "peak_stress_mape_pct": 100.0 * float(ood_pk[sel].mean()),
            "cqr_coverage": coverage(o_y[node_idx], o_lo[node_idx],
                                      o_hi[node_idx]),
            "cqr_mean_width_MPa": float(ood_w[sel].mean()),
            "mean_ensemble_std_MPa": float(ood_std[sel].mean()),
        }

    # --- Deferral decision rule ---
    # Set threshold from ID test std distribution; e.g. 95th percentile as
    # the allowed false-alarm rate at "calibration" (report a sweep too).
    T_grid = np.linspace(0, max(id_std.max(), ood_std.max()) * 1.05, 200)
    far = np.array([(id_std > T).mean() for T in T_grid])     # false alarms on ID
    det = np.array([(ood_std > T).mean() for T in T_grid])    # detections on OOD
    # Primary operating point: threshold = 95th percentile of ID std.
    T_star = float(np.quantile(id_std, 0.95))
    far_star = float((id_std > T_star).mean())
    det_star = float((ood_std > T_star).mean())
    # Secondary operating points for the table.
    ops = {}
    for q in (0.80, 0.90, 0.95, 0.99):
        T = float(np.quantile(id_std, q))
        ops[f"q{int(100*q):02d}"] = {
            "threshold_MPa": T,
            "id_false_alarm_rate": float((id_std > T).mean()),
            "ood_detection_rate": float((ood_std > T).mean()),
        }
    # Per-condition detection at T_star.
    det_by_cond = {}
    for tag in cond_order:
        sel = np.where(directions == tag)[0]
        if sel.size == 0:
            continue
        det_by_cond[tag] = float((ood_std[sel] > T_star).mean())

    # AUROC of "OOD vs ID" based on per-sample std.
    # Labels: 0 = ID (negative), 1 = OOD (positive).
    scores = np.concatenate([id_std, ood_std])
    labels = np.concatenate([np.zeros_like(id_std), np.ones_like(ood_std)])
    order = np.argsort(-scores)  # descending
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / max(labels.sum(), 1)
    fpr = fps / max((1 - labels).sum(), 1)
    auroc = float(np.trapezoid(tpr, fpr))

    deferral = {
        "score": "per_sample_mean_ensemble_std_MPa",
        "threshold_policy": "95th percentile of ID-test distribution",
        "T_star_MPa": T_star,
        "id_false_alarm_rate": far_star,
        "ood_detection_rate": det_star,
        "ood_detection_rate_by_condition": det_by_cond,
        "auroc_ood_vs_id": auroc,
        "operating_points": ops,
    }

    results = {
        "summary": summary,
        "by_condition": conditions,
        "deferral": deferral,
        "seeds": list(args.seeds),
        "n_ood_nodes": int(o_y.size),
        "n_id_nodes": int(t_y.size),
    }
    (out_dir / "ood_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # --- Paper tables ---
    tables = Path("paper/tables"); tables.mkdir(parents=True, exist_ok=True)

    # phase3_id_vs_ood table body: 4 columns (per-node MAPE, peak MAPE,
    # CQR cov, CQR width, ensemble std)
    s_id = summary["id_test"]; s_ood = summary["ood"]
    rows = [
        f"Per-node MAPE [\\%] & {s_id['per_node_mape_pct']:.2f} & "
        f"{s_ood['per_node_mape_pct']:.2f} & "
        f"{s_ood['per_node_mape_pct']/s_id['per_node_mape_pct']:.1f}$\\times$ \\\\",
        f"Peak-stress MAPE [\\%] & {s_id['peak_stress_mape_pct']:.2f} & "
        f"{s_ood['peak_stress_mape_pct']:.2f} & "
        f"{s_ood['peak_stress_mape_pct']/max(s_id['peak_stress_mape_pct'],1e-6):.1f}$\\times$ \\\\",
        f"CQR coverage at 90\\% nominal [\\%] & "
        f"{100*s_id['cqr_coverage']:.2f} & {100*s_ood['cqr_coverage']:.2f} & "
        f"${100*(s_ood['cqr_coverage']-s_id['cqr_coverage']):+.2f}$ pp \\\\",
        f"CQR mean interval width [MPa] & "
        f"{s_id['cqr_mean_width_MPa']:.3f} & {s_ood['cqr_mean_width_MPa']:.3f} & "
        f"{s_ood['cqr_mean_width_MPa']/max(s_id['cqr_mean_width_MPa'],1e-6):.1f}$\\times$ \\\\",
        f"Mean ensemble std [MPa] & "
        f"{s_id['mean_ensemble_std_MPa']:.3f} & "
        f"{s_ood['mean_ensemble_std_MPa']:.3f} & "
        f"{s_ood['mean_ensemble_std_MPa']/max(s_id['mean_ensemble_std_MPa'],1e-6):.1f}$\\times$ \\\\",
    ]
    (tables / "phase3_id_vs_ood_rows.tex").write_text("\n".join(rows) + "\n")

    # phase3_conditional: per-condition breakdown
    cond_rows = []
    pretty = {"R_low": "$R$-low", "R_high": "$R$-high", "p_low": "$p$-low",
              "p_high": "$p$-high", "W_low": "$W$-low", "W_high": "$W$-high",
              "corner": "Corner ($\\geq 2$ OOD)"}
    for tag in cond_order:
        if tag not in conditions:
            continue
        c = conditions[tag]
        det_tag = det_by_cond.get(tag, 0.0)
        cond_rows.append(
            f"{pretty[tag]} & {c['n_samples']} & "
            f"{c['per_node_mape_pct']:.2f} & "
            f"{100*c['cqr_coverage']:.2f} & "
            f"{c['cqr_mean_width_MPa']:.3f} & "
            f"{c['mean_ensemble_std_MPa']:.3f} & "
            f"{100*det_tag:.1f} \\\\")
    (tables / "phase3_conditional_rows.tex").write_text(
        "\n".join(cond_rows) + "\n")

    # phase3_deferral operating points
    op_rows = []
    for q, o in ops.items():
        op_rows.append(
            f"{int(q[1:])}\\% (T = {o['threshold_MPa']:.3f} MPa) & "
            f"{100*o['id_false_alarm_rate']:.1f} & "
            f"{100*o['ood_detection_rate']:.1f} \\\\")
    (tables / "phase3_deferral_rows.tex").write_text("\n".join(op_rows) + "\n")

    # --- Figures ---
    figs = Path("paper/figures"); figs.mkdir(parents=True, exist_ok=True)

    # Fig 1: ID vs OOD bar chart (accuracy, coverage, width, std)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["ID test", "OOD"]
    # Accuracy
    ax = axes[0]
    vals_pn = [s_id["per_node_mape_pct"], s_ood["per_node_mape_pct"]]
    vals_pk = [s_id["peak_stress_mape_pct"], s_ood["peak_stress_mape_pct"]]
    x = np.arange(2)
    ax.bar(x - 0.18, vals_pn, 0.35, label="per-node", color="C0")
    ax.bar(x + 0.18, vals_pk, 0.35, label="peak", color="C1")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("MAPE [%]"); ax.set_title("Accuracy"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    # Coverage
    ax = axes[1]
    covs = [100*s_id["cqr_coverage"], 100*s_ood["cqr_coverage"]]
    bars = ax.bar(x, covs, 0.55, color=["C0", "C3"])
    ax.axhline(90, color="k", linestyle="--", linewidth=0.8,
               label="nominal 90%")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("empirical coverage [%]"); ax.set_title("CQR coverage")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
    for b, v in zip(bars, covs):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=9)
    # Width + ensemble std
    ax = axes[2]
    widths = [s_id["cqr_mean_width_MPa"], s_ood["cqr_mean_width_MPa"]]
    stds = [s_id["mean_ensemble_std_MPa"], s_ood["mean_ensemble_std_MPa"]]
    ax.bar(x - 0.18, widths, 0.35, label="CQR width", color="C0")
    ax.bar(x + 0.18, stds, 0.35, label="ensemble std", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("MPa"); ax.set_title("Uncertainty scale")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Phase-3: ID test vs.\\ OOD at $\\alpha = 0.10$")
    fig.tight_layout(); fig.savefig(figs / "fig_phase3_id_vs_ood_accuracy.pdf")
    plt.close(fig)

    # Fig 2: per-condition breakdown
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    present = [t for t in cond_order if t in conditions]
    pn_vals = [conditions[t]["per_node_mape_pct"] for t in present]
    cov_vals = [100*conditions[t]["cqr_coverage"] for t in present]
    std_vals = [conditions[t]["mean_ensemble_std_MPa"] for t in present]
    width_vals = [conditions[t]["cqr_mean_width_MPa"] for t in present]
    xs = np.arange(len(present))
    labels = [pretty[t] for t in present]
    ax = axes[0]
    ax.bar(xs, pn_vals, 0.7, color="C0")
    ax.axhline(s_id["per_node_mape_pct"], color="k", linestyle="--",
               linewidth=0.8, label=f"ID {s_id['per_node_mape_pct']:.2f}%")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("per-node MAPE [%]"); ax.set_title("Accuracy by condition")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(xs, cov_vals, 0.7, color="C3")
    ax.axhline(90, color="k", linestyle="--", linewidth=0.8,
               label="nominal 90%")
    ax.axhline(100*s_id["cqr_coverage"], color="C2", linestyle=":",
               linewidth=0.8,
               label=f"ID {100*s_id['cqr_coverage']:.1f}%")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("CQR coverage [%]"); ax.set_title("Coverage by condition")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax = axes[2]
    ax.bar(xs - 0.18, width_vals, 0.35, label="CQR width", color="C0")
    ax.bar(xs + 0.18, std_vals, 0.35, label="ensemble std", color="C3")
    ax.axhline(s_id["cqr_mean_width_MPa"], color="C0", linestyle=":",
               linewidth=0.8)
    ax.axhline(s_id["mean_ensemble_std_MPa"], color="C3", linestyle=":",
               linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("MPa"); ax.set_title("Uncertainty by condition")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Phase-3: OOD breakdown by pre-registered extrapolation direction")
    fig.tight_layout(); fig.savefig(figs / "fig_phase3_ood_breakdown.pdf")
    plt.close(fig)

    # Fig 3: OOD example fields (best, median, worst OOD by per-sample error)
    def _split(arr, offs):
        return [arr[offs[i]:offs[i+1]] for i in range(len(offs) - 1)]
    o_pos_list = _split(ood["pos"].reshape(-1, 2), o_off)
    order = np.argsort(ood_err)
    pick = [int(order[0]), int(order[len(order)//2]), int(order[-1])]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for row, gi in enumerate(pick):
        s, e = o_off[gi], o_off[gi+1]
        pos = o_pos_list[gi]
        yt = o_y[s:e]; yp = o_mu[s:e]
        err = np.abs(yp - yt); sd_ = o_sd[s:e]
        vmax = float(max(yt.max(), yp.max()))
        panels = [
            (yt, "ground truth $\\sigma_{vm}$ [MPa]", "viridis", 0, vmax),
            (yp, "ensemble mean [MPa]", "viridis", 0, vmax),
            (err, "|error| [MPa]", "magma", 0, None),
            (sd_, "ensemble std [MPa]", "magma", 0, None),
        ]
        for ax, (vals, title, cmap, vmn, vmx) in zip(axes[row], panels):
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=vals, s=4, cmap=cmap,
                            vmin=vmn, vmax=vmx)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=9)
        tag = ["best", "median", "worst"][row]
        dtag = str(directions[gi])
        axes[row, 0].set_ylabel(
            f"{tag} OOD ({dtag}, mean |err| {err.mean():.3f} MPa, "
            f"mean std {sd_.mean():.3f} MPa)", fontsize=9)
    fig.suptitle("Phase-3 OOD examples: truth / mean / |error| / ensemble std")
    fig.tight_layout(); fig.savefig(figs / "fig_phase3_ood_examples.pdf")
    plt.close(fig)

    # Fig 4: deferral ROC + histogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(100 * far, 100 * det, "-", color="C0", label="deferral rule")
    ax.plot([100 * far_star], [100 * det_star], "o", color="C3",
            markersize=10,
            label=f"T* = {T_star:.3f} MPa (ID q95)")
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.6, alpha=0.5,
            label="chance")
    ax.set_xlabel("false-alarm rate on ID test [%]")
    ax.set_ylabel("detection rate on OOD [%]")
    ax.set_title(f"Deferral ROC (AUROC {auroc:.3f})")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    ax = axes[1]
    bins = np.linspace(
        0, max(id_std.max(), ood_std.max()) * 1.05, 40)
    ax.hist(id_std, bins=bins, alpha=0.7, color="C0", label="ID test (N=500)",
            density=True)
    ax.hist(ood_std, bins=bins, alpha=0.6, color="C3", label="OOD (N=250)",
            density=True)
    ax.axvline(T_star, color="k", linestyle="--", linewidth=0.8,
               label=f"T* = {T_star:.3f} MPa")
    ax.set_xlabel("per-sample mean ensemble std [MPa]")
    ax.set_ylabel("density")
    ax.set_title("Per-sample uncertainty: ID vs OOD")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Phase-3 deferral rule: defer when per-sample ensemble std exceeds T*")
    fig.tight_layout(); fig.savefig(figs / "fig_phase3_deferral_roc.pdf")
    plt.close(fig)

    print(f"[phase3] wrote figures -> {figs}")
    print(f"[phase3] wrote tables  -> {tables}")
    print(f"[phase3] wrote results -> {out_dir / 'ood_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base", default="runs/ensemble")
    ap.add_argument("--out-dir", default="runs/cqr")
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[0, 101, 202, 303, 404])
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    compute_all(args)


if __name__ == "__main__":
    main()
