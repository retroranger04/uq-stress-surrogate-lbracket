"""
Phase-1 runtime helpers: loading pre-packaged .pt bundles, computing
normalization stats from a list of Data objects, applying stats in-place,
training loop, metric computation.

Kept as a small module alongside the Phase-0 `train.py` skeleton so the .pt
bundle workflow (what Phase 1+ actually uses) stays distinct from the
npz-directory workflow that built the bundles. See agent_log.md 2026-04-20
for the split rationale.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .dataset import FeatureStats, NUM_NODE_FEATURES, NUM_EDGE_FEATURES
from .gnn import MeshGNN, MeshGNNConfig


def load_bundle(path: str | Path) -> list[Data]:
    return torch.load(str(path), weights_only=False)


def compute_stats_from_list(items: list[Data],
                            max_samples: int = 200) -> FeatureStats:
    n = min(len(items), max_samples)
    xs, es, ys = [], [], []
    for i in range(n):
        d = items[i]
        xs.append(d.x.numpy())
        es.append(d.edge_attr.numpy())
        ys.append(d.y.numpy())
    X = np.concatenate(xs, 0)
    E = np.concatenate(es, 0)
    Y = np.concatenate(ys, 0)
    return FeatureStats(
        node_mean=torch.from_numpy(X.mean(0).astype(np.float32)),
        node_std=torch.from_numpy(np.maximum(X.std(0), 1e-4).astype(np.float32)),
        edge_mean=torch.from_numpy(E.mean(0).astype(np.float32)),
        edge_std=torch.from_numpy(np.maximum(E.std(0), 1e-4).astype(np.float32)),
        y_mean=torch.from_numpy(Y.mean(0).astype(np.float32)),
        y_std=torch.from_numpy(np.maximum(Y.std(0), 1e-4).astype(np.float32)),
    )


def apply_stats_inplace(items: list[Data], stats: FeatureStats) -> None:
    """Apply input normalization to every Data in the list in-place."""
    for d in items:
        d.x = (d.x - stats.node_mean) / stats.node_std
        d.edge_attr = (d.edge_attr - stats.edge_mean) / stats.edge_std


def save_stats(stats: FeatureStats, path: str | Path) -> None:
    torch.save({
        "node_mean": stats.node_mean, "node_std": stats.node_std,
        "edge_mean": stats.edge_mean, "edge_std": stats.edge_std,
        "y_mean": stats.y_mean, "y_std": stats.y_std,
    }, str(path))


def load_stats(path: str | Path) -> FeatureStats:
    d = torch.load(str(path), weights_only=False)
    return FeatureStats(**d)


@dataclass
class RunCfg:
    out_dir: str
    hidden: int = 128
    num_layers: int = 5
    lr: float = 5e-4
    lr_min: float = 1e-5
    epochs: int = 200
    batch_size: int = 8
    patience: int = 25
    seed: int = 0
    device: str = "cuda"
    loss: str = "huber"          # "mse" or "huber"
    huber_delta: float = 1.0


def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(cfg: RunCfg) -> MeshGNN:
    return MeshGNN(MeshGNNConfig(
        in_node_dim=NUM_NODE_FEATURES,
        in_edge_dim=NUM_EDGE_FEATURES,
        hidden=cfg.hidden, num_layers=cfg.num_layers, out_dim=1,
    ))


def loss_fn(cfg: RunCfg) -> nn.Module:
    if cfg.loss == "mse":
        return nn.MSELoss()
    return nn.HuberLoss(delta=cfg.huber_delta)


@torch.no_grad()
def eval_metrics(model: MeshGNN, loader: DataLoader, device: torch.device,
                 eps: float = 1e-3) -> dict:
    """Per-node MAPE + peak-stress MAPE + percentile abs errors.

    MAPE uses a stabilized denominator max(|y|, eps) [MPa] so near-zero
    stresses don't inflate the percent error.
    """
    model.eval()
    abs_errs, rel_errs = [], []
    peak_aps = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        err = (pred - batch.y).abs().squeeze(-1).cpu().numpy()
        y = batch.y.squeeze(-1).cpu().numpy()
        abs_errs.append(err)
        denom = np.maximum(np.abs(y), eps)
        rel_errs.append(err / denom)
        # per-graph peak
        ptr = batch.ptr.cpu().numpy()
        p = pred.squeeze(-1).cpu().numpy()
        for i in range(len(ptr) - 1):
            yi = y[ptr[i]:ptr[i+1]]
            pi = p[ptr[i]:ptr[i+1]]
            yp = float(yi.max()); pp = float(pi.max())
            peak_aps.append(abs(pp - yp) / max(abs(yp), eps))
    abs_err = np.concatenate(abs_errs)
    rel_err = np.concatenate(rel_errs)
    return {
        "per_node_mape": float(rel_err.mean() * 100.0),
        "peak_mape": float(np.mean(peak_aps) * 100.0),
        "abs_err_p50": float(np.percentile(abs_err, 50)),
        "abs_err_p90": float(np.percentile(abs_err, 90)),
        "abs_err_p99": float(np.percentile(abs_err, 99)),
        "abs_err_max": float(abs_err.max()),
    }


def train_one(cfg: RunCfg, train_items: list[Data], val_items: list[Data],
              progress: Callable[[int, dict], None] | None = None) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_items, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_items, batch_size=cfg.batch_size)

    model = build_model(cfg).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
    lossf = loss_fn(cfg)

    best_val = math.inf
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        tr_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = lossf(pred, batch.y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.detach()))
        tr_mean = float(np.mean(tr_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                val_losses.append(float(lossf(pred, batch.y)))
        va_mean = float(np.mean(val_losses))
        sched.step()

        dt = time.time() - t0
        row = dict(epoch=epoch, train=tr_mean, val=va_mean,
                   lr=opt.param_groups[0]["lr"], wall_s=dt)
        history.append(row)
        if progress is not None:
            progress(epoch, row)

        if va_mean < best_val - 1e-6:
            best_val, best_epoch, stale = va_mean, epoch, 0
            torch.save({"model": model.state_dict(),
                         "cfg": asdict(cfg)}, out / "best.pt")
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    (out / "history.json").write_text(json.dumps({
        "config": asdict(cfg),
        "history": history,
        "best_val": best_val,
        "best_epoch": best_epoch,
    }, indent=2))
    return {"best_val": best_val, "best_epoch": best_epoch, "history": history}


def load_best(cfg_like: RunCfg, ckpt_path: str | Path,
              device: torch.device) -> MeshGNN:
    blob = torch.load(str(ckpt_path), weights_only=False, map_location=device)
    cfg = RunCfg(**blob["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model
