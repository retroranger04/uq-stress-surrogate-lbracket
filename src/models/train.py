"""
Training skeleton for the Phase-1 GNN surrogate.

Not runnable end-to-end until the real Kaggle-published training dataset is
available (Phase C of Day 3 finishes the publishing). The structure is
complete \u2014 dataset path + output dir are the only knobs Day-4 should need
to twist. Ensemble training (Phase 2 of the paper) reuses this same loop
with different seeds and a different output dir.

Design choices locked in `paper/NOTES.md` "Phase 1 experiment plan":
    - Loss: per-node Huber (delta=1.0 MPa)
    - Optimizer: Adam (default beta)
    - LR schedule: cosine decay to 1e-5
    - Batch size: 8 graphs
    - Epochs: 200 with early-stopping patience 25 on val MSE
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from .gnn import MeshGNN, MeshGNNConfig
from .dataset import (
    LBracketStressDataset, FeatureStats, compute_stats,
    lhs_stratified_split, NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
)


@dataclass
class TrainConfig:
    data_root: str
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
    huber_delta: float = 1.0
    flat_npz: bool = False


def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _gather_params(dataset: LBracketStressDataset) -> np.ndarray:
    rows = []
    for i in range(len(dataset._paths)):
        d = dataset.get(i)
        rows.append(d.params.squeeze(0).numpy())
    return np.stack(rows, axis=0)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader, FeatureStats]:
    # Load the full set once (no stats), collect parameter rows, split.
    full = LBracketStressDataset(cfg.data_root, flat=cfg.flat_npz)
    params = _gather_params(full)
    train_idx, val_idx, test_idx = lhs_stratified_split(
        params, train_frac=0.8, val_frac=0.1, seed=cfg.seed,
    )

    paths = full.paths
    train_set_raw = LBracketStressDataset(
        cfg.data_root, sample_paths=[paths[i] for i in train_idx],
        flat=cfg.flat_npz,
    )
    stats = compute_stats(train_set_raw, max_samples=min(200, len(train_idx)))

    train_set = LBracketStressDataset(
        cfg.data_root, sample_paths=[paths[i] for i in train_idx],
        stats=stats, flat=cfg.flat_npz,
    )
    val_set = LBracketStressDataset(
        cfg.data_root, sample_paths=[paths[i] for i in val_idx],
        stats=stats, flat=cfg.flat_npz,
    )
    test_set = LBracketStressDataset(
        cfg.data_root, sample_paths=[paths[i] for i in test_idx],
        stats=stats, flat=cfg.flat_npz,
    )
    return (
        DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=cfg.batch_size),
        DataLoader(test_set, batch_size=cfg.batch_size),
        stats,
    )


def _forward_loss(model: MeshGNN, batch, loss_fn) -> tuple[torch.Tensor, torch.Tensor]:
    pred = model(batch.x, batch.edge_index, batch.edge_attr)
    loss = loss_fn(pred, batch.y)
    return pred, loss


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, stats = build_loaders(cfg)
    model = MeshGNN(MeshGNNConfig(
        in_node_dim=NUM_NODE_FEATURES,
        in_edge_dim=NUM_EDGE_FEATURES,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        out_dim=1,
    )).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    loss_fn = nn.HuberLoss(delta=cfg.huber_delta)

    best_val = math.inf
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = _forward_loss(model, batch, loss_fn)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, loss = _forward_loss(model, batch, loss_fn)
                val_losses.append(float(loss))
        scheduler.step()

        tr_mean = float(np.mean(train_losses)) if train_losses else math.nan
        va_mean = float(np.mean(val_losses)) if val_losses else math.nan
        history.append(dict(epoch=epoch, train=tr_mean, val=va_mean,
                            lr=optimizer.param_groups[0]["lr"]))

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


def _parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flat-npz", action="store_true")
    ns = ap.parse_args()
    return TrainConfig(
        data_root=ns.data_root, out_dir=ns.out_dir,
        hidden=ns.hidden, num_layers=ns.num_layers,
        lr=ns.lr, epochs=ns.epochs, batch_size=ns.batch_size,
        seed=ns.seed, device=ns.device, flat_npz=ns.flat_npz,
    )


if __name__ == "__main__":
    train(_parse_args())
