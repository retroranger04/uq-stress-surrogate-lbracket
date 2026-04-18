"""
Phase-1 training CLI: train a single MeshGraphNet on the .pt bundles.

Usage:
    python scripts/phase1_train.py --out runs/baseline --epochs 200 --seed 0
    python scripts/phase1_train.py --out runs/ens/seed1 --epochs 200 --seed 1 --stats runs/baseline/stats.pt

Reuses src/models/runtime.py. If --stats is passed, loads pre-computed
normalization stats instead of recomputing from the train split — used for
ensemble members so they share the same input normalization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.runtime import (
    RunCfg, apply_stats_inplace, compute_stats_from_list, eval_metrics,
    load_bundle, load_stats, save_stats, train_one,
)
from torch_geometric.loader import DataLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["mse", "huber"], default="huber")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--stats", type=str, default="",
                    help="load normalization stats from path instead of recomputing")
    ap.add_argument("--tick-every", type=int, default=1,
                    help="print one line every N epochs (stdout-flushed)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"[phase1] loading bundles from {args.data_dir}", flush=True)
    t0 = time.time()
    tr = load_bundle(Path(args.data_dir) / "train.pt")
    va = load_bundle(Path(args.data_dir) / "val.pt")
    print(f"[phase1] loaded tr={len(tr)} va={len(va)} in {time.time()-t0:.1f}s",
          flush=True)

    if args.stats:
        stats = load_stats(args.stats)
        print(f"[phase1] loaded stats from {args.stats}", flush=True)
    else:
        stats = compute_stats_from_list(tr, max_samples=200)
        save_stats(stats, out / "stats.pt")
        print(f"[phase1] computed + saved stats -> {out/'stats.pt'}", flush=True)

    apply_stats_inplace(tr, stats)
    apply_stats_inplace(va, stats)

    cfg = RunCfg(
        out_dir=str(out),
        hidden=args.hidden, num_layers=args.num_layers,
        lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience, seed=args.seed,
        loss=args.loss, huber_delta=args.huber_delta,
    )
    (out / "cfg.json").write_text(json.dumps(cfg.__dict__, indent=2))
    print(f"[phase1] cfg: {cfg}", flush=True)

    def _tick(epoch: int, row: dict) -> None:
        if epoch % args.tick_every == 0 or row["val"] < row.get("val", 1e18):
            print(f"  epoch {epoch:3d} train={row['train']:.4f} "
                  f"val={row['val']:.4f} lr={row['lr']:.2e} "
                  f"dt={row['wall_s']:.1f}s "
                  f"vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
                  flush=True)

    t0 = time.time()
    result = train_one(cfg, tr, va, progress=_tick)
    wall = time.time() - t0
    print(f"[phase1] done in {wall/60:.1f}m. best_val={result['best_val']:.4f} "
          f"@epoch {result['best_epoch']}", flush=True)

    # Report final validation metrics from the best checkpoint.
    from src.models.runtime import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(str(out / "best.pt"), weights_only=False, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    val_loader = DataLoader(va, batch_size=args.batch_size)
    metrics = eval_metrics(model, val_loader, device)
    (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[phase1] val metrics: {json.dumps(metrics, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
