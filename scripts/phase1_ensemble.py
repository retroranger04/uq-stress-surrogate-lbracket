"""
Phase-1 Deep Ensemble training: train M=5 models with the chosen Phase-1
hyperparameters using independent random seeds. Uses shared normalization
stats computed once from the train split (identical input scaling across
members, per Lakshminarayanan et al. 2017).

Usage:
    python scripts/phase1_ensemble.py --base runs/ensemble \\
        --hidden 128 --num-layers 5 --lr 5e-4 --epochs 200 --loss huber
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.runtime import (
    compute_stats_from_list, load_bundle, save_stats,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base", required=True)
    ap.add_argument("--num-members", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--loss", choices=["mse", "huber"], default="huber")
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[101, 202, 303, 404, 505])
    args = ap.parse_args()

    base = Path(args.base); base.mkdir(parents=True, exist_ok=True)

    # Pre-compute + save shared stats so each member reuses it.
    stats_path = base / "stats.pt"
    if not stats_path.exists():
        print(f"[ens] computing shared normalization stats from {args.data_dir}/train.pt")
        tr = load_bundle(Path(args.data_dir) / "train.pt")
        stats = compute_stats_from_list(tr, max_samples=200)
        save_stats(stats, stats_path)
        del tr
        print(f"[ens] saved {stats_path}")

    seeds = args.seeds[: args.num_members]
    assert len(seeds) == args.num_members

    member_summaries = []
    for i, seed in enumerate(seeds):
        mdir = base / f"seed{seed}"
        if (mdir / "best.pt").exists():
            print(f"[ens] member {i} (seed={seed}) already trained, skipping")
            continue
        cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / "phase1_train.py"),
            "--data-dir", args.data_dir,
            "--out", str(mdir),
            "--hidden", str(args.hidden),
            "--num-layers", str(args.num_layers),
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--patience", str(args.patience),
            "--seed", str(seed),
            "--loss", args.loss,
            "--stats", str(stats_path),
        ]
        print(f"[ens] === member {i} (seed={seed}) ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        print(f"[ens] member {i} done in {(time.time()-t0)/60:.1f}m (rc={rc})",
              flush=True)
        if rc != 0:
            raise RuntimeError(f"member {i} failed with rc={rc}")

    # Summarize
    for seed in seeds:
        mdir = base / f"seed{seed}"
        vm = json.loads((mdir / "val_metrics.json").read_text())
        member_summaries.append({"seed": seed, **vm})
    (base / "members.json").write_text(json.dumps(member_summaries, indent=2))
    print(json.dumps(member_summaries, indent=2))


if __name__ == "__main__":
    main()
