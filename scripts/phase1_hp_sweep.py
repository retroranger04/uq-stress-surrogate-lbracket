"""
Phase-1 lightweight hyperparameter sweep.

Trains three candidate configs for a short budget, ranks by validation
per-node MAPE. Decision is locked before Step-6 ensemble training. The
test split is NEVER touched here.

Candidates (per paper/NOTES.md Phase-1 experiment plan):
    - (H=128, L=5, lr=5e-4)  # default; matches the already-trained baseline
    - (H=64,  L=5, lr=5e-4)  # narrower hidden
    - (H=128, L=3, lr=5e-4)  # shallower processor
    - (H=128, L=5, lr=1e-3)  # larger LR

To avoid redoing the baseline we use runs/baseline's best.pt as the
(128, 5, 5e-4) datapoint. The remaining three run for --epochs epochs.

Usage:
    python scripts/phase1_hp_sweep.py --epochs 25 --base runs/baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.runtime import (
    RunCfg, apply_stats_inplace, build_model, eval_metrics, load_bundle,
    load_stats,
)


def _eval_best(run_dir: Path, stats_path: Path, data_dir: Path,
               device: torch.device) -> dict:
    va = load_bundle(data_dir / "val.pt")
    stats = load_stats(stats_path)
    apply_stats_inplace(va, stats)
    blob = torch.load(str(run_dir / "best.pt"), weights_only=False,
                      map_location=device)
    cfg = RunCfg(**blob["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    loader = DataLoader(va, batch_size=16, shuffle=False)
    return eval_metrics(model, loader, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="runs/hp_sweep")
    ap.add_argument("--base", default="runs/baseline",
                    help="existing baseline run dir (128/5/5e-4)")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)
    stats = base / "stats.pt"
    assert stats.exists(), f"missing {stats}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = [
        {"tag": "h64_L5_lr5e-4",  "hidden": 64,  "num_layers": 5, "lr": 5e-4},
        {"tag": "h128_L3_lr5e-4", "hidden": 128, "num_layers": 3, "lr": 5e-4},
    ]

    summary = []
    # Baseline entry (reusing full-budget run)
    base_m = _eval_best(base, stats, Path(args.data_dir), device)
    base_m.update({"tag": "h128_L5_lr5e-4 (baseline)", "hidden": 128,
                   "num_layers": 5, "lr": 5e-4, "budget": "full"})
    summary.append(base_m)
    print(f"[hp] baseline val: {base_m}", flush=True)

    for c in candidates:
        mdir = out / c["tag"]
        if not (mdir / "best.pt").exists():
            cmd = [
                sys.executable, "-u",
                str(ROOT / "scripts" / "phase1_train.py"),
                "--data-dir", args.data_dir,
                "--out", str(mdir),
                "--hidden", str(c["hidden"]),
                "--num-layers", str(c["num_layers"]),
                "--lr", str(c["lr"]),
                "--epochs", str(args.epochs),
                "--batch-size", "8",
                "--patience", "200",   # no early stop; short budget
                "--seed", "0",
                "--loss", "huber",
                "--stats", str(stats),
            ]
            print(f"[hp] === {c['tag']} ({args.epochs} ep) ===", flush=True)
            t0 = time.time()
            rc = subprocess.call(cmd)
            print(f"[hp] {c['tag']} done in {(time.time()-t0)/60:.1f}m "
                  f"(rc={rc})", flush=True)
            if rc != 0:
                raise RuntimeError(f"candidate {c['tag']} failed")
        m = _eval_best(mdir, stats, Path(args.data_dir), device)
        m.update({**c, "budget": f"{args.epochs}ep"})
        summary.append(m)
        print(f"[hp] {c['tag']} val: {m}", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    ranked = sorted(summary, key=lambda r: r["per_node_mape"])
    print("\n[hp] ranked by val per-node MAPE (lower=better):")
    for r in ranked:
        print(f"  {r['per_node_mape']:.3f}%  peak={r['peak_mape']:.3f}%  "
              f"tag={r['tag']}  budget={r['budget']}")
    winner = ranked[0]
    (out / "winner.json").write_text(json.dumps(winner, indent=2))
    print(f"\n[hp] winner: {winner['tag']}")


if __name__ == "__main__":
    main()
