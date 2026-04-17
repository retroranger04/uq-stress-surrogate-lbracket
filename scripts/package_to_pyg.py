"""
Package Day-3 sweep outputs into PyTorch Geometric `.pt` bundles.

Reads the per-sample `.npz` shards produced by the Kaggle sweeps and emits:
    data/train.pt  (~80% of main sweep, LHS-stratified)
    data/val.pt    (~10% of main sweep, LHS-stratified)
    data/test.pt   (~10% of main sweep, LHS-stratified, in-distribution)
    data/ood.pt    (100% of OOD sweep, pre-registered protocol)
    data/split_manifest.json  (for reproducibility)

Each .pt is a list of torch_geometric.data.Data objects. The OOD split is
NEVER mixed into train/val/test \u2014 that commitment is enforced here and
logged in the manifest.

Usage:
    python scripts/package_to_pyg.py \\
        --train-root data/day3_main \\
        --ood-root data/day3_ood \\
        --out data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.dataset import npz_to_data, lhs_stratified_split


def load_all(samples_dir: Path) -> tuple[list, list[str]]:
    paths = sorted(samples_dir.glob("*.npz"))
    data_list = []
    names = []
    for p in paths:
        try:
            data_list.append(npz_to_data(p))
            names.append(p.name)
        except Exception as e:
            print(f"  SKIP {p.name}: {type(e).__name__}: {e}")
    return data_list, names


def _finite_check(data) -> bool:
    return bool(
        torch.isfinite(data.x).all().item()
        and torch.isfinite(data.edge_attr).all().item()
        and torch.isfinite(data.y).all().item()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, required=True,
                    help="dir containing samples/*.npz and manifest.json from the main sweep")
    ap.add_argument("--ood-root", type=Path, required=True,
                    help="same for the OOD sweep")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Main sweep --------------------------------------------------------
    main_data, main_names = load_all(args.train_root / "samples")
    finite_mask = [_finite_check(d) for d in main_data]
    excluded_main = [n for n, ok in zip(main_names, finite_mask) if not ok]
    main_data = [d for d, ok in zip(main_data, finite_mask) if ok]
    main_names = [n for n, ok in zip(main_names, finite_mask) if ok]
    params = np.stack([d.params.squeeze(0).numpy() for d in main_data], axis=0)
    print(f"Main sweep: {len(main_data)} finite samples "
          f"(excluded {len(excluded_main)} non-finite)")

    tr_idx, va_idx, te_idx = lhs_stratified_split(
        params, train_frac=0.8, val_frac=0.1, seed=args.seed
    )
    train_set = [main_data[i] for i in tr_idx]
    val_set = [main_data[i] for i in va_idx]
    test_set = [main_data[i] for i in te_idx]

    # --- OOD sweep (all OOD samples stay in ood.pt) ------------------------
    ood_data, ood_names = load_all(args.ood_root / "samples")
    ood_finite_mask = [_finite_check(d) for d in ood_data]
    excluded_ood = [n for n, ok in zip(ood_names, ood_finite_mask) if not ok]
    ood_data = [d for d, ok in zip(ood_data, ood_finite_mask) if ok]
    ood_names = [n for n, ok in zip(ood_names, ood_finite_mask) if ok]
    print(f"OOD sweep: {len(ood_data)} finite samples "
          f"(excluded {len(excluded_ood)} non-finite)")

    # --- Write ------------------------------------------------------------
    torch.save(train_set, args.out / "train.pt")
    torch.save(val_set,   args.out / "val.pt")
    torch.save(test_set,  args.out / "test.pt")
    torch.save(ood_data,  args.out / "ood.pt")
    print(f"wrote train={len(train_set)} val={len(val_set)} "
          f"test={len(test_set)} ood={len(ood_data)}")

    # Per-direction OOD breakdown for the manifest.
    ood_by_direction: dict = {}
    for d in ood_data:
        k = getattr(d, "direction", "")
        ood_by_direction[k] = ood_by_direction.get(k, 0) + 1

    manifest = dict(
        seed=args.seed,
        train=dict(count=len(train_set),
                    names=[main_names[i] for i in tr_idx]),
        val=dict(count=len(val_set),
                  names=[main_names[i] for i in va_idx]),
        test=dict(count=len(test_set),
                   names=[main_names[i] for i in te_idx]),
        ood=dict(count=len(ood_data),
                  by_direction=ood_by_direction,
                  names=ood_names),
        excluded=dict(main=excluded_main, ood=excluded_ood),
        isolation_check=(
            set(main_names[i] for i in tr_idx).isdisjoint(
                main_names[i] for i in va_idx
            )
            and set(main_names[i] for i in tr_idx).isdisjoint(
                main_names[i] for i in te_idx
            )
            and set(main_names[i] for i in va_idx).isdisjoint(
                main_names[i] for i in te_idx
            )
        ),
    )
    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    assert manifest["isolation_check"], "train/val/test index sets are not disjoint"
    print(f"wrote {args.out/'split_manifest.json'}")


if __name__ == "__main__":
    main()
