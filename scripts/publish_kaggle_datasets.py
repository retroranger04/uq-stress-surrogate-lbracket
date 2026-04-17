"""
Publish Day-3 sweeps as two versioned Kaggle datasets.

Datasets:
    retroranger04/lbracket-stress-train  \u2014 main sweep (train+val+test .pt +
        shards + manifest + pre_manifest + split_manifest)
    retroranger04/lbracket-stress-ood    \u2014 OOD sweep (ood.pt + shards +
        pre-registered OOD manifest)

The Kaggle dataset owner is `retroranger04` (the user's GitHub/Kaggle account
per CLAUDE.md) which differs from the kernel owner `retroranger`. Both exist.

Usage:
    python scripts/publish_kaggle_datasets.py \\
        --train-root data/day3_main \\
        --ood-root data/day3_ood \\
        --packaged data \\
        --owner retroranger04
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stage_dir(src_samples: Path, manifest_json: Path, pre_manifest_json: Path,
              extra_files: list[Path], staging: Path) -> None:
    """Assemble a folder with everything that should ship in the dataset."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / "samples").mkdir()
    for p in sorted(src_samples.glob("*.npz")):
        shutil.copy(p, staging / "samples" / p.name)
    shutil.copy(manifest_json, staging / "manifest.json")
    if pre_manifest_json.exists():
        shutil.copy(pre_manifest_json, staging / "sweep_pre_manifest.json")
    for p in extra_files:
        if p.exists():
            shutil.copy(p, staging / p.name)


def init_and_publish(staging: Path, owner: str, slug: str, title: str,
                     version_notes: str, first_push: bool) -> None:
    meta = {
        "id": f"{owner}/{slug}",
        "title": title,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    if first_push:
        cmd = ["kaggle", "datasets", "create", "-p", str(staging),
               "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(staging),
               "-m", version_notes, "--dir-mode", "zip"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"kaggle datasets {'create' if first_push else 'version'} failed for {slug}")


def dataset_exists(owner: str, slug: str) -> bool:
    r = subprocess.run(
        ["kaggle", "datasets", "status", f"{owner}/{slug}"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--ood-root", type=Path, required=True)
    ap.add_argument("--packaged", type=Path, default=Path("data"),
                    help="where train.pt / val.pt / test.pt / ood.pt live")
    ap.add_argument("--owner", default="retroranger04")
    ap.add_argument("--notes", default="Day-3 sweep v2 (post-pyvista-hang fix)")
    ap.add_argument("--staging", type=Path, default=Path("data/kaggle_staging"))
    args = ap.parse_args()

    train_slug = "lbracket-stress-train"
    ood_slug = "lbracket-stress-ood"

    # --- Train dataset ----------------------------------------------------
    train_stage = args.staging / train_slug
    stage_dir(
        src_samples=args.train_root / "samples",
        manifest_json=args.train_root / "manifest.json",
        pre_manifest_json=args.train_root / "sweep_pre_manifest.json",
        extra_files=[
            args.packaged / "train.pt",
            args.packaged / "val.pt",
            args.packaged / "test.pt",
            args.packaged / "split_manifest.json",
        ],
        staging=train_stage,
    )
    init_and_publish(
        staging=train_stage, owner=args.owner, slug=train_slug,
        title="L-Bracket Stress Surrogate \u2014 Training Set",
        version_notes=args.notes,
        first_push=not dataset_exists(args.owner, train_slug),
    )

    # --- OOD dataset ------------------------------------------------------
    ood_stage = args.staging / ood_slug
    stage_dir(
        src_samples=args.ood_root / "samples",
        manifest_json=args.ood_root / "manifest.json",
        pre_manifest_json=args.ood_root / "sweep_pre_manifest.json",
        extra_files=[args.packaged / "ood.pt"],
        staging=ood_stage,
    )
    init_and_publish(
        staging=ood_stage, owner=args.owner, slug=ood_slug,
        title="L-Bracket Stress Surrogate \u2014 OOD Test Set",
        version_notes=args.notes,
        first_push=not dataset_exists(args.owner, ood_slug),
    )

    print("\nPublished dataset URLs:")
    print(f"  https://www.kaggle.com/datasets/{args.owner}/{train_slug}")
    print(f"  https://www.kaggle.com/datasets/{args.owner}/{ood_slug}")


if __name__ == "__main__":
    main()
