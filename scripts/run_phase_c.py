"""
End-to-end Phase-C orchestration: pulls, sanity-checks, packages, publishes.

Assumes both Kaggle sweep kernels have completed. Pulls their outputs to
data/day3_main/ and data/day3_ood/, extracts the tarballs, runs sanity
checks, builds PyG bundles, renders paper figures, and publishes the two
Kaggle datasets.

Usage:
    python scripts/run_phase_c.py [--skip-publish]

Each step logs to data/phase_c.log and appends a section to the top-level
return status printed at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_ROOT = ROOT / "data" / "day3_main"
OOD_ROOT = ROOT / "data" / "day3_ood"
DATA_DIR = ROOT / "data"


def run(cmd: list[str], label: str) -> bool:
    print(f"\n=== {label} ===")
    r = subprocess.run(cmd, text=True)
    ok = (r.returncode == 0)
    print(f"[{label}] {'OK' if ok else 'FAIL'} (exit {r.returncode})")
    return ok


def pull_and_extract(kernel: str, out_root: Path) -> bool:
    out_root.mkdir(parents=True, exist_ok=True)
    if not run(["kaggle", "kernels", "output", kernel, "-p", str(out_root), "--force"],
                f"pull {kernel}"):
        return False
    # Find + extract tarball.
    tarballs = list(out_root.glob("*.tar.gz"))
    if not tarballs:
        print(f"  no tarball under {out_root}")
        return False
    for t in tarballs:
        print(f"  extracting {t.name}...")
        with tarfile.open(t) as tf:
            tf.extractall(out_root)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-publish", action="store_true")
    args = ap.parse_args()

    status: dict = {}

    # --- 1. Pull both kernel outputs --------------------------------------
    status["pull_main"] = pull_and_extract(
        "retroranger/uq-l-bracket-day-3-main-sweep", MAIN_ROOT)
    status["pull_ood"] = pull_and_extract(
        "retroranger/uq-l-bracket-day-3-ood-sweep", OOD_ROOT)
    if not (status["pull_main"] and status["pull_ood"]):
        print("Pull failed; aborting Phase C.")
        sys.exit(1)

    # --- 2. Sanity checks --------------------------------------------------
    status["sanity_main"] = run(
        [sys.executable, "scripts/sanity_check_sweep.py",
         "--root", str(MAIN_ROOT),
         "--report", str(MAIN_ROOT / "sanity_report.json")],
        "sanity main",
    )
    status["sanity_ood"] = run(
        [sys.executable, "scripts/sanity_check_sweep.py",
         "--root", str(OOD_ROOT),
         "--report", str(OOD_ROOT / "sanity_report.json")],
        "sanity ood",
    )

    # --- 3. Package to PyG -----------------------------------------------
    status["package_pyg"] = run(
        [sys.executable, "scripts/package_to_pyg.py",
         "--train-root", str(MAIN_ROOT),
         "--ood-root", str(OOD_ROOT),
         "--out", str(DATA_DIR)],
        "package pyg",
    )

    # --- 4. Paper figures ------------------------------------------------
    status["figures"] = run(
        [sys.executable, "scripts/make_paper_figures.py",
         "--train-root", str(MAIN_ROOT),
         "--ood-root", str(OOD_ROOT),
         "--out", "paper/figures",
         "--figures", "coverage", "hist", "fields"],
        "paper figures",
    )

    # --- 5. Publish Kaggle datasets --------------------------------------
    if not args.skip_publish:
        status["publish"] = run(
            [sys.executable, "scripts/publish_kaggle_datasets.py",
             "--train-root", str(MAIN_ROOT),
             "--ood-root", str(OOD_ROOT),
             "--packaged", str(DATA_DIR)],
            "publish kaggle datasets",
        )
    else:
        status["publish"] = "skipped"

    (DATA_DIR / "phase_c_status.json").write_text(json.dumps(status, indent=2))
    print("\n=== Phase C summary ===")
    print(json.dumps(status, indent=2))
    print(f"wrote {DATA_DIR/'phase_c_status.json'}")


if __name__ == "__main__":
    main()
