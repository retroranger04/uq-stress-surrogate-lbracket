"""
Sanity-check a Day-3 sweep output (main or OOD).

Takes a directory of per-sample `.npz` shards + manifest.json and emits a
JSON report covering:
  - NaN/Inf scan per sample
  - Mesh validity (positive cell area, no isolated nodes)
  - Peak stress distribution summary
  - Parameter coverage summary (per-axis min/max/quantiles)

Usage:
    python scripts/sanity_check_sweep.py \\
        --root data/day3_main \\
        --report data/day3_main/sanity_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cell_areas(coords_t3: np.ndarray, elem_t3: np.ndarray) -> np.ndarray:
    a = coords_t3[elem_t3[:, 0]]
    b = coords_t3[elem_t3[:, 1]]
    c = coords_t3[elem_t3[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                         - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def check_sample(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    vm_l2 = z["vm_l2"]; vm_t3 = z["vm_t3"]
    coords_t3 = z["coords_t3"]; elem_t3 = z["elem_t3"]

    nan_l2 = int(np.isnan(vm_l2).sum() + np.isinf(vm_l2).sum())
    nan_t3 = int(np.isnan(vm_t3).sum() + np.isinf(vm_t3).sum())

    areas = _cell_areas(coords_t3, elem_t3)
    n_zero = int((areas <= 0).sum())

    used_nodes = np.unique(elem_t3.ravel())
    n_isolated = coords_t3.shape[0] - used_nodes.size

    issues: list[str] = []
    if nan_l2 or nan_t3:
        issues.append(f"NaN/Inf in vm (l2={nan_l2}, t3={nan_t3})")
    if n_zero:
        issues.append(f"{n_zero} zero/negative-area cells")
    if n_isolated:
        issues.append(f"{n_isolated} isolated nodes")

    return dict(
        path=path.name,
        params=[float(v) for v in z["params"]],
        peak_vm=float(z["peak_vm"]),
        n_nodes_t3=int(coords_t3.shape[0]),
        n_elem_t3=int(elem_t3.shape[0]),
        min_area=float(areas.min()) if areas.size else float("nan"),
        nan_l2=nan_l2, nan_t3=nan_t3,
        n_isolated=n_isolated,
        issues=issues,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing samples/ + manifest.json")
    ap.add_argument("--report", required=True, help="output JSON path")
    args = ap.parse_args()

    root = Path(args.root)
    samples = sorted((root / "samples").glob("*.npz"))
    if not samples:
        raise SystemExit(f"no samples found under {root/'samples'}")

    per_sample = [check_sample(p) for p in samples]
    n_with_issues = sum(1 for r in per_sample if r["issues"])

    peak_vals = np.array([r["peak_vm"] for r in per_sample])
    params = np.array([r["params"] for r in per_sample])

    report = dict(
        n_samples=len(per_sample),
        n_with_issues=n_with_issues,
        peak_vm_stats=dict(
            min=float(peak_vals.min()), max=float(peak_vals.max()),
            mean=float(peak_vals.mean()), std=float(peak_vals.std()),
            q25=float(np.quantile(peak_vals, 0.25)),
            q50=float(np.quantile(peak_vals, 0.50)),
            q75=float(np.quantile(peak_vals, 0.75)),
        ),
        param_stats={
            name: dict(
                min=float(col.min()), max=float(col.max()),
                mean=float(col.mean()), std=float(col.std()),
            )
            for name, col in zip(("R", "p", "W"), params.T)
        },
        per_sample=per_sample,
    )
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"report \u2192 {args.report}")
    print(f"n_samples={report['n_samples']}  "
          f"n_with_issues={report['n_with_issues']}")
    print(f"peak_vm range {report['peak_vm_stats']['min']:.2f}"
          f"\u2026{report['peak_vm_stats']['max']:.2f} MPa "
          f"(mean {report['peak_vm_stats']['mean']:.2f})")


if __name__ == "__main__":
    main()
