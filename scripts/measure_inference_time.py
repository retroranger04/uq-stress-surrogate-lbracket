"""
End-to-end inference wall-clock benchmark for the 5-member Deep Ensemble.

Measures the deployment-configuration forward-pass latency for one full
ensemble prediction (all 5 members run sequentially, no cross-member
batching) on a single representative test sample, plus the single-member
latency for comparison. Reports full timing statistics in milliseconds.

This is a read-only measurement: it loads the frozen Phase-1 ensemble
members + shared normalization stats and the held-out test bundle, and
writes a small JSON summary. It does NOT retrain, refit, or modify any
model or data artifact.

Run (from project root, venv active):
    python scripts/measure_inference_time.py
or explicitly with the venv interpreter:
    ./venv/Scripts/python.exe scripts/measure_inference_time.py

Outputs
-------
runs/ensemble/inference_timing.json   (summary; stdout mirrors it)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.runtime import (  # noqa: E402
    RunCfg, apply_stats_inplace, build_model, load_bundle, load_stats,
)

# The five frozen Deep-Ensemble members (CLAUDE.md: seeds {0,101,202,303,404}).
SEEDS = [0, 101, 202, 303, 404]
WARMUP = 20
ITERS = 100


def load_member(member_dir: Path, device: torch.device):
    """Replicates scripts/phase1_eval.py::_load_member — build MeshGNN from
    the checkpoint's own cfg, load weights, set eval()."""
    blob = torch.load(str(member_dir / "best.pt"), weights_only=False,
                      map_location=device)
    cfg = RunCfg(**blob["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def summarize_ms(times_ms: list[float]) -> dict:
    return {
        "min": min(times_ms),
        "median": statistics.median(times_ms),
        "mean": statistics.fmean(times_ms),
        "max": max(times_ms),
        "std": statistics.pstdev(times_ms),
    }


def fmt_row(label: str, s: dict) -> str:
    return (f"{label:<16} "
            f"min={s['min']:8.3f}  median={s['median']:8.3f}  "
            f"mean={s['mean']:8.3f}  max={s['max']:8.3f}  std={s['std']:7.3f}")


def main() -> None:
    warnings: list[str] = []

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available — refusing to benchmark on CPU.")
        sys.exit(1)
    device = torch.device("cuda")

    # ---- GPU sanity check ----
    dev_name = torch.cuda.get_device_name(0)
    print(f"[gpu] device               : {dev_name}")
    print(f"[gpu] torch                : {torch.__version__}")
    print(f"[gpu] python               : {sys.version.split()[0]}")

    base = ROOT / "runs" / "ensemble"

    # ---- Load the 5 ensemble members ----
    models = []
    for seed in SEEDS:
        mdir = base / f"seed{seed}"
        if not (mdir / "best.pt").exists():
            print(f"[FATAL] missing checkpoint: {mdir / 'best.pt'}")
            sys.exit(1)
        m = load_member(mdir, device)
        # confirm every parameter actually landed on CUDA
        p_dev = next(m.parameters()).device
        if p_dev.type != "cuda":
            warnings.append(f"member seed{seed} parameters on {p_dev}, not cuda")
        models.append(m)
    print(f"[load] {len(models)} ensemble members on CUDA, all .eval()")

    # ---- Normalization stats ----
    stats = load_stats(base / "stats.pt")

    # ---- Test bundle: list of Data objects ----
    test = load_bundle(ROOT / "data" / "test.pt")
    print(f"[data] test bundle: {len(test)} Data objects (list)")

    # ---- Choose a representative sample: median by node count ----
    node_counts = [int(d.x.shape[0]) for d in test]
    order = sorted(range(len(test)), key=lambda i: node_counts[i])
    sample_idx = order[len(order) // 2]  # median-difficulty by mesh size
    sample = test[sample_idx]
    n_nodes = int(sample.x.shape[0])
    n_edges = int(sample.edge_index.shape[1])
    print(f"[pick] sample idx {sample_idx}: {n_nodes} nodes, {n_edges} edges "
          f"(median of node-count distribution "
          f"[min {min(node_counts)}, max {max(node_counts)}])")

    # ---- Normalize the chosen sample (x + edge_attr in-place) and move to GPU ----
    apply_stats_inplace([sample], stats)
    x = sample.x.to(device)
    edge_index = sample.edge_index.to(device)
    edge_attr = sample.edge_attr.to(device)

    # ---- Timing helpers -------------------------------------------------
    @torch.no_grad()
    def run_ensemble():
        outs = []
        for m in models:
            outs.append(m(x, edge_index, edge_attr))
        return outs

    @torch.no_grad()
    def run_single():
        return models[0](x, edge_index, edge_attr)

    def time_loop(fn, iters: int) -> list[float]:
        times_ms: list[float] = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        return times_ms

    # ---- Ensemble: warmup (untimed) then measure ----
    print(f"[warm] {WARMUP} untimed ensemble forward passes ...")
    with torch.no_grad():
        for _ in range(WARMUP):
            run_ensemble()
            torch.cuda.synchronize()

    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[gpu] memory_allocated     : {mem_gb:.3f} GB "
          f"(post-warmup; >0 confirms tensors + weights on GPU)")

    print(f"[time] {ITERS} timed ensemble forward passes ...")
    ens_ms = time_loop(run_ensemble, ITERS)

    # ---- Single member: warmup then measure ----
    print(f"[warm] {WARMUP} untimed single-member forward passes ...")
    with torch.no_grad():
        for _ in range(WARMUP):
            run_single()
            torch.cuda.synchronize()
    print(f"[time] {ITERS} timed single-member forward passes ...")
    single_ms = time_loop(run_single, ITERS)

    ens_stats = summarize_ms(ens_ms)
    single_stats = summarize_ms(single_ms)
    ratio = ens_stats["median"] / single_stats["median"]

    # No autocast / AMP was used anywhere above -> full fp32.
    param_dtype = next(models[0].parameters()).dtype
    if param_dtype != torch.float32:
        warnings.append(f"member dtype is {param_dtype}, not float32")

    # ---- Report ----
    print("\n===== RESULTS =====")
    print(f"sample_index          : {sample_idx}")
    print(f"node_count            : {n_nodes}")
    print(f"edge_count            : {n_edges}")
    print(fmt_row("ensemble(5) ms", ens_stats))
    print(fmt_row("single-member ms", single_stats))
    print(f"ensemble/single ratio : {ratio:.3f}x  (expected ~5x)")
    print(f"precision             : {param_dtype} (no autocast / AMP)")
    print(f"gpu_memory_allocated  : {mem_gb:.3f} GB")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("WARNINGS: none")

    summary = {
        "device": dev_name,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "sample_index": sample_idx,
        "node_count": n_nodes,
        "edge_count": n_edges,
        "warmup_iters": WARMUP,
        "timed_iters": ITERS,
        "precision": str(param_dtype),
        "gpu_memory_allocated_gb": mem_gb,
        "ensemble_ms": ens_stats,
        "single_member_ms": single_stats,
        "ensemble_over_single_ratio": ratio,
        "warnings": warnings,
    }
    out_path = base / "inference_timing.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
