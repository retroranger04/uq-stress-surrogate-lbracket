"""
Local WSL2 FEA sweep runner for the L-bracket surrogate (Day 3 recovery).

Replaces the old Kaggle notebook pipeline after four consecutive Kaggle failures
(see agent_log.md 2026-04-17 pivot entry). Runs a single-process sequential
sweep inside a WSL2 Ubuntu conda env with fenics-dolfinx=0.9.* + gmsh.

Contract:
    - Output one .npz per sample under <output>/samples/, plus a manifest.json.
    - .npz keys match src/models/dataset.py expectations:
        params, peak_vm, peak_xy, load_w_mpa,
        coords_l2, vm_l2,
        coords_t3, vm_t3, elem_t3,
        dof_{clamped,loaded,fillet,hole1,hole2}_{l2,t3},
        (optional) direction, kind.
    - SIGINT-safe: the flag flips after the current sample finishes cleanly,
      never mid-write. Partially-written samples are written atomically via
      tmpfile + rename.
    - Resume-capable: on startup, any sample whose final .npz exists is
      skipped. Kill the script at sample 500, restart, it picks up at 501.
    - RAM guardrail: if psutil reports > 80% system RAM before a sample, the
      runner waits until < 75% before proceeding.
    - Progress every 10 samples: completed / elapsed / eta / RAM%.

Expected invocation inside WSL2 Ubuntu (fenicsx conda env active):
    conda activate fenicsx
    cd /mnt/a/AntigravityWF/projects/uq-stress-surrogate-lbracket
    python scripts/run_sweep_local.py --mode main --output ~/lbracket-sweep/output/main --target 1000
    python scripts/run_sweep_local.py --mode ood  --output ~/lbracket-sweep/output/ood
    python scripts/run_sweep_local.py --mode validate --output ~/lbracket-sweep/output/validate

The runner reads FEA source from /mnt/a/... but writes all output + tmp mesh
to ~/lbracket-sweep/... to avoid the /mnt filesystem I/O penalty during the
compute-heavy phase.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.fea import config as cfg
from src.fea import ood_config as ood_cfg
from src.fea.constants import E_MPA, NU
from src.fea.geometry import LBracketParams, check_validity
from src.fea.mesh import write_msh

# Physical-group tags must match src/fea/geometry.py.
TAG_CLAMPED = 10
TAG_LOADED = 20
TAG_FILLET = 30
TAG_HOLE1 = 40
TAG_HOLE2 = 41
TAGS = {"clamped": TAG_CLAMPED, "loaded": TAG_LOADED,
        "fillet": TAG_FILLET, "hole1": TAG_HOLE1, "hole2": TAG_HOLE2}

RAM_HIGH_PCT = 80.0
RAM_RESUME_PCT = 75.0

# ---------------------------------------------------------------------------
# SIGINT handling
# ---------------------------------------------------------------------------

STOP_REQUESTED = False

def _sigint(_signum, _frame):
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        print("\n[SIGINT] graceful stop requested — will exit after current sample.", flush=True)
        STOP_REQUESTED = True
    else:
        print("\n[SIGINT] second interrupt — forcing exit.", flush=True)
        sys.exit(130)

signal.signal(signal.SIGINT, _sigint)


# ---------------------------------------------------------------------------
# RAM guardrail
# ---------------------------------------------------------------------------

def ram_pct() -> float:
    return psutil.virtual_memory().percent


def wait_for_ram_headroom():
    pct = ram_pct()
    if pct <= RAM_HIGH_PCT:
        return
    print(f"[GUARD] RAM {pct:.1f}% > {RAM_HIGH_PCT:.0f}% — pausing until < {RAM_RESUME_PCT:.0f}%", flush=True)
    while not STOP_REQUESTED and ram_pct() > RAM_RESUME_PCT:
        time.sleep(5)
    print(f"[GUARD] RAM back to {ram_pct():.1f}% — resuming", flush=True)


# ---------------------------------------------------------------------------
# Solver wrapper with T3 + DOF extraction
# ---------------------------------------------------------------------------

def solve_and_extract(params: LBracketParams,
                      load_w_mpa: float,
                      tmpdir: Path) -> dict:
    """Mesh, solve, and assemble the full per-sample .npz payload."""
    # Lazy dolfinx imports — the module-level import is kept out so the FEA
    # env only matters when we actually solve.
    from mpi4py import MPI
    import ufl
    from dolfinx import fem, io
    from dolfinx.fem.petsc import LinearProblem
    from petsc4py import PETSc

    msh_path = tmpdir / "bracket.msh"
    write_msh(params, msh_path,
              h_coarse=cfg.MESH_H_COARSE_MM,
              h_fine=cfg.MESH_H_FINE_MM,
              refine_dist=cfg.MESH_REFINE_DIST_MM)

    comm = MPI.COMM_WORLD
    mesh_obj, _cell_tags, facet_tags = io.gmshio.read_from_msh(
        str(msh_path), comm, gdim=2
    )

    V = fem.functionspace(mesh_obj, ("Lagrange", 2, (mesh_obj.geometry.dim,)))
    Vs2 = fem.functionspace(mesh_obj, ("Lagrange", 2))
    Vs1 = fem.functionspace(mesh_obj, ("Lagrange", 1))

    mu = E_MPA / (2.0 * (1.0 + NU))
    lam_ps = E_MPA * NU / (1.0 - NU * NU)

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lam_ps * ufl.tr(eps(u)) * ufl.Identity(2) + 2.0 * mu * eps(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    clamped_facets = facet_tags.find(TAG_CLAMPED)
    clamped_dofs = fem.locate_dofs_topological(V, 1, clamped_facets)
    zero = np.zeros(mesh_obj.geometry.dim, dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(zero, clamped_dofs, V)

    ds = ufl.Measure("ds", domain=mesh_obj, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=mesh_obj)
    traction = fem.Constant(mesh_obj, PETSc.ScalarType((0.0, -load_w_mpa)))

    a = ufl.inner(sigma(u), eps(v)) * dx
    L = ufl.inner(traction, v) * ds(TAG_LOADED)

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh = problem.solve()

    # --- von Mises field on Lagrange-2 -----------------------------------
    sg = sigma(uh)
    vm_expr = ufl.sqrt(
        sg[0, 0] ** 2 - sg[0, 0] * sg[1, 1] + sg[1, 1] ** 2 + 3.0 * sg[0, 1] ** 2
    )
    vm2 = fem.Function(Vs2)
    vm2.interpolate(fem.Expression(vm_expr, Vs2.element.interpolation_points()))
    vm_l2 = vm2.x.array.copy()
    coords_l2 = Vs2.tabulate_dof_coordinates()[:, :2].copy()

    peak_idx = int(np.argmax(vm_l2))
    peak_vm = float(vm_l2[peak_idx])
    peak_xy = (float(coords_l2[peak_idx, 0]), float(coords_l2[peak_idx, 1]))

    # --- T3 corner-node view --------------------------------------------
    vm1 = fem.Function(Vs1)
    vm1.interpolate(vm2)
    vm_t3 = vm1.x.array.copy()
    coords_t3 = Vs1.tabulate_dof_coordinates()[:, :2].copy()

    # Triangle connectivity in Vs1 DOF indices.
    dofmap1 = Vs1.dofmap
    num_cells = mesh_obj.topology.index_map(2).size_local
    elem_t3 = np.empty((num_cells, 3), dtype=np.int64)
    for c in range(num_cells):
        elem_t3[c] = dofmap1.cell_dofs(c)

    # --- Boundary DOF indices per physical tag, for both spaces ----------
    out: dict = {}
    for name, tag in TAGS.items():
        facets = facet_tags.find(tag)
        out[f"dof_{name}_l2"] = fem.locate_dofs_topological(Vs2, 1, facets).astype(np.int64)
        out[f"dof_{name}_t3"] = fem.locate_dofs_topological(Vs1, 1, facets).astype(np.int64)

    out.update({
        "params":     np.array([params.R, params.p, params.W], dtype=np.float64),
        "peak_vm":    np.float64(peak_vm),
        "peak_xy":    np.array(peak_xy, dtype=np.float64),
        "load_w_mpa": np.float64(load_w_mpa),
        "coords_l2":  coords_l2.astype(np.float64),
        "vm_l2":      vm_l2.astype(np.float64),
        "coords_t3":  coords_t3.astype(np.float64),
        "vm_t3":      vm_t3.astype(np.float64),
        "elem_t3":    elem_t3,
        "n_dofs_l2":  np.int64(vm_l2.size),
        "n_nodes_t3": np.int64(vm_t3.size),
        "n_cells":    np.int64(num_cells),
    })

    # Explicit cleanup so the next sample starts from a clean heap.
    del problem, uh, vm1, vm2, V, Vs1, Vs2, mesh_obj, facet_tags, clamped_dofs, bc
    del sg, vm_expr, traction, a, L, ds, dx, u, v

    return out


# ---------------------------------------------------------------------------
# Atomic .npz write
# ---------------------------------------------------------------------------

def write_npz_atomic(path: Path, payload: dict, extra_meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if extra_meta:
        # npz keys must be str; keep direction/kind as np arrays of 0-d str
        for k, v in extra_meta.items():
            payload[k] = np.array(v)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------

def lhs_main_samples(target_valid: int, oversample: int,
                     seed: int = 42, max_reseed: int = 10) -> list[LBracketParams]:
    samples: list[LBracketParams] = []
    seen: set[tuple] = set()
    for rs in range(max_reseed):
        sampler = qmc.LatinHypercube(d=3, seed=seed + rs)
        u = sampler.random(n=oversample)
        R = cfg.R_MIN_MM + u[:, 0] * (cfg.R_MAX_MM - cfg.R_MIN_MM)
        p = cfg.P_MIN_MM + u[:, 1] * (cfg.P_MAX_MM - cfg.P_MIN_MM)
        W = cfg.W_MIN_MM + u[:, 2] * (cfg.W_MAX_MM - cfg.W_MIN_MM)
        for Ri, pi, Wi in zip(R, p, W):
            key = (round(float(Ri), 6), round(float(pi), 6), round(float(Wi), 6))
            if key in seen:
                continue
            seen.add(key)
            prm = LBracketParams(R=float(Ri), p=float(pi), W=float(Wi))
            if check_validity(prm).ok:
                samples.append(prm)
                if len(samples) >= target_valid:
                    return samples
    return samples


def ood_single_param_samples(max_reseed: int = 10) -> list[tuple[LBracketParams, str, str]]:
    """Returns (params, kind='single', direction='R_low'|'R_high'|...) tuples."""
    out: list[tuple[LBracketParams, str, str]] = []
    trn = {
        "R": (cfg.R_MIN_MM, cfg.R_MAX_MM),
        "p": (cfg.P_MIN_MM, cfg.P_MAX_MM),
        "W": (cfg.W_MIN_MM, cfg.W_MAX_MM),
    }
    dir_ranges = {
        ("R", "low"):  ood_cfg.R_OOD_LOW_RANGE,  ("R", "high"): ood_cfg.R_OOD_HIGH_RANGE,
        ("p", "low"):  ood_cfg.P_OOD_LOW_RANGE,  ("p", "high"): ood_cfg.P_OOD_HIGH_RANGE,
        ("W", "low"):  ood_cfg.W_OOD_LOW_RANGE,  ("W", "high"): ood_cfg.W_OOD_HIGH_RANGE,
    }
    order = [("R", "low"), ("R", "high"), ("p", "low"), ("p", "high"), ("W", "low"), ("W", "high")]
    seed = ood_cfg.SEED_SINGLE
    for i, (param, side) in enumerate(order):
        target = ood_cfg.SAMPLES_PER_DIRECTION
        got: list[LBracketParams] = []
        for rs in range(max_reseed):
            s = seed + 7 * i + rs
            sampler = qmc.LatinHypercube(d=3, seed=s)
            u = sampler.random(n=target * 4)
            # column 0 -> OOD range for `param`, columns 1,2 -> training ranges for the other two params in fixed RpW order
            other_params = [x for x in ("R", "p", "W") if x != param]
            lo0, hi0 = dir_ranges[(param, side)]
            lo1, hi1 = trn[other_params[0]]
            lo2, hi2 = trn[other_params[1]]
            vals = {
                param: lo0 + u[:, 0] * (hi0 - lo0),
                other_params[0]: lo1 + u[:, 1] * (hi1 - lo1),
                other_params[1]: lo2 + u[:, 2] * (hi2 - lo2),
            }
            for j in range(len(u)):
                prm = LBracketParams(R=float(vals["R"][j]),
                                     p=float(vals["p"][j]),
                                     W=float(vals["W"][j]))
                if check_validity(prm).ok:
                    got.append(prm)
                    if len(got) >= target:
                        break
            if len(got) >= target:
                break
        for prm in got[:target]:
            out.append((prm, "single", f"{param}_{side}"))
    return out


def ood_corner_samples(max_reseed: int = 10) -> list[tuple[LBracketParams, str, str]]:
    """Corner OOD: >=2 params simultaneously outside training range."""
    trn = {
        "R": (cfg.R_MIN_MM, cfg.R_MAX_MM),
        "p": (cfg.P_MIN_MM, cfg.P_MAX_MM),
        "W": (cfg.W_MIN_MM, cfg.W_MAX_MM),
    }
    Rlo, Rhi = ood_cfg.R_FULL_OOD_BOX
    Plo, Phi = ood_cfg.P_FULL_OOD_BOX
    Wlo, Whi = ood_cfg.W_FULL_OOD_BOX
    target = ood_cfg.N_CORNER_OOD
    out: list[tuple[LBracketParams, str, str]] = []
    for rs in range(max_reseed):
        s = ood_cfg.SEED_CORNER + rs
        sampler = qmc.LatinHypercube(d=3, seed=s)
        u = sampler.random(n=ood_cfg.CORNER_OOD_OVERSAMPLE)
        R = Rlo + u[:, 0] * (Rhi - Rlo)
        P = Plo + u[:, 1] * (Phi - Plo)
        W = Wlo + u[:, 2] * (Whi - Wlo)
        for Ri, pi, Wi in zip(R, P, W):
            # Count how many params are out of training range.
            n_out = 0
            if not (trn["R"][0] <= Ri <= trn["R"][1]): n_out += 1
            if not (trn["p"][0] <= pi <= trn["p"][1]): n_out += 1
            if not (trn["W"][0] <= Wi <= trn["W"][1]): n_out += 1
            if n_out < 2:
                continue
            prm = LBracketParams(R=float(Ri), p=float(pi), W=float(Wi))
            if check_validity(prm).ok:
                out.append((prm, "corner", "corner"))
                if len(out) >= target:
                    return out
    return out


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def sample_npz_path(output_dir: Path, idx: int) -> Path:
    return output_dir / "samples" / f"sample_{idx:05d}.npz"


def run_sweep(output_dir: Path,
              sample_list: list,
              load_w_mpa: float,
              label: str):
    """sample_list: list of LBracketParams OR list of (LBracketParams, kind, direction)."""
    output_dir = Path(output_dir).expanduser()
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="lbracket_mesh_", dir=str(output_dir)))
    try:
        total = len(sample_list)
        t0 = time.monotonic()
        n_done = 0
        n_new = 0
        n_fail = 0
        solve_times: list[float] = []
        last_progress = t0
        print(f"[{label}] sweep start: {total} samples → {output_dir}", flush=True)
        for i, entry in enumerate(sample_list):
            if STOP_REQUESTED:
                break
            out_path = sample_npz_path(output_dir, i)
            if out_path.exists():
                n_done += 1
                continue

            if isinstance(entry, tuple):
                params, kind, direction = entry
                meta = {"direction": direction, "kind": kind}
            else:
                params = entry
                meta = {"direction": "", "kind": "main"}

            wait_for_ram_headroom()
            if STOP_REQUESTED:
                break

            ram_before = ram_pct()
            t_s = time.monotonic()
            try:
                payload = solve_and_extract(params, load_w_mpa, tmp_root)
                write_npz_atomic(out_path, payload, meta)
                dt = time.monotonic() - t_s
                solve_times.append(dt)
                n_done += 1
                n_new += 1
                del payload
            except Exception as e:
                n_fail += 1
                print(f"[{label}] sample {i} FAILED (R={params.R:.3f} p={params.p:.3f} W={params.W:.3f}): {type(e).__name__}: {e}", flush=True)
            finally:
                gc.collect()
                # Clean per-sample mesh files
                for f in tmp_root.iterdir():
                    try: f.unlink()
                    except Exception: pass

            # Progress every 10 new samples or every 60s
            now = time.monotonic()
            if n_new and (n_new % 10 == 0 or (now - last_progress) > 60):
                elapsed = now - t0
                avg = (sum(solve_times[-20:]) / max(1, len(solve_times[-20:])))
                remaining = total - (i + 1)
                eta_s = remaining * avg
                print(f"[{label}] {i+1}/{total} done={n_done} new={n_new} fail={n_fail} "
                      f"elapsed={elapsed/60:.1f}m avg={avg:.1f}s eta={eta_s/60:.1f}m "
                      f"RAM={ram_pct():.1f}% (before={ram_before:.1f}%)", flush=True)
                last_progress = now

        # Write/update manifest
        manifest = {
            "label": label,
            "total_scheduled": total,
            "n_done": n_done,
            "n_new_this_run": n_new,
            "n_fail_this_run": n_fail,
            "load_w_mpa": load_w_mpa,
            "mesh": {
                "h_coarse": cfg.MESH_H_COARSE_MM,
                "h_fine": cfg.MESH_H_FINE_MM,
                "refine_dist": cfg.MESH_REFINE_DIST_MM,
            },
            "stopped_early": bool(STOP_REQUESTED),
            "timestamp_end": time.time(),
            "solve_time_stats": {
                "mean_s": float(np.mean(solve_times)) if solve_times else 0.0,
                "p50_s": float(np.median(solve_times)) if solve_times else 0.0,
                "p95_s": float(np.percentile(solve_times, 95)) if solve_times else 0.0,
                "n": len(solve_times),
            },
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[{label}] sweep end: done={n_done}/{total} new={n_new} fail={n_fail} "
              f"stopped_early={STOP_REQUESTED}", flush=True)
        return manifest
    finally:
        try:
            for f in tmp_root.iterdir():
                try: f.unlink()
                except Exception: pass
            tmp_root.rmdir()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_validate(args):
    """Single nominal sample for Day 2 cross-check. Writes solve timings."""
    p = LBracketParams(R=cfg.NOMINAL["R"], p=cfg.NOMINAL["p"], W=cfg.NOMINAL["W"])
    out = Path(args.output).expanduser()
    (out / "samples").mkdir(parents=True, exist_ok=True)
    print(f"[validate] nominal R={p.R} p={p.p} W={p.W}, w=1.0 MPa (Day 2 reference)", flush=True)
    ram0 = ram_pct()
    tmp = Path(tempfile.mkdtemp(prefix="lbracket_validate_", dir=str(out)))
    try:
        t0 = time.monotonic()
        payload = solve_and_extract(p, 1.0, tmp)
        dt = time.monotonic() - t0
        ram1 = ram_pct()
        write_npz_atomic(out / "samples" / "nominal_w1MPa.npz", payload,
                         {"direction": "", "kind": "validate"})
        del payload
        gc.collect()
        ram2 = ram_pct()
        blob = np.load(out / "samples" / "nominal_w1MPa.npz")
        print(f"[validate] solve_time={dt:.2f}s peak_vm={float(blob['peak_vm']):.3f} MPa "
              f"peak_xy=({float(blob['peak_xy'][0]):.2f},{float(blob['peak_xy'][1]):.2f}) "
              f"n_dofs={int(blob['n_dofs_l2'])}", flush=True)
        print(f"[validate] RAM before={ram0:.1f}% after_solve={ram1:.1f}% after_cleanup={ram2:.1f}%", flush=True)
        # Day 2 reference: 45.643 MPa at w=1 MPa, peak at (22.78, 19.60) mm, level 3.
        ref = 45.643
        err = abs(float(blob['peak_vm']) - ref) / ref * 100.0
        print(f"[validate] vs Day 2 (45.643 MPa): Δ={err:.2f}% (tolerance 2%)", flush=True)
        return {"peak_vm": float(blob['peak_vm']),
                "ref": ref,
                "pct_err": err,
                "solve_time_s": dt,
                "ram_before": ram0,
                "ram_after_solve": ram1,
                "ram_after_cleanup": ram2,
                "n_dofs_l2": int(blob['n_dofs_l2']),
                "n_nodes_t3": int(blob['n_nodes_t3']),
                "n_cells": int(blob['n_cells'])}
    finally:
        for f in tmp.iterdir():
            try: f.unlink()
            except Exception: pass
        tmp.rmdir()


def mode_main(args):
    n_target = args.target if args.target else cfg.TARGET_VALID
    oversample = args.oversample if args.oversample else cfg.LHS_OVERSAMPLE
    print(f"[main] generating LHS samples: target={n_target}, oversample={oversample}, seed=42", flush=True)
    samples = lhs_main_samples(n_target, oversample, seed=42)
    print(f"[main] generated {len(samples)} valid samples after filter", flush=True)
    if args.limit:
        samples = samples[:args.limit]
        print(f"[main] limited to first {len(samples)} samples per --limit", flush=True)
    return run_sweep(Path(args.output), samples, cfg.LOAD_W_MPA, label="main")


def mode_ood(args):
    single = ood_single_param_samples()
    corner = ood_corner_samples()
    print(f"[ood] single-param samples: {len(single)}, corner samples: {len(corner)}", flush=True)
    samples = single + corner
    return run_sweep(Path(args.output), samples, cfg.LOAD_W_MPA, label="ood")


def mode_smoke(args):
    """5-sample corners + center smoke test."""
    cs = [
        LBracketParams(R=cfg.R_MIN_MM, p=cfg.P_MIN_MM, W=cfg.W_MIN_MM),
        LBracketParams(R=cfg.R_MAX_MM, p=cfg.P_MAX_MM, W=cfg.W_MAX_MM),
        LBracketParams(R=cfg.R_MIN_MM, p=cfg.P_MAX_MM, W=cfg.W_MIN_MM),
        LBracketParams(R=cfg.R_MAX_MM, p=cfg.P_MIN_MM, W=cfg.W_MAX_MM),
        LBracketParams(R=cfg.NOMINAL["R"], p=cfg.NOMINAL["p"], W=cfg.NOMINAL["W"]),
    ]
    # Drop any that violate validity (corner W=24+R=10 is borderline).
    cs = [c for c in cs if check_validity(c).ok]
    print(f"[smoke] {len(cs)} valid corner+center samples", flush=True)
    return run_sweep(Path(args.output), cs, cfg.LOAD_W_MPA, label="smoke")


def mode_lhs(args):
    """N-sample LHS checkpoint (e.g. 50)."""
    n = args.target or 50
    print(f"[lhs] generating {n} LHS samples, seed=42", flush=True)
    samples = lhs_main_samples(n, n * 2, seed=42)
    return run_sweep(Path(args.output), samples, cfg.LOAD_W_MPA, label=f"lhs{n}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["validate", "smoke", "lhs", "main", "ood"])
    ap.add_argument("--output", required=True, help="Output dir (WSL2 native FS recommended)")
    ap.add_argument("--target", type=int, default=None, help="Valid-sample target (main/lhs)")
    ap.add_argument("--oversample", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Cap number of samples run")
    args = ap.parse_args()

    if args.mode == "validate":
        return mode_validate(args)
    if args.mode == "smoke":
        return mode_smoke(args)
    if args.mode == "lhs":
        return mode_lhs(args)
    if args.mode == "main":
        return mode_main(args)
    if args.mode == "ood":
        return mode_ood(args)


if __name__ == "__main__":
    main()
