"""
Produce paper-ready figures from the Day-3 sweep outputs.

Figures written to paper/figures/:
    fig_geometry_schematic.pdf  \u2014 annotated bracket diagram (Methods)
    fig_lhs_coverage.pdf        \u2014 3-panel train+OOD parameter coverage
    fig_stress_hist.pdf         \u2014 peak stress distribution histogram
    fig_example_fields.pdf      \u2014 3-panel von Mises field visualization

Usage:
    python scripts/make_paper_figures.py \\
        --train-root data/day3_main \\
        --ood-root data/day3_ood \\
        --out paper/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fea.constants import (
    A_VERT_LEN_MM, B_HORIZ_LEN_MM, HOLE_RADIUS_MM, HOLE1_OFFSET_FROM_TOP_MM,
)
from src.fea.config import (
    R_MIN_MM, R_MAX_MM, P_MIN_MM, P_MAX_MM, W_MIN_MM, W_MAX_MM,
)
from src.fea.ood_config import (
    R_OOD_LOW_RANGE, R_OOD_HIGH_RANGE,
    P_OOD_LOW_RANGE, P_OOD_HIGH_RANGE,
    W_OOD_LOW_RANGE, W_OOD_HIGH_RANGE,
)
from src.fea.geometry import LBracketParams, build_outer_boundary, build_holes


# --- fig_geometry_schematic ------------------------------------------------

def fig_geometry_schematic(out: Path) -> None:
    # Nominal parameters to draw labeled; parametric labels R, p, W annotated.
    params = LBracketParams(R=7.0, p=55.0, W=18.0)
    bnd = build_outer_boundary(params)
    holes = build_holes(params)
    A = A_VERT_LEN_MM; B = B_HORIZ_LEN_MM
    W = params.W; R = params.R; p = params.p

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    # Outer polyline with an arc from P4 to P5 (inside fillet).
    P1, P2, P3, P4, P5, P6, P7 = bnd.vertices
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 'k-', lw=1.3)
    ax.plot([P2[0], P3[0]], [P2[1], P3[1]], 'k-', lw=1.3)
    ax.plot([P3[0], P4[0]], [P3[1], P4[1]], 'k-', lw=1.3)
    th = np.linspace(-np.pi / 2, np.pi, 60)
    AC = np.array(bnd.arc_center)
    arc = AC[:, None] + R * np.stack([np.cos(th), np.sin(th)], axis=0)
    ax.plot(arc[0], arc[1], 'k-', lw=1.3)
    ax.plot([P5[0], P6[0]], [P5[1], P6[1]], 'k-', lw=1.3)
    ax.plot([P6[0], P7[0]], [P6[1], P7[1]], 'k-', lw=1.3)
    ax.plot([P7[0], P1[0]], [P7[1], P1[1]], 'k-', lw=1.3)

    for h in holes:
        ax.add_patch(Circle(h.center, h.radius, ec='k', fc='white', lw=1.1))

    # Clamped face hatching.
    clamped = Rectangle((-3.5, 0), 3.5, A, facecolor='none',
                         edgecolor='0.35', hatch='////', lw=0.0)
    ax.add_patch(clamped)
    ax.plot([0, 0], [0, A], 'k-', lw=1.8)

    # Distributed load arrows on the top face.
    for x in np.linspace(W + R + 2, B - 2, 8):
        ax.add_patch(FancyArrowPatch((x, W + 6), (x, W + 0.5),
                                      arrowstyle='-|>', color='C3',
                                      mutation_scale=10, lw=1.1))
    ax.text((W + R + B) / 2, W + 8.5, 'w', color='C3',
            ha='center', va='bottom', fontsize=12)

    # Parametric labels R, p, W.
    ax.annotate(f'R', xy=AC + np.array([R / np.sqrt(2), R / np.sqrt(2)]) * 0.9,
                 xytext=AC + np.array([15, 16]),
                 arrowprops=dict(arrowstyle='-|>', color='C0'), color='C0',
                 fontsize=11, ha='left')
    ax.annotate(f'p', xy=(p, W / 2), xytext=(p, -10),
                 arrowprops=dict(arrowstyle='-|>', color='C0'), color='C0',
                 fontsize=11, ha='center')
    ax.annotate('W', xy=(B / 2, W), xytext=(B / 2, W + 14),
                 arrowprops=dict(arrowstyle='-|>', color='C0'), color='C0',
                 fontsize=11, ha='center')
    ax.annotate('Hole 1', xy=holes[0].center, xytext=(-28, 50),
                 arrowprops=dict(arrowstyle='-'), fontsize=9)
    ax.annotate('Hole 2', xy=holes[1].center, xytext=(holes[1].center[0] + 4, -6),
                 arrowprops=dict(arrowstyle='-'), fontsize=9)

    ax.set_xlim(-10, B + 8); ax.set_ylim(-14, A + 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Parametric L-bracket (clamped back face, distributed load $w$)')
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


# --- fig_lhs_coverage ------------------------------------------------------

def _gather_params(root: Path, kind_label: str = "") -> tuple[np.ndarray, list[str]]:
    samples = sorted((root / "samples").glob("*.npz"))
    rows: list[np.ndarray] = []
    dirs: list[str] = []
    for p in samples:
        z = np.load(p, allow_pickle=False)
        rows.append(z["params"])
        d = str(z["direction"]) if "direction" in z.files else kind_label
        dirs.append(d)
    if not rows:
        return np.empty((0, 3), dtype=np.float32), []
    return np.stack(rows, axis=0), dirs


def fig_lhs_coverage(out: Path, train_root: Path, ood_root: Path) -> None:
    tr, _ = _gather_params(train_root, "train")
    od, od_dirs = _gather_params(ood_root, "ood")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    pairs = [(0, 1, 'R [mm]', 'p [mm]', (R_MIN_MM, R_MAX_MM), (P_MIN_MM, P_MAX_MM)),
             (0, 2, 'R [mm]', 'W [mm]', (R_MIN_MM, R_MAX_MM), (W_MIN_MM, W_MAX_MM)),
             (1, 2, 'p [mm]', 'W [mm]', (P_MIN_MM, P_MAX_MM), (W_MIN_MM, W_MAX_MM))]
    for ax, (a, b, xl, yl, xb, yb) in zip(axes, pairs):
        if tr.size:
            ax.scatter(tr[:, a], tr[:, b], s=6, color='C0',
                        alpha=0.45, label='train')
        if od.size:
            is_corner = np.array([d == 'corner' for d in od_dirs])
            ax.scatter(od[~is_corner, a], od[~is_corner, b], s=16,
                        marker='x', color='C3', label='OOD single')
            ax.scatter(od[is_corner, a], od[is_corner, b], s=22,
                        marker='*', color='C2', label='OOD corner')
        ax.axvspan(xb[0], xb[1], color='C0', alpha=0.08)
        ax.axhspan(yb[0], yb[1], color='C0', alpha=0.08)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    axes[-1].legend(loc='upper right', fontsize=8)
    fig.suptitle('LHS parameter coverage \u2014 train (blue) vs. OOD (red/green)',
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


# --- fig_stress_hist -------------------------------------------------------

def fig_stress_hist(out: Path, train_root: Path, ood_root: Path) -> None:
    def peaks(root: Path) -> np.ndarray:
        return np.array([float(np.load(p, allow_pickle=False)["peak_vm"])
                         for p in sorted((root / "samples").glob("*.npz"))])
    tr = peaks(train_root)
    od = peaks(ood_root)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bins = np.linspace(min(tr.min() if tr.size else 0,
                            od.min() if od.size else 0) - 5,
                        max(tr.max() if tr.size else 0,
                            od.max() if od.size else 0) + 5, 40)
    if tr.size:
        ax.hist(tr, bins=bins, alpha=0.65, color='C0', label=f'train (n={tr.size})')
    if od.size:
        ax.hist(od, bins=bins, alpha=0.65, color='C3', label=f'OOD (n={od.size})')
    ax.axvline(205.0, ls='--', color='k', lw=0.8)
    ax.text(205.2, ax.get_ylim()[1] * 0.9, r'$\sigma_y$', fontsize=9)
    ax.set_xlabel('Peak von Mises stress [MPa]')
    ax.set_ylabel('Samples')
    ax.set_title('Peak stress distribution across the sweep')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


# --- fig_example_fields ---------------------------------------------------

def fig_example_fields(out: Path, train_root: Path) -> None:
    samples = sorted((train_root / "samples").glob("*.npz"))
    if not samples:
        raise SystemExit("no training samples found for fig_example_fields")
    peaks = np.array([float(np.load(p, allow_pickle=False)["peak_vm"])
                       for p in samples])
    order = np.argsort(peaks)
    picks = [order[len(order) // 20],       # low-peak sample
             order[len(order) // 2],        # median
             order[-len(order) // 20 - 1]]  # high-peak sample

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, idx in zip(axes, picks):
        z = np.load(samples[idx], allow_pickle=False)
        coords = z["coords_t3"]; vm = z["vm_t3"]; tri_idx = z["elem_t3"]
        t = Triangulation(coords[:, 0], coords[:, 1], tri_idx)
        tpc = ax.tripcolor(t, vm, shading='gouraud', cmap='viridis')
        R_, p_, W_ = z["params"]
        ax.set_title(f'R={R_:.2f}, p={p_:.1f}, W={W_:.1f}\n'
                      f'peak = {float(z["peak_vm"]):.1f} MPa',
                      fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tpc, ax=ax, shrink=0.85, label='$\\sigma_{vM}$ [MPa]')
    fig.suptitle('Representative stress fields (low / median / high peak)',
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


# --- driver ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, default=None)
    ap.add_argument("--ood-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--figures", nargs="*", default=None,
                    help="subset: schematic | coverage | hist | fields")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    want = set(args.figures) if args.figures else {"schematic", "coverage", "hist", "fields"}

    if "schematic" in want:
        fig_geometry_schematic(args.out / "fig_geometry_schematic.pdf")
        print("wrote fig_geometry_schematic.pdf")
    if "coverage" in want and args.train_root and args.ood_root:
        fig_lhs_coverage(args.out / "fig_lhs_coverage.pdf",
                          args.train_root, args.ood_root)
        print("wrote fig_lhs_coverage.pdf")
    if "hist" in want and args.train_root and args.ood_root:
        fig_stress_hist(args.out / "fig_stress_hist.pdf",
                         args.train_root, args.ood_root)
        print("wrote fig_stress_hist.pdf")
    if "fields" in want and args.train_root:
        fig_example_fields(args.out / "fig_example_fields.pdf", args.train_root)
        print("wrote fig_example_fields.pdf")


if __name__ == "__main__":
    main()
