"""
Matplotlib-based visualizer for the L-bracket.

Kept deliberately minimal — this is not a production plotting module, just a
faithful 2D rendering of the geometry described in geometry.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, FancyArrow, Polygon

from .constants import A_VERT_LEN_MM, B_HORIZ_LEN_MM
from .geometry import (
    LBracketParams,
    build_outer_boundary,
    build_holes,
    check_validity,
)


def draw_bracket(ax, params: LBracketParams,
                 show_load: bool = True,
                 show_clamp: bool = True,
                 show_labels: bool = True,
                 title: str | None = None):
    """Render a single bracket onto the given Axes."""
    bnd = build_outer_boundary(params)
    holes = build_holes(params)
    P1, P2, P3, P4, P5, P6, P7 = bnd.vertices

    # --- Outer material polygon (with inside fillet approximated as a polyline
    #     so matplotlib fills the interior correctly). The arc is sampled with
    #     32 segments which is visually indistinguishable from smooth at this
    #     scale. Drawn AFTER we erase hole interiors by drawing white circles.

    arc_pts = _sample_inside_fillet(bnd, n=32)
    polygon_pts = [P1, P2, P3, P4] + arc_pts + [P5, P6, P7]
    outer = Polygon(polygon_pts, closed=True,
                    facecolor="#d9d9d9", edgecolor="black", linewidth=1.4,
                    zorder=1)
    ax.add_patch(outer)

    # Holes — drawn in white so they visually cut through the material fill.
    for h in holes:
        ax.add_patch(Circle(h.center, h.radius,
                            facecolor="white", edgecolor="black",
                            linewidth=1.0, zorder=2))
        if show_labels:
            ax.annotate(h.name, h.center,
                        xytext=(h.center[0] + 2.5, h.center[1] + 2.5),
                        fontsize=7, color="#444")

    # --- Clamped face (hatch on the back of vertical flange, x = 0). Drawn as
    #     a thin filled strip outside the material boundary so it reads as a
    #     fixed wall without modifying the geometry.
    if show_clamp:
        clamp_strip = Polygon([(-3.0, 0), (0.0, 0), (0.0, A_VERT_LEN_MM),
                               (-3.0, A_VERT_LEN_MM)],
                              closed=True,
                              facecolor="none", edgecolor="black",
                              hatch="////", linewidth=0.8, zorder=0)
        ax.add_patch(clamp_strip)
        if show_labels:
            ax.annotate("clamped", (-3.0, A_VERT_LEN_MM / 2),
                        xytext=(-14, A_VERT_LEN_MM / 2), fontsize=8,
                        rotation=90, ha="center", va="center", color="black")

    # --- Distributed load arrows on top of horizontal flange (y = W,
    #     x in [W+R, B]). Drawn pointing downward.
    if show_load:
        W = params.W
        R = params.R
        x_start = W + R
        x_end = B_HORIZ_LEN_MM
        # 8 arrows spaced along the loaded segment.
        n_arrows = 8
        for k in range(n_arrows):
            x = x_start + (x_end - x_start) * (k + 0.5) / n_arrows
            ax.add_patch(FancyArrow(x, W + 8, 0, -6,
                                    width=0.4, head_width=1.6, head_length=1.8,
                                    length_includes_head=True,
                                    facecolor="#c0392b", edgecolor="#c0392b",
                                    zorder=3))
        if show_labels:
            ax.annotate("distributed load w",
                        ((x_start + x_end) / 2, W + 9.5),
                        fontsize=8, color="#c0392b", ha="center")

    # --- Parameter callout box (top-right, outside the bracket).
    if show_labels:
        txt = f"R = {params.R:.1f} mm\np = {params.p:.1f} mm\nW = {params.W:.1f} mm"
        ax.text(B_HORIZ_LEN_MM * 0.55, A_VERT_LEN_MM * 0.85, txt,
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#888"))

    # Plot cosmetics.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-18, B_HORIZ_LEN_MM + 6)
    ax.set_ylim(-6, A_VERT_LEN_MM + 14)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linestyle=":")

    # Validity watermark if invalid.
    vr = check_validity(params)
    if not vr.ok:
        ax.text(0.5, 0.5, f"INVALID\n{vr.reason}",
                transform=ax.transAxes, fontsize=12, color="#c0392b",
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="#c0392b", pad=6))


def _sample_inside_fillet(bnd, n: int = 32) -> list:
    """Return n sampled (x,y) points along the inside fillet arc going from
    P4 (arc_start) to P5 (arc_end), inclusive of interior samples only.
    The returned list should be inserted between P4 and P5 in the traversal.
    """
    cx, cy = bnd.arc_center
    R = bnd.arc_radius
    # P4 is at angle -pi/2 (south of center), P5 is at angle pi (west of center).
    # Traverse CCW from -pi/2 up to pi would be the long way round; we want the
    # short 90° arc, which goes CW (angle decreasing from -pi/2 -> -pi).
    a0 = -math.pi / 2.0
    a1 = -math.pi
    pts = []
    for k in range(1, n):
        t = k / n
        a = a0 + (a1 - a0) * t
        pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
    return pts


# ---------------------------------------------------------------------------
# High-level renderers used by the live-discussion script
# ---------------------------------------------------------------------------

def render_single(params: LBracketParams, path, title: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_bracket(ax, params, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_param_sweep(param_name: str,
                       values: Iterable[float],
                       fixed: dict,
                       path,
                       suptitle: str | None = None):
    """Render a row of brackets varying one parameter; remaining params fixed.

    param_name: one of {"R", "p", "W"}
    values:     ordered list of values for that parameter (len 3 is standard:
                lower bound, midpoint, upper bound)
    fixed:      dict of the other two parameter names to their held-constant values
    """
    values = list(values)
    fig, axes = plt.subplots(1, len(values),
                             figsize=(6 * len(values), 6),
                             sharey=True)
    if len(values) == 1:
        axes = [axes]

    for ax, v in zip(axes, values):
        kwargs = dict(fixed)
        kwargs[param_name] = v
        params = LBracketParams(**kwargs)
        vr = check_validity(params)
        tag = "" if vr.ok else " (INVALID)"
        draw_bracket(ax, params, title=f"{param_name} = {v:.1f} mm{tag}")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
