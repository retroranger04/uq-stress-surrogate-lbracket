"""
Parametric L-bracket geometry.

Coordinate convention (used everywhere downstream — solver, mesh, visualizer):
    Origin: back-bottom outer corner of the L (the clamped face is x = 0).
    +x: along the horizontal flange toward its free tip.
    +y: along the vertical flange toward its top.
    z:  out of plane (thickness direction, not modeled in 2D plane stress).

Outer boundary traversed counterclockwise (material on the left):
    P1 (0, 0)     back-bottom outer corner
      -> bottom of horizontal flange
    P2 (B, 0)     free-tip bottom
      -> free-tip end face
    P3 (B, W)     free-tip top
      -> top of horizontal flange (LOADED FACE, load applied downward)
    P4 (W+R, W)   fillet start (tangent on horizontal flange top)
      -> inside fillet arc (center at (W+R, W+R), radius R)
    P5 (W, W+R)   fillet end (tangent on vertical flange inside face)
      -> vertical flange inside face
    P6 (W, A)     vertical flange top-inside corner
      -> top of vertical flange
    P7 (0, A)     back-top outer corner
      -> back face of vertical flange (CLAMPED FACE)
      -> close back to P1

Inner boundaries (holes, interior loops — traversed CW so material stays on the left):
    Hole 1 at (W/2, A - hole1_offset)  — centered on vertical flange
    Hole 2 at (p, W/2)                 — centered on horizontal flange

The parametric variables are R, p, W; everything else is fixed in constants.py.

All modeling decisions encoded here appear in the accompanying paper's Methods
section; the working principle is "no silent assumptions".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

from .constants import (
    A_VERT_LEN_MM,
    B_HORIZ_LEN_MM,
    HOLE_RADIUS_MM,
    HOLE1_OFFSET_FROM_TOP_MM,
    CLEARANCE_MM,
)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LBracketParams:
    """Variable design parameters for one L-bracket sample.

    All units millimetres. Fixed dimensions live in constants.py and are pulled
    in implicitly by the builder functions below.
    """
    R: float   # inside fillet radius
    p: float   # Hole-2 x-position along horizontal flange centerline
    W: float   # in-plane flange width (symmetric on both arms)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validity checks — reject infeasible parameter combinations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidityResult:
    ok: bool
    reason: Optional[str] = None


def check_validity(params: LBracketParams,
                   clearance: float = CLEARANCE_MM) -> ValidityResult:
    """Check geometric feasibility of (R, p, W).

    Returns ValidityResult(ok=False, reason=...) for the first violated
    constraint, so the caller gets a readable diagnostic. The checks are
    deliberately redundant where they can catch nearby failure modes — cheap
    to run, and the sweep will LHS-sample thousands of candidates.
    """
    A = A_VERT_LEN_MM
    B = B_HORIZ_LEN_MM
    r_h = HOLE_RADIUS_MM
    m = clearance
    y_hole1 = A - HOLE1_OFFSET_FROM_TOP_MM  # absolute y of Hole 1 center

    R = params.R
    p = params.p
    W = params.W

    # --- Basic positivity and sanity ---------------------------------------

    if W <= 0 or R < 0 or p <= 0:
        return ValidityResult(False, f"non-positive dimensions: W={W}, R={R}, p={p}")

    # Fillet must fit along both flanges. For the equal-arm L (A = B = 80 mm)
    # and the ranges we care about (R <~ 20, W <~ 50) this is trivially true,
    # but we check it so a future widening of ranges can't silently break.
    if W + R > min(A, B):
        return ValidityResult(False,
            f"fillet runs off the flange: W+R={W+R} > min(A,B)={min(A,B)}")

    # --- Flange width must accommodate both holes transversely -------------

    # Hole 1 is at x = W/2 on the vertical flange; Hole 2 is at y = W/2 on the
    # horizontal flange. Both need clearance r_h + m to the nearer free edge
    # (which is W/2 away). Binding constraint: W/2 >= r_h + m => W >= 2(r_h+m).
    min_W = 2.0 * (r_h + m)
    if W < min_W:
        return ValidityResult(False,
            f"flange too narrow for holes: W={W} < {min_W} (=2*(r_h+clearance))")

    # --- Hole 1 must sit in the clear vertical-flange region above fillet --

    # Hole 1 y-center is fixed at A - 40 = 40 mm. The fillet top tangent is at
    # y = W + R. To keep Hole 1 perimeter clear of the fillet we require
    #     y_hole1 - r_h - m >= W + R
    # i.e. the hole's lowest edge (with margin) sits above the fillet top.
    if y_hole1 - r_h - m < W + R:
        return ValidityResult(False,
            f"Hole 1 conflicts with fillet: y_hole1-r_h-m={y_hole1-r_h-m:.2f} "
            f"< W+R={W+R:.2f}")

    # Hole 1 must also clear the top edge of the vertical flange.
    if y_hole1 + r_h + m > A:
        return ValidityResult(False,
            f"Hole 1 too close to top of vertical flange: "
            f"y_hole1+r_h+m={y_hole1+r_h+m:.2f} > A={A}")

    # --- Hole 2 must sit on the horizontal-flange arm, clear of fillet -----

    # Simplest sufficient condition: hole fully to the right of the fillet's
    # rightmost tangent (W+R). Slightly conservative vs. computing the true
    # arc distance, but avoids numerical fuss and the bracket's mechanics are
    # not interesting for a hole that lives inside the corner region anyway.
    if p - r_h - m < W + R:
        return ValidityResult(False,
            f"Hole 2 too close to fillet: p-r_h-m={p-r_h-m:.2f} "
            f"< W+R={W+R:.2f}")

    if p + r_h + m > B:
        return ValidityResult(False,
            f"Hole 2 too close to free tip: p+r_h+m={p+r_h+m:.2f} > B={B}")

    # --- Fillet non-degeneracy ---------------------------------------------

    # R = 0 is a legal "sharp corner" case in principle, but our mesher treats
    # R > 0 as an arc primitive — require a small positive floor to keep the
    # .geo file well-formed.
    if R < 0.5:
        return ValidityResult(False, f"fillet radius too small: R={R} < 0.5")

    return ValidityResult(True, None)


# ---------------------------------------------------------------------------
# Boundary primitives — same representation used by visualizer and .geo emitter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OuterBoundary:
    """Describes the outer L outline as a polyline with one arc segment.

    vertices: list of (x, y) corner points P1..P7 in CCW order.
    arc: (start_pt, end_pt, center, radius) for the inside fillet, inserted
         between P4 and P5 of vertices.
    """
    vertices: list              # [(x, y), ...] length 7, CCW
    arc_start: tuple            # P4 = (W+R, W)
    arc_end: tuple              # P5 = (W, W+R)
    arc_center: tuple           # (W+R, W+R)
    arc_radius: float           # R


def build_outer_boundary(params: LBracketParams) -> OuterBoundary:
    A = A_VERT_LEN_MM
    B = B_HORIZ_LEN_MM
    W = params.W
    R = params.R

    P1 = (0.0, 0.0)
    P2 = (B, 0.0)
    P3 = (B, W)
    P4 = (W + R, W)
    P5 = (W, W + R)
    P6 = (W, A)
    P7 = (0.0, A)

    return OuterBoundary(
        vertices=[P1, P2, P3, P4, P5, P6, P7],
        arc_start=P4,
        arc_end=P5,
        arc_center=(W + R, W + R),
        arc_radius=R,
    )


@dataclass(frozen=True)
class Hole:
    center: tuple        # (x, y)
    radius: float        # mm
    name: str            # "hole1" or "hole2"


def build_holes(params: LBracketParams) -> list:
    y1 = A_VERT_LEN_MM - HOLE1_OFFSET_FROM_TOP_MM
    return [
        Hole(center=(params.W / 2.0, y1), radius=HOLE_RADIUS_MM, name="hole1"),
        Hole(center=(params.p, params.W / 2.0), radius=HOLE_RADIUS_MM, name="hole2"),
    ]


# ---------------------------------------------------------------------------
# Feature locations (used by the mesher for sizing fields and by the solver
# for tagging boundary conditions). Centralized here so everyone agrees.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NamedBoundaries:
    """Endpoints of named linear segments, in CCW order around the outer loop.

    Used to (1) apply the clamped BC to the back face, (2) apply the distributed
    load to the top-of-horizontal-flange segment, (3) tag the fillet arc and
    hole perimeters for local mesh refinement.
    """
    clamped_face: tuple          # (P7, P1) — back face, x=0, y in [0, A]
    loaded_face: tuple           # (P3, P4) — top of horizontal flange
    # (The fillet arc and hole circles are first-class primitives elsewhere.)


def named_boundaries(params: LBracketParams) -> NamedBoundaries:
    bnd = build_outer_boundary(params)
    P1, _P2, P3, P4, _P5, _P6, P7 = bnd.vertices
    return NamedBoundaries(
        clamped_face=(P7, P1),
        loaded_face=(P3, P4),
    )


# ---------------------------------------------------------------------------
# Gmsh .geo emission — text output. Portable, debuggable, independent of
# whether the machine has the gmsh Python API installed.
# ---------------------------------------------------------------------------

_GEO_HEADER = """// Auto-generated by src/fea/geometry.py for the UQ stress-surrogate project.
// Do not edit by hand — regenerate via build_geo() with updated params.
//
// Coordinate convention and physical groups match src/fea/geometry.py.
// Physical tags:
//   Surface 1 -> bracket body (2D domain)
//   Curve 10  -> clamped face (back of vertical flange)
//   Curve 20  -> loaded face (top of horizontal flange)
//   Curve 30  -> inside fillet arc
//   Curve 40  -> hole 1 perimeter
//   Curve 41  -> hole 2 perimeter
"""


def build_geo(params: LBracketParams,
              h_coarse: float = 4.0,
              h_fine: float = 0.5,
              refine_dist: float = 6.0) -> str:
    """Return a Gmsh .geo file as a string.

    Arguments:
        h_coarse: characteristic element size far from stress-concentrators.
        h_fine:   characteristic element size at the fillet arc and hole edges.
        refine_dist: distance (mm) over which the size field transitions from
                     h_fine to h_coarse away from a refined feature.

    The default values are reasonable starting points; the mesh convergence
    study tunes (h_fine, refine_dist) and the chosen values become the standard
    for the full sweep.
    """
    vr = check_validity(params)
    if not vr.ok:
        raise ValueError(f"invalid geometry: {vr.reason}")

    bnd = build_outer_boundary(params)
    holes = build_holes(params)
    P1, P2, P3, P4, P5, P6, P7 = bnd.vertices
    AC = bnd.arc_center  # (W+R, W+R)

    lines = [_GEO_HEADER]
    lines.append(f"// params: R={params.R}  p={params.p}  W={params.W}")
    lines.append(f"h_coarse = {h_coarse};")
    lines.append(f"h_fine   = {h_fine};")
    lines.append("")

    # --- Points -----------------------------------------------------------
    # Seven outer vertices + one fillet center + two hole centers + 4 points
    # per hole circle (Gmsh needs >= 3 points on a Circle arc, we use 4).
    def pt(i, x, y, lc):
        return f"Point({i}) = {{{x:.6f}, {y:.6f}, 0.0, {lc}}};"

    lines.append("// --- outer boundary points ---")
    for i, (x, y) in enumerate([P1, P2, P3, P4, P5, P6, P7], start=1):
        # Use h_fine on fillet tangent points (P4, P5) and h_coarse elsewhere.
        lc = "h_fine" if i in (4, 5) else "h_coarse"
        lines.append(pt(i, x, y, lc))

    lines.append("")
    lines.append("// --- fillet arc center ---")
    lines.append(pt(8, AC[0], AC[1], "h_coarse"))  # center has no mesh role

    lines.append("")
    lines.append("// --- hole points (center + 4 cardinal points per hole) ---")
    hole_center_tags = {}
    hole_cardinal_tags = {}
    tag = 9
    for hi, h in enumerate(holes):
        cx, cy = h.center
        lines.append(pt(tag, cx, cy, "h_fine"))
        hole_center_tags[h.name] = tag
        tag += 1
        cardinals = [
            (cx + h.radius, cy),
            (cx,            cy + h.radius),
            (cx - h.radius, cy),
            (cx,            cy - h.radius),
        ]
        ctags = []
        for cx_, cy_ in cardinals:
            lines.append(pt(tag, cx_, cy_, "h_fine"))
            ctags.append(tag)
            tag += 1
        hole_cardinal_tags[h.name] = ctags

    # --- Curves -----------------------------------------------------------
    lines.append("")
    lines.append("// --- outer boundary curves ---")
    # CCW order of straight segments + one arc between P4 and P5.
    # Tagging scheme: loaded=20, clamped=10, fillet=30.
    # Straight segments numbered 101..106 except the arc (103 replaced by Circle).
    lines.append("Line(101) = {1, 2};")  # bottom of horizontal flange
    lines.append("Line(102) = {2, 3};")  # free-tip end face
    lines.append("// top of horizontal flange — LOADED FACE")
    lines.append("Line(103) = {3, 4};")
    lines.append("// inside fillet arc — the concentration site")
    lines.append("Circle(104) = {4, 8, 5};")  # start, center, end
    lines.append("Line(105) = {5, 6};")  # inside face of vertical flange
    lines.append("Line(106) = {6, 7};")  # top of vertical flange
    lines.append("// back face — CLAMPED")
    lines.append("Line(107) = {7, 1};")

    lines.append("")
    lines.append("// --- hole circles (4 quarter arcs each, CCW) ---")
    hole_loop_tags = {}
    curve_tag = 200
    for h in holes:
        ctr = hole_center_tags[h.name]
        c = hole_cardinal_tags[h.name]  # E, N, W, S
        arcs = []
        # E->N, N->W, W->S, S->E — CCW quarter arcs
        pairs = [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]
        for s, e in pairs:
            lines.append(f"Circle({curve_tag}) = {{{s}, {ctr}, {e}}};")
            arcs.append(curve_tag)
            curve_tag += 1
        hole_loop_tags[h.name] = arcs

    # --- Loops + plane surface -------------------------------------------
    lines.append("")
    lines.append("// --- curve loops ---")
    lines.append("Curve Loop(1) = {101, 102, 103, 104, 105, 106, 107};")
    # For interior holes, Gmsh expects the hole loop with a sign that makes
    # the material domain well-oriented. We list the 4 CCW arcs; Gmsh auto-
    # orients against the outer loop when both are given to Plane Surface.
    hole_loop_ids = []
    lid = 2
    for h in holes:
        arcs = hole_loop_tags[h.name]
        lines.append(f"Curve Loop({lid}) = {{{', '.join(str(a) for a in arcs)}}};")
        hole_loop_ids.append(lid)
        lid += 1

    surface_loops = ["1"] + [str(i) for i in hole_loop_ids]
    lines.append(f"Plane Surface(1) = {{{', '.join(surface_loops)}}};")

    # --- Physical groups (tags referenced from the solver) ---------------
    lines.append("")
    lines.append("// --- physical groups (consumed by the FEniCSx solver) ---")
    lines.append('Physical Surface("bracket", 1) = {1};')
    lines.append('Physical Curve("clamped", 10) = {107};')
    lines.append('Physical Curve("loaded",  20) = {103};')
    lines.append('Physical Curve("fillet",  30) = {104};')
    hole_curves = hole_loop_tags["hole1"]
    lines.append(f'Physical Curve("hole1",   40) = {{{", ".join(str(a) for a in hole_curves)}}};')
    hole_curves = hole_loop_tags["hole2"]
    lines.append(f'Physical Curve("hole2",   41) = {{{", ".join(str(a) for a in hole_curves)}}};')

    # --- Distance-based mesh size field ----------------------------------
    # Refine near the fillet arc and both hole perimeters; coarsen outward.
    lines.append("")
    lines.append("// --- size field: refine near fillet and hole edges ---")
    lines.append("Field[1] = Distance;")
    lines.append("Field[1].CurvesList = {104, " +
                 ", ".join(str(a) for a in hole_loop_tags["hole1"] + hole_loop_tags["hole2"]) +
                 "};")
    lines.append("Field[1].Sampling = 200;")
    lines.append("Field[2] = Threshold;")
    lines.append("Field[2].InField = 1;")
    lines.append("Field[2].SizeMin = h_fine;")
    lines.append("Field[2].SizeMax = h_coarse;")
    lines.append("Field[2].DistMin = 0.0;")
    lines.append(f"Field[2].DistMax = {refine_dist};")
    lines.append("Background Field = 2;")
    lines.append("Mesh.MeshSizeExtendFromBoundary = 0;")
    lines.append("Mesh.MeshSizeFromPoints = 0;")
    lines.append("Mesh.MeshSizeFromCurvature = 0;")

    lines.append("")
    lines.append("// --- T6 (Lagrange order-2 triangle) elements ---")
    lines.append("Mesh.ElementOrder = 2;")
    lines.append("Mesh.Algorithm = 6;  // Frontal-Delaunay — robust on curved features")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Convenience — one-shot save to disk
# ---------------------------------------------------------------------------

def save_geo(params: LBracketParams, path, **kwargs) -> str:
    """Write a .geo to `path` and return its content."""
    content = build_geo(params, **kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return content
