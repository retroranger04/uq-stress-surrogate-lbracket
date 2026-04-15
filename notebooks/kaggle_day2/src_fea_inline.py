"""
Inlined copy of src/fea/ for the self-contained Kaggle notebook.
GENERATED — do not edit by hand. Regenerate with
    python scripts/assemble_kaggle_notebook.py
"""

from __future__ import annotations


# ======================================================================
# BEGIN constants.py
# ======================================================================
"""
Locked (non-parametric) constants for the L-bracket study.

Every number here is a *deliberate* modeling choice. See inline comments for the
justification; the same choices are mirrored in agent_log.md (Day 2 entry) and
will be stated explicitly in paper/main.tex Methods.

Units: millimetres, newtons, megapascals. Consistent set: mm-N-MPa
    stress [MPa] = force [N] / area [mm^2]
    E [MPa] comes from GPa * 1000
    sigma_y [MPa] is native
"""

# --- Geometry (fixed) -------------------------------------------------------

# Equal-arm symmetric L. Both flange lengths = 80 mm.
A_VERT_LEN_MM = 80.0
B_HORIZ_LEN_MM = 80.0

# Out-of-plane thickness. Ratio t/A = 1/10 justifies plane-stress formulation.
T_OUT_OF_PLANE_MM = 8.0

# Through-hole diameter (M8 bolt clearance). Both holes share this diameter.
HOLE_DIAMETER_MM = 8.0
HOLE_RADIUS_MM = HOLE_DIAMETER_MM / 2.0

# Hole 1 is centered on the vertical flange, 40 mm from the top edge.
# Position is fixed across all samples; only Hole 2's p-position varies.
HOLE1_OFFSET_FROM_TOP_MM = 40.0

# --- Material (AISI 304 annealed, ASM Handbook Vol 1 / ASTM A240) ----------

E_GPA = 193.0                     # Young's modulus [GPa] (asm_handbook_vol1)
E_MPA = E_GPA * 1000.0            # [MPa] — native unit for solver

NU = 0.29                         # Poisson ratio [-] (asm_handbook_vol1)

SIGMA_Y_MPA = 205.0               # min yield [MPa] (astm_a240, ASTM A240/A240M-22)

RHO_KG_M3 = 8000.0                # density [kg/m^3] (asm_handbook_vol1).
# Self-weight is omitted from the loading model: applied w exceeds gravitational
# body force by ~3 orders of magnitude, so its contribution to peak stress is
# below discretization error from mesh refinement. Density is recorded for
# provenance only — it is not consumed by the current solver.

# --- Convention: clearance margin used in validity checks ------------------

# Minimum edge clearance between any hole perimeter and any adjacent boundary
# (free edge, fillet arc, or back face). Prevents degenerate mesh regions and
# mechanically-unrealistic geometry. 2 mm is a conservative engineering choice
# for an 8 mm hole in a thin-flanged stainless bracket.
CLEARANCE_MM = 2.0


# ======================================================================
# BEGIN geometry.py
# ======================================================================
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

All modeling decisions encoded here are logged in agent_log.md (Day 2) and will
appear in paper Methods; the working principle is "no silent assumptions".
"""


import math
from dataclasses import dataclass, asdict
from typing import Optional

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


# ======================================================================
# BEGIN analytical.py
# ======================================================================
"""
Analytical cross-check: closed-form net-section stress on an un-notched L.

Geometry: the full L outline with NO holes and the fillet replaced by a sharp
inside corner (i.e., R -> 0 limiting case). This geometry is not used in the
sweep; it exists solely to produce an independent analytical prediction we can
compare against the FEA pipeline to confirm correctness before we trust it.

Modelling idealization
----------------------
Treat the horizontal flange as a cantilever beam, clamped at the inside corner
(x = W), loaded by a uniformly distributed traction w [MPa] applied downward
on the top face along x in [W, B]. In 2D plane-stress the problem is
per-unit-depth; the cantilever beam properties are then:

    span         L_h = B - W
    load per len q   = w   [N/mm^2 * 1mm_depth = N/mm]
    root moment  M/t = w * L_h^2 / 2                [N*mm / mm_depth]
    section mod  S/t = W^2 / 6                       [mm^2]
    root bending stress   sigma = (M/t) / (S/t) = 3 * w * L_h^2 / W^2

The peak tensile/compressive bending stress occurs at the top (compressive)
and bottom (tensile) fibres of the horizontal flange at the root section.
The corresponding von Mises equals |sigma| (uniaxial state).

This ignores the inside-corner stress singularity (a sharp L has infinite
stress at the re-entrant corner — that's why the real bracket has a fillet).
Compared against FEA on the un-notched L, we therefore sample stress at a
cross-section DISPLACED from the corner, e.g. x = W + probe_offset, to avoid
the singularity region. The predicted value at that section is:

    M(x) / t = w * (B - x)^2 / 2
    sigma_b(x) = 3 * w * (B - x)^2 / W^2
"""


from dataclasses import dataclass

@dataclass(frozen=True)
class BendingPrediction:
    x: float               # section location (mm from back face)
    sigma_bending: float   # peak fibre bending stress at section [MPa]
    load_w_mpa: float      # distributed load applied [MPa]
    W_mm: float            # flange width [mm]
    L_remaining: float     # (B - x), cantilever tail length at section [mm]


def bending_stress_at_section(x_mm: float,
                              W_mm: float,
                              load_w_mpa: float) -> BendingPrediction:
    """Closed-form peak bending stress at a cross-section of the horizontal
    flange on the un-notched L, treated as a cantilever beam.

    x_mm : location of the section, measured from the back face. Must lie in
           (W_mm, B_HORIZ_LEN_MM).
    """
    if not (W_mm < x_mm < B_HORIZ_LEN_MM):
        raise ValueError(f"x={x_mm} outside cantilever span (W, B)")

    L_rem = B_HORIZ_LEN_MM - x_mm
    sigma = 3.0 * load_w_mpa * L_rem**2 / W_mm**2
    return BendingPrediction(x=x_mm, sigma_bending=sigma,
                             load_w_mpa=load_w_mpa, W_mm=W_mm,
                             L_remaining=L_rem)


def root_bending_stress(W_mm: float, load_w_mpa: float) -> float:
    """Convenience: peak bending stress at the root section x = W (the inside
    corner). This is the theoretical maximum for the un-notched cantilever,
    but note it coincides with the sharp-corner singularity — use it only as
    an upper-bound reference, not for pointwise FEA comparison.
    """
    L_h = B_HORIZ_LEN_MM - W_mm
    return 3.0 * load_w_mpa * L_h**2 / W_mm**2


# ---------------------------------------------------------------------------
# Simplified .geo for the un-notched reference geometry.
# ---------------------------------------------------------------------------
#
# The simplified L has the same outer outline as the full bracket but:
#   - no holes
#   - fillet arc replaced by a right-angle inside corner at (W, W)
#
# Physical-group tags match the full geometry where possible so the solver
# script can run unchanged on either mesh:
#   Surface 1  -> bracket body
#   Curve 10   -> clamped face (back of vertical flange)
#   Curve 20   -> loaded face (top of horizontal flange)
# No "fillet" or "hole" tags on the simplified mesh.

_SIMPLE_GEO_HEADER = """// Auto-generated un-notched L reference geometry.
// Generated by src/fea/analytical.build_simplified_geo().
// Purpose: Day-2 analytical cross-check of the FEA pipeline.
"""


def build_simplified_geo(W_mm: float,
                         h_coarse: float = 3.0,
                         h_fine: float = 0.5,
                         refine_dist: float = 8.0) -> str:
    """Return a Gmsh .geo string for the un-notched L with width W_mm.

    Refinement is applied near the sharp inside corner (which is the hot-spot
    the FEA will over-predict due to the singularity — expected behaviour).
    """
    A = A_VERT_LEN_MM
    B = B_HORIZ_LEN_MM
    W = W_mm

    P = [
        (0.0, 0.0),        # P1
        (B,   0.0),        # P2
        (B,   W),          # P3
        (W,   W),          # P4  <- sharp inside corner (replaces fillet)
        (W,   A),          # P5
        (0.0, A),          # P6
    ]

    lines = [_SIMPLE_GEO_HEADER,
             f"// simplified L: W={W}",
             f"h_coarse = {h_coarse};",
             f"h_fine   = {h_fine};",
             ""]

    for i, (x, y) in enumerate(P, start=1):
        # Put h_fine on the sharp inside corner (P4).
        lc = "h_fine" if i == 4 else "h_coarse"
        lines.append(f"Point({i}) = {{{x:.6f}, {y:.6f}, 0.0, {lc}}};")

    lines += [
        "",
        "Line(101) = {1, 2};",      # bottom
        "Line(102) = {2, 3};",      # free-tip end
        "Line(103) = {3, 4};",      # LOADED top of horizontal flange
        "Line(104) = {4, 5};",      # inside face of vertical flange
        "Line(105) = {5, 6};",      # top of vertical flange
        "Line(106) = {6, 1};",      # CLAMPED back face
        "",
        "Curve Loop(1) = {101, 102, 103, 104, 105, 106};",
        "Plane Surface(1) = {1};",
        "",
        'Physical Surface("bracket", 1) = {1};',
        'Physical Curve("clamped", 10) = {106};',
        'Physical Curve("loaded",  20) = {103};',
        "",
        "// refine near the sharp inside corner (expected to singularity)",
        "Field[1] = Distance;",
        "Field[1].PointsList = {4};",
        "Field[1].Sampling = 100;",
        "Field[2] = Threshold;",
        "Field[2].InField = 1;",
        "Field[2].SizeMin = h_fine;",
        "Field[2].SizeMax = h_coarse;",
        "Field[2].DistMin = 0.0;",
        f"Field[2].DistMax = {refine_dist};",
        "Background Field = 2;",
        "Mesh.MeshSizeExtendFromBoundary = 0;",
        "Mesh.MeshSizeFromPoints = 0;",
        "Mesh.MeshSizeFromCurvature = 0;",
        "Mesh.ElementOrder = 2;",
        "Mesh.Algorithm = 6;",
        "",
    ]
    return "\n".join(lines) + "\n"


# ======================================================================
# BEGIN mesh.py
# ======================================================================
"""
Gmsh wrapper: takes an LBracketParams and writes a .msh for FEniCSx.

Runs via the `gmsh` Python API when available; falls back to the `gmsh` CLI
otherwise. Both paths consume the .geo file produced by geometry.build_geo.

This module is designed to run on Kaggle CPU (where FEniCSx + gmsh are both
installed). It is import-safe locally — the gmsh import is lazy.
"""


import shutil
import subprocess
import tempfile
from pathlib import Path

def write_msh(params: LBracketParams,
              out_path,
              h_coarse: float = 4.0,
              h_fine: float = 0.5,
              refine_dist: float = 6.0,
              keep_geo: bool = False) -> Path:
    """Generate a .msh file for the given parameters.

    Returns the path to the written .msh. If `keep_geo` is True, also retains
    the intermediate .geo next to the .msh (useful when debugging the mesher).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    geo_content = build_geo(params,
                            h_coarse=h_coarse,
                            h_fine=h_fine,
                            refine_dist=refine_dist)

    # Write the .geo near the .msh so relative paths work for either backend.
    geo_path = out_path.with_suffix(".geo")
    geo_path.write_text(geo_content, encoding="utf-8")

    # Prefer the Python API — it reports errors via exceptions which surface
    # cleaner tracebacks through Kaggle logs than the CLI's stderr does.
    try:
        import gmsh  # noqa: WPS433 — lazy import, optional locally
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.merge(str(geo_path))
            gmsh.model.mesh.generate(2)
            # Force order 2 in case the .geo directive is overridden by the API.
            gmsh.model.mesh.setOrder(2)
            gmsh.write(str(out_path))
        finally:
            gmsh.finalize()
    except ImportError:
        # CLI fallback. `gmsh -2 <geo> -o <msh> -order 2` builds a 2D mesh
        # with Lagrange order 2 elements.
        gmsh_bin = shutil.which("gmsh")
        if gmsh_bin is None:
            raise RuntimeError(
                "gmsh is not available: neither the Python `gmsh` package nor "
                "the `gmsh` CLI binary could be found on PATH."
            )
        cmd = [gmsh_bin, "-2", str(geo_path),
               "-o", str(out_path), "-order", "2", "-v", "2"]
        subprocess.run(cmd, check=True)

    if not keep_geo:
        geo_path.unlink(missing_ok=True)

    return out_path


# ======================================================================
# BEGIN solver.py
# ======================================================================
"""
FEniCSx plane-stress linear-elasticity solver for the L-bracket.

Consumes a .msh produced by src/fea/mesh.py and returns the displacement and
von Mises stress fields along with the peak stress value.

Every modeling choice here is deliberate; each is annotated inline. The same
justifications are logged in agent_log.md (Day 2) and will appear in
paper/main.tex Methods.

Run environment: Kaggle CPU notebook with FEniCSx (dolfinx) installed. This
module imports dolfinx lazily so the rest of the package remains importable
on a Windows-native dev machine where FEniCSx is not available.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Physical-group tags must stay in sync with build_geo() in geometry.py.
TAG_DOMAIN = 1
TAG_CLAMPED = 10
TAG_LOADED = 20
TAG_FILLET = 30
TAG_HOLE1 = 40
TAG_HOLE2 = 41


@dataclass
class FEAResult:
    params: LBracketParams
    load_w_mpa: float              # applied distributed load [MPa]
    peak_vm_mpa: float             # peak von Mises stress over all DOFs
    peak_location_xy: tuple        # (x, y) of the peak vm DOF, mm
    n_dofs: int                    # problem size (scalar field count)
    h_fine: float                  # mesh refinement at fillet
    h_coarse: float                # far-field mesh size
    vm_field: Optional[np.ndarray] = None   # nodal von Mises [MPa]
    coords: Optional[np.ndarray] = None     # nodal (x,y) coords [mm]


def solve_lbracket(msh_path,
                   params: LBracketParams,
                   load_w_mpa: float,
                   h_fine: float,
                   h_coarse: float,
                   return_fields: bool = True) -> FEAResult:
    """Solve 2D plane-stress linear elasticity on the given L-bracket mesh.

    Parameters
    ----------
    msh_path : path-like
        .msh file (quadratic triangles, physical groups per geometry.build_geo).
    params : LBracketParams
        Design parameters — recorded on the result for provenance.
    load_w_mpa : float
        Magnitude of the downward distributed traction on the top of the
        horizontal flange. Units MPa (= N/mm^2). Applied as -y traction; its
        resultant force per unit depth equals load_w_mpa * (B - (W+R)).
    h_fine, h_coarse : float
        Recorded on the result for the convergence-study sweep.
    return_fields : bool
        If True, attach the nodal von Mises field + coordinates (useful for
        plotting & GNN training later). If False, only the summary stats are
        returned (cheap).

    Notes on the modeling choices
    -----------------------------
    - Plane stress: justified by t/A = 1/10; out-of-plane stress << in-plane.
    - Linear elastic isotropic material, AISI 304: E = 193 GPa, nu = 0.29.
    - Quadratic triangular elements for the displacement (Lagrange order 2).
      Stresses recovered by differentiation are piecewise linear within each
      element and projected to a nodal field for plotting and peak-finding.
    - Self-weight omitted (see constants.py).
    - BC: homogeneous Dirichlet (u = 0) on the back face of the vertical
      flange (physical tag 10). Everything else is a free or loaded surface.
    - Load: constant traction -w*e_y on the top of the horizontal flange
      (physical tag 20). All other surfaces (holes, fillet, free tip, bottom
      of horizontal flange, top of vertical flange) are traction-free.
    """
    # Imports kept local so the rest of the package stays import-safe on
    # machines without dolfinx. On Kaggle these imports succeed.
    from mpi4py import MPI
    import dolfinx
    import ufl
    from dolfinx import fem, mesh as dmesh, io
    from dolfinx.fem.petsc import LinearProblem
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD

    # --- Read the mesh ----------------------------------------------------
    mesh_obj, cell_tags, facet_tags = io.gmshio.read_from_msh(
        str(msh_path), comm, gdim=2
    )

    # --- Function space: vector Lagrange order 2 on triangles -------------
    V = fem.functionspace(mesh_obj, ("Lagrange", 2, (mesh_obj.geometry.dim,)))

    # --- Plane-stress constitutive law -----------------------------------
    # sigma = lam_ps * tr(eps) * I + 2 * mu * eps
    # with the plane-stress-effective Lamé first parameter
    #     lam_ps = E*nu / (1 - nu^2)
    # and mu = E / (2*(1+nu)).
    E = E_MPA
    nu = NU
    mu = E / (2.0 * (1.0 + nu))
    lam_ps = E * nu / (1.0 - nu * nu)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        eps = epsilon(u)
        return lam_ps * ufl.tr(eps) * ufl.Identity(2) + 2.0 * mu * eps

    # --- Trial / test / BC -----------------------------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Clamped facets are those tagged 10 on the mesh boundary.
    clamped_facets = facet_tags.find(TAG_CLAMPED)
    clamped_dofs = fem.locate_dofs_topological(V, 1, clamped_facets)
    zero = np.zeros(mesh_obj.geometry.dim, dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(zero, clamped_dofs, V)

    # Measure restricted to the loaded facets.
    ds = ufl.Measure("ds", domain=mesh_obj, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=mesh_obj)

    # Traction: -y direction, magnitude load_w_mpa (N/mm^2 = MPa).
    traction = fem.Constant(mesh_obj,
                            PETSc.ScalarType((0.0, -load_w_mpa)))

    # --- Weak form -------------------------------------------------------
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.inner(traction, v) * ds(TAG_LOADED)

    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options={
                                "ksp_type": "preonly",
                                "pc_type": "lu",
                                "pc_factor_mat_solver_type": "mumps",
                            })
    uh = problem.solve()

    # --- Recover von Mises stress at mesh nodes --------------------------
    # Project sigma onto a DG-1 tensor space then compute sqrt(3/2 * s:s)
    # where s = sigma - 1/3 tr(sigma) I (deviatoric). For plane stress we
    # must include the through-thickness component sigma_zz = 0 in the
    # deviator to recover the correct 3D von Mises from 2D fields:
    #     svm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
    sig = sigma(uh)
    s_xx = sig[0, 0]
    s_yy = sig[1, 1]
    s_xy = sig[0, 1]
    vm_expr = ufl.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)

    # Project onto a Lagrange order-2 scalar space so nodal values coincide
    # with the displacement mesh (keeps field sizes compatible downstream).
    Vs = fem.functionspace(mesh_obj, ("Lagrange", 2))
    vm = fem.Function(Vs)
    vm_expr_fn = fem.Expression(vm_expr, Vs.element.interpolation_points())
    vm.interpolate(vm_expr_fn)

    vm_array = vm.x.array
    coords = Vs.tabulate_dof_coordinates()[:, :2]

    peak_idx = int(np.argmax(vm_array))
    peak_vm = float(vm_array[peak_idx])
    peak_xy = (float(coords[peak_idx, 0]), float(coords[peak_idx, 1]))

    result = FEAResult(
        params=params,
        load_w_mpa=load_w_mpa,
        peak_vm_mpa=peak_vm,
        peak_location_xy=peak_xy,
        n_dofs=vm_array.size,
        h_fine=h_fine,
        h_coarse=h_coarse,
        vm_field=(vm_array.copy() if return_fields else None),
        coords=(coords.copy() if return_fields else None),
    )
    return result
