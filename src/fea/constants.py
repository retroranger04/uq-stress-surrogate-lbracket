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
