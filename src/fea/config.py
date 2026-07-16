"""
Locked parameter ranges and sweep configuration for the L-bracket surrogate.

Ranges locked 2026-04-16. Do NOT adjust without re-running the visualizer +
reconfirming.

Binding constraint: Hole 1 is fixed at y = 40 mm, so W + R <= 34 mm (after the
hole-radius + clearance margin). The joint corner (R=10, W=24) is exactly on
the boundary — a few percent LHS rejection is expected and handled by
oversampling.
"""

# --- Parameter ranges (inclusive) ------------------------------------------

R_MIN_MM, R_MAX_MM = 3.0, 10.0
P_MIN_MM, P_MAX_MM = 42.0, 72.0
W_MIN_MM, W_MAX_MM = 14.0, 24.0

PARAM_RANGES = {
    "R": (R_MIN_MM, R_MAX_MM),
    "p": (P_MIN_MM, P_MAX_MM),
    "W": (W_MIN_MM, W_MAX_MM),
}

# --- Sweep target ----------------------------------------------------------

# Oversample to absorb LHS corner rejection; target at least TARGET_VALID
# feasible samples for the Day-3 parametric sweep.
TARGET_VALID = 1000
LHS_OVERSAMPLE = 1100          # draws before validity filter
LHS_RESEED_ON_SHORT = True     # if < TARGET_VALID remain after filter, reseed

# --- Reference (nominal) sample used for mesh convergence + cross-checks ---

NOMINAL = dict(
    R=(R_MIN_MM + R_MAX_MM) / 2.0,   # 6.5
    p=(P_MIN_MM + P_MAX_MM) / 2.0,   # 57.0
    W=(W_MIN_MM + W_MAX_MM) / 2.0,   # 19.0
)

# --- Worst-case sample used for load calibration ---------------------------

# Smallest R + thinnest flange + Hole 2 nearest the fillet produces the
# sharpest stress concentration. Used to tune the distributed load w so the
# peak von Mises is ~= 50% of sigma_y (~100 MPa).
WORST_CASE = dict(R=R_MIN_MM, p=P_MIN_MM, W=W_MIN_MM)

# --- Calibrated distributed load (Day-2, locked 2026-04-16) ---------------

# Applied downward traction on the top of the horizontal flange over the
# segment [W+R, B]. Magnitude chosen so that the worst-case sample (R_MIN,
# p_MIN, W_MIN) produces a peak von Mises equal to 50% of yield.
#
# Kaggle day-2 kernel v6 gave peak_vm = 119.808 MPa at w=1 MPa for the worst-
# case sample; the linear-elasticity scaling factor
#     w = 0.5 * sigma_y / peak_vm@1 = 0.5 * 205 / 119.808
# yields the value below. See data/day2_validation/day2_results.json for the
# raw run output.
LOAD_W_MPA: float = 0.8555  # MPa

# --- Converged mesh refinement (Day-2, locked 2026-04-16) -----------------

# Level-3 of the convergence sweep converged to within 0.024% of level-2
# (<< 1% criterion). These are the refinement targets used by the Day-3
# parametric sweep for every sample.
MESH_H_COARSE_MM = 2.0
MESH_H_FINE_MM = 0.2
MESH_REFINE_DIST_MM = 8.0
