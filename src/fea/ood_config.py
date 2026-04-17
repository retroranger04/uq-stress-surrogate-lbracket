"""
Pre-registered OOD protocol (locked 2026-04-16, Day 3, BEFORE any training).

See paper/NOTES.md section "Pre-registered OOD protocol" for the full
rationale and commitments. This module exposes the numerical parameters of
that protocol so the OOD sweep runner (notebooks/day3_ood_sweep.ipynb) can
import them without re-deriving anything. Do NOT adjust these values after
model training has begun \u2014 that would break the pre-registration.

OOD ranges extend each training bound by 20% of the training span. The
feasibility-filter (geometry.check_validity) may further clip these; the
runner logs the accepted counts per direction in the output manifest.
"""

from .config import (
    R_MIN_MM, R_MAX_MM,
    P_MIN_MM, P_MAX_MM,
    W_MIN_MM, W_MAX_MM,
)

# Training-span widths.
_R_SPAN = R_MAX_MM - R_MIN_MM      # 7.0
_P_SPAN = P_MAX_MM - P_MIN_MM      # 30.0
_W_SPAN = W_MAX_MM - W_MIN_MM      # 10.0

# Extrapolation fraction: +20% of the training span beyond each bound.
OOD_FRAC = 0.20

# --- OOD sub-intervals per direction (pre-feasibility filter) -------------

R_OOD_LOW_RANGE = (R_MIN_MM - OOD_FRAC * _R_SPAN, R_MIN_MM)   # [1.60, 3.00)
R_OOD_HIGH_RANGE = (R_MAX_MM, R_MAX_MM + OOD_FRAC * _R_SPAN)   # (10.00, 11.40]

P_OOD_LOW_RANGE = (P_MIN_MM - OOD_FRAC * _P_SPAN, P_MIN_MM)   # [36.00, 42.00)
P_OOD_HIGH_RANGE = (P_MAX_MM, P_MAX_MM + OOD_FRAC * _P_SPAN)   # (72.00, 78.00]

W_OOD_LOW_RANGE = (W_MIN_MM - OOD_FRAC * _W_SPAN, W_MIN_MM)   # [12.00, 14.00)
W_OOD_HIGH_RANGE = (W_MAX_MM, W_MAX_MM + OOD_FRAC * _W_SPAN)   # (24.00, 26.00]

# Full expanded box for corner OOD draws (before the in/out filter).
R_FULL_OOD_BOX = (R_OOD_LOW_RANGE[0], R_OOD_HIGH_RANGE[1])    # [1.60, 11.40]
P_FULL_OOD_BOX = (P_OOD_LOW_RANGE[0], P_OOD_HIGH_RANGE[1])    # [36.00, 78.00]
W_FULL_OOD_BOX = (W_OOD_LOW_RANGE[0], W_OOD_HIGH_RANGE[1])    # [12.00, 26.00]

# --- Sample-count targets -------------------------------------------------

SINGLE_PARAM_DIRECTIONS = [
    ("R", "low"),  ("R", "high"),
    ("p", "low"),  ("p", "high"),
    ("W", "low"),  ("W", "high"),
]
SAMPLES_PER_DIRECTION = 10           # 6 directions * 10 = 60 single-param OOD
N_SINGLE_PARAM_OOD = SAMPLES_PER_DIRECTION * len(SINGLE_PARAM_DIRECTIONS)  # 60

N_CORNER_OOD = 40                    # >=2 parameters simultaneously OOD
CORNER_OOD_OVERSAMPLE = 600          # oversample the LHS box before filtering

OOD_TOTAL_TARGET = N_SINGLE_PARAM_OOD + N_CORNER_OOD  # 100

# --- Seeds (pinned for reproducibility, locked in pre-registration) -------

SEED_SINGLE = 43
SEED_CORNER = 44

# Reseed policy: same as the main sweep \u2014 increment seed by 1 and redraw the
# deficit if the initial draw yields fewer than the target count after
# feasibility + OOD filters.
OOD_RESEED_ON_SHORT = True
OOD_MAX_RESEEDS = 10
