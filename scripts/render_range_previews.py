"""
Generate geometry previews at proposed lower / midpoint / upper bounds for
each of R, p, W. Used in the live Day-2 range-locking discussion with Arpit.

Tentative proposed ranges (justification in the console output):
    R in [3.0, 15.0] mm
    p in [35.0, 70.0] mm
    W in [16.0, 32.0] mm

Nominal midpoint for the "other two" parameters is used when sweeping one at a
time. The midpoint sample (R=9, p=52.5, W=24) is also rendered on its own as
the reference-geometry candidate for the mesh convergence study.
"""
from __future__ import annotations

from pathlib import Path
import sys

# allow "python scripts/render_range_previews.py" from project root
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.fea.geometry import LBracketParams, check_validity      # noqa: E402
from src.fea.visualizer import render_single, render_param_sweep  # noqa: E402


OUT = _HERE.parent / "data" / "day2_previews"
OUT.mkdir(parents=True, exist_ok=True)

# Proposed ranges (to be confirmed live with user).
# Binding constraint: Hole 1 is fixed at y = A - 40 = 40 mm, so
#   y_hole1 - r_hole - clearance = 40 - 4 - 2 = 34 must exceed W + R.
# This caps R_max + W_max at 34. Ranges chosen so every (lower, mid, upper)
# triple in the independent sweeps below is valid; at the joint-corner
# (R_max, W_max) a modest LHS rejection rate is expected and acceptable.
R_RANGE = (3.0, 6.5, 10.0)
P_RANGE = (42.0, 57.0, 72.0)
W_RANGE = (14.0, 19.0, 24.0)

NOMINAL = dict(R=R_RANGE[1], p=P_RANGE[1], W=W_RANGE[1])


def main():
    # One-bracket midpoint render — used as the reference sample.
    render_single(LBracketParams(**NOMINAL), OUT / "nominal.png",
                  title=f"Nominal midpoint  R={NOMINAL['R']}  "
                        f"p={NOMINAL['p']}  W={NOMINAL['W']}")

    # Three-panel sweeps per parameter.
    render_param_sweep("R", R_RANGE,
                       fixed=dict(p=NOMINAL["p"], W=NOMINAL["W"]),
                       path=OUT / "sweep_R.png",
                       suptitle="R sweep  (p, W held at midpoint)")

    render_param_sweep("p", P_RANGE,
                       fixed=dict(R=NOMINAL["R"], W=NOMINAL["W"]),
                       path=OUT / "sweep_p.png",
                       suptitle="p sweep  (R, W held at midpoint)")

    render_param_sweep("W", W_RANGE,
                       fixed=dict(R=NOMINAL["R"], p=NOMINAL["p"]),
                       path=OUT / "sweep_W.png",
                       suptitle="W sweep  (R, p held at midpoint)")

    # Validity report for every rendered sample.
    print("\n== Validity check on all rendered samples ==")
    all_samples = [("nominal", LBracketParams(**NOMINAL))]
    for name, rng in [("R", R_RANGE), ("p", P_RANGE), ("W", W_RANGE)]:
        for v in rng:
            kw = dict(NOMINAL)
            kw[name] = v
            all_samples.append((f"{name}={v}", LBracketParams(**kw)))
    for tag, params in all_samples:
        vr = check_validity(params)
        flag = "OK " if vr.ok else "FAIL"
        print(f"  [{flag}] {tag:15s}  R={params.R:5.1f}  p={params.p:5.1f}  "
              f"W={params.W:5.1f}   {vr.reason or ''}")

    print(f"\nWrote previews to {OUT}")


if __name__ == "__main__":
    main()
