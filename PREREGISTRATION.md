# Pre-Registration: Out-of-Distribution Evaluation Protocol

## What pre-registration means here, and why it matters

This paper's central contribution is a deployment-calibrated reliability
layer (Deep Ensemble + conformalized quantile regression) wrapped around a
GNN stress surrogate. A reliability claim is only as credible as the
out-of-distribution (OOD) evaluation used to test it, and an OOD evaluation
designed *after* seeing model behavior is vulnerable to post-hoc
cherry-picking of extrapolation directions, magnitudes, sample counts, or
exclusion rules that flatter the result.

To rule that out structurally, the OOD protocol below — parameter ranges,
extrapolation directions, sample counts, seeding, and exclusion rules — was
committed to version control **before any model was trained**. This file
reproduces that commit verbatim (see "Full protocol text" below) and records
the git evidence that establishes the ordering, so the claim is independently
falsifiable: anyone can check out the repository history and confirm the
protocol predates the first training run.

## Git evidence

- **Pre-registration commit:** `fa793a34e7b4f117d1cb37cb688f34667dbd323a`
  (short: `fa793a3`) — *"Clean up failed Kaggle Day 3 artifacts — pivoting to
  local WSL2 execution"*, committed 2026-04-17 20:47:42 +05:30. This commit
  introduced the `## Pre-registered OOD protocol` section into
  `paper/NOTES.md` (verified via `git log --all --oneline -- paper/NOTES.md`
  plus a content check across both commits that ever touched the file — only
  this commit and the initial skeleton commit `d8052ae` modify
  `paper/NOTES.md` in the entire history, and `d8052ae` does not contain the
  protocol text).
- **First training commit:** `71b9a26ca55a400b6d2b71f422b7923b6212edb2`
  (short: `71b9a26`) — *"Phase 1: GNN surrogate + Deep Ensemble +
  test-set eval"*, committed 2026-04-18 11:24:24 +05:30. This is the first
  commit in the repository's history that trains a model (the bare GNN
  surrogate and the 5-member Deep Ensemble); no training occurs in any
  earlier commit.
- **Time delta:** 14 hours 36 minutes 42 seconds — the pre-registration
  commit precedes the first training commit by roughly half a day, crossing
  one calendar-day boundary (2026-04-17 to 2026-04-18).

> **Note on commit hashes.** This repository's history was rewritten with
> `git filter-repo` prior to public release to strip large PDF corpus files
> out of `raw/papers/`. That rewrite changed every commit hash in the
> repository. The hashes above are the current, post-rewrite hashes and are
> the ones that resolve against the published history.

## Full protocol text

*(Reproduced verbatim from `paper/NOTES.md` as committed in `fa793a3`.)*

### Pre-registered OOD protocol (locked 2026-04-16, Day 3, BEFORE any model training)

**Purpose of pre-registration.** The contribution of this paper is a
deployment-calibrated reliability layer. An honest evaluation of that layer
requires an OOD test set whose design is fixed *before* the surrogate is
trained, so that post-hoc cherry-picking of extrapolation directions,
magnitudes, or sample counts is structurally ruled out. This section fixes
those choices; the pre-registered document will be referenced verbatim from
the paper's Methods § OOD evaluation when Phase 3 reports results.

**Training ranges (locked Day 2; see the accompanying paper for methodology details):**

| Parameter | Training range |
|-----------|----------------|
| R (inside fillet radius) | [3.0, 10.0] mm |
| p (Hole 2 x-position)    | [42.0, 72.0] mm |
| W (flange width)          | [14.0, 24.0] mm |

**1. Extrapolation magnitude — moderate ±20% of the training span.**

Each parameter's OOD range extends 20% of its training span beyond each
training bound (feasibility permitting). The resulting per-parameter
pre-feasibility OOD ranges are:

| Parameter | OOD-low (pre-feas.) | OOD-high (pre-feas.) | Notes on feasibility |
|-----------|---------------------|----------------------|----------------------|
| R         | [1.60, 3.00) mm     | (10.00, 11.40] mm    | both directions fully feasible (R≥0.5 floor, W+R≤34 trivially) |
| p         | [36.00, 42.00) mm   | (72.00, 78.00] mm    | OOD-high clipped at p=74 by Hole-2 tip clearance |
| W         | [12.00, 14.00) mm   | (24.00, 26.00] mm    | OOD-low clipped at W=12 by 2(r_h+clearance) |

Rationale: +20% is the standard surrogate-paper setting — clearly outside
training but leaving enough feasible volume after rejection. +10% was ruled
out as not clearly-OOD-enough to stress a well-regularized GNN; +30% was
ruled out because p-high becomes almost entirely infeasible and R-low runs
into the mesher's arc-primitive floor.

**2. Extrapolation directions — both, where feasible.**

For each parameter, OOD samples are drawn from both the OOD-low and OOD-high
sub-intervals (modulo the feasibility clips above). This probes both tails
symmetrically and avoids the structural bias of a worst-stress-only OOD
design. p-high is included despite the narrow (72, 74] feasible band:
expected yield of ~1–2 feasible samples out of the p-high draws is
acceptable since p-low carries most of the p-direction OOD signal.

**3. Combination strategy — mix of single-parameter and corner OOD.**

- **Single-parameter OOD (60 samples).** One parameter drawn from its OOD
  range, the other two drawn from their *training* ranges. Balanced across
  the 6 axial directions (R-low, R-high, p-low, p-high, W-low, W-high),
  10 samples per direction. Provides clean attribution of surrogate error
  to individual parameter axes.
- **Corner OOD (40 samples).** Two or three parameters simultaneously out
  of their training ranges. Drawn by LHS over the full expanded
  [R, p, W] box and rejection-filtered to keep only rows where
  `count_out_of_training_range >= 2`. Probes joint extrapolation — the
  regime where surrogate failure modes typically compound.

**4. Total OOD sample count — ~100.**

`N_OOD = 100` target (60 single + 40 corner), oversampled to absorb
geometric-validity rejection. Large enough for per-direction error
statistics (~10 samples per axial direction); small enough to run the full
OOD sweep in under 2 h on Kaggle CPU alongside the main sweep.

**Random seeding.** Seeds pinned for reproducibility and locked here so the
set cannot be silently redrawn:

- Single-parameter OOD: seed 43 (one LHS draw per direction, scaled to the
  direction's specific OOD-box).
- Corner OOD: seed 44 (single LHS draw over the full expanded box with
  feasibility + ≥2-OOD-params filter).

Reseed-on-short policy: if the oversampled draws yield fewer than the
target count after both filters, increment the seed by 1 and redraw the
deficit. Same behaviour as the main sweep's `LHS_RESEED_ON_SHORT`. All
reseed events are logged for reproducibility.

**What this protocol commits to.**

- The OOD ranges above are fixed. They will not be narrowed, widened, or
  redirected after training results are known.
- The train/test/OOD split assignments are fixed once the datasets are
  published to Kaggle (train = main sweep dataset; OOD = this OOD set;
  within-distribution test = a held-out 10% of the main sweep's LHS
  samples, chosen by LHS-stratified split in `src/models/dataset.py`).
- Reported OOD metrics (per-node MAPE, peak-stress MAPE, conformal
  coverage, prediction-interval width) will be computed on the full 100
  OOD samples — no post-hoc exclusion of "outlier" samples allowed.
- Any deviation from this protocol encountered during Phase 3 must be
  documented with justification *and* the original unmodified result
  also reported in the accompanying paper.

**Recorded artifacts.** The Kaggle OOD dataset
(`retroranger04/lbracket-stress-ood`) will carry the seeds, bounds, and
actual accepted parameter triples in its `manifest.json`, allowing the
exact OOD set to be reconstructed from the seed pair (43, 44).
