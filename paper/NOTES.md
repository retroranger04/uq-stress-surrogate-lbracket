# paper/NOTES.md

Working notes for the manuscript. Authoritative decision log lives in `../agent_log.md` — this file cross-references it.

## Venue formatting checklist

<!-- Populate when CAISc 2026 CFP / template lands (~April 15). Items to track: document class, page limit, bibliography style, figure/table rules, anonymization requirements, submission portal. -->

## Phase draft targets

- **Phase 1 (April 20, ~2pp):** Introduction skeleton, Methods surrogate + ensembles, first pass at Experiments + Results for the surrogate baseline.
- **Phase 2 (April 23, ~5pp):** + Related Work, + CQR in Methods, + CQR-vs-ensemble comparison in Results.
- **Phase 3 (April 26, ~8pp):** + OOD protocol + deployment demo in Methods/Results, Discussion, Conclusion, final abstract.

## Pre-registered OOD protocol (locked 2026-04-16, Day 3, BEFORE any model training)

**Purpose of pre-registration.** The contribution of this paper is a
deployment-calibrated reliability layer. An honest evaluation of that layer
requires an OOD test set whose design is fixed *before* the surrogate is
trained, so that post-hoc cherry-picking of extrapolation directions,
magnitudes, or sample counts is structurally ruled out. This section fixes
those choices; the pre-registered document will be referenced verbatim from
the paper's Methods § OOD evaluation when Phase 3 reports results.

**Training ranges (locked Day 2, see `agent_log.md` Day 2 entry):**

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
reseed events logged in `agent_log.md`.

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
  logged in `agent_log.md` with justification *and* the original
  unmodified result also reported.

**Recorded artifacts.** The Kaggle OOD dataset
(`retroranger04/lbracket-stress-ood`) will carry the seeds, bounds, and
actual accepted parameter triples in its `manifest.json`, allowing the
exact OOD set to be reconstructed from the seed pair (43, 44).

## Open questions

<!-- Accumulate unresolved issues here; move to agent_log.md once decided. -->

## Phase 1 experiment plan (drafted 2026-04-16, Day 3)

**Scope.** Phase 1 trains the bare GNN surrogate (no UQ yet) plus Deep
Ensembles. Establishes the accuracy floor against which Phase 2's CQR
layer is evaluated.

### Architecture

MeshGraphNets-style encoder / processor / decoder per Pfaff et al. 2021
(`pfaff2021meshgraphnets`). Implemented at `src/models/gnn.py`:

- **Encoder.** Node-MLP maps per-node features (2D position, one-hot
  boundary type over {clamped, loaded, fillet, hole1, hole2, free},
  distance-to-fillet, distance-to-nearest-hole, per-graph params
  broadcast as node features) to hidden dim `H`. Edge-MLP maps edge
  features (Δx, Δy, ‖Δ‖) to hidden dim `H`.
- **Processor.** `L` message-passing layers with residual connections.
  Each layer: edge update `e' = MLP(cat(e, n_i, n_j)) + e`; node update
  `n' = MLP(cat(n, aggregate_j(e_ij))) + n`. Aggregate = sum
  (permutation-invariant).
- **Decoder.** Node-MLP maps hidden → 1-dim per-node von Mises stress.

### Hyperparameter sweep

| Knob | Values |
|------|--------|
| Hidden dim `H` | {64, 128, 256} |
| Num processor layers `L` | {3, 5, 7} |
| Learning rate (initial) | {1e-3, 5e-4} |

Cosine LR decay to 1e-5 over the full budget. Optimizer: Adam (default
β). Batch size: 8 graphs. Budget: 200 epochs with early stopping on
validation per-node MSE (patience 25 epochs). Trained with standard
PyTorch + PyTorch-Geometric.

### Evaluation metrics (same metrics used in Phase 1, 2, 3)

- **Per-node stress MAPE** — mean absolute percent error over all nodes,
  all samples in the split. Primary surrogate-quality metric.
- **Peak stress MAPE** — absolute percent error on the peak von Mises
  value per sample, averaged over samples. This is the deployment-
  relevant scalar (what a design engineer actually reads off).
- **Spatial error distribution** — per-node absolute error characterized
  by its 50th / 90th / 99th / 100th percentiles (across all nodes across
  all samples), to surface tail-heavy pathologies that MAPE hides.

### Baseline comparison plan

The bare GNN (single model, best hyperparameter setting) establishes the
accuracy floor. Deep Ensembles do not change the point prediction
significantly (ensemble mean) — their value is the epistemic-uncertainty
estimate, evaluated via:

- **Calibration of the ensemble variance as an uncertainty proxy** —
  reliability diagram of nominal-coverage-vs-actual-coverage when
  variance is converted to a Gaussian prediction interval. Phase 1 just
  reports this *raw* number to set the scene for Phase 2 CQR.

### Ensemble training protocol

Per Lakshminarayanan et al. 2017 (`lakshminarayanan2017ensembles`):

- `M = 5` members. Evidence from the original paper: M=5 captures most of
  the variance benefit; marginal returns drop sharply for M > 5.
- Independent random initializations per member (different torch seed for
  parameters + for the dataloader-shuffle stream).
- Identical architecture + hyperparameters across members.
- Identical training data across members; member diversity arises from
  initialization + SGD trajectory only.
- Member predictions aggregated at inference: mean for point prediction,
  variance for epistemic uncertainty.

### What Phase 1's paper draft adds beyond Day 3's FEA Methods

- Methods § GNN architecture (MeshGraphNets walk-through, tied to
  `pfaff2021meshgraphnets` + `maurizi2022gnn` for stress-field context
  + `gladstone2024gnn` for multi-GNN extension rationale).
- Methods § Deep Ensembles (training protocol verbatim from this file).
- Experiments § Train/val/test split protocol (LHS-stratified split, size
  numbers from actual main-sweep yield).
- Experiments § Metrics definitions.
- Results § Phase-1 table of per-node MAPE, peak MAPE, percentile errors
  on within-distribution test.
- Results § Ensemble-variance reliability-diagram figure (raw,
  pre-conformal).

## Decisions log

Authoritative log: `../agent_log.md`. Reference specific entries here when a paper-level decision is made (e.g., "see agent_log.md 2026-04-20 entry on ensemble size").
