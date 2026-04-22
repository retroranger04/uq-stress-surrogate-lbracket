# Uncertainty-Aware Neural Surrogate for Parametric L-Bracket Stress Prediction

This project builds a fast AI model that predicts mechanical stress in structural L-shaped brackets, replacing expensive finite-element simulations (minutes per run) with millisecond-speed predictions. The key contribution is a reliability layer that quantifies how confident the model is in each prediction — and raises a flag when the engineer should run the full simulation instead. We validate this on a parametric 2D L-bracket with three variable geometric parameters (fillet radius, hole position, flange width), using a pre-registered out-of-distribution (OOD) test protocol that was locked before any model training began.

## Key Results

- **1.81% per-node MAPE** on the held-out test set (500 samples); 0.42% error on peak stress prediction — the safety-critical quantity.
- **Guaranteed 90% coverage**: the conformal prediction interval contains the true stress value at 91.28% of nodes, meeting the formal finite-sample coverage theorem.
- **Uncertainty tracks difficulty**: sample-level Pearson correlation between ensemble spread and prediction error = **0.944**, validating the reliability signal.
- **OOD detection**: a simple threshold deferral rule achieves **34% OOD detection at 5% false-alarm rate** on the in-distribution test set; corner-extrapolation cases (all three parameters simultaneously out of range) are caught **58%** of the time.
- **OOD coverage gracefully degrades** from 91.3% (in-distribution) to 83.7% (OOD) — the reliability layer correctly signals the regime where the guarantee no longer applies.

## Repository Structure

```
paper/       LaTeX source, figures, tables, and bibliography for the CAISc 2026 submission
src/         Python source: FEA pipeline (src/fea/), GNN model (src/models/), UQ layer (src/uq/), evaluation (src/eval/)
scripts/     End-to-end pipeline scripts: FEA sweep, GNN training, CQR calibration, OOD evaluation, figure generation
runs/        Saved model checkpoints and evaluation artifacts (baseline, HP sweep, ensemble, CQR)
data/        Dataset bundles — gitignored; regenerate with scripts/run_sweep_local.py + scripts/package_to_pyg.py
notebooks/   Kaggle validation notebook from Day-2 pipeline verification
raw/         Literature corpus (raw/papers/), bibliography, and Graphify knowledge graph
tests/       Unit tests
archive/     Superseded files kept for reference
```

## Method Overview

**GNN surrogate.** A MeshGraphNet-style graph neural network operates directly on the FEA mesh: each node carries geometric features (coordinates, boundary type, parameter values, distance to fillet/holes) and each edge carries relative position. Five message-passing layers with residual connections map these to per-node von Mises stress predictions. The model has 0.85M parameters and runs on a mid-range laptop GPU in milliseconds per bracket geometry.

**Deep Ensemble.** Five independently-seeded copies of the GNN are trained from scratch. At inference time, the mean of their predictions is the point estimate; the standard deviation across members is the epistemic uncertainty signal. This uncertainty is what makes the reliability layer possible: the ensemble disagrees most in regions where prediction errors are large (sample-level Pearson = 0.944).

**Conformalized Quantile Regression (CQR).** A thin calibration layer adjusts the raw Gaussian intervals from the ensemble using a held-out calibration set of 500 samples. The result is a distribution-free, finite-sample coverage guarantee (Romano et al., 2019): for any target coverage level α, the calibrated interval contains the true stress value at least (1−α) of the time on in-distribution inputs. No retraining is required — CQR is a pure post-hoc retrofit.

## Reproducing Results

**Hardware used:** RTX 4060 Laptop GPU (8 GB VRAM), 24 GB RAM; FEA sweeps run in WSL2 (Ubuntu 24.04) with a Miniforge3 conda environment.

**Key dependencies:** Python 3.14, PyTorch 2.11.0+cu126, PyTorch Geometric 2.7.0, FEniCSx 0.9.x (WSL2 only), scipy, numpy, matplotlib.

**1. Regenerate dataset** (WSL2, ~52 min end-to-end):
```bash
python scripts/run_sweep_local.py --mode main --target 5000
python scripts/run_sweep_local.py --mode ood
python scripts/package_to_pyg.py --seed 0
```

**2. Train the ensemble** (Windows, RTX 4060, ~5 h total):
```bash
python scripts/phase1_train.py          # baseline
python scripts/phase1_ensemble.py       # members 1-4
python scripts/phase1_eval.py           # final test evaluation
```

**3. Calibrate CQR and evaluate OOD:**
```bash
python scripts/phase2_cqr.py            # CQR calibration + test coverage
python scripts/phase3_ood.py            # OOD evaluation + deferral rule
```

**4. Regenerate paper figures:**
```bash
python scripts/make_paper_figures.py
```

## Citation

```bibtex
@inproceedings{mathur2026uqlbracket,
  title={Uncertainty-Aware Neural Surrogate for Parametric L-Bracket Stress Prediction},
  author={Mathur, Arpit},
  year={2026},
  note={Under review}
}
```

## License

License TBD.

## Acknowledgments

AI coding tools were used extensively throughout this project. Full disclosure — including session logs, orchestrator identity, and a detailed AI Involvement Checklist — is provided in `paper/ai_disclosure.tex` and in `agent_log.md`.
