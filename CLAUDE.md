# CLAUDE.md — uq-stress-surrogate-lbracket

Conventions for future Claude Code sessions in this project. Read this before touching anything.

## Project goal

Build an uncertainty-aware neural surrogate for parametric 2D L-bracket stress prediction. A GNN (PyTorch Geometric) trained on FEniCSx simulations predicts von Mises stress fields for L-brackets parameterized by hole diameter, hole position, and fillet radius. Deep Ensembles plus Conformalized Quantile Regression layered on top for deployment-calibrated uncertainty. **The contribution is the reliability layer, not the surrogate.** Target venue: CAISc 2026 Open-Ended Problems (BITS Pilani). Drafting venue-agnostically until CFP/template arrive.

## Compute split (strict) — revised 2026-04-17

- **Local WSL2 (Ubuntu):** FEniCSx parametric sweeps. All FEA runs here now.
- **Local 4060 (Windows native):** model development — code, plotting, small smoke runs.
- **Kaggle GPU:** reserved for GNN / ensemble / CQR training only. **Not used for FEA.**

## WSL2 FEA runtime (locked 2026-04-17 Day-3 closeout)

- Distro: `Ubuntu-24.04` (`wsl --install -d Ubuntu-24.04 --no-launch`,
  default user `arpit`, systemd off).
- Conda env: `~/miniforge3/envs/fenicsx` — `fenics-dolfinx=0.9.*`, `mpich`,
  `python-gmsh`, `pyvista`, `psutil`. dolfinx 0.10 breaks the 0.9
  `LinearProblem` constructor — keep the `=0.9.*` pin.
- Working dir: `~/lbracket-sweep/` (WSL native FS). Reads source from
  `/mnt/a/AntigravityWF/projects/uq-stress-surrogate-lbracket/` but writes
  all `.npz` + tmp meshes to `~/lbracket-sweep/output/` to avoid the
  `/mnt/` performance penalty. Raw shards are copied back into
  `data/day3_{main,ood}/` after the sweep. `data/` stays gitignored.
- Sweep entry point: `scripts/run_sweep_local.py --mode {validate,smoke,lhs,
  main,ood}`. Sequential (`n_workers=1`), SIGINT-safe, resume-capable
  (existing `sample_NNNNN.npz` is skipped on restart), atomic `.npz` write
  via `tmpfile + os.replace`, RAM guardrail pauses at >80% and resumes at
  <75%. Progress every 10 samples. Regenerating the full 1100-sample
  dataset end-to-end takes ~8 min on this laptop — no need to treat it as
  a long-running artifact.

### Why the pivot off Kaggle for FEA (2026-04-17)

Four Kaggle attempts at the Day-3 sweep (v1 pyvista hang, v2 capture_output
silence, v3 browser-lag firehose, v4 recursive-spawn deadlock) confirmed that
the FEniCSx + multiprocessing(spawn) + Kaggle notebook combination is too
fragile for an ~80-min unattended sweep. Each failure cost hours and the
diagnostic feedback loop was blind (no live CLI log access). Running FEA
locally in WSL2 gives synchronous access to stdout, debuggers, and per-sample
artifacts — worth the machine time.

## Environment constraints

- Workspace is `A:\AntigravityWF\projects\uq-stress-surrogate-lbracket\` and its subfolders only. Do not touch `C:\`.
- Windows host for model/plotting work; **WSL2 Ubuntu for FEA runs** (pivot 2026-04-17, previously forbidden).
- Python **3.14** venv at `venv\`, created with `--system-site-packages` so it inherits the globally-installed torch. (Original spec called for 3.11 + cu121; the Day 1 bootstrap reused the pre-existing global torch 2.11.0+cu126 to avoid a redundant ~2 GB download. Deviation logged in `agent_log.md`.)
- Torch: `2.11.0+cu126`, CUDA 12.6 wheels, running on RTX 4060 Laptop GPU (driver 560.94, CUDA 12.6 capable). If rebuilding on a fresh machine, install from the matching CUDA 12.6 index: `pip install torch --index-url https://download.pytorch.org/whl/cu126`.

## Git identity (local only, not global)

- `user.name` = `Arpit Mathur`
- `user.email` = `retroranger24@gmail.com`

Repo is private until 2026-04-26.

## 2-attempt infrastructure failure rule

If an infrastructure step (install, CUDA check, Kaggle pull, push, etc.) fails, retry **once** with a clear variation. If it fails again, stop and escalate with the error and concrete options. No debug loops.

## agent_log.md is the source of truth

Every significant decision, deviation, result, and escalation gets logged to `agent_log.md` at the project root as it happens. This log is the raw material for the paper's AI Collaboration Disclosure section. Do not edit old entries — append.

## raw/ ownership

`raw/papers/` and `raw/venue/` are populated by a separate Sonnet corpus curator in dedicated sessions. Do not add, edit, or reorganize files under `raw/` from this (orchestrator) session. Treat it as read-only for your purposes.

## Deferred decisions (do not decide in this session)

- Parameter ranges for hole diameter, hole position, fillet radius → **Day 2**.
- Material class (mild / structural / stainless) and handbook-cited E and ν values → **Day 2**, based on engineering intent for the part.
- CAISc venue formatting (template, page limits, bibliography style) → when CFP/template arrive (~April 15). Until then, draft venue-agnostically on the generic article template.

## Locked-at-Day-2 design choices

- 2D plane stress.
- Fixed vertical tip load.
- Fixed outer bracket dimensions.

## Paper

- Venue-agnostic `article` LaTeX template in `paper/`.
- `paper/ai_disclosure.tex` drafted continuously from `agent_log.md` following the Matthew Schwartz "Vibe Physics" precedent.
- `paper/bibliography.bib` is curator-owned. Do not edit from orchestrator sessions.

## Phase targets

- **Phase 1 (April 20):** surrogate + ensembles, ~2pp draft. **DONE 2026-04-18.**
- **Phase 2 (April 23):** + CQR + comparison, ~5pp draft. **DONE 2026-04-18.**
- **Phase 3 (April 26):** + OOD + deployment demo, ~8pp draft. **DONE 2026-04-19.**
- Offline April 27 – May 14. Submission May 14-15.

## Status (end of Phase 3)

- Ensemble (5 members) + CQR layer are frozen. Do not retrain.
- OOD evaluation + deferral rule + full paper draft complete.
- Remaining work before submission: (a) conference-template migration
  when CFP/template arrives; (b) drop in the realistic bracket
  illustration that replaces the `fig:bracket_photo` placeholder in
  `paper/main.tex`; (c) final submission formatting during the
  May 14-15 window. All three are mechanical — no new experiments or
  result recomputation should be needed.

## Model training conventions (locked 2026-04-18, Phase 1)

- **Runtime helpers:** `src/models/runtime.py`. `load_bundle(path)` for the
  `.pt` splits; `compute_stats_from_list` / `apply_stats_inplace` for input
  normalization; `train_one(cfg, tr, va, progress=...)` for the training
  loop with checkpointing; `eval_metrics(model, loader, device, eps_mpa=1)`
  for per-node MAPE + peak MAPE + abs-error percentiles.
- **Phase-1 architecture (locked):** MeshGraphNet encoder / L=5 processor /
  decoder, H=128, Huber(δ=1 MPa), Adam lr 5e-4 cosine-decayed to 1e-5,
  bs=8, patience 10–12 on val Huber. Baseline: 60 ep. Ensemble members:
  45 ep. Defined as the default in `scripts/phase1_train.py`; change the
  CLI args, not the runtime helpers.
- **Batch size on RTX 4060 Laptop:** `bs=8` only. `bs=16` regressed to
  ~10× per-step time despite fitting in VRAM — root cause not worth
  chasing for Phase 2/3. If training time matters for future runs,
  reduce epochs or model size, not batch size.
- **Checkpoint locations.** Each run dir carries `best.pt`, `stats.pt`,
  `cfg.json`, `history.json`, `val_metrics.json`. `best.pt` is
  `{"model": state_dict, "cfg": RunCfg-as-dict}` — load via
  `runtime.load_best(cfg_like, ckpt_path, device)`.
- **Phase-1 trained artifacts (Phase 2 CQR consumes these directly):**
  - `runs/baseline/best.pt` — single-model baseline.
  - `runs/ensemble/stats.pt` — shared input-normalization stats.
  - `runs/ensemble/seed{0,101,202,303,404}/best.pt` — the 5 Deep-Ensemble
    members. Load the list with `[load_best(_, d/'best.pt', dev) for d
    in [runs/ensemble/seed0, ...]]`.
- **Test-set discipline.** `data/test.pt` is used only by
  `scripts/phase1_eval.py` for the final frozen evaluation; no script
  should ever call it during hyperparameter selection.
  `data/ood.pt` stays untouched until Phase 3.
- **HP-sweep CLI.** `scripts/phase1_hp_sweep.py --base runs/baseline
  --epochs N` runs the remaining grid points and ranks by val per-node
  MAPE. Outputs `summary.json` + `winner.json` at `runs/hp_sweep/`.

## CQR inference conventions (locked 2026-04-18, Phase 2)

- **Core module:** `src/uq/cqr.py`. Primitives: `z_for_alpha`,
  `base_quantiles(mu, sigma, alpha)`, `conformity_scores(y, q_lo, q_hi)`,
  `conformal_quantile(scores, alpha)` (with finite-sample correction),
  `calibrated_interval(mu, sigma, alpha, q_hat)`, `coverage`, `width`.
- **Base quantile rule:** Gaussian `mu ± z·sigma` with
  `z = Φ⁻¹(1 − α/2)`. `mu`, `sigma` are ensemble mean and std across the
  5 Phase-1 members.
- **Primary α:** 0.10 (90% nominal coverage). Sweep also reports
  α ∈ {0.20, 0.15, 0.10, 0.05}.
- **Calibration set:** `data/val.pt` (500 samples). Cell-wise pooling per
  Gopakumar 2024 — all nodes × all val samples form one score pool;
  `Q_hat` is the scalar output.
- **Phase-2 artifacts (Phase 3 consumes these):**
  - `runs/cqr/preds_{val,test}.npz` — cached per-graph ensemble
    predictions (keys: `mean`, `std`, `gt`, `pos`, `params`, `offsets`).
    Offsets are the node-flat per-graph boundaries.
  - `runs/cqr/calibration.json` — full sweep + conditional + informativeness.
  - `runs/cqr/cqr_primary.pt` — `{alpha, q_hat, z, base_quantile_rule,
    ensemble_seeds, stats_path, paper_ref}`. Load with `torch.load(...,
    weights_only=False)`.
- **Inference for new samples (Phase-3 pattern):** run the 5 members to
  get per-node `mu, sigma`; call `base_quantiles(mu, sigma, alpha)` to
  get the raw interval; apply the saved `q_hat` via
  `calibrated_interval` (or manually `lo = q_lo − q_hat`,
  `hi = q_hi + q_hat`). Coverage on OOD inputs is NOT guaranteed by
  Theorem 1 — Phase 3 reports degradation as the headline OOD result.
- **Pipeline CLI.** `scripts/phase2_cqr.py` regenerates everything
  end-to-end (predictions cache + calibration sweep + tables + figures).
  `--recompute` forces re-inference; otherwise the `.npz` caches are
  reused.
- **Test-set discipline (unchanged).** `data/test.pt` is used only for
  the final frozen evaluation here. `data/ood.pt` stays untouched.
