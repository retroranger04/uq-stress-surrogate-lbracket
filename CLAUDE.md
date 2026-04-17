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

- **Phase 1 (April 20):** surrogate + ensembles, ~2pp draft.
- **Phase 2 (April 23):** + CQR + comparison, ~5pp draft.
- **Phase 3 (April 26):** + OOD + deployment demo, ~8pp draft.
- Offline April 27 – May 14. Submission May 14-15.
