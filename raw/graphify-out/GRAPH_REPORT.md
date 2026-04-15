# Graph Report - papers  (2026-04-15)

## Corpus Check
- 14 files · ~180,422 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 333 nodes · 378 edges · 34 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `MeshGraphNets Framework` - 19 edges
2. `Psaros 2023: UQ in Scientific Machine Learning` - 13 edges
3. `Stress Field Prediction in Fiber-Reinforced Composites using Deep Learning` - 11 edges
4. `Predicting Stress, Strain and Deformation Fields in Materials and Structures with GNNs` - 11 edges
5. `Conformal Prediction (CP)` - 11 edges
6. `BNNs for Predicting Uncertainty in Full-Field Material Response (Pasparakis et al. 2024)` - 11 edges
7. `Mesh-based GNN Surrogates for Time-Independent PDEs` - 10 edges
8. `Uncertainty Quantification of Surrogate Models using Conformal Prediction` - 10 edges
9. `StressGAN: A Generative Deep Learning Model for 2D Stress Distribution Prediction` - 10 edges
10. `Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles` - 9 edges

## Surprising Connections (you probably didn't know these)
- `CONCEPT: GNN as Mesh-Based Surrogate for Unstructured Geometry (beyond regular grid CNNs)` --semantically_similar_to--> `StressNet: Multi-Channel CNN Architecture for Stress Prediction`  [INFERRED] [semantically similar]
  papers/pyg_docs.md → papers/nie2020stress.pdf
- `von Mises Stress Field as Prediction Target (sigma_vm = sqrt(sx^2 + sy^2 - sx*sy + 3*tau_xy^2))` --semantically_similar_to--> `FEniCSx Von Mises Post-processing: sigma_vm = sqrt(3/2 * s:s), s = sigma - (1/3)tr(sigma)*I (deviatoric)`  [INFERRED] [semantically similar]
  papers/nie2020stress.pdf → papers/dokken_fenicsx.md
- `FEniCSx Linear Elasticity PDE: -div(sigma(u)) = f, sigma = lambda*tr(eps)*I + 2*mu*eps` --semantically_similar_to--> `Linear Elasticity FEA Governing Equations: Ku=F, sigma=CgBu, plane-strain constitutive law`  [INFERRED] [semantically similar]
  papers/dokken_fenicsx.md → papers/nie2020stress.pdf
- `CONCEPT: GNN as Mesh-Based Surrogate for Unstructured Geometry (beyond regular grid CNNs)` --semantically_similar_to--> `Bayesian U-Net Architecture: Modified encoder-decoder CNN with probabilistic filter parameters`  [INFERRED] [semantically similar]
  papers/pyg_docs.md → papers/pasparakis2024bayesian.pdf
- `MeshConv Layer (MeshGraphNets-style): edge_mlp(x_i || x_j || edge_attr), node_mlp(x || aggr_out)` --rationale_for--> `CONCEPT: ML Surrogate for Stress Field Prediction (replaces expensive FEA at inference)`  [INFERRED]
  papers/pyg_docs.md → papers/nie2020stress.pdf

## Hyperedges (group relationships)
- **Conformal Prediction as Post-Hoc UQ for Neural Surrogate Models** — angelopoulos2023conformal_conformal_prediction, gopakumar2024conformal_cp_framework, shared_surrogate_modeling, shared_uncertainty_quantification [INFERRED 0.95]
- **GNN-Based Mesh Surrogate for Stress/Strain Field Prediction in Solid Mechanics** — maurizi2022gnn_graph_neural_network, gladstone2024gnn_edge_augmented_gnn, shared_stress_field_prediction, shared_finite_element_method [INFERRED 0.90]
- **Deep Learning Architectures for 2D/3D Stress Field Prediction (CNN, GAN, GNN)** — bhaduri2022stress_unet, jiang2021stressgan_cgan, maurizi2022gnn_graph_neural_network, shared_stress_field_prediction [INFERRED 0.88]
- **Encode-Process-Decode Architectural Components** — pfaff2021meshgraphnets_encoder, pfaff2021meshgraphnets_processor, pfaff2021meshgraphnets_decoder, pfaff2021meshgraphnets_forward_euler_integrator [EXTRACTED 1.00]
- **Multigraph Edge Type Set** — pfaff2021meshgraphnets_mesh_edges, pfaff2021meshgraphnets_world_edges, pfaff2021meshgraphnets_mesh_nodes [EXTRACTED 1.00]
- **Processor Message Passing Update Functions** — pfaff2021meshgraphnets_edge_update_fM, pfaff2021meshgraphnets_edge_update_fW, pfaff2021meshgraphnets_node_update_fV, pfaff2021meshgraphnets_residual_connection, pfaff2021meshgraphnets_relu_mlp [EXTRACTED 1.00]
- **Cloth Simulation Datasets** — pfaff2021meshgraphnets_dataset_flagsimple, pfaff2021meshgraphnets_dataset_flagdynamic, pfaff2021meshgraphnets_dataset_spheredynamic, pfaff2021meshgraphnets_solver_arcsim, pfaff2021meshgraphnets_cloth_pde [EXTRACTED 1.00]
- **Fluid Simulation Datasets** — pfaff2021meshgraphnets_dataset_cylinderflow, pfaff2021meshgraphnets_dataset_airfoil, pfaff2021meshgraphnets_navier_stokes, pfaff2021meshgraphnets_eulerian_system [EXTRACTED 1.00]
- **Structural Mechanics Dataset** — pfaff2021meshgraphnets_dataset_deformingplate, pfaff2021meshgraphnets_solver_comsol, pfaff2021meshgraphnets_hyperelasticity_pde, pfaff2021meshgraphnets_lagrangian_system, pfaff2021meshgraphnets_von_mises_stress_output [EXTRACTED 1.00]
- **Training Regime Components** — pfaff2021meshgraphnets_one_step_supervision, pfaff2021meshgraphnets_training_noise, pfaff2021meshgraphnets_noise_target_adjustment, pfaff2021meshgraphnets_l2_loss, pfaff2021meshgraphnets_adam_optimizer, pfaff2021meshgraphnets_feature_normalization, pfaff2021meshgraphnets_v100_gpu [EXTRACTED 1.00]
- **Adaptive Remeshing Pipeline** — pfaff2021meshgraphnets_sizing_field, pfaff2021meshgraphnets_learned_remeshing, pfaff2021meshgraphnets_generic_remesher, pfaff2021meshgraphnets_local_remesher_ops, pfaff2021meshgraphnets_sizing_field_estimation, pfaff2021meshgraphnets_anisotropic_delaunay [EXTRACTED 1.00]
- **Ablation Studies and Baselines** — pfaff2021meshgraphnets_ablation_world_edges, pfaff2021meshgraphnets_ablation_relative_encoding, pfaff2021meshgraphnets_ablation_history, pfaff2021meshgraphnets_baseline_gcn, pfaff2021meshgraphnets_baseline_gns, pfaff2021meshgraphnets_baseline_unet, pfaff2021meshgraphnets_gns_mesh_pos_hybrid, pfaff2021meshgraphnets_gcn_mlp_hybrid [EXTRACTED 1.00]
- **Key Design Rationale Nodes** — pfaff2021meshgraphnets_rationale_relative_encoding, pfaff2021meshgraphnets_rationale_world_edges, pfaff2021meshgraphnets_rationale_mesh_edges, pfaff2021meshgraphnets_rationale_noise_injection, pfaff2021meshgraphnets_rationale_learned_sizing, pfaff2021meshgraphnets_rationale_graph_over_cnn, pfaff2021meshgraphnets_rationale_h1_history [INFERRED 0.95]
- **Node and Edge Feature Encoding Components** — pfaff2021meshgraphnets_mesh_edge_features, pfaff2021meshgraphnets_world_edge_features, pfaff2021meshgraphnets_node_type_embedding, pfaff2021meshgraphnets_mesh_space_coordinate, pfaff2021meshgraphnets_world_space_coordinate, pfaff2021meshgraphnets_relative_edge_features [EXTRACTED 1.00]
- **Output Integration Schemes** — pfaff2021meshgraphnets_forward_euler_integrator, pfaff2021meshgraphnets_first_order_integration, pfaff2021meshgraphnets_second_order_integration, pfaff2021meshgraphnets_output_features_pi [EXTRACTED 1.00]
- **Bayesian UQ Methods Family** — psaros2023uq_hmc, psaros2023uq_ld, psaros2023uq_mfvi, psaros2023uq_mcd, psaros2023uq_la [EXTRACTED 1.00]
- **Ensemble UQ Methods Family** — psaros2023uq_deep_ensembles, psaros2023uq_snapshot_ensembles, psaros2023uq_swag [EXTRACTED 1.00]
- **Primary UQ Evaluation Metrics Triad** — psaros2023uq_rl2e, psaros2023uq_mpl, psaros2023uq_rmsce [EXTRACTED 1.00]
- **Post-Training Calibration Methods Set** — psaros2023uq_variance_reweighting, psaros2023uq_cdf_calibration, psaros2023uq_crude_calibration [EXTRACTED 1.00]
- **Neural PDE UQ Methods** — psaros2023uq_u_pinn, psaros2023uq_pi_gan_fp, psaros2023uq_gp_pi_gan, psaros2023uq_u_pi_gan, psaros2023uq_u_nnpc [EXTRACTED 1.00]
- **Neural Operator UQ Methods** — psaros2023uq_u_deeponet, psaros2023uq_pa_bnn_fp, psaros2023uq_pa_gan_fp [EXTRACTED 1.00]
- **UQ Taxonomy: Aleatoric + Epistemic = Total** — psaros2023uq_aleatoric_uncertainty, psaros2023uq_epistemic_uncertainty, psaros2023uq_total_uncertainty [EXTRACTED 1.00]
- **Heteroscedastic Noise Modeling Methods** — psaros2023uq_h_hmc_fp, psaros2023uq_h_deepens, psaros2023uq_heteroscedastic_noise, psaros2023uq_student_t_noise [EXTRACTED 0.95]
- **Comparative Study Benchmark Problems** — psaros2023uq_discontinuous_fn_approx, psaros2023uq_diffusion_reaction_eq, psaros2023uq_stochastic_elliptic_eq, psaros2023uq_porous_media_flow [EXTRACTED 1.00]
- **Key Findings: Deep Ensembles and HMC Dominant** — psaros2023uq_hmc_dens_best_calibration, psaros2023uq_la_mfvi_sens_tradeoff, psaros2023uq_plurality_metrics, psaros2023uq_calibration_small_dataset [EXTRACTED 1.00]

## Communities

### Community 0 - "CNN/DL Stress Surrogates"
Cohesion: 0.07
Nodes (41): FEM Surrogate via Deep Learning, Fiber-Reinforced Composite Material System, Image-to-Image Mapping for Stress Prediction, Stress Field Prediction in Fiber-Reinforced Composites using Deep Learning, Rationale: U-Net chosen for skip connections enabling context propagation and high-resolution detail, Skip Connections (U-Net), StressNet (SE-Res-FCN baseline), Transfer Learning (few-fiber to many-fiber) (+33 more)

### Community 1 - "Conformal Prediction Theory"
Cohesion: 0.09
Nodes (30): Calibration Set, Conformal Prediction (CP), Conformalized Quantile Regression (CQR), Conformal Prediction Under Covariate Shift, Marginal Coverage Guarantee, Distribution-Free Uncertainty Quantification, Exchangeability Assumption, Nonconformity Score (+22 more)

### Community 2 - "MeshGraphNets Ablations"
Cohesion: 0.07
Nodes (29): Ablation: No Relative Encoding (RMSE 26.5 on Airfoil), Ablation: No World Edges (+51%/+92% RMSE on cloth), Adam Optimizer (lr 1e-4 to 1e-6, 10M steps), Dataset: AirfoilSteady (steady-state prediction, GCN baseline validation), Baseline: GCN (Kipf & Welling, no edge messages), Baseline: GNS (Sanchez-Gonzalez et al., particle-based), Baseline: UNet CNN (Thuerey et al., grid-based), Dataset Split: 1000 Train / 100 Val / 100 Test Trajectories (+21 more)

### Community 3 - "ML Surrogate Concepts"
Cohesion: 0.1
Nodes (28): CONCEPT: FEA as Ground Truth Data Generator for ML Surrogate Training, CONCEPT: Image-to-Image Regression (microstructure/geometry image -> stress field image), CONCEPT: ML Surrogate for Stress Field Prediction (replaces expensive FEA at inference), CONCEPT: Uncertainty Quantification for NN Surrogates in Solid Mechanics, FEniCSx Linear Elasticity Tutorial — Dokken (Reference Document), DOLFINx Python Implementation: LinearProblem, VectorElement Lagrange, PETSc LU solver, FEniCSx Linear Elasticity PDE: -div(sigma(u)) = f, sigma = lambda*tr(eps)*I + 2*mu*eps, Lame Parameters: lambda = E*nu/((1+nu)(1-2nu)), mu = E/(2(1+nu)) (+20 more)

### Community 4 - "Deep Ensembles and UQ"
Cohesion: 0.09
Nodes (27): Bayesian Neural Network Functional Prior (BNN-FP), Deep Ensembles (DEns), Deep Ensembles Capture Multiple Loss Landscape Modes, U-DeepONet (DEns) Epistemic Uncertainty Increases for OOD, Deep Operator Network (DeepONet), Epistemic Uncertainty Increases with NN Expressivity, Fourier Neural Operator (FNO), Functional Prior (FP) via GAN (+19 more)

### Community 5 - "UQ Calibration Metrics"
Cohesion: 0.09
Nodes (24): Aleatoric Uncertainty, Calibration Plot / Reliability Diagram, Conformal Prediction Not Covered (Noted as Gap), CRPS Not Explicitly Used as Primary Metric, Benchmark: Nonlinear Diffusion-Reaction Equation, Benchmark: Discontinuous Function Approximation, Epistemic Uncertainty, Gaussian Likelihood Function (+16 more)

### Community 6 - "MeshGraphNets Datasets"
Cohesion: 0.09
Nodes (23): Barycentric Interpolation for Dynamic Mesh History, Compressible NS Encoding: u_ij, |u_ij| (mesh); n_i, w_i, rho_i (node), Dataset: Airfoil (compressible NS, 2D Eulerian), Dataset: CylinderFlow (incompressible NS, 2D Eulerian), Dataset: DeformingPlate (hyper-elastic, tetrahedral), Eulerian System (fixed mesh, evolving fields), First-Order Integration (q^{t+1} = p_i + q^t_i), Hyper-Elastic Encoding: u_ij, |u_ij|, x_ij, |x_ij| (mesh); x_ij, |x_ij| (world); n_i (node) (+15 more)

### Community 7 - "GNN Mesh Surrogate Training"
Cohesion: 0.14
Nodes (19): CONCEPT: GNN as Mesh-Based Surrogate for Unstructured Geometry (beyond regular grid CNNs), Adam Optimizer with Exponential LR Decay, batch size 256 (TensorFlow, GTX 1080Ti), Mean Relative Error (MRE) Metric: |y - yhat| / (eps + max(y, yhat)), MSE Loss for Training (stress field pixel-wise mean squared error), Rationale: Multi-channel input (geometry + load-x + load-y + BC-x + BC-y) encodes arbitrary 2D BCs as separate image planes, StressNet Result: 2.04% Mean Relative Error on Test Set (100k/20.96k split), SE-ResNet Module: Squeeze-and-Excitation + Residual Block in StressNet, StressNet: Multi-Channel CNN Architecture for Stress Prediction (+11 more)

### Community 8 - "MeshGraphNets Dynamics"
Cohesion: 0.12
Nodes (19): Ablation: History Size h (h=1 best, extra h causes overfitting), Acceleration Output (x_ddot_i, cloth second-order), Adaptive Remeshing, Anisotropic Delaunay Criterion (edge flip condition), Cloth Encoding: u_ij, |u_ij|, x_ij, |x_ij| (mesh edge); x_ij, |x_ij| (world edge); n_i, velocity history (node), Cloth Dynamics PDE (second-order system), Dataset: FlagDynamic (cloth, dynamic mesh), Dataset: FlagSimple (cloth, static mesh) (+11 more)

### Community 9 - "Encoder-Processor Architecture"
Cohesion: 0.12
Nodes (18): Encoder Module, Encoder MLPs (epsilon_M, epsilon_W, epsilon_V), Generalization to Larger/Unseen Geometries and Parameters, Next-Step World-Space Velocity as Input for Kinematic Nodes, Kinematic Nodes (fixed or scripted motion), Latent Vector Size 128, LayerNorm on MLP Outputs, Mesh Edge Feature Encoding (u_ij, |u_ij|, x_ij, |x_ij|) (+10 more)

### Community 10 - "Bayesian Approx Methods"
Cohesion: 0.12
Nodes (17): Bayesian Model Average (BMA), Laplace Approximation (LA), LA, MFVI, SEns: Good Performance-Cost Tradeoff, Langevin Dynamics (LD), Monte Carlo Dropout (MCD), MCD Limitation: Epistemic Uncertainty Does Not Scale with Noise/Data Size, Markov Chain Monte Carlo (MCMC) Methods, Mean-Field Variational Inference (MFVI) (+9 more)

### Community 11 - "Decoder Architecture"
Cohesion: 0.15
Nodes (15): Decoder Module, Decoder MLP (delta_V), Mesh Edge Update Function f_M, World Edge Update Function f_W, Encode-Process-Decode Architecture, Forward-Euler Integrator (delta_t=1), GraphNet Blocks (Battaglia et al.), Reference: Graph Networks (Battaglia et al. 2018) (+7 more)

### Community 12 - "CQR Algorithm"
Cohesion: 0.22
Nodes (10): CONCEPT: Conformal Prediction — Distribution-Free Finite-Sample Coverage Guarantee, CQR Algorithm: Split data, fit quantile regressors q_lo/q_hi, compute conformity scores E_i = max(q_lo(X_i)-Y_i, Y_i-q_hi(X_i)), calibrate quantile, CQR Theorem 2: Asymmetric (Two-Tailed) Conformalization — independent left/right tail control, Split Conformal Prediction: Use held-out calibration set to compute empirical quantile Q_{1-alpha} of residuals for valid coverage, CQR Conformity Score: E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i)), CQR Theorem 1: Finite-sample, distribution-free coverage guarantee P{Y_n+1 in C(X_n+1)} >= 1 - alpha under exchangeability, Rationale: CQR over split conformal because fixed-width intervals waste coverage on low-variance regions; quantile regression adapts width to local heteroscedasticity, Conformalized Quantile Regression (Romano, Patterson, Candes, NeurIPS 2019) (+2 more)

### Community 13 - "Post-Training Calibration"
Cohesion: 0.29
Nodes (7): Calibration on Training Set Causes Overfitting, Post-Training Calibration Effective Even with 2-14 Points, CDF Modification Calibration (Kuleshov et al.), CRUDE Calibration Method, Model Misspecification Uncertainty Source, Post-Training Calibration Methods, Variance Re-weighting Calibration (Levi et al.)

### Community 14 - "GP and Physics-Informed UQ"
Cohesion: 0.67
Nodes (3): Gaussian Process + PI-GAN (GP+PI-GAN), Gaussian Process (GP) Regression, Uncertain Physics-Informed GAN (U-PI-GAN)

### Community 15 - "World Edges Rationale"
Cohesion: 1.0
Nodes (2): Rationale: World Edges Needed for Non-Local Mesh-Space Interactions (collision/contact), World-Space Computation (captures contact/collision)

### Community 16 - "Mesh Edges Rationale"
Cohesion: 1.0
Nodes (2): Mesh-Space Computation (approximates differential operators), Rationale: Mesh Edges Capture Rest-State and Material Internal Dynamics

### Community 17 - "Stochastic PDE Methods"
Cohesion: 1.0
Nodes (2): NNPC+ Architecture for Stochastic PDEs, Uncertain NN Polynomial Chaos (U-NNPC+)

### Community 18 - "Computational Cost Analysis"
Cohesion: 1.0
Nodes (2): Computational Time Comparison (CPU/GPU per Method), Performance vs Computational Cost Analysis

### Community 19 - "Adaptive Prediction Sets"
Cohesion: 1.0
Nodes (1): Adaptive Prediction Sets (APS)

### Community 20 - "Conformal Risk Control"
Cohesion: 1.0
Nodes (1): Conformal Risk Control

### Community 21 - "Cross-Validation Conformal"
Cohesion: 1.0
Nodes (1): Jackknife+ / CV+

### Community 22 - "Hyperelastic Materials"
Cohesion: 1.0
Nodes (1): Mooney-Rivlin Hyper-Elastic Model

### Community 23 - "Neural PDE Surrogates"
Cohesion: 1.0
Nodes (1): Neural-PDE Solver (FNO, U-Net, GNN) as Surrogate

### Community 24 - "Quantile Calibration"
Cohesion: 1.0
Nodes (1): Quantile Estimation from Calibration Scores

### Community 25 - "Cantilever Beam Dataset"
Cohesion: 1.0
Nodes (1): Coarse-Mesh Cantilever Beam Dataset

### Community 26 - "Mesh Coordinates"
Cohesion: 1.0
Nodes (1): Mesh-Space Coordinate (u_i)

### Community 27 - "World Coordinates"
Cohesion: 1.0
Nodes (1): World-Space Coordinate (x_i)

### Community 28 - "Time Series Datasets"
Cohesion: 1.0
Nodes (1): Dataset: 250-600 Time Steps per Trajectory

### Community 29 - "Training Hyperparameters"
Cohesion: 1.0
Nodes (1): Batch Size (1 for cloth, 2 for others)

### Community 30 - "Deterministic PDE Forward"
Cohesion: 1.0
Nodes (1): Forward Deterministic PDE Problem

### Community 31 - "Mixed Deterministic PDE"
Cohesion: 1.0
Nodes (1): Mixed Deterministic PDE Problem

### Community 32 - "Mixed Stochastic PDE"
Cohesion: 1.0
Nodes (1): Mixed Stochastic PDE Problem

### Community 33 - "Uncertainty Dispersion Metrics"
Cohesion: 1.0
Nodes (1): SDCV - Dispersion of Uncertainty Metric

## Knowledge Gaps
- **159 isolated node(s):** `Distribution-Free Uncertainty Quantification`, `Prediction Sets / Intervals`, `Marginal Coverage Guarantee`, `Adaptive Prediction Sets (APS)`, `Conformal Prediction Under Covariate Shift` (+154 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `World Edges Rationale`** (2 nodes): `Rationale: World Edges Needed for Non-Local Mesh-Space Interactions (collision/contact)`, `World-Space Computation (captures contact/collision)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mesh Edges Rationale`** (2 nodes): `Mesh-Space Computation (approximates differential operators)`, `Rationale: Mesh Edges Capture Rest-State and Material Internal Dynamics`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Stochastic PDE Methods`** (2 nodes): `NNPC+ Architecture for Stochastic PDEs`, `Uncertain NN Polynomial Chaos (U-NNPC+)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Computational Cost Analysis`** (2 nodes): `Computational Time Comparison (CPU/GPU per Method)`, `Performance vs Computational Cost Analysis`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Adaptive Prediction Sets`** (1 nodes): `Adaptive Prediction Sets (APS)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Conformal Risk Control`** (1 nodes): `Conformal Risk Control`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cross-Validation Conformal`** (1 nodes): `Jackknife+ / CV+`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Hyperelastic Materials`** (1 nodes): `Mooney-Rivlin Hyper-Elastic Model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Neural PDE Surrogates`** (1 nodes): `Neural-PDE Solver (FNO, U-Net, GNN) as Surrogate`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Quantile Calibration`** (1 nodes): `Quantile Estimation from Calibration Scores`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cantilever Beam Dataset`** (1 nodes): `Coarse-Mesh Cantilever Beam Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mesh Coordinates`** (1 nodes): `Mesh-Space Coordinate (u_i)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `World Coordinates`** (1 nodes): `World-Space Coordinate (x_i)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Time Series Datasets`** (1 nodes): `Dataset: 250-600 Time Steps per Trajectory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Training Hyperparameters`** (1 nodes): `Batch Size (1 for cloth, 2 for others)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deterministic PDE Forward`** (1 nodes): `Forward Deterministic PDE Problem`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mixed Deterministic PDE`** (1 nodes): `Mixed Deterministic PDE Problem`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mixed Stochastic PDE`** (1 nodes): `Mixed Stochastic PDE Problem`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Uncertainty Dispersion Metrics`** (1 nodes): `SDCV - Dispersion of Uncertainty Metric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Encode-Process-Decode Architecture` connect `Decoder Architecture` to `Encoder-Processor Architecture`, `MeshGraphNets Ablations`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `MeshGraphNets Framework` connect `MeshGraphNets Ablations` to `Encoder-Processor Architecture`, `Decoder Architecture`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `Encoder Module` connect `Encoder-Processor Architecture` to `Decoder Architecture`, `MeshGraphNets Datasets`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Psaros 2023: UQ in Scientific Machine Learning` (e.g. with `Conformal Prediction Not Covered (Noted as Gap)` and `CRPS Not Explicitly Used as Primary Metric`) actually correct?**
  _`Psaros 2023: UQ in Scientific Machine Learning` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Distribution-Free Uncertainty Quantification`, `Prediction Sets / Intervals`, `Marginal Coverage Guarantee` to the rest of the system?**
  _159 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CNN/DL Stress Surrogates` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._
- **Should `Conformal Prediction Theory` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._