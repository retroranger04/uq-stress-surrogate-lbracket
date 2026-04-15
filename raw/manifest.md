# raw/manifest.md — UQ Stress Surrogate L-Bracket Corpus
*Generated: 2026-04-15 by Sonnet corpus-curator agent.*
*Curator-owned. Do NOT edit from orchestrator sessions.*

**Corpus size:** 14 files acquired (12 PDFs + 2 markdown reference docs).
**Skipped:** 1 paper (paywalled, no preprint) — see Appendix.
**Cap:** 15 of 18 items used (including 2 reference docs).

---

## BUCKET A — Neural Surrogates for Structural/Stress Prediction

### nie2020stress
- **Full citation:** Nie Z., Jiang H., Kara L.B. "Stress Field Prediction in Cantilevered Structures Using Convolutional Neural Networks." *J. Comput. Inf. Sci. Eng.* 20(1):011002, ASME, 2020.
- **Local file:** `raw/papers/nie2020stress.pdf` (8.5 MB, arXiv:1808.08914)
- **Source:** arXiv preprint
- **Year:** 2020 | **Venue:** JCISE (ASME) | **~Citations:** ~200
- **Code:** ✓ github.com/zhenguonie/stress_net | **Affiliation:** CMU VDEL (Kara)
- **Cite-worthy:** Yes (References section)
- **Role in project:** Foundational CNN baseline for 2D von Mises stress field prediction on cantilevered geometries. Establishes the task framing we inherit. Our GNN-based approach contrasts with their fully-convolutional fixed-grid approach.

### jiang2021stressgan
- **Full citation:** Jiang H., Nie Z., Yeo R., Barati Farimani A., Kara L.B. "StressGAN: A Generative Deep Learning Model for Two-Dimensional Stress Distribution Prediction." *J. Appl. Mech.* 88(5):051005, ASME, 2021.
- **Local file:** `raw/papers/jiang2021stressgan.pdf` (2.1 MB, arXiv:2006.11376)
- **Source:** arXiv preprint (DOI: 10.1115/1.4049805)
- **Year:** 2021 | **Venue:** JAM (ASME) | **~Citations:** ~100
- **Code:** — | **Affiliation:** CMU VDEL (Kara)
- **Cite-worthy:** Yes (References section)
- **Role in project:** GAN-based extension showing richer output distributions for stress prediction. Positions our work: we use GNN + calibrated uncertainty rather than a GAN architecture, providing coverage guarantees not available from GAN outputs.

### maurizi2022gnn
- **Full citation:** Maurizi M., Gao C., Berto F. "Predicting Stress, Strain and Deformation Fields in Materials and Structures with Graph Neural Networks." *Sci. Rep.* 12:21834, 2022.
- **Local file:** `raw/papers/maurizi2022gnn.pdf` (5.6 MB, publisher PDF from Nature; also arXiv:2205.06675)
- **Source:** Nature publisher PDF (open access, CC BY 4.0; DOI: 10.1038/s41598-022-26424-3)
- **Year:** 2022 | **Venue:** Scientific Reports (Nature) | **~Citations:** ~120
- **Code:** ✓ github.com/marcomau06/GNNs_fields_prediction | **Affiliation:** Politecnico di Milano / DTU
- **Cite-worthy:** Yes (References section)
- **Role in project:** **Closest methodological cousin.** GNN on unstructured mesh predicts stress, strain, and deformation for diverse material systems including composites and lattice metamaterials. We adopt the same mesh-to-graph paradigm and extend it with a UQ layer (Deep Ensembles + CQR).

### bhaduri2022stress
- **Full citation:** Bhaduri A., Gupta A., Graham-Brady L. "Stress Field Prediction in Fiber-Reinforced Composite Materials Using a Deep Learning Approach." *Composites Part B: Engineering* 238:109879, Elsevier, 2022.
- **Local file:** `raw/papers/bhaduri2022stress.pdf` (14 MB, arXiv:2111.05271)
- **Source:** arXiv preprint (DOI: 10.1016/j.compositesb.2022.109879)
- **Year:** 2022 | **Venue:** Composites Part B (Elsevier) | **~Citations:** ~80
- **Code:** — | **Affiliation:** Johns Hopkins (Graham-Brady group)
- **Cite-worthy:** Yes (References section)
- **Role in project:** CNN stress prediction in fiber-reinforced composites. Different geometry (fiber RVE vs. L-bracket) but same von Mises stress task framing. Cited in Related Work to bound the domain of DL surrogates for stress prediction prior to GNN approaches.

### gladstone2024gnn
- **Full citation:** Gladstone R.J., Rahmani H., Suryakumar V., Meidani H., D'Elia M., Zareei A. "Mesh-Based GNN Surrogates for Time-Independent PDEs." *Sci. Rep.* 14:3394, 2024.
- **Local file:** `raw/papers/gladstone2024gnn.pdf` (2.8 MB, publisher PDF from Nature)
- **Source:** Nature publisher PDF (open access; DOI: 10.1038/s41598-024-53185-y)
- **Year:** 2024 | **Venue:** Scientific Reports (Nature) | **~Citations:** ~30
- **Code:** — | **Affiliation:** UIUC / Sandia National Laboratories
- **Cite-worthy:** Yes (References section)
- **Role in project:** 2024 direct prior art. Develops edge-augmented GNN and multi-GNN architectures that outperform vanilla MeshGraphNets on time-independent solid mechanics with rotation/translation invariance. Directly informs our architecture choices and provides a recent comparison point.

---

## BUCKET B — Mesh-GNN Methodology

### pfaff2021meshgraphnets
- **Full citation:** Pfaff T., Fortunato M., Sanchez-Gonzalez A., Battaglia P.W. "Learning Mesh-Based Simulation with Graph Networks." In *ICLR 2021* (Spotlight), 2021.
- **Local file:** `raw/papers/pfaff2021meshgraphnets.pdf` (13 MB, arXiv:2010.03409)
- **Source:** arXiv preprint
- **Year:** 2021 | **Venue:** ICLR 2021 (Spotlight) | **~Citations:** ~2000
- **Code:** ✓ deepmind/deepmind-research | **Affiliation:** DeepMind
- **Cite-worthy:** Yes (References section)
- **Role in project:** **Non-negotiable foundational paper.** Defines the mesh-to-graph encoder, multi-scale processor, and node/edge decoder architecture that every GNN surrogate in this corpus builds on. Introduces world-edge vs. mesh-edge node type distinction, which we use for boundary condition encoding.

---

## BUCKET C — Foundational UQ

### lakshminarayanan2017ensembles
- **Full citation:** Lakshminarayanan B., Pritzel A., Blundell C. "Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles." In *NeurIPS 2017*, pp. 6402–6413, 2017.
- **Local file:** `raw/papers/lakshminarayanan2017ensembles.pdf` (1.5 MB, arXiv:1612.01474)
- **Source:** arXiv preprint
- **Year:** 2017 | **Venue:** NeurIPS 2017 | **~Citations:** ~8500
- **Code:** — | **Affiliation:** DeepMind
- **Cite-worthy:** Yes (References section)
- **Role in project:** Defines the Deep Ensembles UQ method we implement as the primary UQ layer. Key recipe: M=5 independently-initialized members, proper scoring rule (NLL) training, randomised training data order. We cite this for ensemble design choices in Methods §Deep Ensembles.

### romano2019cqr
- **Full citation:** Romano Y., Patterson E., Candès E.J. "Conformalized Quantile Regression." In *NeurIPS 2019*, pp. 3538–3548, 2019.
- **Local file:** `raw/papers/romano2019cqr.pdf` (1.4 MB, arXiv:1905.03222)
- **Source:** arXiv preprint
- **Year:** 2019 | **Venue:** NeurIPS 2019 | **~Citations:** ~750
- **Code:** — | **Affiliation:** Stanford
- **Cite-worthy:** Yes (References section)
- **Role in project:** Defines the CQR algorithm (pinball loss + conformal calibration of quantile regression intervals). We implement CQR as our calibration layer on top of Deep Ensembles. Cited in Methods §CQR Calibration for the loss function, calibration procedure, and finite-sample marginal coverage theorem.

### angelopoulos2023conformal
- **Full citation:** Angelopoulos A.N., Bates S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *Foundations and Trends in Machine Learning* 16(4):494–591, 2023.
- **Local file:** `raw/papers/angelopoulos2023conformal.pdf` (5.1 MB, arXiv:2107.07511)
- **Source:** arXiv preprint (DOI: 10.1561/2200000101)
- **Year:** 2023 | **Venue:** Foundations and Trends in ML (Now Publishers) | **~Citations:** ~836
- **Code:** ✓ github.com/aangelopoulos/conformal-prediction | **Affiliation:** UC Berkeley
- **Cite-worthy:** Yes (References section)
- **Role in project:** Comprehensive reference for conformal prediction theory. Cited to ground all conformal claims: marginal coverage guarantee (Definition 2.1), exchangeability assumption, split-conformal protocol, and relationship between CQR and full conformal.

---

## BUCKET D — UQ Applied to Engineering / Physics Surrogates

### psaros2023uq
- **Full citation:** Psaros A.F., Meng X., Zou Z., Guo L., Karniadakis G.E. "Uncertainty Quantification in Scientific Machine Learning: Methods, Metrics, and Comparisons." *J. Comput. Phys.* 477:111902, Elsevier, 2023.
- **Local file:** `raw/papers/psaros2023uq.pdf` (21 MB, arXiv:2201.07766)
- **Source:** arXiv preprint (DOI: 10.1016/j.jcp.2022.111902)
- **Year:** 2023 | **Venue:** Journal of Computational Physics (Elsevier) | **~Citations:** ~150
- **Code:** — | **Affiliation:** Brown University (Karniadakis group)
- **Cite-worthy:** Yes (References section)
- **Role in project:** Comprehensive SciML UQ survey. Cited in Introduction and Related Work to frame the broader challenge of UQ in neural surrogates and to position Deep Ensembles + CQR within the taxonomy of methods (ensemble methods, Bayesian approximations, dropout, conformal).

### olivier2021bayesian ← **SKIPPED — PAYWALLED**
- **Full citation:** Olivier A., Shields M.D., Graham-Brady L. "Bayesian Neural Networks for Uncertainty Quantification in Data-Driven Materials Modeling." *Comput. Methods Appl. Mech. Eng.* 386:114079, Elsevier, 2021.
- **Local file:** NOT ACQUIRED
- **Reason:** No open-access preprint (arXiv, engrXiv, or institutional repo) found after two searches. Publisher version behind ScienceDirect paywall.
- **BibTeX DOI:** 10.1016/j.cma.2021.114079 (placeholder entry in bibliography.bib — add PDF manually if access obtained)
- **Note:** Still included in bibliography.bib as a commented-out placeholder. If Arpit has institutional access, add the PDF as `raw/papers/olivier2021bayesian.pdf` and uncomment the BibTeX entry.

### gopakumar2024conformal
- **Full citation:** Gopakumar V., Gray A., Oskarsson J., Zanisi L., Pamela S., Giles D., Kusner M.J., Deisenroth M.P. "Uncertainty Quantification of Surrogate Models Using Conformal Prediction." *Machine Learning: Science and Technology* 7(1):015025, IOP Publishing, 2026.
- **Local file:** `raw/papers/gopakumar2024conformal.pdf` (6.9 MB, arXiv:2408.09881)
- **Source:** arXiv preprint (DOI: 10.1088/2632-2153/ae2e7b; open access CC BY 4.0, published Feb 2026)
- **Year:** 2026 (BibTeX year field) | **arXiv:** 2024 | **Venue:** Machine Learning: Science and Technology (IOP) | **~Citations:** ~25
- **Code:** — | **Affiliation:** UCL / UKAEA (Deisenroth group)
- **Cite-worthy:** Yes (References section)
- **Role in project:** **Strongest 2024–2026 addition.** Applies conformal prediction to PDE surrogate models with marginal coverage guarantee, model-agnostically across FNO, U-Net, GNN, ViT. Our work specialises this to GNN-on-mesh structural surrogates. Cited in Methods §CQR Calibration and Related Work.

### pasparakis2024bayesian
- **Full citation:** Pasparakis G.D., Graham-Brady L., Shields M.D. "Bayesian Neural Networks for Predicting Uncertainty in Full-Field Material Response." *Comput. Methods Appl. Mech. Eng.* 432:117409, Elsevier, 2024.
- **Local file:** `raw/papers/pasparakis2024bayesian.pdf` (3.1 MB, arXiv:2406.14838)
- **Source:** arXiv preprint (DOI: 10.1016/j.cma.2024.117409)
- **Year:** 2024 | **Venue:** CMAME (Elsevier) | **~Citations:** ~15
- **Code:** — | **Affiliation:** Johns Hopkins (Graham-Brady / Shields group)
- **Cite-worthy:** Yes (References section)
- **Role in project:** 2024 direct comparison. BNN-U-Net for full-field von Mises stress prediction with UQ in composite/polycrystal microstructures. Positioned in Related Work as the BNN baseline our Deep Ensembles + CQR approach competes with: we offer stronger calibration guarantees (conformal coverage) and no prior specification requirement.

---

## BUCKET E — Reference Docs (cite in Methods only, not References section)

### dokken_fenicsx
- **Full citation:** Dokken J.S. "The FEniCSx Tutorial." Chapter: The Equations of Linear Elasticity. 2023. https://jsdokken.com/dolfinx-tutorial/ (CC BY 4.0)
- **Local file:** `raw/papers/dokken_fenicsx.md` (4.4 KB, markdown)
- **Source:** Web fetch (jsdokken.com) + manual enrichment with project-specific notes
- **Cite-worthy:** No (reference-only, cite in Methods §FEA Data Pipeline)
- **Role in project:** FEA data generation grounding. Covers DOLFINx variational formulation, Lamé parameters, weak form, von Mises stress postprocessing, parametric mesh workflow.

### pyg_docs
- **Full citation:** PyG Team. "Creating Message Passing Networks." PyTorch Geometric v2.6, 2024. https://pytorch-geometric.readthedocs.io/en/2.6.0/notes/create_gnn.html
- **Local file:** `raw/papers/pyg_docs.md` (5.6 KB, markdown)
- **Source:** GitHub RST source (raw.githubusercontent.com) + enrichment with project-specific GNN patterns
- **Cite-worthy:** No (reference-only, cite in Methods §GNN Architecture)
- **Role in project:** MessagePassing class API, message/aggregate/update pattern, edge feature handling. Grounding for GNN implementation decisions.

---

## Appendix — Dropped Candidates

| Paper | Why Dropped |
|-------|-------------|
| Zhu Z., Zabaras N., Koutsourelakis P., Perdikaris P. "Physics-Constrained Deep Learning for High-dimensional Surrogate Modeling and UQ without Labeled Data." JCP 394:56–81, 2019. arXiv:1901.06314 | Methodology mismatch: physics-constrained training without labeled data is structurally different from our data-driven surrogate with post-hoc conformal calibration. Psaros 2023 survey covers this branch more comprehensively. |
| [Anonymous] 2025 — "Physics-informed machine learning for near real-time stress prediction on a structural component: Application for landing gears." Eng. Appl. Artif. Intell., 2025. | PINN-based inference, no UQ layer, no GNN. Landing gear geometry is superficially similar (structural component) but technically distinct. Architecture and contribution are orthogonal to this project. |
| January 2026 landing-gear bracket paper | Not located. No matching paper found in arXiv or journal sweeps for January 2026. |
