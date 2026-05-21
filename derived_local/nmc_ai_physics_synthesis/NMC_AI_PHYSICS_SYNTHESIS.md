# NMC AI Physics Synthesis

## Scope

This report consolidates the Alek_Jiho NMC charge/discharge photometry analyses into one auditable view. It is generated from derived outputs on Isambard and should be treated as a synthesis of computational evidence, not as a manual curation substitute.

## Current Evidence Base

- Multi-cycle ROI/echem rows: 52
- Distinct cycles in coupled ROI/echem table: 11
- Event-reference cycles: 4
- Calibrated front-QC ROI rows: 12
- Manual-QC pending front ROIs: 12

## Main Findings

- Persistence is the strongest raw next-frame baseline; DMD/velocity/learned residual experiments are most useful as residual and latent descriptors.
- ROI event/control optical differences survive event-reference-cycle centering, especially cumulative normalized change, first-last decorrelation, latent net displacement, high-fraction growth, and ROI mean trend.
- Frame count and protocol-block position strongly couple to ROI dynamics, so echem/protocol context must be a model covariate and a guardrail.
- Cycles 86 and 116 remain the strongest synchronized event-timing regimes; cycles 60 and 156 provide stronger single-particle morphology/latent-movement examples.
- Apparent front tracking currently indicates optical-front contraction/loss more than clean expanding diffusion fronts.

## Model Readout

- Strict no-selection-QC random forest: ROC-AUC 0.651, balanced accuracy 0.573.
- Strict no-selection-QC logistic: ROC-AUC 0.625, balanced accuracy 0.562.
- All physics plus QC random forest: ROC-AUC 0.797, balanced accuracy 0.688.

Interpretation: the stricter model is above random but not deployable. QC/acquisition features improve apparent performance and should be treated as leakage-sensitive guardrails.

## Top Within-Reference Optical Separations

- cumulative_abs_norm_change_centered_by_reference: event-control 0.010, p=2.738e-05
- first_last_corr_centered_by_reference: event-control -0.027, p=4.710e-04
- latent_net_displacement_centered_by_reference: event-control 0.846, p=6.186e-04
- high_fraction_delta_centered_by_reference: event-control 0.048, p=6.840e-04
- roi_norm_mean_slope_centered_by_reference: event-control 5.536e-05, p=7.076e-04
- roi_norm_mean_delta_centered_by_reference: event-control 0.010, p=8.084e-04
- low_fraction_delta_centered_by_reference: event-control -0.046, p=9.843e-04
- centroid_path_px_centered_by_reference: event-control -1.512, p=0.003

## Top ROI/Echem Or Protocol Couplings

- n_frames_percentile vs latent_net_displacement: rho=0.776, p=1.340e-11, n=52
- n_frames_percentile vs cumulative_abs_norm_change: rho=0.756, p=9.501e-11, n=52
- n_frames_percentile vs first_last_corr: rho=-0.700, p=7.504e-09, n=52
- n_frames_percentile vs latent_path_length: rho=0.697, p=9.707e-09, n=52
- n_frames_percentile vs latent_mean_step: rho=0.697, p=9.707e-09, n=52
- cycles_from_block_start vs latent_mean_step: rho=0.609, p=1.635e-06, n=52
- cycles_from_block_start vs latent_path_length: rho=0.609, p=1.635e-06, n=52
- n_frames_percentile vs dmd_minus_persistence_mse: rho=0.595, p=3.280e-06, n=52

## Highest-Priority ROI Candidates

- cycle62_rank3_obj9 (control, cycle 62): rollout/mobility score 3.058, latent path 14.524, first-last corr 0.824
- cycle156_rank5_obj4 (event, cycle 156): rollout/mobility score 3.038, latent path 19.837, first-last corr 0.878
- cycle156_rank7_obj27 (event, cycle 156): rollout/mobility score 3.019, latent path 22.511, first-last corr 0.853
- cycle156_rank2_obj2 (event, cycle 156): rollout/mobility score 2.865, latent path 19.329, first-last corr 0.911
- cycle60_rank3_obj9 (event, cycle 60): rollout/mobility score 2.846, latent path 22.630, first-last corr 0.869
- cycle60_rank6_obj26 (event, cycle 60): rollout/mobility score 2.615, latent path 16.147, first-last corr 0.834
- cycle60_rank4_obj5 (event, cycle 60): rollout/mobility score 2.596, latent path 14.736, first-last corr 0.844
- cycle156_rank8_obj10 (event, cycle 156): rollout/mobility score 2.519, latent path 19.657, first-last corr 0.902
- cycle86_rank4_obj9 (event, cycle 86): rollout/mobility score 2.327, latent path 18.205, first-last corr 0.976
- cycle60_rank2_obj2 (event, cycle 60): rollout/mobility score 2.308, latent path 19.447, first-last corr 0.904

## Completion Audit

- Implement paper-inspired agentic workflows in separate Isambard folders: implemented. Evidence: agentic_research outputs plus derived tier1/tier2/tier3 experiment folders were created on Isambard and compact outputs synced locally. Limitation: The synthesis script summarizes the outputs but does not rerun the original literature analysis.
- Focus on Alek_Jiho NMC degradation dataset on Isambard: implemented. Evidence: Synthesis reads Isambard derived directory with 52 ROI rows, 11 cycles, and 4 event-reference cycles. Limitation: The current multi-cycle ROI cohort is selected around event/reference cycles, not every raw video in the full dataset.
- Next-frame prediction and rollout: implemented_with_guardrail. Evidence: Persistence, velocity, low-rank DMD, PCA latent trajectories, PCA-ridge, and residual-CNN guardrails were run. Persistence is best across cycles: True. Limitation: Learned/full rollout models do not yet beat persistence robustly; use residuals and latent paths as descriptors rather than claiming superior prediction.
- Track phase-boundary movement: implemented_as_proxy. Evidence: Front/phase mobility descriptors and selected-front tracking exist; calibration table has 12 ROI rows. Limitation: Front masks are automatic; all selected calibrated front ROIs remain manual-QC pending.
- Extract diffusion coefficients: partial_proxy_only. Evidence: Provisional 0.096 um/px calibration and apparent diffusion proxies were computed for 12 front ROIs. Limitation: The values are apparent optical-front contraction/expansion proxies, not validated diffusion coefficients.
- Identify degradation modes: implemented_as_hypothesis_ranking. Evidence: Joint physics/rollout/echem degradation mode tables and multi-cycle ROI mobility rankings exist. Limitation: Modes are unsupervised/automatic and tied to the selected ROI cohort.
- Correlate degradation with cycles, particle regions, and echem/protocol context: implemented_with_guardrail. Evidence: Multi-cycle ROI echem coupling found strong frame-count/protocol correlations and within-reference event/control optical shifts. Limitation: Protocol/frame-count confounding is strong; correlations are not causal physics.
- Keep objectives and observations updated: implemented. Evidence: OBJECTIVES_OBSERVATIONS.md contains chronological experiment summaries and guardrails. Limitation: It is long and narrative; the new synthesis is the compact index.
- Keep GitHub updated: local_commits_ready_remote_unverified. Evidence: Local git commits exist through the latest echem coupling and this synthesis can be committed. Limitation: Remote push must be verified separately because prior pushes were blocked by approval/network state.

## Recommended Next Experiments

1. Manual QC the selected front/particle ROI previews and update the manifest with accepted/rejected labels.
2. Expand the ROI cohort across more cycles after QC to reduce event-reference and protocol confounding.
3. Fit echem/protocol-conditioned rollout and event-ranking models, reporting persistence-normalized residuals and uncertainty coverage.
4. Recompute apparent diffusion/front-motion proxies only on QC-accepted fronts with confirmed spatial/time calibration.
5. Convert the top ROI candidates into a labeled degradation-mode benchmark for future self-supervised video models.

## Guardrail

The current outputs support a physics-extraction workflow and ranked hypotheses. They do not yet support a claim of calibrated diffusion coefficients or a deployable automated degradation detector.
