# NMC AI Physics Synthesis

## Scope

This report consolidates the Alek_Jiho NMC charge/discharge photometry analyses into one auditable view. It is generated from derived outputs on Isambard and should be treated as a synthesis of computational evidence, not as a manual curation substitute.

## Current Evidence Base

- Multi-cycle ROI/echem rows: 52
- Distinct cycles in coupled ROI/echem table: 11
- Event-reference cycles: 4
- Calibrated front-QC ROI rows: 12
- Manual-QC pending front ROIs: 12
- ROI/front QC package candidates: 24
- Control-balanced front QC candidates: 40
- Residual physics mode clusters: 4

## Main Findings

- Persistence is the strongest raw next-frame baseline; DMD/velocity/learned residual experiments are most useful as residual and latent descriptors.
- Prefix-only cropped ROI forecasts predict later front-direction residual class best: random_forest at prefix 0.500 gives AUC 0.726, while permutation-null support is strongest for the front-positive residual target.
- ROI event/control optical differences survive event-reference-cycle centering, especially cumulative normalized change, first-last decorrelation, latent net displacement, high-fraction growth, and ROI mean trend.
- Frame count and protocol-block position strongly couple to ROI dynamics, so echem/protocol context must be a model covariate and a guardrail.
- After residualizing available protocol/echem covariates and event-reference fixed effects, event/control separation remains in ROI mean delta, high-fraction delta, first-last correlation, cumulative change, DMD residual, and latent displacement.
- Cycles 86 and 116 remain the strongest synchronized event-timing regimes; cycles 60 and 156 provide stronger single-particle morphology/latent-movement examples.
- Apparent front tracking currently indicates optical-front contraction/loss more than clean expanding diffusion fronts.
- Threshold sweeps show robust event/control differences in phase-fraction slope, but radius-derived diffusion proxies remain weaker and threshold-sensitive.
- Protocol-conditioned front residuals preserve phase-slope sign consistency, but not front-magnitude or diffusion-proxy separability.
- Automatic front-QC sensitivity keeps the positive phase-front residual in 5 strata: all_front_rois, complete_threshold_sweep, q70_phase_ci_excludes_zero, q70_phase_ci_positive, review_panel_selected; review-panel diffusion proxy differences are selection-sensitive and not calibrated transport.
- Protocol-adjusted residual mode taxonomy chooses k=4; its most event-enriched mode is optical_brightening_decorrelating_rollout_hard_front_positive with event fraction 0.846 and Fisher p=0.003.
- A QC review packet prioritizes 30 ROI/front candidates, and a control-balanced front package adds 24 control candidates for manual accept/reject review before publication-scale diffusion claims.

## Model Readout

- Strict no-selection-QC random forest: ROC-AUC 0.651, balanced accuracy 0.573.
- Strict no-selection-QC logistic: ROC-AUC 0.625, balanced accuracy 0.562.
- All physics plus QC random forest: ROC-AUC 0.797, balanced accuracy 0.688.
- Protocol-conditioned residual logistic: ROC-AUC 0.672, balanced accuracy 0.682.
- Protocol-conditioned front-residual logistic: ROC-AUC 0.453, balanced accuracy 0.312.

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

## Prefix-Only ROI Forecasts

- front_positive_residual_binary prefix_only random_forest f=0.500: AUC 0.726, balanced accuracy 0.512
- front_positive_residual_binary prefix_only random_forest f=0.750: AUC 0.712, balanced accuracy 0.547
- front_positive_residual_binary prefix_only logistic_l2 f=0.750: AUC 0.687, balanced accuracy 0.608
- front_positive_residual_binary prefix_only random_forest f=0.250: AUC 0.654, balanced accuracy 0.586
- front_positive_residual_binary prefix_only logistic_l2 f=0.250: AUC 0.648, balanced accuracy 0.554
- front_positive_residual_binary prefix_only logistic_l2 f=0.500: AUC 0.635, balanced accuracy 0.624
- front_positive_residual_binary prefix_plus_context random_forest f=0.500: AUC 0.613, balanced accuracy 0.528
- front_positive_residual_binary prefix_plus_context random_forest f=0.750: AUC 0.601, balanced accuracy 0.531
- front_positive_residual_binary prefix_plus_context random_forest f=0.250: AUC 0.567, balanced accuracy 0.530
- front_positive_residual_binary prefix_plus_context logistic_l2 f=0.250: AUC 0.392, balanced accuracy 0.439
- Null check is_event_roi f=0.250: observed AUC 0.573, null p95 0.687, empirical p=0.285
- Null check is_event_enriched_mode f=0.750: observed AUC 0.636, null p95 0.713, empirical p=0.184
- Null check front_positive_residual_binary f=0.750: observed AUC 0.687, null p95 0.656, empirical p=0.026
- Regression mode_review_priority prefix_plus_context random_forest f=0.500: MAE ratio vs median baseline 0.419, rho 0.374
- Regression mode_review_priority prefix_plus_context random_forest f=0.250: MAE ratio vs median baseline 0.426, rho 0.326
- Regression mode_review_priority prefix_only random_forest f=0.250: MAE ratio vs median baseline 0.429, rho 0.322
- Regression mode_review_priority prefix_only random_forest f=0.500: MAE ratio vs median baseline 0.431, rho 0.363
- Guardrail: Prefix-only ROI forecasts use cropped particle-region sequences, but the cohort is small and selected around event-reference cycles; treat results as physics-signal triage, not a deployable predictor.

## Top Protocol-Conditioned Event Effects

- roi_norm_mean_delta_protocol_residual: event-control 0.003, p=4.424e-05
- high_fraction_delta_protocol_residual: event-control 0.015, p=8.238e-05
- low_fraction_delta_protocol_residual: event-control -0.015, p=4.707e-04
- first_last_corr_protocol_residual: event-control -0.018, p=0.019
- cumulative_abs_norm_change_protocol_residual: event-control 0.004, p=0.019
- dmd_minus_persistence_mse_protocol_residual: event-control 4.515e-04, p=0.041
- latent_net_displacement_protocol_residual: event-control 0.377, p=0.043
- low_rank_dmd_mse_protocol_residual: event-control 4.593e-04, p=0.046

## Threshold-Robust Front Readout

- phase_slope_median_per_s: event-control 1.094e-06, p=0.017
- phase_slope_abs_median_per_s: event-control 7.931e-07, p=0.031
- threshold_robust_phase_score: event-control 0.156, p=0.041
- phase_slope_positive_fraction: event-control 0.139, p=0.041
- radius2_slope_median_px2_per_s: event-control 5.002e-04, p=0.259
- diffusion_proxy_median_um2_per_s: event-control 1.152e-06, p=0.259
- diffusion_proxy_abs_median_um2_per_s: event-control -4.722e-07, p=0.680
- threshold_robust_diffusion_score: event-control 0.011, p=0.993
- By-event strongest cases include cycle 156 phase-slope separation and cycle 86 radius/diffusion-proxy differences; all remain apparent optical proxies.

## Protocol-Conditioned Front Effects

- phase_slope_positive_fraction_protocol_residual: event-control 0.064, p=8.252e-04
- phase_slope_negative_fraction_protocol_residual: event-control -0.064, p=8.252e-04
- threshold_robust_phase_score_protocol_residual: event-control 0.090, p=0.393
- diffusion_proxy_iqr_um2_per_s_protocol_residual: event-control -3.988e-07, p=0.404
- radius2_slope_iqr_px2_per_s_protocol_residual: event-control -1.731e-04, p=0.404
- phase_slope_iqr_per_s_protocol_residual: event-control -9.686e-08, p=0.414
- radius2_slope_negative_fraction_protocol_residual: event-control -0.099, p=0.468
- radius2_slope_positive_fraction_protocol_residual: event-control 0.099, p=0.468
- Conditioning preserves phase-slope sign consistency but weakens magnitude and diffusion-proxy effects; front residuals are not a standalone detector.

## Front QC Sensitivity

- all_front_rois: n=52, event/control 24/28, phase-CI-positive fraction 0.500
- complete_threshold_sweep: n=52, event/control 24/28, phase-CI-positive fraction 0.500
- front_quality_top_half: n=26, event/control 20/6, phase-CI-positive fraction 0.577
- front_quality_top_quartile: n=13, event/control 8/5, phase-CI-positive fraction 0.462
- q70_phase_ci_excludes_zero: n=30, event/control 18/12, phase-CI-positive fraction 0.867
- q70_phase_ci_positive: n=26, event/control 18/8, phase-CI-positive fraction 1.000
- review_panel_selected: n=24, event/control 18/6, phase-CI-positive fraction 0.750
- review_panel_not_fragmented: n=7, event/control 7/0, phase-CI-positive fraction 0.857
- review_panel_no_auto_flags: n=5, event/control 5/0, phase-CI-positive fraction 1.000
- all_front_rois phase-sign residual: median event-control 0.047, bootstrap p05 0.034, MW p=8.252e-04, permutation p=0.047
- complete_threshold_sweep phase-sign residual: median event-control 0.047, bootstrap p05 0.034, MW p=8.252e-04, permutation p=0.042
- front_quality_top_half phase-sign residual: median event-control 0.093, bootstrap p05 -0.051, MW p=0.116, permutation p=0.024
- front_quality_top_quartile phase-sign residual: median event-control 0.139, bootstrap p05 -0.038, MW p=0.158, permutation p=0.036
- q70_phase_ci_excludes_zero phase-sign residual: median event-control 0.088, bootstrap p05 0.047, MW p=1.065e-04, permutation p=9.995e-04
- Diffusion-proxy separations do not remain globally robust; review-panel diffusion differences are selection-biased and still require manual QC.
- Control-balanced QC augmentation selects 40 ROIs (16 event / 24 control), with non-fragmented counts {'control': 4, 'event': 6}.

## Residual Physics Mode Taxonomy

- Selected k=4 with silhouette=0.204 and mean seed-stability ARI=0.935.
- optical_brightening_decorrelating_rollout_hard_front_positive: n=13, event fraction 0.846, p=0.003, cycles 60.0;156.0;62.0;86.0
- optical_loss_rollout_hard: n=5, event fraction 0.400, p=1.000, cycles 157.0;156.0;62.0
- near_baseline_or_context_like: n=24, event fraction 0.333, p=0.102, cycles 86.0;58.0;118.0;116.0
- front_negative_high_apparent_front_proxy: n=10, event fraction 0.300, p=0.309, cycles 88.0;90.0;116.0;60.0
- Guardrail: Mode labels are protocol-adjusted computational hypotheses from automatic ROI candidates; they require QC labels before being treated as biological/material degradation modes.

## Cycle/Region Mode Context

- cycle 60: n=6, event-enriched mode fraction=0.833, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;front_negative_high_apparent_front_proxy
- cycle 156: n=6, event-enriched mode fraction=0.667, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;optical_loss_rollout_hard
- cycle 62: n=4, event-enriched mode fraction=0.250, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;optical_loss_rollout_hard;front_negative_high_apparent_front_proxy
- cycle 158: n=4, event-enriched mode fraction=0.250, top modes=near_baseline_or_context_like;optical_brightening_decorrelating_rollout_hard_front_positive;front_negative_high_apparent_front_proxy
- cycle 86: n=6, event-enriched mode fraction=0.167, top modes=near_baseline_or_context_like;optical_brightening_decorrelating_rollout_hard_front_positive
- cycle 116: n=6, event-enriched mode fraction=0.167, top modes=near_baseline_or_context_like;front_negative_high_apparent_front_proxy;optical_brightening_decorrelating_rollout_hard_front_positive
- region x2_y3: n=15, event-enriched mode fraction=0.467, event fraction=0.600
- region x1_y1: n=3, event-enriched mode fraction=0.333, event fraction=0.333
- region x1_y2: n=13, event-enriched mode fraction=0.308, event fraction=0.308
- region x3_y1: n=15, event-enriched mode fraction=0.067, event fraction=0.400
- Context-only leave-cycle-out classifier: pooled ROC-AUC 0.429, pooled balanced accuracy 0.433; descriptive context is not a standalone detector.

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

## Residual-Mode Review Priorities

- cycle60_rank3_obj9 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.673
- cycle60_rank6_obj26 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.596
- cycle60_rank4_obj5 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.481
- cycle62_rank3_obj9 (control, cycle 62): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.308
- cycle156_rank7_obj27 (event, cycle 156): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.260
- cycle157_rank2_obj2 (control, cycle 157): optical_loss_rollout_hard, priority 3.106
- cycle156_rank5_obj4 (event, cycle 156): optical_brightening_decorrelating_rollout_hard_front_positive, priority 2.933
- cycle62_rank4_obj1 (control, cycle 62): optical_loss_rollout_hard, priority 2.923
- cycle60_rank2_obj2 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 2.923
- cycle58_rank3_obj9 (control, cycle 58): near_baseline_or_context_like, priority 2.913

## Completion Audit

- Implement paper-inspired agentic workflows in separate Isambard folders: implemented. Evidence: agentic_research outputs plus derived tier1/tier2/tier3 experiment folders were created on Isambard and compact outputs synced locally. Limitation: The synthesis script summarizes the outputs but does not rerun the original literature analysis.
- Focus on Alek_Jiho NMC degradation dataset on Isambard: implemented. Evidence: Synthesis reads Isambard derived directory with 52 ROI rows, 11 cycles, and 4 event-reference cycles. Limitation: The current multi-cycle ROI cohort is selected around event/reference cycles, not every raw video in the full dataset.
- Next-frame prediction and rollout: implemented_with_guardrail. Evidence: Persistence, velocity, low-rank DMD, PCA latent trajectories, PCA-ridge, residual-CNN guardrails, and prefix-only ROI forecasts were run. Persistence is best across raw pixel rollouts: True; best prefix classifier target is front_positive_residual_binary with AUC 0.726. Limitation: Learned/full rollout models do not yet beat persistence robustly; use residuals, latent paths, and prefix forecasts as physics descriptors rather than claiming superior pixel prediction.
- Track phase-boundary movement: implemented_as_proxy. Evidence: Front/phase mobility descriptors, selected-front tracking, and threshold-robust sweeps exist; threshold sweep covers 52 ROI rows. Limitation: Front masks are automatic; after protocol/echem conditioning, front-direction sign consistency survives more strongly than front-magnitude metrics and is robust in 5 automatic QC strata.
- Extract diffusion coefficients: partial_proxy_only. Evidence: Provisional 0.096 um/px apparent diffusion proxies were computed and stress-tested across 7 thresholds with bootstrap slopes. Limitation: Global threshold-robust phase slopes separate event/control ROIs, but QC-stratified diffusion proxies are inconsistent and conditioned diffusion-proxy residuals remain non-significant.
- Identify degradation modes: implemented_as_hypothesis_ranking. Evidence: Joint physics/rollout/echem mode tables exist, residual taxonomy found 4 protocol-adjusted modes, and cycle/region context maps them across 11 cycles and 7 coarse regions. Limitation: Modes are unsupervised/automatic and tied to the selected ROI cohort; residual taxonomy silhouette is modest and cycle/region context is descriptive.
- Correlate degradation with cycles, particle regions, and echem/protocol context: implemented_with_guardrail. Evidence: Multi-cycle ROI echem coupling found strong frame-count/protocol correlations; protocol-conditioned residual tests still show event/control optical shifts and phase-front sign consistency. Limitation: Residualization reduces but does not eliminate confounding, and front residual classifiers are not deployable with 52 automatic ROIs.
- Keep objectives and observations updated: implemented. Evidence: OBJECTIVES_OBSERVATIONS.md contains chronological experiment summaries and guardrails. Limitation: It is long and narrative; the new synthesis is the compact index.
- Keep GitHub updated: implemented_with_verification. Evidence: Scoped analysis scripts, compact derived outputs, and observations are committed and pushed after each completed increment. Limitation: The synthesis report records workflow state at generation time; final push status should still be checked with git status and git log.

## Recommended Next Experiments

1. Manual QC the ROI/front package panels and update `manual_qc_status` with accepted/rejected labels.
2. Expand the ROI cohort across more cycles after QC to reduce event-reference and protocol confounding.
3. Recompute apparent diffusion/front-motion proxies only on QC-accepted fronts with confirmed spatial/time calibration.
4. Convert the top ROI candidates into a labeled degradation-mode benchmark for future self-supervised video models.
5. Grow protocol-conditioned residual models after adding more QC-accepted cycles and particle regions.

## Guardrail

The current outputs support a physics-extraction workflow and ranked hypotheses. They do not yet support a claim of calibrated diffusion coefficients or a deployable automated degradation detector.
