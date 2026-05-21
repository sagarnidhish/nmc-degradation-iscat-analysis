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
- Manual-QC label workbook candidates: 47
- Manual-QC gated accepted fronts: 0
- Prefix feature-importance audit features: 54
- Spatiotemporal degradation graph nodes/edges: 52 / 510
- Phase-kinetics ROI rows/features: 52 / 40
- Calibration metadata HDF5/camera-timing files: 33 / 32
- Calibration claim-risk families/source tables: 11 / 11
- Particle trace cycle rows/drop cycles: 89 / 4
- Particle precursor event/control anchors: 4 / 24
- ROI trace-fusion rows/predictors: 52 / 100
- ROI trace-fusion cycle-null points: 11
- Precursor-informed review candidates: 47
- Precursor visual-bundle candidates/assets: 12 / 12
- Within-cycle echem shape cycles/features: 81 / 48
- Control-balanced QC sensitivity robust strata: 6

## Main Findings

- Persistence is the strongest raw next-frame baseline; DMD/velocity/learned residual experiments are most useful as residual and latent descriptors.
- Prefix-only cropped ROI forecasts still rank the front-direction residual class highest: random_forest at prefix 0.500 gives AUC 0.691, but after excluding raw frame-index features the audited permutation null is not significant.
- The 75%-prefix feature-importance audit is descriptive but not independently significant: pooled OOF AUC 0.447, null empirical p=0.714; strongest ablation groups are mean_intensity_trace, frame_texture_level.
- ROI event/control optical differences survive event-reference-cycle centering, especially cumulative normalized change, first-last decorrelation, latent net displacement, high-fraction growth, and ROI mean trend.
- Frame count and protocol-block position strongly couple to ROI dynamics, so echem/protocol context must be a model covariate and a guardrail.
- After residualizing available protocol/echem covariates and event-reference fixed effects, event/control separation remains in ROI mean delta, high-fraction delta, first-last correlation, cumulative change, DMD residual, and latent displacement.
- Cycles 86 and 116 remain the strongest synchronized event-timing regimes; cycles 60 and 156 provide stronger single-particle morphology/latent-movement examples.
- Apparent front tracking currently indicates optical-front contraction/loss more than clean expanding diffusion fronts.
- Threshold sweeps show robust event/control differences in phase-fraction slope, but radius-derived diffusion proxies remain weaker and threshold-sensitive.
- Calibration metadata audit finds camera-timing datasets in 32 of 33 scanned HDF5 files and no HDF5 pixel-size attributes; sampled timing rows can be sparse segment/cycle timing, while the 96 nm/px scale remains slide-derived pending raw microscope metadata confirmation.
- Calibration claim-risk register audits 11 front/kinetic/diffusion claim families; it classifies diffusion-like values as apparent proxies and keeps manual-QC-gated diffusion/front claims pending.
- Protocol-conditioned front residuals preserve phase-slope sign consistency, but not front-magnitude or diffusion-proxy separability.
- Automatic front-QC sensitivity keeps the positive phase-front residual in 5 strata: all_front_rois, complete_threshold_sweep, q70_phase_ci_excludes_zero, q70_phase_ci_positive, review_panel_selected; review-panel diffusion proxy differences are selection-sensitive and not calibrated transport.
- Protocol-adjusted residual mode taxonomy chooses k=4; its most event-enriched mode is optical_brightening_decorrelating_rollout_hard_front_positive with event fraction 0.846 and Fisher p=0.003.
- A QC review packet prioritizes 30 ROI/front candidates, a control-balanced front package adds 24 control candidates, and the manual-QC label workbook deduplicates these into 47 pending ROI labels.
- Control-balanced QC sensitivity keeps positive phase-front residuals robust in 6 automatic strata, including the balanced selected panel; diffusion-proxy residuals remain non-significant in that balanced panel.
- Manual-QC gated front-effect tests are status `ready_for_manual_labels` with 0 accepted fronts, so no manual-QC-filtered diffusion/front claim is emitted yet.
- Spatiotemporal graph tests show strong same-cycle spatial homophily in front-positive residuals and event-enriched residual modes, but cross-cycle nearest-neighbor front/event labels do not show simple propagation and remain cohort-design sensitive.
- Optical phase-kinetics fits add transition-sharpness and Avrami-style descriptors: event-enriched residual modes have larger q70/q80 transformed-fraction deltas and faster q60/q70 logistic rates, while kinetic fit quality/rates remain strongly coupled to frame count.
- The larger four-particle cycle table shows leakage-conscious early-warning signal for future abrupt drops: any-drop within 8 cycles has mean AUC 0.883 with empirical null p=0.002; synchronized 2+ drops are also detectable but with only two positive cycles.
- Event-aligned precursor windows show lower pre-event capacity/CE and higher cross-particle delta dispersion versus matched non-event anchors; the strongest precursor window test is pre8_to_pre5 capacity_mAh with p=0.002.
- ROI trace-fusion links lagged global particle-trace state to localized front behavior at the ROI-row level: strongest focused context-residual association is trace_lag2_particle_norm_range vs phase_slope_positive_fraction_protocol_residual, rho=0.725, p=2.688e-07.
- Cycle-collapsed ROI trace-fusion null audit reduces 52 ROI rows to 11 cycle points; top surviving collapsed association is trace_lag8_frames_percentile vs mode_review_priority, rho=0.834, empirical p=0.002.
- Precursor-informed ROI review ranks 47 pending manual-QC candidates; the top candidate is cycle156_rank7_obj27 with score 5.527.
- A visual review bundle now packages 12 top precursor-informed ROI candidates; 12 have at least one copied QC/preview asset and a contact sheet for manual inspection.
- Within-cycle echem shape descriptors add raw voltage/current trajectory and dQ/dV-proxy context for 81 observed cycles; strongest ROI association is shape_V_q95 vs mode_review_priority, rho=-0.864, but direct event-cycle shape tests are weak and shape terms remain protocol/capacity guardrails.

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

- front_positive_residual_binary prefix_only random_forest f=0.500: AUC 0.691, balanced accuracy 0.472
- front_positive_residual_binary prefix_plus_context random_forest f=0.250: AUC 0.524, balanced accuracy 0.586
- front_positive_residual_binary prefix_only random_forest f=0.250: AUC 0.520, balanced accuracy 0.534
- front_positive_residual_binary prefix_plus_context random_forest f=0.500: AUC 0.506, balanced accuracy 0.464
- front_positive_residual_binary prefix_only logistic_l2 f=0.500: AUC 0.476, balanced accuracy 0.479
- front_positive_residual_binary prefix_only random_forest f=0.750: AUC 0.409, balanced accuracy 0.510
- front_positive_residual_binary prefix_only logistic_l2 f=0.750: AUC 0.407, balanced accuracy 0.466
- front_positive_residual_binary prefix_only logistic_l2 f=0.250: AUC 0.387, balanced accuracy 0.390
- front_positive_residual_binary prefix_plus_context random_forest f=0.750: AUC 0.356, balanced accuracy 0.484
- front_positive_residual_binary prefix_plus_context logistic_l2 f=0.250: AUC 0.316, balanced accuracy 0.210
- Null check is_event_roi f=0.750: observed AUC 0.562, null p95 0.675, empirical p=0.292
- Null check is_event_enriched_mode f=0.750: observed AUC 0.630, null p95 0.715, empirical p=0.189
- Null check front_positive_residual_binary f=0.500: observed AUC 0.476, null p95 0.668, empirical p=0.515
- Regression mode_review_priority prefix_plus_context random_forest f=0.250: MAE ratio vs median baseline 0.951, rho -0.054
- Regression mode_review_priority prefix_plus_context random_forest f=0.750: MAE ratio vs median baseline 0.979, rho 0.027
- Regression mode_review_priority prefix_plus_context random_forest f=0.500: MAE ratio vs median baseline 0.989, rho -0.046
- Regression mode_review_priority prefix_only random_forest f=0.750: MAE ratio vs median baseline 1.208, rho 0.033
- Guardrail: Prefix-only ROI forecasts use cropped particle-region sequences, but the cohort is small and selected around event-reference cycles; treat results as physics-signal triage, not a deployable predictor.

## Prefix ROI Feature Importance

- Rows/features: 52 / 54
- Model/target: logistic_l2 / front_positive_residual_binary
- Pooled OOF AUC: 0.447; balanced accuracy 0.489
- Null check: p95 AUC 0.638, empirical p=0.714
- Group ablation remove mean_intensity_trace: AUC drop 0.171, n_features=18
- Group ablation remove frame_texture_level: AUC drop -0.002, n_features=5
- Group ablation remove stage_drift: AUC drop -0.023, n_features=3
- Group ablation remove temporal_change_energy: AUC drop -0.048, n_features=14
- Group ablation remove bright_dark_fraction: AUC drop -0.074, n_features=14
- Feature permutation stage_prefix_step_max (stage_drift): AUC drop 0.132, positive-fold fraction 0.500
- Feature permutation low_fraction_prefix_std (bright_dark_fraction): AUC drop 0.009, positive-fold fraction 0.250
- Feature permutation average_intensity_prefix_mean (mean_intensity_trace): AUC drop 0.008, positive-fold fraction 0.250
- Feature permutation average_intensity_prefix_last (mean_intensity_trace): AUC drop 0.008, positive-fold fraction 0.250
- Feature permutation roi_norm_mean_prefix_first (mean_intensity_trace): AUC drop 0.007, positive-fold fraction 0.500
- Feature permutation first_q30_threshold (bright_dark_fraction): AUC drop 0.006, positive-fold fraction 0.250
- Guardrail: Feature importance is computed on the small 52-ROI selected cohort with leave-event-reference-cycle-out folds. Treat it as mechanistic triage for early particle-region video signals, not causal proof or a deployable detector.

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
- Control-balanced sensitivity balanced_qc_not_fragmented phase-sign residual: event/control 6/4, median event-control 0.083, bootstrap p05 -0.004, MW p=0.139, permutation p=0.046.
- Control-balanced sensitivity balanced_qc_selected phase-sign residual: event/control 16/24, median event-control 0.091, bootstrap p05 0.047, MW p=0.003, permutation p=0.008.
- Control-balanced robust positive phase-residual strata: all_front_rois, balanced_qc_selected, complete_threshold_sweep, original_qc_selected, q70_phase_ci_excludes_zero, q70_phase_ci_positive.

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

## Spatiotemporal Degradation Graph

- Graph size: 52 ROI nodes and 510 directed nearest-neighbor edges.
- Homophily same_cycle_spatial_knn front_positive_residual_binary: same fraction 0.936, null mean 0.489, empirical p=9.990e-04
- Homophily same_cycle_spatial_knn is_event_roi: same fraction 1.000, null mean 0.473, empirical p=9.990e-04
- Homophily same_cycle_spatial_knn is_event_enriched_mode: same fraction 0.769, null mean 0.647, empirical p=0.006
- Homophily same_reference_spatial_knn is_event_enriched_mode: same fraction 0.673, null mean 0.648, empirical p=0.264
- Homophily next_cycle_spatial_knn is_event_enriched_mode: same fraction 0.602, null mean 0.645, empirical p=0.888
- Homophily previous_cycle_spatial_knn is_event_enriched_mode: same fraction 0.544, null mean 0.628, empirical p=0.975
- Continuous neighbor same_cycle_spatial_knn phase_slope_positive_fraction_protocol_residual: rho 0.696, null p95 0.197, empirical p=9.990e-04
- Continuous neighbor same_reference_spatial_knn threshold_robust_phase_score_protocol_residual: rho 0.275, null p95 0.096, empirical p=0.003
- Continuous neighbor same_cycle_spatial_knn mode_review_priority: rho 0.814, null p95 0.784, empirical p=0.010
- Continuous neighbor next_cycle_spatial_knn mode_review_priority: rho 0.783, null p95 0.776, empirical p=0.024
- Continuous neighbor next_cycle_spatial_knn phase_slope_positive_fraction_protocol_residual: rho -0.219, null p95 0.124, empirical p=0.024
- Continuous neighbor next_cycle_spatial_knn diffusion_proxy_abs_median_um2_per_s_protocol_residual: rho -0.227, null p95 0.061, empirical p=0.036
- Temporal lag is_event_roi: n=30, previous-neighbor metric 0.167
- Temporal lag is_event_enriched_mode: n=30, previous-neighbor metric 0.441
- Temporal lag front_positive_residual_binary: n=30, previous-neighbor metric 0.426
- Temporal lag mode_review_priority: n=30, previous-neighbor metric NA
- Distance gradient is_event_roi: positive-positive median distance 277.440 px vs other 10.419 px, p=2.672e-04
- Distance gradient front_positive_residual_binary: positive-positive median distance 196.291 px vs other 28.496 px, p=0.129
- Distance gradient is_event_enriched_mode: positive-positive median distance 97.506 px vs other 103.716 px, p=0.672
- Guardrail: Spatiotemporal graph tests use automatic ROI coordinates and automatic residual labels on a selected 52-ROI cohort. They test clustering/propagation hypotheses for review prioritization, not causal material degradation mechanisms.

## Phase Kinetics Avrami Audit

- ROI/features: 52 / 40
- event_enriched_mode_vs_other q60_logistic_k_per_s: median difference 2.660e-04, p=0.014
- event_enriched_mode_vs_other q70_fraction_delta: median difference 0.006, p=0.017
- event_enriched_mode_vs_other q70_transformed_fraction_delta: median difference 0.006, p=0.017
- event_vs_control q70_avrami_r2: median difference 0.045, p=0.022
- event_vs_control q80_time_of_max_abs_rate_frac: median difference -0.132, p=0.024
- event_vs_control q60_avrami_r2: median difference 0.025, p=0.028
- event_enriched_mode_vs_other q80_positive_rate_fraction: median difference 0.021, p=0.032
- event_enriched_mode_vs_other q80_time_of_max_abs_rate_frac: median difference -0.200, p=0.034
- Correlation q70_logistic_r2 vs n_frames_percentile: rho -0.801, p=2.796e-12
- Correlation q60_logistic_r2 vs n_frames_percentile: rho -0.776, p=9.625e-11
- Correlation roi_norm_rate_sign_consistency vs n_frames_percentile: rho 0.728, p=9.758e-10
- Correlation roi_norm_max_abs_rate_per_s vs n_frames_percentile: rho 0.701, p=7.281e-09
- Correlation q80_max_abs_rate_per_s vs n_frames_percentile: rho 0.699, p=8.018e-09
- Correlation q60_max_abs_rate_per_s vs n_frames_percentile: rho 0.690, p=1.525e-08
- Correlation q70_max_abs_rate_per_s vs n_frames_percentile: rho 0.686, p=2.043e-08
- Correlation q70_variation_to_net_abs vs n_frames_percentile: rho -0.675, p=4.098e-08
- Guardrail: Kinetic fits are optical phase-fraction proxies from cropped particle ROIs using provisional timing. Avrami/logistic parameters are descriptive and not calibrated reaction constants.

## Calibration Metadata Audit

- HDF5 files/movie datasets/camera timing: 33 / 32 / 32
- HDF5 calibration-like attribute hits: 0
- Median sampled HDF5 timing FPS proxy: 0.099
- PPTX files scanned/calibration hits: 3 / 3
- PPTX hit Degradation Paper Outline.pptx slide 16: We also plot the minimum viable facet size based on the point spread function of the microscope as ~0.2um^2 (dotted line)(check value). ADD COMPARISON TO MANUFACTURER’S EXPECTED FACET SIZES. The consistency of facet size
- PPTX hit Degradation Paper Outline.pptx slide 3: an improved microscope design, iterating on the instrument used by Merryweather et al., that offers 96nm pixel size and 180x120um FoV , compared to XXnm and YYxZZum . Example image of an NMC-811 electrode with 60%(?) act
- PPTX hit Degradation Project.pptx slide 3: and wouldn’t cycle again. Switched to second cell… Cell 2: Cycled from pristine. No bubbles visible FOV: full field of view, including 4 drilled holes on left half, and undrilled holes on right half. Imaging camera expos
- Guardrail: This audit verifies metadata evidence for timebase/spatial calibration. Camera timing is present in sampled HDF5 files, but the sampled timing rows can represent sparse segment/cycle timing rather than true camera frame cadence. Physical pixel-size evidence should be treated as slide-derived unless raw HDF5 attributes or microscope metadata explicitly confirm it.

## Calibration Claim Risk Register

- Claim families/source tables: 11 / 11
- Calibration evidence: 32 HDF5 timing files, 0 HDF5 spatial-calibration attrs, 3 PPTX hits
- event_candidate_fronts: proxy_only - Use as automatic front-candidate ranking only; do not interpret as calibrated diffusion.
- roi_phase_boundary_mobility: optical_front_proxy - Describe as apparent optical boundary mobility in cropped particle ROIs.
- multi_cycle_roi_mobility: optical_front_proxy - Use for event/control morphology comparisons, not transport constants.
- multi_cycle_threshold_robust_fronts: phase_sign_usable_diffusion_proxy_only - Phase-slope sign/fraction trends are usable optical proxies; diffusion numbers are apparent um^2/s proxies pending calibration provenance.
- front_roi_calibration_qc: apparent_um_scale_proxy - Keep units as apparent um^2/s optical-front proxies and pair with QC flags/manual labels.
- phase_kinetics_avrami: descriptive_optical_kinetics_proxy - Report as optical transition-sharpness/rate descriptors, not reaction constants.
- manual_qc_gated_front_effects: publication_gate_pending - No manual-QC accepted front/diffusion claim should be made until this table is populated.
- roi_front_qc_package: review_prioritization_only - Use to assign labels and inspect artifacts; it does not validate diffusion by itself.
- Guardrail: This register audits wording risk, not numerical correctness. Current front/kinetic outputs are strongest as optical particle-region proxies; diffusion-like values remain apparent proxies until spatial calibration, true frame cadence, masks, and manual QC are jointly validated.

## Particle Trace Physics Audit

- Cycle rows/range: 89 rows, cycles 2.000-158.000
- Drop cycles: any=4, synchronized 2+=2, synchronized 3+=2
- Trace-state clustering: k=2, silhouette=0.266
- Future-drop classifier future_any_drop_within_8cycles: folds=5, AUC 0.883, balanced accuracy 0.648
- Future-drop classifier future_sync2_drop_within_8cycles: folds=3, AUC 0.827, balanced accuracy 0.828
- Future-drop null future_any_drop_within_8cycles: observed AUC 0.883, null p95 0.679, empirical p=0.002
- Future-drop null future_sync2_drop_within_8cycles: observed AUC 0.827, null p95 0.744, empirical p=0.010
- Event feature any_abrupt_drop mean_delta_prev: median pos-neg -0.116, p=1.029e-05
- Event feature any_abrupt_drop max_abs_delta_prev: median pos-neg 0.162, p=0.001
- Event feature any_abrupt_drop mean_abs_delta_prev: median pos-neg 0.089, p=0.003
- Event feature any_abrupt_drop particle_norm_mean: median pos-neg -0.096, p=0.059
- Event feature any_abrupt_drop delta_std_across_particles: median pos-neg 0.049, p=0.067
- Event feature any_abrupt_drop frames_percentile: median pos-neg -0.298, p=0.255
- Trace/echem correlation V_max vs particle_norm_mean: rho 0.626, p=4.144e-10
- Trace/echem correlation V_max vs particle_norm_range: rho 0.475, p=7.293e-06
- Trace/echem correlation V_max vs particle_norm_std: rho 0.459, p=1.656e-05
- Trace/echem correlation coulombic_efficiency_pct vs particle_norm_mean: rho -0.456, p=1.909e-05
- Trace/echem correlation capacity_mAh vs particle_norm_mean: rho 0.454, p=2.081e-05
- Trace/echem correlation V_max vs mean_abs_delta_prev: rho -0.250, p=0.025
- Guardrail: This audit uses the larger four-particle cycle intensity table, not video ROI masks. It tests cycle-level photometry/echem physics hypotheses and early-warning signals, but cannot localize phase fronts or validate diffusion without ROI/video QC.

## Particle Event Precursor Atlas

- Anchors: 4 event anchors, 17 candidate controls, 24 matched controls
- Precursor pre8_to_pre5 capacity_mAh min_value: event-control -0.024, p=0.002
- Precursor pre16_to_pre9 coulombic_efficiency_pct min_value: event-control -1.110, p=0.002
- Precursor pre16_to_pre9 coulombic_efficiency_pct mean_value: event-control -0.421, p=0.002
- Precursor pre16_to_pre9 delta_std_across_particles max_value: event-control 0.040, p=0.007
- Precursor pre16_to_pre9 n_frames slope_per_cycle: event-control -2.175, p=0.012
- Precursor pre16_to_pre9 frames_percentile slope_per_cycle: event-control -0.019, p=0.021
- Precursor pre16_to_pre9 coulombic_efficiency_pct max_value: event-control -0.299, p=0.024
- Precursor pre4_to_pre1 particle_norm_mean min_value: event-control 0.047, p=0.027
- Precursor pre8_to_pre5 mean_delta_prev max_value: event-control -0.017, p=0.030
- Precursor pre16_to_pre9 V_max slope_per_cycle: event-control -5.163e-04, p=0.037
- All-window event_cycle max_abs_delta_prev max_value: event-control 0.136, p=0.002
- All-window event_cycle max_abs_delta_prev mean_value: event-control 0.136, p=0.002
- All-window event_cycle max_abs_delta_prev median_value: event-control 0.136, p=0.002
- All-window event_cycle max_abs_delta_prev min_value: event-control 0.136, p=0.002
- All-window pre8_to_pre5 capacity_mAh min_value: event-control -0.024, p=0.002
- Guardrail: Precursor windows are aligned to four detected abrupt-drop cycles and matched non-event anchors from the four-particle cycle table. Results show cycle-level trace precursors for review and hypothesis generation, not localized phase-front motion or calibrated diffusion.

## ROI Trace Fusion Audit

- Fusion rows/predictors: 52 ROI rows, 100 lagged trace/context predictors
- Focused residual association trace_lag2_particle_norm_range vs phase_slope_positive_fraction_protocol_residual: rho=0.725, p=2.688e-07, n=38
- Focused residual association trace_lag2_particle_norm_std vs phase_slope_positive_fraction_protocol_residual: rho=0.725, p=2.688e-07, n=38
- Focused residual association trace_lag4_delta_std_across_particles vs phase_slope_positive_fraction_protocol_residual: rho=-0.705, p=7.719e-07, n=38
- Focused residual association lag16_trace_predprob_future_any_drop_within_8cycles vs phase_slope_positive_fraction_protocol_residual: rho=-0.652, p=9.324e-06, n=38
- Focused residual association trace_lag4_particle_norm_cv vs phase_slope_positive_fraction_protocol_residual: rho=0.652, p=9.324e-06, n=38
- Focused residual association trace_lag2_particle_norm_cv vs phase_slope_positive_fraction_protocol_residual: rho=0.641, p=1.450e-05, n=38
- Focused residual association trace_lag16_frames_percentile vs phase_slope_positive_fraction_protocol_residual: rho=0.600, p=6.805e-05, n=38
- Focused residual association trace_lag16_max_abs_delta_prev vs phase_slope_positive_fraction_protocol_residual: rho=0.600, p=6.805e-05, n=38
- Event-enriched mode precursor test trace_lag16_delta_std_across_particles: median diff -0.007, p=0.005
- Event-enriched mode precursor test trace_lag8_frames_percentile: median diff 0.185, p=0.006
- Event-enriched mode precursor test trace_lag8_n_frames: median diff 17.000, p=0.006
- Event-enriched mode precursor test trace_lag8_future_sync2_drop_within_8cycles: median diff -1.000, p=0.011
- Event-enriched mode precursor test trace_lag16_V_max: median diff 0.002, p=0.013
- Guardrail: Trace lags are cycle-level four-particle/echem summaries attached to selected ROI rows by cycle number. Associations are useful for linking global precursor state to ROI/front outcomes, but rows are not independent within cycle and this does not prove localized causality.

## ROI Trace Fusion Cycle Null

- Cycle-collapsed points: 11 from 52 ROI rows
- Event-reference cycles: 4
- Predictors/permutations: 68 / 10000
- Cycle-collapsed trace_lag8_frames_percentile vs mode_review_priority: rho 0.834, empirical p=0.002, n=11
- Cycle-collapsed trace_lag8_n_frames vs mode_review_priority: rho 0.834, empirical p=0.003, n=11
- Cycle-collapsed lag8_trace_predprob_future_any_drop_within_8cycles vs mode_review_priority: rho -0.827, empirical p=0.003, n=11
- Cycle-collapsed trace_lag16_frames_percentile vs mode_review_priority: rho 0.813, empirical p=0.003, n=11
- Cycle-collapsed trace_lag16_n_frames vs mode_review_priority: rho 0.813, empirical p=0.005, n=11
- Cycle-collapsed lag4_trace_predprob_future_any_drop_within_8cycles vs q70_logistic_k_per_s: rho -0.800, empirical p=0.005, n=11
- Reference-centered trace_lag8_particle_norm_mean vs phase_slope_median_per_s_protocol_residual: rho 0.918, empirical p=2.000e-04, n=11
- Reference-centered trace_lag8_n_frames vs mode_review_priority: rho 0.881, empirical p=8.999e-04, n=11
- Reference-centered trace_lag8_frames_percentile vs mode_review_priority: rho 0.858, empirical p=0.002, n=11
- Reference-centered lag16_trace_predprob_future_any_drop_within_8cycles vs phase_slope_positive_fraction_protocol_residual: rho -0.809, empirical p=0.003, n=11
- Guardrail: This audit collapses repeated ROI rows to one median point per cycle before testing trace/front associations. It is deliberately conservative for the 52-ROI cohort; surviving tests are stronger evidence, while lost tests indicate cycle-clustering sensitivity.

## Precursor-Informed ROI Review

- Review candidates: 47
- Event/control candidates: 23 / 24
- Review tiers: {'high': 12, 'medium': 18, 'routine': 17}
- cycle156_rank7_obj27 (event, cycle 156): score 5.527, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle60_rank5_obj18 (event, cycle 60): score 5.094, tier high, reason event-cycle ROI;high precursor-context cycle;large front-direction residual;kinetic proxy available
- cycle156_rank6_obj3 (event, cycle 156): score 5.086, tier high, reason event-cycle ROI;high precursor-context cycle;kinetic proxy available
- cycle156_rank5_obj4 (event, cycle 156): score 5.054, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle156_rank2_obj2 (event, cycle 156): score 4.926, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle156_rank1_obj1 (event, cycle 156): score 4.834, tier high, reason event-cycle ROI;high precursor-context cycle;kinetic proxy available
- cycle156_rank8_obj10 (event, cycle 156): score 4.741, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle60_rank3_obj9 (event, cycle 60): score 4.632, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;large front-direction residual;kinetic proxy available
- Guardrail: This is a review-prioritization manifest. It combines automatic ROI/front/mode/kinetic descriptors with cycle-level precursor context to decide what to inspect first; it does not assign manual QC labels or validate diffusion/front claims.

## Precursor Visual Review Bundle

- Ranked candidates included: 12
- Candidates with visual assets: 12
- Contact sheet: /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_review_visual_bundle/top_candidate_contact_sheet.png
- Visual rank 1 cycle156_rank7_obj27 (event, cycle 156): score 5.527, tier high
- Visual rank 2 cycle60_rank5_obj18 (event, cycle 60): score 5.094, tier high
- Visual rank 3 cycle156_rank6_obj3 (event, cycle 156): score 5.086, tier high
- Visual rank 4 cycle156_rank5_obj4 (event, cycle 156): score 5.054, tier high
- Visual rank 5 cycle156_rank2_obj2 (event, cycle 156): score 4.926, tier high
- Visual rank 6 cycle156_rank1_obj1 (event, cycle 156): score 4.834, tier high
- Visual rank 7 cycle156_rank8_obj10 (event, cycle 156): score 4.741, tier high
- Visual rank 8 cycle60_rank3_obj9 (event, cycle 60): score 4.632, tier high
- Guardrail: This bundle copies existing automatic QC/preview assets for manual inspection. It does not create labels, adjudicate particle identity, or validate diffusion/front interpretability.

## Within-Cycle Echem Shape Audit

- Echem shape cycles/features: 81 / 48
- ROI rows joined: 52
- ROI shape correlation shape_V_q95 vs mode_review_priority: rho=-0.864, p=2.718e-12, n=38
- ROI shape correlation neg_dq_abs_peak_frac vs mode_review_priority: rho=0.841, p=3.806e-11, n=38
- ROI shape correlation pos_dq_abs_midV_frac vs mode_review_priority: rho=0.840, p=4.174e-11, n=38
- ROI shape correlation all_dq_abs_entropy vs mode_review_priority: rho=-0.839, p=4.679e-11, n=38
- ROI shape correlation pos_dq_abs_peak_frac vs mode_review_priority: rho=0.839, p=4.679e-11, n=38
- ROI shape correlation pos_dq_abs_highV_frac vs mode_review_priority: rho=-0.839, p=4.787e-11, n=38
- ROI shape correlation pos_dq_abs_entropy vs mode_review_priority: rho=-0.801, p=1.607e-09, n=38
- ROI shape correlation all_dq_abs_peak_frac vs mode_review_priority: rho=0.790, p=3.651e-09, n=38
- Cycle shape correlation shape_charge_mAh_neg_abs vs capacity_mAh: rho=1.000, p=1.253e-149, n=81
- Cycle shape correlation neg_dq_abs_total_mAh vs capacity_mAh: rho=1.000, p=7.534e-126, n=81
- Cycle shape correlation echem_shape_duration_s vs capacity_mAh: rho=0.985, p=1.680e-61, n=81
- Cycle shape correlation shape_charge_mAh_abs vs capacity_mAh: rho=0.985, p=1.680e-61, n=81
- Cycle shape correlation shape_charge_mAh_signed vs coulombic_efficiency_pct: rho=-0.968, p=2.136e-49, n=81
- Cycle shape correlation shape_dIdt_slope vs capacity_mAh: rho=0.934, p=5.846e-37, n=81
- Event shape test neg_dq_abs_peak_voltage: median event-control 0.000, p=0.230
- Event shape test shape_dVdt_abs_p95: median event-control 0.003, p=0.255
- Event shape test neg_dq_abs_total_mAh: median event-control -0.010, p=0.406
- Event shape test shape_charge_mAh_neg_abs: median event-control -0.010, p=0.406
- Guardrail: Within-cycle echem shape features are computed from raw time/potential/current rows for observed particle/ROI cycles. dQ/dV terms are proxy descriptors from current-time integration over voltage bins, not calibrated electrochemical capacity analysis.

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

## Manual QC Label Workbook

- Deduplicated ROI candidates: 47
- Role counts: {'control': 24, 'event': 23}
- Priority tiers: {'high': 12, 'medium': 17, 'routine': 18}
- Manual-QC status counts: {'pending': 47}
- Guardrail: This workbook is a manual-label template. It deduplicates review candidates and preserves pending status, but it does not assign accept/reject labels or validate diffusion.

## Manual-QC Gated Front Effects

- Status: ready_for_manual_labels
- Joined ROI rows: 52
- Accepted front-effect rows: 0
- Accepted diffusion-interpretable rows: 0
- Guardrail: Only manually accepted particle/front labels are eligible for gated front-effect tests. Current pending labels intentionally produce no accepted-front physics claims.

## Control-Balanced QC Sensitivity

- Robust positive phase-residual strata: all_front_rois, balanced_qc_selected, complete_threshold_sweep, original_qc_selected, q70_phase_ci_excludes_zero, q70_phase_ci_positive
- Guardrail: Control-balanced QC sensitivity compares the original event-heavy review panel with the augmented event/control-balanced panel. It audits selection bias in phase-front directionality, but remains an automatic candidate review and not a manual accept/reject QC result.

## Completion Audit

- Implement paper-inspired agentic workflows in separate Isambard folders: implemented. Evidence: agentic_research outputs plus derived tier1/tier2/tier3 experiment folders were created on Isambard and compact outputs synced locally. Limitation: The synthesis script summarizes the outputs but does not rerun the original literature analysis.
- Focus on Alek_Jiho NMC degradation dataset on Isambard: implemented. Evidence: Synthesis reads Isambard derived directory with 52 ROI rows, 11 cycles, and 4 event-reference cycles. Limitation: The current multi-cycle ROI cohort is selected around event/reference cycles, not every raw video in the full dataset.
- Next-frame prediction and rollout: implemented_with_guardrail. Evidence: Persistence, velocity, low-rank DMD, PCA latent trajectories, PCA-ridge, residual-CNN guardrails, and prefix-only ROI forecasts were run. Persistence is best across raw pixel rollouts: True; best prefix classifier target is front_positive_residual_binary with AUC 0.691. Limitation: Learned/full rollout models do not yet beat persistence robustly; use residuals, latent paths, and prefix forecasts as physics descriptors rather than claiming superior pixel prediction.
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
