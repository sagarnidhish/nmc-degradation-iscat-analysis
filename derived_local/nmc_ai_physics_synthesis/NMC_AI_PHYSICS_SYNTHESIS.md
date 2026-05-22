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
- Echem-shape-conditioned ROI/front rows/shape PCs: 52 / 6
- Physics-consistency matrix ROI/cycles: 52 / 11
- Cycle state-space rows/clusters: 89 / 4
- Cycle hazard warning evaluated cycles/events: 62 / 4
- Cycle-state ROI bridge rows/cycles: 52 / 11
- Particle-mask stability ROI/frame rows: 52 / 4992
- Masked ROI rollout frame rows: 4992
- Masked rollout cycle-warning ROI cycles/features: 11 / 105
- Masked residual transition ROI/method rows: 156
- Masked residual state-transfer anchor/full cycles: 11 / 89
- Transfer-ranked reconstructed cycles/ROI rows: 12 / 48
- Transfer-ranked masked rollout ROI/frame rows: 48 / 4608
- Transfer-ranked front physics ROI/cycles: 48 / 12
- Transfer-ranked residual transition timing ROI/method rows: 48 / 144
- Multi-cohort future-drop model rows/features: 59 / 44
- Active-learning QC candidates/visual/immediate: 97 / 47 / 4
- Balanced future-drop ROI rows/cycles/features: 72 / 24 / 26
- Cross-cohort rollout transfer selected/transfer ROIs: 11 / 48
- Diffusion sanity selected-front/publication candidates: 12 / 0
- Control-balanced high-res front tracking/sanity candidates: 40 / 0
- Weak-label benchmark trainable positives/negatives: 3 / 4
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
- Diffusion proxy sanity audit rejects calibrated-diffusion promotion for the selected high-resolution front set: 0 automatic positive candidates and 0 publication candidates; median selected-front apparent D is -3.647e-04 um2/s and only 0.083 of selected fronts are nonnegative.
- Control-balanced high-resolution front tracking expands this check to 40 ROIs ({'control': 24, 'event': 16}); it still yields 0 automatic positive diffusion candidates and event/control selected-D separation remains non-significant (top p=0.314).
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
- Echem-shape-conditioned residual audit uses 45 shape features compressed to 6 PCs; phase-slope positive-fraction residual remains the strongest event/control readout after shape conditioning (p=0.004), while diffusion residuals remain non-significant and the shape-residual classifier is poor.
- Physics-consistency claim matrix scores 52 ROI rows across front, optical-change, rollout, kinetics, precursor, echem-shape, and mode-taxonomy pillars; 2 rows are cross-modal high priority, but all 52 remain `manual_qc_required_no_physics_claim`.
- Cycle state-space transition audit builds a 4-state cycle manifold from trace plus echem-shape features; PC2 is the strongest future 8-cycle abrupt-drop separator (permutation p=0.016), the shuffled-fold classifier reaches mean AUC 0.781, and stricter temporal holdout reaches AUC 0.779 across 2 usable blocks.
- Rolling-origin cycle hazard warning audit evaluates 62 cycles for future 8-cycle abrupt drops; best AUC is 0.783 with permutation p=0.048, and 8-cycle pre-event warnings hit 0.750 of event cycles.
- Cycle-state to ROI/front bridge links state PC2 to ROI physics-consistency after collapsing repeated ROI rows to 11 cycles: top collapsed test cycle_state_pc2 vs mode_taxonomy_score, rho=0.855, permutation p=0.002.
- Particle-mask stability audit confirms ROI-only crops can be processed with a history-aware particle support guardrail: median fallback fraction 0.000, accepted-area CV 0.042, centroid path 73.607 px; event/control mask instability is not significantly different in the current cohort.
- Masked ROI rollout audit scores held-out predictions only inside accepted particle masks; persistence remains best for 52 of 52 ROIs, while low-rank DMD particle MSE tracks cumulative optical change (top rho=0.637, p=3.909e-07).
- Cycle-collapsed masked-rollout warning audit covers 11 observed ROI cycles; strongest tests align residual jumps with same-cycle abrupt drops (top permutation p=0.014), while future-drop evaluation is underpowered with only 1 positive 8-cycle warning case.
- Masked residual state-transfer warning expands the masked-residual signature from 11 video-backed cycles to 89 cycle-state rows; the transferred score separates future 8-cycle drops (AUC 0.708, permutation p=0.004), but anchor leave-one-cycle transfer is weak (rho=-0.155, p=0.650) and cycle-state PC2 remains the stronger direct future8 baseline (AUC 0.772).
- Transfer-ranked ROI reconstruction converts that state-transfer hypothesis list back into direct video crops: 12 cycles yielded 960 reconstructed components and 48 ROI rows; masked rollout on the exported crops again picks persistence as best for 48 of 48 ROIs, while low-rank DMD particle residuals remain much larger than nonparticle context.
- Transfer-ranked front physics audit links those crops to phase/front proxies across 48 ROIs; strongest ROI-level future8 association is radius2_slope_positive_fraction with median positive-negative 0.857, AUC 0.781, and permutation p=3.999e-04; radius/diffusion-like values remain apparent optical-front proxies only.
- Transfer-ranked residual transition timing gives a stronger temporal residual/phase link than the broader event-control cohort: low_rank_dmd weighted_center_distance_to_transition_frac median distance 0.087 versus null mean 0.250, p=2.000e-04; top future8 timing target is persistence particle_to_nonparticle_mse_ratio_median, AUC 0.832.
- Cross-cohort rollout transfer audit shows the late transfer-ranked crops are a distinct video-dynamics domain: selected-cohort DMD evaluated on transfer-ranked ROIs has median particle MSE 0.020, 3.494x the transfer-internal DMD baseline (p=3.277e-09), while pooled training is close to transfer-internal (1.061x).
- Multi-cohort future-drop model combines selected and transfer-ranked ROI physics features across 59 rows / 44 features; leave-cycle random forest reaches AUC 0.886/AP 0.914 with 40-permutation p=0.024, but leave-cohort transfer is not evaluable because the selected cohort has no positive future8 labels.
- Active-learning QC prioritization merges manual-QC, precursor, weak-model, front, and timing evidence into 97 review candidates; 47 have visual assets and 4 are immediate manual-QC picks, led by cycle116_rank7_obj37. No manual labels are assigned.
- Balanced future-drop direct-video audit removes the transfer-ranked class imbalance by sampling 24 cycles and 72 ROI rows with equal weak future8 positives/negatives; leave-cycle logistic_l2 reaches AUC 0.716/AP 0.761, permutation p=0.049. Top positive-associated features are radius2/front-motion proxies and particle-mask rollout residual fractions, still under optical-proxy/manual-QC guardrails.
- Balanced future particle-mask stability audit covers 72 ROIs / 6912 frames; median fallback fraction is 0.000, and the strongest future8 mask-stability contrast is accepted_centroid_max_step_px with p=0.175, so the balanced future signal is not explained by a simple mask-instability split.
- Masked residual transition timing finds low-rank DMD residual weighted centers are closer to automatic phase-transition centers than random at borderline strength (empirical p=0.056), but peak-frame timing is not aligned and persistence particle/nonparticle ratios track kinetic rates.
- Weak-label degradation benchmark converts consensus physics/mode/mask evidence into a guarded manifest: 7 trainable weak rows (3 positive / 4 negative), and only 1 leave-reference fold is class-balanced enough for binary evaluation.

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

## Diffusion Proxy Sanity Audit

- Selected high-resolution front ROIs: 12
- Automatic positive diffusion-proxy candidates: 0
- Publication diffusion candidates after manual-QC gate: 0
- Median selected/threshold apparent D: -3.647e-04 / -6.470e-07 um2/s
- Estimator consensus counts: {'mixed': 6, 'negative': 5, 'positive': 1}
- Gate low_drift_relative_to_motion: 12/12 pass (1.000)
- Gate threshold_nonnegative: 5/12 pass (0.417)
- Gate q70_bootstrap_positive: 2/12 pass (0.167)
- Gate selected_nonnegative: 1/12 pass (0.083)
- Gate selected_fit_good: 1/12 pass (0.083)
- Gate manual_qc_accepted: 0/12 pass (0.000)
- Candidate check cycle116_front1_obj6 (event, cycle 116): selected D -3.345e-04, threshold D -1.021e-06, selected R2 0.215, consensus negative, manual pending
- Candidate check cycle116_front2_obj2 (event, cycle 116): selected D -5.845e-04, threshold D -1.065e-06, selected R2 0.222, consensus negative, manual pending
- Candidate check cycle116_front3_obj9 (event, cycle 116): selected D -6.054e-04, threshold D 2.138e-06, selected R2 0.299, consensus mixed, manual pending
- Candidate check cycle116_front4_obj58 (None, cycle 116): selected D -3.712e-04, threshold D NA, selected R2 0.267, consensus mixed, manual pending
- Candidate check cycle116_front7_obj37 (event, cycle 116): selected D -5.000e-04, threshold D 4.589e-06, selected R2 0.612, consensus mixed, manual pending
- Candidate check cycle116_front8_obj20 (event, cycle 116): selected D 8.568e-05, threshold D -5.566e-06, selected R2 0.091, consensus mixed, manual pending
- Diffusion-sanity link selected_r2 vs mask_instability_score: rho=0.218, p=0.519, n=11
- Diffusion-sanity link selected_r2 vs front_quality_score: rho=-0.209, p=0.537, n=11
- Diffusion-sanity link selected_diffusion_um2_per_s vs threshold_robust_diffusion_score: rho=0.182, p=0.593, n=11
- Diffusion-sanity link drift_to_motion_ratio vs threshold_robust_diffusion_score: rho=0.182, p=0.593, n=11
- Guardrail: This audit checks whether optical radius-squared front proxies behave like calibrated diffusion estimates. Publication diffusion claims require manual QC plus positive, estimator-consistent, low-drift, threshold-robust slopes; otherwise values remain apparent optical-front proxies.

## Control-Balanced Front Tracking And Diffusion Sanity

- High-resolution tracked ROIs: 40
- Cohort counts: {'control': 24, 'event': 16}
- Automatic/publication diffusion candidates: 0 / 0
- Median selected/threshold apparent D: -1.029e-04 / -1.021e-06 um2/s
- Estimator consensus counts: {'negative': 15, 'mixed': 14, 'positive': 11}
- Balanced gate low_drift_relative_to_motion: 40/40 pass (1.000)
- Balanced gate threshold_nonnegative: 16/40 pass (0.400)
- Balanced gate q70_bootstrap_positive: 9/40 pass (0.225)
- Balanced gate selected_nonnegative: 8/40 pass (0.200)
- Balanced gate selected_fit_good: 1/40 pass (0.025)
- Balanced gate manual_qc_accepted: 0/40 pass (0.000)
- Balanced event/control diffusion_proxy_median_um2_per_s: median diff 1.195e-06, p=0.314, n=16/24
- Balanced event/control threshold_robust_diffusion_score: median diff 0.019, p=0.456, n=16/24
- Balanced event/control drift_to_motion_ratio: median diff -0.001, p=0.516, n=16/24
- Balanced event/control selected_diffusion_um2_per_s: median diff -4.076e-05, p=0.858, n=16/24
- Balanced event/control selected_radius2_um2_per_s: median diff -1.630e-04, p=0.858, n=16/24
- Balanced candidate check cycle156_front8_obj10 (event, cycle 156): selected D 5.881e-05, selected R2 0.068, consensus positive, manual pending
- Balanced candidate check cycle157_front2_obj2 (control, cycle 157): selected D 6.320e-06, selected R2 6.553e-04, consensus positive, manual pending
- Balanced candidate check cycle157_front4_obj8 (control, cycle 157): selected D -2.030e-05, selected R2 0.003, consensus positive, manual pending
- Balanced candidate check cycle158_front2_obj1 (control, cycle 158): selected D -3.503e-05, selected R2 0.030, consensus positive, manual pending
- Balanced candidate check cycle58_front4_obj2 (control, cycle 58): selected D -2.425e-04, selected R2 0.046, consensus positive, manual pending
- Guardrail: This audit checks whether optical radius-squared front proxies behave like calibrated diffusion estimates. Publication diffusion claims require manual QC plus positive, estimator-consistent, low-drift, threshold-robust slopes; otherwise values remain apparent optical-front proxies.

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

## Echem-Shape-Conditioned ROI/Front Effects

- Rows event/control: 24 / 28
- Shape PCA: 45 features, 6 PCs, total explained variance 0.997
- Shape-residual classifier: ROC-AUC 0.469, balanced accuracy 0.448
- Shape-conditioned phase_slope_positive_fraction_protocol_residual: event-control residual median 0.031, p=0.004
- Shape-conditioned high_fraction_delta_protocol_residual: event-control residual median 0.002, p=0.008
- Shape-conditioned q70_transformed_fraction_delta: event-control residual median 0.005, p=0.010
- Shape-conditioned low_fraction_delta_protocol_residual: event-control residual median -0.007, p=0.016
- Shape-conditioned q80_transformed_fraction_delta: event-control residual median 0.003, p=0.021
- Shape-conditioned dmd_minus_persistence_mse_protocol_residual: event-control residual median 4.775e-04, p=0.041
- Shape-conditioned mode_review_priority: event-control residual median 0.135, p=0.300
- Shape-conditioned threshold_robust_phase_score_protocol_residual: event-control residual median -0.013, p=0.393
- Shape context fit mode_review_priority: variance explained 0.839, n=52
- Shape context fit first_last_corr_protocol_residual: variance explained 0.583, n=52
- Shape context fit q60_logistic_k_per_s: variance explained 0.544, n=48
- Shape context fit q70_logistic_k_per_s: variance explained 0.515, n=50
- Shape context fit cumulative_abs_norm_change_protocol_residual: variance explained 0.423, n=52
- Shape context fit latent_net_displacement_protocol_residual: variance explained 0.413, n=52
- Shape PC correlation echem_shape_pc3 vs mode_review_priority: rho=-0.707, p=4.516e-09, n=52
- Shape PC correlation echem_shape_pc6 vs phase_slope_positive_fraction_protocol_residual: rho=0.586, p=4.918e-06, n=52
- Shape PC correlation echem_shape_pc6 vs high_fraction_delta_protocol_residual: rho=0.516, p=9.166e-05, n=52
- Shape PC correlation echem_shape_pc4 vs cumulative_abs_norm_change_protocol_residual: rho=-0.490, p=2.252e-04, n=52
- Guardrail: Echem-shape conditioning uses low-dimensional PCA/ridge covariates on a small automatically selected ROI cohort. Surviving residuals are evidence that optical/front signals are not fully explained by measured within-cycle echem shape, but they are not causal proof or calibrated transport constants.

## Probabilistic Rollout Calibration

- Frame rows / ROI-method rows: 4992 / 156
- ROI / event-reference cycles: 52 / 4
- Near-transition frame fraction: 0.264
- 95% empirical coverage low_rank_dmd all: global 0.921, local 0.921, n=1664
- 95% empirical coverage low_rank_dmd event_roi: global 0.871, local 0.871, n=768
- 95% empirical coverage low_rank_dmd near_transition: global 0.909, local 0.941, n=440
- 95% empirical coverage persistence all: global 0.955, local 0.955, n=1664
- 95% empirical coverage persistence event_roi: global 0.945, local 0.945, n=768
- 95% empirical coverage persistence near_transition: global 0.975, local 0.952, n=440
- 95% empirical coverage velocity all: global 0.942, local 0.942, n=1664
- 95% empirical coverage velocity event_roi: global 0.917, local 0.917, n=768
- 95% empirical coverage velocity near_transition: global 0.952, local 0.959, n=440
- Residual test low_rank_dmd late_minus_early_rollout: median diff 0.006, p=8.875e-29, n=572/572
- Residual test low_rank_dmd event_minus_control: median diff 0.004, p=5.586e-15, n=768/896
- Residual test velocity event_minus_control: median diff 1.569e-04, p=0.110, n=768/896
- Residual test persistence event_minus_control: median diff 4.067e-05, p=0.161, n=768/896
- Residual test velocity late_minus_early_rollout: median diff -5.112e-05, p=0.516, n=572/572
- Calibration/physics link low_rank_dmd mae_max vs cumulative_abs_first_last_first: rho=0.631, p=5.263e-07, n=52
- Calibration/physics link low_rank_dmd mae_mean vs cumulative_abs_first_last_first: rho=0.600, p=2.617e-06, n=52
- Calibration/physics link low_rank_dmd q90_undercoverage_rate vs cumulative_abs_first_last_first: rho=0.586, p=5.079e-06, n=52
- Calibration/physics link persistence mae_max vs q70_transformed_fraction_delta_first: rho=0.569, p=1.078e-05, n=52
- Calibration/physics link low_rank_dmd mae_max vs first_last_corr_first: rho=-0.542, p=3.274e-05, n=52
- Undercoverage priority cycle156_rank7_obj27 low_rank_dmd: q90 undercoverage 1.000, priority 3.843, role event
- Undercoverage priority cycle156_rank5_obj4 low_rank_dmd: q90 undercoverage 0.344, priority 3.699, role event
- Undercoverage priority cycle156_rank7_obj27 velocity: q90 undercoverage 1.000, priority 3.606, role event
- Undercoverage priority cycle156_rank8_obj10 low_rank_dmd: q90 undercoverage 0.125, priority 3.516, role event
- Undercoverage priority cycle156_rank7_obj27 persistence: q90 undercoverage 1.000, priority 3.471, role event
- Undercoverage priority cycle156_rank2_obj2 low_rank_dmd: q90 undercoverage 0.781, priority 3.401, role event
- Guardrail: Empirical residual quantiles are a calibration audit for ROI-only rollout baselines. They are not a generative uncertainty model, not calibrated diffusion, and not manual QC.

## Masked Residual State Transfer Warning

- Anchor/full cycles/permutations: 11 / 89 / 5000
- Signature features: 8
- Anchor leave-one-cycle transfer: rho=-0.155, p=0.650, n=11
- Signature feature low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_max_delta_prev_observed_roi_cycle: median positive-negative 12.290, permutation p=0.013
- Signature feature low_rank_dmd_particle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.009, permutation p=0.014
- Signature feature low_rank_dmd_nonparticle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.002, permutation p=0.024
- Signature feature low_rank_dmd_particle_mse_mean_median_delta_prev_observed_roi_cycle: median positive-negative 0.005, permutation p=0.043
- Signature feature low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_max: median positive-negative 8.364, permutation p=0.045
- Signature feature low_rank_dmd_nonparticle_mse_mean_median_delta_prev_observed_roi_cycle: median positive-negative 7.885e-04, permutation p=0.050
- Transfer target future_any_drop_within_16cycles transferred_masked_residual_signature: median positive-negative 0.631, AUC 0.676, permutation p=3.999e-04, n=40/49
- Transfer target any_abrupt_drop mean_abs_delta_prev: median positive-negative 0.089, AUC 0.946, permutation p=0.001, n=4/84
- Transfer target future_any_drop_within_8cycles transferred_masked_residual_signature: median positive-negative 0.648, AUC 0.708, permutation p=0.004, n=20/69
- Transfer target any_abrupt_drop state_step_norm: median positive-negative 5.544, AUC 0.887, permutation p=0.013, n=4/84
- Transfer target future_sync2_drop_within_8cycles frames_percentile: median positive-negative -0.362, AUC 0.856, permutation p=0.014, n=8/81
- Transfer target future_any_drop_within_8cycles cycle_state_pc2: median positive-negative 0.730, AUC 0.772, permutation p=0.016, n=20/69
- Transfer target future_any_drop_within_16cycles mean_abs_delta_prev: median positive-negative -0.011, AUC 0.652, permutation p=0.020, n=40/48
- Transfer target future_any_drop_within_16cycles state_step_norm: median positive-negative -0.813, AUC 0.668, permutation p=0.021, n=40/48
- Transfer/context link cycle_state_pc6: rho=0.541, p=4.330e-08, n=89
- Transfer/context link axis_step: rho=0.522, p=1.779e-07, n=88
- Transfer/context link coulombic_efficiency_pct: rho=-0.499, p=2.162e-06, n=81
- Transfer/context link cycle_state_pc5: rho=-0.399, p=1.098e-04, n=89
- Transfer/context link cycle_state_pc2: rho=0.354, p=6.631e-04, n=89
- Transfer/context link frames_percentile: rho=0.319, p=0.002, n=89
- Transfer-ranked cycle 150: score 40.972, future8=1, future16=1, abrupt=0
- Transfer-ranked cycle 146: score 3.217, future8=0, future16=1, abrupt=0
- Transfer-ranked cycle 156: score 1.612, future8=0, future16=0, abrupt=1
- Transfer-ranked cycle 151: score 1.439, future8=1, future16=1, abrupt=0
- Transfer-ranked cycle 152: score 1.188, future8=1, future16=1, abrupt=0
- Transfer-ranked cycle 154: score 1.010, future8=1, future16=1, abrupt=0
- Guardrail: Masked residual signature is learned from 11 video-backed ROI cycles and transferred through the cycle-state/echem manifold to 89 cycles; use as hypothesis-ranking evidence, not a deployable warning model or direct video residual measurement for unexported cycles.

## Transfer-Ranked ROI Reconstruction And Masked Rollout

- Sampled cycles/reconstructed candidates/ROI rows: 12 / 960 / 48
- Exported ROI sequences: 48 at 96 frames per ROI
- Masked rollout ROI/frame rows: 48 / 4608
- Best method counts inside particle masks: {'persistence': 48}
- Transfer sampled cycle 150 rank 1: score 40.972, future8=1, future16=1, candidates=80
- Transfer sampled cycle 146 rank 2: score 3.217, future8=0, future16=1, candidates=80
- Transfer sampled cycle 156 rank 3: score 1.612, future8=0, future16=0, candidates=80
- Transfer sampled cycle 151 rank 4: score 1.439, future8=1, future16=1, candidates=80
- Transfer sampled cycle 152 rank 5: score 1.188, future8=1, future16=1, candidates=80
- Transfer sampled cycle 154 rank 6: score 1.010, future8=1, future16=1, candidates=80
- Transfer sampled cycle 153 rank 7: score 0.871, future8=1, future16=1, candidates=80
- Transfer sampled cycle 40 rank 8: score 0.784, future8=0, future16=0, candidates=80
- Transfer sequence cycle 116.0: mean ROI delta -10.252, norm delta -4.391e-04, n=4
- Transfer sequence cycle 146.0: mean ROI delta -3.926, norm delta -2.887e-04, n=4
- Transfer sequence cycle 147.0: mean ROI delta 6.302, norm delta 9.638e-04, n=4
- Transfer sequence cycle 148.0: mean ROI delta 1.052, norm delta 5.269e-04, n=4
- Transfer sequence cycle 150.0: mean ROI delta -4.974, norm delta -1.556e-04, n=4
- Transfer sequence cycle 151.0: mean ROI delta 2.846, norm delta 6.862e-05, n=4
- Transfer sequence cycle 152.0: mean ROI delta -5.496, norm delta -3.611e-04, n=4
- Transfer sequence cycle 153.0: mean ROI delta -13.522, norm delta -7.464e-04, n=4
- Transfer-ranked persistence: particle-MSE median 1.847e-04, nonparticle-MSE median 8.412e-05, particle/nonparticle ratio median 2.155
- Transfer-ranked velocity: particle-MSE median 4.602e-04, nonparticle-MSE median 2.397e-04, particle/nonparticle ratio median 1.827
- Transfer-ranked low_rank_dmd: particle-MSE median 0.006, nonparticle-MSE median 6.980e-04, particle/nonparticle ratio median 8.308
- Transfer-ranked difficult ROI cycle151_rank4_obj3 low_rank_dmd cycle 151: particle MSE 0.014, particle/nonparticle ratio 8.196
- Transfer-ranked difficult ROI cycle153_rank7_obj4 low_rank_dmd cycle 153: particle MSE 0.013, particle/nonparticle ratio 7.656
- Transfer-ranked difficult ROI cycle156_rank3_obj2 low_rank_dmd cycle 156: particle MSE 0.013, particle/nonparticle ratio 4.500
- Transfer-ranked difficult ROI cycle154_rank6_obj2 low_rank_dmd cycle 154: particle MSE 0.012, particle/nonparticle ratio 8.299
- Transfer-ranked difficult ROI cycle151_rank4_obj2 low_rank_dmd cycle 151: particle MSE 0.012, particle/nonparticle ratio 9.524
- Transfer-ranked difficult ROI cycle153_rank7_obj2 low_rank_dmd cycle 153: particle MSE 0.011, particle/nonparticle ratio 12.454
- Transfer-ranked ROI candidate cycle 150 obj 4: validation score 32.064, mean abs z 29.677, future8=1
- Transfer-ranked ROI candidate cycle 150 obj 1: validation score 31.252, mean abs z 25.298, future8=1
- Transfer-ranked ROI candidate cycle 150 obj 2: validation score 30.291, mean abs z 20.694, future8=1
- Transfer-ranked ROI candidate cycle 150 obj 3: validation score 30.065, mean abs z 19.134, future8=1
- Transfer-ranked ROI candidate cycle 146 obj 4: validation score 25.085, mean abs z 13.691, future8=0
- Transfer-ranked ROI candidate cycle 146 obj 3: validation score 24.258, mean abs z 20.728, future8=0
- Guardrail: Automatic transfer-ranked ROI candidates reconstructed from sampled HDF5 cycle segments; compatible with particle-region sequence export, but not manual annotations or validated fronts. Held-out rollout errors are scored inside automatic history-aware particle masks; this is not manual segmentation or a new learned video model.

## Transfer-Ranked Front Physics Audit

- Threshold-front ROI/cycles/quantiles: 48 / 12 / 7
- Front-physics target positives: {'any_abrupt_drop': 8, 'future_any_drop_within_16cycles': 36, 'future_any_drop_within_8cycles': 28}
- Front target future_any_drop_within_8cycles radius2_slope_positive_fraction: median positive-negative 0.857, AUC 0.781, permutation p=3.999e-04, n=28/20
- Front target future_any_drop_within_8cycles radius2_slope_median_px2_per_s: median positive-negative 0.002, AUC 0.846, permutation p=5.999e-04, n=28/20
- Front target future_any_drop_within_8cycles q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 0.002, AUC 0.827, permutation p=7.998e-04, n=28/20
- Front target future_any_drop_within_8cycles diffusion_proxy_median_um2_per_s: median positive-negative 5.147e-06, AUC 0.846, permutation p=9.998e-04, n=28/20
- Front target any_abrupt_drop roi_norm_mean_delta_last_minus_first: median positive-negative 0.008, AUC 0.684, permutation p=9.998e-04, n=8/40
- Front target future_any_drop_within_8cycles persistence_particle_mse_mean: median positive-negative 6.220e-04, AUC 0.945, permutation p=0.002, n=28/20
- Front target future_any_drop_within_8cycles q70_phase_slope_bootstrap_p50: median positive-negative -1.709e-06, AUC 0.759, permutation p=0.003, n=28/20
- Front target future_any_drop_within_8cycles phase_slope_median_per_s: median positive-negative -1.805e-06, AUC 0.770, permutation p=0.004, n=28/20
- Front/residual link roi_norm_mean_delta_last_minus_first vs roi_mean_delta_last_minus_first: rho=0.965, p=1.998e-28, n=48
- Front/residual link persistence_particle_mse_mean vs future_any_drop_within_8cycles: rho=0.760, p=3.886e-10, n=48
- Front/residual link radius2_slope_median_px2_per_s vs persistence_particle_mse_mean: rho=0.724, p=6.169e-09, n=48
- Front/residual link diffusion_proxy_median_um2_per_s vs persistence_particle_mse_mean: rho=0.724, p=6.169e-09, n=48
- Front/residual link q70_radius2_slope_bootstrap_p50_px2_per_s vs persistence_particle_mse_mean: rho=0.703, p=2.481e-08, n=48
- Front/residual link q70_radius2_slope_bootstrap_p50_px2_per_s vs low_rank_dmd_particle_mse_mean: rho=0.679, p=1.156e-07, n=48
- Cycle-collapsed front target future_any_drop_within_8cycles persistence_particle_mse_mean: median positive-negative 5.592e-04, AUC 0.971, permutation p=0.058, n=7/5
- Cycle-collapsed front target future_any_drop_within_16cycles diffusion_proxy_abs_median_um2_per_s: median positive-negative 3.253e-06, AUC 0.926, permutation p=0.063, n=9/3
- Cycle-collapsed front target future_any_drop_within_8cycles threshold_robust_diffusion_score: median positive-negative -0.294, AUC 0.743, permutation p=0.065, n=7/5
- Cycle-collapsed front target future_any_drop_within_8cycles q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 0.003, AUC 0.886, permutation p=0.086, n=7/5
- Cycle-collapsed front target future_any_drop_within_8cycles radius2_slope_median_px2_per_s: median positive-negative 0.003, AUC 0.886, permutation p=0.089, n=7/5
- Cycle-collapsed front target future_any_drop_within_8cycles diffusion_proxy_median_um2_per_s: median positive-negative 6.872e-06, AUC 0.886, permutation p=0.092, n=7/5
- Transfer front cycle 40: phase slope 1.580e-06, apparent D median 1.685e-06, phase score 0.661, n=4
- Transfer front cycle 116: phase slope 3.254e-06, apparent D median -1.678e-06, phase score 0.969, n=4
- Transfer front cycle 146: phase slope 2.641e-06, apparent D median -5.688e-06, phase score 0.609, n=4
- Transfer front cycle 147: phase slope 3.909e-06, apparent D median -6.335e-06, phase score 0.729, n=4
- Transfer front cycle 148: phase slope 3.377e-06, apparent D median -4.354e-06, phase score 0.812, n=4
- Transfer front cycle 150: phase slope 2.471e-06, apparent D median 1.369e-06, phase score 1.010, n=4
- Transfer front cycle 151: phase slope 1.226e-06, apparent D median 4.742e-06, phase score 0.398, n=4
- Transfer front cycle 152: phase slope -1.547e-07, apparent D median 5.194e-06, phase score -0.047, n=4
- Front-physics review ROI cycle151_rank4_obj2: score 3.479, future8=1, phase score 1.260, apparent D 9.838e-06, DMD particle MSE 0.012
- Front-physics review ROI cycle150_rank1_obj3: score 3.104, future8=1, phase score 1.260, apparent D 5.160e-06, DMD particle MSE 0.006
- Front-physics review ROI cycle152_rank5_obj2: score 3.104, future8=1, phase score 0.885, apparent D 1.171e-05, DMD particle MSE 0.011
- Front-physics review ROI cycle156_rank3_obj2: score 2.990, future8=0, phase score 1.198, apparent D 1.925e-06, DMD particle MSE 0.013
- Front-physics review ROI cycle151_rank4_obj3: score 2.833, future8=1, phase score 0.760, apparent D 3.555e-06, DMD particle MSE 0.014
- Front-physics review ROI cycle154_rank6_obj2: score 2.813, future8=1, phase score 0.531, apparent D 1.407e-05, DMD particle MSE 0.012
- Front-physics review ROI cycle146_rank2_obj1: score 2.771, future8=0, phase score 0.885, apparent D -5.805e-06, DMD particle MSE 0.006
- Front-physics review ROI cycle153_rank7_obj4: score 2.760, future8=1, phase score 0.594, apparent D 1.101e-05, DMD particle MSE 0.013
- Guardrail: Transfer-ranked front descriptors are automatic optical phase/radius proxies from ROI crops. Diffusion-like values are apparent front-motion descriptors, not calibrated transport coefficients, and the cohort is warning-ranked rather than an event/control design.

## Transfer-Ranked Residual Transition Timing

- Phase-kinetic ROI/timing rows/permutations: 48 / 144 / 5000
- Timing target positives: {'any_abrupt_drop': 8, 'future_any_drop_within_16cycles': 36, 'future_any_drop_within_8cycles': 28}
- Transfer timing alignment low_rank_dmd weighted_center_distance_to_transition_frac: median distance 0.087, null mean 0.250, empirical p=2.000e-04, n=48
- Transfer timing alignment persistence weighted_center_distance_to_transition_frac: median distance 0.156, null mean 0.252, empirical p=0.004, n=48
- Transfer timing alignment velocity weighted_center_distance_to_transition_frac: median distance 0.170, null mean 0.251, empirical p=0.012, n=48
- Transfer timing alignment low_rank_dmd peak_distance_to_transition_frac: median distance 0.421, null mean 0.439, empirical p=0.400, n=48
- Transfer timing alignment persistence peak_distance_to_transition_frac: median distance 0.318, null mean 0.329, empirical p=0.441, n=48
- Transfer timing alignment velocity peak_distance_to_transition_frac: median distance 0.324, null mean 0.312, empirical p=0.591, n=48
- Transfer timing target future_any_drop_within_8cycles persistence particle_to_nonparticle_mse_ratio_median: median positive-negative 0.648, AUC 0.832, permutation p=3.999e-04, n=28/20
- Transfer timing target future_any_drop_within_16cycles persistence particle_to_nonparticle_mse_ratio_median: median positive-negative 0.645, AUC 0.808, permutation p=7.998e-04, n=36/12
- Transfer timing target future_any_drop_within_8cycles persistence residual_peak_particle_mse: median positive-negative 0.010, AUC 0.879, permutation p=0.002, n=28/20
- Transfer timing target future_any_drop_within_8cycles velocity residual_peak_particle_mse: median positive-negative 0.012, AUC 0.923, permutation p=0.002, n=28/20
- Transfer timing target any_abrupt_drop low_rank_dmd near_transition_residual_fraction: median positive-negative -0.073, AUC 0.772, permutation p=0.002, n=8/40
- Transfer timing target future_any_drop_within_16cycles low_rank_dmd weighted_center_distance_to_transition_frac: median positive-negative -0.199, AUC 0.778, permutation p=0.003, n=36/12
- Transfer timing target future_any_drop_within_8cycles low_rank_dmd weighted_center_distance_to_transition_frac: median positive-negative -0.116, AUC 0.645, permutation p=0.003, n=28/20
- Transfer timing target future_any_drop_within_16cycles low_rank_dmd peak_distance_to_transition_frac: median positive-negative -0.146, AUC 0.736, permutation p=0.004, n=36/12
- Transfer timing/target link velocity residual_peak_particle_mse vs future_any_drop_within_8cycles: rho=0.723, p=6.528e-09, n=48
- Transfer timing/target link persistence residual_peak_particle_mse vs future_any_drop_within_8cycles: rho=0.647, p=6.850e-07, n=48
- Transfer timing/target link velocity residual_peak_particle_mse vs future_any_drop_within_16cycles: rho=0.629, p=1.717e-06, n=48
- Transfer timing/target link velocity particle_to_nonparticle_mse_ratio_median vs future_any_drop_within_16cycles: rho=0.583, p=1.346e-05, n=48
- Transfer timing/target link persistence particle_to_nonparticle_mse_ratio_median vs future_any_drop_within_8cycles: rho=0.567, p=2.610e-05, n=48
- Transfer timing/target link persistence residual_peak_particle_mse vs future_any_drop_within_16cycles: rho=0.542, p=6.981e-05, n=48
- Near-transition transfer ROI cycle152_rank5_obj3 persistence: near residual fraction 0.700, peak MSE 0.011, future8=1
- Near-transition transfer ROI cycle152_rank5_obj4 persistence: near residual fraction 0.693, peak MSE 0.010, future8=1
- Near-transition transfer ROI cycle152_rank5_obj1 persistence: near residual fraction 0.660, peak MSE 0.017, future8=1
- Near-transition transfer ROI cycle152_rank5_obj4 velocity: near residual fraction 0.528, peak MSE 0.010, future8=1
- Near-transition transfer ROI cycle152_rank5_obj3 velocity: near residual fraction 0.504, peak MSE 0.010, future8=1
- Near-transition transfer ROI cycle156_rank3_obj1 persistence: near residual fraction 0.486, peak MSE 7.274e-04, future8=0
- Guardrail: Transfer-ranked transition timing uses automatic phase-fraction kinetics and masked rollout residuals from warning-ranked ROI crops. It tests temporal alignment and future-label association only; it is not manual phase-boundary annotation or calibrated transport.

## Multi-Cohort Future-Drop Weak Model

- ROI rows selected/transfer/features: 11 / 48 / 44
- RF trees/permutation null: 80 / 40
- Multi-cohort OOF logistic_l2: AUC 0.692, AP 0.777, scored 58 rows over 13/13 folds
- Multi-cohort OOF random_forest: AUC 0.886, AP 0.914, scored 58 rows over 13/13 folds
- Multi-cohort permutation null random_forest: observed AUC 0.886, null mean 0.441, p95 0.591, empirical p=0.024
- Multi-cohort feature velocity_particle_mse_fraction_of_full_mean: median positive-negative 0.599, oriented AUC 0.968, p=1.012e-09, n=28/30
- Multi-cohort feature velocity_particle_to_nonparticle_mse_ratio_mean: median positive-negative 1.573, oriented AUC 0.964, p=1.354e-09, n=28/30
- Multi-cohort feature velocity_particle_mse_mean: median positive-negative 0.001, oriented AUC 0.964, p=1.354e-09, n=28/30
- Multi-cohort feature persistence_particle_mse_mean: median positive-negative 6.253e-04, oriented AUC 0.940, p=8.944e-09, n=28/30
- Multi-cohort feature persistence_particle_to_nonparticle_mse_ratio_mean: median positive-negative 1.431, oriented AUC 0.920, p=4.132e-08, n=28/30
- Multi-cohort feature persistence_particle_mse_fraction_of_full_mean: median positive-negative 0.629, oriented AUC 0.901, p=1.640e-07, n=28/30
- Multi-cohort feature phase_slope_positive_fraction: median positive-negative -0.071, oriented AUC 0.750, p=1.377e-05, n=28/30
- Multi-cohort feature radius2_slope_median_px2_per_s: median positive-negative 0.002, oriented AUC 0.832, p=1.467e-05, n=28/30
- Multi-cohort importance logistic_l2 velocity_particle_mse_fraction_of_full_mean: 1.301
- Multi-cohort importance logistic_l2 persistence_particle_mse_fraction_of_full_mean: 0.931
- Multi-cohort importance logistic_l2 velocity_particle_to_nonparticle_mse_ratio_mean: 0.719
- Multi-cohort importance logistic_l2 transferred_masked_residual_signature: 0.703
- Multi-cohort importance logistic_l2 stage_drift_xy_sampled: 0.651
- Multi-cohort importance logistic_l2 object_mean_abs_z: 0.588
- Multi-cohort importance logistic_l2 persistence_particle_to_nonparticle_mse_ratio_mean: 0.527
- Multi-cohort importance logistic_l2 roi_mean_delta_last_minus_first: 0.498
- Leave-cohort selected -> transfer_ranked logistic_l2: status missing_class, train positives/negatives 0/10, test positives/negatives 28/20
- Leave-cohort selected -> transfer_ranked random_forest: status missing_class, train positives/negatives 0/10, test positives/negatives 28/20
- Leave-cohort transfer_ranked -> selected logistic_l2: status missing_class, train positives/negatives 28/20, test positives/negatives 0/10
- Leave-cohort transfer_ranked -> selected random_forest: status missing_class, train positives/negatives 28/20, test positives/negatives 0/10
- Guardrail: Future-drop labels are weak cycle-level labels projected onto ROI rows. Grouped splits reduce cycle leakage, but this is a review-prioritization model, not a deployment-ready degradation detector or manual-QC label set.

## Active-Learning QC Prioritization

- Candidate/visual/immediate rows: 97 / 47 / 4
- Tier counts: {'control_balance_review': 47, 'front_diffusion_guardrail_review': 2, 'immediate_manual_qc': 4, 'model_boundary_case': 8, 'standard_manual_qc': 36}
- Active-QC ROI cycle116_rank7_obj37: rank 1, tier immediate_manual_qc, score 0.463, cycle 116, tags model_boundary_case;control_balance_review;visual_asset_available, model p=0.456
- Active-QC ROI cycle156_rank3_obj2: rank 2, tier model_boundary_case, score 0.462, cycle 156, tags model_boundary_case;control_balance_review, model p=0.529
- Active-QC ROI cycle147_rank11_obj2: rank 3, tier model_boundary_case, score 0.437, cycle 147, tags model_boundary_case;control_balance_review, model p=0.503
- Active-QC ROI cycle86_rank4_obj9: rank 4, tier immediate_manual_qc, score 0.414, cycle 86, tags control_balance_review;visual_asset_available, model p=0.369
- Active-QC ROI cycle151_rank4_obj2: rank 5, tier front_diffusion_guardrail_review, score 0.406, cycle 151, tags high_future_drop_probability;front_diffusion_guardrail_review, model p=0.929
- Active-QC ROI cycle156_rank3_obj1: rank 6, tier model_boundary_case, score 0.396, cycle 156, tags model_boundary_case;control_balance_review, model p=0.428
- Active-QC ROI cycle156_rank3_obj4: rank 7, tier control_balance_review, score 0.391, cycle 156, tags control_balance_review, model p=0.640
- Active-QC ROI cycle146_rank2_obj2: rank 8, tier control_balance_review, score 0.390, cycle 146, tags control_balance_review, model p=0.663
- Active-QC cycle 116: max score 0.463, mean score 0.230, ROI 11, immediate 1
- Active-QC cycle 156: max score 0.462, mean score 0.360, ROI 10, immediate 1
- Active-QC cycle 147: max score 0.437, mean score 0.367, ROI 4, immediate 0
- Active-QC cycle 86: max score 0.414, mean score 0.291, ROI 6, immediate 1
- Active-QC cycle 151: max score 0.406, mean score 0.353, ROI 4, immediate 0
- Active-QC cycle 146: max score 0.390, mean score 0.294, ROI 4, immediate 0
- Active-QC reason control_balance_review: n=54
- Active-QC reason visual_asset_available: n=47
- Active-QC reason high_future_drop_probability: n=19
- Active-QC reason model_boundary_case: n=9
- Active-QC reason general_pending_qc: n=6
- Active-QC reason front_diffusion_guardrail_review: n=4
- Active-QC reason transition_timing_residual_review: n=3
- Guardrail: Active-learning QC ranks pending ROI review candidates from automatic and weak-label evidence only; manual labels, diffusion claims, and deployment decisions remain withheld until human QC.

## Balanced Future-Drop Direct-Video ROI Audit

- Reconstructed cycles/candidates/ROI rows: 24 / 1920 / 72
- Exported ROI sequences: 72
- Physics ROI/cycles/features: 72 / 24 / 26
- Label counts: [{'cohort_role': 'future8_negative', 'future_any_drop_within_8cycles': 0, 'n': 36}, {'cohort_role': 'future8_positive', 'future_any_drop_within_8cycles': 1, 'n': 36}]
- Masked rollout best-method counts: {'persistence': 72}
- Mask stability ROI/frames: 72 / 6912; overall {'median_fallback_frame_fraction': 0.0, 'median_accepted_area_cv': 0.07390347879875156, 'median_centroid_path_px': 133.39177039942166, 'median_mask_instability_score': 1.4906788486948246}
- Mask stability role summary: [{'cohort_role': 'future8_negative', 'n_roi': 36, 'fallback_frame_fraction': 0.0, 'accepted_area_cv': 0.07014676858327966, 'accepted_centroid_path_px': 132.55635556011174, 'mask_instability_score': 1.4676753658649946}, {'cohort_role': 'future8_positive', 'n_roi': 36, 'fallback_frame_fraction': 0.0, 'accepted_area_cv': 0.07785297724433327, 'accepted_centroid_path_px': 135.94103030938612, 'mask_instability_score': 1.5007457893873566}]
- Balanced future OOF logistic_l2: AUC 0.716, AP 0.761, scored 72 rows (36/36 pos/neg)
- Balanced future OOF random_forest: AUC 0.673, AP 0.736, scored 72 rows (36/36 pos/neg)
- Balanced future permutation null logistic_l2: observed AUC 0.716, null mean 0.489, p95 0.701, empirical p=0.049
- Balanced future ROI feature radius2_slope_median_px2_per_s: median positive-negative 0.001, oriented AUC 0.717, permutation p=0.024, n=36/36
- Balanced future ROI feature diffusion_proxy_median_um2_per_s: median positive-negative 2.622e-06, oriented AUC 0.717, permutation p=0.024, n=36/36
- Balanced future ROI feature q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 0.001, oriented AUC 0.710, permutation p=0.024, n=36/36
- Balanced future ROI feature persistence_particle_mse_fraction_of_full_mean: median positive-negative 0.282, oriented AUC 0.709, permutation p=0.024, n=36/36
- Balanced future ROI feature velocity_particle_mse_fraction_of_full_mean: median positive-negative 0.345, oriented AUC 0.704, permutation p=0.024, n=36/36
- Balanced future ROI feature velocity_particle_to_nonparticle_mse_ratio_mean: median positive-negative 0.495, oriented AUC 0.672, permutation p=0.024, n=36/36
- Balanced future ROI feature persistence_particle_to_nonparticle_mse_ratio_mean: median positive-negative 0.447, oriented AUC 0.673, permutation p=0.049, n=36/36
- Balanced future ROI feature transferred_masked_residual_signature: median positive-negative 0.416, oriented AUC 0.681, permutation p=0.073, n=36/36
- Balanced future cycle feature persistence_particle_mse_fraction_of_full_mean: median positive-negative 0.246, oriented AUC 0.729, permutation p=0.049, n=12/12
- Balanced future cycle feature radius2_slope_median_px2_per_s: median positive-negative 8.206e-04, oriented AUC 0.722, permutation p=0.049, n=12/12
- Balanced future cycle feature persistence_particle_to_nonparticle_mse_ratio_mean: median positive-negative 0.520, oriented AUC 0.694, permutation p=0.073, n=12/12
- Balanced future cycle feature q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 8.303e-04, oriented AUC 0.729, permutation p=0.098, n=12/12
- Balanced future cycle feature velocity_particle_mse_fraction_of_full_mean: median positive-negative 0.278, oriented AUC 0.715, permutation p=0.122, n=12/12
- Balanced future cycle feature transferred_masked_residual_signature: median positive-negative 0.416, oriented AUC 0.681, permutation p=0.122, n=12/12
- Balanced future mask-stability accepted_centroid_max_step_px: future8 positive-negative 2.153, MW p=0.175, n=36/36
- Balanced future mask-stability accepted_area_cv: future8 positive-negative 0.008, MW p=0.248, n=36/36
- Balanced future mask-stability accepted_centroid_path_px: future8 positive-negative 3.385, MW p=0.395, n=36/36
- Balanced future mask-stability candidate_area_cv: future8 positive-negative 0.005, MW p=0.414, n=36/36
- Balanced future mask-stability mask_instability_score: future8 positive-negative 0.033, MW p=0.461, n=36/36
- Balanced future mask-stability low_area_fraction: future8 positive-negative 0.000, MW p=0.569, n=36/36
- Front-script guardrail: Diffusion values are apparent optical-front proxies computed as slope(radius^2) * pixel_size_um^2 / 4. They are not validated Li diffusion coefficients and require manual front QC.
- Audit guardrail: Balanced direct-video ROI audit uses weak cycle-level future8 labels projected to automatic particle-region candidates. Cycle-group splits and cycle-collapsed tests reduce leakage, but this is still review-prioritization evidence, not manual QC or deployable detection.

## Cross-Cohort Rollout Transfer

- ROI cohorts selected/transfer-ranked: 11 / 48
- Low-rank rank/train fraction: 16 / 0.670
- Transfer model selected_internal on selected: median particle MSE 0.002, DMD/persistence 22.808, particle/nonparticle ratio 5.416, internal ratio NA, p=NA
- Transfer model pooled on selected: median particle MSE 0.006, DMD/persistence 40.246, particle/nonparticle ratio 4.473, internal ratio 3.521, p=0.115
- Transfer model transfer_ranked_internal on selected: median particle MSE 0.016, DMD/persistence 102.029, particle/nonparticle ratio 5.186, internal ratio 10.285, p=8.113e-04
- Transfer model transfer_ranked_internal on transfer_ranked: median particle MSE 0.006, DMD/persistence 20.622, particle/nonparticle ratio 8.308, internal ratio NA, p=NA
- Transfer model pooled on transfer_ranked: median particle MSE 0.006, DMD/persistence 21.376, particle/nonparticle ratio 8.449, internal ratio 1.061, p=0.674
- Transfer model selected_internal on transfer_ranked: median particle MSE 0.020, DMD/persistence 65.440, particle/nonparticle ratio 7.493, internal ratio 3.494, p=3.277e-09
- Transfer/error link transfer_ranked_internal on transfer_ranked cycleNo vs particle_mse_mean: rho=0.569, p=2.493e-05, n=48
- Transfer/error link pooled on transfer_ranked cycleNo vs particle_mse_mean: rho=0.511, p=2.092e-04, n=48
- Transfer/error link selected_internal on selected cycleNo vs particle_mse_mean: rho=-0.866, p=5.670e-04, n=11
- Transfer/error link selected_internal on transfer_ranked validation_score_first vs dmd_particle_mse_ratio_vs_persistence: rho=0.408, p=0.004, n=48
- Transfer/error link transfer_ranked_internal on selected validation_score_first vs particle_mse_mean: rho=-0.782, p=0.004, n=11
- Transfer/error link selected_internal on transfer_ranked validation_score_first vs particle_mse_mean: rho=0.381, p=0.008, n=48
- Transfer-ranked hard ROI cycle116_rank12_obj2 via selected_internal: particle MSE 0.054, DMD/persistence 844.134, cycle 116
- Transfer-ranked hard ROI cycle152_rank5_obj3 via selected_internal: particle MSE 0.045, DMD/persistence 84.097, cycle 152
- Transfer-ranked hard ROI cycle40_rank8_obj3 via selected_internal: particle MSE 0.043, DMD/persistence 443.572, cycle 40
- Transfer-ranked hard ROI cycle152_rank5_obj1 via selected_internal: particle MSE 0.040, DMD/persistence 46.942, cycle 152
- Transfer-ranked hard ROI cycle150_rank1_obj4 via selected_internal: particle MSE 0.040, DMD/persistence 325.487, cycle 150
- Transfer-ranked hard ROI cycle154_rank6_obj3 via selected_internal: particle MSE 0.039, DMD/persistence 104.473, cycle 154
- Guardrail: Cross-cohort low-rank rollout transfer compares interpretable linear dynamics across automatic ROI cohorts. It is evidence about video-domain generalization and difficult particle-local dynamics, not manual QC, calibrated diffusion, or a deployable learned video predictor.

## Masked Residual Transition Timing

- ROI/method rows/permutations: 156 / 5000
- ROI count: 52
- Alignment low_rank_dmd weighted_center_distance_to_transition_frac: median distance 0.196, null mean 0.250, empirical p=0.056, n=52
- Alignment low_rank_dmd peak_distance_to_transition_frac: median distance 0.453, null mean 0.419, empirical p=0.714, n=52
- Alignment velocity weighted_center_distance_to_transition_frac: median distance 0.279, null mean 0.250, empirical p=0.798, n=52
- Alignment persistence peak_distance_to_transition_frac: median distance 0.346, null mean 0.299, empirical p=0.828, n=52
- Alignment velocity peak_distance_to_transition_frac: median distance 0.357, null mean 0.299, empirical p=0.886, n=52
- Alignment persistence weighted_center_distance_to_transition_frac: median distance 0.319, null mean 0.253, empirical p=0.966, n=52
- Event/control timing persistence near_transition_residual_fraction: median event-control 0.069, p=0.068, n=24/28
- Event/control timing low_rank_dmd peak_distance_to_transition_frac: median event-control 0.122, p=0.183, n=24/28
- Event/control timing velocity near_transition_residual_fraction: median event-control 0.053, p=0.209, n=24/28
- Event/control timing low_rank_dmd near_minus_far_particle_mse_median: median event-control -5.138e-04, p=0.317, n=24/28
- Event/control timing low_rank_dmd particle_to_nonparticle_mse_ratio_median: median event-control -0.444, p=0.354, n=24/28
- Event/control timing low_rank_dmd weighted_center_distance_to_transition_frac: median event-control 0.042, p=0.393, n=24/28
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q70_max_abs_rate_per_s: rho=0.518, p=8.420e-05, n=52
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q80_transformed_fraction_delta: rho=0.490, p=2.286e-04, n=52
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q80_max_abs_rate_per_s: rho=0.489, p=2.310e-04, n=52
- Timing/kinetics link low_rank_dmd near_minus_far_particle_mse_median vs q70_transformed_fraction_delta: rho=-0.474, p=3.847e-04, n=52
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q70_transformed_fraction_delta: rho=0.438, p=0.001, n=52
- Timing/kinetics link low_rank_dmd near_minus_far_particle_mse_median vs q80_transformed_fraction_delta: rho=-0.421, p=0.002, n=52
- Near-transition residual ROI cycle158_rank2_obj1 velocity (control, cycle 158): near fraction 0.860, peak distance 0.021, weighted distance 0.048
- Near-transition residual ROI cycle158_rank2_obj1 persistence (control, cycle 158): near fraction 0.773, peak distance 0.021, weighted distance 0.046
- Near-transition residual ROI cycle86_rank4_obj9 persistence (event, cycle 86): near fraction 0.696, peak distance 0.044, weighted distance 0.006
- Near-transition residual ROI cycle86_rank8_obj17 persistence (event, cycle 86): near fraction 0.645, peak distance 0.054, weighted distance 0.008
- Near-transition residual ROI cycle86_rank5_obj8 persistence (event, cycle 86): near fraction 0.626, peak distance 0.044, weighted distance 0.001
- Near-transition residual ROI cycle86_rank3_obj5 persistence (event, cycle 86): near fraction 0.597, peak distance 0.033, weighted distance 0.015
- Guardrail: Automatic phase-kinetic transition timing and masked rollout residual timing audit; not manual front annotation or calibrated transport.

## Masked Rollout Cycle Warning

- ROI cycles/features/permutations: 11 / 105 / 5000
- Target positive counts: {'any_abrupt_drop': 4, 'synchronized_drop_2plus': 2, 'future_any_drop_within_4cycles': 1, 'future_any_drop_within_8cycles': 1, 'future_any_drop_within_16cycles': 1, 'future_sync2_drop_within_4cycles': 0, 'future_sync2_drop_within_8cycles': 0, 'future_sync2_drop_within_16cycles': 0}
- Target test any_abrupt_drop low_rank_dmd_particle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.009, MW p=0.038, permutation p=0.014, n=4/6
- Target test any_abrupt_drop low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_max_delta_prev_observed_roi_cycle: median positive-negative 12.290, MW p=0.019, permutation p=0.016, n=4/6
- Target test synchronized_drop_2plus low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_median_delta_prev_observed_roi_cycle: median positive-negative -2.591, MW p=0.044, permutation p=0.022, n=2/8
- Target test synchronized_drop_2plus low_rank_dmd_particle_mse_fraction_of_full_mean_median_delta_prev_observed_roi_cycle: median positive-negative -0.425, MW p=0.044, permutation p=0.024, n=2/8
- Target test any_abrupt_drop low_rank_dmd_nonparticle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.002, MW p=0.038, permutation p=0.024, n=4/6
- Target test synchronized_drop_2plus low_rank_dmd_particle_mse_fraction_of_full_mean_median: median positive-negative -0.430, MW p=0.073, permutation p=0.036, n=2/9
- Cycle context link persistence_particle_mse_fraction_of_full_mean_max vs cycle_state_pc2: rho=0.955, p=4.989e-06, permutation p=2.000e-04, n=11
- Cycle context link persistence_particle_mse_fraction_of_full_mean_median_rolling2_mean vs frames_percentile: rho=0.927, p=4.052e-05, permutation p=2.000e-04, n=11
- Cycle context link persistence_particle_mse_fraction_of_full_mean_median vs cycle_state_pc2: rho=0.909, p=1.056e-04, permutation p=3.999e-04, n=11
- Cycle context link persistence_particle_to_nonparticle_mse_ratio_mean_median_rolling2_mean vs frames_percentile: rho=0.890, p=2.380e-04, permutation p=3.999e-04, n=11
- Cycle context link persistence_particle_to_nonparticle_mse_ratio_mean_median vs cycle_state_pc2: rho=0.900, p=1.600e-04, permutation p=5.999e-04, n=11
- Cycle context link persistence_particle_mse_fraction_of_full_mean_max vs frames_percentile: rho=0.904, p=1.332e-04, permutation p=7.998e-04, n=11
- Warning-ranked cycle 60: score 3.091, abrupt_drop=1, future8=0, n_roi=6
- Warning-ranked cycle 156: score 3.091, abrupt_drop=1, future8=0, n_roi=6
- Warning-ranked cycle 62: score 2.818, abrupt_drop=0, future8=0, n_roi=4
- Warning-ranked cycle 157: score 2.727, abrupt_drop=0, future8=0, n_roi=4
- Warning-ranked cycle 158: score 2.545, abrupt_drop=0, future8=0, n_roi=4
- Warning-ranked cycle 86: score 2.273, abrupt_drop=1, future8=0, n_roi=6
- Guardrail: Cycle-level audit from selected masked ROI rollout residuals; small selected cycle set, no deployable warning model.

## Masked ROI Rollout Audit

- ROI/frame metric rows: 52 / 4992
- Best method counts inside particle masks: {'persistence': 52}
- DMD spectral radius: 1.002
- persistence: particle-MSE median 1.146e-04, nonparticle-MSE median 7.408e-05, particle/nonparticle ratio median 1.461
- velocity: particle-MSE median 2.595e-04, nonparticle-MSE median 2.040e-04, particle/nonparticle ratio median 1.317
- low_rank_dmd: particle-MSE median 0.007, nonparticle-MSE median 0.001, particle/nonparticle ratio median 5.233
- Masked rollout event/control low_rank_dmd particle_mse_mean: median event-control 0.003, p=0.015
- Masked rollout event/control low_rank_dmd particle_mae_mean: median event-control 0.013, p=0.015
- Masked rollout event/control low_rank_dmd particle_mse_ratio_vs_persistence: median event-control 16.640, p=0.026
- Masked rollout event/control low_rank_dmd nonparticle_mse_mean: median event-control 3.952e-04, p=0.048
- Masked rollout event/control low_rank_dmd particle_to_nonparticle_mse_ratio_mean: median event-control -0.505, p=0.364
- Masked rollout event/control velocity particle_mse_fraction_of_full_mean: median event-control 0.023, p=0.640
- Masked rollout/physics link persistence particle_mse_fraction_of_full_mean vs cumulative_abs_first_last: rho=0.637, p=3.909e-07, n=52
- Masked rollout/physics link low_rank_dmd particle_mse_mean vs cumulative_abs_first_last: rho=0.603, p=2.273e-06, n=52
- Masked rollout/physics link persistence particle_mse_fraction_of_full_mean vs first_last_corr: rho=-0.585, p=5.151e-06, n=52
- Masked rollout/physics link persistence particle_to_nonparticle_mse_ratio_mean vs cumulative_abs_first_last: rho=0.567, p=1.170e-05, n=52
- Masked rollout/physics link low_rank_dmd particle_to_nonparticle_mse_ratio_mean vs high_fraction_slope_per_s: rho=-0.555, p=1.995e-05, n=52
- Masked rollout/physics link low_rank_dmd particle_mse_mean vs first_last_corr: rho=-0.544, p=3.088e-05, n=52
- Particle-rollout difficult ROI cycle156_rank7_obj27 low_rank_dmd (event, cycle 156): particle MSE 0.022, particle/nonparticle ratio 6.368
- Particle-rollout difficult ROI cycle86_rank8_obj17 low_rank_dmd (event, cycle 86): particle MSE 0.018, particle/nonparticle ratio 3.990
- Particle-rollout difficult ROI cycle157_rank2_obj2 low_rank_dmd (control, cycle 157): particle MSE 0.016, particle/nonparticle ratio 8.877
- Particle-rollout difficult ROI cycle156_rank2_obj2 low_rank_dmd (event, cycle 156): particle MSE 0.016, particle/nonparticle ratio 3.066
- Particle-rollout difficult ROI cycle62_rank3_obj9 low_rank_dmd (control, cycle 62): particle MSE 0.015, particle/nonparticle ratio 8.474
- Guardrail: Held-out rollout errors are scored inside automatic history-aware particle masks; this is not manual segmentation or a new learned video model.

## Particle Mask Stability Audit

- ROI/frame rows: 52 / 4992
- Overall median fallback fraction: 0.000
- Overall median accepted-area CV: 0.042
- Overall median centroid path: 73.607 px
- Overall median instability score: 0.809
- control masks: n=28, fallback median 0.000, area-CV median 0.036, centroid-path median 73.607 px
- event masks: n=24, fallback median 0.000, area-CV median 0.045, centroid-path median 68.920 px
- Event/control mask test fallback_frame_fraction: median event-control 0.000, p=0.298
- Event/control mask test centroid_jump_fraction: median event-control 0.000, p=0.298
- Event/control mask test accepted_area_cv: median event-control 0.008, p=0.435
- Event/control mask test accepted_area_fraction_iqr: median event-control 0.002, p=0.576
- Event/control mask test candidate_area_cv: median event-control 0.002, p=0.790
- Event/control mask test accepted_centroid_path_px: median event-control -4.687, p=0.847
- Mask/physics link accepted_centroid_path_px vs high_fraction_slope_per_s: rho=-0.370, p=0.007, n=52
- Mask/physics link mask_instability_score vs high_fraction_slope_per_s: rho=-0.353, p=0.010, n=52
- Mask/physics link accepted_area_fraction_iqr vs high_fraction_slope_per_s: rho=-0.341, p=0.013, n=52
- Mask/physics link accepted_area_fraction_iqr vs evidence_score: rho=-0.269, p=0.054, n=52
- Mask/physics link accepted_area_fraction_iqr vs mean_drop_frac: rho=-0.250, p=0.074, n=52
- High-instability ROI cycle86_rank8_obj17 (event, cycle 86): score 2.228, area-CV 0.228, centroid path 275.841 px
- High-instability ROI cycle156_rank2_obj2 (event, cycle 156): score 2.155, area-CV 0.155, centroid path 197.337 px
- High-instability ROI cycle86_rank6_obj78 (event, cycle 86): score 2.117, area-CV 0.117, centroid path 263.636 px
- High-instability ROI cycle88_rank2_obj4 (control, cycle 88): score 2.115, area-CV 0.115, centroid path 301.115 px
- High-instability ROI cycle116_rank2_obj2 (event, cycle 116): score 2.113, area-CV 0.113, centroid path 290.833 px
- Guardrail: guardrail audit of particle-region mask stability, not manual segmentation

## Physics Consistency Claim Matrix

- ROI/cycles: 52 / 11
- Tier counts: {'cross_modal_high_priority': 2, 'cross_modal_review_priority': 8, 'discordant_guardrail': 14, 'front_kinetic_consistent': 4, 'rollout_mode_consistent': 3, 'routine_or_low_consistency': 21}
- Claim readiness: {'manual_qc_required_no_physics_claim': 52}
- Manual-QC accepted rows: 0
- Rank 1 cycle156_rank7_obj27 (event, cycle 156): score 5.790, support 6, tier cross_modal_high_priority
- Rank 2 cycle156_rank5_obj4 (event, cycle 156): score 5.503, support 6, tier cross_modal_review_priority
- Rank 3 cycle156_rank2_obj2 (event, cycle 156): score 5.486, support 6, tier cross_modal_high_priority
- Rank 4 cycle158_rank2_obj1 (control, cycle 158): score 4.968, support 6, tier cross_modal_review_priority
- Rank 5 cycle156_rank8_obj10 (event, cycle 156): score 3.925, support 4, tier cross_modal_review_priority
- Rank 6 cycle158_rank4_obj8 (control, cycle 158): score 3.656, support 5, tier cross_modal_review_priority
- Rank 7 cycle158_rank6_obj3 (control, cycle 158): score 3.146, support 3, tier front_kinetic_consistent
- Rank 8 cycle156_rank6_obj3 (event, cycle 156): score 2.681, support 3, tier front_kinetic_consistent
- Event/control pillar test optical_change_score: median event-control 0.574, MW p=0.001, permutation p=0.002
- Event/control pillar test mode_taxonomy_score: median event-control 0.644, MW p=0.060, permutation p=0.011
- Event/control pillar test front_direction_score: median event-control 0.531, MW p=0.010, permutation p=0.041
- Event/control pillar test rollout_residual_score: median event-control 0.354, MW p=0.023, permutation p=0.056
- Event/control pillar test physics_consistency_score: median event-control 0.775, MW p=0.071, permutation p=0.239
- Event/control pillar test precursor_context_score: median event-control 0.159, MW p=0.425, permutation p=0.396
- Guardrail: This matrix is a multimodal consistency and review-prioritization audit. It does not assign manual QC labels and does not validate calibrated diffusion or material degradation mechanisms.

## Cycle State-Space Transition Audit

- Cycle rows/features: 89 / 107
- Echem-shape cycles joined: 81
- Chosen state clusters: 4 with silhouette 0.634
- Degradation axis oriented to mean_abs_delta_prev with rho 0.326
- Future-drop classifier: shuffled-fold AUC 0.781, balanced accuracy 0.731
- Expanding temporal holdout: AUC 0.779, balanced accuracy 0.645, evaluated blocks 2 / 4, purge 8 cycles
- Future-drop test cycle_state_pc2: positive-negative median 0.730, permutation p=0.016, MW p=2.322e-04
- Future-drop test mean_abs_delta_prev: positive-negative median -0.012, permutation p=0.080, MW p=0.036
- Future-drop test state_step_norm: positive-negative median -0.694, permutation p=0.101, MW p=0.097
- Future-drop test degradation_state_axis: positive-negative median -2.269, permutation p=0.184, MW p=0.295
- Future-drop test cycle_state_pc1: positive-negative median 2.269, permutation p=0.199, MW p=0.295
- Future-drop test capacity_mAh: positive-negative median -0.014, permutation p=0.205, MW p=0.033
- State correlation degradation_state_axis vs cycleNo: rho=-0.738, p=1.594e-16, n=89
- State correlation cycle_state_pc1 vs cycleNo: rho=0.738, p=1.594e-16, n=89
- State correlation degradation_state_axis vs capacity_mAh: rho=-0.710, p=1.133e-13, n=81
- State correlation cycle_state_pc1 vs capacity_mAh: rho=0.710, p=1.133e-13, n=81
- State correlation degradation_state_axis vs frames_percentile: rho=-0.641, p=1.360e-11, n=89
- State 1: n=58, cycles 2-158, future8 rate=0.259
- State 2: n=1, cycles 150-150, future8 rate=1.000
- State 0: n=29, cycles 112-149, future8 rate=0.138
- State 3: n=1, cycles 126-126, future8 rate=0.000
- Transition 0->2: n=1, next future8 rate=1.000, step norm=29.647
- Transition 2->1: n=1, next future8 rate=1.000, step norm=28.533
- Transition 1->0: n=1, next future8 rate=1.000, step norm=8.756
- Transition 1->1: n=56, next future8 rate=0.250, step norm=1.833
- Guardrail: Cycle state-space clusters use four-particle trace summaries and echem-shape descriptors at cycle resolution. They are degradation-state hypotheses and early-warning covariates, not localized ROI/front validation or calibrated diffusion measurements.

## Cycle Hazard Warning Audit

- Target/events: future_any_drop_within_8cycles / 4 event cycles [60.0, 86.0, 116.0, 156.0]
- Rolling-origin purge/min-train/permutation nulls: 8 / 24 / 20
- Feature set particle_trace_echem_with_acquisition: AUC 0.783, AP 0.573, Brier 0.313, balanced accuracy 0.789, n=62, positives=16
- Feature set particle_trace_echem_no_acquisition: AUC 0.742, AP 0.490, Brier 0.393, balanced accuracy 0.705, n=62, positives=16
- Feature set combined_with_cycle_state: AUC 0.739, AP 0.485, Brier 0.382, balanced accuracy 0.705, n=62, positives=16
- Feature set particle_trace: AUC 0.637, AP 0.402, Brier 0.290, balanced accuracy 0.579, n=62, positives=16
- Feature set echem_shape_cycle: AUC 0.306, AP 0.192, Brier 0.484, balanced accuracy 0.326, n=62, positives=16
- Null check: observed AUC 0.783, null p95 0.581, empirical p=0.048
- Lead-time 4 cycles: hit rate 0.500, median max probability 0.993, median lead 4.000 cycles
- Lead-time 8 cycles: hit rate 0.750, median max probability 1.000, median lead 6.000 cycles
- Lead-time 16 cycles: hit rate 0.750, median max probability 1.000, median lead 6.000 cycles
- Ablation remove particle_trace: AUC 0.492, drop vs best 0.291
- Ablation remove echem_shape: AUC 0.712, drop vs best 0.071
- Ablation remove acquisition_context: AUC 0.742, drop vs best 0.041
- Ablation remove cycle_echem: AUC 0.780, drop vs best 0.003
- Warning probability link cycle_state_pc2: rho=0.456, p=1.958e-04, n=62
- Warning probability link mean_abs_delta_prev: rho=-0.258, p=0.043, n=62
- Warning probability link shape_V_q95: rho=-0.113, p=0.414, n=54
- Warning probability link capacity_mAh: rho=-0.108, p=0.439, n=54
- Warning probability link cycle_state_pc1: rho=-0.081, p=0.532, n=62
- Guardrail: Rolling-origin cycle-level warning audit using particle trace/echem/state descriptors. It tests early-warning covariates, not localized ROI fronts, manual degradation labels, or calibrated diffusion.

## Cycle State To ROI/Front Bridge

- ROI rows/cycles joined: 52 / 11
- Predictors/targets: 9 / 13
- Row bridge cycle_state_pc2 vs physics_consistency_score: rho=0.702, permutation p=4.998e-04, n=52
- Row bridge cycle_state_pc2 vs kinetic_transition_score: rho=0.695, permutation p=4.998e-04, n=52
- Row bridge cycle_state_pc2 vs precursor_context_score: rho=0.687, permutation p=4.998e-04, n=47
- Row bridge cycle_state_pc3 vs kinetic_transition_score: rho=0.598, permutation p=4.998e-04, n=52
- Row bridge cycle_state_cluster vs kinetic_transition_score: rho=0.595, permutation p=4.998e-04, n=52
- Cycle-collapsed bridge cycle_state_pc2 vs mode_taxonomy_score: rho=0.855, permutation p=0.002, n=11
- Cycle-collapsed bridge cycle_state_pc2 vs physics_consistency_score: rho=0.836, permutation p=0.002, n=11
- Cycle-collapsed bridge cycle_state_pc2 vs kinetic_transition_score: rho=0.764, permutation p=0.009, n=11
- Cycle-collapsed bridge cycle_state_pc3 vs mode_taxonomy_score: rho=0.709, permutation p=0.021, n=11
- Cycle-collapsed bridge cycle_state_pc3 vs kinetic_transition_score: rho=0.682, permutation p=0.026, n=11
- Reference-centered bridge axis_step_ref_centered vs precursor_context_score_ref_centered: rho=0.790, permutation p=4.998e-04, n=47
- Reference-centered bridge state_step_norm_ref_centered vs precursor_context_score_ref_centered: rho=0.712, permutation p=4.998e-04, n=47
- Reference-centered bridge axis_step_ref_centered vs optical_change_score_ref_centered: rho=0.549, permutation p=4.998e-04, n=52
- Reference-centered bridge axis_step_ref_centered vs front_direction_score_ref_centered: rho=0.451, permutation p=4.998e-04, n=52
- Cycle-state cluster 1: ROI n=42, cycles=9, cross-modal priority fraction=0.238
- Cycle-state cluster 0: ROI n=10, cycles=2, cross-modal priority fraction=0.000
- Guardrail: Cycle-state to ROI/front bridge joins cycle-level state coordinates to selected automatic ROI rows. Row-level associations are not independent within cycle; reference-centered and cycle-collapsed tests are the stricter evidence. This does not create manual QC labels or calibrated diffusion claims.

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

## Weak-Label Degradation Benchmark

- ROI rows: 52
- Trainable weak-label rows: 7
- Positive/negative weak labels: 3 / 4
- Label counts: {'review_control_uncertain': 19, 'review_positive_uncertain': 15, 'review_uncertain': 11, 'weak_event_enriched_front_mode': 3, 'weak_low_consistency_control': 4}
- Leave-reference usable binary folds: 1 / 4
- Weak positive cycle156_rank7_obj27 (event, cycle 156): physics score 5.790, mode optical_brightening_decorrelating_rollout_hard_front_positive
- Weak positive cycle156_rank8_obj10 (event, cycle 156): physics score 3.925, mode optical_brightening_decorrelating_rollout_hard_front_positive
- Weak positive cycle60_rank6_obj26 (event, cycle 60): physics score 1.683, mode optical_brightening_decorrelating_rollout_hard_front_positive
- Weak negative cycle118_rank2_obj2 (control, cycle 118): physics score -3.012, mode near_baseline_or_context_like
- Weak negative cycle90_rank3_obj4 (control, cycle 90): physics score -1.988, mode front_negative_high_apparent_front_proxy
- Weak negative cycle58_rank1_obj1 (control, cycle 58): physics score -1.885, mode near_baseline_or_context_like
- Weak negative cycle90_rank4_obj6 (control, cycle 90): physics score -1.493, mode front_negative_high_apparent_front_proxy
- Split guardrail fold 2 holdout 86: weak_label_class_missing_in_train_or_test (0 positive / 2 negative test)
- Split guardrail fold 3 holdout 116: weak_label_class_missing_in_train_or_test (0 positive / 1 negative test)
- Split guardrail fold 4 holdout 156: weak_label_class_missing_in_train_or_test (2 positive / 0 negative test)
- Guardrail: This benchmark contains weak consensus labels for model development and review prioritization only. It is not a manual-QC label set and must not be used to claim validated degradation modes or calibrated diffusion.

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
- Select and guard particle-region-only ROIs: implemented_with_guardrail. Evidence: Model inputs use cropped ROI tensors and the particle-mask stability audit covers 52 ROI rows / 4992 frames; median fallback fraction is 0.000. Limitation: The audit uses automatic contrast/history masks and is not a manual segmentation of each particle boundary.
- Track phase-boundary movement: implemented_as_proxy. Evidence: Front/phase mobility descriptors, selected-front tracking, and threshold-robust sweeps exist; threshold sweep covers 52 ROI rows. Limitation: Front masks are automatic; after protocol/echem conditioning, front-direction sign consistency survives more strongly than front-magnitude metrics and is robust in 5 automatic QC strata.
- Extract diffusion coefficients: partial_proxy_only. Evidence: Provisional 0.096 um/px apparent diffusion proxies were computed and stress-tested across 7 thresholds with bootstrap slopes; the stricter diffusion sanity audit finds 0 automatic positive candidates and 0 publication candidates; the control-balanced high-resolution rerun tracks 40 ROIs and still finds 0 publication candidates. Limitation: Global threshold-robust phase slopes separate event/control ROIs, but QC-stratified diffusion proxies are inconsistent, conditioned diffusion-proxy residuals remain non-significant, and selected-front radius-squared slopes fail sign/fit/manual-QC gates.
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
