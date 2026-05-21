# ROI Trace Fusion Cycle Null

Cycle-collapsed permutation audit for lagged trace/front associations.

- ROI rows collapsed: 52 -> 11 cycle points
- Event-reference cycles: 4
- Predictors tested: 68
- Permutations per test: 1000

## Top Cycle-Collapsed Tests
- lag8_trace_predprob_future_any_drop_within_8cycles vs mode_review_priority: rho=-0.827, empirical p=0.003, n=11
- trace_lag8_n_frames vs mode_review_priority: rho=0.834, empirical p=0.004, n=11
- trace_lag16_n_frames vs mode_review_priority: rho=0.813, empirical p=0.004, n=11
- lag4_trace_predprob_future_any_drop_within_8cycles vs q70_logistic_k_per_s: rho=-0.8, empirical p=0.004, n=11
- trace_lag8_frames_percentile vs mode_review_priority: rho=0.834, empirical p=0.005, n=11
- trace_lag16_capacity_mAh vs q70_logistic_k_per_s: rho=0.755, empirical p=0.00599, n=11
- trace_lag16_frames_percentile vs mode_review_priority: rho=0.813, empirical p=0.00699, n=11
- trace_lag4_frames_percentile vs mode_review_priority: rho=0.745, empirical p=0.00699, n=11
- trace_lag16_frames_percentile vs q70_logistic_k_per_s: rho=0.74, empirical p=0.00999, n=11
- trace_lag2_frames_percentile vs mode_review_priority: rho=0.733, empirical p=0.00999, n=11

## Top Reference-Centered Tests
- trace_lag8_particle_norm_mean vs phase_slope_median_per_s_protocol_residual: rho=0.918, empirical p=0.000999, n=11
- trace_lag8_n_frames vs mode_review_priority: rho=0.881, empirical p=0.002, n=11
- trace_lag8_frames_percentile vs mode_review_priority: rho=0.858, empirical p=0.002, n=11
- lag16_trace_predprob_future_any_drop_within_8cycles vs phase_slope_positive_fraction_protocol_residual: rho=-0.809, empirical p=0.004, n=11
- trace_lag16_frames_percentile vs threshold_robust_phase_score_protocol_residual: rho=0.781, empirical p=0.00799, n=11
- trace_lag16_mean_abs_delta_prev vs mode_review_priority: rho=0.755, empirical p=0.00899, n=11
- trace_lag2_particle_norm_mean vs mode_review_priority: rho=0.755, empirical p=0.00899, n=11
- lag16_trace_predprob_future_any_drop_within_8cycles vs threshold_robust_phase_score_protocol_residual: rho=-0.745, empirical p=0.00999, n=11
- trace_lag2_capacity_mAh vs q70_logistic_k_per_s: rho=-0.833, empirical p=0.00999, n=8
- trace_lag8_delta_std_across_particles vs threshold_robust_phase_score_protocol_residual: rho=0.745, empirical p=0.015, n=11

## Guardrail

This audit collapses repeated ROI rows to one median point per cycle before testing trace/front associations. It is deliberately conservative for the 52-ROI cohort; surviving tests are stronger evidence, while lost tests indicate cycle-clustering sensitivity.
