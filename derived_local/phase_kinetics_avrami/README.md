# Phase Kinetics Avrami Audit

Optical phase-fraction kinetic summaries from cropped NMC particle ROI movies.

- ROI rows: 52
- Kinetic features: 40

## Top Group Tests
- event_enriched_mode_vs_other q60_logistic_k_per_s: median diff=0.000266, p=0.01373
- event_enriched_mode_vs_other q70_fraction_delta: median diff=0.00632, p=0.01694
- event_enriched_mode_vs_other q70_transformed_fraction_delta: median diff=0.00632, p=0.01694
- event_vs_control q70_avrami_r2: median diff=0.0452, p=0.0223
- event_vs_control q80_time_of_max_abs_rate_frac: median diff=-0.132, p=0.02388
- event_vs_control q60_avrami_r2: median diff=0.0253, p=0.02827
- event_enriched_mode_vs_other q80_positive_rate_fraction: median diff=0.0208, p=0.03241
- event_enriched_mode_vs_other q80_time_of_max_abs_rate_frac: median diff=-0.2, p=0.03358

## Top Correlations
- q70_logistic_r2 vs n_frames_percentile: rho=-0.801, p=2.796e-12
- q60_logistic_r2 vs n_frames_percentile: rho=-0.776, p=9.625e-11
- roi_norm_rate_sign_consistency vs n_frames_percentile: rho=0.728, p=9.758e-10
- roi_norm_max_abs_rate_per_s vs n_frames_percentile: rho=0.701, p=7.281e-09
- q80_max_abs_rate_per_s vs n_frames_percentile: rho=0.699, p=8.018e-09
- q60_max_abs_rate_per_s vs n_frames_percentile: rho=0.690, p=1.525e-08
- q70_max_abs_rate_per_s vs n_frames_percentile: rho=0.686, p=2.043e-08
- q70_variation_to_net_abs vs n_frames_percentile: rho=-0.675, p=4.098e-08

## Guardrail

Kinetic fits are optical phase-fraction proxies from cropped particle ROIs using provisional timing. Avrami/logistic parameters are descriptive and not calibrated reaction constants.
