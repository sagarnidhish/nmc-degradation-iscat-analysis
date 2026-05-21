# Protocol-Conditioned Front Effects

Residualized threshold-robust front metrics against protocol/echem context and event-reference-cycle fixed effects.

- Source rows: 52
- Event ROI: 24
- Control ROI: 28
- Residualized front features: 17
- Covariates: n_frames_percentile, cycles_from_block_start, cycles_to_block_end, block_fraction_elapsed, V_mean, I_mean_mA, I_abs_mean_mA, duration_s, timing_elapsed_s

## Top Adjusted Event-Control Tests

- phase_slope_positive_fraction_protocol_residual: event-control 0.06396, p=0.0008252
- phase_slope_negative_fraction_protocol_residual: event-control -0.06396, p=0.0008252
- threshold_robust_phase_score_protocol_residual: event-control 0.08979, p=0.3934
- diffusion_proxy_iqr_um2_per_s_protocol_residual: event-control -3.988e-07, p=0.4036
- radius2_slope_iqr_px2_per_s_protocol_residual: event-control -0.0001731, p=0.4036
- phase_slope_iqr_per_s_protocol_residual: event-control -9.686e-08, p=0.414
- radius2_slope_negative_fraction_protocol_residual: event-control -0.09879, p=0.4682
- radius2_slope_positive_fraction_protocol_residual: event-control 0.09879, p=0.4682

## Residual Classifier

- Leave-event-reference-out logistic mean ROC-AUC: 0.453
- Leave-event-reference-out logistic mean balanced accuracy: 0.312

## Guardrail

Residualized threshold-front effects test whether front metrics survive protocol/echem context. They remain automatic ROI/front proxies and are not calibrated diffusion coefficients.
