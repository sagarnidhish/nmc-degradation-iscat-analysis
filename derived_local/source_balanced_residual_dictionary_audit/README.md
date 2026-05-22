# Source-Balanced Residual Dictionary Audit

ROI sequences: 96 across 48 cycles and 14 sources.
Future8/future16 positive ROI sequences: 28 / 48.
PCA components: 16; explained residual variance sum: 0.522.

## Top Grouped Metrics
- cycleNo future_any_drop_within_16cycles residual_dictionary: AUC=0.602, AP=0.581, n=96
- cycleNo future_any_drop_within_16cycles mask_front_scalar: AUC=0.593, AP=0.578, n=96
- cycleNo future_any_drop_within_16cycles residual_dictionary_plus_mask_front: AUC=0.573, AP=0.555, n=96
- source_stem future_any_drop_within_16cycles mask_front_scalar: AUC=0.456, AP=0.466, n=96
- source_stem future_any_drop_within_16cycles residual_dictionary: AUC=0.375, AP=0.448, n=96
- source_stem future_any_drop_within_16cycles residual_dictionary_plus_mask_front: AUC=0.322, AP=0.396, n=96
- cycleNo future_any_drop_within_8cycles mask_front_scalar: AUC=0.641, AP=0.508, n=96
- cycleNo future_any_drop_within_8cycles residual_dictionary_plus_mask_front: AUC=0.614, AP=0.437, n=96
- cycleNo future_any_drop_within_8cycles residual_dictionary: AUC=0.600, AP=0.438, n=96
- source_stem future_any_drop_within_8cycles mask_front_scalar: AUC=0.418, AP=0.255, n=96

## Top Scalar Feature Tests
- future_any_drop_within_16cycles masked_minus_background_mean_slope: AUC=0.690, AP=0.696, eta2=0.634
- future_any_drop_within_16cycles resdict_pc01_mean: AUC=0.668, AP=0.670, eta2=0.187
- future_any_drop_within_16cycles resdict_pc02_slope: AUC=0.668, AP=0.719, eta2=0.199
- future_any_drop_within_16cycles resdict_pc09_slope: AUC=0.653, AP=0.674, eta2=0.150
- future_any_drop_within_16cycles resdict_pc03_mean: AUC=0.641, AP=0.666, eta2=0.329
- future_any_drop_within_16cycles resdict_pc02_last_minus_first: AUC=0.637, AP=0.617, eta2=0.112
- future_any_drop_within_16cycles resdict_pc09_last_minus_first: AUC=0.629, AP=0.640, eta2=0.114
- future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC=0.626, AP=0.589, eta2=0.365
- future_any_drop_within_16cycles resdict_pc12_mean: AUC=0.613, AP=0.629, eta2=0.066
- future_any_drop_within_16cycles front_radius_q80_slope_px_per_norm_time: AUC=0.611, AP=0.615, eta2=0.104

## Guardrail
Residual dictionary bases are label-free PCA summaries of automatic source-balanced ROI crops. They are useful for ranking dynamics hypotheses, not a trained deployable predictor or calibrated physics model.
