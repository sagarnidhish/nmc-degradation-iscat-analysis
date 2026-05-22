# Source-Balanced Expansion Transport/Front Audit

Apparent optical-flow transport and front-radius proxy audit on the expanded source-balanced ROI sequence cohort.

- Rows OK/failed: 96 / 0
- Cycles/sources: 48 / 14
- Future8/future16 positive rows: 28 / 48
- Flow method: farneback

## Top Feature Tests

- future_any_drop_within_16cycles gradient_aligned_flow_slope raw: AUC 0.6662326388888888, AP 0.6834679080096558, source-stratified p 0.138
- future_any_drop_within_16cycles divergence_q90 source_residual: AUC 0.6419270833333333, AP 0.7226016256741855, source-stratified p 0.034
- future_any_drop_within_16cycles gradient_aligned_flow_mean raw: AUC 0.6046006944444444, AP 0.5414173587493619, source-stratified p 0.246
- future_any_drop_within_16cycles context_flow_mag_slope raw: AUC 0.5933159722222222, AP 0.6185550194891046, source-stratified p 0.224
- future_any_drop_within_16cycles divergence_slope source_residual: AUC 0.5928819444444444, AP 0.5559653241999977, source-stratified p 0.066
- future_any_drop_within_16cycles divergence_slope raw: AUC 0.5902777777777778, AP 0.5551569633842665, source-stratified p 0.22
- future_any_drop_within_16cycles radial_flow_mean raw: AUC 0.5833333333333333, AP 0.5548716175483217, source-stratified p 0.458
- future_any_drop_within_16cycles curl_abs_slope raw: AUC 0.5807291666666667, AP 0.6372613535290399, source-stratified p 0.08
- future_any_drop_within_16cycles intensity_delta_abs_slope source_residual: AUC 0.5716145833333334, AP 0.6126657869053757, source-stratified p 0.336
- future_any_drop_within_16cycles expansion_transport_front_score source_residual: AUC 0.5716145833333333, AP 0.6099185950542106, source-stratified p 0.012

## Top Candidates

- source_balanced_cycle108_rank6_obj1_12_c2_x10_070723: score 0.8916666666666666, future8=1.0, future16=1.0
- source_balanced_cycle152_rank23_obj1_17_c2_x10_HighHighCOV_150723: score 0.8557291666666667, future8=1.0, future16=1.0
- source_balanced_cycle6_rank25_obj1_2_c2_x14_200623: score 0.85, future8=0.0, future16=0.0
- source_balanced_cycle84_rank48_obj2_9_c2_x10_010723: score 0.8380208333333333, future8=1.0, future16=1.0
- source_balanced_cycle142_rank18_obj2_16_c2_x10_HighHighCOV_130723: score 0.8026041666666666, future8=0.0, future16=1.0
- source_balanced_cycle152_rank23_obj2_17_c2_x10_HighHighCOV_150723: score 0.8, future8=1.0, future16=1.0
- source_balanced_cycle110_rank7_obj1_12_c2_x10_070723: score 0.7791666666666668, future8=1.0, future16=1.0
- source_balanced_cycle125_rank11_obj1_14_c2_x10_HighCOV_110723: score 0.7755208333333334, future8=0.0, future16=0.0

## Guardrail

Expanded-cohort apparent transport/front descriptors are computed from automatic fixed ROI crops. They test generalization of image-motion and front-radius proxies across broader source-balanced cycles, but they are not manual particle labels, calibrated transport, diffusion coefficients, or degradation causality.
