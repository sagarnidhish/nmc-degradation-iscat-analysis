# Source-Balanced Pre-Event Sequence Audit

- ROI sequences: 128 / manifest rows 128
- Cycles/sources: 64 / 14
- Bin counts: {'post_event_1_16': 40, 'near_pre_event_1_8': 32, 'mid_pre_event_9_16': 22, 'far_pre_event_17_32': 22, 'no_near_event_control': 12}
- Target positives: {'target_near_pre_vs_rest': 32, 'target_pre16_clean_vs_post_control': 54, 'target_any_pre_vs_post_control': 76}

## Top Scalar Tests

- target_any_pre_vs_post_control frame_diff_mse_p95 raw: AUC=0.632338056680162, AP=0.7778649138487013, p=0.011245551234578478
- target_any_pre_vs_post_control spatial_std_slope raw: AUC=0.6022267206477733, AP=0.6972092233902059, p=0.05027332425144095
- target_any_pre_vs_post_control dark_fraction_025_delta raw: AUC=0.5975455465587045, AP=0.6863939027582681, p=0.06160652806931172
- target_any_pre_vs_post_control frame_diff_mse_mean raw: AUC=0.5948886639676113, AP=0.7362899685083091, p=0.06922723901934166
- target_any_pre_vs_post_control roi_norm_mean_delta raw: AUC=0.5860323886639677, AP=0.6923871359410272, p=0.09953195053307463
- target_any_pre_vs_post_control temporal_gradient_signed_mean raw: AUC=0.5860323886639677, AP=0.6923871359410272, p=0.09953195053307463
- target_any_pre_vs_post_control bright_fraction_060_delta raw: AUC=0.5836285425101215, AP=0.6686967624039207, p=0.10889337621021267
- target_any_pre_vs_post_control roi_norm_mean_abs_step_p95 raw: AUC=0.5819838056680162, AP=0.74038795330825, p=0.11653125658978951

## Guardrail

Event-relative sequence features are computed from automatic fixed particle crops and labels are derived from abrupt-event cycle proximity. Positive readouts indicate pre/post/control optical-dynamics separation for follow-up QC and modeling, not validated precursors, particle identities, phase boundaries, diffusion coefficients, or causal degradation mechanisms.
