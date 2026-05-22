# Source-Balanced Future-Specific Residual Audit

Rows/cycles/sources: 96 / 48 / 14
Event cycles: [60.0, 86.0, 116.0, 156.0]
Label counts: {'current_any_event': 0, 'future8': 28, 'future16': 48, 'past8': 10, 'past16': 30, 'no_past8_rows': 86, 'no_past16_rows': 66}

## Primary Source-Residual Candidate
- all_rows: future16 AUC=0.637, AP=0.637, n=96, positives=48
- exclude_past16: future16 AUC=0.589, AP=0.708, n=66, positives=38
- pre_first_event: future16 AUC=0.541, AP=0.329, n=32, positives=10

## Top Model Deltas vs Past-Event Context
- cycleNo full_future_any_event_within_16cycles mask_contrast_source_residual_plus_context: delta AUC=0.065, AUC=0.820, base=0.755
- cycleNo full_future_any_event_within_16cycles primary_source_residual_plus_context: delta AUC=0.055, AUC=0.810, base=0.755
- cycleNo full_future_any_event_within_16cycles primary_rank_plus_context: delta AUC=0.022, AUC=0.777, base=0.755
- source_stem full_future_any_event_within_16cycles mask_contrast_source_residual_plus_context: delta AUC=0.017, AUC=0.737, base=0.720
- source_stem full_future_any_event_within_16cycles primary_source_residual_plus_context: delta AUC=0.004, AUC=0.724, base=0.720
- source_stem full_future_any_event_within_8cycles q80_front_source_residual_plus_context: delta AUC=0.003, AUC=0.819, base=0.815
- source_stem full_future_any_event_within_16cycles primary_rank_plus_context: delta AUC=0.003, AUC=0.723, base=0.720
- source_stem full_future_any_event_within_8cycles primary_rank_plus_context: delta AUC=0.002, AUC=0.817, base=0.815

## Guardrail
Past-event context uses event-cycle labels derived from full particle abrupt-drop cycles. Excluding or modeling past windows tests future specificity on the source-balanced ROI table; it does not prove causality, source transfer, or calibrated phase/diffusion physics.
