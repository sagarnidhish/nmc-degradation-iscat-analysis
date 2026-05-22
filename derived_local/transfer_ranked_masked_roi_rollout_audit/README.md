# Masked ROI Rollout Audit

This audit reruns the existing simple rollout baseline scoring inside the accepted particle support from the history-aware mask guardrail.

- ROI tensors: 48
- Frame metric rows: 4608
- Best-method counts inside particle masks: {'persistence': 48}

Outputs:

- `masked_roi_rollout_frame_metrics.csv`: held-out frame metrics by ROI, method, and mask region.
- `masked_roi_rollout_per_roi_metrics.csv`: ROI-method aggregates.
- `masked_roi_rollout_method_ratios.csv`: particle-MSE ratios versus persistence.
- `masked_roi_rollout_event_control_tests.csv`: event/control tests on masked errors.
- `masked_roi_rollout_correlations.csv`: links between masked rollout errors and mask/front descriptors.
- `masked_roi_rollout_audit_summary.json`: compact summary.
