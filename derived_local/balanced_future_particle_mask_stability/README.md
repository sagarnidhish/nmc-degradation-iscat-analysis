# Particle Mask Stability Audit

This folder audits a history-aware particle-mask guardrail on the existing ROI-only tensors.

The candidate mask is built from frame-local contrast plus temporal-standard-deviation evidence inside a sequence-level central prior. Frames with implausible area, fragmentation, or centroid jumps fall back to the previous accepted mask.

Key results:

- ROI tensors audited: 72 across 6912 frames.
- Median fallback-frame fraction: 0.000.
- Median accepted-mask area CV: 0.074.
- Median accepted-centroid path: 133.392 px.
- Median mask-instability score: 1.491.
- Strongest event/control mask-stability test: fallback_frame_fraction median event-control nan, p=nan.

Interpretation:

The audit is a mask-stability guardrail, not manual segmentation. Current automatic ROI masks do not show a systematic event/control instability difference, so the robust phase-front direction signal is less likely to be explained by event ROIs having worse particle masks.

Outputs:

- `particle_mask_stability_per_roi.csv`: ROI-level fallback and stability metrics.
- `particle_mask_stability_frame_summary.csv`: per-frame candidate and accepted mask diagnostics.
- `particle_mask_stability_event_control_tests.csv`: event/control Mann-Whitney tests.
- `particle_mask_stability_correlations.csv`: Spearman correlations with rollout/front descriptors.
- `particle_mask_stability_audit_summary.json`: compact project summary.
