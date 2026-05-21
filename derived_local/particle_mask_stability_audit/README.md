# Particle Mask Stability Audit

This folder audits a history-aware particle-mask guardrail on the existing ROI-only tensors.

The candidate mask is built from frame-local contrast plus temporal-standard-deviation evidence inside a sequence-level central prior. Frames with implausible area, fragmentation, or centroid jumps fall back to the previous accepted mask.

Key results:

- ROI tensors audited: 52 across 4992 frames.
- Median fallback-frame fraction: 0.000.
- Median accepted-mask area CV: 0.042.
- Median accepted-centroid path: 73.607 px.
- Median mask-instability score: 0.809.
- Strongest event/control mask-stability test: fallback_frame_fraction median event-control 0.000, p=0.298.
- Strongest descriptor correlation: accepted_centroid_path_px vs high_fraction_slope_per_s rho=-0.370, p=0.00695.

Interpretation:

The audit is a mask-stability guardrail, not manual segmentation. Current automatic ROI masks do not show a systematic event/control instability difference, so the robust phase-front direction signal is less likely to be explained by event ROIs having worse particle masks.

Outputs:

- `particle_mask_stability_per_roi.csv`: ROI-level fallback and stability metrics.
- `particle_mask_stability_frame_summary.csv`: per-frame candidate and accepted mask diagnostics.
- `particle_mask_stability_event_control_tests.csv`: event/control Mann-Whitney tests.
- `particle_mask_stability_correlations.csv`: Spearman correlations with rollout/front descriptors.
- `particle_mask_stability_audit_summary.json`: compact project summary.
