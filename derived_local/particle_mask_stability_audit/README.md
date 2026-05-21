# Particle Mask Stability Audit

This folder audits a history-aware particle-mask guardrail on the existing ROI-only tensors.

The candidate mask is built from frame-local contrast plus temporal-standard-deviation evidence inside a sequence-level central prior. Frames with implausible area, fragmentation, or centroid jumps fall back to the previous accepted mask.

Outputs:

- `particle_mask_stability_per_roi.csv`: ROI-level fallback and stability metrics.
- `particle_mask_stability_frame_summary.csv`: per-frame candidate and accepted mask diagnostics.
- `particle_mask_stability_event_control_tests.csv`: event/control Mann-Whitney tests.
- `particle_mask_stability_correlations.csv`: Spearman correlations with rollout/front descriptors.
- `particle_mask_stability_audit_summary.json`: compact project summary.
