# Source-Balanced Sequence Source-Control Audit

Source/cycle-control stress test for source-balanced particle-region rollout features.

- ROI rows/cycles/sources: 96 / 48 / 14
- Source/cycle-stratified permutations per scalar test: 1000
- Verdict: not_source_controlled_predictive;use_for_review_negative_controls
- Strict scalar rows: 0
- Source-heldout model rows with AUC >= 0.65: 0

## Top Source-Stratified Scalars
- future_any_drop_within_16cycles temporal_energy_late_minus_early (raw): AUC 0.561, source p 0.002, AP 0.548
- future_any_drop_within_16cycles temporal_energy_late_minus_early (within_source_z): AUC 0.591, source p 0.003, AP 0.570
- future_any_drop_within_16cycles temporal_energy_late_minus_early (source_residual): AUC 0.615, source p 0.004, AP 0.606
- future_any_drop_within_16cycles temporal_energy_late_minus_early (within_source_rank): AUC 0.573, source p 0.006, AP 0.551
- future_any_drop_within_16cycles roi_norm_mean_positive_step_fraction (raw): AUC 0.602, source p 0.013, AP 0.603
- future_any_drop_within_8cycles stage_drift_xy_recomputed (within_source_rank): AUC 0.543, source p 0.028, AP 0.322
- future_any_drop_within_8cycles stage_drift_xy_recomputed (raw): AUC 0.534, source p 0.039, AP 0.348
- future_any_drop_within_8cycles persistence_mse_p95 (within_source_rank): AUC 0.553, source p 0.045, AP 0.329

## Top Source-Heldout Models
- future_any_drop_within_16cycles rollout_raw: AUC 0.639, AP 0.656, n 96
- future_any_drop_within_16cycles rollout_raw_plus_context: AUC 0.604, AP 0.639, n 96
- future_any_drop_within_16cycles rollout_within_source_z: AUC 0.533, AP 0.551, n 96
- future_any_drop_within_16cycles rollout_source_residual: AUC 0.529, AP 0.560, n 96
- future_any_drop_within_16cycles rollout_within_source_rank: AUC 0.497, AP 0.514, n 96
- future_any_drop_within_8cycles rollout_source_residual: AUC 0.489, AP 0.381, n 96
- future_any_drop_within_8cycles rollout_within_source_z: AUC 0.445, AP 0.291, n 96
- future_any_drop_within_8cycles rollout_within_source_rank: AUC 0.437, AP 0.312, n 96

## Guardrail
This audit stress-tests source-balanced rollout descriptors under source/cycle controls and weak future-drop labels. Within-source transforms are useful for review and negative-control design, but are not prospective source-transfer models. Results do not assign manual QC labels, validate degradation mechanisms, or calibrate diffusion.
