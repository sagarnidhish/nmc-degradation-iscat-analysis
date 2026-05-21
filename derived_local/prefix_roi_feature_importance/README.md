# Prefix ROI Feature Importance

Interpretability audit for early cropped particle-ROI features predicting later positive phase-front residual direction.

- ROI rows: 52
- Prefix fraction: 0.75
- Target: front_positive_residual_binary
- Pooled OOF AUC: 0.447
- Null p>=observed: 0.7143

## Top Feature Groups
- remove mean_intensity_trace: AUC drop=0.171, n_features=18
- remove frame_texture_level: AUC drop=-0.002, n_features=5
- remove stage_drift: AUC drop=-0.023, n_features=3
- remove temporal_change_energy: AUC drop=-0.048, n_features=14
- remove bright_dark_fraction: AUC drop=-0.074, n_features=14

## Top Permutation Features
- stage_prefix_step_max (stage_drift): AUC drop=0.132, positive-fold fraction=0.50
- low_fraction_prefix_std (bright_dark_fraction): AUC drop=0.009, positive-fold fraction=0.25
- average_intensity_prefix_mean (mean_intensity_trace): AUC drop=0.008, positive-fold fraction=0.25
- average_intensity_prefix_last (mean_intensity_trace): AUC drop=0.008, positive-fold fraction=0.25
- roi_norm_mean_prefix_first (mean_intensity_trace): AUC drop=0.007, positive-fold fraction=0.50
- first_q30_threshold (bright_dark_fraction): AUC drop=0.006, positive-fold fraction=0.25
- high_fraction_prefix_slope (bright_dark_fraction): AUC drop=0.006, positive-fold fraction=0.50
- first_frame_mean (frame_texture_level): AUC drop=0.006, positive-fold fraction=0.50

## Guardrail

Feature importance is computed on the small 52-ROI selected cohort with leave-event-reference-cycle-out folds. Treat it as mechanistic triage for early particle-region video signals, not causal proof or a deployable detector.
