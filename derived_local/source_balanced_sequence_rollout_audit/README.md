# Source-Balanced Sequence Rollout Audit

Fast ROI-only temporal prediction and dynamics features for the source-balanced particle crop cohort.

- ROI sequences: 96
- Cycles: 48
- Sources: 14
- Future8 positive sequences: 28
- Future16 positive sequences: 48

## Top ROI Feature Tests

- future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.6258680555555555, AP 0.5892585924909939, source eta2 0.365275723531651
- future_any_drop_within_16cycles raw_roi_mean_delta_last_minus_first: AUC 0.6059027777777778, AP 0.5492814723006487, source eta2 0.5335359243634783
- future_any_drop_within_16cycles roi_norm_mean_positive_step_fraction: AUC 0.6024305555555556, AP 0.6030268835081076, source eta2 0.2326142302023308
- future_any_drop_within_16cycles stage_drift_xy_recomputed: AUC 0.5625000000000001, AP 0.5901497821350763, source eta2 0.334536996595898
- future_any_drop_within_16cycles temporal_energy_late_minus_early: AUC 0.5607638888888888, AP 0.547762690487468, source eta2 0.293601502621033
- future_any_drop_within_16cycles persistence_mse_late_mean: AUC 0.5564236111111112, AP 0.6186599590041513, source eta2 0.7870212701334101
- future_any_drop_within_16cycles velocity_mse_p95: AUC 0.5555555555555555, AP 0.6158099984034713, source eta2 0.717289190438265
- future_any_drop_within_16cycles velocity_minus_persistence_mse: AUC 0.5499131944444444, AP 0.609137488776376, source eta2 0.8250007218155716
- future_any_drop_within_16cycles persistence_mse_p95: AUC 0.5477430555555556, AP 0.6143705150395018, source eta2 0.726596695686589
- future_any_drop_within_16cycles velocity_mse_mean: AUC 0.5399305555555556, AP 0.6049250416307346, source eta2 0.8242515582666822
- future_any_drop_within_16cycles temporal_energy_mean: AUC 0.5347222222222222, AP 0.6043464637209901, source eta2 0.7429485346702067
- future_any_drop_within_16cycles object_mean_abs_z: AUC 0.5264756944444444, AP 0.5098387014615868, source eta2 0.2683029446959361

## Top Cycle Feature Tests

- future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.6475694444444444, AP 0.5940168774046461
- future_any_drop_within_16cycles stage_drift_xy_recomputed: AUC 0.5625000000000001, AP 0.5901497821350763
- future_any_drop_within_16cycles object_mean_abs_z: AUC 0.5625, AP 0.5588909106318675
- future_any_drop_within_16cycles temporal_energy_late_minus_early: AUC 0.5538194444444444, AP 0.5562730306944456
- future_any_drop_within_16cycles velocity_mse_mean: AUC 0.5190972222222222, AP 0.6037099242770925
- future_any_drop_within_16cycles persistence_mse_mean: AUC 0.5017361111111112, AP 0.5981577065105004
- future_any_drop_within_16cycles temporal_energy_mean: AUC 0.47916666666666663, AP 0.5038576225475571
- future_any_drop_within_8cycles object_mean_abs_z: AUC 0.6323529411764706, AP 0.3632226771213945

## Guardrail

Source-balanced rollout features are computed from automatic particle-centered crops and weak future labels. They quantify ROI-only temporal prediction difficulty and optical drift/intensity dynamics, not manual QC, causal degradation, or calibrated diffusion.
