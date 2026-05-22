# Source-Balanced Sequence Rollout Audit

Fast ROI-only temporal prediction and dynamics features for the source-balanced particle crop cohort.

- ROI sequences: 128
- Cycles: 64
- Sources: 14
- Future8 positive sequences: 32
- Future16 positive sequences: 66

## Top ROI Feature Tests

- future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.6280547409579668, AP 0.6203479026636833, source eta2 0.413983768799865
- future_any_drop_within_16cycles raw_roi_mean_delta_last_minus_first: AUC 0.6265884652981427, AP 0.6238243824564498, source eta2 0.4591139102176846
- future_any_drop_within_16cycles temporal_energy_late_minus_early: AUC 0.6055718475073314, AP 0.6531597889170409, source eta2 0.2823313068811859
- future_any_drop_within_16cycles temporal_energy_p95: AUC 0.5843108504398827, AP 0.685731575966003, source eta2 0.8374275676480238
- future_any_drop_within_16cycles roi_norm_mean_positive_step_fraction: AUC 0.5801564027370478, AP 0.6199349849450737, source eta2 0.12356133059905998
- future_any_drop_within_16cycles persistence_mse_late_mean: AUC 0.5760019550342131, AP 0.6697149887810505, source eta2 0.6646846224893593
- future_any_drop_within_16cycles persistence_mse_mean: AUC 0.5562072336265884, AP 0.6497151951426368, source eta2 0.7541982340374864
- future_any_drop_within_16cycles velocity_mse_mean: AUC 0.5542521994134897, AP 0.633698895929802, source eta2 0.783022062622372
- future_any_drop_within_16cycles object_mean_abs_z: AUC 0.5540078201368523, AP 0.5287839547855898, source eta2 0.24062877316727443
- future_any_drop_within_16cycles velocity_minus_persistence_mse: AUC 0.5532746823069403, AP 0.6304795618641398, source eta2 0.7970380411936245
- future_any_drop_within_16cycles temporal_energy_mean: AUC 0.5505865102639296, AP 0.6329928973957843, source eta2 0.7281690036904107
- future_any_drop_within_16cycles object_area_ds_px: AUC 0.5359237536656891, AP 0.5142867487630473, source eta2 0.10986075143354154

## Top Cycle Feature Tests

- future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.6852394916911047, AP 0.6504161796298411
- future_any_drop_within_16cycles temporal_energy_late_minus_early: AUC 0.6041055718475073, AP 0.6607888194743683
- future_any_drop_within_16cycles object_mean_abs_z: AUC 0.5962854349951124, AP 0.583938038279014
- future_any_drop_within_16cycles persistence_mse_mean: AUC 0.533724340175953, AP 0.6450247215196919
- future_any_drop_within_16cycles velocity_mse_mean: AUC 0.530791788856305, AP 0.631364677244829
- future_any_drop_within_16cycles temporal_energy_mean: AUC 0.5259042033235581, AP 0.6275366981195754
- future_any_drop_within_16cycles stage_drift_xy_recomputed: AUC 0.5259042033235581, AP 0.5701276624439541
- future_any_drop_within_8cycles object_mean_abs_z: AUC 0.6354166666666667, AP 0.31745602814044177

## Guardrail

Source-balanced rollout features are computed from automatic particle-centered crops and weak future labels. They quantify ROI-only temporal prediction difficulty and optical drift/intensity dynamics, not manual QC, causal degradation, or calibrated diffusion.
