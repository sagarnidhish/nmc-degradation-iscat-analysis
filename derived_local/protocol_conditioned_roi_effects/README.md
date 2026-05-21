# Protocol-Conditioned ROI Effects

Residualized ROI optical/rollout descriptors against available protocol/echem covariates and event-reference-cycle fixed effects, then retested event/control separation.

- Source rows: 52
- Event ROI: 24
- Control ROI: 28
- Residualized features: 19
- Covariates: n_frames_percentile, cycles_from_block_start, cycles_to_block_end, block_fraction_elapsed, V_mean, I_mean_mA, I_abs_mean_mA, duration_s

## Top Adjusted Event-Control Tests

- roi_norm_mean_delta_protocol_residual: event-control 0.003224, p=4.424e-05
- high_fraction_delta_protocol_residual: event-control 0.01455, p=8.238e-05
- low_fraction_delta_protocol_residual: event-control -0.01504, p=0.0004707
- first_last_corr_protocol_residual: event-control -0.0184, p=0.01927
- cumulative_abs_norm_change_protocol_residual: event-control 0.003864, p=0.01927
- dmd_minus_persistence_mse_protocol_residual: event-control 0.0004515, p=0.04069
- latent_net_displacement_protocol_residual: event-control 0.3768, p=0.04253
- low_rank_dmd_mse_protocol_residual: event-control 0.0004593, p=0.04642

## Residual Classifier

- Leave-event-reference-out logistic mean ROC-AUC: 0.672
- Leave-event-reference-out logistic mean balanced accuracy: 0.682

## Guardrail

Residualization reduces measured context effects but cannot prove causal event physics with 52 automatically selected ROIs.
