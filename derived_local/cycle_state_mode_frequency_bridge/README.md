# Cycle-State Mode-Frequency Bridge

Cycle-level bridge from echem/cycle-state transition descriptors to automatic ROI degradation-mode composition.

- Cycles: 11
- ROI rows represented: 52
- Mode-frequency targets: 4
- Best macro model: cycle_state_only MAE 0.261
- Context macro MAE reduction: 0.043

## Feature Sets

- context_only: 28 features
- cycle_state_only: 12 features
- echem_only: 61 features
- cycle_state_plus_context: 40 features
- echem_plus_context: 84 features
- cycle_state_echem_context: 96 features

## Top Metrics

- cycle_state_only -> macro_mode_fraction: MAE 0.261, R2 NA, rho NA
- context_only -> macro_mode_fraction: MAE 0.303, R2 NA, rho NA
- cycle_state_plus_context -> macro_mode_fraction: MAE 0.348, R2 NA, rho NA
- cycle_state_echem_context -> macro_mode_fraction: MAE 0.395, R2 NA, rho NA
- echem_only -> macro_mode_fraction: MAE 0.396, R2 NA, rho NA
- echem_plus_context -> macro_mode_fraction: MAE 0.409, R2 NA, rho NA
- cycle_state_only -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.312, R2 -1.251, rho -0.196
- context_only -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.328, R2 -1.272, rho -0.182
- cycle_state_plus_context -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.390, R2 -2.075, rho -0.225
- echem_only -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.603, R2 -8.539, rho -0.569
- cycle_state_echem_context -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.605, R2 -7.048, rho -0.440
- echem_plus_context -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.626, R2 -8.038, rho -0.507

## Permutation Null

- cycle_state_only: observed macro MAE 0.261, null mean 0.290, p=0.381
- echem_plus_context: observed macro MAE 0.409, null mean 0.435, p=0.381
- context_only: observed macro MAE 0.303, null mean 0.343, p=0.429
- cycle_state_echem_context: observed macro MAE 0.395, null mean 0.391, p=0.524
- cycle_state_plus_context: observed macro MAE 0.348, null mean 0.344, p=0.667
- echem_only: observed macro MAE 0.396, null mean 0.377, p=0.714

## Guardrail

Cycle-state mode-frequency bridge predicts automatic ROI mode composition at cycle resolution from cycle/echem descriptors. It is a degradation-mode organization audit, not manual QC, causal proof, or calibrated diffusion validation.
