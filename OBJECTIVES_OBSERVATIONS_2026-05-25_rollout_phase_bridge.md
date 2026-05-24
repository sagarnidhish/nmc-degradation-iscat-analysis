# Objectives And Observations (Addendum)

Date: 2026-05-25

## New Experiment

Rollout / phase-kinetics bridge audit.

Purpose:

- join source-balanced rollout difficulty metrics to source-balanced optical phase-kinetics descriptors;
- test whether persistence/velocity error and temporal energy track masked-minus-background slope, phase-fraction slopes, and Avrami/logistic descriptors;
- separate raw effects from source confounding with source-residual correlations.

Why this matters:

- rollout residuals are a natural measure of how hard the particle-region dynamics are to predict;
- phase-kinetics features are a compact proxy for boundary propagation and reaction-front shape;
- correlating the two can show whether the model difficulty is driven by physically meaningful motion instead of only acquisition noise.

Planned outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_phase_kinetics_bridge/rollout_phase_kinetics_bridge_merged.csv`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_phase_kinetics_bridge/rollout_phase_kinetics_bridge_correlations.csv`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_phase_kinetics_bridge/rollout_phase_kinetics_bridge_source_residual_correlations.csv`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_phase_kinetics_bridge/rollout_phase_kinetics_bridge_event_tests.csv`

Guardrail:

This is an association audit over automatic particle-only crops. It is intended for follow-up hypothesis ranking, not for manual phase-boundary labels or calibrated diffusion claims.

