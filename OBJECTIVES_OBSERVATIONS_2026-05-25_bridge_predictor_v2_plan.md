# Rollout / Phase Kinetics Bridge and Predictor v2

## Why v2 exists

The first bridge and predictor attempts matched exact ROI ids and produced no usable overlap because the rollout and phase-kinetics audits were built from related but not identical ROI exports. V2 joins at the shared source/cycle level.

## Working hypothesis

If phase kinetics adds signal beyond rollout difficulty, then a source-held-out model should improve event-risk ranking when phase and rollout features are combined, and the bridge should show stable source/cycle associations instead of an empty join.

## Outputs to inspect

- `derived/rollout_phase_kinetics_bridge_v2/rollout_phase_kinetics_bridge_v2_summary.json`
- `derived/rollout_phase_kinetics_predictor_v2/rollout_phase_kinetics_predictor_v2_summary.json`
- `derived/rollout_phase_kinetics_predictor_v2/rollout_phase_kinetics_predictor_v2_metrics.csv`

## Guardrail

These are association and ranking audits over automatic particle-only features. They are for hypothesis ranking, not diffusion calibration or deployment.
