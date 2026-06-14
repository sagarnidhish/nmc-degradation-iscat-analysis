# Objectives And Observations (Addendum)

Date: 2026-05-25

## New Experiment

Source-heldout rollout/phase-kinetics predictor.

Purpose:

- compare rollout-only, phase-only, and combined feature families on future-event labels;
- test whether phase-fraction / Avrami / front-gradient descriptors add event-risk signal beyond raw rollout difficulty;
- keep the split by source to avoid memorizing one dataset family.

Why this is useful:

- if the combined model beats rollout-only, then the rollout residuals are not just pixel noise;
- if phase-only already wins, the physics signal is in the kinetics descriptors;
- if combined wins by a lot, then the right path is a multimodal physics-aware predictor rather than a pure next-frame model.

Planned outputs:

- `/scratch/<account>/<username>/Alek_Jiho/derived/rollout_phase_kinetics_predictor/rollout_phase_kinetics_predictor_metrics.csv`
- `/scratch/<account>/<username>/Alek_Jiho/derived/rollout_phase_kinetics_predictor/rollout_phase_kinetics_predictor_summary.json`

Guardrail:

This is a source-held-out weak-label classifier over automatic particle-only crops. It is a ranking experiment for physics extraction, not a deployable warning system.

