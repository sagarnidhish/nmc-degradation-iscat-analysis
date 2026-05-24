# Objectives And Observations (Addendum)

Date: 2026-05-25

## Bridge Fix

The first rollout/phase-kinetics bridge matched exact ROI ids and produced zero overlap because the rollout and phase-kinetics audits were derived from related but not identical ROI exports.

## Fix

Use a source/cycle-level bridge instead:

- aggregate rollout features by `source_stem` and `cycleNo`,
- aggregate phase-kinetics features by `source_stem` and `cycleNo`,
- join those source/cycle summaries and compute associations there.

## Why This Is Better

- It preserves the source-held-out structure.
- It keeps the analysis particle-only.
- It avoids false zero-overlap failures from incompatible ROI naming.

## Planned Output

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_phase_kinetics_bridge_v2/`

## Guardrail

This bridge is still an association audit only. It does not claim calibrated diffusion or manual phase-boundary labels.

