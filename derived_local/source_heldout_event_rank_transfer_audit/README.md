# Source-Heldout Event Rank Transfer Audit

Tests whether automatic particle-region features can rank near-pre-event ROIs in held-out acquisition sources.

- Input dossier rows: 128
- Sources: 14
- Candidate raw automatic features: 60
- Held-out folds: 5

## Transfer Score Tests

- near_vs_any_non_near: AUC 0.832, AP 0.924, eligible held-out sources 5, sign-flip p 0.125

## Guardrail

Held-out-source transfer scores are automatic particle-region ranking diagnostics. They do not add manual labels, calibrated velocities, diffusion coefficients, or deployment validation.
