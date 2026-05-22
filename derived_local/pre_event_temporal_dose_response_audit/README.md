# Pre-Event Temporal Dose-Response Audit

Tests whether particle-region descriptors increase as the next event approaches.

- Pre-event rows: 76
- Pre-event cycles: 38
- Pre-event sources: 10

## Key Feature Tests

- qc_review_score: source-centered rho 0.110, raw rho 0.417, median source slope 6.18e-05, positive source slope fraction 0.556
- transport_mechanism_score: source-centered rho 0.011, raw rho 0.404, median source slope 0.0132, positive source slope fraction 0.556
- front_kinetic_score: source-centered rho -0.007, raw rho 0.354, median source slope -0.008, positive source slope fraction 0.444

## Guardrail

Temporal dose-response tests use automatic particle-region optical/front descriptors and event-relative labels. They support precursor-ranking hypotheses only, not causal mechanisms, calibrated phase-boundary velocities, or diffusion coefficients.
