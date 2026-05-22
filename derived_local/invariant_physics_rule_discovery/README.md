# Invariant Physics Rule Discovery

Sparse leave-source if/then rule mining over interpretable NMC particle, front, rollout, and echem descriptors.

- Evaluation rows: 72
- Cycles/sources: 24 / 9
- Target: future_any_drop_within_16cycles
- Candidate rules: 171
- Best rule: low(particle_std_diff_positive_fraction)
- Best precision/recall/lift: 0.8888888888888888 / 0.42105263157894735 / 1.1228070175438596

Guardrail: Rules use automatic ROI masks, weak future labels, and data-derived thresholds under leave-source evaluation. They are sparse review-prioritization hypotheses only; source/outcome imbalance, acquisition coupling, missing manual QC, and uncalibrated diffusion/front proxies remain hard claim limits.
