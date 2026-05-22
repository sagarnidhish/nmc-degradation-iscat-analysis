# Signed Optical-Loss Mechanism Audit

Builds signed optical-loss, front-contraction, rollout-difficulty, and echem-state axes for source-aware mechanism triage.

- Rows/eval rows: 172 / 72
- Cycles/sources: 24 / 9
- Axes: signed_optical_loss_axis, front_contraction_axis, rollout_difficulty_axis, echem_degraded_state_axis, combined_loss_mechanism_axis

Guardrail: Signed optical-loss axes are computed from automatic ROI/video/echem descriptors and weak future labels. They support mechanism triage for optical loss/contraction versus front expansion, but they do not validate manual particle identity, front masks, calibrated diffusion, or deployable warnings.
