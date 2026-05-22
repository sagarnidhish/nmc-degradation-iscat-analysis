# Post-Remeasurement Diffusion Gate Audit

- Target: `cycle78_rank22_obj2`
- Target q70 status: passed_for_target_packet
- Candidate q70 blocker removed: True
- Publication-ready after remeasurement: False

Outputs:
- `post_remeasurement_diffusion_candidate_update.csv`
- `post_remeasurement_diffusion_gate_table.csv`
- `post_remeasurement_diffusion_scenario_table.csv`
- `post_remeasurement_diffusion_gate_summary.json`

Guardrail:
This audit propagates a target-specific automatic q70 CI update through existing readiness gates. It does not rerun global diffusion readiness, accept manual labels, verify raw calibration metadata, or create publication-ready diffusion coefficients.
