# Current Claim Readiness Matrix

The project supports source-aware particle-region review ranking and optical-proxy hypothesis generation, while calibrated diffusion, causal mechanism, and deployable warning claims remain blocked.

- Claims audited: 7
- Supported/operational claim IDs: C01_event_local_particle_readiness_ranker, C02_transport_mechanism_candidate_dossier, C04_next_frame_rollout_physics_descriptor
- Blocked/not-supported claim IDs: C03_broad_future_drop_generalization, C05_calibrated_diffusion_coefficients, C07_deployable_future_warning_detector

## Status Counts

- supported_for_review_ranking: 1
- ready_for_manual_qc_handoff: 1
- not_supported_as_broad_predictor: 1
- supported_as_descriptor_not_superior_predictor: 1
- blocked: 1
- partial_proxy_only: 1
- blocked_for_deployment: 1

## Claim Rows

- C01_event_local_particle_readiness_ranker supported_for_review_ranking: Automatic particle-region descriptors can rank near-pre-event ROIs for manual review.
- C02_transport_mechanism_candidate_dossier ready_for_manual_qc_handoff: The project has a concrete prioritized review dossier for apparent transport/front/kinetic candidates.
- C03_broad_future_drop_generalization not_supported_as_broad_predictor: Apparent transport/front descriptors generalize as a broad future-drop predictor across expanded ROI crops.
- C04_next_frame_rollout_physics_descriptor supported_as_descriptor_not_superior_predictor: Next-frame prediction and rollout residuals are useful particle-region physics descriptors.
- C05_calibrated_diffusion_coefficients blocked: The current pipeline can report calibrated material diffusion coefficients.
- C06_phase_boundary_tracking partial_proxy_only: The videos support automatic phase-boundary/front tracking as a physical proxy.
- C07_deployable_future_warning_detector blocked_for_deployment: The current automatic video/echem features form a deployable future degradation warning detector.

## Guardrail

Claim readiness is wording guidance over existing automatic audits. It does not add manual labels, calibrated spatial metadata, causal tests, or deployable validation.
