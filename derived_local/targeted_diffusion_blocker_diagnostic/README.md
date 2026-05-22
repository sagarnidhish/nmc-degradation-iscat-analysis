# Targeted Diffusion Blocker Diagnostic

Rechecks diffusion-readiness candidates against threshold-sweep front metrics and same-source/cycle reference groups.

- Target candidates with threshold rows: 20
- Threshold variant rows: 140

## Diagnostic Actions

- fit_quality_blocked_retrack_front: 16
- guarded_review_only: 2
- remeasure_q70_ci_and_manual_front_qc: 1
- source_context_common_not_candidate_specific: 1

## Top Remeasurement Candidates

- cycle78_rank22_obj2: action remeasure_q70_ci_and_manual_front_qc, score 74.749, positive-D fraction 1.0, q70 R2 0.4236646385622081
- cycle60_rank2_obj2: action guarded_review_only, score 24.103, positive-D fraction 1.0, q70 R2 0.465762670919851
- cycle62_rank1_obj4: action fit_quality_blocked_retrack_front, score 22.253, positive-D fraction 1.0, q70 R2 0.1321503522734332
- cycle156_rank2_obj2: action fit_quality_blocked_retrack_front, score 18.234, positive-D fraction 1.0, q70 R2 0.1072221258500542
- cycle116_rank7_obj37: action fit_quality_blocked_retrack_front, score 18.124, positive-D fraction 1.0, q70 R2 0.1217668068233499
- cycle156_rank7_obj27: action fit_quality_blocked_retrack_front, score 17.550, positive-D fraction 1.0, q70 R2 0.0319379661746297
- cycle58_rank4_obj2: action fit_quality_blocked_retrack_front, score 17.274, positive-D fraction 1.0, q70 R2 0.0463618936749117
- cycle157_rank4_obj8: action fit_quality_blocked_retrack_front, score 16.885, positive-D fraction 1.0, q70 R2 0.065878890002745

## Guardrail

This diagnostic ranks diffusion-blocker follow-up candidates using existing automatic threshold/front tables. It does not accept manual labels, relax readiness gates, or emit calibrated diffusion coefficients.
