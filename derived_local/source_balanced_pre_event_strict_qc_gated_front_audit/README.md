# Source-Balanced Pre-Event Strict QC-Gated Front Audit

Automatic stop/go gates for front and diffusion interpretation. These gates prioritize manual review and explicitly prevent automatic diffusion claims.

- Candidates scored: 24
- Manual front-review candidates: 1
- Automatic diffusion-claim candidates: 0
- Gate pass counts: {'stable_mask_gate': 24, 'centroid_stability_gate': 20, 'visual_sanity_gate': 9, 'visual_mode_gate': 6, 'front_trace_fit_gate': 4, 'front_direction_agreement_gate': 11, 'manual_front_review_gate': 1, 'automatic_diffusion_claim_gate': 0}

## Top Manual-Review Candidates
- rank 8 source_balanced_cycle80_rank62_obj2_9_c2_x10_010723 score=0.600, sanity=0.761, visual_qc=0.506

## Guardrail

Strict QC gates are automatic review-prioritization gates. They do not create manual labels, validate particle identity or front masks, calibrate diffusion, prove phase-boundary motion, or establish causality. Rows failing the diffusion gate must not be used for material diffusion coefficient claims.
