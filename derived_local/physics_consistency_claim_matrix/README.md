# Physics Consistency Claim Matrix

Multimodal ROI-level consistency audit across front direction, optical change, rollout residuals, kinetics, precursor context, echem-shape context, and residual mode taxonomy.

- ROI rows scored: 52
- Cycles represented: 11
- Tier counts: {'routine_or_low_consistency': 21, 'discordant_guardrail': 14, 'cross_modal_review_priority': 8, 'front_kinetic_consistent': 4, 'rollout_mode_consistent': 3, 'cross_modal_high_priority': 2}
- Claim readiness counts: {'manual_qc_required_no_physics_claim': 52}
- Manual-QC accepted rows: 0
- Calibration evidence: 32/33 HDF5 timing files; spatial status slide_derived_96nm_px

## Top Candidates

- cycle156_rank7_obj27 (event, cycle 156.0): score 5.790, support 6, tier cross_modal_high_priority; multi-pillar support;multiple strong pillars;front/kinetic agreement;rollout/mode agreement;precursor context
- cycle156_rank5_obj4 (event, cycle 156.0): score 5.503, support 6, tier cross_modal_review_priority; multi-pillar support;multiple strong pillars;front/kinetic agreement;rollout/mode agreement;precursor context;automatic artifact caution
- cycle156_rank2_obj2 (event, cycle 156.0): score 5.486, support 6, tier cross_modal_high_priority; multi-pillar support;multiple strong pillars;front/kinetic agreement;rollout/mode agreement;precursor context
- cycle158_rank2_obj1 (control, cycle 158.0): score 4.968, support 6, tier cross_modal_review_priority; multi-pillar support;multiple strong pillars;front/kinetic agreement;rollout/mode agreement;precursor context;automatic artifact caution
- cycle156_rank8_obj10 (event, cycle 156.0): score 3.925, support 4, tier cross_modal_review_priority; multi-pillar support;multiple strong pillars;front/kinetic agreement;precursor context
- cycle158_rank4_obj8 (control, cycle 158.0): score 3.656, support 5, tier cross_modal_review_priority; multi-pillar support;multiple strong pillars;front/kinetic agreement;precursor context;discordant pillars;automatic artifact caution
- cycle158_rank6_obj3 (control, cycle 158.0): score 3.146, support 3, tier front_kinetic_consistent; multiple strong pillars;front/kinetic agreement;precursor context;automatic artifact caution
- cycle156_rank6_obj3 (event, cycle 156.0): score 2.681, support 3, tier front_kinetic_consistent; multiple strong pillars;front/kinetic agreement;precursor context;discordant pillars

## Interpretation

This matrix is a multimodal consistency and review-prioritization audit. It does not assign manual QC labels and does not validate calibrated diffusion or material degradation mechanisms.
