# Precursor-Informed ROI Review

Ranked manual-QC candidate manifest combining ROI physics, front proxies, kinetics, graph context, and cycle-level precursor severity.

- Review candidates: 47
- Event/control candidates: 23 / 24
- Tier counts: {'medium': 18, 'routine': 17, 'high': 12}

## Top Candidates
- cycle156_rank7_obj27 (event, cycle 156.0): score=5.527, tier=high, reason=event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle60_rank5_obj18 (event, cycle 60.0): score=5.094, tier=high, reason=event-cycle ROI;high precursor-context cycle;large front-direction residual;kinetic proxy available
- cycle156_rank6_obj3 (event, cycle 156.0): score=5.086, tier=high, reason=event-cycle ROI;high precursor-context cycle;kinetic proxy available
- cycle156_rank5_obj4 (event, cycle 156.0): score=5.054, tier=high, reason=event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle156_rank2_obj2 (event, cycle 156.0): score=4.926, tier=high, reason=event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle156_rank1_obj1 (event, cycle 156.0): score=4.834, tier=high, reason=event-cycle ROI;high precursor-context cycle;kinetic proxy available
- cycle156_rank8_obj10 (event, cycle 156.0): score=4.741, tier=high, reason=event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle60_rank3_obj9 (event, cycle 60.0): score=4.632, tier=high, reason=event-cycle ROI;high precursor-context cycle;event-enriched residual mode;large front-direction residual;kinetic proxy available
- cycle60_rank2_obj2 (event, cycle 60.0): score=4.176, tier=high, reason=event-cycle ROI;high precursor-context cycle;event-enriched residual mode;large front-direction residual;kinetic proxy available
- cycle62_rank2_obj2 (control, cycle 62.0): score=4.145, tier=high, reason=high precursor-context cycle;large front-direction residual;kinetic proxy available

## Guardrail

This is a review-prioritization manifest. It combines automatic ROI/front/mode/kinetic descriptors with cycle-level precursor context to decide what to inspect first; it does not assign manual QC labels or validate diffusion/front claims.
