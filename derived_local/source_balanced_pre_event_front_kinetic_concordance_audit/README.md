# Source-Balanced Pre-Event Front/Kinetic Concordance Audit

Ranks source-balanced pre-event ROI candidates by agreement between particle-mask optical kinetics, front-consensus evidence, visual QC, and strict front gates.

- Rows: 128
- Loaded kinetic rows: 128
- Sources: 14
- Tier counts: {'routine_or_low_concordance': 99, 'front_only_guardrail': 13, 'kinetic_only_guardrail': 7, 'near_pre_front_kinetic_review': 4, 'front_kinetic_concordant': 4, 'strict_gate_manual_front_review': 1}

## Top Candidates
- source_balanced_cycle154_rank30_obj2_17_c2_x10_HighHighCOV_150723 near_pre_event_1_8: score=1.669, kinetic=2.030, front=1.585, tier=near_pre_front_kinetic_review
- source_balanced_cycle46_rank46_obj2_6_c2_x10_270623_2 mid_pre_event_9_16: score=1.468, kinetic=-0.179, front=3.901, tier=front_only_guardrail
- source_balanced_cycle32_rank39_obj1_4_c2_x10_240623 far_pre_event_17_32: score=1.369, kinetic=0.525, front=2.881, tier=front_kinetic_concordant
- source_balanced_cycle2_rank31_obj2_2_c2_x14_200623 no_near_event_control: score=1.335, kinetic=-0.112, front=4.165, tier=front_only_guardrail
- source_balanced_cycle151_rank28_obj1_17_c2_x10_HighHighCOV_150723 near_pre_event_1_8: score=1.224, kinetic=0.967, front=1.741, tier=near_pre_front_kinetic_review
- source_balanced_cycle8_rank33_obj1_2_c2_x14_200623 no_near_event_control: score=1.135, kinetic=-0.325, front=3.983, tier=front_only_guardrail
- source_balanced_cycle78_rank61_obj2_9_c2_x10_010723 near_pre_event_1_8: score=1.129, kinetic=1.360, front=0.671, tier=near_pre_front_kinetic_review
- source_balanced_cycle80_rank62_obj2_9_c2_x10_010723 near_pre_event_1_8: score=1.124, kinetic=0.672, front=1.413, tier=strict_gate_manual_front_review

## Top Event Tests
- near_vs_any_non_near raw kinetic_evidence_score: AUC=0.770, median diff=0.956, p=5.005e-06
- near_vs_any_non_near raw front_kinetic_concordance_score: AUC=0.768, median diff=0.561, p=3.904e-05
- near_vs_mid_pre raw kinetic_evidence_score: AUC=0.820, median diff=0.969, p=7.743e-05
- near_vs_post_control raw kinetic_evidence_score: AUC=0.746, median diff=0.9, p=0.0001681
- near_vs_far_pre raw kinetic_evidence_score: AUC=0.778, median diff=1.02, p=0.0005781
- near_vs_post_control raw front_kinetic_concordance_score: AUC=0.749, median diff=0.506, p=0.0007042

## Guardrail

Front/kinetic concordance is an automatic review-prioritization score joining optical phase kinetics, front proxies, visual QC, and strict gates. It does not assign manual labels, validate phase-boundary motion, calibrate diffusion, or prove degradation causality.
