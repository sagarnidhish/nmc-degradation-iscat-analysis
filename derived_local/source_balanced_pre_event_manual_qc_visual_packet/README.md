# Source-Balanced Pre-Event Manual-QC Visual Packet

This packet renders the manual-QC decision queue, not just the earlier consensus top candidates.

## Key Counts

- Queue rows: 128
- Requested/rendered: 40 / 40
- Sources rendered: 12
- Action tiers rendered: {'review_front_only_guardrail': 11, 'routine_or_low_concordance': 11, 'review_kinetic_only_guardrail': 8, 'review_front_and_kinetics_first': 5, 'context_control_concordant': 4, 'review_strict_front_gate': 1}
- Event bins rendered: {'near_pre_event_1_8': 23, 'post_event_1_16': 6, 'mid_pre_event_9_16': 4, 'far_pre_event_17_32': 4, 'no_near_event_control': 3}
- Contact sheet: `/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_manual_qc_visual_packet/manual_qc_visual_contact_sheet.png`

## Top Rendered Candidates

- rank 1 source_balanced_cycle80_rank62_obj2_9_c2_x10_010723 review_strict_front_gate near_pre_event_1_8 score=1.181
- rank 2 source_balanced_cycle151_rank28_obj1_17_c2_x10_HighHighCOV_150723 review_front_and_kinetics_first near_pre_event_1_8 score=1.083
- rank 3 source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723 review_front_and_kinetics_first near_pre_event_1_8 score=1.062
- rank 4 source_balanced_cycle78_rank61_obj2_9_c2_x10_010723 review_front_and_kinetics_first near_pre_event_1_8 score=1.061
- rank 5 source_balanced_cycle154_rank30_obj2_17_c2_x10_HighHighCOV_150723 review_front_and_kinetics_first near_pre_event_1_8 score=1.054
- rank 6 source_balanced_cycle152_rank29_obj2_17_c2_x10_HighHighCOV_150723 review_front_and_kinetics_first near_pre_event_1_8 score=1.005
- rank 7 source_balanced_cycle46_rank46_obj2_6_c2_x10_270623_2 review_front_only_guardrail mid_pre_event_9_16 score=0.897
- rank 8 source_balanced_cycle150_rank27_obj2_17_c2_x10_HighHighCOV_150723 review_kinetic_only_guardrail near_pre_event_1_8 score=0.839
- rank 9 source_balanced_cycle151_rank28_obj2_17_c2_x10_HighHighCOV_150723 review_kinetic_only_guardrail near_pre_event_1_8 score=0.821
- rank 10 source_balanced_cycle150_rank27_obj1_17_c2_x10_HighHighCOV_150723 review_kinetic_only_guardrail near_pre_event_1_8 score=0.809
- rank 11 source_balanced_cycle56_rank51_obj2_7_c2_x10_290623 review_kinetic_only_guardrail near_pre_event_1_8 score=0.782
- rank 12 source_balanced_cycle2_rank31_obj2_2_c2_x14_200623 review_front_only_guardrail no_near_event_control score=0.778

## Guardrail

This packet renders automatic particle-crop visualizations for manual inspection. It does not assign labels, validate particle identity, validate front masks, calibrate diffusion, prove phase-boundary motion, or establish causality.
