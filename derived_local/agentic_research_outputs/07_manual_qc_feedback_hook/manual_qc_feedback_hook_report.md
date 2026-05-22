# Manual-QC Feedback Hook

Feedback state: `manual_labels_needed`

## Feedback Rules

| gate | required_feedback | current_status | downstream_script |
| --- | --- | --- | --- |
| front_or_diffusion_claim | manual accept label for ROI/front mask plus timing and spatial calibration evidence | waiting_for_labels | tier4_manual_qc_gated_front_effects.py |
| degradation_mode_training_label | manual reject/accept/artifact label joined to automatic ROI evidence ledger | waiting_for_labels | tier4_qc_decision_evidence_ledger.py |
| agentic_hypothesis_status_update | ledger status update from proposed/blocked_by_qc to supported/weakened after manual labels | ready_for_update | agentic_research/06_closed_loop_hypothesis_ledger/run_hypothesis_ledger.py |

## Review Queue

| roi_id | cycleNo |
| --- | --- |
| cycle116_rank7_obj37 | 116 |
| cycle118_rank1_obj7 | 118 |
| cycle116_rank5_obj43 | 116 |
| cycle116_rank2_obj2 | 116 |
| cycle116_rank8_obj20 | 116 |
| cycle116_rank1_obj6 | 116 |
| cycle88_rank3_obj6 | 88 |
| cycle88_rank2_obj4 | 88 |
| cycle88_rank4_obj8 | 88 |
| cycle86_rank5_obj8 | 86 |
| cycle158_rank4_obj8 | 158 |
| cycle58_rank4_obj2 | 58 |
| cycle58_rank6_obj29 | 58 |
| cycle158_rank6_obj3 | 158 |
| cycle86_rank8_obj17 | 86 |
| cycle58_rank3_obj9 | 58 |
| cycle86_rank3_obj5 | 86 |
| cycle86_rank4_obj9 | 86 |
| cycle60_rank5_obj18 | 60 |
| cycle157_rank1_obj1 | 157 |
| cycle157_rank4_obj8 | 157 |
| cycle157_rank6_obj3 | 157 |
| cycle156_rank8_obj10 | 156 |
| cycle62_rank4_obj1 | 62 |
| cycle62_rank1_obj4 | 62 |
| cycle158_rank2_obj1 | 158 |
| cycle62_rank2_obj2 | 62 |
| cycle60_rank1_obj1 | 60 |
| cycle60_rank6_obj26 | 60 |
| cycle60_rank4_obj5 | 60 |

## Guardrail

This hook defines how manual QC should update downstream AI/physics claims. It does not fabricate labels.
