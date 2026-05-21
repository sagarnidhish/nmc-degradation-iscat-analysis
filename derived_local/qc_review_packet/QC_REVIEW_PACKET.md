# QC Review Packet

Prioritized ROI/front candidates for manual review before treating phase-front or diffusion proxies as physical estimates.

- Candidates: 30
- Event candidates: 16
- Control candidates: 14

## Top Candidates

- cycle60_rank3_obj9 (event, cycle 60.0): score 5.322; event_roi;hard_rollout;diffusion_proxy;conditioned_front_direction;
- cycle156_rank7_obj27 (event, cycle 156.0): score 5.264; event_roi;hard_rollout;phase_front;diffusion_proxy;conditioned_optical_shift;
- cycle156_rank2_obj2 (event, cycle 156.0): score 5.226; event_roi;hard_rollout;phase_front;conditioned_optical_shift;
- cycle60_rank2_obj2 (event, cycle 60.0): score 5.173; event_roi;hard_rollout;phase_front;conditioned_front_direction;conditioned_optical_shift;
- cycle156_rank1_obj1 (event, cycle 156.0): score 5.029; event_roi;phase_front;diffusion_proxy;conditioned_optical_shift;
- cycle156_rank5_obj4 (event, cycle 156.0): score 4.938; event_roi;hard_rollout;conditioned_optical_shift;
- cycle62_rank3_obj9 (control, cycle 62.0): score 4.904; hard_rollout;
- cycle157_rank2_obj2 (control, cycle 157.0): score 4.875; diffusion_proxy;conditioned_optical_shift;
- cycle156_rank6_obj3 (event, cycle 156.0): score 4.856; event_roi;phase_front;conditioned_optical_shift;
- cycle60_rank4_obj5 (event, cycle 60.0): score 4.803; event_roi;hard_rollout;diffusion_proxy;conditioned_front_direction;
- cycle60_rank6_obj26 (event, cycle 60.0): score 4.553; event_roi;hard_rollout;conditioned_front_direction;
- cycle60_rank1_obj1 (event, cycle 60.0): score 4.428; event_roi;conditioned_front_direction;
- cycle62_rank2_obj2 (control, cycle 62.0): score 4.423; conditioned_front_direction;conditioned_optical_shift;
- cycle158_rank2_obj1 (control, cycle 158.0): score 4.284; conditioned_front_direction;conditioned_optical_shift;
- cycle62_rank1_obj4 (control, cycle 62.0): score 4.149; 

## Review Instructions

For each candidate, inspect available ROI preview, front crop, tracking plot, and rollout preview paths. Set `manual_qc_decision` to `accept`, `reject`, or `uncertain`, and record particle/front-mask concerns in `manual_qc_notes`.

## Guardrail

This is a review packet, not manual QC. Use manual_qc_decision/status columns to record accepted/rejected particle/front masks before publication-scale diffusion claims.
