# QC Decision Evidence Ledger

Reviewer-facing consolidation of pending ROI/front candidates. No manual labels are assigned.

## Summary

- Candidates: 47
- Manual status counts: {'pending': 47}
- Decision action counts: {'routine_pending_review': 19, 'review_but_diffusion_guarded': 16, 'high_priority_review': 5, 'review_artifact_or_reject_first': 4, 'review_for_possible_accept_first': 3}
- Visual-asset candidates: 47

## Top Review Queue

- rank 1: cycle156_rank7_obj27 cycle 156.0 action=review_for_possible_accept_first score=0.948 risk=0.000 reasons=auto_likely_interpretable;cross_modal_high_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- rank 2: cycle156_rank8_obj10 cycle 156.0 action=review_for_possible_accept_first score=0.889 risk=0.000 reasons=auto_likely_interpretable;cross_modal_review_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- rank 3: cycle156_rank2_obj2 cycle 156.0 action=review_for_possible_accept_first score=0.888 risk=0.125 reasons=auto_likely_interpretable;cross_modal_high_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- rank 4: cycle156_rank1_obj1 cycle 156.0 action=high_priority_review score=0.847 risk=0.125 reasons=auto_likely_interpretable;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- rank 5: cycle156_rank6_obj3 cycle 156.0 action=high_priority_review score=0.827 risk=0.250 reasons=auto_likely_interpretable;front_mask_candidate;diffusion_claim_guardrail;visual_assets_available
- rank 6: cycle60_rank3_obj9 cycle 60.0 action=high_priority_review score=0.776 risk=0.375 reasons=cross_modal_review_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- rank 7: cycle62_rank3_obj9 cycle 62.0 action=high_priority_review score=0.733 risk=0.375 reasons=front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- rank 8: cycle157_rank1_obj1 cycle 157.0 action=high_priority_review score=0.710 risk=0.000 reasons=auto_likely_interpretable;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- rank 9: cycle60_rank6_obj26 cycle 60.0 action=review_but_diffusion_guarded score=0.681 risk=0.250 reasons=cross_modal_review_priority;diffusion_claim_guardrail;visual_assets_available
- rank 10: cycle60_rank2_obj2 cycle 60.0 action=routine_pending_review score=0.673 risk=0.250 reasons=cross_modal_review_priority;diffusion_proxy_candidate;visual_assets_available
- rank 11: cycle156_rank5_obj4 cycle 156.0 action=review_artifact_or_reject_first score=0.647 risk=0.625 reasons=artifact_risk_review;cross_modal_review_priority;multi_pillar_support;front_mask_candidate;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- rank 12: cycle116_rank7_obj37 cycle 116.0 action=routine_pending_review score=0.645 risk=0.375 reasons=front_mask_candidate;diffusion_proxy_candidate;visual_assets_available

## Artifact/Reject-First Queue

- cycle156_rank5_obj4 cycle 156.0 artifact_score=0.920 risk=0.625 reasons=artifact_risk_review;cross_modal_review_priority;multi_pillar_support;front_mask_candidate;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- cycle86_rank4_obj9 cycle 86.0 artifact_score=0.865 risk=0.500 reasons=artifact_risk_review;cross_modal_review_priority;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- cycle157_rank2_obj2 cycle 157.0 artifact_score=0.794 risk=0.625 reasons=artifact_risk_review;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- cycle158_rank2_obj1 cycle 158.0 artifact_score=0.787 risk=0.500 reasons=artifact_risk_review;cross_modal_review_priority;multi_pillar_support;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- cycle86_rank5_obj8 cycle 86.0 artifact_score=0.768 risk=0.500 reasons=artifact_risk_review;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- cycle60_rank3_obj9 cycle 60.0 artifact_score=0.679 risk=0.375 reasons=cross_modal_review_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- cycle86_rank8_obj17 cycle 86.0 artifact_score=0.673 risk=0.375 reasons=diffusion_claim_guardrail;visual_assets_available
- cycle62_rank4_obj1 cycle 62.0 artifact_score=0.673 risk=0.625 reasons=artifact_risk_review;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available

## Guardrail

This ledger does not assign manual QC labels. It prioritizes pending particle/front candidates for human review using existing automatic evidence and keeps all physics claims guarded until particle identity, front mask, and calibration checks are manually accepted.
