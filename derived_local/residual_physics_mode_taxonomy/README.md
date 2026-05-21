# Residual Physics Mode Taxonomy

Protocol-adjusted degradation-mode hypotheses from optical residuals, rollout/latent descriptors, threshold-front residuals, and QC-priority context.

- ROI rows: 52
- Event ROI: 24
- Control ROI: 28
- Selected k: 4
- Best silhouette: 0.204
- Mean seed-stability ARI: 0.935

## Modes

- optical_brightening_decorrelating_rollout_hard_front_positive: n=13, event fraction=0.85, p=0.00282, cycles=60.0;156.0;62.0;86.0
- optical_loss_rollout_hard: n=5, event fraction=0.40, p=1, cycles=157.0;156.0;62.0
- near_baseline_or_context_like: n=24, event fraction=0.33, p=0.102, cycles=86.0;58.0;118.0;116.0
- front_negative_high_apparent_front_proxy: n=10, event fraction=0.30, p=0.309, cycles=88.0;90.0;116.0;60.0

## Top Review ROIs

- cycle60_rank3_obj9 (event, cycle 60.0): optical_brightening_decorrelating_rollout_hard_front_positive, priority=3.673
- cycle60_rank6_obj26 (event, cycle 60.0): optical_brightening_decorrelating_rollout_hard_front_positive, priority=3.596
- cycle60_rank4_obj5 (event, cycle 60.0): optical_brightening_decorrelating_rollout_hard_front_positive, priority=3.481
- cycle62_rank3_obj9 (control, cycle 62.0): optical_brightening_decorrelating_rollout_hard_front_positive, priority=3.308
- cycle156_rank7_obj27 (event, cycle 156.0): optical_brightening_decorrelating_rollout_hard_front_positive, priority=3.260
- cycle157_rank2_obj2 (control, cycle 157.0): optical_loss_rollout_hard, priority=3.106
- cycle156_rank5_obj4 (event, cycle 156.0): optical_brightening_decorrelating_rollout_hard_front_positive, priority=2.933
- cycle62_rank4_obj1 (control, cycle 62.0): optical_loss_rollout_hard, priority=2.923
- cycle60_rank2_obj2 (event, cycle 60.0): optical_brightening_decorrelating_rollout_hard_front_positive, priority=2.923
- cycle58_rank3_obj9 (control, cycle 58.0): near_baseline_or_context_like, priority=2.913
- cycle156_rank1_obj1 (event, cycle 156.0): optical_loss_rollout_hard, priority=2.875
- cycle156_rank6_obj3 (event, cycle 156.0): optical_loss_rollout_hard, priority=2.798

## Guardrail

Mode labels are protocol-adjusted computational hypotheses from automatic ROI candidates; they require QC labels before being treated as biological/material degradation modes.
