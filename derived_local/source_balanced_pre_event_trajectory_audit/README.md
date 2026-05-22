# Source-Balanced Pre-Event Trajectory Audit

Cycle-level event-distance tests for the source-balanced pre-event ROI cohort. The audit aggregates the two ROI proposals per sampled cycle, then tests whether front/mask/diffusion-like and rollout proxies vary monotonically as cycles approach abrupt optical event cycles.

Key files:

- `source_balanced_pre_event_trajectory_cycle_features.csv`: one row per source/cycle/event-distance.
- `source_balanced_pre_event_trajectory_feature_tests.csv`: Spearman/slope/permutation tests versus event proximity.
- `source_balanced_pre_event_trajectory_full_event_deltas.csv`: near-minus-far deltas for source/event groups containing both bins.
- `source_balanced_pre_event_trajectory_summary.json`: compact machine-readable summary.
