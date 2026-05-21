# Masked Rollout Cycle Warning Audit

This folder collapses masked particle-rollout residuals to observed ROI cycles and joins larger particle-trace future-drop labels.

- ROI cycles: 11
- Rollout-derived features tested: 105
- Permutations per test: 5000

Outputs:

- `masked_rollout_cycle_warning_joined.csv`: cycle-level masked-rollout features plus trace targets.
- `masked_rollout_cycle_warning_target_tests.csv`: binary target tests with permutation p-values.
- `masked_rollout_cycle_warning_correlations.csv`: correlations with trace/state context.
- `masked_rollout_cycle_warning_ranked_cycles.csv`: cycles ranked by masked rollout warning score.
- `masked_rollout_cycle_warning_summary.json`: compact summary.
