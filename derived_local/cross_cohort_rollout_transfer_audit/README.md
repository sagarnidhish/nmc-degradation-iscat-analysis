# Cross-Cohort Rollout Transfer Audit

Fits low-rank masked ROI rollout dynamics on selected, transfer-ranked, and pooled cohorts, then evaluates each model on both cohorts.

- Selected ROI sequences: 11
- Transfer-ranked ROI sequences: 48
- Rank: 16

Outputs:

- `cross_cohort_rollout_frame_metrics.csv`: held-out frame metrics by model, cohort, ROI, and method.
- `cross_cohort_rollout_per_roi_metrics.csv`: ROI-method aggregates and DMD/velocity ratios versus persistence.
- `cross_cohort_rollout_domain_shift.csv`: model-transfer summaries versus within-cohort DMD baselines.
- `cross_cohort_rollout_correlations.csv`: links between transfer errors, cycle, validation score, and mask behavior.
- `cross_cohort_rollout_transfer_summary.json`: compact summary.
