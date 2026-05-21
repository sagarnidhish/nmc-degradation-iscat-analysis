# Weak-Label Degradation Benchmark

Consensus weak-label manifest for ROI video model development and degradation-mode review.

- ROI rows: 52
- Trainable weak-label rows: 7
- Positive/negative weak labels: 3 / 4
- Leave-reference folds: 4 (1 usable binary folds)
- Label counts: {'review_control_uncertain': 19, 'review_positive_uncertain': 15, 'review_uncertain': 11, 'weak_low_consistency_control': 4, 'weak_event_enriched_front_mode': 3}

## Guardrail

This benchmark contains weak consensus labels for model development and review prioritization only. It is not a manual-QC label set and must not be used to claim validated degradation modes or calibrated diffusion.
