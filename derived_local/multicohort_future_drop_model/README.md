# Multi-Cohort Future-Drop Model

Weak future-drop model table combining selected and transfer-ranked ROI video physics features.

- ROI rows: 59
- Features: 44
- Cycle-group OOF metrics: [{'model': 'logistic_l2', 'group_col': 'cycleNo', 'n_scored': 58, 'n_positive_scored': 28, 'n_negative_scored': 30, 'pooled_oof_roc_auc': 0.6916666666666667, 'pooled_oof_average_precision': 0.7773557378543848, 'n_scored_folds': 13, 'n_total_folds': 13}, {'model': 'random_forest', 'group_col': 'cycleNo', 'n_scored': 58, 'n_positive_scored': 28, 'n_negative_scored': 30, 'pooled_oof_roc_auc': 0.8857142857142857, 'pooled_oof_average_precision': 0.9137697862759439, 'n_scored_folds': 13, 'n_total_folds': 13}]

Guardrail: Future-drop labels are weak cycle-level labels projected onto ROI rows. Grouped splits reduce cycle leakage, but this is a review-prioritization model, not a deployment-ready degradation detector or manual-QC label set.
