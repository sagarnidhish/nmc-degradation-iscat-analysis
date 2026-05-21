# Prefix ROI Forecast

Prefix-only forecasts from early cropped particle ROI frames.

- ROI rows: 52
- Prefix rows: 156
- Prefix features: 56

## Top Classification
- front_positive_residual_binary prefix_only random_forest f=0.5: AUC=0.691, bal_acc=0.472
- front_positive_residual_binary prefix_plus_context random_forest f=0.25: AUC=0.524, bal_acc=0.586
- front_positive_residual_binary prefix_only random_forest f=0.25: AUC=0.520, bal_acc=0.534
- front_positive_residual_binary prefix_plus_context random_forest f=0.5: AUC=0.506, bal_acc=0.464
- front_positive_residual_binary prefix_only logistic_l2 f=0.5: AUC=0.476, bal_acc=0.479
- front_positive_residual_binary prefix_only random_forest f=0.75: AUC=0.409, bal_acc=0.510
- front_positive_residual_binary prefix_only logistic_l2 f=0.75: AUC=0.407, bal_acc=0.466
- front_positive_residual_binary prefix_only logistic_l2 f=0.25: AUC=0.387, bal_acc=0.390

## Guardrail

Prefix-only ROI forecasts use cropped particle-region sequences, but the cohort is small and selected around event-reference cycles; treat results as physics-signal triage, not a deployable predictor.
