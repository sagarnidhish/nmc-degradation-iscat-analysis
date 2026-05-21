# Prefix ROI Forecast

Prefix-only forecasts from early cropped particle ROI frames.

- ROI rows: 52
- Prefix rows: 156
- Prefix features: 58

## Top Classification
- front_positive_residual_binary prefix_only random_forest f=0.5: AUC=0.726, bal_acc=0.512
- front_positive_residual_binary prefix_only random_forest f=0.75: AUC=0.712, bal_acc=0.547
- front_positive_residual_binary prefix_only logistic_l2 f=0.75: AUC=0.687, bal_acc=0.608
- front_positive_residual_binary prefix_only random_forest f=0.25: AUC=0.654, bal_acc=0.586
- front_positive_residual_binary prefix_only logistic_l2 f=0.25: AUC=0.648, bal_acc=0.554
- front_positive_residual_binary prefix_only logistic_l2 f=0.5: AUC=0.635, bal_acc=0.624
- front_positive_residual_binary prefix_plus_context random_forest f=0.5: AUC=0.613, bal_acc=0.528
- front_positive_residual_binary prefix_plus_context random_forest f=0.75: AUC=0.601, bal_acc=0.531

## Guardrail

Prefix-only ROI forecasts use cropped particle-region sequences, but the cohort is small and selected around event-reference cycles; treat results as physics-signal triage, not a deployable predictor.
