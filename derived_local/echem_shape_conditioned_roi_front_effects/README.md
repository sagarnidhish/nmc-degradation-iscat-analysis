# Echem-Shape-Conditioned ROI/Front Effects

Residualizes protocol-conditioned ROI/front optical readouts against low-dimensional within-cycle echem shape PCs and event-reference-cycle fixed effects.

- Source rows: 52
- Event/control ROI: 24 / 28
- Shape features used: 45
- Shape PCs: 6 explaining 0.997 variance

## Top Shape-Conditioned Event-Control Tests

- phase_slope_positive_fraction_protocol_residual: residual median event-control 0.03055, p=0.003747
- high_fraction_delta_protocol_residual: residual median event-control 0.002399, p=0.007991
- q70_transformed_fraction_delta: residual median event-control 0.004886, p=0.01045
- low_fraction_delta_protocol_residual: residual median event-control -0.006718, p=0.01579
- q80_transformed_fraction_delta: residual median event-control 0.003385, p=0.02124
- dmd_minus_persistence_mse_protocol_residual: residual median event-control 0.0004775, p=0.04069
- mode_review_priority: residual median event-control 0.1352, p=0.2997
- threshold_robust_phase_score_protocol_residual: residual median event-control -0.0128, p=0.3934

## Shape Context Fit

- mode_review_priority: variance explained 0.839
- first_last_corr_protocol_residual: variance explained 0.583
- q60_logistic_k_per_s: variance explained 0.544
- q70_logistic_k_per_s: variance explained 0.515
- cumulative_abs_norm_change_protocol_residual: variance explained 0.423
- latent_net_displacement_protocol_residual: variance explained 0.413

## Residual Classifier

- Leave-event-reference-out logistic mean ROC-AUC: 0.469
- Leave-event-reference-out logistic mean balanced accuracy: 0.448

## Guardrail

Echem-shape conditioning uses low-dimensional PCA/ridge covariates on a small automatically selected ROI cohort. Surviving residuals are evidence that optical/front signals are not fully explained by measured within-cycle echem shape, but they are not causal proof or calibrated transport constants.
