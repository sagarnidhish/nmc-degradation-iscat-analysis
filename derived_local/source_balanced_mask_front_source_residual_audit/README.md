# Source-Balanced Mask/Front Source-Residual Audit

Rows/features/sources: 96 / 54 / 14

## Future16 Best Transforms
- raw: masked_minus_background_mean_slope AUC=0.690, AP=0.696, eta2=0.634
- source_residual: front_radius_q80_slope_px_per_norm_time AUC=0.631, AP=0.634, eta2=0.000
- within_source_rank: front_radius_q80_slope_px_per_norm_time AUC=0.656, AP=0.677, eta2=0.078

## Guardrail
Source residualization tests whether automatic mask/front proxies survive source structure. Passing this audit would still be weak-label, automatic-mask evidence; failing it means the feature is useful mainly for QC/source triage.
