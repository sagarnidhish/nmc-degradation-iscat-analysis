# Cycle Hazard Warning Audit

Rolling-origin warning audit for future abrupt particle drops using cycle-level photometry/echem descriptors.

- Cycles: 89
- Event cycles: [60.0, 86.0, 116.0, 156.0]
- Target: future_any_drop_within_8cycles
- Best feature set: particle_trace_echem_with_acquisition

Outputs:

- `cycle_hazard_warning_predictions.csv`: rolling-origin probabilities by cycle and feature set.
- `cycle_hazard_warning_feature_set_summary.csv`: AUC/AP/Brier summaries.
- `cycle_hazard_warning_lead_time.csv`: pre-event warning hits and lead cycles.
- `cycle_hazard_warning_group_ablation.csv`: feature-group ablation summary for the best model.
- `cycle_hazard_warning_probability_correlations.csv`: warning probability links to interpretable cycle descriptors.
