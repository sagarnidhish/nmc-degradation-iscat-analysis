# Echem Optical Regime Atlas

Cycle-level electrochemical regime descriptors joined to optical/AI degradation evidence.

- Cycles: 89
- Echem regime features: 44
- Missing echem-shape cycles: 8
- Extreme-or-missing CE cycles: 10

## Regime Summary
- pc1_mid: n=29, median cycle=102, median consensus=0.542, modal votes=1, event rate=0.103, future8 rate=0.379, extreme/missing CE rate=0.276
- pc1_high: n=30, median cycle=53, median consensus=0.527, modal votes=1, event rate=0, future8 rate=0.167, extreme/missing CE rate=0.0333
- pc1_low: n=30, median cycle=135, median consensus=0.429, modal votes=0.833, event rate=0.0333, future8 rate=0.133, extreme/missing CE rate=0.0333

## Top Binary Echem/Optical Tests
- pos_dq_abs_peak_voltage vs multimodal_outlier_without_trace_drop: median positive-negative 0.05, p=0.000145, n=81
- all_dq_abs_peak_voltage vs multimodal_outlier_without_trace_drop: median positive-negative 0.05, p=0.00538, n=81
- shape_charge_mAh_abs vs future_any_drop_within_8cycles: median positive-negative -0.0253, p=0.03, n=81
- capacity_fade_from_first_mAh vs future_any_drop_within_8cycles: median positive-negative 0.0137, p=0.0329, n=81
- capacity_fraction_of_first vs future_any_drop_within_8cycles: median positive-negative -0.0207, p=0.0329, n=81
- capacity_mAh vs future_any_drop_within_8cycles: median positive-negative -0.0137, p=0.0329, n=81
- shape_charge_mAh_neg_abs vs future_any_drop_within_8cycles: median positive-negative -0.0137, p=0.0339, n=81
- dqdv_entropy_asymmetry vs future_any_drop_within_8cycles: median positive-negative 0.0269, p=0.0393, n=81
- pos_dq_abs_peak_voltage vs future_any_drop_within_8cycles: median positive-negative 0, p=0.0445, n=81
- charge_discharge_capacity_abs_gap_mAh vs multimodal_outlier_without_trace_drop: median positive-negative 0.378, p=0.0499, n=81

## Top Echem/Optical Correlations
- shape_dVdt_abs_p95 vs cross_modal_consensus_score: rho=0.617, p=8.51e-10, n=81
- all_dq_abs_midV_frac vs particle_norm_cv: rho=-0.404, p=0.000187, n=81
- pos_dq_abs_highV_frac vs particle_norm_cv: rho=0.393, p=0.000285, n=81
- all_dq_abs_peak_voltage vs max_abs_delta_prev: rho=-0.394, p=0.000299, n=80
- pos_dq_abs_midV_frac vs particle_norm_cv: rho=-0.389, p=0.000335, n=81
- neg_dq_abs_peak_frac vs particle_norm_cv: rho=-0.385, p=0.000393, n=81
- shape_dVdt_abs_p95 vs n_modal_votes: rho=0.373, p=0.0006, n=81
- echem_regime_pc3 vs particle_norm_cv: rho=0.356, p=0.000611, n=89
- echem_regime_pc4 vs cross_modal_consensus_score: rho=-0.352, p=0.000711, n=89
- all_dq_abs_entropy vs particle_norm_cv: rho=0.36, p=0.000973, n=81

## Guardrail

This atlas uses echem shape and dQ/dV-like proxy descriptors to organize optical degradation hypotheses. It is not calibrated dQ/dV, not a mechanistic phase diagram, and does not remove the acquisition/frame-count confounder by itself.
