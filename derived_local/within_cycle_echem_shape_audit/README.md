# Within-Cycle Echem Shape Audit

Voltage/current trajectory and dQ/dV-proxy descriptors joined to particle-trace and ROI optical physics outputs.

- Observed cycles requested: 89
- Echem shape cycles found: 81
- Shape features: 48
- ROI rows joined: 52

## Top ROI Shape Correlations
- shape_V_q95 vs mode_review_priority: rho=-0.864, p=2.72e-12, n=38
- neg_dq_abs_peak_frac vs mode_review_priority: rho=0.841, p=3.81e-11, n=38
- pos_dq_abs_midV_frac vs mode_review_priority: rho=0.84, p=4.17e-11, n=38
- all_dq_abs_entropy vs mode_review_priority: rho=-0.839, p=4.68e-11, n=38
- pos_dq_abs_peak_frac vs mode_review_priority: rho=0.839, p=4.68e-11, n=38
- pos_dq_abs_highV_frac vs mode_review_priority: rho=-0.839, p=4.79e-11, n=38
- pos_dq_abs_entropy vs mode_review_priority: rho=-0.801, p=1.61e-09, n=38
- all_dq_abs_peak_frac vs mode_review_priority: rho=0.79, p=3.65e-09, n=38
- all_dq_abs_midV_frac vs mode_review_priority: rho=0.79, p=3.71e-09, n=38
- pos_dqdv_abs_integral_proxy vs mode_review_priority: rho=0.758, p=3.58e-08, n=38

## Top Cycle Shape Correlations
- shape_charge_mAh_neg_abs vs capacity_mAh: rho=1, p=1.25e-149, n=81
- neg_dq_abs_total_mAh vs capacity_mAh: rho=1, p=7.53e-126, n=81
- echem_shape_duration_s vs capacity_mAh: rho=0.985, p=1.68e-61, n=81
- shape_charge_mAh_abs vs capacity_mAh: rho=0.985, p=1.68e-61, n=81
- shape_charge_mAh_signed vs coulombic_efficiency_pct: rho=-0.968, p=2.14e-49, n=81
- shape_dIdt_slope vs capacity_mAh: rho=0.934, p=5.85e-37, n=81
- all_dq_abs_total_mAh vs capacity_mAh: rho=0.924, p=9.07e-35, n=81
- neg_dq_abs_highV_frac vs capacity_mAh: rho=0.916, p=3.95e-33, n=81
- neg_dq_abs_entropy vs capacity_mAh: rho=0.891, p=8.25e-29, n=81
- all_dq_abs_highV_frac vs capacity_mAh: rho=0.879, p=4.03e-27, n=81

## Guardrail

Within-cycle echem shape features are computed from raw time/potential/current rows for observed particle/ROI cycles. dQ/dV terms are proxy descriptors from current-time integration over voltage bins, not calibrated electrochemical capacity analysis.
