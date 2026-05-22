# Source-Balanced Pre-Event Echem/Front Coupling Audit

- ROI rows/cycles/sources: 128 / 64 / 14
- Echem features used: 39
- Optical/front targets used: 20

## Top Raw Event-Bin Effects
- clean_pre_1_8_vs_post_control masked_minus_background_mean_slope: n=32 vs 52, median diff=0.005637, p=5.478e-06, AUC=0.797
- clean_pre_1_8_vs_post_control kymograph_temporal_energy: n=32 vs 52, median diff=3.87e-06, p=0.0008984, AUC=0.717
- clean_pre_1_8_vs_post_control front_radius_q60_slope_px_per_norm_time: n=21 vs 20, median diff=0.2503, p=0.02942, AUC=0.700
- clean_pre_1_8_vs_post_control phase_fraction_slope_per_norm_time: n=32 vs 52, median diff=0.01072, p=0.04225, AUC=0.633
- clean_pre_1_8_vs_post_control masked_minus_background_mean_median: n=32 vs 52, median diff=0.00232, p=0.0514, AUC=0.627
- clean_pre_1_8_vs_post_control front_radius2_slope_px2_per_norm_time: n=23 vs 31, median diff=2.569, p=0.05886, AUC=0.652

## Top Source+Echem Residual Event-Bin Effects
- clean_pre_1_8_vs_post_control front_radius_slope_px_per_norm_time: n=23 vs 31, residual median diff=0.2104, p=0.08024, AUC=0.641
- clean_pre_1_8_vs_post_control front_radius2_slope_px2_per_norm_time: n=23 vs 31, residual median diff=2.616, p=0.09655, AUC=0.634
- clean_pre_1_8_vs_post_control front_radius_q60_slope_px_per_norm_time: n=21 vs 20, residual median diff=0.3228, p=0.1878, AUC=0.621
- clean_pre_1_8_vs_post_control masked_minus_background_mean_slope: n=32 vs 52, residual median diff=0.0001509, p=0.3176, AUC=0.566
- clean_pre_1_8_vs_post_control mask_centroid_path_px: n=32 vs 52, residual median diff=0.6647, p=0.3312, AUC=0.564
- clean_pre_1_8_vs_post_control front_gradient_peak_radius_slope_px_per_norm_time: n=29 vs 41, residual median diff=-0.2199, p=0.4244, AUC=0.443

## Top Echem/Optical Correlations
- shape_V_mean vs kymograph_temporal_energy: n=122, rho=0.592, p=7.049e-13
- all_dq_abs_highV_frac vs kymograph_temporal_energy: n=122, rho=0.540, p=1.344e-10
- all_dq_abs_midV_frac vs kymograph_temporal_energy: n=122, rho=-0.512, p=1.648e-09
- neg_dq_abs_peak_voltage vs masked_minus_background_mean_slope: n=122, rho=0.495, p=6.771e-09
- dqdv_integral_asymmetry vs kymograph_temporal_energy: n=122, rho=-0.490, p=9.901e-09
- neg_dq_abs_peak_voltage vs kymograph_temporal_energy: n=122, rho=-0.487, p=1.328e-08

## Guardrail

Cycle-level echem descriptors are joined to ROI-level automatic front/kymograph proxies, so rows are clustered by cycle and source. Residualization is an explanatory stress test, not causal evidence, calibrated diffusion, validated phase-boundary tracking, or a deployable warning model.
