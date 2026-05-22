# Source-Balanced Pre-Event Front Consensus Audit

- Rows/cycles/sources: 128 / 64 / 14
- Consensus features tested: front_consensus_score, front_residual_outward_z_mean, front_raw_outward_z_mean, front_quantile_positive_fraction, front_q_slope_mean_source_echem_residual, front_q_slope_mean_px_per_norm_time, front_q_slope_cv, front_quality_score

## Top Event-Bin Consensus Tests
- near_vs_any_non_near front_quantile_positive_fraction: n=32 vs 96, median diff=0.3333, AUC=0.599, p=0.07165
- near_vs_any_non_near front_residual_outward_z_mean: n=31 vs 83, median diff=0.2416, AUC=0.594, p=0.1248
- near_vs_any_non_near front_raw_outward_z_mean: n=31 vs 83, median diff=0.2936, AUC=0.592, p=0.1329
- near_vs_any_non_near front_consensus_score: n=32 vs 96, median diff=1.191, AUC=0.578, p=0.19
- near_vs_any_non_near front_q_slope_mean_px_per_norm_time: n=27 vs 70, median diff=0.09149, AUC=0.554, p=0.4094
- near_vs_any_non_near front_quality_score: n=32 vs 96, median diff=0, AUC=0.458, p=0.4738
- near_vs_any_non_near front_q_slope_mean_source_echem_residual: n=27 vs 70, median diff=-0.001894, AUC=0.539, p=0.5541
- near_vs_any_non_near front_q_slope_cv: n=21 vs 46, median diff=0.02202, AUC=0.488, p=0.8764

## Top Matched Consensus Tests
- near_vs_mid_pre same_source front_consensus_score: n=20, median diff=4.946, sign-flip p=0.000999
- near_vs_mid_pre same_source front_raw_outward_z_mean: n=20, median diff=2.473, sign-flip p=0.000999
- near_vs_mid_pre source_penalized_echem_context front_consensus_score: n=32, median diff=2.428, sign-flip p=0.000999
- near_vs_mid_pre same_source front_q_slope_mean_px_per_norm_time: n=18, median diff=1.365, sign-flip p=0.000999
- near_vs_mid_pre source_penalized_echem_context front_q_slope_mean_px_per_norm_time: n=18, median diff=1.174, sign-flip p=0.000999
- near_vs_any_non_near same_source front_raw_outward_z_mean: n=27, median diff=1.074, sign-flip p=0.000999
- near_vs_mid_pre source_penalized_echem_context front_raw_outward_z_mean: n=28, median diff=0.9088, sign-flip p=0.000999
- near_vs_mid_pre source_penalized_echem_context front_residual_outward_z_mean: n=28, median diff=0.8169, sign-flip p=0.000999

## Guardrail

The consensus score combines automatic radial-front, kymograph, and source+echem residual proxies. It tests internal coherence and prioritizes manual QC, but it is still not calibrated diffusion, validated phase-boundary tracking, particle-identity proof, causal degradation evidence, or a deployable warning model.
