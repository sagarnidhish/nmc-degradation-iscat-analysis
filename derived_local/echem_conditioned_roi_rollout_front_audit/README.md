# Echem-Conditioned ROI Rollout/Front Audit

This audit joins cycle-level electrochemical regime descriptors to balanced future ROI physics rows.
It compares acquisition/context features with echem-regime features for continuous ROI rollout and front-motion targets under leave-one-cycle-out regression.

- ROI rows: 72
- Cycles: 24
- Targets: low_rank_dmd_particle_mse_fraction_of_full_mean, low_rank_dmd_particle_to_nonparticle_mse_ratio_mean, persistence_particle_mse_fraction_of_full_mean, velocity_particle_mse_fraction_of_full_mean, transferred_masked_residual_signature, phase_slope_abs_median_per_s, threshold_robust_phase_score, diffusion_proxy_abs_median_um2_per_s, radius2_slope_median_px2_per_s, phase_slope_positive_fraction, roi_norm_mean_delta_last_minus_first, object_mean_residual
- Feature set sizes: {'acquisition_context': 10, 'echem_regime': 49, 'echem_plus_acquisition': 59}

Guardrail: ROI rows are automatic, clustered by cycle, and front/diffusion variables are proxy measurements; use this as a weak-label explanatory audit, not calibrated electrochemical mechanism proof.
