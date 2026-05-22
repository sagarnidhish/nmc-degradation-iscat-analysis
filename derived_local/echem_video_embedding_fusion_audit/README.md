# Echem Video Embedding Fusion Audit

Fuses masked particle-video embeddings with cycle-level echem regime descriptors and compares feature families under leave-one-cycle-out models.

- Rows: 172
- Cycles: 34
- Classification targets: future_any_drop_within_8cycles, future_any_drop_within_16cycles
- Regression targets: transferred_masked_residual_signature, threshold_robust_phase_score, phase_slope_abs_median_per_s, diffusion_proxy_abs_median_um2_per_s, persistence_particle_mse_fraction_of_full_mean, velocity_particle_mse_fraction_of_full_mean, low_rank_dmd_particle_mse_fraction_of_full_mean, object_mean_residual, cross_modal_consensus_score, particle_norm_cv
- Feature set sizes: {'acquisition_context': 24, 'echem_regime': 51, 'video_scalar': 48, 'video_embedding': 16, 'video_all': 64, 'video_plus_echem': 115, 'video_plus_echem_acquisition': 138}

Guardrail: This is a grouped weak-label fusion audit over automatic masked-video embeddings, echem descriptors, and automatic ROI physics. It supports representation design and review prioritization, not deployable warning, manual particle/front labels, or calibrated diffusion claims.
