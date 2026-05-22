# Source-Balanced Pre-Event Masked Rollout Benchmark

- Input rows / OK / failed: 128 / 128 / 0
- Methods: ['persistence', 'velocity', 'pixel_linear', 'particle_mean_drift', 'radial_profile_trend']
- Best method by median particle gain: {'method': 'persistence', 'particle_mse_median': 0.0014861117404862563, 'particle_mse_mean': 0.003116806463893457, 'particle_mse_ratio_vs_persistence_median': 1.0, 'particle_mse_gain_vs_persistence_median': 0.0, 'wins_vs_persistence': 0}
- Top event test: {'target': 'near_vs_post_control', 'feature': 'persistence_particle_mse', 'transform': 'raw', 'n': 84, 'n_positive': 32, 'direction': 'higher_in_positive', 'oriented_auc': 0.6616586538461539, 'average_precision': 0.6188209476546704, 'median_positive_minus_negative': 0.0006750457312271704, 'mwu_p': 0.013393147791856587, 'spearman_rho': 0.2719679533106391, 'spearman_p': 0.012326949902347287}

Outputs:
- `source_balanced_pre_event_masked_rollout_per_roi.csv`
- `source_balanced_pre_event_masked_rollout_frame_samples.csv`
- `source_balanced_pre_event_masked_rollout_method_summary.csv`
- `source_balanced_pre_event_masked_rollout_event_tests.csv`
- `source_balanced_pre_event_masked_rollout_source_summary.csv`
- `source_balanced_pre_event_masked_rollout_summary.json`

Guardrail:
Rollouts use history-derived automatic particle masks and held-out future frames within each ROI crop. They test interpretable particle-region prediction baselines, not manual segmentation, calibrated phase-boundary motion, or material diffusion.
