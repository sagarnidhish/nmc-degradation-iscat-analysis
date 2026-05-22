# Source-Balanced Pre-Event Transport/Kinetic Fusion Audit

- Rows/cycles/sources: 128 / 64 / 14
- Event bins: {'post_event_1_16': 40, 'near_pre_event_1_8': 32, 'mid_pre_event_9_16': 22, 'far_pre_event_17_32': 22, 'no_near_event_control': 12}
- Feature set sizes: {'transport_only': 8, 'rollout_only': 5, 'kinetic_front': 14, 'transport_plus_kinetic_front': 22, 'all_core_no_visual': 27}
- Top event test: {'target': 'near_vs_any_non_near', 'score': 'manual_qc_augmented_fusion_score', 'n': 128, 'n_positive': 32, 'direction': 'higher_in_positive', 'oriented_auc': 0.7848307291666667, 'average_precision': 0.6163904199439082, 'median_positive_minus_negative_oriented': 0.8807904078524048, 'mwu_p': 1.4926703171131616e-06, 'spearman_rho_oriented': 0.4272591328629309, 'spearman_p': 4.911097448218114e-07, 'source_stratified_permutation_p': 0.003992015968063872}
- Top leave-source model: {'target': 'near_vs_post_control', 'feature_set': 'transport_only', 'status': 'ok', 'n_features': 8, 'features': ['abs_radial_flow_mean', 'abs_radial_flow_q90', 'particle_flow_mag_mean', 'particle_flow_mag_q90', 'curl_mean', 'curl_q90', 'flow_acceleration_mean', 'apparent_transport_instability_score'], 'n_eval': 18.0, 'n_positive': 12.0, 'n_sources_eval': 2.0, 'n_folds': 2, 'n_skipped_folds': 9, 'roc_auc': 0.888888888888889, 'average_precision': 0.9594017094017094, 'spearman_rho': 0.6360351905939566, 'spearman_p': 0.004548232236704297, 'n_scored': nan}

Outputs:
- `source_balanced_pre_event_transport_kinetic_fusion_per_roi.csv`
- `source_balanced_pre_event_transport_kinetic_fusion_event_tests.csv`
- `source_balanced_pre_event_transport_kinetic_fusion_leave_source_models.csv`
- `source_balanced_pre_event_transport_kinetic_fusion_ranked_candidates.csv`

Guardrail:
Fusion scores join automatic transport, rollout, front, and kinetic descriptors from history-derived particle ROI crops. They are ranking and source-aware hypothesis-generation tools, not manual QC labels, calibrated phase-boundary velocities, diffusion coefficients, or causal degradation proof.
