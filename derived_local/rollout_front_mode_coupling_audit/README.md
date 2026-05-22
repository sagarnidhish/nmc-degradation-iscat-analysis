# Rollout Front/Mode Coupling Audit

- Rows / sources / cycles / modes: 128 / 14 / 64 / 2
- Top feature test: {'target': 'future8', 'feature': 'qc_review_score', 'transform': 'raw', 'n': 128, 'n_positive': 32, 'direction': 'higher_in_positive', 'oriented_auc': 0.8313802083333333, 'average_precision': 0.7186165521579553, 'median_positive_minus_negative': 0.26197916666666676, 'mwu_p': 2.154457602779011e-08, 'spearman_rho': 0.49708548257652985, 'spearman_p': 2.3951728148448424e-09}
- Top source-residual correlation: {'rollout_feature': 'persistence_rollout_hybrid_mse', 'physics_feature': 'observable_tail_score', 'transform': 'source_residual', 'n': 128, 'spearman_rho': 0.42140684634331277, 'spearman_p': 7.285428386574524e-07, 'abs_rho': 0.42140684634331277}

Outputs:
- `rollout_front_mode_coupling_merged.csv`
- `rollout_front_mode_coupling_feature_tests.csv`
- `rollout_front_mode_coupling_correlations.csv`
- `rollout_front_mode_coupling_mode_summary.csv`
- `rollout_front_mode_coupling_review_queue.csv`
- `rollout_front_mode_coupling_summary.json`

Guardrail:
This audit couples automatic particle-only rollout residuals to automatic front/phase/transport/mode descriptors. It nominates physically interpretable review rows and source-robust associations, but does not validate manual phase boundaries or calibrated diffusion coefficients.
