# History/Fallback Masked Rollout Ablation

- Input / OK / failures / sources: 128 / 128 / 0 / 14
- Median fallback frame fraction: 0.28645833333333337
- Median one-step adaptive/hybrid MSE: 4.1262255804869073e-05 / 3.672906023113427e-05
- Median latent-linear gain vs persistence, history/hybrid masks: 0.15822063730376987 / 0.08166849667499436
- Top event test: {'target': 'near_vs_far_pre', 'feature': 'fallback_frame_fraction', 'transform': 'raw', 'n': 54, 'n_positive': 32, 'direction': 'higher_in_positive', 'oriented_auc': 0.7897727272727273, 'average_precision': 0.799273100960755, 'median_positive_minus_negative': 0.2864583333333333, 'mwu_p': 0.00032801530289219854, 'spearman_rho': 0.4946261408860768, 'spearman_p': 0.00014363490031071374}

Outputs:
- `history_fallback_masked_rollout_ablation_per_roi.csv`
- `history_fallback_masked_rollout_ablation_event_tests.csv`
- `history_fallback_masked_rollout_ablation_source_summary.csv`
- `history_fallback_masked_rollout_ablation_method_summary.csv`
- `history_fallback_masked_rollout_ablation_failures.csv`
- `history_fallback_masked_rollout_ablation_summary.json`

Guardrail:
This is an automatic particle-mask ablation and compact latent-linear rollout benchmark. It uses history/fallback particle support for robustness under drift-correction blur, but it does not provide manual segmentation, validated phase-boundary velocities, or calibrated diffusion coefficients.
