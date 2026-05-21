# Cycle/Region Residual Mode Context

Cycle, echem, protocol, and coarse particle-region context for protocol-adjusted residual physics modes.

- ROI rows: 52
- Cycles: 11
- Event-enriched mode: `optical_brightening_decorrelating_rollout_hard_front_positive`
- Event-enriched mode fraction: 0.250

## Top Cycle Summaries

- cycle 60.0: n=6, event-enriched fraction=0.833, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;front_negative_high_apparent_front_proxy
- cycle 156.0: n=6, event-enriched fraction=0.667, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;optical_loss_rollout_hard
- cycle 62.0: n=4, event-enriched fraction=0.250, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;optical_loss_rollout_hard;front_negative_high_apparent_front_proxy
- cycle 158.0: n=4, event-enriched fraction=0.250, top modes=near_baseline_or_context_like;optical_brightening_decorrelating_rollout_hard_front_positive;front_negative_high_apparent_front_proxy
- cycle 86.0: n=6, event-enriched fraction=0.167, top modes=near_baseline_or_context_like;optical_brightening_decorrelating_rollout_hard_front_positive
- cycle 116.0: n=6, event-enriched fraction=0.167, top modes=near_baseline_or_context_like;front_negative_high_apparent_front_proxy;optical_brightening_decorrelating_rollout_hard_front_positive
- cycle 58.0: n=4, event-enriched fraction=0.000, top modes=near_baseline_or_context_like
- cycle 88.0: n=4, event-enriched fraction=0.000, top modes=front_negative_high_apparent_front_proxy;near_baseline_or_context_like

## Strongest Context Correlations

- mode_review_priority vs n_frames_percentile: rho=0.716, p=2.46e-09, n=52
- mode_review_priority vs block_fraction_elapsed: rho=-0.615, p=0.0193, n=14
- mode_review_priority vs V_mean: rho=-0.503, p=0.0013, n=38
- is_event_enriched_mode vs block_fraction_elapsed: rho=-0.434, p=0.121, n=14
- mode_pc1 vs object_y_full_approx: rho=0.387, p=0.00464, n=52
- mode_review_priority vs cycles_to_block_end: rho=0.378, p=0.00569, n=52
- mode_review_priority vs V_range: rho=-0.371, p=0.0219, n=38
- is_high_apparent_front_proxy_mode vs block_fraction_elapsed: rho=0.367, p=0.197, n=14
- is_event_enriched_mode vs n_frames_percentile: rho=0.343, p=0.0127, n=52
- mode_pc1 vs object_x_full_approx: rho=-0.307, p=0.0269, n=52
- mode_review_priority vs cycles_from_block_start: rho=0.294, p=0.0342, n=52
- mode_pc1 vs object_candidate_rank: rho=0.287, p=0.0393, n=52

## Guardrail

Cycle/region mode context is descriptive and uses automatic ROI coordinates plus automatic residual-mode labels; use it to prioritize manual review, not as proof of spatial mechanism or calibrated degradation physics.
