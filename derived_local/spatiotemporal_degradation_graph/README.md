# Spatiotemporal Degradation Graph

Nearest-neighbor graph audit for NMC ROI degradation modes, front residuals, cycles, and particle regions.

- Nodes: 52
- Edges: 510
- k-neighbors: 3

## Top Homophily Tests
- same_cycle_spatial_knn front_positive_residual_binary: same=0.936, null_mean=0.489, p=0.0010
- same_cycle_spatial_knn is_event_roi: same=1.000, null_mean=0.473, p=0.0010
- same_cycle_spatial_knn is_event_enriched_mode: same=0.769, null_mean=0.647, p=0.0060
- same_reference_spatial_knn is_event_enriched_mode: same=0.673, null_mean=0.648, p=0.2637
- next_cycle_spatial_knn is_event_enriched_mode: same=0.602, null_mean=0.645, p=0.8881
- previous_cycle_spatial_knn is_event_enriched_mode: same=0.544, null_mean=0.628, p=0.9750
- same_reference_spatial_knn front_positive_residual_binary: same=0.404, null_mean=0.490, p=0.9930
- same_reference_spatial_knn is_event_roi: same=0.378, null_mean=0.474, p=0.9950

## Top Continuous Neighbor Tests
- same_cycle_spatial_knn phase_slope_positive_fraction_protocol_residual: rho=0.696, null_mean=-0.012, p=0.0010
- same_reference_spatial_knn threshold_robust_phase_score_protocol_residual: rho=0.275, null_mean=-0.067, p=0.0030
- same_cycle_spatial_knn mode_review_priority: rho=0.814, null_mean=0.737, p=0.0100
- next_cycle_spatial_knn mode_review_priority: rho=0.783, null_mean=0.723, p=0.0240
- next_cycle_spatial_knn phase_slope_positive_fraction_protocol_residual: rho=-0.219, null_mean=-0.008, p=0.0240
- next_cycle_spatial_knn diffusion_proxy_abs_median_um2_per_s_protocol_residual: rho=-0.227, null_mean=-0.064, p=0.0360
- previous_cycle_spatial_knn threshold_robust_phase_score_protocol_residual: rho=0.221, null_mean=-0.066, p=0.0549
- same_reference_spatial_knn phase_slope_positive_fraction_protocol_residual: rho=-0.174, null_mean=-0.017, p=0.0889

## Guardrail

Spatiotemporal graph tests use automatic ROI coordinates and automatic residual labels on a selected 52-ROI cohort. They test clustering/propagation hypotheses for review prioritization, not causal material degradation mechanisms.
