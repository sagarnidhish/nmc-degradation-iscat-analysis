# Source-Balanced Pre-Event Echem-Matched Far-Control Audit

- Rows/cycles/sources: 128 / 64 / 14
- Echem match features: ['capacity_mAh', 'capacity_fade_from_first_mAh', 'capacity_fraction_of_first', 'coulombic_efficiency_pct', 'charge_discharge_capacity_abs_gap_mAh', 'voltage_peak_hysteresis_proxy', 'highV_charge_discharge_imbalance', 'midV_charge_discharge_imbalance', 'lowV_charge_discharge_imbalance', 'echem_outlier_score', 'echem_regime_pc1', 'echem_regime_pc2', 'echem_regime_pc3', 'echem_regime_pc4', 'shape_V_mean', 'shape_V_range', 'shape_I_abs_mean_mA', 'all_dq_abs_highV_frac', 'all_dq_abs_midV_frac', 'all_dq_abs_lowV_frac']
- Context match features: ['cycleNo', 'local_cycle_index', 'expansion_cycle_rank', 'object_candidate_rank', 'object_x_full_approx', 'object_y_full_approx', 'object_area_ds_px', 'object_mean_abs_z', 'crop_x0', 'crop_y0', 'roi_norm_mean_first', 'stage_drift_xy_sampled', 'mask_base_area_fraction']
- Pair counts: {'global_echem_context': 32, 'same_source_class_context_only': 32, 'same_source_class_echem_context': 32}

## Top Paired Near-vs-Far Tests
- same_source_class_context_only front_radius2_slope_px2_per_norm_time: n=19, median near-far diff=15.95, positive fraction=0.842, p=0.000999
- same_source_class_echem_context front_radius2_slope_px2_per_norm_time: n=20, median near-far diff=8.794, positive fraction=0.750, p=0.000999
- same_source_class_context_only front_radius_slope_px_per_norm_time: n=19, median near-far diff=1.128, positive fraction=0.842, p=0.000999
- same_source_class_context_only masked_minus_background_mean_median: n=32, median near-far diff=0.005962, positive fraction=0.656, p=0.000999
- same_source_class_context_only kymograph_temporal_energy: n=32, median near-far diff=5.358e-06, positive fraction=0.688, p=0.001998
- same_source_class_echem_context kymograph_temporal_energy: n=32, median near-far diff=2.533e-06, positive fraction=0.625, p=0.001998
- same_source_class_echem_context mask_centroid_path_px: n=32, median near-far diff=1.092, positive fraction=0.781, p=0.002997
- same_source_class_context_only mask_centroid_path_px: n=32, median near-far diff=0.9568, positive fraction=0.750, p=0.003996
- same_source_class_echem_context front_radius_slope_px_per_norm_time: n=20, median near-far diff=0.6746, positive fraction=0.750, p=0.003996
- global_echem_context masked_minus_background_mean_slope: n=32, median near-far diff=0.002636, positive fraction=0.688, p=0.003996

## Guardrail

This is a cross-source far-control stress test because no same-source near-vs-far lattice exists. Source-class and echem/context matching reduce obvious confounding but cannot prove particle identity, causality, calibrated diffusion, or validated phase-boundary tracking.
