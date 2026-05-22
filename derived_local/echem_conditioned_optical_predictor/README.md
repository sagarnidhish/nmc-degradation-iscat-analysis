# Echem-Conditioned Optical Predictor

Controlled cycle-level comparison of acquisition context versus echem regime descriptors for weak optical degradation targets.

- Cycles: 89
- Targets: future_any_drop_within_8cycles, high_cross_modal_consensus_q75, high_particle_norm_cv_q75, high_roi_phase_slope_abs_q75, high_state_step_norm_q75
- Feature sets: {'acquisition_context': 7, 'echem_regime': 49, 'echem_plus_acquisition': 56, 'cycle_state_upper_bound': 14}

## Top Metrics
- cycle_block_cv high_roi_phase_slope_abs_q75 echem_regime: AUC=1, AP=1, n=24, positives=6
- cycle_block_cv high_roi_phase_slope_abs_q75 echem_plus_acquisition: AUC=0.991, AP=0.976, n=24, positives=6
- cycle_block_cv high_roi_phase_slope_abs_q75 acquisition_context: AUC=0.944, AP=0.735, n=24, positives=6
- cycle_block_cv high_roi_phase_slope_abs_q75 cycle_state_upper_bound: AUC=0.926, AP=0.766, n=24, positives=6
- cycle_block_cv high_state_step_norm_q75 cycle_state_upper_bound: AUC=0.857, AP=0.745, n=88, positives=22
- cycle_block_cv high_cross_modal_consensus_q75 cycle_state_upper_bound: AUC=0.846, AP=0.716, n=89, positives=23
- cycle_block_cv high_cross_modal_consensus_q75 echem_plus_acquisition: AUC=0.804, AP=0.632, n=89, positives=23
- cycle_block_cv high_cross_modal_consensus_q75 echem_regime: AUC=0.76, AP=0.424, n=89, positives=23
- cycle_block_cv high_cross_modal_consensus_q75 acquisition_context: AUC=0.73, AP=0.62, n=89, positives=23
- cycle_block_cv high_particle_norm_cv_q75 cycle_state_upper_bound: AUC=0.659, AP=0.434, n=89, positives=23
- cycle_block_cv high_particle_norm_cv_q75 echem_regime: AUC=0.634, AP=0.392, n=89, positives=23
- cycle_block_cv high_particle_norm_cv_q75 echem_plus_acquisition: AUC=0.609, AP=0.387, n=89, positives=23

## Top Echem Feature-Set Gains
- cycle_block_cv high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC=0.152, base=0.482, compare=0.634
- cycle_block_cv high_particle_norm_cv_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.127, base=0.482, compare=0.609
- cycle_block_cv future_any_drop_within_8cycles echem_regime_minus_acquisition: delta AUC=0.124, base=0.316, compare=0.44
- rolling_origin_block future_any_drop_within_8cycles echem_regime_minus_acquisition: delta AUC=0.112, base=0.463, compare=0.574
- rolling_origin_block future_any_drop_within_8cycles echem_plus_acquisition_minus_acquisition: delta AUC=0.0759, base=0.463, compare=0.539
- cycle_block_cv high_cross_modal_consensus_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.0744, base=0.73, compare=0.804
- cycle_block_cv high_roi_phase_slope_abs_q75 echem_regime_minus_acquisition: delta AUC=0.0556, base=0.944, compare=1
- cycle_block_cv high_roi_phase_slope_abs_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.0463, base=0.944, compare=0.991
- rolling_origin_block high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC=0.0358, base=0.371, compare=0.407
- cycle_block_cv high_cross_modal_consensus_q75 echem_regime_minus_acquisition: delta AUC=0.0303, base=0.73, compare=0.76

## Guardrail

This is a cycle-level weak-label model comparison with blocked cycle-CV and rolling-origin block splits. Echem-regime gains show conditional association, not deployable prediction, causal mechanism, calibrated dQ/dV, or validated front/diffusion physics. Rare 2-4 positive targets are excluded from the model-comparison table, and the permutation null shuffles labels against held-out prediction scores rather than retraining every permutation.
