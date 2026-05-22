# Echem-Conditioned Optical Predictor

Controlled cycle-level comparison of acquisition context versus echem regime descriptors for weak optical degradation targets.

- Cycles: 89
- Targets: future_any_drop_within_8cycles, high_cross_modal_consensus_q75, high_particle_norm_cv_q75, high_roi_phase_slope_abs_q75, high_state_step_norm_q75
- Feature sets: {'acquisition_context': 7, 'echem_regime': 49, 'echem_plus_acquisition': 56, 'cycle_state_upper_bound': 14}

## Top Metrics
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_plus_acquisition: AUC=0.981, AP=0.958, n=24, positives=6
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_regime: AUC=0.944, AP=0.842, n=24, positives=6
- leave_one_cycle high_roi_phase_slope_abs_q75 acquisition_context: AUC=0.944, AP=0.735, n=24, positives=6
- leave_one_cycle high_roi_phase_slope_abs_q75 cycle_state_upper_bound: AUC=0.935, AP=0.717, n=24, positives=6
- leave_one_cycle high_state_step_norm_q75 cycle_state_upper_bound: AUC=0.903, AP=0.838, n=88, positives=22
- leave_one_cycle high_cross_modal_consensus_q75 cycle_state_upper_bound: AUC=0.834, AP=0.681, n=89, positives=23
- leave_one_cycle future_any_drop_within_8cycles cycle_state_upper_bound: AUC=0.812, AP=0.537, n=89, positives=20
- leave_one_cycle high_cross_modal_consensus_q75 echem_plus_acquisition: AUC=0.789, AP=0.638, n=89, positives=23
- leave_one_cycle high_cross_modal_consensus_q75 echem_regime: AUC=0.785, AP=0.509, n=89, positives=23
- leave_one_cycle high_particle_norm_cv_q75 cycle_state_upper_bound: AUC=0.727, AP=0.518, n=89, positives=23
- leave_one_cycle high_cross_modal_consensus_q75 acquisition_context: AUC=0.688, AP=0.583, n=89, positives=23
- leave_one_cycle future_any_drop_within_8cycles acquisition_context: AUC=0.676, AP=0.367, n=89, positives=20

## Top Echem Feature-Set Gains
- leave_one_cycle high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC=0.116, base=0.527, compare=0.643
- leave_one_cycle high_particle_norm_cv_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.115, base=0.527, compare=0.642
- rolling_origin high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC=0.101, base=0.502, compare=0.603
- leave_one_cycle high_cross_modal_consensus_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.101, base=0.688, compare=0.789
- leave_one_cycle high_cross_modal_consensus_q75 echem_regime_minus_acquisition: delta AUC=0.0968, base=0.688, compare=0.785
- rolling_origin high_particle_norm_cv_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.0755, base=0.502, compare=0.577
- rolling_origin high_cross_modal_consensus_q75 echem_regime_minus_acquisition: delta AUC=0.0595, base=0.626, compare=0.686
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.037, base=0.944, compare=0.981
- rolling_origin high_cross_modal_consensus_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.0101, base=0.626, compare=0.636
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_regime_minus_acquisition: delta AUC=0, base=0.944, compare=0.944

## Guardrail

This is a cycle-level weak-label model comparison. Echem-regime gains show conditional association, not deployable prediction, causal mechanism, calibrated dQ/dV, or validated front/diffusion physics. Rare 2-4 positive targets are excluded from the model-comparison table, and the permutation null shuffles labels against held-out prediction scores rather than retraining every permutation.
