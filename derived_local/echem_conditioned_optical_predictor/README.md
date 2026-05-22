# Echem-Conditioned Optical Predictor

Controlled cycle-level comparison of acquisition context versus echem regime descriptors for weak optical degradation targets.

- Cycles: 89
- Targets: future_any_drop_within_8cycles, synchronized_multimodal_candidate, multimodal_outlier_without_trace_drop, high_cross_modal_consensus_q75, high_particle_norm_cv_q75, high_roi_phase_slope_abs_q75, high_state_step_norm_q75
- Feature sets: {'acquisition_context': 7, 'echem_regime': 49, 'echem_plus_acquisition': 56, 'cycle_state_upper_bound': 14}

## Top Metrics
- leave_one_cycle synchronized_multimodal_candidate cycle_state_upper_bound: AUC=0.994, AP=0.833, n=89, positives=2
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_plus_acquisition: AUC=0.991, AP=0.976, n=24, positives=6
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_regime: AUC=0.981, AP=0.958, n=24, positives=6
- leave_one_cycle synchronized_multimodal_candidate acquisition_context: AUC=0.966, AP=0.325, n=89, positives=2
- leave_one_cycle high_roi_phase_slope_abs_q75 acquisition_context: AUC=0.944, AP=0.735, n=24, positives=6
- leave_one_cycle high_roi_phase_slope_abs_q75 cycle_state_upper_bound: AUC=0.944, AP=0.735, n=24, positives=6
- leave_one_cycle high_state_step_norm_q75 cycle_state_upper_bound: AUC=0.899, AP=0.835, n=88, positives=22
- leave_one_cycle high_cross_modal_consensus_q75 cycle_state_upper_bound: AUC=0.829, AP=0.684, n=89, positives=23
- leave_one_cycle future_any_drop_within_8cycles cycle_state_upper_bound: AUC=0.808, AP=0.53, n=89, positives=20
- leave_one_cycle high_cross_modal_consensus_q75 echem_regime: AUC=0.799, AP=0.522, n=89, positives=23
- leave_one_cycle high_cross_modal_consensus_q75 echem_plus_acquisition: AUC=0.797, AP=0.64, n=89, positives=23
- leave_one_cycle high_particle_norm_cv_q75 cycle_state_upper_bound: AUC=0.723, AP=0.515, n=89, positives=23

## Top Echem Feature-Set Gains
- leave_one_cycle high_cross_modal_consensus_q75 echem_regime_minus_acquisition: delta AUC=0.113, base=0.686, compare=0.799
- leave_one_cycle high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC=0.113, base=0.527, compare=0.64
- leave_one_cycle high_cross_modal_consensus_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.111, base=0.686, compare=0.797
- leave_one_cycle high_particle_norm_cv_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.103, base=0.527, compare=0.63
- rolling_origin high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC=0.084, base=0.525, compare=0.609
- rolling_origin high_cross_modal_consensus_q75 echem_regime_minus_acquisition: delta AUC=0.0668, base=0.625, compare=0.692
- rolling_origin high_particle_norm_cv_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.0519, base=0.525, compare=0.577
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.0463, base=0.944, compare=0.991
- leave_one_cycle high_roi_phase_slope_abs_q75 echem_regime_minus_acquisition: delta AUC=0.037, base=0.944, compare=0.981
- rolling_origin high_cross_modal_consensus_q75 echem_plus_acquisition_minus_acquisition: delta AUC=0.0137, base=0.625, compare=0.639

## Guardrail

This is a cycle-level weak-label model comparison. Echem-regime gains show conditional association, not deployable prediction, causal mechanism, calibrated dQ/dV, or validated front/diffusion physics.
