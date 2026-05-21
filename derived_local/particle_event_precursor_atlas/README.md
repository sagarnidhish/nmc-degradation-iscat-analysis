# Particle Event Precursor Atlas

Event-aligned cycle-level precursor windows from the normalized four-particle trace table.

- Event anchors: 4
- Candidate control anchors: 17
- Matched control anchors: 24

## Top Precursor Window Tests
- pre8_to_pre5 capacity_mAh min_value: event-control=-0.0238, p=0.00171
- pre16_to_pre9 coulombic_efficiency_pct min_value: event-control=-1.11, p=0.00194
- pre16_to_pre9 coulombic_efficiency_pct mean_value: event-control=-0.421, p=0.00223
- pre16_to_pre9 delta_std_across_particles max_value: event-control=0.0396, p=0.00665
- pre16_to_pre9 n_frames slope_per_cycle: event-control=-2.18, p=0.0118
- pre16_to_pre9 frames_percentile slope_per_cycle: event-control=-0.0191, p=0.0212
- pre16_to_pre9 coulombic_efficiency_pct max_value: event-control=-0.299, p=0.024
- pre4_to_pre1 particle_norm_mean min_value: event-control=0.0473, p=0.0267
- pre8_to_pre5 mean_delta_prev max_value: event-control=-0.0174, p=0.03
- pre16_to_pre9 V_max slope_per_cycle: event-control=-0.000516, p=0.0367

## Top All Window Tests
- event_cycle max_abs_delta_prev max_value: event-control=0.136, p=0.00167
- event_cycle max_abs_delta_prev mean_value: event-control=0.136, p=0.00167
- event_cycle max_abs_delta_prev median_value: event-control=0.136, p=0.00167
- event_cycle max_abs_delta_prev min_value: event-control=0.136, p=0.00167
- pre8_to_pre5 capacity_mAh min_value: event-control=-0.0238, p=0.00171
- post1_to_post8 max_abs_delta_prev max_value: event-control=-0.0585, p=0.00174
- event_cycle mean_delta_prev max_value: event-control=-0.109, p=0.00176
- event_cycle mean_delta_prev mean_value: event-control=-0.109, p=0.00176
- event_cycle mean_delta_prev median_value: event-control=-0.109, p=0.00176
- event_cycle mean_delta_prev min_value: event-control=-0.109, p=0.00176

## Guardrail

Precursor windows are aligned to four detected abrupt-drop cycles and matched non-event anchors from the four-particle cycle table. Results show cycle-level trace precursors for review and hypothesis generation, not localized phase-front motion or calibrated diffusion.
