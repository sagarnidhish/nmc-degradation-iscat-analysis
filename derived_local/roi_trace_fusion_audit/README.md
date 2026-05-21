# ROI Trace Fusion Audit

Cycle-level four-particle trace state joined onto ROI/front/kinetic residual descriptors.

- ROI rows: 52
- Event ROI rows: 24
- Event-enriched mode rows: 13
- Trace lag predictors: 100

## Top Precursor Context-Residual Associations
- trace_lag2_particle_norm_range vs phase_slope_positive_fraction_protocol_residual: rho=0.725, p=2.69e-07, n=38
- trace_lag2_particle_norm_std vs phase_slope_positive_fraction_protocol_residual: rho=0.725, p=2.69e-07, n=38
- trace_lag4_delta_std_across_particles vs phase_slope_positive_fraction_protocol_residual: rho=-0.705, p=7.72e-07, n=38
- lag16_trace_predprob_future_any_drop_within_8cycles vs phase_slope_positive_fraction_protocol_residual: rho=-0.652, p=9.32e-06, n=38
- trace_lag4_particle_norm_cv vs phase_slope_positive_fraction_protocol_residual: rho=0.652, p=9.32e-06, n=38
- trace_lag2_particle_norm_cv vs phase_slope_positive_fraction_protocol_residual: rho=0.641, p=1.45e-05, n=38
- trace_lag16_frames_percentile vs phase_slope_positive_fraction_protocol_residual: rho=0.6, p=6.81e-05, n=38
- trace_lag16_max_abs_delta_prev vs phase_slope_positive_fraction_protocol_residual: rho=0.6, p=6.81e-05, n=38
- trace_lag16_n_frames vs phase_slope_positive_fraction_protocol_residual: rho=0.6, p=6.81e-05, n=38
- trace_lag4_V_max vs phase_slope_positive_fraction_protocol_residual: rho=0.586, p=0.000111, n=38

## Top Event-Enriched Mode Precursor Tests
- is_event_enriched_mode trace_lag16_delta_std_across_particles: median diff=-0.00726, p=0.00491
- is_event_enriched_mode trace_lag8_frames_percentile: median diff=0.185, p=0.0063
- is_event_enriched_mode trace_lag8_n_frames: median diff=17, p=0.0063
- is_event_enriched_mode trace_lag8_future_sync2_drop_within_8cycles: median diff=-1, p=0.0113
- is_event_enriched_mode trace_lag16_V_max: median diff=0.00154, p=0.0135
- is_event_enriched_mode trace_lag16_frames_percentile: median diff=0.135, p=0.0146
- is_event_enriched_mode trace_lag16_n_frames: median diff=12, p=0.0146
- is_event_enriched_mode trace_lag16_coulombic_efficiency_pct: median diff=-0.442, p=0.0169
- is_event_enriched_mode trace_lag2_particle_norm_mean: median diff=0.0254, p=0.0169
- is_event_enriched_mode trace_lag2_mean_delta_prev: median diff=0.0439, p=0.019

## Full Audit Note

Full CSV tables also include lag-0/event-label checks, which are useful sanity checks but not interpreted as precursor evidence.

## Guardrail

Trace lags are cycle-level four-particle/echem summaries attached to selected ROI rows by cycle number. Associations are useful for linking global precursor state to ROI/front outcomes, but rows are not independent within cycle and this does not prove localized causality.
