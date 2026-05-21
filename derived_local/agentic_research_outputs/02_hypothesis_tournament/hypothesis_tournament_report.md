# Hypothesis Tournament

Ranked hypotheses using current derived evidence, required controls, and skeptical risks.

| id | tournament_score | support_score | statement | critical_next_tests | skeptical_risk |
| --- | --- | --- | --- | --- | --- |
| H3_event_precursors_are_weak_but_present | 16.87 | 1 | Prior-cycle optical features contain weak but nonzero warning signal for abrupt degradation events. | leave_one_particle_out; cycle_blocked_cv; calibrated_hazard_model | Rare events and particle grouping can inflate apparent precursor skill. |
| H1_sync_events_are_real_degradation | 16 | 1 | The same-cycle abrupt optical drops around cycles 86 and 116 are coordinated degradation events rather than independent particle noise. | shuffle_cycle_control; raw_roi_visual_qc; background_region_control | Protocol or frame-count artifacts can create synchronized apparent events. |
| H4_degradation_modes_are_particle_region_specific | 16 | 1 | Particles and regions differ in degradation mode, with synchronized events superimposed on heterogeneous gradual drift. | particle_random_effects; region_feature_clustering; spatial_covariate_controls | Only four example particles are currently summarized, limiting spatial inference. |
| H5_observable_forecasting_more_useful_than_pixel_rollout | 16 | 1 | For this project stage, forecasting physical observables and event hazards is more scientifically useful than unconstrained pixel rollout. | observable_forecast_vs_persistence; event_hazard_calibration; phase_or_protocol_stratified_metrics | Observable compression may miss spatial crack morphology or phase-front movement. |
| H6_apparent_transport_requires_validation | 16 | 1 | Phase-boundary/front-motion and apparent diffusion extraction should be treated as readiness-ranked proxies until visual validation and spatial calibration are complete. | front_overlay_review; monotonicity_filter; calibration_metadata_check | Apparent coefficients can be physically misleading without calibration. |
| H2_short_frame_protocol_context | 14.67 | 1 | Synchronized event cycles are associated with unusually short cycle-frame regimes and protocol context changes. | local_protocol_window_scan; frame_count_matched_controls | Frame-count association may be an imaging artifact, not electrochemical degradation. |

## Rule

High score means the hypothesis is ready for targeted validation, not that it is proven.
