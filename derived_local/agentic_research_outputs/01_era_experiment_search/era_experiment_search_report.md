# ERA-Style Experiment Search

Root: `/scratch/u6hp/nsagar.u6hp/Alek_Jiho`

## Top Ranked Experiments

| id | status | readiness | era_score | question | priority_reason |
| --- | --- | --- | --- | --- | --- |
| E1_frame_count_matched_event_artifact_control | ready | 1 | 43.5 | Are synchronized events still enriched after matching cycles by frame count and protocol block? | Separates physical degradation from short-cycle imaging/protocol artifact. |
| E2_protocol_local_echem_window_scan | ready | 1 | 40.6 | Do cycles 86/116 show local dV/dt, dQ/dV, current, or capacity anomalies compared with neighboring cycles? | Directly tests electrochemical context around synchronized optical events. |
| E5_roi_event_visual_qc_manifest | ready | 1 | 39.4 | Do raw ROI frames around event cycles show real particle changes rather than threshold/drift artifacts? | Required before claiming physical abrupt degradation from optical traces. |
| E3_particle_event_hazard_calibration | ready | 1 | 36.51 | Can grouped hazard models forecast abrupt particle events under leave-one-particle and blocked-cycle validation? | Turns weak precursor signal into a calibrated degradation-risk model. |
| E4_degradation_mode_clustering | ready | 1 | 33.4 | Do particles/cycles cluster into gradual drift, abrupt drop, recovery, and stable modes? | Extracts interpretable degradation modes rather than binary event labels only. |
| E6_observable_sequence_forecast | ready | 1 | 29.9 | Can past particle optical observables plus future protocol/echem context predict future degradation observables? | Bridges the NMC degradation data and the SP observable-forecasting strategy. |
| E7_front_proxy_readiness_scan | ready | 1 | 29.9 | Which particles/cycles are suitable for phase-front/proxy tracking and apparent transport fits? | Prevents over-interpreting diffusion coefficients before front-quality validation. |

## Interpretation

The highest-ranked ready experiments should be executed first because they test artifact controls and protocol-local physics before adding heavier sequence models.
