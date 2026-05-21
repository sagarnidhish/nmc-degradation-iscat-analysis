# Closed-Loop Analysis Update

## Current Observations

| topic | strength | observation | evidence |
| --- | --- | --- | --- |
| event_synchrony | moderate | Synchronized optical drop events have already passed a permutation sanity check. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_synchrony/event_synchrony_summary.json |
| event_echem | moderate | 81 cycles matched to electrochemistry; coarse cycle summaries do not fully explain events. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_echem_coupling/event_echem_coupling_summary.json |
| protocol_context | moderate | Synchronized event cycles are linked to unusually short frame-count regimes. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_protocol_context/event_protocol_context_summary.json |
| event_forecasting | weak_to_moderate | Best transparent leave-one-particle event-forecast F1 is 0.435. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets/particle_event_feature_baselines.csv |

## Next Action Queue

| priority | action | expected_output |
| --- | --- | --- |
| 1 | Run frame-count-matched shuffled controls for synchronized event cycles. | matched_control_p_values.csv and artifact-risk summary |
| 2 | Run protocol-local echem window scan around cycles 86 and 116. | local_echem_window_features.csv with neighbor-cycle controls |
| 3 | Generate ROI/event visual QC manifest for raw frame inspection. | event_roi_qc_manifest.csv and review checklist |
| 4 | Fit grouped degradation-mode clustering and hazard calibration. | degradation_modes.csv, hazard_calibration.csv |
