# Closed-Loop Analysis Update

## Current Observations

| topic | strength | observation | evidence |
| --- | --- | --- | --- |
| event_synchrony | moderate | Synchronized optical drop events have already passed a permutation sanity check. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_synchrony/event_synchrony_summary.json |
| event_echem | moderate | 81 cycles matched to electrochemistry; coarse cycle summaries do not fully explain events. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_echem_coupling/event_echem_coupling_summary.json |
| protocol_context | moderate | Synchronized event cycles are linked to unusually short frame-count regimes. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_protocol_context/event_protocol_context_summary.json |
| event_candidate_fronts | exploratory | Detected 88 candidate particle/front regions; outputs are apparent front proxies, not calibrated diffusion coefficients. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_candidate_fronts/event_candidate_fronts_summary.json |
| validated_front_rois | exploratory_to_moderate | Selected 10 cycle-86/116 candidate ROIs for next tracking from 16 scored candidates; selections are algorithmic and still need manual QC/spatial calibration. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/validated_front_rois/validated_front_rois_summary.json |
| selected_front_roi_tracking | exploratory_to_moderate | Tracked 10 selected front ROIs at full-pixel crop scale; cycle 86 mean radius2 slope -0.121 px2/s, R2 0.184; cycle 116 mean radius2 slope -0.201 px2/s, R2 0.297. These are apparent front-motion proxies, not calibrated diffusion coefficients. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/selected_front_roi_tracking/selected_front_roi_tracking_summary.json |
| roi_rollout_baselines | moderate_baseline | Evaluated 11 particle-ROI sequences with persistence, velocity, and low-rank DMD rollouts; DMD spectral radius 1.002. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_rollout_baselines/roi_rollout_baseline_summary.json |
| event_forecasting | weak_to_moderate | Best transparent leave-one-particle event-forecast F1 is 0.435. | /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets/particle_event_feature_baselines.csv |

## Next Action Queue

| priority | action | expected_output |
| --- | --- | --- |
| 1 | Run frame-count-matched shuffled controls for synchronized event cycles. | matched_control_p_values.csv and artifact-risk summary |
| 2 | Run protocol-local echem window scan around cycles 86 and 116. | local_echem_window_features.csv with neighbor-cycle controls |
| 3 | Generate ROI/event visual QC manifest for raw frame inspection. | event_roi_qc_manifest.csv and review checklist |
| 4 | Add manual QC/spatial calibration metadata for selected front ROIs and convert pixel-scale slopes to calibrated units. | front_roi_qc_calibration.csv and calibrated_front_tracking.csv |
| 5 | Train event-conditioned ROI next-frame models against persistence/DMD baselines and test rollout residuals as degradation features. | roi_event_model_metrics.csv and rollout_residual_degradation_features.csv |
