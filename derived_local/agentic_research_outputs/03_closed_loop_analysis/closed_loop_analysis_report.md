# Closed-Loop Analysis Update

## Current Observations

| topic | strength | observation | evidence |
| --- | --- | --- | --- |
| event_synchrony | moderate | Synchronized optical drop events have already passed a permutation sanity check. | /scratch/<account>/<username>/Alek_Jiho/derived/event_synchrony/event_synchrony_summary.json |
| event_echem | moderate | 81 cycles matched to electrochemistry; coarse cycle summaries do not fully explain events. | /scratch/<account>/<username>/Alek_Jiho/derived/event_echem_coupling/event_echem_coupling_summary.json |
| protocol_context | moderate | Synchronized event cycles are linked to unusually short frame-count regimes. | /scratch/<account>/<username>/Alek_Jiho/derived/event_protocol_context/event_protocol_context_summary.json |
| event_candidate_fronts | exploratory | Detected 88 candidate particle/front regions; outputs are apparent front proxies, not calibrated diffusion coefficients. | /scratch/<account>/<username>/Alek_Jiho/derived/event_candidate_fronts/event_candidate_fronts_summary.json |
| validated_front_rois | exploratory_to_moderate | Selected 10 cycle-86/116 candidate ROIs for next tracking from 16 scored candidates; selections are algorithmic and still need manual QC/spatial calibration. | /scratch/<account>/<username>/Alek_Jiho/derived/validated_front_rois/validated_front_rois_summary.json |
| selected_front_roi_tracking | exploratory_to_moderate | Tracked 12 selected front ROIs at full-pixel crop scale; cycle 86 mean radius2 slope -0.127 px2/s, R2 0.178; cycle 116 mean radius2 slope -0.167 px2/s, R2 0.284. These are apparent front-motion proxies, not calibrated diffusion coefficients. | /scratch/<account>/<username>/Alek_Jiho/derived/selected_front_roi_tracking/selected_front_roi_tracking_summary.json |
| roi_rollout_baselines | moderate_baseline | Evaluated 11 particle-ROI sequences with persistence, velocity, and low-rank DMD rollouts; DMD spectral radius 1.002. | /scratch/<account>/<username>/Alek_Jiho/derived/roi_rollout_baselines/roi_rollout_baseline_summary.json |
| roi_event_conditioned_nextframe | moderate_baseline | Trained a PCA-ridge event-conditioned next-frame model on 11 selected ROI sequences; cycle 86 residual mean 0.00171; cycle 116 residual mean 0.000611. Persistence remains a strong recursive baseline, so residuals are the main degradation descriptor. | /scratch/<account>/<username>/Alek_Jiho/derived/roi_event_conditioned_nextframe/roi_event_model_summary.json |
| roi_residual_cnn_fast | negative_baseline | A leave-one-cycle residual CNN on 11 selected ROIs did not beat persistence; overall relative MSE improvement -4.23. | /scratch/<account>/<username>/Alek_Jiho/derived/roi_residual_cnn_fast/roi_residual_cnn_summary.json |
| roi_joint_physics_degradation_modes | moderate_synthesis | Combined rollout residual, front tracking, ROI physics, residual-CNN guardrails, and cycle evidence into 11 ROI joint modes; selected k=2 with silhouette 0.335. Top ROIs: cycle86_front4_obj9 score 2.86; cycle116_front3_obj9 score 1.72. | /scratch/<account>/<username>/Alek_Jiho/derived/roi_joint_physics_degradation_modes/roi_joint_physics_degradation_modes_summary.json |
| event_vs_control_roi_physics | moderate_control_check | Compared 11 event ROIs against 16 matched control ROIs; strongest shifts: high_fraction_slope_per_frame p=0.000503; roi_mean_slope_per_frame p=0.00123. Leave-pair classifier did not generalize well, so controls are a guardrail. | /scratch/<account>/<username>/Alek_Jiho/derived/event_vs_control_roi_physics/event_vs_control_roi_physics_summary.json |
| event_control_roi_comparison_expanded | moderate_control_check | Expanded matched controls to 32 ROIs; strongest event-control shifts: first_last_corr d=-1.5, p=0.0187; cumulative_abs_norm_change d=1.48, p=0.01; expanded classifier mean ROC-AUC 0.29. | /scratch/<account>/<username>/Alek_Jiho/derived/event_control_roi_comparison_expanded/event_control_roi_comparison_summary.json |
| event_forecasting | weak_to_moderate | Best transparent leave-one-particle event-forecast F1 is 0.435. | /scratch/<account>/<username>/Alek_Jiho/derived/particle_event_targets/particle_event_feature_baselines.csv |

## Next Action Queue

| priority | action | expected_output |
| --- | --- | --- |
| 1 | Run frame-count-matched shuffled controls for synchronized event cycles. | matched_control_p_values.csv and artifact-risk summary |
| 2 | Run protocol-local echem window scan around cycles 86 and 116. | local_echem_window_features.csv with neighbor-cycle controls |
| 3 | Generate ROI/event visual QC manifest for raw frame inspection. | event_roi_qc_manifest.csv and review checklist |
| 4 | Add manual QC/spatial calibration metadata for selected front ROIs and convert pixel-scale slopes to calibrated units. | front_roi_qc_calibration.csv and calibrated_front_tracking.csv |
| 5 | Add manual QC/spatial calibration for top joint-mode ROIs and expand event/control extraction to additional non-synchronized candidate cycles. | calibrated top-mode ROI report and expanded multi-cycle ROI cohort |
