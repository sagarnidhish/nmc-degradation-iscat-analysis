# Source-Balanced Pre-Event Optical-Flow Transport Audit

- Input rows / OK / failed: 128 / 128 / 0
- Flow method: farneback
- Event bins: {'near_pre_event_1_8': 32, 'mid_pre_event_9_16': 22, 'far_pre_event_17_32': 22, 'post_event_1_16': 40, 'no_near_event_control': 12}
- Top event test: {'target': 'near_vs_any_non_near', 'feature': 'abs_radial_flow_mean', 'transform': 'raw', 'n': 128, 'n_positive': 32, 'direction': 'higher_in_positive', 'oriented_auc': 0.755859375, 'average_precision': 0.5958296393840454, 'median_positive_minus_negative': 3.38066806299152e-07, 'mwu_p': 1.542964977888066e-05, 'spearman_rho': 0.38380077534887275, 'spearman_p': 7.745133696932763e-06}

Outputs:
- `source_balanced_pre_event_optical_flow_transport_per_roi.csv`
- `source_balanced_pre_event_optical_flow_transport_frame_samples.csv`
- `source_balanced_pre_event_optical_flow_transport_event_tests.csv`
- `source_balanced_pre_event_optical_flow_transport_method_summary.csv`
- `source_balanced_pre_event_optical_flow_transport_source_summary.csv`
- `source_balanced_pre_event_optical_flow_transport_failures.csv`

Guardrail:
Apparent optical-flow transport is computed inside history-derived automatic particle masks on normalized ROI crops. It is an image-motion proxy for hypothesis ranking, not calibrated particle velocity, phase-boundary velocity, material flux, or diffusion.
