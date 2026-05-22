# Source-Balanced Mask/Front Sanity Audit

ROI sequences: 96 across 48 cycles and 14 sources.
Future8/future16 positive ROI sequences: 28 / 48.

This packet estimates automatic crop-local masks, centroid stability, radial front radii at q60/q70/q80, and an approximate q70 radius-squared diffusion proxy from source-balanced ROI tensors.

## Top ROI Feature Tests
- future_any_drop_within_16cycles masked_minus_background_mean_slope: AUC=0.690, AP=0.696, direction=higher_in_positive, eta2=0.634
- future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC=0.626, AP=0.589, direction=lower_in_positive, eta2=0.365
- future_any_drop_within_16cycles front_radius_q80_slope_px_per_norm_time: AUC=0.612, AP=0.616, direction=lower_in_positive, eta2=0.104
- future_any_drop_within_16cycles front_radius_q60_slope_px_per_norm_time: AUC=0.611, AP=0.649, direction=higher_in_positive, eta2=0.370
- future_any_drop_within_16cycles mask_area_fraction_median: AUC=0.607, AP=0.606, direction=higher_in_positive, eta2=0.317
- future_any_drop_within_16cycles mask_base_area_fraction: AUC=0.595, AP=0.610, direction=higher_in_positive, eta2=0.243
- future_any_drop_within_16cycles front_radius_q60_median_px: AUC=0.578, AP=0.535, direction=lower_in_positive, eta2=0.156
- future_any_drop_within_16cycles front_radius_q80_median_px: AUC=0.576, AP=0.538, direction=lower_in_positive, eta2=0.153
- future_any_drop_within_16cycles front_gradient_peak_radius_median_px: AUC=0.564, AP=0.550, direction=higher_in_positive, eta2=0.215
- future_any_drop_within_16cycles mask_centroid_max_step_px: AUC=0.560, AP=0.618, direction=higher_in_positive, eta2=0.695

## Guardrail
Automatic crop-local masks/front radii are source-balanced sanity proxies from resized ROI tensors; they are not manual particle masks, not calibrated fronts, and apparent diffusion uses an approximate 0.192 um/output-pixel scale.
