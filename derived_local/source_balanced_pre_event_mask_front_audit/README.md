# Source-Balanced Mask/Front Sanity Audit

ROI sequences: 128 across 64 cycles and 14 sources.
Future8/future16 positive ROI sequences: 32 / 66.

This packet estimates automatic crop-local masks, centroid stability, radial front radii at q60/q70/q80, and an approximate q70 radius-squared diffusion proxy from source-balanced ROI tensors.

## Top ROI Feature Tests
- future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC=0.628, AP=0.620, direction=lower_in_positive, eta2=0.414
- future_any_drop_within_16cycles masked_minus_background_mean_slope: AUC=0.624, AP=0.655, direction=higher_in_positive, eta2=0.578
- future_any_drop_within_16cycles front_radius_q80_slope_px_per_norm_time: AUC=0.617, AP=0.644, direction=lower_in_positive, eta2=0.153
- future_any_drop_within_16cycles mask_centroid_max_step_px: AUC=0.593, AP=0.669, direction=higher_in_positive, eta2=0.755
- future_any_drop_within_16cycles mask_area_fraction_slope: AUC=0.561, AP=0.567, direction=higher_in_positive, eta2=0.371
- future_any_drop_within_16cycles front_radius_q60_slope_px_per_norm_time: AUC=0.555, AP=0.628, direction=higher_in_positive, eta2=0.281
- future_any_drop_within_16cycles object_mean_abs_z: AUC=0.554, AP=0.529, direction=lower_in_positive, eta2=0.241
- future_any_drop_within_16cycles front_radius_q70_positive_step_fraction: AUC=0.549, AP=0.587, direction=higher_in_positive, eta2=0.364
- future_any_drop_within_16cycles mask_area_fraction_iqr: AUC=0.548, AP=0.584, direction=higher_in_positive, eta2=0.384
- future_any_drop_within_16cycles mask_base_area_fraction: AUC=0.538, AP=0.569, direction=higher_in_positive, eta2=0.216

## Guardrail
Automatic crop-local masks/front radii are source-balanced sanity proxies from resized ROI tensors; they are not manual particle masks, not calibrated fronts, and apparent diffusion uses an approximate 0.192 um/output-pixel scale.
