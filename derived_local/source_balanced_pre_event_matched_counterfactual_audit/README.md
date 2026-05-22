# Source-Balanced Pre-Event Matched Counterfactual Audit

- Rows/cycles/sources: 128 / 64 / 14
- Context match features: cycleNo, local_cycle_index, object_x_full_approx, object_y_full_approx, object_area_ds_px, object_mean_abs_z, crop_x0, crop_y0, roi_mean_first, roi_norm_mean_first, mask_base_area_fraction
- Pair counts: {'near_vs_far_pre|source_penalized_global': 32, 'near_vs_post_control|same_source': 12, 'near_vs_post_control|source_penalized_global': 32}

## Top Matched Physics Tests
- near_vs_far_pre source_penalized_global masked_minus_background_mean_median: n=32, median diff=0.003018, sign-flip p=0.000999
- near_vs_far_pre source_penalized_global masked_minus_background_mean_slope: n=32, median diff=0.002954, sign-flip p=0.004995
- near_vs_far_pre source_penalized_global mask_centroid_path_px: n=32, median diff=0.6576, sign-flip p=0.007992
- near_vs_far_pre source_penalized_global front_gradient_peak_radius_slope_px_per_norm_time: n=29, median diff=-2.702, sign-flip p=0.02697
- near_vs_far_pre source_penalized_global mask_area_fraction_slope: n=32, median diff=0.002631, sign-flip p=0.1169
- near_vs_far_pre source_penalized_global front_radius_q60_slope_px_per_norm_time: n=10, median diff=0.9884, sign-flip p=0.1239
- near_vs_far_pre source_penalized_global apparent_diffusion_q70_um2_per_norm_time: n=9, median diff=-0.2037, sign-flip p=0.3946
- near_vs_far_pre source_penalized_global front_radius_q70_median_px: n=32, median diff=0, sign-flip p=0.4026
- near_vs_post_control same_source front_gradient_peak_radius_slope_px_per_norm_time: n=6, median diff=1.126, sign-flip p=0.06294
- near_vs_post_control same_source masked_minus_background_mean_slope: n=12, median diff=0.001309, sign-flip p=0.08092

## Guardrail

Nearest-neighbor matching uses automatic context/baseline descriptors only and controls observed confounding, not unobserved acquisition or particle-identity effects. Pairs reuse controls and all ROI masks/fronts are automatic. Results prioritize manual QC and physics follow-up; they do not validate phase boundaries, calibrated diffusion coefficients, degradation causality, or deployable warnings.
