# Source-Balanced Pre-Event Same-Source Ladder Audit

- Rows/cycles/sources: 128 / 64 / 14
- Ladder source counts: {'sources_with_near_rows': 5, 'sources_with_near_mid_ladder': 3, 'sources_with_near_far_ladder': 0, 'sources_with_near_post_ladder': 2}
- Pair counts: {'near_vs_any_non_near_same_source': 32, 'near_vs_mid_pre_same_source': 20, 'near_vs_post_control_same_source': 12}

## Top Same-Source Paired Physics Tests
- near_vs_any_non_near_same_source mask_centroid_path_px: n=32, median near-control diff=0.784, sign-flip p=0.001998
- near_vs_any_non_near_same_source apparent_diffusion_q70_um2_per_norm_time: n=16, median near-control diff=0.6656, sign-flip p=0.003996
- near_vs_any_non_near_same_source masked_minus_background_mean_slope: n=32, median near-control diff=0.001643, sign-flip p=0.004995
- near_vs_any_non_near_same_source front_radius_q60_slope_px_per_norm_time: n=15, median near-control diff=1.769, sign-flip p=0.005994
- near_vs_any_non_near_same_source front_radius_q70_slope_px_per_norm_time: n=16, median near-control diff=1.036, sign-flip p=0.007992
- near_vs_any_non_near_same_source apparent_diffusion_q70_px2_per_norm_time: n=16, median near-control diff=18.06, sign-flip p=0.008991
- near_vs_any_non_near_same_source mask_area_fraction_slope: n=32, median near-control diff=-0.006282, sign-flip p=0.01499
- near_vs_any_non_near_same_source front_radius_q80_slope_px_per_norm_time: n=12, median near-control diff=0.5889, sign-flip p=0.3477
- near_vs_mid_pre_same_source apparent_diffusion_q70_px2_per_norm_time: n=14, median near-control diff=19.63, sign-flip p=0.000999
- near_vs_mid_pre_same_source mask_centroid_path_px: n=20, median near-control diff=1.628, sign-flip p=0.000999

## Top Within-Source Clock Tests
- front_radius_q60_slope_px_per_norm_time: n=58, rho=0.1086, p=0.4171
- front_radius_q70_slope_px_per_norm_time: n=69, rho=0.09242, p=0.4501
- mask_area_fraction_slope: n=128, rho=0.08125, p=0.3619
- apparent_diffusion_q70_px2_per_norm_time: n=69, rho=0.0752, p=0.5391
- apparent_diffusion_q70_um2_per_norm_time: n=69, rho=0.0752, p=0.5391
- front_radius_q70_median_px: n=128, rho=-0.07151, p=0.4225
- front_radius_q80_median_px: n=128, rho=-0.06808, p=0.4451
- front_gradient_peak_radius_slope_px_per_norm_time: n=111, rho=0.05608, p=0.5588

## Guardrail

This audit is strictly within-source but opportunistic: sources have incomplete event-relative ladders, pairs reuse controls, and all particle masks/fronts are automatic. It tests whether observed near-pre signals survive same-source local controls; it does not validate particle identity, calibrated diffusion, phase-boundary motion, causality, or deployable warnings.
