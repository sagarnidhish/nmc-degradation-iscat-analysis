# Source-Balanced Pre-Event Echem-Matched Residual Audit

- Rows/cycles/sources: 128 / 64 / 14
- Match features: 54 (15 context, 39 echem)
- Pair counts: {'near_vs_any_non_near|same_source': 32, 'near_vs_any_non_near|source_class_penalized_echem_context': 32, 'near_vs_any_non_near|source_penalized_echem_context': 32, 'near_vs_far_pre|source_class_penalized_echem_context': 32, 'near_vs_far_pre|source_penalized_echem_context': 32, 'near_vs_mid_pre|same_source': 20, 'near_vs_mid_pre|source_class_penalized_echem_context': 32, 'near_vs_mid_pre|source_penalized_echem_context': 32, 'near_vs_post_control|same_source': 12, 'near_vs_post_control|source_class_penalized_echem_context': 32, 'near_vs_post_control|source_penalized_echem_context': 32}

## Top Residual Matched Tests
- near_vs_mid_pre same_source apparent_diffusion_q70_um2_per_norm_time: n=14, median diff=0.9303, sign-flip p=0.000999
- near_vs_mid_pre source_class_penalized_echem_context radial_profile_last_minus_first_l1: n=32, median diff=-0.00273, sign-flip p=0.001998
- near_vs_mid_pre same_source front_radius_q70_slope_px_per_norm_time: n=14, median diff=1.427, sign-flip p=0.002997
- near_vs_any_non_near same_source apparent_diffusion_q70_um2_per_norm_time: n=16, median diff=0.6654, sign-flip p=0.002997
- near_vs_mid_pre source_penalized_echem_context radial_profile_last_minus_first_l1: n=32, median diff=-0.002833, sign-flip p=0.003996
- near_vs_mid_pre same_source radial_profile_last_minus_first_l1: n=20, median diff=-0.002534, sign-flip p=0.004995
- near_vs_mid_pre source_penalized_echem_context front_radius_q70_slope_px_per_norm_time: n=14, median diff=0.7439, sign-flip p=0.005994
- near_vs_post_control source_penalized_echem_context front_radius_q70_slope_px_per_norm_time: n=8, median diff=-1.14, sign-flip p=0.006993
- near_vs_mid_pre source_penalized_echem_context apparent_diffusion_q70_um2_per_norm_time: n=14, median diff=0.6309, sign-flip p=0.006993
- near_vs_post_control same_source radial_profile_last_minus_first_l1: n=12, median diff=0.005251, sign-flip p=0.007992

## Guardrail

This audit pairs automatic ROI rows on observed source, acquisition, baseline, and cycle-level echem descriptors, then tests automatic source+echem residual front/kymograph proxies. Matching reuses controls and cannot resolve unobserved acquisition effects, particle identity, manual front validity, calibrated diffusion, phase-boundary mechanism, or degradation causality.
