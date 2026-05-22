# Source-Balanced Pre-Event Phase Kinetics Audit

Particle-region-only optical phase-fraction kinetics for the source-balanced pre-event ROI cohort.

- Input ROI rows: 128
- Loaded ROI tensors: 128
- Sources: 14
- Kinetic features tested: 56
- Event-bin counts: {'post_event_1_16': 40, 'near_pre_event_1_8': 32, 'mid_pre_event_9_16': 22, 'far_pre_event_17_32': 22, 'no_near_event_control': 12}

## Top Event Tests
- near_vs_any_non_near raw masked_minus_bg_slope: AUC=0.816, median diff=0.00376, p=8.993e-08
- near_vs_post_control raw masked_minus_bg_slope: AUC=0.799, median diff=0.00386, p=4.802e-06
- near_vs_mid_pre raw masked_minus_bg_slope: AUC=0.849, median diff=0.00419, p=1.547e-05
- near_vs_far_pre raw masked_minus_bg_slope: AUC=0.825, median diff=0.00325, p=5.755e-05
- near_vs_any_non_near raw q75_phase_fraction_slope: AUC=0.734, median diff=0.00956, p=7.693e-05
- near_vs_any_non_near raw q75_avrami_n: AUC=0.723, median diff=0.628, p=0.0001691
- near_vs_any_non_near raw q65_phase_fraction_slope: AUC=0.710, median diff=0.0135, p=0.0003903
- near_vs_any_non_near raw q75_logistic_amp: AUC=0.709, median diff=0.0831, p=0.0004258

## Top Matched Tests
- global_echem_context near-vs-far_pre_event_17_32 masked_mean_total_variation: n=32, median diff=0.00843, p=0.0004998
- global_echem_context near-vs-far_pre_event_17_32 masked_minus_bg_slope: n=32, median diff=0.00747, p=0.0004998
- global_echem_context near-vs-far_pre_event_17_32 q55_phase_fraction_delta: n=32, median diff=0.0215, p=0.0004998
- global_echem_context near-vs-far_pre_event_17_32 q55_phase_fraction_slope: n=32, median diff=0.0523, p=0.0004998
- global_echem_context near-vs-far_pre_event_17_32 q65_phase_fraction_delta: n=32, median diff=0.0163, p=0.0004998
- global_echem_context near-vs-far_pre_event_17_32 q65_phase_fraction_slope: n=32, median diff=0.0492, p=0.0004998
- global_echem_context near-vs-far_pre_event_17_32 q65_phase_fraction_slope_r2: n=32, median diff=-0.0448, p=0.0004998
- global_echem_context near-vs-far_pre_event_17_32 q65_avrami_r2: n=32, median diff=0.044, p=0.0004998

## Guardrail

Source-balanced pre-event phase kinetics are automatic optical phase-fraction proxies inside stable particle masks with prior-mask fallback. Logistic/Avrami values are descriptive timing summaries, not calibrated reaction constants, manual phase labels, validated particle identities, or diffusion coefficients.
