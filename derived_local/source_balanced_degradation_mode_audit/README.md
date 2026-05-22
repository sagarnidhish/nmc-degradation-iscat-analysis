# Source-Balanced Degradation Mode Audit

Rows/cycles/sources: 96 / 48 / 14
Chosen k: 4; source-mode change fraction: 0.244

## Modes
- mode 0 source_residual_front_geometry_state: n=70, sources=13, future16=0.543, past16=0.329, phases={'pre_event_8': 20, 'far_from_event': 19, 'pre_event_16': 18, 'post_event_8': 8, 'post_event_16': 5}
- mode 1 source_residual_temporal_dictionary_state: n=1, sources=1, future16=1.000, past16=0.000, phases={'pre_event_16': 1}
- mode 2 source_residual_temporal_dictionary_state: n=2, sources=1, future16=0.000, past16=1.000, phases={'post_event_16': 2}
- mode 3 source_residual_temporal_dictionary_state: n=23, sources=10, future16=0.391, past16=0.217, phases={'far_from_event': 9, 'pre_event_8': 8, 'post_event_16': 3, 'post_event_8': 2, 'pre_event_16': 1}

## Top Enrichment
- mode 2 post_event_16: fraction=1.000 vs 0.085, p=0.00987
- mode 3 pre_event_16: fraction=0.043 vs 0.260, p=0.0364
- mode 0 pre_event_16: fraction=0.257 vs 0.077, p=0.0871
- mode 2 past16: fraction=1.000 vs 0.298, p=0.0954
- mode 0 post_event_16: fraction=0.071 vs 0.192, p=0.128
- mode 1 pre_event_16: fraction=1.000 vs 0.200, p=0.208
- mode 0 future16: fraction=0.543 vs 0.385, p=0.251
- mode 3 far_from_event: fraction=0.391 vs 0.260, p=0.293

## Guardrail
Modes are unsupervised source-residual clusters of automatic particle-region optical/residual proxies. They organize degradation-state hypotheses and review candidates, but they do not prove calibrated phase boundaries, diffusion coefficients, or causal event mechanisms.
