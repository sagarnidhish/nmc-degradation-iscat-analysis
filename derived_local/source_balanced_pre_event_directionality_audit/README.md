# Source-Balanced Pre-Event Directionality Audit

- Rows/cycles/sources: 128 / 64 / 14
- Event-bin counts: {'post_event_1_16': 40, 'near_pre_event_1_8': 32, 'mid_pre_event_9_16': 22, 'far_pre_event_17_32': 22, 'no_near_event_control': 12}
- Features tested: 57
- Within-source permutations: 0

## Guardrail

Temporal directionality uses automatic pre-event ROI crops and weak event-relative bins. AUC and event-clock rows are descriptive readouts; optional within-source clock permutations can be enabled with --n-perm. This tests whether optical/front/rollout proxies are ordered around event time; it does not validate causal precursors, particle identity, calibrated phase boundaries, or diffusion coefficients.
