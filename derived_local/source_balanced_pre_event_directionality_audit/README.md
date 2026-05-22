# Source-Balanced Pre-Event Directionality Audit

- Rows/cycles/sources: 128 / 64 / 14
- Event-bin counts: {'far_pre_event_17_32': 22, 'mid_pre_event_9_16': 22, 'near_pre_event_1_8': 32, 'no_near_event_control': 12, 'post_event_1_16': 40}
- Features tested: 57
- Within-source permutations: 250

## Guardrail

Temporal directionality uses automatic pre-event ROI crops, weak event-relative bins, and within-source clock permutations. AUC rows remain descriptive readouts. This tests whether optical/front/rollout proxies are ordered around event time; it does not validate causal precursors, particle identity, calibrated phase boundaries, or diffusion coefficients.
