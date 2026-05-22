# Source-Balanced Pre-Event Readout Audit

- Rows/cycles/sources: 128 / 64 / 14
- Event-bin counts: {'post_event_1_16': 40, 'near_pre_event_1_8': 32, 'mid_pre_event_9_16': 22, 'far_pre_event_17_32': 22, 'no_near_event_control': 12}
- Features tested: 44

## Guardrail

Event-relative readouts use automatic ROI crops and weak event-distance bins. They test pre/post/control organization of optical/front/rollout proxies, not manual particle identity, causal mechanism, calibrated phase boundaries, or diffusion coefficients.
