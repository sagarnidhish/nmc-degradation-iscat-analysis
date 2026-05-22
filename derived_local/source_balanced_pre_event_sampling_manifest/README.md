# Source-Balanced Pre-Event Sampling Manifest

- Ranked cycles: 89
- Event cycles: [60.0, 86.0, 116.0, 156.0]
- Selected cycles: 64 (13 new vs existing video cohorts)
- Sources selected: 14
- Sampled cycles: 64
- Reconstructed candidates: 3840
- ROI rows: 128

## Cycle Bins

{
  "far_pre_event_17_32": 11,
  "mid_pre_event_9_16": 11,
  "near_pre_event_1_8": 16,
  "no_near_event_control": 6,
  "post_event_1_16": 20
}

## Guardrail

Pre-event sampling candidates are automatic particle-like proposals from sampled HDF5 frames. They broaden event-relative/source-balanced coverage for follow-up ROI video export and future-specific modeling, but do not validate particle identity, phase fronts, diffusion, or causal degradation mechanisms.
