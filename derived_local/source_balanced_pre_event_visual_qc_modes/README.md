# Source-Balanced Pre-Event Visual QC Modes

Automatic visual/front plausibility audit for the consensus review packet.
This is a prioritization layer for manual QC, not a manual label set.

- Candidates scored: 24
- Priority-tier counts: {'routine_or_low_front_plausibility': 16, 'front_plausible_followup': 8}
- Mode counts: {'low front-plausibility context': 23, 'moderate front-like followup': 1}
- Top candidate: source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723 (front_plausible_followup, score 0.5899088357604971)

## Outputs

- `source_balanced_pre_event_visual_qc_modes.csv`
- `source_balanced_pre_event_visual_qc_mode_summary.csv`
- `visual_qc_priority_contact_sheet.png`
- `source_balanced_pre_event_visual_qc_modes_summary.json`

Guardrail: Automatic visual/front plausibility scores and clusters are review priorities only. They do not validate particle identity, front masks, manual QC, calibrated diffusion, phase-boundary motion, or causality.
