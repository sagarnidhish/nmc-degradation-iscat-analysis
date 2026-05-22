# Source-Balanced Pre-Event Manual-QC Decision Packet

This packet merges front/kinetic concordance, source-stratified null evidence, strict front gates, and rendered visual assets into a manual-review queue.

## Key Counts

- ROI rows: 128
- Sources: 14
- Rows with rendered visual assets: 24
- Manual front-review gate rows: 1
- Automatic diffusion-claim gate rows: 0
- Action counts: {'routine_or_low_concordance': 96, 'review_front_only_guardrail': 13, 'review_kinetic_only_guardrail': 9, 'review_front_and_kinetics_first': 5, 'context_control_concordant': 4, 'review_strict_front_gate': 1}

## Outputs

- Decision queue: `source_balanced_pre_event_manual_qc_decision_queue.csv`
- Top review queue: `source_balanced_pre_event_manual_qc_top40.csv`
- Visual asset manifest: `source_balanced_pre_event_manual_qc_visual_asset_manifest.csv`
- Source summary: `source_balanced_pre_event_manual_qc_source_summary.csv`
- Action summary: `source_balanced_pre_event_manual_qc_action_summary.csv`
- JSON summary: `source_balanced_pre_event_manual_qc_decision_summary.json`

## Guardrail

Manual-QC decision packet only prioritizes review. It carries source-null and visual-QC evidence forward, but it does not create manual labels, validate fronts, calibrate diffusion, or make source-invariant causal claims.
