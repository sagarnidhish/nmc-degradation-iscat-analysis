# Source-Balanced Pre-Event Blinded Manual-QC Workbook

This packet randomizes the top rendered manual-QC candidates and separates reviewer-visible image paths from the hidden event/source/action key.

## Counts

- Candidate rows: 40
- Sources in hidden key: 12
- Event bins in hidden key: {'near_pre_event_1_8': 23, 'post_event_1_16': 6, 'mid_pre_event_9_16': 4, 'far_pre_event_17_32': 4, 'no_near_event_control': 3}
- Action tiers in hidden key: {'routine_or_low_concordance': 11, 'review_front_only_guardrail': 11, 'review_kinetic_only_guardrail': 8, 'review_front_and_kinetics_first': 5, 'context_control_concordant': 4, 'review_strict_front_gate': 1}

## Files

- `source_balanced_pre_event_manual_qc_blinded_workbook.csv`: reviewer-facing label sheet.
- `source_balanced_pre_event_manual_qc_blinded_key.csv`: hidden key with event/source/action metadata.
- `source_balanced_pre_event_manual_qc_blinded_review.html`: image review page using relative paths.
- `source_balanced_pre_event_manual_qc_rubric.json`: explicit label schema.
- `source_balanced_pre_event_manual_qc_blind_summary.json`: machine-readable summary.

## Guardrail

This workbook randomizes and blinds manual review but does not assign labels. Do not open/use the hidden key until reviewer decisions are frozen; diffusion claims remain blocked unless manual labels support particle identity, front mask quality, and front-motion interpretability.
