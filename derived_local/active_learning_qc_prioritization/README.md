# Active-Learning QC Prioritization

This audit ranks ROI candidates for manual review. It does not assign degradation labels or validate diffusion/front claims.

- Candidate rows: 97
- Immediate manual-QC rows: 4
- Rows with visual assets: 47
- Guardrail: Active-learning QC ranks pending ROI review candidates from automatic and weak-label evidence only; manual labels, diffusion claims, and deployment decisions remain withheld until human QC.

Primary outputs:

- `active_learning_qc_priority_table.csv`
- `active_learning_qc_cycle_summary.csv`
- `active_learning_qc_reason_counts.csv`
- `active_learning_qc_summary.json`
