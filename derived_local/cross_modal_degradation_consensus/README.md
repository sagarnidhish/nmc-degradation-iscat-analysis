# Cross-Modal Degradation Consensus

This audit ranks cycles by agreement across particle traces, acquisition context, integrated event evidence, ROI fronts, masked/rollout residuals, and electrochemical state features.

- Cycles scored: 89
- Cycles with at least one modal vote: 53
- Median consensus score: 0.499
- Modal vote threshold: 0.85

Guardrail: the score is not a calibrated degradation probability. Frame-count/acquisition receives its own vote so cycles 86/116 can remain high-confidence optical events while still flagged as acquisition-confounded.

Top cycles are in `cross_modal_consensus_top_cycles.csv`; full joined table is in `cross_modal_consensus_cycle_table.csv`.
