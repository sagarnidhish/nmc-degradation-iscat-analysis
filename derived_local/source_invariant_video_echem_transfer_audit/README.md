# Source-Invariant Video/Echem Transfer Audit

Tests whether removing train-source mean directions or source-confounded features rescues leave-source weak future-drop prediction.

- Rows: 172
- Cycles: 34
- Sources: 12
- Feature sets: {'acquisition_context': 28, 'echem_regime': 55, 'video_all': 64, 'video_plus_echem': 119}
- Methods: ['raw', 'source_mean_resid_1', 'source_mean_resid_2', 'source_mean_resid_4', 'source_mean_resid_8', 'source_confound_filter_0.10', 'source_confound_filter_0.25', 'source_confound_filter_0.50']

Guardrail: Source-invariant projections and source-confound filters are trained without held-out source labels, but labels remain source-composition imbalanced. These results test robustness for review prioritization only; they do not validate source-transferable warning, causal degradation mechanisms, manual QC labels, or calibrated diffusion.
