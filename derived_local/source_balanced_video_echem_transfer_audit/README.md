# Source-Balanced Video/Echem Transfer Audit

Tests whether source/class balancing and within-source feature ranks rescue leave-source future-drop prediction.

- Rows: 172
- Cycles: 34
- Sources: 12
- Feature sets: {'acquisition_context': 28, 'echem_regime': 55, 'video_all': 64, 'video_plus_echem': 119}

Guardrail: Leave-source labels are highly source-composition imbalanced. Source weighting and within-source rank normalization test domain robustness and review-prioritization only; they do not validate source-transferable warning, causal degradation mechanisms, manual QC labels, or calibrated diffusion.
