# Acquisition-Residualized Video/Echem Warning Audit

Tests whether video/echem features retain future-drop signal after fold-local acquisition/context residualization.

- Rows: 172
- Cycles: 34
- Sources: 12
- Feature sets: {'acquisition_context': 40, 'echem_regime': 57, 'video_scalar': 48, 'video_embedding': 16, 'video_all': 64, 'video_plus_echem': 121}

Guardrail: This audit residualizes candidate video/echem features against acquisition/context inside each held-out fold. It tests context-resistant weak-label signal for prioritizing next analyses, not deployable warning, causal mechanism, manual QC labels, or calibrated diffusion.
