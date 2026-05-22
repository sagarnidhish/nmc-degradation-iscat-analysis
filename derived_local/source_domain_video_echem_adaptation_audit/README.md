# Source-Domain Video/Echem Adaptation Audit

Tests whether source centering or CORAL-style alignment rescues leave-source future16 video/echem transfer.

- Rows: 172
- Cycles: 34
- Sources: 12
- Feature sets: {'acquisition_context': 40, 'video_all': 64, 'echem_regime': 82, 'video_plus_echem': 146}

Guardrail: This is an unlabeled target-source adaptation audit over weak labels and automatic ROI descriptors. Source centering/CORAL use held-out source feature distributions but not held-out labels. Results diagnose domain shift; they are not deployable warnings or causal physics proof.
