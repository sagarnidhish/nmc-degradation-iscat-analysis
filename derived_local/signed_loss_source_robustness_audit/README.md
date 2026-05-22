# Signed Loss Source Robustness Audit

Tests signed optical-loss axes under within-source rank/centering, source-mean-only, source-balanced bootstrap, source-macro AUC, and label-permutation controls.

- Rows: 172
- Future16 labeled rows: 72
- Cycles: 34
- Sources: 12

Guardrail: Source robustness transforms distinguish within-source signal from source-level composition. High raw AUC with high source-mean-only AUC or weak within-source permutation evidence should be treated as source/context-sensitive review evidence, not a source-independent detector.
