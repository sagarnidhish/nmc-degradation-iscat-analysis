# Acquisition-Residualized Video Physics Benchmark

Leave-one-cycle benchmark for future weak-label signal after acquisition/context conditioning.

- Rows: 172
- Cycles: 34
- Targets: future_any_drop_within_8cycles, future_any_drop_within_16cycles
- Feature groups: {'acquisition_context': 27, 'echem_context': 107, 'residual_dictionary_raw': 39, 'handcrafted_particle_raw': 89, 'video_pca_raw': 16, 'all_video_raw': 144, 'context_plus_residual_dictionary': 66, 'context_plus_handcrafted_particle': 116, 'context_plus_video_pca': 43, 'context_plus_all_video': 171, 'echem_context_plus_all_video': 251}

Guardrail: This is a weak-label, leave-one-cycle benchmark over automatically selected ROI embeddings. A strong acquisition-context score is treated as design/context structure, not a deployable warning model. Residualized video scores test whether particle-region video descriptors add signal after context conditioning.
