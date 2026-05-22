# Echem-Conditioned Residual Dictionary Fusion Audit

Fuses label-free residual-dictionary video descriptors with echem regime and acquisition/context features under leave-one-cycle weak-label evaluation.

- Rows: 172
- Cycles: 34
- Feature sets: {'acquisition_context': 11, 'echem_regime': 75, 'residual_dictionary': 39, 'handcrafted_scalar': 62, 'pca_video': 16, 'residual_dictionary_plus_echem': 125, 'residual_dictionary_plus_handcrafted': 101, 'residual_dictionary_handcrafted_echem': 187, 'pca_video_plus_echem': 102, 'handcrafted_plus_echem': 148}

Guardrail: Fusion uses weak cycle labels, automatic ROI crops, and echem/acquisition covariates. It tests representation conditioning and review prioritization, not deployable warning, manual particle/front labels, causal echem mechanism, or calibrated diffusion.
