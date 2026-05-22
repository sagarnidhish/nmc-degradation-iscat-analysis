# Residual Dictionary Embedding Audit

Learns a PCA dictionary on next-frame residual fields from ROI videos and audits residual-coefficient descriptors under leave-one-cycle weak-label splits.

- Rows: 172
- Cycles: 34
- Dictionary: {'rank': 8, 'downsample': 4, 'stride': 2, 'n_residual_samples': 800, 'explained_variance_ratio_sum': 0.5991450548171997}
- Feature sets: {'residual_dictionary': 39, 'handcrafted_scalar': 55, 'pca_video': 16, 'residual_dictionary_plus_handcrafted': 94}

Guardrail: The residual dictionary is label-free and uses automatic ROI crops. It is a fast temporal-residual representation audit for model design and review prioritization, not a deployable detector, manual front label, or calibrated diffusion measurement.
