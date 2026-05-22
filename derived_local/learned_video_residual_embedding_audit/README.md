# Learned Video Residual Embedding Audit

Self-supervised residual-CNN encoder over ROI videos. It predicts next-frame residuals label-free, then audits learned latent/residual descriptors under leave-one-cycle weak-label splits.

- Rows: 172
- Cycles: 34
- Training: {'n_pairs_used': 7000, 'downsample': 2, 'pair_stride': 2, 'channels': 12, 'n_epochs_run': 14, 'best_val_loss': 0.0004373242554720491, 'final_train_loss': 0.0003976869063100163}
- Feature set sizes: {'learned_latent': 96, 'learned_residual': 9, 'learned_all': 105, 'handcrafted_scalar': 55, 'pca_video': 16, 'learned_plus_handcrafted': 160}

Guardrail: The residual CNN is trained label-free on automatic ROI crops and evaluated only through weak cycle labels. Learned embeddings support representation design and review prioritization, not deployable prediction, manual particle/front labels, or calibrated diffusion.
