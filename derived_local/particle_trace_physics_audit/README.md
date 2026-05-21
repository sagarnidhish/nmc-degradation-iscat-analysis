# Particle Trace Physics Audit

Cycle-level audit over the larger normalized four-particle intensity table.

- Cycle rows: 89
- Cycle range: 2-158
- Any-drop cycles: 4
- Synchronized 2+ drop cycles: 2
- Chosen trace-state k: 2 (silhouette=0.266)

## Top Event Feature Tests
- any_abrupt_drop mean_delta_prev: median pos-neg=-0.116, p=1.03e-05
- any_abrupt_drop max_abs_delta_prev: median pos-neg=0.162, p=0.00133
- any_abrupt_drop mean_abs_delta_prev: median pos-neg=0.0894, p=0.00275
- any_abrupt_drop particle_norm_mean: median pos-neg=-0.0963, p=0.0586
- any_abrupt_drop delta_std_across_particles: median pos-neg=0.049, p=0.0668
- any_abrupt_drop frames_percentile: median pos-neg=-0.298, p=0.255
- any_abrupt_drop n_frames: median pos-neg=-24.5, p=0.255
- any_abrupt_drop V_min: median pos-neg=-0.00521, p=0.434

## Future Drop Classifier
- future_any_drop_within_8cycles: folds=5, mean AUC=0.883, balanced accuracy=0.648
- future_sync2_drop_within_8cycles: folds=3, mean AUC=0.827, balanced accuracy=0.828
- null future_any_drop_within_8cycles: observed AUC=0.883, null p95=0.679, empirical p=0.0020
- null future_sync2_drop_within_8cycles: observed AUC=0.827, null p95=0.744, empirical p=0.0100

## Guardrail

This audit uses the larger four-particle cycle intensity table, not video ROI masks. It tests cycle-level photometry/echem physics hypotheses and early-warning signals, but cannot localize phase fronts or validate diffusion without ROI/video QC.
