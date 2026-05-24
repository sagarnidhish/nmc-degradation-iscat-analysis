# Objectives And Observations (Addendum)

Date: 2026-05-25

## New Experiment

Context-length sweep for the ROI-only next-frame baseline on the source-balanced cohort.

Why this matters:

- If short history performs nearly as well as longer context, the video dynamics may be close to a low-order Markov process under the current crop/normalization.
- If longer context helps materially, that suggests the particle-region sequence carries temporal memory worth modeling with a stronger rollout architecture.

Planned outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/next_frame_baseline_context_sweep_source_balanced_v1/context_2/`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/next_frame_baseline_context_sweep_source_balanced_v1/context_4/`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/next_frame_baseline_context_sweep_source_balanced_v1/context_8/`

Guardrail:

Pixel-space next-frame prediction is still a control model, not a physics claim. The experiment is useful because it brackets how much signal remains after particle-only cropping and drift-tolerant preprocessing.

