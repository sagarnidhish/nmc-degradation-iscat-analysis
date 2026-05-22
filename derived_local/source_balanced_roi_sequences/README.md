# Source-Balanced ROI Sequences

Particle-region-only crop tensors for source-balanced expansion candidates. Use `frames_norm` from each NPZ as model inputs for next-frame and rollout experiments.

Crops are fixed and padded around automatically reconstructed particle coordinates to tolerate drift-correction blur and small coordinate shifts.

- ROI sequences: 96
- Cycles: 48
- Sources: 14
- Failures: 0

Guardrail: Source-balanced sequences are fixed padded particle-region crops around automatic reconstructed candidates. They broaden source coverage for model/physics tests, but are not manual particle annotations or validated front labels.
