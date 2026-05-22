# Residualized Future8 Video-Physics Benchmark

This benchmark tests whether short-horizon future8 weak labels carry source-robust video/optical-physics signal after acquisition residualization and echem-context comparison.

- Rows/cycles/sources: 172 / 34 / 12
- Future8 optical status: `not_supported_after_controls`
- Fused video+echem incremental status: `not_incremental_over_echem_context`

Outputs:
- `residualized_future8_video_physics_metrics.csv`
- `residualized_future8_video_physics_predictions.csv`
- `residualized_future8_video_physics_deltas.csv`
- `residualized_future8_video_physics_summary.json`

Guardrail: Future8 labels are treated as weak warning labels. This benchmark supports a video-physics claim only if optical features survive source/source-cohort holdout, acquisition residualization, cycle balancing, source-stratified permutation, and echem-context comparison.
