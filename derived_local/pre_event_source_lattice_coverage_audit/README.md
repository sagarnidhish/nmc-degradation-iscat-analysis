# Pre-Event Source Lattice Coverage Audit

- Cycle rows/sources/HDF5-backed sources: 89 / 16 / 16
- Event-relative bins: {'post_event_1_16': 29, 'near_pre_event_1_8': 20, 'far_pre_event_17_32': 14, 'mid_pre_event_9_16': 14, 'no_near_event_control': 8, 'current_event': 4}
- Near-source counts: {'sources_with_near': 5, 'sources_with_near_mid': 3, 'sources_with_near_far': 0, 'sources_with_near_post_or_control': 2, 'near_sources_no_within_source_control': 0}

## Recommended Design

- Do not claim same-source near-vs-far evidence from current raw coverage: no source with near-pre rows also has far-pre rows.
- Use same-source near-vs-mid and near-vs-post/control as local opportunistic checks where available.
- Use cross-source far-pre controls only with explicit source/acquisition-class matching or source-residualization.
- Acquire or export additional far-pre cycles only if new raw source/movie coverage exists; it is not latent in exampleParticles for current near-pre sources.

## Guardrail

This is a coverage/design audit over the raw particle cycle/source index. It identifies feasible controls and missing same-source ladders; it does not add manual labels, validate particles/fronts, calibrate diffusion, or establish precursor causality.
