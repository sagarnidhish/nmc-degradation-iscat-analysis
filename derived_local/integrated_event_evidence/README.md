# Integrated NMC Event Evidence

Combines synchrony, electrochemistry, protocol context, persistence/recovery QC, and sampled full-frame proxy QC into one degradation-mode evidence table.

## Mode Counts

| Mode | Count |
|---|---:|
| `synchronized_persistent_drop_low_frame_count` | 2 |
| `single_persistent_drop_needs_roi_qc` | 2 |

## Ranked Event Cycles

| rank | cycleNo | mode | score | particles | mean_drop_frac | sustained | frame_pct | V_mean | I_mean_mA | next_action |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 116.0 | `synchronized_persistent_drop_low_frame_count` | 12.930 | `particle0,particle1,particle2` | 0.192 | 3/3 | 0.011 | 3.904 | 0.007041 | manual ROI preview validation; compare with exact object detector coordinates; test if low frame count is protocol or acquisition artifact |
| 2 | 86.0 | `synchronized_persistent_drop_low_frame_count` | 12.399 | `particle0,particle1,particle2` | 0.142 | 3/3 | 0.079 | 3.822 | 0.003486 | manual ROI preview validation; compare with exact object detector coordinates; test if low frame count is protocol or acquisition artifact |
| 3 | 60.0 | `single_persistent_drop_needs_roi_qc` | 5.377 | `particle3` | 0.171 | 1/1 | 0.371 | 3.83 | 0.02164 | prioritize particle-level crop validation before mechanistic interpretation |
| 4 | 156.0 | `single_persistent_drop_needs_roi_qc` | 4.817 | `particle1` | 0.166 | 1/1 | 0.910 |  |  | prioritize particle-level crop validation before mechanistic interpretation |

## Guardrails

- Chopped cycle HDF5 files referenced by exampleParticles.csv are not present on Isambard.
- Full-frame proxy masks are not final particle/object masks; fallback thresholding was used in sampled segments.
- Frame-count associations are hypothesis-generating and may reflect protocol/acquisition coupling.
- No diffusion coefficient should be interpreted from these NMC event tables without spatial calibration and validated particle-boundary tracking.
