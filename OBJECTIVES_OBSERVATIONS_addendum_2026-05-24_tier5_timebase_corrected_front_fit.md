# Objectives And Observations (Addendum)

Date: 2026-05-24

Topic: Tier5 timebase-corrected front fit (NMC)

## Problem

The diffusion/transport proxy work is (correctly) blocked by:

1. Per-source / per-ROI timebase instability (irregular `camera_timing`).
2. Fragile radius^2-vs-time fits that degrade when assuming uniform dt or using coarse sampled timing.

This prevents any credible claim about calibrated diffusion, and also makes it hard to rank ROIs/cycles by transport-like signal without entanglement with acquisition artifacts.

## Action Taken

Added `scripts/tier5_timebase_corrected_front_fit.py`.

What it does:

- Takes an ROI sequence manifest (e.g. `derived/selected_roi_sequences/selected_roi_sequence_manifest.csv`).
- Loads each ROI NPZ (`frames_norm`, `frame_indices`) which is already a particle-region crop.
- Reads `camera_timing` only at those frame indices from the corresponding HDF5 (`{source_stem}.hdf5`).
- Fits radius^2 front proxies vs real elapsed time for three simple thresholds (thr30/thr50/thr70).
- Reports robust bootstrap uncertainty and timing diagnostics (`dt_max_to_median_ratio`, dt quantiles).

## Planned Outputs (Isambard)

Write under:

`/scratch/<account>/<username>/Alek_Jiho/derived/timebase_corrected_front_fit_v1/`

Files:

- `timebase_corrected_front_fit_rois.csv`
- `timebase_corrected_front_fit_timing.csv`
- `timebase_corrected_front_fit_summary.json`
- `README.md`

## Guardrail

These outputs are apparent kinematics for ranking/diagnosis and follow-up experiment design.

They are not a calibrated material diffusion measurement without:

- provenance for spatial calibration metadata (beyond slide-derived 96 nm/px), and
- manual QC acceptance for the underlying ROI/front traces.

