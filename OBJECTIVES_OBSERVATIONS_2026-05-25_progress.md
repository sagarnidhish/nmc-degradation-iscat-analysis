# Objectives And Observations

Date: 2026-05-25

## Objective

Use the Alek_Jiho NMC degradation data on Isambard to build AI analyses that go beyond pixel error and extract physics-facing signals from particle-region-only charge photometry videos.

## What Has Been Added

- `scripts/tier5_timebase_corrected_front_fit.py`
- `scripts/tier5_segmented_timebase_front_fit.py`
- `scripts/tier5_next_frame_baseline.py`
- SLURM submit scripts for the above
- `OBJECTIVES_OBSERVATIONS_addendum_2026-05-24_tier5_timebase_corrected_front_fit.md`

## Verified Observations

- Timebase-corrected front fits on the synchronized event ROI set completed successfully: `11/11` rows usable.
- The per-ROI timing was stable for that set: `timing_dt_max_to_median_ratio_p90 ≈ 1.0094`.
- The threshold-50 radius^2 fit quality is still weak: median `r2 ≈ 0.092`.
- The apparent-D estimates remain small and sign-changing across thresholds, which supports the current guardrail that these are not calibrated diffusion coefficients.
- The source-balanced next-frame baseline completed on `68` train and `28` test sequences.
- That baseline reached test `MAE median ≈ 0.00710` and `MAE p90 ≈ 0.01455` on the ROI-normalized pixel scale.

## Current Queue State

- `4759844` pending: timebase-corrected front fits on transfer-ranked ROIs
- `4759855` pending: timebase-corrected front fits on source-balanced ROIs
- `4759869` pending: segmented timebase front fits on source-balanced ROIs

## Immediate Next Experiments

1. Consume the pending timebase-corrected and segmented front-fit jobs.
2. Compare segmented-window fits against the global fits to find stable positive-expansion intervals.
3. Use the next-frame baseline as the control model for future physics-aware auxiliary heads.

