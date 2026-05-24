# Objectives And Observations (Addendum)

Date: 2026-05-25

## Source-Balanced Timebase-Corrected Front Fits

The source-balanced front-fit sweep has now completed.

## Verified Metrics

- Rows: `96`
- Usable fits: `96/96`
- `timing_dt_max_to_median_ratio_p90`: `1.1062`
- `thr50_radius2_slope_r2` median: `0.0460`
- `thr30_apparent_D_um2_per_s` median: `6.23e-09`
- `thr50_apparent_D_um2_per_s` median: `-3.49e-07`
- `thr70_apparent_D_um2_per_s` median: `-8.99e-07`

## Interpretation

This is the cleanest broad front-fit evidence yet, and it says the same thing as the earlier smaller runs:

- the particle-only timebase is stable enough to fit,
- but the radius^2 front proxy is still too weak and sign-unstable for calibrated diffusion,
- so the current best use of this result is as a ranking/falsification feature and as an auxiliary input to a later multi-task model.

## Next Step

Use the front-fit outputs as conditioning for a multi-task model that predicts:

- next-frame residuals,
- rollout residual energy,
- phase-fraction slope,
- front-radius slope,
- and future-event risk.

