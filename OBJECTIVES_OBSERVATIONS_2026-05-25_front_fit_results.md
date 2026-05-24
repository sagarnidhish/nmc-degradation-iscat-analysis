# Objectives And Observations (Addendum)

Date: 2026-05-25

## Timebase-Corrected Front Fits

Two broader front-fit sweeps completed on Isambard:

- `source_balanced_timebase_corrected_front_fit_v1`
- `timebase_corrected_front_fit_transfer_ranked_v1`

## Verified Metrics

### Source-balanced ROI set

- Rows: `128`
- Usable fits: `128/128`
- `timing_dt_max_to_median_ratio_p90`: `1.1110`
- `thr50_radius2_slope_r2` median: `0.0561`
- `thr30_apparent_D_um2_per_s` median: `6.23e-09`
- `thr50_apparent_D_um2_per_s` median: `-3.84e-07`
- `thr70_apparent_D_um2_per_s` median: `-1.05e-06`

### Transfer-ranked ROI set

- Rows: `48`
- Usable fits: `48/48`
- `timing_dt_max_to_median_ratio_p90`: `1.0835`
- `thr50_radius2_slope_r2` median: `0.0813`
- `thr30_apparent_D_um2_per_s` median: `4.45e-07`
- `thr50_apparent_D_um2_per_s` median: `-2.11e-07`
- `thr70_apparent_D_um2_per_s` median: `1.63e-06`

## Observation

The front-radius proxy is still too unstable for a calibrated diffusion claim:

- fit quality is low at the median,
- apparent-D signs flip across thresholds,
- the timebase is acceptable enough to fit, but not enough to make the linear-radius proxy physically trustworthy on its own.

This is still useful as a ranking and falsification tool. It says where the current front-frontier model is weak and where a segmented or threshold-robust model might recover better behavior.

## Next Step

Use these front-fit tables as a conditioning signal in the rollout/phase bridge and in the eventual multi-task physics-aware model:

- next-frame prediction,
- rollout residual energy,
- phase-fraction slopes,
- front-radius slopes,
- future event risk.

