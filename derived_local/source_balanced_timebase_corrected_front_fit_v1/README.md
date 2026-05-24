# Timebase-Corrected Front Fit

Re-fit ROI front/radius^2 slopes using per-frame `camera_timing` values for the exact ROI frame indices.

- Rows in manifest: 128
- Rows with successful timebase-corrected fits: 128
- Pixel size assumption: 0.096 um/px

## Outputs

- `timebase_corrected_front_fit_rois.csv`: per-ROI threshold front slopes (px^2/s) and apparent D (um^2/s) with bootstrap CIs.
- `timebase_corrected_front_fit_timing.csv`: per-ROI timing diagnostics (dt quantiles, dt max/median ratio).

## Guardrail

Timebase-corrected front fits are apparent kinematics for ranking/diagnosis. They are not calibrated diffusion coefficients.
