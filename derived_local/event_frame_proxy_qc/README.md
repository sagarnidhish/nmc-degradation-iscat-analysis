# NMC Event Frame Proxy QC

Samples full HDF5 movies around event-cycle segments and builds a fixed proxy ROI mask from temporal variability and brightness. This is a bounded visual/QC check because chopped cycle HDF5 files are unavailable on Isambard.

## Summary

- Rows sampled: 11
- Event rows: 4
- Neighbor rows: 7
- Mean event ROI fraction: 0.0629
- Mean neighbor ROI fraction: 0.0410
- Mean event sampled XY stage drift: 0.1757
- Mean neighbor sampled XY stage drift: 0.1485

## Sampled Segments

| cycleNo | event | source_stem | local_idx | proxy_roi_fraction | roi_cv_time | stage_drift_xy | preview |
|---:|---|---|---:|---:|---:|---:|---|
| 86.0 | yes | `10_c2_x10_030723` | 1 | 0.0811 | 0.0121 | 0.2693 | `previews/cycle_86_10_c2_x10_030723_local1.png` |
| 88.0 | no | `10_c2_x10_030723` | 3 | 0.0398 | 0.01168 | 0.1414 | `previews/cycle_88_10_c2_x10_030723_local3.png` |
| 90.0 | no | `10_c2_x10_030723` | 5 | 0.0533 | 0.006972 | 0.1414 | `previews/cycle_90_10_c2_x10_030723_local5.png` |
| 116.0 | yes | `13_c2_x6_100723` | 1 | 0.0900 | 0.01088 | 0.1414 | `previews/cycle_116_13_c2_x6_100723_local1.png` |
| 118.0 | no | `13_c2_x6_100723` | 3 | 0.0488 | 0.01196 | 0.25 | `previews/cycle_118_13_c2_x6_100723_local3.png` |
| 156.0 | yes | `18_c2_xN_HighHighCOV_170723` | 0 | 0.0230 | 0.02115 | 0.1118 | `previews/cycle_156_18_c2_xN_HighHighCOV_170723_local0.png` |
| 157.0 | no | `18_c2_xN_HighHighCOV_170723` | 1 | 0.0281 | 0.02877 | 0.1118 | `previews/cycle_157_18_c2_xN_HighHighCOV_170723_local1.png` |
| 158.0 | no | `18_c2_xN_HighHighCOV_170723` | 2 | 0.0269 | 0.01978 | 0.1118 | `previews/cycle_158_18_c2_xN_HighHighCOV_170723_local2.png` |
| 58.0 | no | `7_c2_x10_290623` | 3 | 0.0408 | 0.01476 | 0.1414 | `previews/cycle_58_7_c2_x10_290623_local3.png` |
| 60.0 | yes | `7_c2_x10_290623` | 5 | 0.0573 | 0.01385 | 0.1803 | `previews/cycle_60_7_c2_x10_290623_local5.png` |
| 62.0 | no | `7_c2_x10_290623` | 7 | 0.0495 | 0.01253 | 0.1414 | `previews/cycle_62_7_c2_x10_290623_local7.png` |

## Interpretation Guardrail

The ROI is a proxy mask at downsampled full-frame resolution. It is useful for checking gross frame quality, drift, and whether a stable bright/dynamic particle-like region exists, but it is not the original object detector output and should not be used as final particle segmentation.
