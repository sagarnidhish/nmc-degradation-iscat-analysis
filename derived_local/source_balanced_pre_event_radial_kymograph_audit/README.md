# Source-Balanced Pre-Event Radial Kymograph Audit

- ROI/cycles/sources: 128 / 64 / 14
- Rendered kymographs: 32

## Top Near-Vs-Far Tests
- front_radius2_slope_px2_per_norm_time: AUC=0.705, median diff=3.455, p=0.04025
- front_radius_slope_px_per_norm_time: AUC=0.686, median diff=0.2983, p=0.06242
- front_radius_monotonic_fraction: AUC=0.634, median diff=-0.005263, p=0.08447
- kymograph_temporal_energy: AUC=0.628, median diff=2.306e-06, p=0.1151
- phase_fraction_slope_per_norm_time: AUC=0.619, median diff=0.009027, p=0.1416
- radial_profile_last_minus_first_l1: AUC=0.558, median diff=-0.00108, p=0.4759

## Guardrail

Radial kymographs are automatic fixed-crop optical front proxies from source-balanced ROI tensors. Centroids, thresholds, and gradient fronts are not manual segmentations. Radius-squared slopes are apparent mobility descriptors only, not calibrated diffusion coefficients or validated phase-boundary tracks.
