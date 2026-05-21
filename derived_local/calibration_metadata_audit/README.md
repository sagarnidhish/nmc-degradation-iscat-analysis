# Calibration Metadata Audit

Metadata-only audit for spatial and time calibration evidence behind apparent front/diffusion proxies.

- HDF5 files discovered before optional cap: 33
- Max HDF5 files requested: 12
- HDF5 files scanned: 12
- HDF5 files with movie datasets: 12
- HDF5 files with camera timing: 12
- HDF5 files with calibration-like attributes: 0
- Median sampled HDF5 timing FPS proxy: 0.013331411250651681
- PPTX files scanned/calibration hits: 3 / 3

## Interpretation

This audit verifies metadata evidence for timebase/spatial calibration. Camera timing is present in sampled HDF5 files, but the sampled timing rows can represent sparse segment/cycle timing rather than true camera frame cadence. Physical pixel-size evidence should be treated as slide-derived unless raw HDF5 attributes or microscope metadata explicitly confirm it.
