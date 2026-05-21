# Calibration Metadata Audit

Metadata-only audit for spatial and time calibration evidence behind apparent front/diffusion proxies.

- HDF5 files scanned: 33
- HDF5 files with movie datasets: 32
- HDF5 files with camera timing: 32
- HDF5 files with calibration-like attributes: 0
- Median sampled HDF5 camera FPS: 0.09937929871365499
- PPTX files scanned/calibration hits: 3 / 3

## Interpretation

This audit verifies metadata evidence for timebase/spatial calibration. Camera timing is present in HDF5 files, but physical pixel-size evidence should be treated as slide-derived unless raw HDF5 attributes or microscope metadata explicitly confirm it.
