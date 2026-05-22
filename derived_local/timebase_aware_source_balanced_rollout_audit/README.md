# Timebase-Aware Source-Balanced Rollout Audit

Joins HDF5 timebase quality onto source-balanced particle-region rollout features.

- ROI rows/cycles/sources: 96 / 48 / 14
- Timebase classes: {'unknown': 40, 'strict': 30, 'pause_heavy': 26}
- Verdict: rollout_signal_timebase_sensitive_pause_heavy_enriched

## Key Results

- Best strict-timebase target AUC: 0.8300000000000001
- Best pause-heavy target AUC: 1.0
- Best source-heldout timebase delta AUC: 0.20703124999999983

## Guardrail

This audit tests whether source-balanced ROI rollout signals are stable under HDF5 timebase quality classes. It uses automatic ROI crops and source-level timing provenance, not manual particle labels, prospective validation, or calibrated diffusion.
