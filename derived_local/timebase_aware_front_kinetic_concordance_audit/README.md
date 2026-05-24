# Timebase-Aware Front/Kinetic Concordance Audit

Joins HDF5 timebase provenance onto source-balanced pre-event front/kinetic candidates.

- ROI rows/cycles/sources: 128 / 64 / 14
- Timebase classes: {'unknown': 56, 'strict': 38, 'pause_heavy': 34}
- Verdict: front_kinetic_signal_source_transferable

## Key Results

- Best strict-timebase target AUC: 1.0
- Best pause-heavy target AUC: 1.0
- Best source-heldout front delta AUC: 0.5681818181818181

## Guardrail

This audit tests whether source-balanced front/kinetic candidates are stable under HDF5 timebase quality classes. It uses automatic particle-region crops and source-level timing provenance, not manual particle labels, prospective validation, or calibrated diffusion.
