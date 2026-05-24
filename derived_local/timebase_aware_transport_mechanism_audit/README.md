# Timebase-Aware Transport Mechanism Audit

Joins HDF5 timebase provenance onto source-balanced transport/front mechanism descriptors.

- ROI rows/cycles/sources: 128 / 64 / 14
- Timebase classes: {'unknown': 56, 'strict': 38, 'pause_heavy': 34}
- Verdict: transport_physics_signal_source_transferable

## Key Results

- Best strict-timebase target AUC: 1.0
- Best pause-heavy target AUC: 1.0
- Best source-heldout physics delta AUC: 0.2451171875000001

## Guardrail

This audit tests whether source-balanced transport/front descriptors are stable under HDF5 timebase quality classes. It uses automatic particle-region crops and source-level timing provenance, not manual particle labels, prospective validation, or calibrated diffusion.
