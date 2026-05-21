# Calibration Claim Risk Register

Claim-level calibration guardrail for front, kinetic, mobility, and diffusion-like outputs.

- Claim families audited: 11
- Source tables present: 11
- HDF5 timing evidence: 32/33 scanned files have camera_timing
- HDF5 spatial calibration attrs: 0
- PPTX calibration hits: 3

## Interpretation

This register audits wording risk, not numerical correctness. Current front/kinetic outputs are strongest as optical particle-region proxies; diffusion-like values remain apparent proxies until spatial calibration, true frame cadence, masks, and manual QC are jointly validated.

## Highest-Risk Claim Families

- event_candidate_fronts: Use as automatic front-candidate ranking only; do not interpret as calibrated diffusion.
- roi_phase_boundary_mobility: Describe as apparent optical boundary mobility in cropped particle ROIs.
- multi_cycle_roi_mobility: Use for event/control morphology comparisons, not transport constants.
- multi_cycle_threshold_robust_fronts: Phase-slope sign/fraction trends are usable optical proxies; diffusion numbers are apparent um^2/s proxies pending calibration provenance.
- front_roi_calibration_qc: Keep units as apparent um^2/s optical-front proxies and pair with QC flags/manual labels.
- phase_kinetics_avrami: Report as optical transition-sharpness/rate descriptors, not reaction constants.
- manual_qc_gated_front_effects: No manual-QC accepted front/diffusion claim should be made until this table is populated.
- roi_front_qc_package: Use to assign labels and inspect artifacts; it does not validate diffusion by itself.
