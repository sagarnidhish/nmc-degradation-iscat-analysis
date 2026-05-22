# Balanced Future-Drop ROI Reconstruction

Reconstructs automatic particle-region ROI candidates from future8-positive and future8-negative cycles for direct video physics/model audits.

- Cycles sampled: 24
- Candidate components: 1920
- ROI table rows: 72
- ROI label counts: [{'cohort_role': 'future8_negative', 'future_any_drop_within_8cycles': 0, 'n': 36}, {'cohort_role': 'future8_positive', 'future_any_drop_within_8cycles': 1, 'n': 36}]

Automatic class-balanced future-drop ROI candidates reconstructed from sampled HDF5 cycle segments. Labels are weak cycle-level future8 labels projected to particle-region candidates; outputs are not manual annotations, degradation labels, or validated fronts.
