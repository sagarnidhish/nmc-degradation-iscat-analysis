# Diffusion Unblock Sensitivity Audit

What-if accounting for calibrated diffusion blockers over the current candidate ledger.

- Candidate rows: 60
- Global hard blockers applied to every candidate: ['Control-balanced diffusion sanity candidates', 'Event/control diffusion separability', 'HDF5 spatial calibration metadata present']

## Blockers

- Automatic/publication diffusion candidates: 60 candidate rows; status fail
- Control-balanced diffusion sanity candidates: 60 candidate rows; status fail
- Event/control diffusion separability: 60 candidate rows; status fail
- HDF5 spatial calibration metadata present: 60 candidate rows; status fail
- Radius2 linear-fit quality: 53 candidate rows; status fail
- q70 positive confidence interval: 51 candidate rows; status fail
- Positive front expansion: 35 candidate rows; status
- Manual QC accepted labels: 30 candidate rows; status blocked_pending_manual_qc
- threshold nonnegative: 14 candidate rows; status
- Per-ROI HDF5 timing stability: 11 candidate rows; status fail

## Scenarios

- current_all_guardrails: eligible 0; one blocker remaining 0
- confirm_spatial_calibration_plus_accept_manual_qc_only: eligible 0; one blocker remaining 0
- metadata_manual_qc_and_publication_gate_rechecked: eligible 0; one blocker remaining 0
- external_blockers_cleared_internal_gates_unchanged: eligible 0; one blocker remaining 4
- external_blockers_cleared_plus_timing_and_q70_relaxed: eligible 5; one blocker remaining 17
- all_current_blockers_removed_sanity_upper_bound: eligible 46; one blocker remaining 14

## Nearest Candidates

- cycle78_rank22_obj2: blockers 5, priority 56.47261063486813, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; q70 positive confidence interval
- cycle62_rank1_obj4: blockers 6, priority 15.002721388804652, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; Manual QC accepted labels; Radius2 linear-fit quality
- cycle60_rank2_obj2: blockers 6, priority 15.00025187983658, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; Manual QC accepted labels; Radius2 linear-fit quality
- cycle116_rank7_obj37: blockers 6, priority 10.612143252222973, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; Manual QC accepted labels; Positive front expansion
- cycle140_rank9_obj2: blockers 6, priority 5.4142068261501, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; Per-ROI HDF5 timing stability; q70 positive confidence interval
- cycle153_rank17_obj2: blockers 6, priority 5.410387396133704, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; Per-ROI HDF5 timing stability; q70 positive confidence interval
- cycle139_rank11_obj3: blockers 6, priority 5.407738910757132, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; Per-ROI HDF5 timing stability; q70 positive confidence interval
- cycle108_rank21_obj3: blockers 6, priority 5.407650608408734, remaining Automatic/publication diffusion candidates; Control-balanced diffusion sanity candidates; Event/control diffusion separability; HDF5 spatial calibration metadata present; Positive front expansion; q70 positive confidence interval

## Guardrail

This sensitivity audit removes blockers only in explicit what-if scenarios. It does not change the diffusion readiness status, accept manual labels, relax gates in production, or create calibrated diffusion coefficients.
