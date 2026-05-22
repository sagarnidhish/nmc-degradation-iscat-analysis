# Diffusion Claim Readiness Audit

This folder assembles a go/no-go ledger for calibrated diffusion claims from existing calibration, apparent-diffusion, physics-consistency, control-balanced sanity, and manual-QC outputs.

- Overall status: `not_ready_for_calibrated_diffusion_claim`
- Hard blockers: 8
- Status counts: `{'fail': 7, 'partial': 5, 'pass': 1, 'blocked_pending_manual_qc': 1}`

Hard blockers:
- HDF5 spatial calibration metadata present
- Per-ROI HDF5 timing stability
- Radius2 linear-fit quality
- q70 positive confidence interval
- Automatic/publication diffusion candidates
- Manual QC accepted labels
- Control-balanced diffusion sanity candidates
- Event/control diffusion separability

Outputs:
- `diffusion_claim_readiness_criteria.csv`: gate-level pass/partial/fail ledger.
- `diffusion_claim_readiness_candidates.csv`: ranked ROI/candidate ledger with blockers.
- `diffusion_claim_readiness_summary.json`: compact summary for the synthesis report.

Guardrail: Calibrated diffusion remains blocked unless raw spatial calibration, timestamp semantics, internal front-motion gates, control-balanced sanity checks, and manual QC labels all pass. Current outputs support optical-front proxy review only.
