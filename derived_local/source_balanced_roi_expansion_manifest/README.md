# Source-Balanced ROI Expansion Manifest

- Ranked cycles: 89
- Existing cycle/source keys: 34
- Selected cycles: 48 (41 new vs existing video cohorts)
- Sources selected: 14
- Sampled cycles: 48
- Reconstructed candidates: 2880
- ROI rows: 96

## Label Counts

{
  "future16_positive": 24,
  "future8_positive": 14,
  "same_cycle_drop": 0
}

## Guardrail

Source-balanced expansion candidates are automatic proposals from sampled HDF5 frames. They reduce source/cycle selection bias for follow-up ROI export and manual QC, but do not validate particle identity, fronts, diffusion, or degradation mechanisms.
