# Source-Balanced Residual Dictionary Normalized Readout

Rows/cycles/sources: 96 / 48 / 14
Residual dictionary features: 72; mask/front features: 24

## Future16 Leave-Source Readouts
- future16_leave_source_raw_residual_dictionary: residual_dictionary_raw AUC=0.375, AP=0.448, n=96
- future16_leave_source_source_residual_residual_dictionary: residual_dictionary_source_residual AUC=0.550, AP=0.583, n=96
- future16_leave_source_within_source_rank_residual_dictionary: residual_dictionary_within_source_rank AUC=0.543, AP=0.506, n=96
- future16_leave_source_best: dictionary_recon_error_last_minus_first_source_residual AUC=0.612, AP=0.613, n=96

## Permutation Null
- dictionary_recon_error_last_minus_first_source_residual observed source-heldout future16 AUC=0.612; null p95=0.621; empirical p=0.100

## Guardrail
Source transforms are unsupervised within-source normalizations computed from ROI feature distributions, including held-out source rows without labels. This tests source-normalized readout stability, not a deployable source-transfer warning model.
