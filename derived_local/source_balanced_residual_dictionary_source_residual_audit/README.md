# Source-Balanced Residual Dictionary Source-Residual Audit

Rows/features/sources: 96 / 102 / 14
Feature families: {'residual_dictionary': 72, 'mask_front_scalar': 24, 'object_reconstruction': 6}

## Future16 Best Residual Dictionary Features
- raw: resdict_pc01_mean AUC=0.668, AP=0.670, eta2=0.187
- source_residual: dictionary_recon_error_last_minus_first AUC=0.637, AP=0.637, eta2=0.000
- within_source_rank: dictionary_recon_error_mse_slope AUC=0.574, AP=0.551, eta2=0.004

## Guardrail
Source-normalized residual dictionary tests are in-cohort weak-label audits. They can identify source-robust residual dynamics candidates for follow-up, but they do not prove source-transferable prediction or calibrated phase/diffusion physics.
