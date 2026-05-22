# Source-Balanced Residual-Physics Coupling Audit

Rows/residual features/physics proxies/sources: 96 / 72 / 25 / 14

## Top Source-Residual Couplings for Primary Candidates
- resdict_pc02_slope vs mask_area_fraction_median: rho=0.521, residual AUC=0.515, physics AUC=0.534
- resdict_pc09_slope vs mask_area_fraction_median: rho=-0.501, residual AUC=0.548, physics AUC=0.534
- resdict_pc09_slope vs mask_base_area_fraction: rho=-0.500, residual AUC=0.548, physics AUC=0.526
- resdict_pc02_slope vs mask_base_area_fraction: rho=0.477, residual AUC=0.515, physics AUC=0.526
- resdict_pc09_slope vs mask_centroid_max_step_px: rho=-0.463, residual AUC=0.548, physics AUC=0.499
- resdict_pc02_slope vs mask_centroid_path_px: rho=-0.387, residual AUC=0.515, physics AUC=0.519

## Guardrail
Residual-physics coupling is a source-normalized correlation audit over automatic crop-local optical proxies. It can prioritize follow-up mechanisms, but it does not calibrate diffusion coefficients or prove phase-boundary physics without manual/QC and physical scale validation.
