# Source-Balanced Pre-Event Front/Kinetic Null Audit

Source-aware null tests for the front/kinetic concordance score and component features.

- Rows: 128
- Sources: 14
- Features tested: 9
- Source-stratified permutations per test: 50
- Source-balanced bootstraps per test: 0

## Top Null Tests
- near_vs_mid_pre raw kinetic_evidence_score: AUC=0.820, perm p=0.01961, bootstrap disabled
- near_vs_any_non_near raw masked_minus_bg_slope: AUC=0.816, perm p=0.01961, bootstrap disabled
- near_vs_any_non_near raw kinetic_evidence_score: AUC=0.770, perm p=0.01961, bootstrap disabled
- near_vs_mid_pre raw masked_minus_bg_slope: AUC=0.849, perm p=0.03922, bootstrap disabled
- near_vs_any_non_near raw front_kinetic_concordance_score: AUC=0.768, perm p=0.03922, bootstrap disabled
- near_vs_any_non_near source_residual kinetic_evidence_score: AUC=0.598, perm p=0.03922, bootstrap disabled
- near_vs_post_control source_residual front_kinetic_concordance_score: AUC=0.591, perm p=0.05882, bootstrap disabled
- near_vs_any_non_near within_source_rank kinetic_evidence_score: AUC=0.573, perm p=0.05882, bootstrap disabled

## Top Proximity Tests
- within_source_rank strict_qc_priority_score: rho=-0.450, perm p=0.01961
- within_source_rank qc_evidence_score: rho=-0.351, perm p=0.03922
- source_residual qc_evidence_score: rho=-0.296, perm p=0.05882
- source_residual strict_qc_priority_score: rho=-0.328, perm p=0.09804
- source_residual front_kinetic_product: rho=0.082, perm p=0.09804
- source_residual kinetic_evidence_score: rho=0.087, perm p=0.1176

## Guardrail

Source-aware null tests audit whether front/kinetic concordance survives source-stratified label shuffles and source-balanced resampling. Passing rows remain automatic review-prioritization evidence, not manual labels, causal mechanisms, deployable warnings, or calibrated diffusion coefficients.
