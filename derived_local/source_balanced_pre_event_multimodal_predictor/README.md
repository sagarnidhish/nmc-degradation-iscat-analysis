# Source-Balanced Pre-Event Multimodal Predictor

Leave-source logistic model comparison over echem, front/residual, consensus/QC, and masked phase-kinetic feature families.

- Rows/cycles/sources: 128 / 64 / 14
- Feature family sizes: {'echem_context': 7, 'front_echem_residual': 19, 'consensus_visual_qc': 5, 'phase_kinetics': 64, 'no_kinetics_signal': 31, 'kinetics_plus_front_echem': 88, 'all_signal': 93, 'object_context_guardrail': 6}

## Top Models
- clean_pre16_vs_post_control consensus_visual_qc source_confound_filter_0.25: AUC=0.826, AP=0.879, n=106
- clean_pre16_vs_post_control consensus_visual_qc raw_standard: AUC=0.822, AP=0.875, n=106
- clean_pre16_vs_post_control no_kinetics_signal raw_standard: AUC=0.735, AP=0.777, n=106
- clean_pre16_vs_post_control no_kinetics_signal source_mean_resid_2: AUC=0.695, AP=0.774, n=106
- clean_pre16_vs_post_control no_kinetics_signal source_confound_filter_0.25: AUC=0.653, AP=0.676, n=106
- clean_pre16_vs_post_control all_signal raw_standard: AUC=0.647, AP=0.701, n=106
- clean_pre16_vs_post_control all_signal source_mean_resid_2: AUC=0.613, AP=0.663, n=106
- clean_pre16_vs_post_control object_context_guardrail raw_standard: AUC=0.590, AP=0.621, n=106
- near_vs_far_pre consensus_visual_qc raw_standard: AUC=0.991, AP=0.994, n=54
- near_vs_far_pre echem_context source_mean_resid_2: AUC=0.920, AP=0.937, n=54

## Kinetics Deltas
- near_vs_far_pre raw_standard phase_kinetics vs front_echem_residual: dAUC=0.131, dAP=0.149
- near_vs_post_control source_mean_resid_2 phase_kinetics vs front_echem_residual: dAUC=0.127, dAP=0.109
- clean_pre16_vs_post_control source_confound_filter_0.25 phase_kinetics vs front_echem_residual: dAUC=0.113, dAP=0.048
- clean_pre16_vs_post_control raw_standard phase_kinetics vs front_echem_residual: dAUC=0.098, dAP=0.130
- near_vs_far_pre source_mean_resid_2 phase_kinetics vs front_echem_residual: dAUC=0.095, dAP=0.166
- near_vs_post_control source_confound_filter_0.25 phase_kinetics vs front_echem_residual: dAUC=0.094, dAP=0.033
- near_vs_far_pre source_confound_filter_0.25 phase_kinetics vs front_echem_residual: dAUC=0.065, dAP=0.131
- near_vs_post_control raw_standard phase_kinetics vs front_echem_residual: dAUC=0.059, dAP=0.104

## Guardrail

Leave-source multimodal pre-event models use automatic ROI crops, weak event-relative labels, and analysis-time source-confound transforms. They test whether masked phase kinetics add source-heldout signal to echem/front/QC features; they are not deployable warnings, manual labels, causal mechanisms, calibrated phase-boundaries, or diffusion coefficients.
