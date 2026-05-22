# Agentic Metric Search

Rows scored: 733

## Top Analysis Variants

| source_table | feature_set | method | target | split | auc | average_precision | leakage_penalty | control_bonus | scientific_score | promotion_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| echem_conditioned_residual_dictionary | echem_context | leave_source | future_any_drop_within_8cycles | leave_source | 0.9167 | 0.9577 | 0.08 | 0.22 | 1.464 | promote_for_followup |
| echem_conditioned_residual_dictionary | conditioned_residual_plus_echem_context | leave_source | future_any_drop_within_8cycles | leave_source | 0.9136 | 0.9545 | 0.08 | 0.22 | 1.459 | promote_for_followup |
| echem_conditioned_residual_dictionary | conditioned_residual_plus_handcrafted_echem | leave_source | future_any_drop_within_8cycles | leave_source | 0.902 | 0.9413 | 0.08 | 0.22 | 1.441 | promote_for_followup |
| echem_conditioned_residual_dictionary | conditioned_residual_dictionary | leave_source | future_any_drop_within_16cycles | leave_source | 0.7848 | 0.9432 | 0 | 0.22 | 1.375 | promote_for_followup |
| acquisition_residualized_video_physics | context_plus_residual_dictionary |  | future_any_drop_within_8cycles |  | 1 | 1 | 0.12 | 0.05 | 1.367 | promote_for_followup |
| acquisition_residualized_video_physics | context_plus_video_pca |  | future_any_drop_within_8cycles |  | 1 | 1 | 0.12 | 0.05 | 1.367 | promote_for_followup |
| acquisition_residualized_video_physics | residualized_residual_dictionary_plus_context_logit |  | future_any_drop_within_8cycles |  | 1 | 1 | 0.12 | 0.05 | 1.367 | promote_for_followup |
| acquisition_residualized_video_physics | residualized_video_pca_plus_context_logit |  | future_any_drop_within_8cycles |  | 1 | 1 | 0.12 | 0.05 | 1.367 | promote_for_followup |
| echem_conditioned_residual_dictionary | handcrafted_plus_echem_context | leave_source | future_any_drop_within_8cycles | leave_source | 0.9059 | 0.9461 | 0.08 | 0.13 | 1.357 | promote_for_followup |
| echem_conditioned_residual_dictionary | raw_residual_plus_echem_context | leave_source | future_any_drop_within_8cycles | leave_source | 0.9035 | 0.9446 | 0.08 | 0.13 | 1.354 | promote_for_followup |
| acquisition_residualized_video_physics | context_plus_all_video |  | future_any_drop_within_8cycles |  | 0.99 | 0.991 | 0.12 | 0.05 | 1.352 | promote_for_followup |
| acquisition_residualized_video_physics | context_plus_handcrafted_particle |  | future_any_drop_within_8cycles |  | 0.9861 | 0.9878 | 0.12 | 0.05 | 1.346 | promote_for_followup |
| echem_conditioned_residual_dictionary | echem_context | leave_cycle | future_any_drop_within_8cycles | leave_cycle | 0.9167 | 0.9577 | 0.12 | 0.14 | 1.344 | promote_for_followup |
| echem_conditioned_residual_dictionary | conditioned_residual_plus_echem_context | leave_cycle | future_any_drop_within_8cycles | leave_cycle | 0.9167 | 0.9577 | 0.12 | 0.14 | 1.344 | promote_for_followup |
| echem_residual_dictionary_fusion | residual_dictionary_plus_echem |  | future_any_drop_within_8cycles |  | 0.9167 | 0.9577 | 0.12 | 0.14 | 1.344 | promote_for_followup |
| echem_residual_dictionary_fusion | residual_dictionary_handcrafted_echem |  | future_any_drop_within_8cycles |  | 0.9167 | 0.9577 | 0.12 | 0.14 | 1.344 | promote_for_followup |
| echem_residual_dictionary_fusion | pca_video_plus_echem |  | future_any_drop_within_8cycles |  | 0.9167 | 0.9577 | 0.12 | 0.14 | 1.344 | promote_for_followup |
| echem_residual_dictionary_fusion | handcrafted_plus_echem |  | future_any_drop_within_8cycles |  | 0.9167 | 0.9577 | 0.12 | 0.14 | 1.344 | promote_for_followup |
| echem_conditioned_residual_dictionary | conditioned_residual_plus_handcrafted_echem | leave_cycle | future_any_drop_within_8cycles | leave_cycle | 0.9159 | 0.9569 | 0.12 | 0.14 | 1.343 | promote_for_followup |
| echem_conditioned_residual_dictionary | conditioned_residual_plus_handcrafted_echem | leave_cycle | future_any_drop_within_16cycles | leave_cycle | 0.848 | 0.9613 | 0.04 | 0.14 | 1.333 | promote_for_followup |

## Guardrail

High score ranks computational follow-up candidates only; weak labels, source imbalance, acquisition context, automatic masks, and missing manual QC still block physical mechanism claims.
