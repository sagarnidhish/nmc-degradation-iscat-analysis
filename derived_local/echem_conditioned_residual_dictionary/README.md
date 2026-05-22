# Echem-Conditioned Residual Dictionary

Split-specific residual-dictionary features after subtracting echem/acquisition-predicted residual bases.

- Rows: 172
- Cycles: 34
- Sources: 12
- Residual dictionary features: 39
- Conditioning features: 119

## Top Metrics

- leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem: AUC 0.8479532163742689, AP 0.9613109922185283, p=0.0033222591362126247
- leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_echem_context: AUC 0.8339181286549707, AP 0.9553789936258192, p=0.0033222591362126247
- leave_cycle future_any_drop_within_16cycles handcrafted_plus_echem_context: AUC 0.8210526315789473, AP 0.9558905832389313, p=None
- leave_cycle future_any_drop_within_16cycles echem_context: AUC 0.8058479532163743, AP 0.9514803819444464, p=0.0033222591362126247
- leave_cycle future_any_drop_within_16cycles raw_residual_plus_echem_context: AUC 0.7461988304093568, AP 0.9350801079161684, p=None
- leave_cycle future_any_drop_within_16cycles conditioned_residual_dictionary: AUC 0.6947368421052631, AP 0.9019641999877155, p=0.019933554817275746
- leave_cycle future_any_drop_within_16cycles residual_dictionary_raw: AUC 0.4807017543859649, AP 0.8549923211013175, p=0.5614617940199336
- leave_source future_any_drop_within_16cycles conditioned_residual_dictionary: AUC 0.7847953216374268, AP 0.9431707475860319, p=0.0033222591362126247
- leave_source future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem: AUC 0.7005847953216374, AP 0.9126915366850239, p=0.009966777408637873
- leave_source future_any_drop_within_16cycles conditioned_residual_plus_echem_context: AUC 0.6514619883040935, AP 0.8910633522664138, p=0.05647840531561462
- leave_source future_any_drop_within_16cycles echem_context: AUC 0.6444444444444444, AP 0.883525745902692, p=0.05647840531561462
- leave_source future_any_drop_within_16cycles handcrafted_plus_echem_context: AUC 0.6198830409356725, AP 0.8764540213853513, p=None

## Deltas

- leave_source future_any_drop_within_16cycles conditioned_residual_dictionary_minus_residual_dictionary_raw: delta AUC 0.7263157894736841
- leave_cycle future_any_drop_within_16cycles conditioned_residual_dictionary_minus_residual_dictionary_raw: delta AUC 0.2140350877192982
- leave_source future_any_drop_within_16cycles conditioned_residual_plus_echem_context_minus_raw_residual_plus_echem_context: delta AUC 0.15672514619883038
- leave_source future_any_drop_within_8cycles conditioned_residual_dictionary_minus_residual_dictionary_raw: delta AUC 0.1450617283950617
- leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_echem_context_minus_raw_residual_plus_echem_context: delta AUC 0.08771929824561397
- leave_source future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem_minus_handcrafted_plus_echem_context: delta AUC 0.08070175438596494
- leave_source future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem_minus_conditioned_residual_plus_echem_context: delta AUC 0.0491228070175439
- leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_echem_context_minus_echem_context: delta AUC 0.028070175438596467
- leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem_minus_handcrafted_plus_echem_context: delta AUC 0.026900584795321647
- leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem_minus_conditioned_residual_plus_echem_context: delta AUC 0.014035087719298178

## Guardrail

Echem-conditioned residual dictionary features are split-specific residuals from echem/acquisition predictions of label-free residual bases. They test whether video residual modes add signal beyond measured context, not deployable warning, manual QC, causal mechanism, or calibrated diffusion.
