# Conditioned Residual Physics Atlas

Maps echem-conditioned residual-dictionary modes onto interpretable physics, rollout, particle, mask, echem, and degradation descriptors.

- Rows: 172
- Cycles: 34
- Sources: 12
- Physics descriptor columns screened: 212
- Conditioned residual modes per split: {'leave_cycle': 39, 'leave_source': 39}

## Top Source-Centered Alignments

- leave_source resdict_pc04_mean vs temporal_diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho 0.815008837484705, raw rho 0.5924268548924521
- leave_source resdict_pc04_mean vs diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho 0.815008837484705, raw rho 0.5924268548924521
- leave_source resdict_pc04_mean vs roi_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho 0.8149494605321227, raw rho 0.5924268548924521
- leave_source resdict_pc04_mean vs temporal_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho 0.8149494605321227, raw rho 0.5924268548924521
- leave_source resdict_pc04_mean vs radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho 0.8149494605321227, raw rho 0.5924268548924521
- leave_source resdict_pc04_last_minus_first vs q70_radius2_slope_bootstrap_p50_px2_per_s (front_phase_diffusion): centered rho -0.8147506379248689, raw rho -0.5586464762391082
- leave_source resdict_pc04_last_minus_first vs temporal_diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho -0.8103882214810728, raw rho -0.575790711257567
- leave_source resdict_pc04_last_minus_first vs diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho -0.8103882214810728, raw rho -0.575790711257567
- leave_source resdict_pc04_last_minus_first vs roi_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho -0.8095343859968805, raw rho -0.575790711257567
- leave_source resdict_pc04_last_minus_first vs temporal_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho -0.8095343859968805, raw rho -0.575790711257567
- leave_source resdict_pc04_last_minus_first vs radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho -0.8095343859968805, raw rho -0.575790711257567
- leave_source resdict_pc04_mean vs q70_radius2_slope_bootstrap_p50_px2_per_s (front_phase_diffusion): centered rho 0.7969734055439222, raw rho 0.5662293552212284

## Top Target Tests

- leave_cycle future_any_drop_within_16cycles resdict_pc08_slope: AUC 0.8210526315789473, AP 0.9563026196053434, direction higher_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc08_last_minus_first: AUC 0.8011695906432749, AP 0.9513231451901765, direction higher_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc04_last_minus_first: AUC 0.8, AP 0.9496599690246226, direction higher_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc04_slope: AUC 0.7929824561403509, AP 0.9420405327575161, direction higher_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc04_mean: AUC 0.7859649122807018, AP 0.9299476657146244, direction lower_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc07_mean: AUC 0.7426900584795322, AP 0.8952845853566187, direction higher_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc06_last_minus_first: AUC 0.7415204678362572, AP 0.9190925935658457, direction lower_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc01_last_minus_first: AUC 0.7134502923976608, AP 0.9241851015425127, direction lower_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc06_slope: AUC 0.7064327485380117, AP 0.9110181774257997, direction lower_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc07_last_minus_first: AUC 0.7005847953216373, AP 0.9172418512336229, direction higher_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc07_slope: AUC 0.67953216374269, AP 0.9146492399595771, direction higher_in_positive
- leave_cycle future_any_drop_within_16cycles resdict_pc06_mean: AUC 0.6596491228070176, AP 0.8950573799717594, direction lower_in_positive

## Guardrail

Conditioned residual modes are split-specific, label-free video residual features; source-centered correlations reduce source/acquisition confounding but do not prove causal physics, calibrated diffusion, or deployable warning performance.
