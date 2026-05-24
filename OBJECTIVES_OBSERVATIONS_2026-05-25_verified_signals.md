# Objectives And Observations (Addendum)

Date: 2026-05-25

## Verified Signals

### Phase kinetics

The source-balanced pre-event phase-kinetics audit shows a strong near-event separation on particle-only crops:

- `near_vs_mid_pre masked_minus_bg_slope`: AUC `0.8494`, p `1.55e-05`
- `near_vs_far_pre masked_minus_bg_slope`: AUC `0.8253`, p `5.76e-05`
- `near_vs_any_non_near masked_minus_bg_slope`: AUC `0.8164`, p `8.99e-08`
- `near_vs_post_control masked_minus_bg_slope`: AUC `0.7987`, p `4.80e-06`

Other phase-fraction / Avrami-like descriptors also contribute, but the masked-minus-background slope is the clearest compact kinetics signal.

### Rollout difficulty

The source-balanced rollout audit shows only modest future-drop predictivity from raw rollout features:

- `future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first`: AUC `0.6259`
- `future_any_drop_within_16cycles raw_roi_mean_delta_last_minus_first`: AUC `0.6059`
- `future_any_drop_within_8cycles temporal_energy_late_minus_early`: AUC `0.6224`
- `future_any_drop_within_8cycles roi_norm_mean_late_minus_early`: AUC `0.6208`

Interpretation: the ROI-only dynamics are informative, but not yet strongly deterministic under the current representation.

### Context sweep

The source-balanced next-frame context sweep shows:

- `context=2` is best on median MAE.
- `context=8` improves the upper tail but not the median.

This indicates the model is not gaining much from long temporal memory, though some difficult sequences still benefit from longer history.

## Current Gaps

- The bridge audit between rollout residuals and phase kinetics is queued but not yet materialized.
- The front-fit jobs are still pending behind cluster priority.
- The current evidence supports physics-facing ranking and hypothesis generation, but not calibrated diffusion coefficients.

## Next Recommendation

Move from raw next-frame baselines to a physics-aware multi-task model that predicts:

- next-frame pixels,
- phase-fraction slope,
- front-radius slope,
- rollout residual energy,
- and future event risk.

That is the best remaining path to turn the existing ROI-only signal into a direct physics-extraction experiment.

