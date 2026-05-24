# Objectives And Observations (Addendum)

Date: 2026-05-25

## Experiment Priority Matrix

Based on the verified outputs already on Isambard, the most useful next experiments are ranked as follows:

1. **Multi-task physics-aware predictor**
   - Predict next frame, rollout residual energy, phase-fraction slope, front-radius slope, and future-event risk jointly.
   - Why: this is the shortest path from pure pixel prediction to physics extraction.

2. **Rollout / phase-kinetics bridge**
   - Join rollout difficulty to phase-kinetics descriptors and test source-held-out signal.
   - Why: rollout signal is modest by itself, but the phase-kinetics audit is strong.

3. **Front-fit segmentation / threshold robustness**
   - Search for stable positive-expansion windows rather than forcing a single global radius^2 slope.
   - Why: current global front fits are usable for ranking, but not diffusion-ready.

4. **Short-context next-frame control**
   - Keep `context=2` as the control model, since it is best on median MAE.
   - Why: it is the cleanest baseline for comparing future physics-aware heads.

## Verified Evidence Behind the Ranking

- Phase kinetics: `masked_minus_bg_slope` separates near-event rows with AUC up to `0.8494`.
- Rollout difficulty: best source-balanced ROI feature reaches AUC `0.6259`.
- Context sweep: `context=2` beats `context=4` and `context=8` on median MAE.
- Front fits: source-balanced median `thr50_radius2_slope_r2 = 0.0561`, transfer-ranked median `0.0813`.

## Interpretation

The current best path is not another raw pixel model. It is a multitask, source-held-out model over particle-only crops that exposes physically interpretable heads and uses phase kinetics as a conditioning signal.

## Guardrail

This matrix is for experiment prioritization only. It does not claim calibrated diffusion coefficients or manual phase-boundary validation.

