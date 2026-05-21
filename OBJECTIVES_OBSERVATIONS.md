# Objectives And Observations

Last updated: 2026-05-21

## Working Objective

Use the Alek_Jiho NMC degradation data and the SP/FOV charge-photometry video work to build AI analyses that extract physically useful signals, not just low pixel error. The main targets are particle-region-only prediction, rollout, phase-boundary/proxy tracking, apparent transport/degradation observables, and correlations between cycle number, region, particle behavior, and degradation events.

## Current Evidence

- Alek_Jiho raw/derived data are on Isambard under `/scratch/u6hp/nsagar.u6hp/Alek_Jiho`.
- The large electrochemistry CSV is present at `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/echemDF_full.csv`; Tier 1 path discovery has been updated to search this location.
- Particle intensity event targets are already generated under `derived/particle_event_targets`.
- The event target table has 356 particle-cycle rows from 4 particles across 89 cycles.
- Abrupt optical drop events are concentrated around cycles 86 and 116 for particles 0-2, with particle1 also at 156 and particle3 at 60. This makes synchronized degradation events a strong hypothesis rather than isolated particle noise.
- Transparent leave-one-particle-out feature baselines show weak but nonzero precursor signal: best F1 is about 0.35 at 10 cycles, 0.36 at 20 cycles, and 0.44 at 40 cycles.
- Existing SP work has already extracted ROI masks, phase labels, per-frame optical observables, and boundary/phase-front proxies for all 8 SP videos.
- Existing pixel rollout models remain weaker than desired: Walrus true rollout has particle MAE ratio near 1.0 and unstable mean-intensity slopes; physics-constrained U-Net training histories plateau around validation rollout MAE 5-6.
- Project decision summaries therefore support moving the central target from pixel recreation to physical-observable forecasting under future electrochemical controls.

## Active Isambard Jobs

As of the latest check, the relevant pending jobs are:

- `4687055` `tier1_nmc_eda`
- `4687056` `tier2_nmc_ml`, dependent on Tier 1
- `4687057` `tier3_nmc_dl`, dependent on Tier 2
- `4687712` `obs_res_gru` for SP physical-observable residual forecasting
- `4687884` `fov_particles`, followed by FOV QC, FOV residual GRU, and FOV observable baselines
- `4687040` and `4687041` probabilistic rollout evaluations for v5/v6

## Computational Experiments To Prioritize

1. NMC synchronized-event validation: test whether the shared event cycles around 86/116 align with capacity fade, dQ/dV changes, cycle frame counts, or HighCOV protocol changes. Use shuffled-cycle and shuffled-particle controls.
2. NMC event forecasting: treat abrupt intensity drops as survival/hazard targets and compare transparent features, gradient boosting, and sequence models with grouped cross-validation by particle/session.
3. SP observable residual forecasting: forecast mean, P95/P99, bright fraction, histogram bins, and boundary proxy from past observables plus future current/voltage/time. Evaluate against persistence by phase and around transitions.
4. Particle-region-only video rollouts: keep ROI/mask selection fixed from robust temporal evidence; when drift correction blurs a frame, propagate the prior ROI rather than thresholding the blurred frame independently.
5. Physics-readiness extraction: rank phase-front candidates by coherent-front score, mask uncertainty, monotonicity, and fit quality. Do not interpret apparent diffusion coefficients as physical constants until spatial calibration and manual front validation are available.
6. Probabilistic rollout calibration: evaluate uncertainty using CRPS and physics-observable coverage, not only pixel metrics. Useful uncertainty should widen near transitions and regions with larger mask/phase uncertainty.

## Current Interpretation

The strongest route is a two-track program. For Alek_Jiho NMC degradation, focus first on cycle-level particle-event and electrochemical coupling because the provided CSV/PPT-derived tables already expose degradation structure. For SP videos, focus on ROI physical-observable and boundary-proxy forecasting because ordinary autoregressive pixel rollout has repeatedly plateaued. Pixel models remain useful when constrained by ROI, future EC controls, and physical losses, but the scientific claims should be made from validated observables and controls.

## 2026-05-21 Event Synchrony Result

A new lightweight synchrony analysis was added in `scripts/tier1_event_synchrony_analysis.py` and run on Isambard. It preserves each particle's number of abrupt-drop events while randomly permuting event cycles over that particle's observed cycle grid.

Output directory: `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_synchrony`

Key result:

- Three particles (`particle0`, `particle1`, `particle2`) event together at cycles 86 and 116.
- Permutation p-value for observing a same-cycle maximum of 3 particles: 0.003300.
- Permutation p-value for observing two cycles with at least two particles eventing: 0.016649.
- Those synchronized cycles have low cycle-frame percentiles (cycle 86: 0.079; cycle 116: 0.011), so the next check is whether these cycles correspond to electrochemical/protocol changes rather than independent local particle failures.

Interpretation: synchronized optical degradation timing is now a concrete hypothesis worth validating against echem capacity, dQ/dV, current/voltage protocol, and shuffled-cycle controls.
