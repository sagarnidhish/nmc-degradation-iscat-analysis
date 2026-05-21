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

## 2026-05-21 Event-Echem Coupling Result

A targeted chunked scan of `echemDF_full.csv` was added in `scripts/tier1_event_echem_coupling.py` and run on Isambard while the larger Slurm jobs remained pending.

Output directory: `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_echem_coupling`

Key result:

- 81 of 89 particle cycles matched to electrochemistry summaries.
- The synchronized optical event cycles both have echem matches:
  - cycle 86: 3 particles, frame-count percentile 0.079, `galvanostatic_2`, mean V 3.822, mean current 0.00349 mA.
  - cycle 116: 3 particles, frame-count percentile 0.011, `galvanostatic_10`, mean V 3.904, mean current 0.00704 mA.
- Simple per-cycle echem summaries do not explain event occurrence on their own; strongest exploratory differences are lower frame count / lower frame-count percentile for event cycles, but p-values are not strong with only four event cycles.
- Cycle 156 has an optical event for particle1 but no echem match in this scan, so downstream statistics should treat it carefully.

Interpretation: the synchronized optical events are unlikely to be random particle-local coincidences, but they are not yet explained by coarse per-cycle voltage/current summaries. The next useful analyses are protocol-local checks around the event cycles, visual/ROI validation, and image-derived degradation features within those cycles.

## 2026-05-21 Protocol Context Result

A protocol-context pass was added in `scripts/tier1_event_protocol_context.py` using the compact event/echem cycle table.

Output directory: `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_protocol_context`

Key result:

- Synchronized event cycles 86 and 116 have low frame counts: mean 895.5 frames versus 1036.1 for non-synchronized cycles, Mann-Whitney p = 0.0240.
- The same cycles are at very low frame-count percentiles: 0.079 and 0.011.
- Block-position inference was not reliable because dominant block labels produce 82 inferred block segments across 89 particle cycles; treat those boundary features as exploratory only.

Interpretation: synchronized optical degradation events are now linked most clearly to unusually short cycle-frame regimes, not to coarse voltage/current summaries. The next check should inspect raw frame/video quality and particle ROI behavior around cycles 86 and 116 to distinguish physical abrupt degradation from imaging/protocol/frame-count artifacts.

## 2026-05-21 Event Recovery QC Result

A recovery/persistence QC pass was added in `scripts/tier1_event_recovery_qc.py` and run on Isambard using the provided particle intensity traces, cycle-frame metadata, and source HDF5 address strings.

Output directory: `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_recovery_qc`

Key result:

- The synchronized event cycles are not single-cycle transients in the particle trace table.
- Cycle 86: particles 0/1/2 drop together by a mean 14.2%; all 3 remain sustained into the next observed cycle. The next-cycle recovery fraction is negative on average because the following cycle is even lower.
- Cycle 116: particles 0/1/2 drop together by a mean 19.2%; all 3 remain sustained into the next observed cycle despite partial recovery.
- Single-particle event cycles 60 and 156 also have sustained next-cycle deficits, but they are not synchronized across particles.
- Chopped cycle HDF5 files referenced by the Windows-style `addrs` field are not present on Isambard; only the full session HDF5 files are present. Therefore this QC uses trace-level evidence and full-file metadata, not direct particle crops.

Interpretation: the strongest synchronized events behave like persistent degradation-like optical changes rather than one-cycle trace artifacts. The remaining uncertainty is spatial: raw-frame ROI validation is still needed to confirm the drops are in the particle regions and not due to segmentation/object-detection drift in the original trace extraction.

## 2026-05-21 Full-HDF5 Frame Proxy QC Result

A sampled full-frame QC pass was added in `scripts/tier1_event_frame_proxy_qc.py` and run on Isambard because the chopped cycle HDF5 files referenced by `exampleParticles.csv` are not present on Isambard.

Output directory: `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_frame_proxy_qc`

Key result:

- The script sampled 11 full-HDF5 cycle segments: 4 event cycles and 7 nearest represented neighbor cycles.
- It read only downsampled sampled frames from the full `movie` datasets and generated preview PNGs plus ROI/background metrics.
- Event cycles sampled: 60, 86, 116, 156.
- Synchronized event cycle 86 was sampled from `10_c2_x10_030723.hdf5`, inferred local segment 1, frames 906-1812.
- Synchronized event cycle 116 was sampled from `13_c2_x6_100723.hdf5`, inferred local segment 1, frames 909-1818.
- Mean event proxy ROI fraction was 0.0629 versus 0.0410 for neighbor segments.
- Mean event sampled XY stage drift was 0.176 versus 0.148 for neighbor segments; this is not obviously catastrophic at this coarse sampled scale.
- Proxy masks required fallback thresholding in the sampled segments, so they should be treated as gross ROI/QC masks, not final particle/object masks.

Interpretation: full-HDF5 sampled-frame QC did not reveal an obvious missing-frame/raw-file impossibility for event cycles 86 and 116. It provides preview artifacts for manual review and confirms that bounded particle-region-like proxy masks can be generated from the full movies. The next stronger step is to recover or recreate object-level particle crops/coordinates so the synchronized trace drops can be validated on the exact particle regions rather than downsampled full-frame proxy ROIs.

## 2026-05-21 Integrated Event Evidence Result

An integrated evidence table was added in `scripts/tier1_integrated_event_evidence.py` and run on Isambard. It merges synchrony, electrochemistry, protocol/frame-count context, recovery persistence, and full-HDF5 proxy frame QC into one auditable degradation-mode ranking.

Output directory: `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/integrated_event_evidence`

Key result:

- Cycle 116 is currently the strongest degradation-mode candidate: particles 0/1/2 drop together by mean 19.2%, all remain sustained into the next observed cycle, frame-count percentile is 0.011, mean voltage is 3.904 V, and mean current is 0.00704 mA.
- Cycle 86 is the second strongest candidate: particles 0/1/2 drop together by mean 14.2%, all remain sustained into the next observed cycle, frame-count percentile is 0.079, mean voltage is 3.822 V, and mean current is 0.00349 mA.
- Both are classified as `synchronized_persistent_drop_low_frame_count`.
- Cycles 60 and 156 are classified as `single_persistent_drop_needs_roi_qc`.
- The highest-priority next action is exact particle-region validation for cycles 116 and 86, ideally by recovering original object detector coordinates or reconstructing stable particle crops from full-frame videos.

Interpretation: the strongest NMC finding so far is not a diffusion coefficient yet; it is a ranked, evidence-backed degradation-event hypothesis. The current evidence supports coordinated persistent optical degradation at cycles 86 and 116, with a confounding association to low frame counts that must be separated from protocol/acquisition artifacts before mechanistic interpretation.

## 2026-05-21 Agentic Research Workflow Implementation

Implemented the four paper-inspired workflows as separate Isambard folders under:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg/agentic_research`

The four workflows are:

1. `01_era_experiment_search`: ranks next computational experiments using readiness, physics yield, controls, cost, and current evidence.
2. `02_hypothesis_tournament`: scores degradation/transport hypotheses against available evidence and skeptical risks.
3. `03_closed_loop_analysis`: summarizes current outputs and writes a next-action queue.
4. `04_guarded_code_generation`: generates reviewable experiment stubs for top-ranked next analyses.

Smoke outputs are written on Isambard under:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/agentic_research_outputs`

Current ERA ranking prioritizes: frame-count/protocol matched controls, protocol-local echem window scans, ROI event visual QC, hazard calibration, and degradation-mode clustering.

## 2026-05-21 Frame-Count Matched Control Result

Added and ran:

`scripts/tier1_frame_count_matched_event_controls.py`

Output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/frame_count_matched_controls`

Key result:

- Synchronized event cycles are 86 and 116.
- Even after matched sampling against non-event cycles, their frame counts remain unusually low.
- Matched null with 20,000 draws gives:
  - observed mean frame count: 895.5 vs null mean 898.3, empirical lower-tail p = 0.00005.
  - observed mean frame-count percentile: 0.0449 vs null mean 0.0921, empirical lower-tail p = 0.00005.
- Coarse voltage/current metrics are not similarly extreme under the matched null.

Interpretation: the synchronized optical events remain tightly associated with the shortest cycle-frame regimes. This strengthens the artifact/protocol-risk concern rather than proving physical degradation. Raw ROI/event visual QC around cycles 86 and 116 is now mandatory before making a physical claim.

## 2026-05-21 Protocol-Local Echem Window Scan Result

Added and ran:

`scripts/tier1_protocol_local_window_scan.py`

Output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/protocol_local_window_scan`

Key result:

- Compared synchronized event cycles with neighboring cycles over a +/- 6 cycle window.
- Coarse per-cycle echem features do not show strong local anomalies; best local test was echem point count, p = 0.235.
- Current, voltage, duration, and simple deltas are not significant with this small event set.

Interpretation: event timing is not explained by coarse per-cycle voltage/current summaries. The next echem step should inspect within-cycle traces or protocol metadata, but the highest-priority check remains raw optical/ROI validation.

## 2026-05-21 Degradation Mode Clustering Result

Added and ran:

`scripts/tier2_degradation_mode_clustering.py`

Output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/degradation_mode_clustering`

Key result:

- K-means model selection chose k = 4 with silhouette 0.289.
- One cluster is clearly an `abrupt_drop_risk` mode: 53 particle-cycle rows, event rate 0.151, mean cycle about 89, negative delta/trailing slope, low frame-count percentile.
- The other three clusters are currently labeled stable/slow-drift variants, including later-cycle brighter/high-voltage regimes.

Interpretation: the data support an exploratory degradation-mode view, but the cluster labels are not physical mechanisms yet. They should guide QC and hypothesis generation, not serve as final claims without raw ROI/video validation and particle/region controls.
