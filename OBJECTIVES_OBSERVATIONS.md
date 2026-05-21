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

## 2026-05-21 Matched Controls And Degradation-Mode Clustering

Three additional scripts were validated and run on Isambard:

- `scripts/tier1_frame_count_matched_event_controls.py`
- `scripts/tier1_protocol_local_window_scan.py`
- `scripts/tier2_degradation_mode_clustering.py`

Output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/frame_count_matched_controls`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/protocol_local_window_scan`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/degradation_mode_clustering`

Key result:

- Frame-count/protocol matched controls reinforce that synchronized event cycles 86 and 116 are at the extreme low end of frame count, even among matched non-event cycles.
  - Observed mean frame count: 895.5; matched-null median: 897.5; empirical lower-tail p = 5.0e-5.
  - Observed mean frame-count percentile: 0.0449; matched-null median: 0.0927; empirical lower-tail p = 5.0e-5.
- Coarse local electrochemistry windows around cycles 86 and 116 still do not show strong voltage/current separation from neighboring cycles; all local-window p-values are >0.23.
- Exploratory degradation-mode clustering selected k=4 with silhouette 0.289. One cluster is labeled `abrupt_drop_risk`, containing 53 particle-cycle rows with 15.1% event rate, negative mean delta, and stronger trailing negative drops. Other clusters are stable/slow-drift modes.

Interpretation: low frame count is not merely a global loose correlate; cycles 86 and 116 are extreme even under matched controls. That makes low frame count a serious acquisition/protocol confound to resolve, not a reason to dismiss the synchronized persistent drops. The most defensible current hypothesis is a coordinated degradation-like optical transition that occurs in an unusual low-frame-count acquisition/protocol regime. Exact particle-region crop validation remains the next decisive check.

## 2026-05-21 Event Frame Proxy QC And Integrated Evidence

Pulled the Isambard-only proxy QC and integrated event evidence scripts into the repo:

- `scripts/tier1_event_frame_proxy_qc.py`
- `scripts/tier1_integrated_event_evidence.py`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_frame_proxy_qc`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/integrated_event_evidence`

Key proxy-QC result:

- Full-session HDF5 files are present, but chopped cycle HDF5 files referenced by `exampleParticles.csv` are not present on Isambard.
- The script samples bounded full-frame segments around event cycles and immediate neighbors, builds fixed proxy masks from temporal variability/brightness, and writes preview PNGs.
- All sampled event rows used fallback proxy masks, so these are not final particle/object masks.
- Event-cycle mean ROI CV is similar to neighbor ROI CV, and sampled stage drift is modest but nonzero.

Key integrated evidence result:

- Cycles 116 and 86 are ranked as `synchronized_persistent_drop_low_frame_count`.
- Cycle 116: three particles, mean drop fraction 0.192, sustained in 3/3 particles, frame-count percentile 0.011.
- Cycle 86: three particles, mean drop fraction 0.142, sustained in 3/3 particles, frame-count percentile 0.079.
- Cycles 60 and 156 are single-particle persistent drops needing ROI QC.

Interpretation: the strongest computational finding is now a ranked degradation-event evidence table, not a final mechanism. Cycles 86 and 116 are high-priority synchronized persistent optical-drop candidates, but the low-frame-count coupling and fallback proxy masks mean exact particle/object-detector ROI recovery is required before claiming physical degradation, phase-boundary motion, or diffusion coefficients.

## 2026-05-21 Object Detector Output Search

A direct search for original object-detection outputs was run locally and on Isambard under the Alek_Jiho tree.

Search targets:

- `particleInfo.csv`
- `particleTraces.csv`
- `segMapStack.npy`
- `maxSegMap*`
- `Output*` directories

Result:

- No original `Output - .../particleInfo.csv` or `particleTraces.csv` folders were found under `/scratch/u6hp/nsagar.u6hp/Alek_Jiho`.
- The `exampleParticles.csv` address field points to Windows paths such as `D://NMC_degradation_3_160623_Halfthedata\10_c2_x10_030723chopped\10_c2_x10_030723_cycle1.hdf5`, but those chopped cycle files and their object-detection output folders are not present on Isambard.
- The repo contains legacy object detection code using SEP (`config/ObjectDetection.py`) and matching by `x`, `y`, and `npix`, so particle coordinates can likely be reconstructed from full HDF5 frames, but this is not identical to recovering the original detector outputs.

Interpretation: exact particle-region validation cannot currently use the original coordinate tables because they are missing from the accessible filesystem. The next implementable path is a reconstruction workflow: use full HDF5 segment sampling around cycles 86 and 116, run SEP-style detection on bounded frame subsets, match candidate particles across neighboring local segments, and compare reconstructed particle intensities against the provided trace-level drops.

## 2026-05-21 Candidate Front / Apparent Transport Proxy Result

Added and ran:

`scripts/tier2_event_candidate_fronts.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_candidate_fronts`

Key result:

- The script sampled 11 bounded full-HDF5 segments around event cycles and immediate neighbors, detected candidate particle-like connected components, and extracted high-intensity fraction, front-radius, and apparent radius-squared slope proxies.
- It produced 88 candidate rows, 8 per segment, plus candidate overlay previews and per-candidate traces.
- Event-cycle candidates had higher front-quality scores than neighbor candidates on average: mean 1.418 for event cycles vs 0.834 for neighbors.
- Cycle 86 has several candidate regions with moderate front-radius fit quality; best candidate score is 2.198 with front-radius R2 0.543, monotonic fraction 0.613, and a small negative apparent radius-squared slope proxy.
- Cycle 116 candidates show persistent optical drop behavior but weaker front-radius coherence; best candidate score is 1.325 with front-radius R2 0.105.

Interpretation: this is useful evidence for ranking candidate particle regions and front-like behavior, but it does not yet justify calibrated diffusion coefficients. The front proxies are in downsampled-pixel/frame units and depend on automatically detected candidate masks. The next physics step is manual validation of candidate overlays for cycles 86/116, then rerunning front tracking on validated ROIs with spatial and time calibration.

## 2026-05-21 Event Object Candidate Reconstruction

Added and ran:

`scripts/tier1_reconstruct_event_object_candidates.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_object_candidate_reconstruction`

Local compact copy:

`derived_local/event_object_candidate_reconstruction`

Key result:

- The script reconstructs approximate particle-like object candidates from sampled full-session HDF5 movie segments because the original chopped HDF5 files and legacy object-detector outputs are missing.
- It sampled 13 event/neighbor segments, including event cycles 60, 86, 116, and 156.
- It generated 1,040 ranked connected-component candidates and 159 adjacent-segment nearest-candidate matches.
- Each sampled segment hit the configured cap of 80 candidates, so these tables should be treated as high-recall candidate lists rather than sparse final segmentations.
- Overlay previews were generated for all sampled segments under `overlays/`, with candidate boxes drawn on sampled mean images and background-subtracted z images.

Interpretation: this closes the immediate filesystem gap enough to support manual ROI/candidate validation without waiting for the missing Windows detector outputs. It still does not recover exact legacy particle coordinates, and the 80-candidate cap indicates a noisy/high-recall detector pass. The next useful step is to inspect the overlays for cycles 86 and 116, select validated candidate ROIs, and rerun calibrated front/intensity tracking on those selected regions.

## 2026-05-21 Event ROI Validation / Selection

Added and ran:

`scripts/tier2_select_validated_event_rois.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_roi_validation`

Local compact copy:

`derived_local/event_roi_validation`

Key result:

- The script fused candidate-front metrics with reconstructed object candidates for synchronized event cycles 86 and 116.
- It linked 16 front/object candidates and selected the top 6 ROI candidates per event cycle.
- Cycle 86: 5/6 selected rows passed the automatic `selected_roi_candidate` label. Top selected ROI approximate full-frame coordinate is `(x=1660.7, y=191.1)` with validation score 5.378 and nearest next-sampled-segment distance 0.347 downsampled pixels.
- Cycle 116: 6/6 selected rows passed the automatic `selected_roi_candidate` label. Top selected ROI approximate full-frame coordinate is `(x=820.9, y=380.1)` with validation score 4.261 and nearest next-sampled-segment distance 0.181 downsampled pixels.
- Focused overlays were written for both cycles under `selected_overlays/`.

Interpretation: this creates a concrete, auditable ROI set for the strongest synchronized events. These are still automatically selected candidates, not manually confirmed legacy detector ROIs, but they are now specific enough to feed particle-region-only models and calibrated follow-up tracking. The next modeling step should use `selected_event_rois.csv` to crop fixed/padded particle regions from full HDF5 frames and train/evaluate event-aware next-frame or rollout models on those ROIs rather than on full frames.

## 2026-05-21 Selected ROI Sequence Export

Added and ran:

`scripts/tier2_export_selected_roi_sequences.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/selected_roi_sequences`

Local compact copy, excluding NPZ tensors:

`derived_local/selected_roi_sequences`

Key result:

- Exported particle-region-only tensors from selected synchronized-event ROIs using fixed padded crops around approximate full-frame object coordinates.
- Used 192x192 full-frame crops resized to 96x96 model inputs, with 96 sampled frames per ROI sequence.
- Exported 11 validation-label ROI sequences: 5 from cycle 86 and 6 from cycle 116.
- The remote NPZ files contain `frames`, `frames_norm`, `frame_indices`, `roi_mean`, `roi_norm_mean`, `average_intensity`, and `stage_position` arrays.
- Cycle 86 selected ROI crops have mean raw intensity delta +3.49 across sampled frames and mean normalized delta +0.000164.
- Cycle 116 selected ROI crops have mean raw intensity delta -11.31 across sampled frames and mean normalized delta -0.000690.
- Sampled XY stage drift is small but nonzero: about 0.25 for cycle 86 and 0.224 for cycle 116.

Interpretation: the project now has concrete particle-region-only model inputs for the two strongest synchronized events, satisfying the requirement that modeling should not feed full frames when selected particle regions are available. The small crop-level deltas show that the trace-level degradation signal is subtle after fixed padded cropping; next-frame/rollout models should use ROI-local contrast, temporal differencing, or event-conditioned targets rather than relying only on absolute crop mean intensity.

## 2026-05-21 Validated Front ROI Selection

Added and ran:

`scripts/tier2_select_validated_front_rois.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/validated_front_rois`

Local compact copy:

`derived_local/validated_front_rois`

Key result:

- The script cross-validates candidate front regions against reconstructed object candidates and nearest neighboring-segment support for cycles 86 and 116.
- It scored 16 candidate front/object matches and selected 10 ROIs for the next front-tracking pass, 5 per event cycle.
- All 10 selected rows were labeled `candidate_roi_supported` by the automatic validation tier.
- Cycle 86 top selected front ROI: validation score 6.575, front-quality score 2.198, front-radius R2 0.543, monotonic fraction 0.613, nearest object match 2.41 downsampled pixels, nearest cycle-88 neighbor 2.21 downsampled pixels, and apparent full-pixel radius-squared proxy -0.000610 px^2/s.
- Cycle 116 top selected front ROI: validation score 5.166, front-quality score 1.255, front-radius R2 0.0067, monotonic fraction 0.742, nearest object match 1.91 downsampled pixels, nearest cycle-118 neighbor 2.28 downsampled pixels, and apparent full-pixel radius-squared proxy 0.0000413 px^2/s.
- Camera timing was read from `camera_timing` to normalize apparent transport proxies to seconds; the observed frame spacing is about 10.04 s/frame for both selected event segments.

Interpretation: this is the first front-specific, time-normalized ROI selection layer. It provides a concrete short list for high-resolution front tracking and manual QC, but the apparent transport values are still downsampled/object-candidate proxies rather than calibrated micron-scale diffusion coefficients. Cycle 86 shows the stronger front-fit evidence; cycle 116 remains a strong synchronized optical-drop event but has weaker front-radius fit quality.


## 2026-05-21 ROI-Only Rollout Baselines

Added and ran:

`scripts/tier3_roi_rollout_baselines.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_rollout_baselines`

Local compact copy:

`derived_local/roi_rollout_baselines`

Key result:

- Evaluated particle-region-only next-frame/rollout baselines on the 11 selected ROI sequences from cycles 86 and 116.
- Methods: persistence, velocity extrapolation, and rank-10 low-rank DMD/PCA latent dynamics.
- Persistence is the strongest short-horizon baseline, showing that selected ROI crops evolve slowly at the sampled cadence.
- Cycle 86 persistence: mean MSE 9.02e-5, SSIM 0.969.
- Cycle 116 persistence: mean MSE 1.03e-4, SSIM 0.964.
- Low-rank DMD is weaker than persistence as a pixel predictor but gives an interpretable latent state. DMD spectral radius is 1.0024, close to marginally stable dynamics.
- Cycle 86 has larger latent movement than cycle 116: mean latent path length 11.11 vs 7.59 and net displacement 1.12 vs 0.315.
- Cycle 116 has negative mean latent component-0 shift (-0.0565), while cycle 86 is positive on average (+0.0330).

Interpretation: the first ROI-only rollout experiment shows that simple persistence is a hard baseline for these slowly varying selected particle crops, so future neural next-frame models must beat a strong near-static baseline and should report residual/difference metrics. The low-rank latent dynamics provide a compact physics-facing descriptor: synchronized event cycles 86 and 116 do not behave identically, with cycle 86 showing more latent movement and cycle 116 showing a more consistent negative latent shift. This supports treating degradation mode as a latent trajectory/phase-space problem rather than only a pixel MSE problem.

## 2026-05-21 ROI Physics Descriptors And Mode Clustering

Added and ran:

`scripts/tier3_roi_physics_descriptors.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_physics_descriptors`

Local compact copy:

`derived_local/roi_physics_descriptors`

Key result:

- Extracted physics-facing descriptors from the 11 selected particle-region ROI sequences: ROI mean trends, high/low optical-state fractions, radial moments, apparent front-radius trends, apparent radius-squared slope proxies, temporal residual energy, and cumulative absolute change.
- Small-sample KMeans selected k=3 with silhouette 0.311 using optical/front/dynamics descriptors.
- Most selected ROIs are `near_static_or_mixed`: 3/5 ROIs in cycle 86 and 4/6 ROIs in cycle 116.
- The remaining selected ROIs were assigned to brightening/expanding-front-like clusters: 2/5 in cycle 86 and 2/6 in cycle 116.
- Cycle 86 has higher cumulative absolute crop change than cycle 116: 0.0151 vs 0.00937.
- Cycle 86 has a more negative apparent diffusion/radius-squared proxy than cycle 116: -0.00202 vs -0.000640 normalized-image px^2/frame.
- Cycle 116 has slightly larger temporal difference energy than cycle 86: 1.05e-4 vs 8.43e-5.

Interpretation: the selected event ROIs do not show a single uniform degradation mode. The dominant behavior is slow near-static/mixed evolution, but a minority of ROIs show front-like brightening/expansion signatures. Cycle 86 appears more spatially/cumulatively active in the selected crops, while cycle 116 looks more temporally noisy but less net-displaced. These descriptors are still image-coordinate proxies; calibrated diffusion claims require spatial calibration, manual ROI validation, and time-base alignment.


## 2026-05-21 High-Resolution Selected Front ROI Tracking

Added and ran:

`scripts/tier3_track_selected_front_rois.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/selected_front_roi_tracking`

Local compact copy:

`derived_local/selected_front_roi_tracking`

Key result:

- Re-read full-resolution HDF5 crops around the selected cycle-86 and cycle-116 ROI candidates and tracked signed optical-change fronts over sampled event segments.
- Tracked 12 selected ROIs: 6 from cycle 86 and 6 from cycle 116.
- Fitted apparent full-pixel radius-squared slopes versus elapsed camera time. These are transport proxies, not calibrated diffusion coefficients.
- Cycle 86: mean radius^2 slope -0.127 full-px^2/s, mean apparent diffusion proxy -0.0318 full-px^2/s, mean fit R2 0.178, final active fraction 0.108, corrected ROI mean delta -1.82.
- Cycle 116: mean radius^2 slope -0.167 full-px^2/s, mean apparent diffusion proxy -0.0418 full-px^2/s, mean fit R2 0.284, final active fraction 0.0489, corrected ROI mean delta -6.63.
- Sampled XY stage drift was similar across the tracked cycle groups, about 0.224.

Interpretation: the higher-resolution crop tracking strengthens the view that the strongest synchronized events are optical-loss/contraction-like rather than simple expanding bright fronts. Cycle 116 has a stronger negative front-radius proxy and larger corrected ROI mean loss than cycle 86. Fit quality remains modest, so these values should be treated as ranked apparent transport proxies until spatial calibration, manual ROI validation, and exact time-base checks are complete.

## 2026-05-21 Event-Conditioned ROI Next-Frame Model

Added and ran:

`scripts/tier3_roi_event_conditioned_nextframe.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_event_conditioned_nextframe`

Local compact copy:

`derived_local/roi_event_conditioned_nextframe`

Key result:

- Trained a fast PCA-latent ridge next-frame model on the 11 selected ROI sequences, conditioned on cycle identity, normalized time, and ROI validation score.
- Used stride-2 ROI inputs, 24 PCA components, train fraction 0.67, and ridge alpha 1.0. PCA explained 99.6% of early-frame variance.
- Teacher-forced next-frame prediction: the event-conditioned model improves over persistence for cycle 116 (MSE 8.55e-05 vs 1.04e-04, SSIM 0.986 vs 0.977), but not for cycle 86 (MSE 3.80e-04 vs persistence 9.05e-05).
- Recursive rollout: persistence remains stronger than the event-conditioned model for both cycles, showing that autoregressive drift is still a hard failure mode.
- Rollout residual energy is higher for cycle 86 than cycle 116: mean 0.00171 vs 0.000611, last-step 0.00398 vs 0.00167, residual slope 1.42e-4 vs 4.87e-5 per step.
- Both cycles have negative truth tail ROI deltas, but the event-conditioned rollout tail drifts slightly positive, so residual sign/magnitude should be treated as a degradation descriptor rather than a faithful long-run simulator.

Interpretation: this gives a concrete AI next-frame experiment on particle-region-only battery photometry videos. It confirms that persistence is a strong baseline and that model residuals carry cycle-specific information. The result supports using rollout residual energy, latent displacement, and front-tracking features together as physics-facing degradation descriptors, while reserving stronger video-model claims for a larger ROI set and better calibrated/manual-QC annotations.


## 2026-05-21 ROI Residual CNN Negative Baseline

Added and ran:

`scripts/tier3_roi_residual_cnn.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_residual_cnn_fast`

Local compact copy:

`derived_local/roi_residual_cnn_fast`

Key result:

- Trained a small particle-ROI-only CNN to predict residuals `next_frame - current_frame`, using cycle-holdout evaluation.
- The persistence baseline is equivalent to predicting zero residual; the learned residual model must improve on this to justify heavier neural rollouts.
- The fast diagnostic run used 11 selected ROI sequences, 132 frame pairs, stride 8, CPU training, hidden width 4, and leave-one-cycle-out testing.
- The residual CNN did not beat persistence.
- Cycle 86 holdout: persistence MSE 7.28e-5, model MSE 7.87e-5, relative MSE improvement -0.0937, residual sign accuracy 0.503.
- Cycle 116 holdout: persistence MSE 1.06e-4, model MSE 7.74e-4, relative MSE improvement -7.68, residual sign accuracy 0.492.
- Overall relative MSE improvement is -4.23, and overall relative residual-MSE improvement is -9.09.

Interpretation: the current selected ROI dataset is too small and too cycle-specific for a naive residual CNN to generalize across event cycles. This is an important negative result: larger neural video models should not be judged by raw next-frame MSE alone and should not be scaled until we add more ROI sequences, stronger event/echem conditioning, or train on broader non-event particle crops. For now, persistence plus physics descriptors/front tracking are more reliable baselines than a small supervised residual CNN.

## 2026-05-21 Matched Control ROI Export And Comparison

Added and ran:

- `scripts/tier2_select_control_rois.py`
- `scripts/tier3_compare_event_control_roi_sequences.py`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_selection`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_control_roi_comparison`

Local compact copies:

- `derived_local/control_roi_selection`
- `derived_local/control_roi_sequences`
- `derived_local/event_control_roi_comparison`

Key result:

- Selected 16 matched non-event reconstructed control ROIs from adjacent source-movie segments: 8 from cycle 88 as controls for event cycle 86, and 8 from cycle 118 as controls for event cycle 116.
- Exported the control ROIs as particle-region-only tensors using the same fixed 192x192 full-frame crop and 96x96 model-input format as the event ROIs.
- Compared 11 selected event ROIs against 16 matched control ROIs.
- Event ROIs show higher cumulative normalized crop change than controls: mean 0.01197 vs 0.00834, Cohen d 1.35, Mann-Whitney p 0.019.
- Event ROIs show lower first-last frame correlation than controls: mean 0.9855 vs 0.9948, Cohen d -1.23, p 0.032.
- Event ROIs show higher normalized intensity variation than controls: std mean 0.1391 vs 0.1309, Cohen d 0.924, p 0.028.
- Raw mean-intensity delta is not strongly separated in this matched-control table, reinforcing that spatial/correlation/residual features are more sensitive than whole-crop mean changes.

Interpretation: this broadens the modeling dataset beyond the synchronized-event ROIs and gives a direct matched-control degradation signature. The strongest event-vs-control differences are not simple average brightness shifts; they are higher spatial/temporal change, lower frame-to-frame structural persistence over the segment, and higher normalized spatial intensity variance. These control ROIs are automatic reconstructed candidates, so manual QC is still needed, but they are now usable as non-event particle-region model inputs and as controls for degradation-mode classifiers.

## 2026-05-21 Event-Control ROI Classifier Guardrail

Added and ran:

`scripts/tier3_event_control_roi_classifier.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_control_roi_classifier`

Local compact copy:

`derived_local/event_control_roi_classifier`

Key result:

- Trained a small logistic classifier to separate selected event ROIs from matched non-event control ROIs using particle-only descriptors.
- Evaluation used pair holdout: train on one event/control pair and test on the other, so the model cannot simply memorize one source movie.
- The classifier did not generalize across event/control pairs. Mean accuracy was 0.404, mean ROC AUC 0.0208, and mean average precision 0.277.
- Held out pair 86/88: accuracy 0.308 and ROC AUC 0.0.
- Held out pair 116/118: accuracy 0.500 and ROC AUC 0.0417.
- The largest average positive coefficients were cumulative absolute normalized change and normalized intensity standard deviation; first-last correlation had a negative coefficient, consistent with the matched-control univariate screen.

Interpretation: event/control ROI descriptors are promising as univariate degradation signatures, but they do not yet form a cycle-general classifier. This is a useful guardrail: cycle/source/protocol effects are strong, and robust event-vs-control classification will require more source movies, more non-event controls, and probably echem/protocol conditioning. The current best use of these features is ranked hypothesis generation and matched-control evidence, not a deployed classifier.

## 2026-05-21 Joint ROI Physics Degradation Modes

Added and ran:

`scripts/tier3_joint_roi_physics_degradation_modes.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_joint_physics_degradation_modes`

Local compact copy:

`derived_local/roi_joint_physics_degradation_modes`

Key result:

- Combined selected-ROI physics descriptors, event-conditioned rollout residual features, residual-CNN guardrail metrics, high-resolution front tracking, integrated event evidence, and cycle-level electrochemistry context into one ROI-level joint table.
- The joint table contains 11 selected ROIs and uses 10 interpretable features, including rollout residual energy, radius-squared front slope, active front fraction, corrected ROI mean loss, cumulative absolute change, high-fraction trend, residual-CNN relative improvement, and cycle evidence score.
- KMeans selected k=2 with silhouette 0.335.
- The highest-score joint mode is `highest_score_contracting_optical_loss`; top ROI is `cycle86_front4_obj9` with joint score 2.86, radius-squared slope -0.187 full px^2/s, corrected ROI delta -15.66, active fraction 0.112, and rollout residual energy 0.00219.
- The next-ranked ROI is `cycle116_front3_obj9` with joint score 1.72, radius-squared slope -0.263 full px^2/s, corrected ROI delta -12.11, active fraction 0.0594, and rollout residual energy 0.000648.
- Cycle-level synthesis: cycle 86 has higher mean AI rollout residual energy and active front fraction; cycle 116 has larger mean optical drop fraction, more negative mean radius-squared slope, and stronger corrected ROI mean loss.

Interpretation: the strongest physics-facing story is now a joint degradation descriptor rather than any single video model. Cycle 86 appears more active/difficult for rollout prediction, while cycle 116 looks more like coherent optical loss/contraction with stronger event evidence. These modes are automatic hypothesis rankings from selected ROIs and still need control ROI expansion, manual QC, and spatial calibration before mechanistic diffusion claims.

## 2026-05-21 Event vs Control ROI Physics Check

Added and ran:

- `scripts/tier2_select_control_rois.py`
- `scripts/tier3_event_vs_control_roi_physics.py`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_selection`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_vs_control_roi_physics`

Local compact copies:

- `derived_local/control_roi_selection`
- `derived_local/control_roi_sequences`
- `derived_local/event_vs_control_roi_physics`

Key result:

- Selected 16 matched non-event control ROIs from nearby reconstructed candidates: 8 from cycle 88 as controls for event cycle 86, and 8 from cycle 118 as controls for event cycle 116.
- Exported control ROI particle-region tensors with the same crop size and sampling style as selected event ROIs.
- Compared 11 event ROIs against 16 matched control ROIs using ROI mean trends, high/low optical-state fractions, temporal difference energy, cumulative absolute change, persistence tail MSE, and stage drift.
- Strongest event-vs-control shifts are in high/bright-state growth and mean-intensity slope: high-fraction slope p=5.03e-4 and ROI-mean slope p=0.00123 by Mann-Whitney tests.
- Mean high-fraction slope is higher for event ROIs than controls: 2.86e-4 vs 1.42e-4 per frame.
- Leave-pair-out logistic classification did not generalize: holdout cycle 86 ROC-AUC 0.10 and holdout cycle 116 ROC-AUC 0.313.

Interpretation: matched controls support the idea that selected event ROIs have different optical-state trend structure, especially bright/high-fraction growth, but simple event-vs-control classification is not robust with only two event/control cycle pairs. This is a useful guardrail: the joint degradation modes should be treated as hypothesis rankings until expanded to more cycles and manually QC'd particle regions.

## 2026-05-21 Expanded Control ROI Screen

Added and ran expanded controls using:

- `scripts/tier2_select_control_rois.py --controls-per-event-cycle 24 --max-controls-per-control-cycle 8`
- `scripts/tier2_export_selected_roi_sequences.py` on the expanded control table
- `scripts/tier3_compare_event_control_roi_sequences.py`
- `scripts/tier3_event_control_roi_classifier.py`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_selection_expanded`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_sequences_expanded`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_control_roi_comparison_expanded`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_control_roi_classifier_expanded`

Local compact copies use the same names under `derived_local/`.

Key result:

- Expanded from 16 to 32 automatic matched control ROIs.
- Controls for event cycle 86 now include cycles 88, 90, and 92, with 8 ROIs per cycle; controls for event cycle 116 include cycle 118 with 8 ROIs.
- Event ROI crops have larger cumulative absolute normalized change than expanded controls: 0.01197 vs 0.00848, Cohen's d 1.48, Mann-Whitney p=0.0100.
- Event ROI crops have lower first-last image correlation than expanded controls: 0.9855 vs 0.9947, Cohen's d -1.50, p=0.0187.
- Event ROI crops also have higher normalized intensity standard deviation: 0.1391 vs 0.1312, p=0.0187.
- Pair-holdout logistic classification remains weak even with expanded controls: mean accuracy 0.372 and mean ROC-AUC 0.290.
- Top classifier coefficients rank cumulative absolute change positive, first-last correlation negative, and stage drift negative among the strongest separating features.

Interpretation: expanded controls strengthen the feature-level evidence that event ROIs are more dynamically changing and less temporally correlated than nearby non-event particle-like controls. However, the poor pair-holdout classifier means the current feature set does not yet generalize robustly across event/control cycle pairs. This supports using event-control descriptors as physics evidence and guardrails, not as a final automated event detector.

## 2026-05-21 ROI Phase-Boundary Mobility Proxies

Added and ran:

`scripts/tier3_roi_phase_boundary_mobility.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_phase_boundary_mobility`

Local compact copy:

`derived_local/roi_phase_boundary_mobility`

Key result:

- Computed threshold-front mobility proxies on the selected event ROI tensors and expanded matched controls: bright/high phase fraction slopes, mid-threshold front fraction, high/low apparent radius-squared changes, interface-density changes, centroid path, first-last correlation, and first-last absolute change.
- Compared 11 event ROIs with 32 expanded control ROIs from cycles 88, 90, 92, and 118.
- Event ROIs show faster high-phase fraction growth than controls: mean slope 3.36e-4 vs 1.54e-4 per frame, Mann-Whitney p=0.00097.
- Using HDF5 `camera_timing`, those high-phase growth rates correspond to 3.51e-6/s for events vs 1.60e-6/s for expanded controls; sampled ROI frames are roughly 10 s apart.
- Event ROIs show larger first-last absolute change: 0.01213 vs 0.00828, p=0.00925, and lower first-last correlation: 0.9863 vs 0.9957, p=0.0118.
- The apparent high-phase radius-squared change is larger for events: +7.01 px^2 vs -0.91 px^2, p=0.0100. This is a phase-boundary proxy only, not a calibrated diffusion coefficient.
- Interface density increases slightly in event ROIs but decreases in controls: +0.00128 vs -0.00182, p=0.0241.
- Cycle 86 event ROIs carry the strongest mobility signal: cumulative first-last change 0.01579 vs 0.00835 for its controls, first-last correlation 0.9754 vs 0.9957, and four of the top four mobility-ranked ROIs are cycle 86 events.
- Cycle 116 event ROIs have high bright-fraction growth but much weaker decorrelation and mobility ranking, consistent with the earlier interpretation that cycle 116 is more coherent optical loss/contraction while cycle 86 is more dynamically heterogeneous.

Interpretation: this gives a physics-facing bridge between video prediction residuals and phase-boundary language. The selected event ROIs, especially cycle 86, are not just harder to predict; their threshold-front geometry changes faster and loses structural correlation relative to expanded controls. These are still apparent optical front metrics from fixed particle crops, so they should be treated as calibrated-diffusion candidates only after spatial calibration and manual ROI QC.

## 2026-05-21 Front ROI Calibration QC

Added and ran:

`scripts/tier3_front_roi_calibration_qc.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/front_roi_calibration_qc`

Local compact copy:

`derived_local/front_roi_calibration_qc`

Key result:

- Built a provisional calibration/QC manifest for the selected front-tracking ROIs and joined it to the joint degradation-mode ranking.
- Used provisional pixel calibration 0.096 um/px from `Battery_Degradation_Project/Degradation Paper Outline.pptx` slide text noting 96 nm pixel size and 180x120 um FoV.
- Converted front-tracking radius-squared slopes and apparent diffusion proxies from full-pixel units into um^2/s using pixel area 0.009216 um^2/px^2.
- Top joint-ranked ROI `cycle86_front4_obj9` has radius^2 slope -0.00172 um^2/s and apparent diffusion proxy -0.000431 um^2/s, with radius^2 fit R2 0.258 and no automatic QC warning.
- Top cycle 116 ROI has radius^2 slope -0.00242 um^2/s and apparent diffusion proxy -0.000605 um^2/s, R2 0.299.
- Several lower-ranked ROIs are automatically flagged for low radius^2 fit R2, and all ROIs remain `manual_qc_status=pending`.

Interpretation: this converts the front-tracking outputs into physically interpretable provisional units while keeping a strict guardrail. The signs are mostly negative, so the current front metric is better described as apparent optical-front contraction/loss than diffusion expansion. These values should not be treated as final diffusion coefficients until the 96 nm/px calibration is confirmed from microscope metadata and the front masks/particle identities are manually reviewed.

## 2026-05-21 Expanded Multi-Cycle ROI Cohort

Added and ran:

- `scripts/tier3_build_multicycle_roi_cohort.py`
- `scripts/tier2_export_selected_roi_sequences.py` on the expanded cohort table
- `scripts/tier3_multicycle_roi_cohort_analysis.py`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_cohort`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_analysis`

Local compact copies use the same names under `derived_local/`.

Key result:

- Built an expanded automatic ROI cohort for synchronized event cycles 86/116 and single-particle event cycles 60/156, plus nearby non-event controls.
- The cohort has 52 ROI crops: 24 event ROIs and 28 control ROIs. Control cycles are 58/62 for event 60, 88/90 for event 86, 118 for event 116, and 157/158 for event 156.
- Exported 64-frame fixed particle-region crops for all 52 ROIs from the full HDF5 movies.
- Across the expanded cohort, event ROIs separate from controls most strongly by ROI normalized mean slope (event 1.05e-4 vs control 4.50e-5 per frame, Mann-Whitney p=4.71e-4) and high-fraction slope (event 4.60e-4 vs control 1.87e-4 per frame, p=0.00238).
- Single-particle candidate event cycles 60/156 show larger visible ROI dynamics than their controls: event first-last correlation 0.889 vs control 0.930, cumulative absolute normalized change 0.0392 vs 0.0250, and positive ROI mean delta 0.0151 vs control -0.00273.
- Synchronized event cycles 86/116 still show subtler but consistent event/control shifts: event first-last correlation 0.985 vs control 0.996 and cumulative absolute normalized change 0.0125 vs 0.00880.
- The top expanded-cohort physics-ranked ROIs are mostly cycle 156 event ROIs (`cycle156_rank7_obj27`, `cycle156_rank5_obj4`, `cycle156_rank8_obj10`, `cycle156_rank2_obj2`), followed by cycle 60 and the known cycle 86 ROI `cycle86_rank4_obj9`.

Interpretation: expanding beyond the synchronized cycles changes the story. Cycles 86/116 remain the strongest coordinated degradation-event candidates, but cycles 60/156 have stronger local optical dynamics and may be better for learning particle-region degradation morphology. The useful AI program is therefore two-track: use synchronized cycles for event/protocol/degradation timing, and use the expanded single+sync ROI cohort for front/mobility morphology, rollout residuals, and degradation-mode discovery. All expanded ROIs are still automatic candidates and need manual particle/front QC before mechanistic claims.

## 2026-05-21 Multi-Cycle ROI Cohort And Mobility

Added and ran:

- `scripts/tier3_build_multicycle_roi_cohort.py`
- `scripts/tier3_multicycle_roi_mobility_summary.py`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_cohort`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_mobility`

Local compact copies:

- `derived_local/multi_cycle_roi_cohort`
- `derived_local/multi_cycle_roi_sequences`
- `derived_local/multi_cycle_roi_mobility`

Key result:

- Built an automatic multi-cycle ROI cohort beyond the synchronized cycles: 52 ROIs total, with 24 event ROIs and 28 matched controls.
- Event cycles included 60, 86, 116, and 156. Matched controls came from cycles 58/62 for 60, 88/90 for 86, 118 for 116, and 157/158 for 156.
- Exported all 52 ROIs as fixed 192x192 full-frame padded crops and 96-frame, 96x96 model tensors.
- Overall multi-cycle events have higher timing-normalized high-phase growth than controls: 2.55e-6/s vs 1.45e-6/s, Mann-Whitney p=0.0143.
- Cycle-specific signals differ by degradation mode. Cycle 86 events show strong structural decorrelation and interface growth relative to controls: first-last correlation 0.9750 vs 0.9955, cumulative first-last change 0.0161 vs 0.0090, and interface-density delta +0.00404 vs -0.00247, all p=0.000666 in the small per-cycle test.
- Cycle 156 events show the largest optical-state shift: normalized ROI mean delta +0.0301 vs -0.00593 for controls and high-phase growth 2.24e-6/s vs -4.06e-7/s, both p=0.000666.
- Cycle 60 is more ambiguous because adjacent controls also show large activity; this is a useful guardrail against overinterpreting single-particle events without more controls or manual QC.
- Top multi-cycle mobility-ranked event ROIs are dominated by cycle 156 and cycle 60, while the previously important `cycle86_rank4_obj9` remains high-ranked but no longer dominates the full multi-cycle cohort.

Interpretation: the analysis now has a broader event/control ROI dataset instead of relying only on cycles 86 and 116. The physics picture is heterogeneous: cycle 86 looks like structural/front disorder, cycle 116 remains coherent optical loss with weak decorrelation, cycle 156 is a strong brightening/phase-growth case, and cycle 60 needs caution because controls are also active. This cohort is now ready for broader rollout modeling and cycle-conditioned degradation-mode learning.

## 2026-05-21 Multi-Cycle ROI Rollout Baseline

Added and ran the existing rollout baseline on the expanded 52-ROI cohort:

`python scripts/tier3_roi_rollout_baselines.py --roi-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_rollout_baselines --rank 16 --train-fraction 0.67`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_rollout_baselines`

Local compact copy:

`derived_local/multi_cycle_roi_rollout_baselines`

Key result:

- Evaluated persistence, velocity, and low-rank DMD recursive rollouts for all 52 expanded ROI sequences.
- Persistence remains the strongest pixel-space baseline across all cycles. Example MSE means: cycle 60 persistence 1.27e-4 vs DMD 2.41e-3; cycle 86 persistence 9.83e-5 vs DMD 1.21e-3; cycle 116 persistence 1.00e-4 vs DMD 1.06e-3; cycle 156 persistence 1.16e-4 vs DMD 3.13e-3.
- The value of the rollout pass is therefore not better pixel prediction, but latent/rollout descriptors. Latent net displacement is much larger for single-particle event cycles 60 and 156 than for synchronized cycles 86 and 116: cycle 60 mean 3.12, cycle 156 mean 4.17, cycle 86 mean 1.24, cycle 116 mean 0.383.
- Cycle 156 has the largest event-like latent movement among the candidate event cycles, consistent with the expanded cohort descriptor result that single-particle cycles carry stronger visible morphology.

Interpretation: the expanded AI rollout confirms the project direction. Standard low-rank image rollout does not beat persistence, so the publishable signal should come from physically constrained descriptors: latent displacement, event/control front-mobility shifts, calibrated-but-provisional front tracking, and degradation-mode clustering. The stronger latent dynamics in cycles 60/156 make them useful morphology-training cases, while cycles 86/116 remain the strongest coordinated event-timing cases.

## 2026-05-21 Multi-Cycle Rollout-Mobility Coupling

Added and ran:

`scripts/tier3_multicycle_rollout_mobility_coupling.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_rollout_mobility_coupling`

Local compact copy:

`derived_local/multi_cycle_rollout_mobility_coupling`

Key result:

- Joined the 52-ROI multi-cycle mobility descriptor table with ROI-only rollout baselines and PCA-latent trajectory summaries.
- Persistence remains the strongest raw predictor baseline; event and control ROIs do not separate cleanly by persistence MSE alone (event mean 1.10e-4 vs control 1.20e-4, p=0.762).
- Event ROIs show a larger DMD failure ratio relative to persistence, but this is a trend rather than a significant global separation: DMD/persistence MSE ratio 20.49 vs 14.12, p=0.125.
- The strongest coupling is between simple temporal activity and persistence error: temporal-diff energy vs persistence MSE Spearman rho=0.978, and temporal-diff energy vs persistence SSIM rho=-0.861. This is useful but mostly tautological, so persistence MSE should be treated as an activity proxy rather than a standalone degradation marker.
- More physics-facing latent couplings are strong: cumulative first-last ROI change vs latent net displacement rho=0.889, first-last correlation vs latent net displacement rho=-0.885, and centroid path vs latent path length rho=0.701.
- Integrated event evidence anticorrelates with latent net displacement (rho=-0.806) but correlates with velocity-model failure ratio (rho=0.711), suggesting synchronized-event cycles are not simply the largest latent movers; they are cycles where naive motion extrapolation becomes less reliable.
- Top rollout/mobility difficulty ROIs include one active control (`cycle62_rank3_obj9`) and multiple event ROIs from cycles 156 and 60, with `cycle86_rank4_obj9` still present but no longer the top-ranked ROI in the broader cohort.

Interpretation: the multi-cycle rollout analysis supports a guarded modeling conclusion. Next-frame persistence is a strong baseline and mostly measures short-term activity; the more useful AI/physics signal is the mismatch between simple dynamical models and ROI phase/mobility descriptors. Cycle 156 and cycle 60 contain the hardest high-mobility ROIs, while cycle 86 remains a structurally decorrelating synchronized-event regime and cycle 116 remains coherent/low-decorrelation.

## 2026-05-21 Multi-Cycle ROI Event Predictor

Added and ran:

`python scripts/tier3_multicycle_roi_event_predictor.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_event_predictor`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_event_predictor`

Local compact copy:

`derived_local/multi_cycle_roi_event_predictor`

Key result:

- Trained leakage-guarded event/control classifiers on the 52-ROI expanded cohort using leave-event-reference-cycle-out folds. Direct evidence score, event particle count, cycle number, and event-reference cycle were excluded from features.
- Two feature sets were evaluated: `all_physics_plus_qc`, and `physics_no_selection_qc`, which removes stage drift, validation score, and front-quality score to reduce selection/acquisition leakage.
- With all physics plus QC features, mean leave-cycle-out performance was modest: logistic mean ROC-AUC 0.703 / balanced accuracy 0.609; random forest mean ROC-AUC 0.797 / balanced accuracy 0.688.
- The all-feature logistic model depended most on `stage_drift_xy_sampled`, `roi_norm_mean_slope_r2`, and `front_quality_score`, so that result is partly acquisition/selection-sensitive.
- In the stricter `physics_no_selection_qc` setting, performance dropped but stayed above random on average: logistic mean ROC-AUC 0.625 / balanced accuracy 0.563; random forest mean ROC-AUC 0.651 / balanced accuracy 0.573.
- The stricter physics features ranked ROI mean-slope fit quality, centroid path, cumulative absolute normalized change, rollout SSIM, ROI mean slope, ROI mean delta, high-fraction slope, first-last correlation, and latent path length among the leading descriptors.
- Fold behavior is heterogeneous: cycle 86 and 116 are relatively separable in several folds, while cycle 156 reverses or depends strongly on model class.

Interpretation: ROI physics and rollout descriptors contain some event/control signal, but the current 52-ROI cohort is not enough for a reliable event detector. The predictor is most useful as a ranking and ablation tool: it identifies which features are likely physical (`high_fraction_slope`, `cumulative_abs_norm_change`, `first_last_corr`, latent movement) and which are likely guardrail/artifact-sensitive (`stage_drift_xy_sampled`, validation/front-quality scores). This supports continuing with physics descriptor extraction and manual QC rather than claiming automated degradation detection.

## 2026-05-21 Multi-Cycle ROI Echem Coupling

Added and ran:

`python scripts/tier3_multicycle_roi_echem_coupling.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_echem_coupling`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_echem_coupling`

Local compact copy:

`derived_local/multi_cycle_roi_echem_coupling`

Key result:

- Joined 52 multi-cycle ROI rows across 11 cycles with event-reference metadata, electrochemical summaries where available, protocol-block position, mobility descriptors, rollout baselines, and event-predictor probabilities.
- Event-reference cycles have distinct optical and protocol contexts: cycle 60 has stronger latent movement than cycles 86 and 116, cycle 116 sits at higher mean voltage, and cycle 156 has the largest normalized ROI mean delta but lacks a matched echem row in the current join.
- ROI/echem and ROI/protocol correlations are strong, but many are acquisition/protocol entangled rather than clean degradation physics. Frame-count percentile correlates with latent net displacement (rho=0.776), cumulative absolute normalized change (rho=0.756), first-last correlation (rho=-0.700), and DMD failure over persistence (rho=0.595). Protocol-block position also correlates with latent path and net displacement.
- Current/echem deltas show additional coupling where matched rows exist: `I_mean_delta` anticorrelates with latent net displacement (rho=-0.675, n=38) and correlates with first-last correlation (rho=0.636, n=38).
- Within-event-reference centered event/control tests still show optical separation after removing event-reference-cycle context: cumulative absolute normalized change +0.00953 (p=2.74e-05), first-last correlation -0.0273 (p=4.71e-04), latent net displacement +0.846 (p=6.19e-04), high-fraction delta +0.0478 (p=6.84e-04), ROI mean delta +0.00968 (p=8.08e-04), and high-fraction slope +2.54e-04 (p=0.00407).
- The strict physics event-predictor probability is not significant after within-reference centering (p=0.121), which is an important guardrail against presenting the current classifier as a validated detector.

Interpretation: the new coupling analysis makes the main caveat explicit. ROI morphology, rollout descriptors, and apparent event separability are strongly conditioned by frame count and protocol-block position, so echem/protocol metadata should be used as covariates and guardrails. At the same time, event-vs-control optical shifts survive within-reference centering, supporting a physically useful signal that should be advanced through manual QC, cycle-conditioned models, and echem-aware validation rather than a raw automated detector claim.

## 2026-05-21 NMC AI Physics Synthesis

Added and ran:

`python scripts/tier4_nmc_ai_physics_synthesis.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/nmc_ai_physics_synthesis`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/nmc_ai_physics_synthesis`

Local compact copy:

`derived_local/nmc_ai_physics_synthesis`

Key result:

- Generated a top-level synthesis report and requirement-by-requirement completion audit from the Isambard derived outputs.
- The current evidence base contains 52 multi-cycle ROI/echem rows, 11 cycles, 4 event-reference cycles, and 12 calibrated front-QC ROI rows.
- Persistence is confirmed as the best raw cycle-level MSE baseline across the expanded rollout cohort; DMD, velocity, PCA latent trajectories, PCA-ridge, and residual-CNN runs are most useful as residual and latent physics descriptors.
- Strict no-selection-QC event ranking remains only modest: random forest mean ROC-AUC 0.651 / balanced accuracy 0.573, logistic mean ROC-AUC 0.625 / balanced accuracy 0.562. The all-feature random forest reaches ROC-AUC 0.797 / balanced accuracy 0.688, but that includes QC/acquisition-sensitive features.
- The strongest within-reference event/control optical separations remain cumulative normalized change, first-last decorrelation, latent net displacement, high-fraction growth, and ROI mean trend.
- The strongest ROI/echem/protocol couplings are frame-count percentile versus latent displacement, cumulative absolute change, first-last correlation, and DMD failure over persistence, reinforcing protocol/frame-count confounding as a central guardrail.
- The highest-priority ROI candidates for manual review include `cycle62_rank3_obj9` as an active control and event ROIs from cycles 156, 60, and 86.
- The synthesis explicitly classifies diffusion extraction as `partial_proxy_only`: provisional 0.096 um/px apparent optical-front proxies exist, but calibrated diffusion coefficients are not yet validated because all front ROIs remain manual-QC pending.

Interpretation: the project now has an auditable top-level index tying the requested goals to concrete outputs and limitations. The core AI/physics workflow is implemented as computational hypothesis generation: next-frame/rollout baselines, ROI latent dynamics, phase/front mobility proxies, degradation-mode rankings, and echem/protocol coupling. The remaining scientific gap is not more unconditioned modeling; it is manual ROI/front QC, broader cycle coverage, and echem/protocol-conditioned validation before making calibrated diffusion or deployable detector claims.

## 2026-05-21 Multi-Cycle ROI Event Predictor Null

Added and ran:

`python scripts/tier3_multicycle_roi_event_predictor_null.py --n-permutations 200 --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_event_predictor_null`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_event_predictor_null`

Local compact copy:

`derived_local/multi_cycle_roi_event_predictor_null`

Key result:

- Ran a 200-permutation within-event-reference-cycle label null using the same leave-cycle-out folds and model classes as the 52-ROI event predictor.
- The all-physics-plus-QC random forest exceeded the null: observed mean ROC-AUC 0.797 vs null 0.490, p=0.00498; observed balanced accuracy 0.688 vs null 0.498, p=0.00995; observed average precision 0.809 vs null 0.561, p=0.00995.
- The all-physics-plus-QC logistic model also exceeded the null by ranking metrics: ROC-AUC p=0.0199 and average precision p=0.0149, while balanced accuracy was weaker at p=0.0697.
- The stricter `physics_no_selection_qc` models were only suggestive, not decisive: logistic ROC-AUC p=0.0896 and random forest ROC-AUC p=0.104, with balanced-accuracy p-values 0.254 and 0.184.

Interpretation: the predictor signal is real relative to shuffled labels when QC/acquisition descriptors are included, but the stricter physics-only signal remains borderline in this small cohort. This strengthens the guardrail: the current classifier should be used for feature triage and experiment design, not as evidence of a deployable degradation detector.

## 2026-05-21 Protocol-Conditioned ROI Event Effects

Added and ran:

`python scripts/tier4_protocol_conditioned_roi_effects.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/protocol_conditioned_roi_effects`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/protocol_conditioned_roi_effects`

Local compact copy:

`derived_local/protocol_conditioned_roi_effects`

Key result:

- Residualized 19 ROI optical/rollout descriptors against available protocol/echem covariates (`n_frames_percentile`, protocol-block position, voltage/current summaries, and duration) plus event-reference-cycle fixed effects.
- Protocol/echem context explains substantial variance in several raw ROI descriptors: ROI mean delta 0.890, high-fraction delta 0.865, low-fraction delta 0.846, latent net displacement 0.770, cumulative absolute normalized change 0.766, first-last correlation 0.660, and high-fraction slope 0.654.
- Despite that adjustment, event/control separation survives in residual space: ROI mean delta residual +0.00322 (p=4.42e-05), high-fraction delta residual +0.0145 (p=8.24e-05), low-fraction delta residual -0.0150 (p=4.71e-04), first-last correlation residual -0.0184 (p=0.0193), cumulative absolute normalized change residual +0.00386 (p=0.0193), DMD-minus-persistence residual +4.51e-04 (p=0.0407), latent net displacement residual +0.377 (p=0.0425), and low-rank DMD MSE residual +4.59e-04 (p=0.0464).
- Leave-event-reference-out logistic classification using residualized descriptors gives mean ROC-AUC 0.672 and balanced accuracy 0.682.
- Updated the tier4 synthesis report to include these protocol-conditioned results.

Interpretation: this is the strongest guardrail-aware evidence so far that event ROIs differ from controls beyond obvious frame-count/protocol context. The result does not prove causality because the cohort is still 52 automatically selected ROIs, but it improves the project from raw correlations to conditioned optical/rollout event effects. These residualized descriptors should be the default input for any next event-ranking or degradation-mode model.

## 2026-05-21 Multi-Cycle Threshold-Robust Fronts

Added and ran:

`python scripts/tier3_multicycle_threshold_robust_fronts.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_threshold_robust_fronts --n-bootstrap 200`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_threshold_robust_fronts`

Local compact copy:

`derived_local/multi_cycle_threshold_robust_fronts`

Key result:

- Swept seven bright-phase thresholds (`0.55` to `0.85` early-frame quantiles) for all 52 particle ROI tensors, using camera timing and the provisional 0.096 um/px calibration.
- Bootstrapped the default q70 threshold slopes 200 times per ROI and summarized threshold sign agreement, slope IQR, apparent radius-squared motion, and apparent optical-front diffusion proxy `slope(radius^2) * pixel_size^2 / 4`.
- Event ROIs show globally higher threshold-robust bright-phase growth than controls: median phase-fraction slope 2.57e-6/s vs 1.48e-6/s, p=0.0175; absolute median slope 2.59e-6/s vs 1.79e-6/s, p=0.0310; threshold-robust phase score 0.593 vs 0.438, p=0.0406; positive-threshold-slope fraction 0.970 vs 0.832, p=0.0406.
- Apparent diffusion proxies do not separate events globally: median proxy -7.61e-7 vs -1.91e-6 um^2/s, p=0.259; absolute proxy 2.22e-6 vs 2.69e-6 um^2/s, p=0.680; diffusion robustness score p=0.993.
- Cycle-specific physics remains heterogeneous. Cycle 156 events have strong threshold-robust phase growth relative to controls (2.21e-6/s vs -3.76e-7/s, p=0.000666). Cycle 86 events have weaker contraction/diffusion-proxy magnitude than controls (absolute proxy 8.69e-7 vs 5.00e-6 um^2/s, p=0.0127), consistent with cycle 86 being more structural/decorrelating than a simple expanding-front case.
- Top robust phase-front ROIs include `cycle156_rank6_obj3`, `cycle60_rank2_obj2`, `cycle116_rank1_obj6`, `cycle116_rank2_obj2`, and `cycle156_rank2_obj2`. Top robust diffusion-proxy candidates include both event and active-control ROIs, reinforcing the need for manual front QC.

Interpretation: threshold-swept ROI fronts strengthen the phase-boundary movement evidence while keeping diffusion claims guarded. Bright-phase growth is robust to threshold choice and bootstrapping across the expanded event cohort, but apparent diffusion proxies are threshold- and control-sensitive. The strongest current physics claim is therefore phase/front mobility and optical-state change, not calibrated Li diffusion coefficients.

## 2026-05-21 Protocol-Conditioned Front Effects

Added and ran:

`python scripts/tier4_protocol_conditioned_front_effects.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/protocol_conditioned_front_effects`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/protocol_conditioned_front_effects`

Local compact copy:

`derived_local/protocol_conditioned_front_effects`

Key result:

- Joined the 52-ROI echem/protocol table with the threshold-robust front metrics and residualized 17 front features against frame-count percentile, protocol-block position, voltage/current summaries, duration, timing elapsed, and event-reference-cycle fixed effects.
- Protocol/echem context explains substantial variance in several threshold-front descriptors: phase-slope positive/negative fraction 0.810, phase-slope median 0.576, q70 bootstrap phase slope 0.565, radius2/diffusion IQR 0.585, and phase-slope IQR 0.511.
- Raw front metrics reproduce the threshold-front result: event phase-slope median 2.57e-6/s vs 1.48e-6/s, p=0.0175, and q70 bootstrap phase slope 2.53e-6/s vs 1.44e-6/s, p=0.0202.
- After protocol/echem residualization, the strongest surviving event/control effect is the sign consistency of the phase front: phase-slope positive-fraction residual +0.06396, p=0.000825, with the matching negative-fraction residual -0.06396, p=0.000825.
- Magnitude-based phase-front residuals weaken after conditioning: threshold-robust phase score residual p=0.393, phase-slope median residual p=0.563, and absolute phase-slope residual p=0.833.
- Diffusion-related residuals remain non-significant: diffusion proxy median residual p=0.468, absolute diffusion proxy residual p=0.949, and threshold-robust diffusion score residual p=0.993.
- Leave-event-reference-out logistic classification using only residualized front features is poor (mean ROC-AUC 0.453, balanced accuracy 0.312), so these front residuals should not be used as a standalone detector.

Interpretation: protocol/echem conditioning sharpens the mechanistic guardrail. Event ROIs are more likely to retain a consistently positive bright-front trend after adjusting for acquisition/protocol context, but the magnitude of phase motion and all diffusion-like proxies are largely explained by context or are too noisy in the current automatic ROI set. The useful physics claim remains robust phase-front directionality and optical-state movement, not calibrated diffusion or standalone front-based event detection.

## 2026-05-21 ROI Front QC Package

Added and ran:

`python scripts/tier4_roi_front_qc_package.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_front_qc_package --top-n 24`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_front_qc_package`

Local compact copy:

`derived_local/roi_front_qc_package`

Key result:

- Built a compact manual-review package for 24 high-priority particle ROI/front candidates selected from rollout/mobility difficulty, threshold-robust phase/front rankings, diffusion-proxy magnitude, and protocol-conditioned phase-front sign residuals.
- The corrected package includes 18 event ROIs and 6 controls, including the active control `cycle62_rank3_obj9`, plus event candidates from cycles 60, 86, 116, and 156.
- For each ROI, generated a QC panel with first/middle/last/delta ROI frames, q70 bright-front contour overlays, and the central ROI guard mask; wrote `roi_front_qc_index.html` and `roi_front_qc_manifest.csv` with `manual_qc_status=pending`.
- Automatic review flags show why manual QC is necessary: 17/24 candidates have fragmented q70 masks, 6/24 have q70 radius-slope bootstrap confidence intervals crossing zero, 1/24 is explicitly flagged as an active control, and only 5/24 have no automatic review flags.
- Top review targets include `cycle60_rank3_obj9`, active control `cycle62_rank3_obj9`, `cycle156_rank6_obj3`, `cycle88_rank4_obj8`, `cycle60_rank2_obj2`, `cycle156_rank5_obj4`, `cycle156_rank2_obj2`, `cycle156_rank7_obj27`, `cycle116_rank1_obj6`, and `cycle86_rank4_obj9`.
- Updated the tier4 synthesis report so the completion audit now points to the QC package as the next manual validation artifact.

Interpretation: this closes a practical review gap. The project now has a concrete, compact set of particle ROI/front panels for manual accept/reject labeling, rather than only CSV rankings. The high rate of fragmented masks reinforces the current guardrail: threshold-front and diffusion proxies are useful for hypothesis ranking, but calibrated diffusion or final degradation-mode labels require manual QC on these panels.

## 2026-05-21 QC Review Packet

Added and ran:

`python scripts/tier4_qc_review_packet.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/qc_review_packet`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/qc_review_packet`

Local compact copy:

`derived_local/qc_review_packet`

Key result:

- Built a review-ready manifest of 30 prioritized ROI/front candidates using rollout-mobility difficulty, threshold-robust phase/diffusion proxy scores, protocol-conditioned front-direction residuals, and protocol-conditioned optical-shift residuals.
- The packet includes 16 event candidates and 14 control candidates, with pending columns for `manual_qc_status`, `manual_qc_decision`, and `manual_qc_notes`.
- Top review targets include `cycle60_rank3_obj9`, `cycle156_rank7_obj27`, `cycle156_rank2_obj2`, `cycle60_rank2_obj2`, `cycle156_rank1_obj1`, and the active control `cycle62_rank3_obj9`.
- The manifest attaches available Isambard ROI preview and rollout-preview paths so manual inspection can be done without searching through derived folders.
- Updated the tier4 synthesis to mention this QC packet as the next concrete guardrail before publication-scale front/diffusion claims.

Interpretation: this does not replace manual QC, but it closes the workflow gap by turning the automatic rankings into an actionable review queue. The next scientific claim boundary is clear: accept/reject these particle/front masks before interpreting apparent diffusion proxies as physical transport estimates.

## 2026-05-21 Residual Physics Mode Taxonomy

Added and ran:

`python scripts/tier4_residual_physics_mode_taxonomy.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/residual_physics_mode_taxonomy`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/residual_physics_mode_taxonomy`

Local compact copy:

`derived_local/residual_physics_mode_taxonomy`

Key result:

- Built a protocol-adjusted unsupervised taxonomy from 52 automatic ROI rows using optical residuals, latent/rollout difficulty, protocol-conditioned front residuals, and QC-priority context.
- Model selection chose k=4 with a modest silhouette of 0.204, but the cluster assignments are reasonably seed-stable across 20 KMeans reruns (mean adjusted Rand index 0.935, minimum 0.612).
- Mode names are assigned from standardized centroid deviations rather than raw residual signs, preventing tiny positive residuals from receiving strong mechanism labels.
- The strongest event-enriched mode is `optical_brightening_decorrelating_rollout_hard_front_positive`: 13 ROIs, 11 events, event fraction 0.846, Fisher p=0.00282. Its top cycles are 60, 156, 62, and 86.
- Other modes are `optical_loss_rollout_hard` (5 ROIs, event fraction 0.40), `near_baseline_or_context_like` (24 ROIs, event fraction 0.33), and `front_negative_high_apparent_front_proxy` (10 ROIs, event fraction 0.30).
- Top residual-mode review targets include `cycle60_rank3_obj9`, `cycle60_rank6_obj26`, `cycle60_rank4_obj5`, active control `cycle62_rank3_obj9`, `cycle156_rank7_obj27`, `cycle157_rank2_obj2`, `cycle156_rank5_obj4`, `cycle62_rank4_obj1`, `cycle60_rank2_obj2`, and `cycle58_rank3_obj9`.
- Updated the tier4 synthesis to include this taxonomy in the degradation-mode audit and machine-readable summary.

Interpretation: the current mode taxonomy points to a reproducible event-enriched pattern combining protocol-adjusted optical brightening, decorrelation, rollout difficulty, and positive front-direction residuals. This is useful for manual review and benchmark labeling, but the labels remain computational hypotheses from automatic particle/front candidates until QC decisions are recorded.

## 2026-05-21 Front QC Sensitivity Result

Added and ran:

`python scripts/tier4_front_qc_sensitivity.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/front_qc_sensitivity`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/front_qc_sensitivity`

Local compact copy:

`derived_local/front_qc_sensitivity`

Key result:

- Tested whether threshold-front and protocol-conditioned front effects survive automatic quality strata: complete threshold sweeps, high front-quality score, q70 phase-slope bootstrap CI excluding zero, q70 positive-CI fronts, and the 24-ROI review-panel subset.
- All 52 ROIs had complete threshold sweeps. The review-panel subset has 24 ROIs, but removing fragmented q70 masks leaves 7 event ROIs and 0 controls, so review-panel no-fragment/no-flag strata cannot estimate event/control separation yet.
- The protocol-conditioned phase-slope positive-fraction residual is robust in five strata: all front ROIs, complete threshold sweep, q70 phase CI excluding zero, q70 phase CI positive, and review-panel selected.
- All-front phase-sign residual: median event-control +0.0475, bootstrap 5th percentile +0.0342, Mann-Whitney p=0.000825, permutation median-difference p=0.0470.
- q70 phase-CI-excluding-zero stratum: 18 events and 12 controls; phase-sign residual median event-control +0.0885, bootstrap 5th percentile +0.0475, Mann-Whitney p=0.000106, permutation p=0.0010.
- q70 positive-CI stratum: 18 events and 8 controls; phase-sign residual median event-control +0.0604, bootstrap 5th percentile +0.0273, Mann-Whitney p=0.00209, permutation p=0.0150.
- High-front-quality strata preserve the positive direction but have sparse controls and do not pass the bootstrap-positive criterion.
- Diffusion-proxy effects remain unstable: absolute diffusion proxy is not significant across all ROIs or q70 phase-CI strata. The review-panel subset shows lower event absolute diffusion proxy than controls, but this subset is selection-biased and has only 6 controls.
- Updated the tier4 synthesis to include the QC-sensitivity result as an explicit guardrail around phase-front and diffusion claims.

Interpretation: automatic QC filtering strengthens the confidence that event ROIs have more consistent positive phase-front directionality after protocol/echem conditioning. It does not validate calibrated diffusion coefficients. The control imbalance after removing fragmented review masks shows why manual QC must include accepted controls, not only event-looking candidates.

## 2026-05-21 Cycle/Region Residual Mode Context

Added and ran:

`python scripts/tier4_cycle_region_mode_context.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_region_mode_context`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_region_mode_context`

Local compact copy:

`derived_local/cycle_region_mode_context`

Key result:

- Mapped the residual physics mode assignments across 11 cycles and 7 coarse image-region bins using the ROI approximate object coordinates from `multi_cycle_roi_echem_joined.csv`.
- The event-enriched residual mode `optical_brightening_decorrelating_rollout_hard_front_positive` covers 25.0% of the 52 ROI cohort, but is concentrated in cycle 60 (5/6 ROIs, fraction 0.833) and cycle 156 (4/6 ROIs, fraction 0.667).
- The strongest coarse spatial region for this mode is `x2_y3` with 15 ROIs, event fraction 0.600, and event-enriched-mode fraction 0.467; the right-side `x3` region is depleted for this mode by Fisher level-vs-rest testing (fraction 0.056 vs 0.353 outside, p=0.0210).
- Context correlations show the residual-mode review priority still tracks acquisition/protocol axes: `n_frames_percentile` rho=0.716 (p=2.46e-09), `V_mean` rho=-0.503 (p=0.00130), and cycles-to-block-end rho=0.378 (p=0.00569).
- The event-enriched-mode indicator itself is weaker but still frame-count associated: rho=0.343 with `n_frames_percentile` (p=0.0127).
- A context-only leave-cycle-out logistic probe is unstable: mean fold ROC-AUC 0.711 but pooled ROC-AUC 0.429 and pooled balanced accuracy 0.433 over scored folds. This is evidence against using cycle/region context alone as a detector.
- Updated the tier4 synthesis to include this cycle/region context as a bridge between residual degradation modes, cycle dependence, and coarse particle location.

Interpretation: the residual mode taxonomy is not uniformly distributed over cycles or particle regions. It highlights cycles 60/156 and the coarse `x2_y3` image region as high-priority manual-review strata, while also exposing remaining protocol/acquisition coupling. This moves the project closer to cycle-region degradation mapping, but it remains a QC-prioritization and hypothesis-generation layer rather than proof of spatial degradation mechanism.

## 2026-05-21 Control-Balanced Front QC Package

Added and ran:

`python scripts/tier4_control_balanced_front_qc_package.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_qc_package --n-per-role 16 --n-extra-controls 8`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_qc_package`

Local compact copy:

`derived_local/control_balanced_front_qc_package`

Key result:

- Built a separate visual QC augmentation package to address the front-QC sensitivity gap where strict no-fragment/no-flag review strata had no controls.
- The package contains 40 ROI/front panels: 16 event ROIs and 24 control ROIs, with 22 ROIs not present in the primary QC package.
- It preserves the same panel format as the primary package: first/middle/last/delta ROI frames, q70 bright-front contours, and central ROI guard masks, plus an HTML review index and manifest with pending manual QC columns.
- Automatic flags still show substantial mask risk: 30/40 candidates have fragmented q70 masks, 13/40 have diffusion CI crossing zero, 3/40 are active controls, and 3/40 have threshold-sign instability.
- The augmentation now provides non-fragmented examples in both roles: 4 controls and 6 events. No-auto-flag examples are still control-only (2 controls), so final clean event/control diffusion inference still needs manual acceptance labels.
- Updated the tier4 synthesis to include the control-balanced QC package as the recommended manual review companion to the high-signal primary QC package.

Interpretation: this closes a practical review-design gap rather than a scientific claim gap. We now have enough control-heavy visual panels to manually assess whether front-mask quality artifacts explain the apparent diffusion/control effects. It supports better manual QC and benchmark construction, but calibrated diffusion and final degradation labels remain guarded until those panels are reviewed.

## 2026-05-21 Prefix ROI Forecast Result

Added and ran:

`python scripts/tier4_prefix_roi_forecast.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_forecast --n-permutation 500`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_forecast`

Local compact copy:

`derived_local/prefix_roi_forecast`

Key result:

- Built prefix-only features from cropped particle-region ROI videos using only the first 25%, 50%, or 75% of each ROI sequence. Features include early intensity slopes, high/low fraction changes, temporal difference energy, frame delta summaries, and stage drift proxies.
- Evaluated leave-event-reference-cycle-out classifiers and regressors for event labels, residual mode labels, and later protocol-conditioned front-direction residual outcomes.
- The clearest early-video signal is predicting whether the later protocol-conditioned phase-slope positive-fraction residual is positive. Prefix-only random forest at 50% prefix gives mean ROC-AUC 0.726; prefix-only logistic at 75% gives AUC 0.687 and balanced accuracy 0.608.
- The front-positive residual logistic 75% prefix result passes a 500-shuffle permutation null: observed AUC 0.687, null mean 0.492, null p95 0.656, empirical p=0.0259.
- Event-label prediction is less reliable under the same null: best audited prefix-only logistic AUC 0.573, empirical p=0.285. Event-enriched residual mode prediction is suggestive but not significant: AUC 0.636, empirical p=0.184.
- Prefix features also predict residual-mode review priority better than a median baseline: prefix-only random forest at 25% prefix has mean MAE ratio 0.429 and mean Spearman rho 0.322; prefix-plus-context random forest at 50% has MAE ratio 0.419 and rho 0.374.
- Continuous phase-front residual regression remains weak; classification of front-direction sign is more stable than predicting residual magnitude.

Interpretation: early particle-region ROI video contains useful information about later front-direction behavior, not just final-cycle labels. This supports using prefix/past-observable models as physics-signal triage for front motion, while keeping the current guardrail that the selected 52-ROI cohort is too small and event-centered for deployable forecasting.
