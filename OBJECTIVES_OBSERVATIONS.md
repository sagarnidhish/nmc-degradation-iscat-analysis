# Objectives And Observations

Last updated: 2026-05-22

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

## 2026-05-22 Agentic AI Paper Implementation Review

Reviewed the three May 2026 Nature agentic-AI papers linked by the user:

- Ghareeb et al., "A multi-agent system for automating scientific discovery", DOI `10.1038/s41586-026-10652-y`.
- Gottweis et al., "Accelerating scientific discovery with Co-Scientist", DOI `10.1038/s41586-026-10644-y`.
- Aygun et al., "An AI system to help scientists write expert-level empirical software", DOI `10.1038/s41586-026-10658-6`.

Added a project-specific implementation plan under:

`derived_local/literature_agentic_ai_implementation_plan`

Key conclusion:

- The directly useful techniques are a closed-loop hypothesis ledger, Co-Scientist-style skeptical hypothesis tournament, Robin-style lab-in-the-loop feedback, and ERA-style metric search over analysis code/feature variants.
- The highest-priority implementable module is `tier4_agentic_metric_search.py`, which should search over feature families, source/cycle splits, residualization controls, and weak-label targets using a predeclared scientific score.
- The key guardrail is unchanged: agentic systems may rank and test hypotheses, but should not emit validated battery mechanism claims without source-balanced controls, manual ROI/front QC, and electrochemical consistency checks.

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

## 2026-05-21 Manual QC Label Workbook Result

Added and ran:

`python scripts/tier4_manual_qc_label_workbook.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/manual_qc_label_workbook`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/manual_qc_label_workbook`

Local compact copy:

`derived_local/manual_qc_label_workbook`

Key result:

- Built a single deduplicated manual-QC label template from the primary visual front-QC package, the control-balanced front-QC package, and the ranked QC review packet.
- The workbook contains 47 unique ROI/front candidates: 24 controls and 23 events.
- Priority tiers are 12 high, 17 medium, and 18 routine candidates.
- Source coverage is 40 control-balanced visual-QC rows, 24 primary visual-QC rows, and 30 ranked review-packet rows.
- All 47 labels remain `manual_qc_status=pending`; the script deliberately does not assign accept/reject decisions.
- The label template adds columns for `manual_particle_identity_ok`, `manual_front_mask_ok`, `manual_diffusion_interpretable`, reviewer, and review date so accepted fronts can later be joined back to diffusion/front analyses without hand-merging manifests.

Interpretation: this closes the label-provenance gap between the automatic analyses and eventual manual review. It does not validate diffusion or degradation labels, but it creates the authoritative table needed to convert visual QC decisions into a reproducible accepted-front subset.

## 2026-05-21 Control-Balanced Front QC Sensitivity V2

Added and ran:

`python scripts/tier4_control_balanced_front_qc_sensitivity.py --n-bootstrap 1000 --n-permutation 1000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_qc_sensitivity`

Local compact copy:

`derived_local/control_balanced_front_qc_sensitivity`

Key result:

- Added a second front-QC sensitivity analysis that compares the original high-signal QC package with the control-balanced augmentation.
- The all-front result is unchanged: phase-slope positive-fraction protocol residual remains event-shifted by median event-control 0.047, bootstrap p05 0.034, Mann-Whitney p=0.000825, and permutation p=0.03696.
- The control-balanced selected review set has usable event/control composition (16 event / 24 control) and preserves the phase-sign residual separation: median event-control 0.091, bootstrap p05 0.047, Mann-Whitney p=0.00288, permutation p=0.00799.
- The balanced non-fragmented subset now contains both roles (6 event / 4 control). It keeps the expected direction for phase-sign residual (median event-control 0.083, permutation p=0.04595), but the bootstrap p05 is slightly negative and the subset is too small for a strong claim.
- The original strict non-fragmented and no-auto-flag subsets are event-only (7/0 and 5/0), so they cannot support event/control testing. This confirms that the control-balanced review design is necessary before manual QC labels can be used for final filtering.
- Diffusion-proxy effects remain unstable and selection-sensitive. In the balanced selected set, diffusion-proxy residual median event-control is near zero and non-significant (Mann-Whitney p=0.901, permutation p=0.920).

Interpretation: the phase-front directionality finding survives the control-balanced automatic review panel, which reduces concern that it is only an event-heavy QC selection artifact. The strictest balanced automatic subset is directionally consistent but underpowered. Diffusion remains guarded and should not be interpreted as calibrated Li diffusivity without manual QC and calibration.

## 2026-05-21 Manual-QC Gated Front Effects Result

Added and ran:

`python scripts/tier4_manual_qc_gated_front_effects.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/manual_qc_gated_front_effects --n-bootstrap 1000 --n-permutation 1000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/manual_qc_gated_front_effects`

Local compact copy:

`derived_local/manual_qc_gated_front_effects`

Key result:

- Added a conservative downstream gate that joins the manual-QC label workbook to the 52 automatic threshold-front ROI rows and recomputes front/diffusion statistics only for manually accepted fronts.
- Current status is `ready_for_manual_labels`: 0 manually accepted front-effect rows and 0 diffusion-interpretable rows.
- The joined gate table reports 47 pending workbook labels plus 5 automatic front rows missing from the label workbook; all 52 rows are pending or not accepted.
- The effect-test CSV is intentionally empty apart from headers because no row currently satisfies `manual_qc_decision=accept`, `manual_particle_identity_ok=yes`, and `manual_front_mask_ok=yes`.
- A pending review queue is written so the high-priority event/control rows can be reviewed without rebuilding the automatic analysis.

Interpretation: this creates the hard reproducibility gate for final front/diffusion claims. Automatic analyses can continue to guide review, but publication-facing front-effect and diffusion statistics now require explicit manual acceptance labels before the gated script will emit tests.

## 2026-05-21 Prefix ROI Feature-Importance Audit

Added and ran:

`python scripts/tier4_prefix_roi_feature_importance.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_feature_importance --n-permutation-repeats 50 --n-null 300`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_feature_importance`

Local compact copy:

`derived_local/prefix_roi_feature_importance`

Key result:

- Reran the prefix-forecast table with a stricter feature guardrail that excludes raw `frame_index` features from the model feature set, then reran the feature-importance audit on the guarded prefix table.
- The guarded prefix-only forecast still ranks the front-direction residual class highest: random forest at 50% prefix gives mean AUC 0.691, but the audited front-positive residual permutation null is no longer significant (observed AUC 0.476, null p95 0.668, empirical p=0.515).
- Interpreted the 75% prefix logistic model for the later `front_positive_residual_binary` target using leave-event-reference-cycle-out predictions, permutation feature importance, feature-group ablation, coefficient summaries, univariate tests, and a 300-shuffle label null.
- The audit used 52 ROI rows and 54 prefix features grouped into mean-intensity trace, bright/dark fractions, temporal change energy, frame/texture level, stage drift, and other prefix features.
- The held-out pooled readout is not independently significant: pooled OOF AUC 0.447, balanced accuracy 0.489, null mean AUC 0.500, null p95 AUC 0.638, empirical p=0.714.
- Descriptively, removing `mean_intensity_trace` causes the largest AUC drop (0.171). The remaining group ablations are near-zero or negative, including `frame_texture_level` (-0.002), `stage_drift` (-0.023), `temporal_change_energy` (-0.048), and `bright_dark_fraction` (-0.074). Top permutation features include `stage_prefix_step_max`, `low_fraction_prefix_std`, `average_intensity_prefix_mean`, `average_intensity_prefix_last`, `roi_norm_mean_prefix_first`, and `first_q30_threshold`.
- Univariate checks flag `stage_prefix_step_max`, `stage_prefix_net_drift`, and `prefix_temporal_absdiff_p95`, but these should be interpreted as acquisition/physics-triage signals rather than causal mechanisms.
- The project synthesis now includes a dedicated Prefix ROI Feature Importance section and carries the full summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: the earlier prefix-forecast signal remains useful for triage, but this 75% logistic interpretability audit is mainly a guardrail. It points mainly to early mean-intensity trace features as descriptive contributors while showing that the pooled importance model does not clear a permutation-null test on the selected 52-ROI cohort.

## 2026-05-21 Leakage-Clean Prefix Forecast Revision

Updated and reran:

`python scripts/tier4_prefix_roi_forecast.py --n-permutation 300`

Also updated and reran:

`python scripts/tier4_prefix_roi_feature_importance.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_feature_importance --n-permutation-repeats 50 --n-null 300`

Remote output directories:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_forecast`

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_feature_importance`

Local compact copies:

`derived_local/prefix_roi_forecast`

`derived_local/prefix_roi_feature_importance`

Key result:

- Found and fixed an acquisition-leakage issue in prefix feature selection: `first_frame_index` and `last_frame_index` existed in the prefix feature table, and the old selector could include `first_frame_index` because it matched `first_*`.
- Tightened `tier4_prefix_roi_forecast.py` and `tier4_prefix_roi_feature_importance.py` so strict prefix models exclude frame-index columns and use only visual ROI-prefix features plus the explicitly named context set when requested.
- After removing frame-index leakage, the best front-positive residual classifier is still prefix-only random forest at 50% prefix, but weaker: mean fold ROC-AUC 0.691 and balanced accuracy 0.472. The audited prefix-only logistic null for the same target is no longer significant: observed AUC 0.476, null p95 0.668, empirical p=0.515.
- Event-label and residual-mode prefix-only logistic nulls also remain non-significant after the cleanup: event-label observed AUC 0.563, empirical p=0.292; event-enriched-mode observed AUC 0.630, empirical p=0.189.
- The strict visual-feature importance audit for the 75% logistic front-positive target gives pooled OOF AUC 0.447, balanced accuracy 0.489, null p95 AUC 0.638, empirical p=0.714. Mean-intensity trace features have the largest descriptive group-ablation drop, but the model does not clear the permutation null.
- This revises the previous optimistic prefix interpretation: early particle-region video features may still triage front-direction behavior, but current evidence does not prove a robust leakage-clean predictor on the 52-ROI cohort.

Interpretation: this is a useful negative/control result. It prevents an acquisition index from masquerading as image physics and narrows the credible claim to descriptive early-ROI intensity/texture associations. The next prefix-model step should use stricter leakage controls, more ROI rows, and preferably manual-QC-accepted particle/front labels before claiming deployable forecasting.

## 2026-05-21 Spatiotemporal Degradation Graph Result

Added and ran:

`python scripts/tier4_spatiotemporal_degradation_graph.py --n-permutation 1000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/spatiotemporal_degradation_graph`

Local compact copy:

`derived_local/spatiotemporal_degradation_graph`

Key result:

- Built a directed nearest-neighbor graph over the 52 multi-cycle ROI nodes using approximate particle coordinates, cycle number, and event-reference-cycle blocks. The graph contains 510 directed edges across same-cycle, same-reference, previous-cycle, and next-cycle neighbor definitions.
- Same-cycle spatial neighbors show strong front-direction homophily: `front_positive_residual_binary` same-label fraction 0.936 versus within-reference permutation null mean 0.489, empirical p=0.000999. Same-cycle event-enriched residual modes also cluster: same-label fraction 0.769 versus null mean 0.647, empirical p=0.00599.
- Continuous front-direction residuals are spatially correlated within cycle: `phase_slope_positive_fraction_protocol_residual` neighbor rho 0.696 versus null p95 0.197, empirical p=0.000999.
- A same-reference, not necessarily same-cycle, threshold-robust phase residual correlation remains visible: rho 0.275 versus null p95 0.096, empirical p=0.002997.
- Cross-cycle nearest-neighbor labels do not show simple propagation. Previous-cycle nearest neighbor AUC for current front-positive residual is 0.426, for event-enriched mode is 0.441, and for event label is 0.167. The next-cycle continuous front residual relation is negative (rho -0.219, empirical p=0.02398), consistent with local cycling/reversal or cohort design rather than monotonic spread.
- Distance-gradient checks flag that event-event edges are farther than other same-reference edges, but this is a cohort-design artifact because event/control ROIs are selected in separate cycles. Event-enriched mode positive-positive edges do not have a shorter median distance than other edges (97.5 px versus 103.7 px, p=0.672).

Interpretation: front-direction and residual-mode signals are spatially organized within individual cycles, which supports using local particle-region context for manual review and physics triage. The graph does not prove degradation propagation across cycles; cross-cycle nearest-neighbor behavior is weak or reversed and remains confounded by event/control cycle selection.

## 2026-05-21 Phase Kinetics Avrami Audit

Added and ran:

`python scripts/tier4_phase_kinetics_avrami.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/phase_kinetics_avrami`

Local compact copy:

`derived_local/phase_kinetics_avrami`

Key result:

- Extracted optical phase-fraction trajectories from each cropped particle ROI movie at q60/q70/q80 thresholds and fit descriptive logistic and Avrami-style transformed-fraction summaries.
- The corrected feature list uses only newly generated kinetic features, avoiding accidental reuse of older ROI descriptor columns.
- Event-enriched residual-mode ROIs show faster/stronger optical transformations than other modes: q60 logistic rate median difference 2.66e-04 1/s (p=0.0137), q70 transformed-fraction delta +0.00632 (p=0.0169), and q80 transformed-fraction delta +0.00566 (p=0.0384).
- Event ROIs have better simple Avrami fit quality than controls at q70 and q60: q70 Avrami R2 median difference +0.0452 (p=0.0223), q60 Avrami R2 +0.0253 (p=0.0283). Event q70/q80 maximum-rate timing occurs earlier than controls (q70 timing fraction -0.100, p=0.0362; q80 -0.132, p=0.0239).
- The strongest global correlations are still acquisition/protocol-linked: q70 logistic R2 vs frame-count percentile rho=-0.801, q60 logistic R2 rho=-0.776, ROI norm rate sign consistency rho=0.728, and q80 max-rate rho=0.699. This is a major guardrail for interpreting kinetic parameters.

Interpretation: the kinetic audit adds a physics-shaped layer beyond raw front slopes: event-enriched modes look like sharper/brighter optical phase transformations with earlier transition timing. These are optical transformation proxies, not calibrated reaction constants or diffusion coefficients, and frame-count/protocol coupling must be controlled before using them as mechanistic material parameters.

## 2026-05-21 Particle Trace Physics Audit

Added and ran:

`python scripts/tier4_particle_trace_physics_audit.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_trace_physics_audit --n-permutation 500`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_trace_physics_audit`

Local compact copy:

`derived_local/particle_trace_physics_audit`

Key result:

- Built a cycle-level physics audit over the larger normalized four-particle intensity table rather than the 52 cropped ROI/video cohort.
- The audit covers 89 cycle rows from cycle 2 to 158, with 4 any-drop cycles: 60, 86, 116, and 156. Cycles 86 and 116 are synchronized 3-particle drops; cycles 60 and 156 are single-particle abrupt drops.
- Unsupervised trace-state clustering selects k=2 with silhouette 0.266. This is a coarse state split, not a definitive degradation taxonomy.
- Abrupt-drop cycles have much more negative mean particle-intensity step changes than non-drop cycles: median positive-negative `mean_delta_prev` = -0.116, Mann-Whitney p=1.03e-05. They also have larger `max_abs_delta_prev` (+0.162, p=0.00133) and `mean_abs_delta_prev` (+0.0894, p=0.00275).
- A leakage-conscious leave-cycle-block-out classifier predicts future any-drop cycles within 8 cycles with mean AUC 0.883 and balanced accuracy 0.648. A 500-shuffle null gives null p95 AUC 0.679 and empirical p=0.001996.
- The analogous future synchronized 2+ drop classifier has mean AUC 0.827 and balanced accuracy 0.828, with null p95 AUC 0.744 and empirical p=0.00998, but this target has only two positive synchronized-drop cycles.
- Echem/trace coupling is strong at the cycle level: Vmax correlates with particle norm mean (rho 0.626, p=4.14e-10), particle norm range (rho 0.475, p=7.29e-06), and particle norm std (rho 0.459, p=1.66e-05); capacity correlates with particle norm mean (rho 0.454, p=2.08e-05).
- The project synthesis now includes a Particle Trace Physics Audit section and carries the audit into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this is the first stronger early-warning result from the broader particle trace table rather than only selected ROI videos. It supports the idea that particle-level photometry dynamics before an event carry predictive information about upcoming abrupt degradation, but it cannot localize phase boundaries, infer diffusion, or replace manual ROI/front QC.

## 2026-05-21 Particle Event Precursor Atlas

Added and ran:

`python scripts/tier4_particle_event_precursor_atlas.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_precursor_atlas`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_precursor_atlas`

Local compact copy:

`derived_local/particle_event_precursor_atlas`

Key result:

- Built event-aligned precursor windows from the normalized four-particle cycle table using the four abrupt-drop event anchors: cycles 60, 86, 116, and 156.
- Selected 17 candidate non-event control anchors outside a +/-16 cycle exclusion zone around events, then formed 24 matched control anchors using cycle, frame-count, capacity, and Vmax context.
- The strongest pre-event precursor window is lower capacity 5-8 cycles before an event: `pre8_to_pre5 capacity_mAh min_value` median event-control -0.0238 mAh, Mann-Whitney p=0.00171.
- Longer-range pre-event echem degradation is visible 9-16 cycles before events: `coulombic_efficiency_pct` minimum is lower by -1.11 percentage points (p=0.00194) and mean CE is lower by -0.421 percentage points (p=0.00223).
- Cross-particle trace dispersion is elevated before events: `pre16_to_pre9 delta_std_across_particles max_value` is higher by +0.0396 (p=0.00665), consistent with increasingly heterogeneous particle response before abrupt degradation.
- Some significant precursor terms are acquisition/protocol guardrails rather than material physics: pre16-to-pre9 `n_frames` slope is lower in event windows (event-control -2.18 frames/cycle, p=0.0118) and frame-count percentile slope is also lower (p=0.0212).
- At the event cycle, the atlas recovers the expected abrupt-change signature: `max_abs_delta_prev` event-control +0.136 and `mean_delta_prev` event-control -0.109, both p~0.0017.
- The project synthesis now includes a Particle Event Precursor Atlas section and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this turns the broader cycle-level early-warning classifier into a more physical timeline. Events are preceded by weaker echem state, stronger cross-particle photometry heterogeneity, and then the abrupt intensity drop itself. Because there are only four event anchors and acquisition terms remain significant, this is a review-prioritization and hypothesis-generation result, not a calibrated degradation-propagation model.

## 2026-05-21 ROI Trace Fusion Audit

Added and ran:

`python scripts/tier4_roi_trace_fusion_audit.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_trace_fusion_audit`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_trace_fusion_audit`

Local compact copy:

`derived_local/roi_trace_fusion_audit`

Key result:

- Joined cycle-level four-particle/echem trace state onto the 52 ROI/front/kinetic residual rows using lags of 0, 2, 4, 8, and 16 cycles.
- The full audit correctly recovers lag-0 event-label sanity checks, but the report headline excludes lag-0/event-flag terms when interpreting precursor evidence.
- Among non-lag-0 context-residual associations, two-cycle-prior cross-particle intensity spread is strongly associated with front-positive residual fraction: `trace_lag2_particle_norm_range` and `trace_lag2_particle_norm_std` both have rho=0.725, p=2.69e-07, n=38 after residualizing against cycle number, frame-count percentile, Vmean, and mean current.
- Four-cycle-prior `delta_std_across_particles` is inversely associated with front-positive residual fraction after the same context residualization: rho=-0.705, p=7.72e-07, n=38.
- Event-enriched residual-mode ROIs differ in several lagged precursor/context terms, including lower lag16 cross-particle delta dispersion (median diff -0.00726, p=0.00491), higher lag8 frame-count percentile and frame count (both p=0.00630), and lower lag16 CE (median diff -0.442 percentage points, p=0.0169).

Interpretation: this links global particle-trace precursor state to localized ROI/front behavior more directly than the standalone cycle audit. The strongest front-direction association is with prior cross-particle trace heterogeneity, but ROI rows are clustered by cycle/reference and automatic ROI labels remain pending manual QC, so this is linkage and prioritization evidence rather than causal proof.

## 2026-05-21 Precursor-Informed ROI Review Manifest

Added and ran:

`python scripts/tier4_precursor_informed_roi_review.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_informed_roi_review`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_informed_roi_review`

Local compact copy:

`derived_local/precursor_informed_roi_review`

Key result:

- Built a review-prioritization manifest over 47 pending manual-QC candidates by combining the existing manual-QC workbook priority, event precursor severity, residual-mode priority, front-direction residual magnitude, robust phase score magnitude, kinetic rate magnitude, and event/pre-event cycle bonuses.
- Tiered the candidates into 12 high, 18 medium, and 17 routine review priorities.
- Event-cycle candidates dominate the top tier, especially cycle 156 and cycle 60 ROI rows; the top candidate is `cycle156_rank7_obj27`, followed by `cycle60_rank5_obj18`, `cycle156_rank6_obj3`, `cycle156_rank5_obj4`, and `cycle156_rank2_obj2`.
- The precursor context scores rank event-reference-cycle severity as cycle 156 highest, then 86, 60, and 116 under the selected event-window tests.
- The manifest does not assign manual labels. It gives the next practical inspection order for particle identity, front-mask validity, and diffusion/front interpretability.

Interpretation: this converts the many automatic analyses into a concrete manual-review queue. The highest-value next human/visual QC work is not arbitrary: inspect top-tier precursor-informed candidates first, with particular attention to event-cycle ROIs that combine high precursor severity, event-enriched residual modes, and large front-direction residuals.

## 2026-05-21 Precursor Review Visual Bundle

Added and ran:

`python scripts/tier4_precursor_review_visual_bundle.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_review_visual_bundle --top-n 12`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_review_visual_bundle`

Local compact copy:

`derived_local/precursor_review_visual_bundle`

Key result:

- Packaged the top 12 precursor-informed ROI review candidates into a visual inspection bundle.
- All 12 ranked candidates have at least one copied automatic QC/preview asset on Isambard.
- Created a contact sheet at `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_review_visual_bundle/top_candidate_contact_sheet.png`; the compact contact sheet was synced locally.
- The bundle starts with the same highest-ranked candidates: `cycle156_rank7_obj27`, `cycle60_rank5_obj18`, `cycle156_rank6_obj3`, `cycle156_rank5_obj4`, and `cycle156_rank2_obj2`.
- Per-candidate asset folders remain on Isambard under `precursor_review_visual_bundle/assets`; only the compact index, summary, README, and contact sheet are copied locally.

Interpretation: this does not solve manual QC, but it removes the path-chasing friction. The next human or automated visual review can start from a single ranked contact sheet and per-ROI asset folders while keeping labels, particle identity decisions, and diffusion/front interpretability explicitly unassigned.

## 2026-05-21 ROI Trace Fusion Cycle Null

Added and ran:

`python scripts/tier4_roi_trace_fusion_cycle_null.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_trace_fusion_cycle_null --n-permutation 1000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_trace_fusion_cycle_null`

Local compact copy:

`derived_local/roi_trace_fusion_cycle_null`

Key result:

- Collapsed the 52 ROI trace-fusion rows to 11 median cycle points before retesting lagged trace/front associations with 1,000 permutation shuffles per test.
- This is deliberately more conservative than the row-level fusion audit because it removes repeated ROI rows within the same cycle/reference context.
- The strongest cycle-collapsed association is `trace_lag16_frames_percentile` versus `mode_review_priority`: rho=0.813, empirical p=0.0020, n=11.
- Several surviving collapsed associations involve frame-count or frame-count percentile terms, including lag8 and lag16 `n_frames`/`frames_percentile` versus `mode_review_priority`, which reinforces acquisition/protocol context as a guardrail.
- A reference-centered variant still finds strong frame-count associations and one future-drop/front-residual association: `lag16_trace_predprob_future_any_drop_within_8cycles` versus `phase_slope_positive_fraction_protocol_residual` has rho=-0.809, empirical p=0.0070, n=11.
- The original row-level `trace_lag2_particle_norm_range/std` association with `phase_slope_positive_fraction_protocol_residual` is not the top cycle-collapsed result, so it should be treated as cycle-cluster-sensitive linkage evidence rather than a standalone local causal precursor.

Interpretation: this audit upgrades the trace-fusion story from a simple row-level association to a properly guarded result. Some lagged global trace/echem state still aligns with ROI/front/mode descriptors after cycle collapse, but the strongest surviving signals mix degradation-risk predictions, frame-count/protocol context, and mode/kinetic descriptors. This supports prioritization and hypothesis generation, not a calibrated propagation or diffusion claim.

## 2026-05-21 Within-Cycle Echem Shape Audit

Added and ran:

`python scripts/tier4_within_cycle_echem_shape_audit.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/within_cycle_echem_shape_audit`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/within_cycle_echem_shape_audit`

Local compact copy:

`derived_local/within_cycle_echem_shape_audit`

Key result:

- Scanned the raw `echemDF_full.csv` time/potential/current rows and computed within-cycle trajectory descriptors for observed particle/ROI cycles.
- Matched 81 echem shape cycles out of 89 requested observed cycles, producing 48 shape descriptors including voltage quantiles, current quantiles, integrated signed/absolute charge, dV/dt summaries, and voltage-binned dQ/dV proxy fractions/entropy.
- The echem shape descriptors strongly recover expected electrochemical structure: `shape_charge_mAh_neg_abs` correlates with `capacity_mAh` at rho=0.9999, and `echem_shape_duration_s` correlates with capacity at rho=0.985.
- ROI optical mode-review priority has strong associations with voltage/dQ-shape descriptors, led by `shape_V_q95` versus `mode_review_priority` (rho=-0.864, p=2.72e-12, n=38) and dQ/dV concentration terms such as `neg_dq_abs_peak_frac` (rho=0.841, p=3.81e-11, n=38).
- Direct abrupt-event cycle tests are weak with only three matched event cycles: best any-drop shape test is `neg_dq_abs_peak_voltage`, p=0.230, and `shape_dVdt_abs_p95`, p=0.255.
- ROI event/control binary shape tests are modest: `shape_V_std` separates event/control ROI rows (median diff +0.00579, p=0.00153), while most direct event-label shape terms are weaker.
- The project synthesis now includes a Within-Cycle Echem Shape Audit section and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: raw echem trajectory shape is a useful physics/protocol context layer for optical modes and kinetic/front descriptors, especially voltage-window and dQ/dV-proxy concentration. It does not independently explain abrupt degradation-event timing. Treat these descriptors as covariates and guardrails for ROI optical physics, not calibrated dQ/dV or diffusion constants.

## 2026-05-21 Calibration Metadata Audit

Added and ran a metadata-only audit using top-level HDF5 attributes/datasets, small CSV samples, and PPTX text extraction:

`python scripts/tier4_calibration_metadata_audit.py --base-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho --repo-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/calibration_metadata_audit`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/calibration_metadata_audit`

Local compact copy:

`derived_local/calibration_metadata_audit`

Key result:

- Scanned 33 HDF5 files from the discovered raw HDF5 set; 32 contain `movie` datasets and 32 contain `camera_timing`.
- No HDF5 file exposed calibration-like pixel-size or field-of-view attributes in top-level file/dataset/group metadata.
- HDF5 timing rows yield a median timing-derived FPS proxy of 0.0994, with a range from about 0.0050 to 0.0999 across files.
- That timing result is a metadata guardrail, not a true camera-cadence claim: some files have very short movie stacks or sparse timing, so `camera_timing` may encode sparse segment/cycle timing rather than every exposure.
- PPTX text extraction found slide-derived calibration context: `Degradation Paper Outline.pptx` slide 3 states 96 nm pixel size and 180x120 um field of view; slide 16 mentions a minimum viable facet-size/point-spread-function note; `Degradation Project.pptx` slide 3 mentions exposure time = 2 ms.
- The project synthesis now carries the calibration metadata audit into `nmc_ai_physics_synthesis_summary.json` and the synthesis markdown.

Interpretation: this improves the calibration evidence boundary. HDF5 timing metadata exists but must be interpreted cautiously for cadence; spatial calibration for the current um-scale front/diffusion proxies is still slide-derived, not raw-HDF5-confirmed. Diffusion values therefore remain apparent optical-front proxies until microscope metadata or manual provenance confirms both pixel size and the relevant timebase.

## 2026-05-21 Calibration Claim Risk Register

Added and ran:

`python scripts/tier4_calibration_claim_risk_register.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/calibration_claim_risk_register`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/calibration_claim_risk_register`

Local compact copy:

`derived_local/calibration_claim_risk_register`

Key result:

- Audited 11 front, mobility, kinetics, QC, and diffusion-like claim families; all 11 source tables are present.
- Calibration evidence status is now explicit: 32/33 scanned HDF5 files contain `camera_timing`, 0 scanned HDF5 files expose spatial calibration attributes, and 3 PPTX calibration text hits provide slide-derived context.
- The register classifies early front candidates and ROI/multi-cycle mobility as high-risk proxy-only or optical-front-proxy outputs.
- Threshold-robust fronts and calibrated-front QC are marked as apparent um-scale proxies: phase-slope sign/fraction trends are useful optical readouts, but diffusion-like values remain apparent `um^2/s` proxies.
- Protocol-conditioned front residuals and QC sensitivity tables are medium-risk guardrails, with the robust claim centered on phase-front direction residuals rather than diffusion magnitude.
- Manual-QC-gated front/diffusion effects remain publication-gate pending because there are no accepted manual labels yet.
- The project synthesis now includes a Calibration Claim Risk Register section and carries the risk-register summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this is a useful paper-writing and future-analysis guardrail. It prevents accidental overclaiming by separating strong optical proxy evidence from calibrated transport claims. Current diffusion-like values should continue to be described as apparent optical-front proxies until spatial calibration, true frame cadence, particle/front masks, and manual QC are jointly validated.

## 2026-05-21 Echem-Shape-Conditioned ROI/Front Effects

Added and ran:

`python scripts/tier4_echem_shape_conditioned_roi_front_effects.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_shape_conditioned_roi_front_effects`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_shape_conditioned_roi_front_effects`

Local compact copy:

`derived_local/echem_shape_conditioned_roi_front_effects`

Key result:

- Conditioned 17 ROI/front/kinetic targets on within-cycle echem shape using 45 numeric voltage/current/dQ-proxy descriptors compressed to 6 PCA components; those PCs explain 0.997 of the shape-feature variance.
- The audit uses 52 ROI rows, with 24 event and 28 control rows, and keeps event-reference-cycle fixed effects in the residual model.
- Shape PCs explain substantial variance in several targets: `mode_review_priority` 0.839, `first_last_corr_protocol_residual` 0.583, `q60_logistic_k_per_s` 0.544, and `q70_logistic_k_per_s` 0.515.
- The strongest shape-PC association is `echem_shape_pc3` versus `mode_review_priority`, rho=-0.707, p=4.52e-09, confirming that echem-shape context is a strong protocol/physics covariate for the optical mode ranking.
- After echem-shape conditioning, the front-direction residual remains the strongest event/control readout: `phase_slope_positive_fraction_protocol_residual_shape_residual` median event-control +0.0305, p=0.00375. This retains about 0.64 of the original median effect.
- High/low optical fraction residuals remain but weaken: `high_fraction_delta_protocol_residual_shape_residual` p=0.00799 and `low_fraction_delta_protocol_residual_shape_residual` p=0.0158.
- Shape conditioning makes q70/q80 transformed-fraction deltas more separable than their raw event/control tests in this cohort, suggesting echem-shape adjustment can sharpen optical kinetics contrasts.
- Several earlier ROI effects are largely absorbed by shape context: first-last correlation and latent net displacement no longer separate event/control after shape conditioning.
- Diffusion-like residuals remain non-significant after shape conditioning, consistent with the calibration risk register.
- Leave-event-reference-out logistic classification on shape-conditioned residuals is poor: mean ROC-AUC 0.469 and balanced accuracy 0.448.
- The project synthesis now includes this audit and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this is a stronger covariate test than the previous coarse protocol conditioning. Within-cycle echem trajectory shape explains a lot of optical mode/kinetic variance, but it does not fully explain the robust phase-front direction residual. The credible physics claim remains localized optical phase-front direction/morphology coupled to degradation context, not calibrated diffusion or a deployable detector.

## 2026-05-21 Physics Consistency Claim Matrix

Added and ran:

`python scripts/tier4_physics_consistency_claim_matrix.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/physics_consistency_claim_matrix --n-permutation 5000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/physics_consistency_claim_matrix`

Local compact copy:

`derived_local/physics_consistency_claim_matrix`

Key result:

- Scored 52 ROI rows across 11 cycles using seven evidence pillars: front direction, optical change, rollout residuals, kinetic transition, precursor context, echem-shape context, and residual mode taxonomy.
- Tier counts: 2 `cross_modal_high_priority`, 8 `cross_modal_review_priority`, 4 `front_kinetic_consistent`, 3 `rollout_mode_consistent`, 14 `discordant_guardrail`, and 21 `routine_or_low_consistency`.
- Claim readiness is deliberately conservative: all 52 rows are `manual_qc_required_no_physics_claim`, with 0 manual-QC accepted rows.
- Top ranked candidates are cycle-156 ROIs already prioritized by the precursor-informed review bundle: `cycle156_rank7_obj27`, `cycle156_rank5_obj4`, and `cycle156_rank2_obj2`.
- Event/control pillar tests show the strongest separation in `optical_change_score` (median event-control +0.574, Mann-Whitney p=0.00145, permutation p=0.00240) and `front_direction_score` (+0.531, p=0.00991, permutation p=0.0410).
- `mode_taxonomy_score` is suggestive by permutation (+0.644, p=0.0110) but has weaker asymptotic Mann-Whitney p=0.0599; `rollout_residual_score` is suggestive but just above the permutation cutoff (p=0.0564).
- The overall `physics_consistency_score` is not itself a robust event/control classifier (permutation p=0.239), because high-consistency control rows are also present and useful for review.
- Calibration evidence and claim wording are carried through from the calibration risk register: 32/33 scanned HDF5 files contain timing, HDF5 spatial calibration attributes are absent, and spatial scale remains slide-derived.
- The project synthesis now includes a Physics Consistency Claim Matrix section and carries the matrix into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this matrix is the current best review-prioritization map. It consolidates independent optical/front/rollout/kinetic/precursor/mode evidence into a ranked set of hypotheses while explicitly preventing overclaiming. The most defensible physics pattern remains event-associated optical change plus phase-front direction, with manual QC and calibration still gating any publication-scale diffusion or material-mechanism claim.

## 2026-05-21 Probabilistic Rollout Calibration

Added and ran:

`python scripts/tier4_probabilistic_rollout_calibration.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/probabilistic_rollout_calibration`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/probabilistic_rollout_calibration`

Local compact copy:

`derived_local/probabilistic_rollout_calibration`

Key result:

- Audited 4,992 frame-level ROI-only rollout residual rows, collapsed into 156 ROI-method rows across 52 ROIs and 4 event-reference cycles.
- Near-transition frames, defined from the median q60/q70/q80 max-rate phase-kinetics time, make up 0.264 of evaluated frames.
- Persistence is close to nominal empirical 95% coverage overall: 0.955 global weighted coverage, with event ROI coverage 0.945.
- Low-rank DMD undercovers event ROIs: 95% target coverage is only 0.871 on event ROI frames, versus 0.963 on controls. This supports treating DMD as a residual/latent stress descriptor rather than a calibrated predictor.
- Transition-local quantiles improve low-rank DMD near-transition coverage from 0.909 to 0.941, suggesting transition-aware uncertainty bands are more physically appropriate than a single global residual band.
- The strongest residual-error contrast is low-rank DMD late-vs-early rollout MAE, median +0.00559, p=8.87e-29; DMD event-vs-control MAE is also higher, median +0.00401, p=5.59e-15. Persistence and velocity event/control contrasts are weak by comparison.
- Calibration/physics correlations show low-rank DMD q90 undercoverage tracks cumulative optical change (rho=0.586, p=5.08e-06) and first-last decorrelation (rho=-0.519, p=8.07e-05).
- Top undercovered ROI-method rows again prioritize cycle-156 candidates, especially `cycle156_rank7_obj27`, `cycle156_rank5_obj4`, and `cycle156_rank2_obj2`.
- The project synthesis now includes a Probabilistic Rollout Calibration section and carries this audit into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this fills the uncertainty-calibration gap without pretending to have a final generative uncertainty model. Persistence remains the best-calibrated raw pixel baseline, while DMD undercoverage is a useful physics-facing stress signal that concentrates on high-change event ROIs. For future neural rollout models, uncertainty should be conditioned on transition state and optical-change descriptors, not only on frame index or global residual variance.

## 2026-05-21 Cycle State-Space Transition Audit

Added and ran:

`python scripts/tier4_cycle_state_space_transition_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_space_transition_audit --n-permutation 5000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_space_transition_audit`

Local compact copy:

`derived_local/cycle_state_space_transition_audit`

Key result:

- Built a cycle-level degradation state space from 89 four-particle trace cycles joined to 81 within-cycle echem-shape cycles.
- Used 107 current-cycle features, excluding future labels and abrupt-drop labels from the state-space inputs.
- PCA/state clustering selected 4 cycle-state clusters with silhouette 0.634.
- The degradation axis is strongly coupled to cycle/protocol progression and capacity: rho=-0.738 with `cycleNo`, rho=-0.710 with `capacity_mAh`, and rho=0.433 with `coulombic_efficiency_pct`.
- PC2 is the strongest future-drop separator: future any-drop within 8 cycles has median positive-negative PC2 shift +0.730, Mann-Whitney p=2.32e-4, permutation p=0.0162.
- A compact state-space logistic model for `future_any_drop_within_8cycles` reaches mean ROC-AUC 0.781 and balanced accuracy 0.731 across stratified folds.
- Added an expanding-origin temporal holdout guardrail with an 8-cycle purge; 2 of 4 chronological blocks were evaluable, with mean ROC-AUC 0.779 and balanced accuracy 0.645. This supports signal persistence but shows the balanced-accuracy claim weakens under stricter temporal evaluation.
- State clusters are useful but not definitive: two singleton transition states appear around cycles 126 and 150, so state-transition interpretations remain hypothesis-generating.
- The project synthesis now includes a Cycle State-Space Transition Audit section and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this creates a cycle-level companion to the ROI/front analyses. It suggests that degradation risk is visible in the joint particle-trace/echem-shape trajectory before abrupt drops, especially through a second state-space coordinate, but it remains a cycle-level early-warning analysis rather than localized front validation or calibrated diffusion.

## 2026-05-21 Cycle State-Space Temporal Holdout Guardrail

Command:

`python scripts/tier4_cycle_state_space_transition_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_space_transition_audit --n-permutation 5000`

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_space_transition_audit`

Local synced output:

`derived_local/cycle_state_space_transition_audit`

What changed:

- Added an expanding-origin chronological holdout to the cycle state-space audit, with an 8-cycle purge before each test block and train-only fitting for imputation, scaling, PCA, and logistic regression.
- The shuffled stratified folds remain mean ROC-AUC 0.781 and balanced accuracy 0.731 for `future_any_drop_within_8cycles`.
- The stricter temporal holdout evaluates 2 of 4 later-cycle blocks; 2 blocks are skipped because the train or test slice lacks enough positive/negative class diversity.
- Evaluated temporal blocks reach mean ROC-AUC 0.779 and balanced accuracy 0.645. Fold details: cycles 84-118 AUC 0.600 / balanced accuracy 0.623; cycles 142-158 AUC 0.958 / balanced accuracy 0.667.
- The temporal holdout output is `cycle_state_future_drop_temporal_holdout.csv`, and the project synthesis now reports this guardrail alongside the shuffled-fold result.

Interpretation: the cycle-level early-warning signal is not only an artifact of shuffled neighboring-cycle folds, but the usable chronological evidence is still sparse. Treat it as a promising degradation-state covariate for experiment prioritization and manual review, not as a deployable forecasting model.

## 2026-05-21 Cycle State to ROI/Front Bridge

Command:

`python scripts/tier4_cycle_state_roi_bridge.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_roi_bridge --n-permutation 4000`

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_roi_bridge`

Local synced output:

`derived_local/cycle_state_roi_bridge`

Key result:

- Joined the 89-cycle state-space table to the 52 ROI physics-consistency rows across 11 ROI cycles, then added echem-shape-conditioned ROI/front residual targets.
- Row-level bridge associations are strong but not independent within cycle: `cycle_state_pc2` vs `physics_consistency_score` rho=0.702, permutation p=0.0005; `cycle_state_pc2` vs `kinetic_transition_score` rho=0.695, p=0.0005.
- The stricter cycle-collapsed check keeps the same story over 11 cycle points: `cycle_state_pc2` vs `mode_taxonomy_score` rho=0.855, permutation p=0.0020; `cycle_state_pc2` vs `physics_consistency_score` rho=0.836, p=0.0020; `cycle_state_pc2` vs `kinetic_transition_score` rho=0.764, p=0.0095.
- Event-reference-centered checks show within-reference state-step associations with precursor/optical/front scores, led by `axis_step_ref_centered` vs `precursor_context_score_ref_centered` rho=0.790, p=0.0005, and `axis_step_ref_centered` vs `front_direction_score_ref_centered` rho=0.451, p=0.0005.
- Cycle-state cluster 1 contains 42 ROI rows across 9 cycles and has cross-modal priority fraction 0.238; cluster 0 contains 10 ROI rows across 2 cycles and has cross-modal priority fraction 0.
- The project synthesis now includes a Cycle State To ROI/Front Bridge section and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: the cycle-level AI state coordinate is not isolated from localized ROI/front physics. PC2, already the strongest future-drop coordinate, also tracks the ROI physics-consistency/mode/kinetic scores after collapsing repeated ROI rows by cycle. This makes it a useful bridge variable for experiment prioritization and future model conditioning, while still not validating manual front labels, causal propagation, or calibrated diffusion.

## 2026-05-21 Particle Mask Stability And History-Fallback Audit

Added and ran:

`scripts/tier4_particle_mask_stability_audit.py --manifest /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_mask_stability_audit`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_mask_stability_audit`

Local compact copy:

`derived_local/particle_mask_stability_audit`

Key result:

- Audited 52 ROI-only particle crops and 4,992 frame masks using a sequence-level particle-support prior plus per-frame local-contrast/temporal-variance candidates.
- The fallback rule reuses the previous accepted mask when candidate area, fragmentation, or centroid jump violates rolling-history constraints, addressing drift-correction blur without expanding back to full-frame context.
- Median fallback fraction was 0.000; median accepted-area CV was 0.0418; median accepted centroid path was 73.6 px.
- Event and control ROIs did not differ significantly in mask instability: mask-instability median event-control -0.0518, Mann-Whitney p=0.949.
- The strongest mask/physics association was accepted centroid path versus high-fraction slope, Spearman rho=-0.370, p=0.00695; this is a guardrail signal that front/phase-rate readouts should keep mask stability as a covariate.
- Highest-instability examples for manual review include `cycle86_rank8_obj17`, `cycle156_rank2_obj2`, `cycle86_rank6_obj78`, `cycle88_rank2_obj4`, and `cycle116_rank2_obj2`.

Interpretation: the selected tensors remain particle-region-only, and the automatic history-aware mask guardrail does not show event/control mask-instability leakage in the current cohort. This is not manual segmentation; it is a stability/fallback audit to keep ROI-only AI and front/phase physics claims honest under drift-correction blur.

## 2026-05-21 Weak-Label Degradation Benchmark

Command:

`python scripts/tier4_weak_label_degradation_benchmark.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/weak_label_degradation_benchmark`

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/weak_label_degradation_benchmark`

Local synced output:

`derived_local/weak_label_degradation_benchmark`

Key result:

- Built a consensus weak-label manifest over the 52 selected ROI videos by joining sequence paths, physics-consistency tiers, residual mode labels, precursor review scores, cycle-state bridge coordinates, DMD rollout calibration, and particle-mask stability metrics.
- Only 7 rows survive the conservative trainable weak-label filters: 3 `weak_event_enriched_front_mode` positives and 4 `weak_low_consistency_control` negatives. The remaining 45 rows are explicitly review-only or uncertain.
- Label counts are: 19 `review_control_uncertain`, 15 `review_positive_uncertain`, 11 `review_uncertain`, 4 `weak_low_consistency_control`, and 3 `weak_event_enriched_front_mode`.
- Leave-event-reference-cycle split audit creates 4 folds, but only the cycle-60 holdout fold has both weak positive and weak negative labels in train and test. The cycle-86, cycle-116, and cycle-156 holdouts are missing one weak class in the test set.
- Top weak-positive rows are `cycle156_rank7_obj27`, `cycle156_rank8_obj10`, and `cycle60_rank6_obj26`. Top weak-negative rows include `cycle118_rank2_obj2`, `cycle90_rank3_obj4`, `cycle58_rank1_obj1`, and `cycle90_rank4_obj6`.
- The project synthesis now includes a Weak-Label Degradation Benchmark section and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this turns the automatic evidence into a usable but heavily guarded AI benchmark manifest. The important result is not that we have enough labels to train a final model; it is that most automatic rows are not trustworthy labels yet. Future video models should use this manifest for provenance and split hygiene, while final degradation-mode labels still require manual QC.

## 2026-05-21 Masked ROI Rollout Audit

Added and ran:

`scripts/tier4_masked_roi_rollout_audit.py --manifest /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv --mask-stability /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_mask_stability_audit/particle_mask_stability_per_roi.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_roi_rollout_audit --rank 16 --train-fraction 0.67`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_roi_rollout_audit`

Local compact copy:

`derived_local/masked_roi_rollout_audit`

Key result:

- Re-scored 4,992 held-out frame/method rows inside the accepted particle support from the history-aware mask guardrail, with separate non-particle context metrics.
- Persistence remains the best method inside the particle mask for all 52 ROIs. Median particle MSE: persistence 1.15e-4, velocity 2.60e-4, low-rank DMD 6.55e-3.
- Low-rank DMD errors are much more particle-local than background-local: median particle/non-particle MSE ratio 5.23, versus 1.46 for persistence and 1.32 for velocity.
- Event ROIs have higher low-rank-DMD particle error than controls: median event-control particle MSE +0.00342, Mann-Whitney p=0.0150; particle-MSE ratio versus persistence is also higher in event ROIs, median +16.64, p=0.0257.
- Masked rollout errors link strongly to optical-change descriptors: low-rank-DMD particle MSE vs cumulative_abs_first_last rho=0.603, p=2.27e-6; persistence particle/full-MSE fraction vs cumulative_abs_first_last rho=0.637, p=3.91e-7.
- High particle-rollout difficulty examples include `cycle156_rank7_obj27`, `cycle86_rank8_obj17`, and `cycle157_rank2_obj2`; these should stay high priority for manual QC and mode labeling.

Interpretation: this strengthens the ROI-only modeling guardrail. The full-crop conclusion that persistence is the strongest pixel predictor survives when scoring only accepted particle pixels, while the DMD residual becomes a sharper particle-local physics descriptor associated with cumulative optical change and event/control status. This still does not prove a deployable video model; it supports using masked residuals as interpretable degradation descriptors.

## 2026-05-22 Diffusion Proxy Sanity Audit

Added and ran:

`scripts/tier4_diffusion_proxy_sanity_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/diffusion_proxy_sanity_audit`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/diffusion_proxy_sanity_audit`

Local compact copy:

`derived_local/diffusion_proxy_sanity_audit`

Key result:

- Joined the 12 selected high-resolution front ROIs to threshold-robust front metrics, manual-QC status, and particle-mask stability descriptors.
- The stricter diffusion gate produced 0 automatic positive diffusion-proxy candidates and 0 publication diffusion candidates.
- Median selected-front apparent D was -3.65e-4 um2/s; the threshold-sweep median was -6.47e-7 um2/s.
- Only 1 of 12 selected fronts had a nonnegative selected radius-squared diffusion proxy, only 1 of 12 met the selected-fit R2 gate, and 0 of 12 had accepted manual QC labels.
- Estimator consensus was mixed for 6 fronts, negative for 5 fronts, and positive for 1 front. The one positive-consensus row still failed because the selected high-resolution radius-squared slope was negative and the fit was weak.
- Drift relative to apparent motion was low for all 12 selected fronts, so the primary rejection is not stage drift; it is sign/estimator inconsistency, poor radius-squared fit quality, threshold sensitivity, and missing manual QC.
- The project synthesis now includes a Diffusion Proxy Sanity Audit section and carries the gate counts into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this closes the current diffusion-claim loophole. The project can keep using radius/phase-front slopes as optical particle-region descriptors, but no calibrated diffusion coefficient should be reported from the current selected-front set. A publishable diffusion claim still needs raw microscope calibration provenance, validated timebase, accepted front masks, estimator agreement, and manual ROI QC.

## 2026-05-22 Control-Balanced High-Resolution Front Tracking

Added and ran:

`scripts/tier4_control_balanced_front_tracking_table.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_tracking_table`

Then ran the existing high-resolution tracker, provisional calibration, and diffusion sanity gate on the generated 40-row balanced table:

`python scripts/tier3_track_selected_front_rois.py --roi-table /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_tracking_table/control_balanced_front_rois_for_tracking.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_tracking --crop-size-full 192 --baseline-frames 5`

`python scripts/tier3_front_roi_calibration_qc.py --front-tracking-csv /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_tracking/selected_front_roi_tracking_summary.csv --joint-modes-csv /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_joint_physics_degradation_modes/roi_joint_physics_degradation_modes.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_calibration_qc --top-n 40`

`python scripts/tier4_diffusion_proxy_sanity_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --front-calibration-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_calibration_qc --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_diffusion_proxy_sanity_audit`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_tracking_table`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_tracking`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_calibration_qc`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_diffusion_proxy_sanity_audit`

Local compact copies:

- `derived_local/control_balanced_front_tracking_table`
- `derived_local/control_balanced_front_tracking`
- `derived_local/control_balanced_front_calibration_qc`
- `derived_local/control_balanced_diffusion_proxy_sanity_audit`

Key result:

- The bridge table resolved all 40 control-balanced QC ROIs into high-resolution tracker inputs: 24 controls and 16 events, with 0 missing front/object coordinate rows.
- High-resolution tracking completed for all 40 ROIs. Most cycle-level mean radius-squared slopes remained negative or near zero; fit quality was generally weak outside a few contracting fronts.
- The balanced diffusion sanity audit again produced 0 automatic positive diffusion-proxy candidates and 0 publication diffusion candidates.
- Gate counts on the balanced cohort: low drift 40/40, threshold nonnegative 16/40, q70 bootstrap positive 9/40, selected nonnegative 8/40, selected-fit R2 1/40, manual-QC accepted 0/40.
- Median selected-front apparent D was -1.03e-4 um2/s; median threshold-sweep apparent D was -1.02e-6 um2/s.
- Event/control tests were not significant for selected diffusion proxy: median event-control selected D -4.08e-5 um2/s, Mann-Whitney p=0.858. Threshold median D was also not significant, p=0.314.
- The project synthesis now includes a Control-Balanced Front Tracking And Diffusion Sanity section and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this removes the main weakness of the previous selected-front diffusion sanity audit, which was event-heavy. Even after adding a balanced high-resolution control/event cohort, the radius-squared front proxies do not behave like calibrated diffusion coefficients. They remain useful optical front descriptors and QC priorities, not publishable transport constants.

## 2026-05-22 Masked Rollout Cycle Warning Audit

Added and ran:

`scripts/tier4_masked_rollout_cycle_warning.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_rollout_cycle_warning --n-permutation 5000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_rollout_cycle_warning`

Local compact copy:

`derived_local/masked_rollout_cycle_warning`

Key result:

- Collapsed masked particle-rollout residuals from 52 ROI rows to 11 observed ROI cycles and joined the larger particle-trace abrupt-drop/future-drop targets.
- Tested 105 rollout-derived cycle features with 5,000 permutation checks per binary/correlation statistic.
- Same-cycle abrupt drops show positive jumps in low-rank-DMD particle residuals versus the previous observed ROI cycle: top feature `low_rank_dmd_particle_mse_mean_max_delta_prev_observed_roi_cycle`, median positive-negative +0.00885, Mann-Whitney p=0.0381, permutation p=0.0144.
- Another same-cycle abrupt-drop signal is `low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_max_delta_prev_observed_roi_cycle`, median positive-negative +12.29, Mann-Whitney p=0.0190, permutation p=0.0156.
- Future-warning tests are underpowered in the selected ROI-cycle set: only one observed ROI cycle is positive for future_any_drop_within_4/8/16 cycles, and there are zero future_sync2 positives.
- Masked persistence particle/full-MSE fraction is strongly aligned with cycle-state PC2 and frame-count context across the 11 ROI cycles, e.g. rho=0.955 vs cycle_state_pc2 with permutation p=0.0002. This is a context/selection guardrail, not proof of a general early-warning detector.
- Warning-ranked cycles are headed by 60 and 156, both same-cycle abrupt-drop cycles, followed by 62, 157, 158, and 86.

Interpretation: masked particle-local rollout residuals are useful as same-cycle degradation-state descriptors and review-priority signals. The current selected ROI-cycle subset does not prove future early warning because future-positive counts are too sparse; expanding ROI extraction beyond event-centered windows is still required for a real warning model.

## 2026-05-22 Masked Residual Transition Timing Audit

Added and ran:

`scripts/tier4_masked_residual_transition_timing.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_residual_transition_timing --n-permutation 5000`

Then regenerated the synthesis:

`python scripts/tier4_nmc_ai_physics_synthesis.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/nmc_ai_physics_synthesis`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_residual_transition_timing`

Local compact copy:

`derived_local/masked_residual_transition_timing`

Key result:

- Joined masked ROI rollout frame metrics to automatic phase-kinetics transition timing for 52 ROIs and 156 ROI-method rows, with 5,000 permutation draws for method-level timing alignment.
- Low-rank DMD residual weighted centers are closer to automatic phase-transition centers than the random-timing null at borderline strength: median distance 0.1960 of the evaluation window, null mean 0.2501, null p05 0.1945, empirical p=0.05599.
- Peak-frame timing does not support the same claim. Low-rank DMD peak distance has empirical p=0.714, and persistence/velocity peak timing are also non-significant.
- Event/control timing separation is weak. The strongest trend is persistence near-transition residual fraction, median event-control +0.0692, Mann-Whitney p=0.0678; low-rank DMD and velocity near-transition fractions are not significant.
- Persistence particle/nonparticle residual ratios track kinetic-rate descriptors: rho=0.518 vs q70 max absolute rate, p=8.42e-5; rho=0.490 vs q80 transformed-fraction delta, p=2.29e-4; rho=0.489 vs q80 max absolute rate, p=2.31e-4.
- Low-rank DMD near-minus-far particle residuals are anticorrelated with q70/q80 transformed-fraction deltas, rho=-0.474 and -0.421, respectively.
- Top near-transition residual examples include `cycle158_rank2_obj1` for velocity and persistence, plus event-cycle 86 ROIs (`cycle86_rank4_obj9`, `cycle86_rank8_obj17`, `cycle86_rank5_obj8`, `cycle86_rank3_obj5`) whose residual centers sit close to automatic transition centers.
- The project synthesis now includes a Masked Residual Transition Timing section and carries the summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this adds a timing guardrail between AI residuals and optical phase kinetics. There is a borderline low-rank DMD weighted-center alignment signal and clear kinetic-rate correlations, but no robust peak-frame timing alignment or strong event/control separation. Treat it as evidence that masked residual energy partly reflects phase-transition kinetics, not as a manual front annotation or calibrated transport result.


## 2026-05-22 Masked Residual State Transfer Warning Audit

Added and ran:

`scripts/tier4_masked_residual_state_transfer_warning.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_residual_state_transfer_warning --n-permutation 5000`

Then regenerated the synthesis:

`python scripts/tier4_nmc_ai_physics_synthesis.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/nmc_ai_physics_synthesis`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_residual_state_transfer_warning`

Local compact copy:

`derived_local/masked_residual_state_transfer_warning`

Key result:

- Built a masked-residual signature from the 11 video-backed masked rollout warning cycles using the strongest low-rank-DMD particle residual jump features, then transferred that signature through cycle-state/echem features onto the full 89-cycle trace table.
- The expanded transfer table has many more future positives than the direct masked video cycle table: 20 future_any_drop_within_8cycles positives and 40 future_any_drop_within_16cycles positives.
- Transferred masked-residual signature separates future 16-cycle drops: median positive-negative +0.631, Mann-Whitney p=0.00449, permutation p=0.000400, AUC=0.676.
- It also separates future 8-cycle drops: median positive-negative +0.648, Mann-Whitney p=0.00486, permutation p=0.00380, AUC=0.708.
- Direct cycle-state PC2 remains the stronger simple future-8 baseline on the same 89-cycle table: AUC=0.772, permutation p=0.0158.
- Anchor leave-one-cycle transfer is weak: rho=-0.155, p=0.650 across the 11 video-backed cycles. This is the key guardrail against treating the transferred score as a validated video residual measurement.
- The transferred score correlates with cycle_state_pc6 (rho=0.541), axis_step (rho=0.522), lower coulombic efficiency (rho=-0.499), cycle_state_pc2 (rho=0.354), and future_any_drop_within_8cycles (rho=0.301).
- Top transfer-ranked cycles are dominated by the late high-COV window around cycles 150-156, including cycles 150, 151, 152, 154, 155, and 156.
- The project synthesis now includes a Masked Residual State Transfer Warning section and carries the compact summary into `nmc_ai_physics_synthesis_summary.json`.

Interpretation: this addresses the direct masked-rollout future-warning underpowering by using the 11 video-backed cycles as a particle-local residual anchor and testing the transferred signature across all cycle-state rows. It is useful for hypothesis ranking and for choosing which cycles deserve new ROI/video extraction, but the weak leave-one-cycle anchor check means it is not a deployable warning model and not proof that unexported cycles have measured video residuals.

## 2026-05-22 Cycle Hazard Warning Audit

Added and ran:

`scripts/tier4_cycle_hazard_warning_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_hazard_warning_audit --n-permutation 40`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_hazard_warning_audit`

Local compact copy:

`derived_local/cycle_hazard_warning_audit`

Key result:

- Built a rolling-origin warning audit for future abrupt particle drops within 8 cycles using the 89-cycle cycle-state table. Each prediction only trains on cycles at least 8 cycles before the test cycle; 62 cycles are evaluable after the min-train/class-balance gate.
- Best feature set is particle trace + echem shape + acquisition context: AUC 0.783, average precision 0.573, top-rate balanced accuracy 0.789 across 16 positive future-drop cycles.
- A short 20-permutation rolling-label null gives null mean AUC 0.478, null p95 0.581, empirical p=0.0476. This is a useful but still shallow null; a deeper rerun can raise the permutation count.
- Warning lead-time audit hits 2/4 event cycles within 4 cycles and 3/4 event cycles within 8 or 16 cycles; median lead time is 6 cycles for the 8-cycle lookback. The first event at cycle 60 has no eligible prior predictions after the min-train gate.
- Feature-group ablation shows the warning signal is mainly particle-trace driven: removing particle-trace features drops AUC by 0.291, while removing echem-shape drops 0.071, acquisition context 0.041, and cycle-level echem summaries 0.0027.
- Warning probabilities correlate most with cycle_state_pc2 (rho=0.456, p=1.96e-4), linking the rolling warning model back to the earlier cycle-state manifold.

Interpretation: this strengthens the cycle-level early-warning story using a stricter chronological protocol than the shuffled-fold classifier. The result supports particle-trace/echem warning covariates and review prioritization, but it is not a localized ROI/front detector and does not validate diffusion or degradation labels without manual QC.

## 2026-05-22 Transfer-Ranked ROI Reconstruction And Masked Rollout

Added and ran:

`scripts/tier4_transfer_ranked_roi_reconstruction.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_reconstruction --top-cycles 12 --top-candidates-per-cycle 4`

Then exported particle-region sequences from the generated ROI table and ran the masked rollout audit on those transfer-ranked crops:

`python scripts/tier2_export_selected_roi_sequences.py --roi-table /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_reconstruction/transfer_ranked_roi_table.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_sequences --crop-size-full 192 --output-size 96 --samples-per-roi 96`

`python scripts/tier4_masked_roi_rollout_audit.py --manifest /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_sequences/selected_roi_sequence_manifest.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_masked_roi_rollout_audit --rank 16 --train-fraction 0.67`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_reconstruction`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_masked_roi_rollout_audit`

Local compact copies:

- `derived_local/transfer_ranked_roi_reconstruction`
- `derived_local/transfer_ranked_roi_sequences`
- `derived_local/transfer_ranked_masked_roi_rollout_audit`

Key result:

- Converted the masked residual state-transfer warning ranks into direct video-backed ROI candidates. The top 12 transfer-ranked cycles were sampled from their HDF5 segments; all 12 resolved successfully with 0 missing cycles.
- Reconstructed 960 automatic particle-like candidate components and retained 48 ROI rows, four per transfer-ranked cycle, compatible with the existing ROI sequence exporter.
- The sampled cycles include the late high-COV warning window around cycles 146-156 plus cycle 116 and cycle 40. Top-ranked cycle 150 is future-drop positive within 8 cycles and produced 80 candidates; cycles 151, 152, 153, 154, 155, and 148 are also future8-positive.
- Exported 48 fixed particle-region sequences, each with 96 frames and 96x96 output crops. Cycle 156 transfer-ranked ROIs show a large mean ROI delta (+702.6 raw intensity; +0.0273 normalized), while other late-window cycles show smaller mixed deltas.
- Masked rollout on the transfer-ranked crops produced 4,608 frame-metric rows. Persistence remains the best particle-mask predictor for all 48 ROIs; low-rank DMD has much higher particle-local residuals than nonparticle context, with median particle/nonparticle MSE ratio 8.31.
- The most difficult low-rank-DMD transfer-ranked ROIs include `cycle151_rank4_obj3`, `cycle153_rank7_obj4`, `cycle156_rank3_obj2`, `cycle154_rank6_obj2`, and `cycle151_rank4_obj2`.

Interpretation: this closes the loop from cycle-level warning to direct video data. It does not validate the transfer-ranked crops as manual particle annotations or fronts, but it gives a concrete new ROI cohort for next-frame prediction, masked rollout, and eventual manual QC in cycles that the warning models said were worth inspecting.

## 2026-05-22 Cross-Cohort Rollout Transfer Audit

Added and ran:

`python scripts/tier4_cross_cohort_rollout_transfer_audit.py --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cross_cohort_rollout_transfer_audit --rank 16 --train-fraction 0.67`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cross_cohort_rollout_transfer_audit`

Local compact copy:

`derived_local/cross_cohort_rollout_transfer_audit`

Key result:

- Trained the same interpretable low-rank DMD rollout model separately on the original selected ROI cohort, on the transfer-ranked warning ROI cohort, and on the pooled cohort, then evaluated each model on both cohorts with particle-mask scoring.
- The audit covers 11 selected ROI sequences and 48 transfer-ranked ROI sequences. Each model is evaluated against persistence, velocity, and low-rank DMD rollouts on held-out frame tails.
- Transfer-ranked ROIs are a distinct video-dynamics domain: selected-cohort DMD evaluated on transfer-ranked crops has median particle MSE 0.0204, which is 3.49x the transfer-ranked internal DMD baseline, with Mann-Whitney p=3.28e-9.
- Pooled training transfers much better to transfer-ranked ROIs: median particle MSE 0.00620, only 1.06x the transfer-ranked internal baseline, p=0.674.
- The reverse transfer is also poor: transfer-ranked DMD evaluated on selected ROIs is 10.28x the selected internal baseline, p=8.11e-4.
- Even internal DMD remains worse than persistence in both cohorts, so the result should guide cohort-aware residual modeling and domain adaptation rather than support a deployable low-rank video predictor.

Interpretation: this adds a stricter model-generalization test around next-frame/rollout work. The late warning-window crops are not just more of the same selected event/control videos; they require either cohort-aware training, pooled training, or stronger learned dynamics before rollout errors can be interpreted as robust degradation physics.

## 2026-05-22 Transfer-Ranked Front Physics Audit

Added and ran transfer-ranked threshold/front extraction, then joined those phase/front proxies back to warning-ranked ROI crops, masked rollout residuals, and future-drop labels:

`python scripts/tier3_multicycle_threshold_robust_fronts.py --manifest /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_sequences/selected_roi_sequence_manifest.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_threshold_robust_fronts`

`python scripts/tier4_transfer_ranked_front_physics_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_front_physics_audit --n-permutation 5000`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_threshold_robust_fronts`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_front_physics_audit`

Local compact copies:

- `derived_local/transfer_ranked_threshold_robust_fronts`
- `derived_local/transfer_ranked_front_physics_audit`

Key result:

- The audit covers 48 transfer-ranked ROI crops across 12 warning-ranked cycles with 5,000 permutation tests.
- Future 8-cycle positive rows are common in this warning-ranked panel: 28/48 ROI rows and 7/12 cycles.
- Persistence particle MSE is the strongest ROI-level future8-associated front/residual feature in the joined table: Spearman rho=0.760 versus future8 status and the top target test gives a large positive median difference with permutation support.
- Radius-squared/front-motion proxies correlate strongly with masked rollout residual difficulty: radius2 slope and apparent diffusion-proxy median correlate with low-rank-DMD particle MSE at rho=0.674, and q70 radius2 bootstrap slope correlates at rho=0.679.
- Transfer-ranked cycles 116 and 150 have especially high threshold-robust phase scores, while late-window cycles 151-156 show mixed apparent diffusion signs. This reinforces the earlier guardrail that the values are optical-front proxies, not calibrated Li diffusion coefficients.
- The review-priority table ranks ROIs that jointly combine transfer score, phase/front proxy strength, rollout residual difficulty, and future-drop context for manual inspection.

Interpretation: this connects the cycle-level warning loop back to front/phase physics on direct video crops. It supports using transfer-ranked ROIs for hypothesis ranking and domain-adapted rollout/front modeling, but it still does not justify calibrated diffusion or manual front claims without QC labels and calibration confirmation.

## 2026-05-22 Transfer-Ranked Residual Transition Timing

Added and ran:

`python scripts/tier4_transfer_ranked_residual_transition_timing.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_residual_transition_timing --n-permutation 5000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_residual_transition_timing`

Local compact copy:

`derived_local/transfer_ranked_residual_transition_timing`

Key result:

- Computed optical phase-kinetic transition centers directly from all 48 transfer-ranked ROI movies, then aligned them to held-out masked rollout residual timing for persistence, velocity, and low-rank DMD.
- The audit yields 144 ROI-method timing rows with 5,000-permutation alignment tests.
- Low-rank-DMD residual weighted centers are strongly phase-transition aligned: median distance to transition center is 0.087 of the held-out tail versus null mean 0.250, empirical p=0.000200.
- Persistence and velocity weighted residual centers are also closer than random, but weaker: persistence median distance 0.156, p=0.00360; velocity median distance 0.170, p=0.0122.
- Future-drop-positive transfer-ranked ROIs have higher persistence particle/nonparticle residual ratios: future8 median positive-negative +0.648, AUC=0.832, Mann-Whitney p=1.05e-4, permutation p=0.000400.
- Future8 positives also have much larger persistence residual peaks: median positive-negative +0.0104 particle MSE, AUC=0.879, Mann-Whitney p=9.74e-6, permutation p=0.00180.

Interpretation: unlike the broader event/control timing audit, the warning-ranked transfer cohort gives a strong temporal link between low-rank rollout residual centers and automatic phase-transition centers. This is still an automatic optical proxy, but it strengthens the case that rollout residuals are tracking real transition-like particle dynamics in the late warning-window crops.

## 2026-05-22 Multi-Cohort Future-Drop Weak Model

Added and ran:

`python scripts/tier4_multicohort_future_drop_model.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multicohort_future_drop_model --n-permutation 40 --rf-trees 80`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multicohort_future_drop_model`

Local compact copy:

`derived_local/multicohort_future_drop_model`

Key result:

- Built a combined direct-video feature table from 11 original selected ROI rows and 48 transfer-ranked ROI rows, using 44 optical front, masked rollout, ROI intensity, cycle-state, and consistency features.
- Labels are weak cycle-level future-drop-within-8-cycles labels projected onto ROI rows, not manual degradation labels. The selected cohort contributes no positive future8 ROI rows, so leave-cohort evaluation is intentionally marked missing-class.
- Cycle-grouped out-of-fold random forest scores 58 labeled ROI rows with AUC=0.886 and average precision=0.914. A 40-permutation cycle-label null gives null mean AUC=0.441, null p95=0.591, empirical p=0.0244.
- Logistic L2 is weaker but still above chance: AUC=0.692 and AP=0.777.
- The strongest univariate and model features are masked rollout residual features, especially velocity and persistence particle MSE fractions/ratios, followed by radius2/front-motion proxies and the transferred masked residual signature.
- The strongest univariate feature is velocity particle-MSE fraction of full crop: oriented AUC=0.968, median positive-negative +0.599, Mann-Whitney p=1.01e-9.

Interpretation: this is the most model-like ROI/video warning audit so far, but it remains a weak-label review-prioritization model. The apparent performance mostly reflects transfer-ranked future-positive rows and cannot be treated as deployable until a class-balanced selected/control cohort and manual QC labels are available.

## 2026-05-22 Active-Learning QC Prioritization

Added and ran:

`python scripts/tier4_active_learning_qc_prioritization.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/active_learning_qc_prioritization`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/active_learning_qc_prioritization`

Local compact copy:

`derived_local/active_learning_qc_prioritization`

Key result:

- Merged the manual-QC workbook, precursor-informed review manifest, multi-cohort weak future-drop predictions, transfer-ranked front physics review table, residual transition timing audit, and transfer reconstruction metadata into a single manual-review queue.
- The queue contains 97 ROI candidates across 21 cycles; 47 rows already have at least one visual QC/preview asset.
- Four rows are promoted to immediate manual-QC review, with remaining rows split across model-boundary, front/diffusion guardrail, control-balance, and standard manual-QC tiers.
- Top-ranked candidates include `cycle116_rank7_obj37`, `cycle156_rank3_obj2`, `cycle147_rank11_obj2`, `cycle86_rank4_obj9`, and `cycle151_rank4_obj2`.
- The most common review tags are control-balance review, visual asset available, high future-drop probability, model boundary case, and front/diffusion guardrail review.

Interpretation: this closes the loop from AI/physics discovery into a concrete human-review worklist. It does not assign manual labels or validate diffusion/degradation claims; it prioritizes where manual particle identity, front-mask quality, and diffusion-interpretability checks should be spent next.

## 2026-05-22 Balanced Future-Drop Direct-Video ROI Audit

Added and ran a separate class-balanced direct-video ROI workflow on Isambard:

`python scripts/tier4_balanced_future_roi_reconstruction.py --root /scratch/u6hp/nsagar.u6hp/Alek_Jiho --ranked-cycles /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_residual_state_transfer_warning/masked_residual_state_transfer_ranked_cycles.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_reconstruction --positive-cycles 12 --negative-cycles 12 --top-candidates-per-cycle 3 --samples-per-segment 16`

`python scripts/tier2_export_selected_roi_sequences.py --root /scratch/u6hp/nsagar.u6hp/Alek_Jiho --roi-table /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_reconstruction/balanced_future_roi_table.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_sequences --crop-size-full 192 --output-size 96 --samples-per-roi 96`

`python scripts/tier3_multicycle_threshold_robust_fronts.py --manifest /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_sequences/selected_roi_sequence_manifest.csv --cohort-table /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_reconstruction/balanced_future_roi_table.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_threshold_robust_fronts --n-bootstrap 200`

`python scripts/tier4_masked_roi_rollout_audit.py --manifest /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_sequences/selected_roi_sequence_manifest.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_masked_roi_rollout_audit --rank 16 --train-fraction 0.67`

`python scripts/tier4_balanced_future_roi_physics_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_physics_audit --n-permutation 40`

Remote output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_reconstruction`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_particle_mask_stability`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_threshold_robust_fronts`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_masked_roi_rollout_audit`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_physics_audit`

Local compact copies exclude heavy NPZ/previews/overlays where applicable.

Key result:

- Reconstructed 24 direct-video cycles, 1,920 automatic particle-like components, and 72 ROI rows from a deliberately balanced weak future8 design: 36 positive ROI rows and 36 negative ROI rows across 12 positive and 12 negative cycles.
- Exported 72 fixed 96-frame particle-region crop tensors for front/rollout analyses.
- Masked rollout again picks persistence as best inside particle masks for all 72 ROIs; low-rank DMD remains a useful residual descriptor rather than a better pixel forecaster.
- The balanced future8 physics audit uses 26 direct-video features and leave-cycle-out scoring. Logistic L2 reaches AUC=0.716 and AP=0.761; the 40-permutation cycle-label sanity null gives null mean AUC=0.489, p95=0.701, empirical p=0.0488.
- The strongest ROI-level weak future8 features are radius2/front-motion proxies and masked rollout residual fractions: radius2 slope median and apparent diffusion-proxy median both have oriented AUC=0.717 and permutation p=0.0244; persistence and velocity particle-MSE fractions are also positive-associated.
- The reused threshold-front script's event/control summary is not meaningful for this cohort because labels are `future8_positive` / `future8_negative`, not `event` / `control`; the final balanced audit uses the explicit `future_any_drop_within_8cycles` label.

Interpretation: this addresses the main class-balance weakness of the transfer-ranked video audit. Future-drop signal persists in a more balanced direct-video cohort, but the top features still include apparent optical-front proxies and automatic particle-mask residuals, so this remains weak-label physics-prioritization evidence pending manual QC and calibrated diffusion validation.

## 2026-05-22 Balanced Future-Drop Particle-Mask Stability Guardrail

Updated and reran the particle-mask stability audit on the balanced future-drop ROI tensors:

`python scripts/tier4_particle_mask_stability_audit.py --manifest /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_sequences/selected_roi_sequence_manifest.csv --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_particle_mask_stability`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_particle_mask_stability`

Local compact copy:

`derived_local/balanced_future_particle_mask_stability`

Key result:

- Extended the mask-stability audit to preserve `future8_positive` and `future8_negative` cohort roles and to emit direct future8-positive/minus-negative mask-stability tests.
- The refreshed balanced sequence manifest now carries weak future8 labels and transferred warning scores alongside the ROI tensor paths.
- The audit covers 72 balanced ROI tensors and 6,912 frames.
- Median fallback fraction is 0.0 in both weak future8 classes; median accepted-area CV is 0.0779 for positives versus 0.0701 for negatives.
- The strongest mask-stability contrast is accepted-centroid max step, positive-negative +2.15 px, Mann-Whitney p=0.175. Mask instability score is also non-significant (positive-negative +0.033, p=0.461).

Interpretation: this is a useful artifact guardrail for the balanced future-drop audit. The weak future8 signal in the balanced direct-video cohort is not explained by a simple difference in mask fallback, fragmentation, or aggregate mask instability between positive and negative rows.

## 2026-05-22 Balanced Future Context/Region Confound Audit

Added and ran a context/region guardrail on the balanced future-drop ROI table:

`python scripts/tier4_balanced_future_context_region_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_context_region_audit`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_context_region_audit`

Local compact copy:

`derived_local/balanced_future_context_region_audit`

Key result:

- Audited 72 ROI rows across 24 balanced weak future8 cycles, with 36 positive and 36 negative ROI rows.
- Separated acquisition/spatial context (`source_stem`, local segment, stage drift, object x/y/area, frame indices, spatial bins) from design context (`cycleNo`, balanced rank, transferred warning score, selection subrole).
- Acquisition context alone predicts weak future8 labels strongly: random forest AUC=0.851 and AP=0.888; logistic L2 AUC=0.823 and AP=0.846.
- Design context is perfect by construction: AUC/AP=1.0, confirming selection metadata must not be used as deployment features.
- Physics-only scoring reaches AUC=0.711 and AP=0.683; physics plus acquisition context reaches AUC=0.887 and AP=0.896.
- After residualizing physics features against acquisition context, top front/radius/diffusion residuals are non-significant: radius2 residual AUC=0.552, p=0.447; apparent diffusion residual AUC=0.552, p=0.447.
- Spatial bin label enrichment is not significant: x-bin p=1.0, y-bin p=0.846, xy-region p=0.982.

Interpretation: the balanced future-drop physics signal is useful for review prioritization, but it is not yet a context-independent detector. More balanced acquisition design and manual ROI/front QC are needed before claiming direct video physics as a deployable degradation predictor.

## 2026-05-22 Particle-Masked Video Embedding Audit

Added and ran a self-supervised masked video embedding audit over the existing ROI tensor cohorts on Isambard:

`python scripts/tier4_masked_video_embedding_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_video_embedding_audit --n-permutation 80`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_video_embedding_audit`

Local compact copy:

`derived_local/masked_video_embedding_audit`

Key result:

- Extracted particle-prior masked temporal and spatial-temporal descriptors for 172 ROI tensors: 72 balanced future-drop ROIs, 52 selected event/control ROIs, and 48 transfer-ranked ROIs.
- The embedding uses the same history-derived particle-support prior and mask-stability logic as the ROI-only guardrail, then builds PCA components from prior-masked video bins plus trace/mask descriptors. Heavy NPZ tensors stay remote.
- On the balanced future cohort, leave-cycle grouped logistic scoring over masked video embedding/trace features reaches AUC=0.816 and AP=0.865 across 72 rows (36/36 weak future8 positives/negatives).
- The 80-permutation cycle-label null gives null mean AUC=0.469, p95=0.641, empirical p=0.0123.
- The older selected event/control split is much weaker under the same embedding readout: AUC=0.588 and AP=0.530 across 52 rows (24 event / 28 control), which is a useful guardrail against overgeneralizing the balanced future result.
- The strongest univariate masked-video features for weak future8 are increasing particle-region intensity heterogeneity and edge/gradient energy: `particle_std_last_minus_first` has oriented AUC=0.891 and Mann-Whitney p=1.17e-8; `particle_std_slope` has oriented AUC=0.884 and p=2.11e-8; `particle_gradient_slope` has oriented AUC=0.813 and p=4.95e-6.
- Unsupervised clusters show coherent review groups: cluster 6 is entirely weak future8-positive among labeled balanced rows (future8 positive fraction 1.0, prototype `transfer_ranked::cycle152_rank5_obj2`), while cluster 2 is future8-negative among labeled balanced rows (fraction 0.0, prototype `selected_event_control::cycle156_rank8_obj10`).

Interpretation: this is the strongest particle-masked video-embedding evidence so far for weak future-drop triage, and it is closer to the requested AI-video objective than hand-picked scalar front proxies alone. It remains a weak-label and context-sensitive result: the synthesis context audit shows acquisition/spatial metadata can also predict the balanced future labels strongly, so these embeddings should drive review prioritization and model-design hypotheses rather than deployment or calibrated diffusion claims.

## 2026-05-22 Temporal Directionality Physics Audit

Added and ran a temporal directionality guardrail for the balanced future-drop ROI physics descriptors:

`python scripts/tier4_temporal_directionality_physics_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/temporal_directionality_physics_audit`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/temporal_directionality_physics_audit`

Local compact copy:

`derived_local/temporal_directionality_physics_audit`

Key result:

- Audited 72 balanced direct-video ROI rows across 24 cycles using 33 ROI physics/front/rollout descriptors.
- Computed past-drop windows from the particle-trace cycle table and compared future8 labels against past, time-reversed, and circularly shifted labels.
- Physics-only leave-cycle-out logistic regression predicts balanced future8 drops with AUC=0.799 and AP=0.793.
- Circular time-shift null over 23 nonzero shifts has mean AUC=0.593, p95=0.775, and empirical p=0.0417 versus the observed future8 AUC.
- Top future8 features remain front-motion/apparent-diffusion and particle-mask residual descriptors: q70 radius2 slope p95 AUC=0.731, apparent diffusion median AUC=0.717, radius2 slope median AUC=0.717, persistence particle-MSE fraction AUC=0.709.
- Past8 labels are underpowered in this balanced cohort, with only 3 positive ROI rows and no evaluable leave-cycle-out positive predictions; past16 is evaluable but has a different class balance.
- Time-reversed future8 labels remain nontrivial, especially with random forest AUC=0.750, so directionality is supportive rather than causal proof.
- Timing correlations show aftermath/context structure: diffusion/radius2 slope IQR correlates with cycles since previous drop (rho=0.644, p=5.4e-9), and persistence particle/full-MSE fraction anticorrelates with cycles to next drop (rho=-0.493, p=1.66e-5).

Interpretation: this strengthens the precursor-style evidence because the observed future8 physics model beats circular time-shifted labels, but the nontrivial reversed-label, past-drop timing structure, and context-confound guardrails mean the result remains weak-label hypothesis-ranking evidence pending more balanced acquisition design and manual ROI/front QC.

## 2026-05-22 Apparent Diffusion Calibration Bounds Audit

Added and ran an HDF5-timebase calibration guardrail for the balanced future-drop front/diffusion proxies:

`python scripts/tier4_apparent_diffusion_calibration_bounds.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/apparent_diffusion_calibration_bounds`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/apparent_diffusion_calibration_bounds`

Local compact copy:

`derived_local/apparent_diffusion_calibration_bounds`

Key result:

- Mapped all 72 balanced ROI rows, 504 threshold rows, and 24 cycles back to source HDF5 camera timing metadata.
- ROI elapsed time and HDF5 median camera timing agree closely: median ROI elapsed / HDF5 elapsed ratio = 1.0016.
- Used slide-derived pixel-size assumptions of 0.08, 0.096, and 0.12 um/px; no HDF5 pixel-size attribute is available, so values remain apparent optical-front proxies.
- At the q70 threshold and 96 nm/px, median apparent D is 4.32e-08 um2/s, median absolute apparent D is 2.60e-06 um2/s, and 51.4% of ROI rows are positive.
- q70 future8 separation is non-significant after HDF5-timebase recalibration: apparent D positive-negative median difference = -6.33e-07 um2/s, Mann-Whitney p=0.175.
- Some source files have rare timing gaps: maximum source-level HDF5 dt max/median ratio is 13.78, so timing envelopes should accompany any front-motion claim.
- The strongest calibration correlation is HDF5 dt max/median ratio versus transferred masked residual signature, rho=0.728, p=2.86e-84, showing source/timing context remains an important guardrail.

Interpretation: the video timebase itself is well matched for ROI spans, but the apparent diffusion proxy is threshold/context sensitive and not a calibrated material diffusion coefficient. This supports retaining diffusion-like values as optical-front descriptors for QC and hypothesis ranking only.

## 2026-05-22 Balanced Spatial Front Propagation Audit

Added and ran a balanced-cohort spatial/temporal propagation audit over reconstructed ROI front/rollout physics:

`python scripts/tier4_balanced_spatial_front_propagation_audit.py --derived-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived --out-dir /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_spatial_front_propagation_audit --k-neighbors 3 --n-permutation 1000`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_spatial_front_propagation_audit`

Local compact copy:

`derived_local/balanced_spatial_front_propagation_audit`

Key result:

- Built a spatial kNN graph over 72 balanced ROI nodes across 24 cycles and 9 source videos, yielding 414 edges: 144 same-cycle, 135 next-observed-cycle, and 135 previous-observed-cycle edges.
- Future8 label homophily is high across adjacent observed cycles: next-cycle same-label fraction is 0.867 versus null mean 0.548, permutation p=0.001; same-cycle same-label fraction is 1.0 because labels are cycle-level.
- Front-motion descriptors show strong nearest-neighbor temporal autocorrelation: next-cycle radius2 slope rho=0.594, apparent diffusion proxy rho=0.594, q70 radius2 p50 rho=0.599, phase-slope positive fraction rho=0.622; all permutation p=0.001.
- The only strong source-feature to next-cycle future8 test is phase-slope positive fraction: next-neighbor AUC=0.682 versus null p95=0.593, permutation p=0.001.
- Distance-gradient tests do not show a simple shorter-distance enrichment for future8-positive pairs; next-cycle both-positive median distance is not significantly different from other edges (p=0.831).

Interpretation: nearby reconstructed ROI candidates preserve coherent front/rollout state across adjacent observed cycles, which is useful for spatial review prioritization and candidate propagation hypotheses. It is not particle tracking or causal propagation proof because ROI identities are automatic and future8 labels are cycle-level weak labels.

## 2026-05-22 Cross-Modal Degradation Consensus

Added `scripts/tier4_cross_modal_degradation_consensus.py` and ran it on Isambard to join cycle-level evidence from particle traces, integrated event evidence, hazard probabilities, balanced ROI-front physics, temporal directionality, electrochemical state-space transitions, and joint rollout/physics degradation summaries.

Key result:

- The audit scored 89 cycles; 53 cycles received at least one modal vote at the 0.85 rank threshold.
- Cycles 86 and 116 are the only `synchronized_multimodal_degradation_candidate` cycles. Each has 4 modal votes, synchronized trace drops, low frame-count/acquisition votes, and echem-state votes.
- Cycle 86 ranks first with consensus score 0.813, 4 modal votes, frame percentile 0.079, 3 dropping particles, max particle-trace delta 0.168, hazard probability 0.942, rollout residual energy 0.00171, and state-step norm 8.52.
- Cycle 116 ranks second with consensus score 0.795, 4 modal votes, frame percentile 0.011, 3 dropping particles, max particle-trace delta 0.262, integrated evidence score 12.93, and state-step norm 13.56.
- The synchronized candidate class has median consensus score 0.804, event rate 1.0, and median frame percentile 0.045; this reinforces the event hypothesis while keeping acquisition/frame-count as an explicit confounder.
- Four cycles are `multimodal_outlier_without_trace_drop`; cycles 150 and 151 are notable because both are future8-positive and combine high hazard probability, ROI-front/masked-residual signals, and large echem-state transitions without a current trace drop.
- Consensus score separates current abrupt-drop cycles from non-drop cycles (median 0.732 vs 0.492) but barely separates future8 labels by itself (median 0.508 vs 0.498). Hazard probability and masked residual signatures remain more relevant for future warning than the full consensus score.

Interpretation: cross-modal agreement strengthens cycles 86/116 as coordinated degradation candidates, but the same result also confirms that the strongest synchronized events are acquisition-confounded low-frame-count regimes. Use the consensus table as a QC and hypothesis-ranking index, not as a calibrated probability model.

## 2026-05-22 Echem Optical Breakpoint Audit

Added `scripts/tier4_echem_optical_breakpoint_audit.py` and ran it on Isambard to test whether synchronized optical degradation cycles align with local shifts in electrochemical and cycle-state trajectories. The audit uses compact cycle-level outputs from the cycle-state, particle-trace, integrated-event, and cross-modal consensus analyses rather than rescanning raw videos.

Key result:

- The audit scored 89 cycles and 84 cycle-level echem/trace/state features plus derivatives around synchronized event cycles 86 and 116, using 5,000 bootstrap/permutation draws.
- The strongest synchronized-event-centered shift is `state_step_norm_delta_prev` over +/-4 cycles: median post-pre shift -4.49, IQR-scaled shift -2.26, control-center p95 abs 1.77, empirical p=0.040, bootstrap p=0.0018.
- Other event-centered shifts include `mean_abs_delta_prev_delta_prev` (scaled -1.75, bootstrap p=0.0030), `axis_step_delta_prev` (scaled -1.46, bootstrap p=0.0056), and `cycle_state_pc2` over +/-8 cycles (scaled -1.39, bootstrap p=0.014).
- The future-warning signal is not identical to the event-centered signal: `cycle_state_pc2` separates future8 cycles from non-future8 cycles with median positive-negative 0.730, Mann-Whitney p=2.32e-4, rho=0.393.
- Future synchronized-drop cycles are again associated with low frame count/frame percentile: future_sync2 median frame percentile 0.183 vs 0.545, p=9.67e-4.
- The global breakpoint scan ranks late cycles around 149/150 as the largest electrochemical slope breakpoints, especially coulombic-efficiency and capacity/dQ features. This supports treating cycles 150/151 as late-stage echem transition candidates rather than assuming all electrochemical structure is concentrated at optical event cycles 86/116.

Interpretation: cycles 86/116 are not just isolated optical drops; they sit near measurable cycle-state/echem trajectory changes. However, the largest electrochemical breakpoints occur later near 149/150, so the mechanistic story is multi-stage: early synchronized optical degradation candidates plus later electrochemical trajectory collapse/outlier behavior. This remains a cycle-level association audit, not proof of causality or calibrated transport.
## 2026-05-22 Echem Optical Regime Atlas

Added `scripts/tier4_echem_optical_regime_atlas.py` and ran it on Isambard to organize cycle-level optical degradation evidence by electrochemical regime descriptors. The script joins the cycle-state/echem-shape table, cross-modal degradation consensus, balanced ROI-front physics, and temporal-directionality summaries, then derives charge/discharge capacity asymmetry, voltage peak hysteresis proxies, voltage-bin dQ/dV-proxy allocation, robust echem PCA axes, and an echem-optical priority score.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_optical_regime_atlas`

Local compact copy:

`derived_local/echem_optical_regime_atlas`

Key result:

- The atlas covers 89 cycles and 44 echem-regime features; 8 cycles have missing echem-shape values and 10 cycles have extreme-or-missing coulombic-efficiency fields, which are now explicit flags.
- Echem PC1 is dominated by coulombic-efficiency / signed charge imbalance terms, so it is best treated as an electrochemical-regime plus data-quality axis rather than a pure material state coordinate.
- The PC1-mid regime has the highest future8 rate: 0.379 versus 0.167 in PC1-high and 0.133 in PC1-low. It also has the highest median cross-modal consensus score (0.542).
- Multimodal outliers without current trace drops show a higher positive dQ-proxy peak voltage: `pos_dq_abs_peak_voltage` median +0.050 V versus other cycles, p=1.45e-4; `all_dq_abs_peak_voltage` gives the same +0.050 V shift, p=0.00538.
- Future8-positive cycles have lower capacity/charge throughput: `capacity_fraction_of_first` median shift -0.0207 (p=0.0329), `capacity_fade_from_first_mAh` shift +0.0137 mAh (p=0.0329), and `shape_charge_mAh_abs` shift -0.0253 mAh (p=0.0300).
- The strongest continuous echem-optical link is `shape_dVdt_abs_p95` versus cross-modal consensus score (rho=0.617, p=8.51e-10); it also correlates with modal vote count (rho=0.373, p=6.00e-4).
- Top echem-optical priority cycles are 150, 151, 126, 116, and 86. Cycles 150/151 remain late warning-window multimodal outliers, while 86/116 remain synchronized multimodal optical degradation candidates.

Interpretation: the echem regime atlas supports a multi-stage degradation picture. Synchronized optical events at 86/116 sit in a broader echem/trace transition landscape, while late cycles 150/151 look like electrochemical-regime outliers with future-drop risk but no same-cycle trace drop. The atlas is useful for hypothesis ranking and model conditioning, not calibrated dQ/dV, mechanistic phase-diagram proof, or diffusion validation.


## 2026-05-22 Automatic QC Triage Surrogate

Added `scripts/tier4_automatic_qc_triage_surrogate.py` and ran it on Isambard to turn the pending manual-QC bottleneck into an auditable automatic triage queue. The script consolidates the manual-QC label template, active-learning priority table, control-balanced front-QC sensitivity diagnostics, front-tracking metrics, and physics-consistency matrix. It preserves all manual labels as pending and assigns only surrogate triage tiers.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/automatic_qc_triage_surrogate`

Local compact copy:

`derived_local/automatic_qc_triage_surrogate`

Key result:

- The surrogate scored all 47 pending manual-QC candidates; manual status remains `pending` for all rows.
- Six candidates are `auto_surrogate_likely_interpretable`; five are event ROIs and all six have visual assets. The top likely-interpretable ROI is `cycle156_rank7_obj27` with score 0.919 and artifact risk 0.000.
- Ten candidates are `auto_surrogate_artifact_risk`, led by `cycle156_rank5_obj4` with score 0.598, artifact risk 0.625, fragmentation and diffusion-CI flags.
- Twenty candidates trigger the diffusion guardrail after incorporating suffixed control-balanced QC flags; artifact-risk rows have diffusion-guardrail rate 0.700 versus 0.167 in likely-interpretable rows.
- The likely-interpretable tier has median surrogate score 0.836 and median artifact risk 0.0625. Standard-review rows have median score 0.555 and median risk 0.375. Artifact-risk rows have median score 0.463 and median risk 0.500.
- Internal checks behave as expected: diffusion interpretability is strongly lower in diffusion-guardrail rows (median positive-negative -0.515, p=6.68e-9), front-mask score is higher in likely-interpretable rows (median shift 0.453, p=7.45e-7), and particle-identity score correlates with total surrogate score (rho=0.722, p=9.96e-9).

Interpretation: this closes the immediate review-prioritization gap without pretending to replace manual QC. The output identifies high-yield visual-review candidates and explicit artifact/diffusion guardrails, but it must not be used as particle identity, front-mask acceptance, diffusion interpretability, or degradation-mode validation.
## 2026-05-22 Echem-Conditioned Optical Predictor

Added `scripts/tier4_echem_conditioned_optical_predictor.py` and ran it on Isambard to test whether electrochemical regime descriptors add predictive signal for weak optical degradation targets beyond acquisition/frame-count context. The refreshed audit uses blocked cycle-CV plus rolling-origin block splits to avoid the expensive leave-one retraining loop. Rare 2-4 positive labels are excluded from this model-comparison table and remain descriptive atlas/consensus targets.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_conditioned_optical_predictor`

Local compact copy:

`derived_local/echem_conditioned_optical_predictor`

Key result:

- The refreshed audit covers 89 cycles, 5 weak optical targets, and feature sets of size 7 (`acquisition_context`), 49 (`echem_regime`), 56 (`echem_plus_acquisition`), and 14 (`cycle_state_upper_bound`).
- Blocked cycle-CV high cross-modal consensus prediction has acquisition-only AUC 0.730, echem-regime AUC 0.760, and echem+acquisition AUC 0.804. All three beat the 5,000 held-out score-label shuffle null, with echem+acquisition p=0.00020.
- Blocked cycle-CV high particle heterogeneity is where echem helps most: acquisition-only AUC 0.482, echem-regime AUC 0.634 (delta +0.152, p=0.0340), and echem+acquisition AUC 0.609 (delta +0.127, p=0.0632).
- Rolling-origin block checks are weaker but retain a small high-particle-heterogeneity gain for echem-regime over acquisition (+0.0358 AUC) and a high-consensus gain (+0.0266 AUC).
- High ROI phase-slope targets are highly separable but under-covered: only 24 cycles have the target, with acquisition-only AUC 0.944, echem-regime AUC 1.000, and echem+acquisition AUC 0.991.
- Future8 remains unstable and not a deployable echem warning result: blocked acquisition-only AUC is below random at 0.316, echem-regime rises to 0.440, and echem+acquisition is 0.333; none beats the score-label null.
- The `cycle_state_upper_bound` remains useful as an internal state-control check for optical state labels, reaching blocked-CV AUC 0.857 for high state-step targets and 0.846 for high cross-modal consensus; it does not rescue future8 in the refreshed blocked split.

Interpretation: electrochemical regime descriptors add conditional signal for broad cross-modal optical degradation state and particle heterogeneity, particularly when compared against acquisition-only context. They do not provide a standalone future-drop warning model; echem regime is a conditioning/context axis for optical AI models rather than causal mechanism proof or calibrated dQ/dV.

## 2026-05-22 Echem-Conditioned ROI Rollout/Front Audit

Added `scripts/tier4_echem_conditioned_roi_rollout_front_audit.py` and ran it on Isambard to test whether electrochemical regime descriptors explain ROI-level video-model difficulty and front-motion descriptors beyond acquisition/context variables. The audit joins the `echem_optical_regime_atlas` cycle table to `balanced_future_roi_physics_audit/balanced_future_roi_physics_joined.csv`, then compares `acquisition_context`, `echem_regime`, and `echem_plus_acquisition` feature sets under leave-one-cycle-out ridge regression.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_conditioned_roi_rollout_front_audit`

Local compact copy:

`derived_local/echem_conditioned_roi_rollout_front_audit`

Key result:

- The audit covers 72 ROI rows from 24 cycles, 12 continuous ROI targets, 10 acquisition/context features, and 49 echem-regime features.
- The clearest echem-conditioned win is the transferred masked-residual signature: acquisition-only leave-cycle performance is R2 -0.079 and Spearman -0.086, while echem-regime features reach R2 0.546 and Spearman 0.364 (delta R2 +0.626, delta Spearman +0.450).
- Adding echem to acquisition improves R2 further for the transferred masked-residual signature (R2 0.963, delta R2 +1.042), but the rank correlation is weaker than echem alone (Spearman 0.186), so this looks like a useful explanatory axis rather than a robust deployable predictor.
- Echem features add rank signal for phase-front positive fraction (delta Spearman +0.223 to +0.327), but R2 remains strongly negative. Treat those front-fraction effects as exploratory and sensitive to small-sample/high-dimensional regression.
- Particle rollout fractions are mostly better explained by acquisition/context: persistence particle MSE fraction has acquisition-only Spearman 0.685 and echem+acquisition Spearman 0.729, while echem alone has very poor R2.
- After residualizing acquisition/context, the transferred masked-residual signature is strongly linked to echem proxy shape terms: `pos_dq_abs_entropy` rho=-0.745, `all_dq_abs_peak_frac` rho=0.727, `all_dq_abs_entropy` rho=-0.649, and `neg_dq_abs_peak_voltage` rho=0.592, all with n=54.

Interpretation: the paper-inspired electrochemical-regime conditioning is most actionable for the masked residual signature, which is our strongest bridge between AI video reconstruction difficulty and battery state. Front-motion and diffusion-proxy descriptors remain useful screening variables, but the ROI-level evidence does not justify calibrated transport or phase-front claims without manual QC and stronger physical calibration.

## 2026-05-22 Echem Video Embedding Fusion Audit

Added `scripts/tier4_echem_video_embedding_fusion_audit.py` and ran it on Isambard to test whether label-free masked-video embeddings add degradation signal beyond electrochemical regime and acquisition/context variables. The audit fuses `masked_video_embedding_audit`, `echem_optical_regime_atlas`, and balanced ROI physics rows, then compares acquisition, echem, video-scalar, video-embedding, video-all, video+echem, and video+echem+acquisition feature sets under leave-one-cycle-out models.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_video_embedding_fusion_audit`

Local compact copy:

`derived_local/echem_video_embedding_fusion_audit`

Key result:

- The audit covers 172 masked-video rows across 34 cycles: 72 balanced-future, 52 selected event/control, and 48 transfer-ranked rows. Future-drop labels are available for 72 balanced rows across 24 cycles.
- For future8 classification, acquisition/context alone reaches AUC 1.000, confirming a strong cohort-design/frame/context confounder. Video-all features still carry signal beyond echem alone: AUC 0.823 versus echem AUC 0.632, delta +0.191; video+echem+acquisition reaches AUC 0.916 but does not beat acquisition-only.
- For future16 classification, video+echem is the best tested set: AUC 0.754 versus acquisition AUC 0.729, video-all AUC 0.697, and echem AUC 0.505. The video+echem gain over video-only is +0.057 AUC and +0.081 Spearman rho, with permutation p=0.002 for the fused model.
- For the transferred masked-residual signature, video+echem strongly improves over video-only in regression: R2 0.711 versus -1.832 and Spearman 0.583 versus 0.040. This independently supports echem conditioning for AI residual-difficulty interpretation.
- Scalar video descriptors outperform raw video PCA components for future labels in this small table: future8 video-scalar AUC 0.816 versus video-embedding-only AUC 0.569. The current PCA embedding is useful but not yet a sufficient learned video representation.

Interpretation: masked-video representations add meaningful weak-label degradation signal over echem alone, and echem helps video descriptors for longer-horizon future16 and masked-residual-signature modeling. However, acquisition/context can dominate future8 labels, so the result should guide representation design, balanced sampling, and review prioritization rather than deployment or causal mechanism claims.

## 2026-05-22 QC Decision Evidence Ledger

Added `scripts/tier4_qc_decision_evidence_ledger.py` and ran it on Isambard to consolidate the manual-QC workbook, automatic QC triage surrogate, physics-consistency matrix, active-learning queue, and available future-drop/front evidence into a reviewer-facing decision ledger. This does not assign manual labels; it turns the remaining QC bottleneck into explicit review actions with claim guardrails.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/qc_decision_evidence_ledger`

Local compact copy:

`derived_local/qc_decision_evidence_ledger`

Key result:

- The ledger covers all 47 pending manual-QC candidates, and all 47 have at least one visual asset path.
- Decision actions are: 3 `review_for_possible_accept_first`, 5 `high_priority_review`, 4 `review_artifact_or_reject_first`, 16 `review_but_diffusion_guarded`, and 19 `routine_pending_review`.
- The top possible-accept queue is `cycle156_rank7_obj27`, `cycle156_rank8_obj10`, and `cycle156_rank2_obj2`; all combine automatic likely-interpretable status with cross-modal physics support and front/diffusion-proxy candidate evidence.
- The top artifact/reject-first queue is led by `cycle156_rank5_obj4`, followed by `cycle86_rank4_obj9`, `cycle157_rank2_obj2`, `cycle158_rank2_obj1`, and `cycle86_rank5_obj8`.
- Cycle 156 is the highest-yield immediate review group because it contains all three possible-accept-first candidates plus the strongest artifact-risk counterexample.

Interpretation: this reduces the manual-QC bottleneck without pretending to automate acceptance. The immediate practical next review is a small cycle-156 panel: accept/reject `cycle156_rank7_obj27`, `cycle156_rank8_obj10`, and `cycle156_rank2_obj2` first, then inspect `cycle156_rank5_obj4` as an artifact-risk foil. No calibrated diffusion or validated degradation-mode claim is allowed until manual particle identity and front-mask checks are recorded.

## 2026-05-22 Residual Dictionary Video Embedding Audit

Added `scripts/tier4_residual_dictionary_embedding_audit.py` and ran it on Isambard to test a faster learned temporal-residual representation after the raw masked-video PCA components underperformed. The audit learns a label-free PCA dictionary on next-frame residual fields from selected, transfer-ranked, and balanced ROI crops, summarizes coefficient trajectories per ROI, and evaluates them under leave-one-cycle weak-label splits.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/residual_dictionary_embedding_audit`

Local compact copy:

`derived_local/residual_dictionary_embedding_audit`

Key result:

- The audit covers 172 ROI videos across 34 cycles: 72 balanced-future, 52 selected event/control, and 48 transfer-ranked rows.
- The residual dictionary uses 800 sampled residual frames at 24x24 resolution, rank 8, and explains 59.9% of sampled next-frame residual variance.
- Residual-dictionary features beat raw masked-video PCA components for future8 classification: AUC 0.663 versus 0.566, delta +0.096; the fused residual-dictionary+handcrafted feature set reaches AUC 0.771 versus PCA AUC 0.566, delta +0.204.
- Handcrafted scalar video descriptors remain strongest for future8 in this small table: AUC 0.825 and AP 0.865. Adding the residual dictionary to them reduces AUC to 0.771, so the dictionary is useful as a temporal-residual representation but not yet a better weak-label classifier.
- Future16 remains weak for the residual dictionary: AUC 0.473 alone and 0.527 when fused with handcrafted features. The dictionary does not reproduce the echem+video future16 gain seen in the echem-video fusion audit.
- The residual dictionary correctly reconstructs its own residual-energy quantities under leave-cycle regression (rho about 0.997-0.999 for residual energy and reconstruction-error targets), which validates the feature extraction pipeline but is not an external degradation result.

Interpretation: a label-free next-frame residual basis is more informative than raw video PCA for future8 weak-label triage, but it does not beat the simpler hand-engineered particle/mask temporal descriptors. This suggests the next neural-video step should not merely learn a generic residual basis; it should be echem/context-conditioned or trained with a physics-aware objective around front motion and particle-local residual timing.

## 2026-05-22 Agentic Current Hypothesis Tournament

Added `scripts/tier4_agentic_current_hypothesis_tournament.py` and ran it on Isambard to turn the three new Nature AI-scientist papers into a concrete project tool: generator-style hypothesis cards, skeptical guardrail scoring, tournament ranking, and machine-readable next-experiment specs grounded in the latest NMC photometry evidence.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/agentic_current_hypothesis_tournament`

Local compact copy:

`derived_local/agentic_current_hypothesis_tournament`

Key result:

- The tournament ranks 8 current hypotheses using evidence support, novelty, falsifiability, implementability, guardrail penalty, and cost.
- Rank 1 is echem-conditioned video residuals for longer-horizon weak-label signal: future16 video+echem AUC 0.754 versus echem-only AUC 0.505, empirical p=0.002. The next proposed experiment is an acquisition-residualized video+echem future16 audit with leave-source and leave-cycle splits.
- Rank 2 is the skeptical confound hypothesis: future8 is acquisition/context dominated, with acquisition/context AUC 1.000 versus video-all AUC 0.823. The next proposed experiment is a residualized future8 benchmark before using that label mechanistically.
- Rank 3 is the lab-in-the-loop QC hypothesis: review the cycle-156 mini-batch first because the ledger has 47 pending candidates, including 3 possible-accept-first and 4 artifact/reject-first candidates, with cycle 156 containing both high-support candidates and an artifact foil.
- The emitted `agentic_next_experiment_specs.json` gives success criteria and do-not-claim guardrails for the top five next analyses.

Interpretation: the paper-inspired agentic layer is now implemented as an auditable planning and prioritization tool over our real derived evidence, not a free-form LLM report. It points to the next best computational step as acquisition-residualized video+echem future16 modeling, while explicitly preserving the strongest guardrails: no calibrated diffusion claim without accepted masks/calibration, no deployable warning model from weak labels alone, and no mechanistic future8 claim while acquisition/context remains dominant.

## 2026-05-22 Echem Residual-Dictionary Fusion Audit

Added `scripts/tier4_echem_residual_dictionary_fusion_audit.py` and ran it on Isambard to test whether label-free residual-dictionary video descriptors become more useful after conditioning on echem regime and acquisition/context variables. Before rerunning this audit, I repaired the residual-dictionary output table to the full 172 ROI rows after a bounded diagnostic run had temporarily overwritten the remote feature CSV with a 45-row subset.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_residual_dictionary_fusion_audit`

Local compact copy:

`derived_local/echem_residual_dictionary_fusion_audit`

Key result:

- The refreshed fusion audit covers 172 ROI videos across 34 cycles, matching the repaired residual-dictionary table.
- For future8 labels, acquisition/context alone reaches AUC 1.000/AP 1.000, confirming that this weak label is strongly cohort/context encoded.
- Residual dictionary plus echem/context reaches future8 AUC 0.917/AP 0.958, improving over residual dictionary alone by +0.248 AUC and +0.430 Spearman rho, but still below the acquisition-only upper confound.
- Residual dictionary plus echem/context improves future16 over residual dictionary alone by +0.196 AUC and +0.276 Spearman rho; residual dictionary + handcrafted + echem improves future16 over residual dictionary + handcrafted by +0.230 AUC.
- Echem/context also improves raw PCA-video future8 strongly (+0.347 AUC), showing that much of the gain is context conditioning rather than uniquely residual-dictionary physics.
- Regression readouts support echem as a context bridge for residual descriptors: particle heterogeneity (`particle_norm_cv`) improves from poor residual-dictionary-only regression to residual_dictionary_plus_echem R2 0.327 / Spearman 0.815, while handcrafted scalars remain strongest for that target.

Interpretation: echem/acquisition conditioning makes residual-basis video descriptors more predictive for weak future labels and particle-state readouts, but the perfect acquisition-context future8 score is a hard leakage/design guardrail. The useful claim is representation conditioning for review prioritization and model design, not deployable early warning or causal echem mechanism proof.

## 2026-05-22 Acquisition-Residualized Video Physics Benchmark

Added `scripts/tier4_residualized_future8_video_physics_benchmark.py` and ran it on Isambard to test the skeptical follow-up from the echem residual-dictionary fusion audit: whether particle-region video descriptors retain weak future-label signal after acquisition/context residualization.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/acquisition_residualized_video_physics_benchmark`

Local compact copy:

`derived_local/acquisition_residualized_video_physics_benchmark`

Key result:

- The benchmark covers the same 172 ROI rows across 34 cycles as the repaired residual-dictionary/echem fusion table, with 72 evaluable balanced weak-label rows over 24 cycles.
- Future8 remains acquisition/context dominated: acquisition context alone reaches AUC 1.000/AP 1.000 with cycle-block permutation p=0.002, while raw all-video reaches AUC 0.756/AP 0.814.
- After residualizing all video features against acquisition context, future8 all-video alone falls to AUC 0.319/AP 0.398. Adding the context logit back recovers AUC 0.922/AP 0.959, showing the predictive power is carried by context rather than context-independent video residuals.
- Future16 has modest raw video signal: handcrafted particle descriptors reach AUC 0.796/AP 0.931, all-video reaches AUC 0.733/AP 0.913, and echem-context plus all-video improves over echem-context by +0.097 AUC. But residualized all-video alone is only AUC 0.620/AP 0.873, and residualized all-video plus context logit is below acquisition context by -0.188 AUC.
- The strongest context-residual feature tests are future16 front/diffusion-proxy descriptors (`roi_threshold_robust_diffusion_score` and `threshold_robust_diffusion_score`, direction-free AUC 0.821), still under the same apparent-optical-proxy/manual-QC guardrail.

Interpretation: this is an important negative/guardrail result. Future8 should not be treated as a context-independent video-physics degradation detector in the current weak-label design. Future16 remains the more plausible target for echem-conditioned video modeling, but the evidence should be framed as review-prioritization and hypothesis generation until source/cohort holdouts and manual-QC labels support it.

## 2026-05-22 Acquisition-Residualized Video/Echem Warning Audit

Added `scripts/tier4_acquisition_residualized_video_echem_warning.py` and ran it on Isambard to execute the top current-evidence tournament recommendation: test whether the video+echem future16 signal survives explicit acquisition/context residualization and stricter source holdouts. Candidate echem/video features are residualized against acquisition/context inside each held-out fold before classification.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/acquisition_residualized_video_echem_warning`

Local compact copy:

`derived_local/acquisition_residualized_video_echem_warning`

Key result:

- The audit covers 172 ROI rows, 34 cycles, and 12 source movies, with leave-cycle, leave-source, and source-cohort-key splits.
- Under leave-cycle future16 evaluation, acquisition-residualized video+echem reaches AUC 0.788, AP 0.932, empirical p=0.002, versus acquisition/context-only AUC 0.727. The cycle-balanced residualized mode gives the same AUC/AP in this table, so the cycle-holdout signal is not driven by repeated-cycle weighting.
- In the same leave-cycle setting, residualized video+echem beats raw echem-only by +0.136 AUC and raw video-only by +0.091 AUC, supporting the paper-inspired multimodal conditioning hypothesis.
- Under leave-source future16 evaluation, the signal does not transfer: acquisition-residualized video+echem falls to AUC 0.527, while acquisition/context-only is AUC 0.697. This makes source/movie domain shift a major guardrail.
- Under the stricter source-cohort-key holdout, cycle-balanced residualized video+echem reaches only AUC 0.560/AP 0.837, empirical p=0.248, below acquisition/context-only AUC 0.782. This fails the source-cohort transfer criterion for a deployable warning claim.
- Residualized video-only collapses under leave-cycle future16 (AUC 0.469), so the useful residualized signal is mainly electrochemically conditioned, not generic optical video structure alone.

Interpretation: the top tournament experiment is now executed with its stricter cycle-balanced/source-cohort guardrail. Echem+video carries future16 weak-label signal beyond linear acquisition context when held out by cycle, but it is not yet source- or source-cohort-transferable. The next defensible modeling step is source-domain adaptation or source-balanced sampling before any warning-model claim; the current result should guide representation design and review prioritization only.

## 2026-05-22 Source-Domain Video/Echem Adaptation Audit

Added `scripts/tier4_source_domain_video_echem_adaptation_audit.py` and ran it on Isambard to test whether simple source-domain adaptation can rescue the failed leave-source future16 transfer from the acquisition-residualized warning audit. The audit compares raw, source-centered, CORAL-aligned, acquisition-residualized, and combined variants across acquisition context, video-only, echem-only, and video+echem feature sets.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_domain_video_echem_adaptation_audit`

Local compact copy:

`derived_local/source_domain_video_echem_adaptation_audit`

Key result:

- The audit covers 172 ROI rows, 34 cycles, and 12 source movies; the future16 labels are highly source concentrated, with `17_c2_x10_HighHighCOV_150723` at 30/30 positives and several 3-row sources all negative.
- Acquisition-only leave-source reaches AUC 0.697/AP 0.922 with empirical p=0.012, confirming that source/context structure remains predictive.
- Raw video+echem leave-source is weak at AUC 0.595/AP 0.868, matching the earlier source-transfer warning.
- Source-centered video+echem reaches AUC 0.737/AP 0.931 with empirical p=0.006, improving over acquisition-only by +0.040 AUC and +0.009 AP.
- CORAL alignment fails for this small, imbalanced source layout: video+echem CORAL reaches only AUC 0.420, and acquisition-residualized CORAL/source-centering variants do not help.

Interpretation: unlabeled source-centering partially rescues future16 source transfer, so source-specific feature offsets/domain calibration matter. The result is still not a deployable warning model because source labels and outcomes remain severely entangled; the next defensible step is source-balanced sampling or explicit domain-invariant representation learning, ideally anchored by manual QC labels.

## 2026-05-22 Source-Balanced Video/Echem Transfer Audit

Added `scripts/tier4_source_balanced_video_echem_transfer_audit.py` and ran it on Isambard to test whether source/class weighting and within-source rank normalization make the video+echem future16 signal source-transferable after the source-domain adaptation result.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_video_echem_transfer_audit`

Local compact copy:

`derived_local/source_balanced_video_echem_transfer_audit`

Key result:

- The audit covers 172 ROI rows, 34 cycles, and 12 source movies, with four feature sets: acquisition context (28 features), echem regime (55), video-only (64), and video+echem (119).
- Source composition is the dominant limitation: `17_c2_x10_HighHighCOV_150723` contributes 30 labeled future16 rows and all are positive, while several small sources are all positive, all negative, or unlabeled for future16.
- Future16 acquisition-context raw leave-source remains strongest at AUC 0.704/AP 0.915 with empirical p=0.004.
- Future16 echem source-rank reaches AUC 0.642/AP 0.898 with empirical p=0.032.
- Future16 video+echem improves only modestly from raw AUC 0.594/AP 0.850 to source-rank weighted AUC 0.614/AP 0.868 with empirical p=0.068, still below acquisition context and echem source-rank.
- Video-only future16 remains weak under source holdout at AUC 0.454/AP 0.794.

Interpretation: source-rank normalization and source/class weighting are useful stress tests but do not solve the source-transfer problem. The stronger source-centered adaptation result suggests source offsets matter, while this audit shows simple balancing does not yield a deployable source-transferable warning model. Treat video+echem as a review-prioritization and representation-design signal until we have better balanced sources and manual QC labels.

## 2026-05-22 Learned Video Residual Embedding Audit

Added `scripts/tier4_learned_video_residual_embedding_audit.py` and ran it on Isambard as a label-free next-frame residual representation experiment. The model trains a small residual CNN to predict frame-to-frame residuals from current ROI frames, then summarizes latent channels and residual errors per ROI before evaluating weak labels with leave-cycle splits.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/learned_video_residual_embedding_audit`

Local compact copy:

`derived_local/learned_video_residual_embedding_audit`

Key result:

- The audit covers 172 ROI rows across 34 cycles: 52 selected event/control rows, 48 transfer-ranked rows, and 72 balanced-future rows.
- Training used 7000 frame pairs at downsample 2 with stride 2, 12 channels, and 14 epochs; best validation residual MSE was 4.37e-4.
- Future8 leave-cycle weak-label classification is strong for learned residual-CNN features: `learned_all` AUC 0.849/AP 0.872 with permutation p=0.001, versus PCA-video AUC 0.569/AP 0.622 and handcrafted scalar AUC 0.828/AP 0.867.
- Learned latent features alone reach future8 AUC 0.829/AP 0.850, so the encoder carries useful short-horizon degradation information beyond simple PCA video bins.
- Future16 remains weak for the learned residual representation: `learned_all` AUC 0.538/AP 0.842, while handcrafted scalar features remain stronger at AUC 0.680/AP 0.910.
- The learned features predict their own residual-energy/error targets under leave-cycle regression, confirming the feature extraction path, but that is an internal reconstruction-consistency result rather than external degradation physics.

Interpretation: a small self-supervised residual CNN is now a useful AI representation for short-horizon weak degradation ranking and clearly beats the earlier PCA-video embedding on future8. It does not solve longer-horizon future16/source-transfer behavior, and it does not validate manual particle/front labels or calibrated diffusion. The next modeling use should be source-aware/echem-conditioned residual-CNN features or using these embeddings to prioritize manually QC-reviewed ROI/front candidates.

## 2026-05-22 Source-Invariant Video/Echem Transfer Audit

Added `scripts/tier4_source_invariant_video_echem_transfer_audit.py` and ran it on Isambard to test a stronger source-transfer remedy than weighting or within-source rank normalization. The audit removes training-source mean directions (`source_mean_resid_k`) and filters source-confounded features by train-source eta-squared before leave-source prediction, without using held-out source labels.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_invariant_video_echem_transfer_audit`

Local compact copy:

`derived_local/source_invariant_video_echem_transfer_audit`

Key result:

- The audit covers 172 ROI rows, 34 cycles, and 12 source movies, using the same 72 labeled leave-source evaluation rows over 24 cycles and 9 sources for the weak-label tests.
- For future16, raw video+echem reaches AUC 0.612/AP 0.858/p=0.088, still below acquisition context raw AUC 0.745/AP 0.937/p=0.002.
- Removing the top four training-source mean directions improves video+echem future16 to AUC 0.729/AP 0.927/p=0.004, a +0.117 AUC gain over raw video+echem but still just below acquisition context.
- Video-only features respond strongly: source-confound filtering of the top 50% source-predictive video features reaches future16 AUC 0.770/AP 0.919/p=0.004, a +0.281 AUC gain over raw video-only.
- Future8 remains acquisition/context dominated: acquisition context is AUC 1.000, while best video+echem source-invariant future8 is AUC 0.784 and source-mean removal can also destroy the signal when too many components are removed.

Interpretation: source-invariant projection is the best source-transfer rescue so far for particle-region video/echem features, especially for future16 and video-only descriptors. It still does not clear the warning-model guardrail because acquisition context remains competitive and source/outcome composition is severely entangled. The result supports pursuing explicit domain-invariant video representations and manual-QC-gated targets, not deployable weak-label source transfer.

## 2026-05-22 Source-Invariant Physical Family Audit

Added `scripts/tier4_source_invariant_physical_family_audit.py` and ran it on Isambard to decompose the source-invariant future16 rescue into particle-region physical feature families. The audit evaluates leave-source weak future labels for particle intensity, particle-vs-context contrast, particle gradients, normalized heterogeneity, video embedding PCs, handcrafted particle descriptors, and all video features under raw, source-mean residualized, and source-confound filtered variants.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_invariant_physical_family_audit`

Local compact copy:

`derived_local/source_invariant_physical_family_audit`

Key result:

- The audit covers 172 ROI rows, 34 cycles, and 12 source movies, using 72 leave-source evaluation rows over 24 cycles and 9 sources.
- For future16, normalized heterogeneity is the strongest interpretable family after source-mean projection: `norm_heterogeneity` with `source_mean_resid_4` reaches AUC 0.738/AP 0.917/p=0.002, a +0.433 AUC gain over its raw family readout.
- Particle-vs-context contrast is the next strongest interpretable family: `particle_vs_context` with `source_mean_resid_4` reaches AUC 0.703/AP 0.903/p=0.010.
- The best all-video family result remains `all_video` with source-confound filtering at AUC 0.770/AP 0.919/p=0.002, but the family decomposition shows that normalized heterogeneity and contrast carry most of the interpretable future16 signal.
- Raw self-supervised video embedding alone is weak for future16 (AUC 0.462/AP 0.753), so the current source-invariant rescue is not just opaque embedding capacity.
- Source-confounded features are dominated by derivative-like intensity/gradient descriptors, especially `particle_gradient_diff_abs_mean` (eta2 0.811), `particle_gradient_std` (eta2 0.789), and particle intensity diff quantiles (eta2 about 0.67).

Interpretation: the source-transfer rescue now has a more physical handle. The future16 particle-region signal that survives source-mean removal is concentrated in normalized heterogeneity and particle-vs-context contrast, consistent with degradation-mode/phase-front texture changes rather than raw brightness drift alone. It remains a review-prioritization result under weak labels and automatic masks, not a calibrated mechanism or warning model.

## 2026-05-22 Diffusion Physics Consistency Audit

Added `scripts/tier4_diffusion_physics_consistency_audit.py` and reran it on Isambard to collapse the apparent-diffusion threshold sweep into ROI-level physics gates. The audit asks whether any automatic optical-front ROI satisfies the minimum internal conditions needed before interpreting radius-squared slopes as material-like diffusion: positive expansion across thresholds, adequate linear fit, low threshold sensitivity, stable HDF5 timing, low drift, positive q70 confidence interval, and manual-QC acceptance.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/diffusion_physics_consistency_audit`

Local compact copy:

`derived_local/diffusion_physics_consistency_audit`

Key result:

- The audit covers 72 ROI rows, 24 cycles, 9 sources, and 504 threshold rows from the apparent-diffusion calibration table.
- All 72 ROIs have the full threshold set and 63/72 pass threshold-stability, but only 26/72 pass positive expansion, 7/72 pass the radius-squared fit-quality gate, 24/72 pass stable HDF5 timing, and 0/72 pass the q70 positive-confidence-interval gate.
- Only 1/72 ROIs is automatically physics-consistent (`cycle78_rank22_obj2`), and 0/72 are publication-ready diffusion candidates because q70 CI and manual-QC gates fail.
- The median absolute apparent D is 2.23e-06 um2/s, median positive-D fraction is 0.714, and median radius-squared fit R2 is only 0.055.
- Future16 positives have lower positive-D fraction and lower median signed apparent D than negatives (positive-D fraction oriented AUC 0.705, p=0.013; median apparent D oriented AUC 0.701, p=0.018), which is consistent with optical-front loss/contraction rather than clean expanding diffusion.
- The transferred masked residual signature remains a stronger weak-label correlate than diffusion gates for future8/future16, while physics consistency scores are negatively correlated with that residual signature.

Interpretation: calibrated diffusion remains blocked. The apparent-D quantities are useful as optical-front consistency and guardrail features, but they do not satisfy internal physics gates for material diffusion coefficients. This strengthens the project rule that diffusion claims need manual QC, better front masks, stable timing, and positive estimator agreement before promotion.

## 2026-05-22 Source-Invariant Exact Feature Audit

Added and ran:

`scripts/tier4_source_invariant_interpretable_feature_audit.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_invariant_interpretable_feature_audit`

Local compact copy:

`derived_local/source_invariant_interpretable_feature_audit`

Key result:

- The audit evaluated 72 ROI rows across 24 cycles and 9 sources using weak future16 labels under leave-source splits.
- The strongest univariate descriptor is `particle_vs_context_mean_diff_positive_fraction`: lower values are associated with future16 positives, oriented AUC 0.769, AP 0.904, source eta2 0.390, and median positive-minus-negative -0.0526.
- The best single-feature leave-source transfer model is the same descriptor with AUC 0.727, AP 0.880, empirical permutation p = 0.00798.
- The best small feature set is `particle_vs_context_mean_diff_positive_fraction + particle_mean_last_minus_first + particle_gradient_diff_q90`, with leave-source AUC 0.750, AP 0.905, empirical permutation p = 0.001996.
- Other high-ranking exact descriptors include lower `particle_std_diff_positive_fraction`, higher particle-vs-context/particle-mean last-minus-first trends, and normalized intensity features, but several trend/intensity features remain strongly source-confounded.

Interpretation: the source-invariant family signal now has an exact descriptor hypothesis: future16 positives tend to show fewer positive particle-vs-context frame-to-frame changes, consistent with optical-loss/contraction-like behavior rather than clean expanding fronts. This remains a hypothesis-prioritization result only because labels are weak, acquisition/source imbalance persists, masks are automatic, and manual QC is still absent.

## 2026-05-22 Invariant Physics Rule Discovery

Added and ran:

`scripts/tier4_invariant_physics_rule_discovery.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/invariant_physics_rule_discovery`

Local compact copy:

`derived_local/invariant_physics_rule_discovery`

Key result:

- The rule audit screened 172 joined rows and evaluated 72 future16-labeled rows across 24 cycles and 9 sources.
- It generated 171 sparse candidate rules from particle optical, front/rollout, echem-state, and acquisition-context descriptors.
- The highest-priority sparse rule is `low(particle_std_diff_positive_fraction)`: it covers 27/72 rows, precision 0.889 against a 0.792 base positive rate, recall 0.421, binary AUC 0.611, Fisher greater-tail p = 0.099, and positive hits in 6 sources.
- A related rule, `low(particle_vs_context_mean_diff_positive_fraction)`, covers 21/72 rows with precision 0.905, recall 0.333, lift 1.143, and positive hits in 5 sources.
- The strongest oriented single features include echem capacity/state descriptors as well as optical-difference descriptors; several echem features have higher AUC but also substantial source/context structure.

Interpretation: sparse rules reinforce the exact-feature audit by flagging low positive optical-change fractions as a review-prioritization signature. They are not deployable warning rules: precision gains are modest over the high future16 base rate, source support is uneven, and manual QC/acquisition controls remain required.

## 2026-05-22 Exact Feature Mechanism Consistency Audit

Added and ran:

`scripts/tier4_exact_feature_mechanism_consistency_audit.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/exact_feature_mechanism_consistency_audit`

Local compact copy:

`derived_local/exact_feature_mechanism_consistency_audit`

Key result:

- The audit joined exact particle-region descriptors to temporal directionality, front/rollout, echem-conditioned, and diffusion-consistency tables. The overlap covers 172 rows, 34 cycles, and 12 sources, with 72 rows carrying the balanced front/diffusion mechanism fields.
- The composite `exact_optical_loss_score` is a strong future16 weak-label separator: AUC 0.853, AP 0.960, median positive-minus-negative 0.730, p = 2.998e-05.
- That composite is also source-structured: source eta2 is 0.513, so it is not a source-independent degradation detector.
- The primary exact descriptor, `particle_vs_context_mean_diff_positive_fraction`, remains useful for future16 (oriented AUC 0.769), but its direct source-residualized link to radius-squared slope is weak (rho -0.047).
- The composite score correlates with radius2 slope after source residualization (rho 0.404, p = 4.31e-04) and has high-stratum shifts in rollout residual and radius/diffusion-proxy features, but it does not validate a clean contraction mechanism: front-contraction composite linkage is negative after source residualization (rho -0.398), and diffusion proxy quantities remain apparent/guarded.

Interpretation: this audit prevents overclaiming. Exact particle-region descriptors are useful warning/review features, but the mechanistic story is not simply clean optical-front contraction or calibrated diffusion. The next strongest interpretation is a source/context-sensitive optical-change mode with rollout-residual and radius-proxy associations that needs manual QC and calibration before physical mechanism claims.

## 2026-05-22 Signed Optical-Loss Mechanism Audit

Added `scripts/tier4_signed_optical_loss_mechanism_audit.py` and ran it on Isambard to turn the contraction/loss hypothesis into explicit signed axes: signed optical loss, front contraction, rollout difficulty, echem degraded state, and a combined loss-mechanism axis.

Remote output:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/signed_optical_loss_mechanism_audit`

Local compact copy:

`derived_local/signed_optical_loss_mechanism_audit`

Key result:

- The audit covers 172 total rows and 72 future16-evaluable rows over 24 cycles and 9 sources.
- For future16, the combined loss-mechanism axis is very separable (AUC 0.989/AP 0.997, p=6.79e-9), but it has moderate source eta2 0.556 and therefore remains a guarded synthesis statistic.
- Future16 is also separated by the echem degraded-state axis (AUC 0.821/AP 0.939, eta2 0.055) and the signed optical-loss axis (AUC 0.815/AP 0.940, eta2 0.684).
- Leave-source axis models show all loss-mechanism axes together reach future16 AUC 0.927/AP 0.984; echem-only reaches AUC 0.789 and optical-loss-only reaches AUC 0.732.
- Future8 behaves differently: the best leave-source axis model is optical plus front contraction (AUC 0.797), and the front-contraction axis alone reaches AUC 0.750, suggesting short-horizon labels are more front-motion dominated than future16.
- Mechanism clustering yields an echem-degraded-state-dominant group (39 rows, future16 rate 0.795, future8 rate 0.718), a front-contraction-dominant group (30 rows, future16 rate 0.767, future8 rate 0.167), and a tiny rollout-difficulty-dominant source-local group (3 rows, future16/future8 rate 1.0).

Interpretation: this is the clearest signed-mechanism split so far. Future16 looks like a combined echem-degraded plus optical-loss state, while future8 is more tied to front-contraction/proxy motion. The result is still not a deployable warning or calibrated transport model because weak labels, automatic masks, source structure, and absent manual QC remain limiting.
## 2026-05-22 Cycle-State Mode-Frequency Bridge

Added and ran:

`scripts/tier4_cycle_state_mode_frequency_bridge.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_mode_frequency_bridge`

Local compact copy:

`derived_local/cycle_state_mode_frequency_bridge`

Key result:

- The audit collapsed automatic ROI degradation-mode assignments to 11 cycle-level mode-frequency rows covering 52 ROI rows and 4 mode-fraction targets.
- It tests the agentic hypothesis that cycle-state transitions organize ROI degradation modes, using leave-one-cycle ridge models and explicit cycle/acquisition context baselines.
- The best macro mode-frequency model is `cycle_state_only`: macro MAE 0.261 across held-out cycles.
- The context-only baseline has macro MAE 0.303, so cycle-state features reduce macro MAE by 0.043 in this small cohort.
- The compact permutation null is not significant: cycle-state-only observed macro MAE 0.261 versus null mean 0.290, empirical p = 0.381 with 20 permutations.
- For the optical-brightening/decorrelating/rollout-hard/front-positive mode, echem-plus-context and cycle-state/echem/context models give the best per-mode fits (MAE 0.173 and 0.178; the latter has rho 0.676, p = 0.0225), but sample size is only 11 cycles.
- Cycle-state cluster 1 carries most sampled ROI rows (42/52) and has higher mixed mode fractions, while cluster 0 is smaller and mostly near-baseline/context-like.

Interpretation: cycle-state coordinates appear useful for organizing automatic ROI degradation-mode composition beyond simple context, but the evidence is exploratory rather than confirmatory. The model improves over context-only on macro MAE, yet the permutation null and tiny 11-cycle cohort prevent a strong claim. This should guide which cycles/modes to expand and manually QC next, not be treated as a causal degradation-mode predictor.

## 2026-05-22 Signed Loss Source Robustness Audit

Added and completed:

`scripts/tier4_signed_loss_source_robustness_audit.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/signed_loss_source_robustness_audit`

Local compact copy:

`derived_local/signed_loss_source_robustness_audit`

Key result:

- The audit covers 172 rows, 34 cycles, and 12 sources; 72 rows have future16 weak labels across 9 labeled sources.
- The combined loss-mechanism axis is extremely strong raw (future16 AUC 0.989, balanced source bootstrap mean AUC 0.989), but the source-mean-only transform also reaches AUC 0.942, proving that source composition is a major part of the raw separation.
- After source residualization, the combined axis drops to AUC 0.558 and within-source rank drops to AUC 0.551, so the combined raw axis should not be interpreted as a source-independent detector.
- The signed optical-loss axis raw AUC is 0.815 with high source eta2 0.684. Source-mean-only is still AUC 0.774, source-residual AUC is 0.656, within-source centered-z AUC is 0.607, and within-source rank AUC is 0.514.
- The echem degraded-state axis is less source-structured in raw form (eta2 0.055) and remains the strongest source-residualized future16 axis among the signed axes tested: raw AUC 0.821 and source-residual AUC 0.716.
- Source influence is not dominated by one single source: dropping `17_c2_x10_HighHighCOV_150723` changes the combined-axis AUC by only -0.012, while signed optical-loss is most reduced by dropping `15_c2_x5_HighCOV_120723` or `18_c2_xN_HighHighCOV_170723` (delta -0.046 each).

Interpretation: the signed-loss mechanism result is useful, but this audit moves the claim boundary from “source-transferable warning axis” to “source/context-sensitive degradation-state triage axis.” The echem degraded-state axis is the most robust residual signal, while the optical-loss and combined axes need balanced sources, manual QC, and source-aware calibration before any source-independent warning or mechanism claim.

## 2026-05-22 Echem/Optical Source-Residual Audit

Added and ran:

`scripts/tier4_echem_optical_source_residual_audit.py`

Remote output directory:

`/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_optical_source_residual_audit`

Local compact copy:

`derived_local/echem_optical_source_residual_audit`

Key result:

- The audit combines source-normalized echem-degraded, signed optical-loss, and front-contraction axes from the signed-loss source robustness table.
- Direct source-residual scores recover low-source-eta future16 evidence: echem+optical+front residual AUC 0.809/AP 0.951/source eta2 0.006; echem+optical residual AUC 0.774/AP 0.943/source eta2 0.002.
- Individual source-residual axes are weaker: echem residual AUC 0.716/AP 0.915 and optical residual AUC 0.656/AP 0.899.
- Leave-source logistic evaluation is more conservative: echem+optical source-residual model reaches AUC 0.708/AP 0.922, while echem-only is 0.558 and optical-only is 0.611.
- The raw/source-mean guardrail remains much stronger than source-normalized evidence, so source-normalized echem+optical features should be treated as review evidence rather than a deployable warning model.

Interpretation: after removing much of the source composition signal, echem-degraded state and optical-loss residuals do add complementary weak-label information. This is the most defensible current source-normalized warning signature, but it still relies on weak labels, unlabeled source normalization, and automatic ROI/front descriptors.
## 2026-05-22 Echem-Conditioned Residual Dictionary Audit

Added and ran:

`scripts/tier4_echem_conditioned_residual_dictionary.py`

Output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_conditioned_residual_dictionary`
- `derived_local/echem_conditioned_residual_dictionary`

Key result:

- The audit uses 172 ROI rows across 34 cycles and 12 sources, with 39 residual-dictionary features and 119 echem/acquisition/context conditioning features.
- Echem/context prediction explains a substantial part of some residual bases, especially leave-cycle `resdict_pc03_std` (R2 0.614, rho 0.783) and `resdict_pc04_std` (R2 0.102, rho 0.788).
- For future16, split-specific conditioning rescues residual-dictionary transfer. Leave-source conditioned residual dictionary reaches AUC 0.785 versus raw residual dictionary AUC 0.058, a +0.726 AUC delta; leave-cycle conditioned residual dictionary reaches AUC 0.695 versus raw 0.481.
- Adding echem/context to conditioned residuals gives leave-cycle future16 AUC 0.834 and leave-source AUC 0.651. The strongest leave-cycle future16 set is conditioned residual plus handcrafted plus echem/context at AUC 0.848.
- Future8 remains context dominated: echem/context and context-augmented feature sets are near AUC 0.917 under leave-cycle, while conditioned residual dictionary alone is weak for future8.

Interpretation: the useful representation is not raw learned video residuals alone. Residual dictionary modes become more informative after subtracting echem/acquisition-predictable structure, especially for longer-horizon future16 and leave-source transfer. This supports a context-conditioned residual objective for future AI models, but it remains a weak-label, automatic-mask, non-calibrated audit rather than a deployable warning or causal degradation mechanism.

## 2026-05-22 Source-Balanced ROI Expansion Manifest

Added and ran:

`scripts/tier4_source_balanced_roi_expansion_manifest.py`

Output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_roi_expansion_manifest`
- `derived_local/source_balanced_roi_expansion_manifest`

Key result:

- The expansion pass uses the full 89-cycle cycle-state table rather than only the 30 highest-ranked warning cycles.
- It selected and sampled 48 cycle segments across 14 source movies, with no missing HDF5 segments.
- 41 selected cycle/source pairs are new relative to the existing multi-cycle, transfer-ranked, balanced-future, and control video cohorts.
- The sampled segments produced 2,880 automatic particle-like candidates and a compact 96-row ROI expansion table with the top 2 candidates per cycle.
- The selected cycle set includes 24 future16-positive cycles and 14 future8-positive cycles, plus source representatives and future16-negative controls. Source coverage spans early normal-COV movies, HighCOV movies, and HighHighCOV movies.

Interpretation: this directly addresses the cohort-breadth bottleneck in the current video/physics analyses. It does not claim validated particles, fronts, diffusion, or degradation mechanisms; it creates a source-balanced expansion manifest for follow-up ROI sequence export, mask/front QC, and source-transfer tests on a broader direct-video cohort.
## 2026-05-22 Conditioned Residual Physics Atlas

Added and ran:

`scripts/tier4_conditioned_residual_physics_atlas.py`

Output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/conditioned_residual_physics_atlas`
- `derived_local/conditioned_residual_physics_atlas`

Key result:

- The atlas maps 39 split-specific conditioned residual modes per split against 212 interpretable descriptor columns across 172 ROI rows, 34 cycles, and 12 sources.
- The strongest non-degradation source-centered physics alignment is leave-source `resdict_pc04_mean` versus temporal/diffusion-radius proxies: `temporal_diffusion_proxy_median_um2_per_s` rho 0.815 after source centering, with raw rho 0.592. `resdict_pc04_last_minus_first` shows the opposite-sign radius/diffusion alignment around rho -0.815.
- Category-level source-centered support is broad: leave-source front/phase/diffusion max |rho| 0.815 with 315 strong centered pairs; echem-state max |rho| 0.769 with 1296 strong pairs; rollout-prediction max |rho| 0.767 with 242 strong pairs.
- Individual conditioned modes are predictive for future16 without using the full classifier: leave-cycle `resdict_pc08_slope` reaches AUC 0.821/AP 0.956; `resdict_pc08_last_minus_first` reaches AUC 0.801/AP 0.951; `resdict_pc04_last_minus_first` reaches AUC 0.800/AP 0.950.
- Top mode summaries connect target-predictive residuals to interpretable families: leave-source `residual_energy_mean` aligns with low-rank-DMD particle reconstruction error (centered rho 0.659), and leave-source `resdict_pc04_mean` aligns with apparent diffusion/front-radius motion (centered rho 0.815).

Interpretation: the conditioned residual objective is not just a black-box weak-label predictor. Several residual bases line up, after source centering, with apparent phase/front/diffusion motion, rollout reconstruction difficulty, and echem-regime structure. These are still optical proxies and split-specific residual features, so they support physics triage and model-objective design rather than calibrated diffusion coefficients or causal mechanism claims.
## 2026-05-22 Source-Balanced ROI Sequences and Rollout Audit

Added and ran:

- `scripts/tier4_export_source_balanced_roi_sequences.py`
- `scripts/tier4_source_balanced_sequence_rollout_audit.py`

Output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_sequence_rollout_audit`
- `derived_local/source_balanced_roi_sequences` (manifest/summary only; NPZ tensors stay remote)
- `derived_local/source_balanced_sequence_rollout_audit`

Key result:

- Exported all 96 source-balanced ROI rows into particle-region crop tensors across 48 cycles and 14 sources, with 96 sampled frames per ROI, 192 px full-frame crops resized to 96x96, and zero export failures. This directly converts the source-balanced manifest into model-ready particle-only inputs without copying the NPZ tensors locally.
- The fast rollout/temporal audit computes ROI-only persistence/velocity prediction error, temporal activity, and intensity-drift features over the exported crops. It avoids the expensive full global-DMD run that was too slow for this larger cohort.
- Future16 weak-label signal in this broader cohort is modest and mostly intensity-drift based: ROI normalized mean last-minus-first has AUC 0.626/AP 0.589/source eta2 0.365, with future16 positives showing a more negative normalized intensity drift. Raw ROI mean drift has AUC 0.606/AP 0.549/source eta2 0.534.
- One-step prediction-error features are not robust future16 signals here and remain strongly source structured: persistence late MSE AUC 0.556/source eta2 0.787; velocity MSE mean AUC 0.540/source eta2 0.824.
- At the cycle-collapsed level, normalized ROI intensity drift is again the top future16 feature (AUC 0.648/AP 0.594), while future8 is weak and led by object contrast/late temporal energy around AUC 0.63/0.63.

Interpretation: this is the first broad source-balanced direct-video cohort with particle-only tensors. It supports source-breadth and model-readiness more than strong warning performance: future labels are weak, automatic ROIs are not manually validated, and rollout-error features are heavily source dependent. The useful next step is to combine this cohort with mask/front QC and source-normalized modeling rather than claiming calibrated degradation physics from these automatic crops alone.

## 2026-05-22 Agentic AI Implementation Folders 05-08

Extended the paper-inspired agentic workflow on Isambard with four additional separate folders under `agentic_research` and reran the full workflow against `/scratch/u6hp/nsagar.u6hp/Alek_Jiho`.

New folders:

- `05_agentic_metric_search`: ERA-style benchmark search over real derived metric tables.
- `06_closed_loop_hypothesis_ledger`: persistent claim ledger with support, counter-evidence, falsification test, next script, and status.
- `07_manual_qc_feedback_hook`: lab-in-the-loop feedback gate that turns pending ROI/front labels into explicit downstream requirements.
- `08_guarded_code_acceptance`: acceptance scanner for generated scripts using compile, split/null/schema, and destructive-operation guardrails.

Output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/agentic_research_outputs/05_agentic_metric_search`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/agentic_research_outputs/06_closed_loop_hypothesis_ledger`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/agentic_research_outputs/07_manual_qc_feedback_hook`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/agentic_research_outputs/08_guarded_code_acceptance`
- compact local copy: `derived_local/agentic_research_outputs`

Key result:

- Metric search aggregated 733 model/audit rows from 15 source metric tables and promoted 132 analysis variants for follow-up under the guarded score `AUC + 0.35*AP + 0.10*abs(rho) + control_bonus - leakage_penalty`.
- The top promoted rows come from the echem-conditioned residual dictionary, especially leave-source future8/future16 variants. This is useful for planning, but future8 remains explicitly leakage/context guarded.
- The closed-loop ledger now tracks 5 claims: future16 echem-conditioned residuals under test, future8 context dominance as a supported guardrail, manual QC as blocked-by-QC, calibrated diffusion as still proposed/guarded, and the agentic metric-search planner as implemented.
- The QC feedback hook found 47 pending QC candidates and zero accepted manual labels, so downstream front/diffusion claims remain blocked by manual labels rather than inferred automatically.
- The code acceptance gate scanned 3 generated experiment stubs. All compile, but all remain `needs_revision` because they are placeholders without real split/null/schema logic. This is the correct guardrail outcome.

Interpretation: the three Nature agentic-AI ideas are now implemented as auditable project infrastructure over real Alek/Jiho derived evidence, not as an unrestricted LLM report. The immediate computational value is benchmark-driven prioritization and claim tracking. The scientific bottleneck is still manual ROI/front QC plus source-balanced validation before mechanism-level claims.

## 2026-05-22 Source-Balanced Mask/Front Sanity Audit

Added `scripts/tier4_source_balanced_mask_front_sanity_audit.py` and ran it on Isambard against the source-balanced particle-region crop tensors.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_mask_front_sanity_audit`
- `derived_local/source_balanced_mask_front_sanity_audit`

Cohort/result snapshot:

- 96 ROI sequences across 48 cycles and 14 source movies.
- Future8/future16 positive ROI sequences: 28 / 48.
- Strongest future16 ROI proxy: `masked_minus_background_mean_slope`, AUC 0.690, AP 0.696, median positive-negative 0.00372, source eta2 0.634.
- Strongest future16 cycle-collapsed proxy was the same feature, AUC 0.688, AP 0.703, median positive-negative 0.00253, source eta2 0.649.
- The prior rollout/intensity drift signal remains visible: `roi_norm_mean_delta_last_minus_first` future16 ROI AUC 0.626 and cycle AUC 0.648, lower in positives.
- Simple front-radius and apparent diffusion proxies were weaker: q70 radius-squared apparent diffusion future16 ROI AUC 0.558 and cycle AUC 0.569, with substantial source structure.

Interpretation:

The source-balanced crop cohort contains a reproducible automatic mask/intensity contrast slope associated with future16 drops, stronger than the raw ROI mean drift in this audit. However, the strongest mask/front proxy is still source-structured, and the masks/front radii are crop-local automatic sanity proxies from resized tensors, not manual particle masks or calibrated diffusion measurements. Treat this as a useful QC/mechanism triage feature, not a standalone morphology claim.

## 2026-05-22 Source-Balanced Mask/Front Source-Residual Audit

Added and ran:

`scripts/tier4_source_balanced_mask_front_source_residual_audit.py`

Output directories:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_mask_front_source_residual_audit`
- `derived_local/source_balanced_mask_front_source_residual_audit`

Key result:

- Tested 54 source-balanced mask/front/crop descriptors across 96 ROI rows, 48 cycles, and 14 sources with raw, source-mean-only, source-residual, within-source z, and within-source rank transforms.
- The raw future16 best remains `masked_minus_background_mean_slope` at AUC 0.690/AP 0.696, but source eta2 is 0.634, confirming that this is strongly source structured.
- Source-mean-only coordinates and intensity summaries can score even higher, e.g. `object_y_full_approx` future16 AUC 0.798, proving that source/design structure can dominate apparent weak-label performance if left uncontrolled.
- After source residualization, the best future16 feature switches to `front_radius_q80_slope_px_per_norm_time`, AUC 0.631/AP 0.634, source eta2 ~0, p=0.122.
- Within-source rank gives the same q80 front-slope feature AUC 0.656/AP 0.677, source eta2 0.078, p=0.062.

Interpretation: source residualization demotes the strongest raw optical-contrast signal and leaves only modest front-radius slope evidence. That is still useful: it identifies `front_radius_q80_slope_px_per_norm_time` as a source-robust front/morphology hypothesis for QC and future modeling. It does not validate calibrated phase-boundary motion or diffusion because labels are weak, fronts are automatic, and significance is borderline.

## 2026-05-22 Source-Balanced Residual Dictionary Audit

Added `scripts/tier4_source_balanced_residual_dictionary_audit.py` and ran it on Isambard against the 96 source-balanced particle-region crop tensors. The audit learns label-free PCA bases on next-frame residual fields, summarizes each ROI by residual-dictionary coefficient trajectories, and compares residual descriptors with the automatic mask/front scalar proxies.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_audit`
- `derived_local/source_balanced_residual_dictionary_audit`

Cohort/result snapshot:

- 96 ROI sequences across 48 cycles and 14 source movies.
- PCA used 16 components on 48x48 downsampled residuals and explained 0.522 of sampled residual variance.
- Leave-cycle future16 readout: residual dictionary AUC 0.602/AP 0.581, mask/front scalar AUC 0.593/AP 0.578, combined residual+mask/front AUC 0.573/AP 0.555.
- Leave-source future16 readout failed/inverted: residual dictionary AUC 0.375/AP 0.448; combined residual+mask/front AUC 0.322/AP 0.396. Source transfer remains the dominant guardrail.
- Strongest ROI-level residual dictionary future16 scalar was `resdict_pc01_mean`, AUC 0.668/AP 0.670/source eta2 0.187, close to `resdict_pc02_slope` AUC 0.668/AP 0.719/source eta2 0.199. These residual PCs are less source-structured than the mask/front contrast slope (`masked_minus_background_mean_slope`, AUC 0.690/source eta2 0.634).
- Cycle-collapsed residual dictionary features were stronger: `resdict_pc01_mean` future16 AUC 0.769/AP 0.753/source eta2 0.421 and `resdict_pc09_slope` AUC 0.724/AP 0.755/source eta2 0.422.

Interpretation:

Label-free residual bases capture temporal-dynamics modes that are complementary to crop-local mask/front intensity contrast and less source dominated at the ROI scalar level. However, grouped predictive readouts do not yet transfer across held-out source movies, so the next modeling step should either residualize/domain-adapt these residual coefficients by source or explicitly train source-invariant residual dynamics objectives before making any degradation warning claim.

## 2026-05-22 Source-Balanced Residual Dictionary Source-Residual Audit

Added `scripts/tier4_source_balanced_residual_dictionary_source_residual_audit.py` and ran it on Isambard against the source-balanced residual dictionary feature table. The audit tests raw, source-mean-only, source-residual, within-source-z, and within-source-rank transforms across residual-dictionary, mask/front, and object-reconstruction scalar families.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_source_residual_audit`
- `derived_local/source_balanced_residual_dictionary_source_residual_audit`

Cohort/result snapshot:

- 96 ROI sequences, 48 cycles, 14 source movies.
- Tested 102 physics/ROI descriptors: 72 residual dictionary, 24 mask/front scalar, and 6 object-reconstruction controls.
- Raw future16 best overall remains `masked_minus_background_mean_slope`, AUC 0.690/AP 0.696/source eta2 0.634.
- Raw future16 best residual-dictionary feature remains `resdict_pc01_mean`, AUC 0.668/AP 0.670/source eta2 0.187.
- Source-residual future16 best overall and best residual-dictionary feature is `dictionary_recon_error_last_minus_first`, AUC 0.637/AP 0.637 with source eta2 effectively zero and median positive-negative 2.54e-6.
- Within-source-rank future16 best residual-dictionary feature is weaker: `dictionary_recon_error_mse_slope`, AUC 0.574/AP 0.551/source eta2 0.004.
- Source-mean-only transforms are very strong, e.g. residual `resdict_pc01_mean` future16 AUC 0.824, confirming that source-level acquisition/protocol structure carries a large share of the apparent degradation signal.

Interpretation:

This audit separates three effects: source-level structure is very predictive, raw crop-local mask/front contrast is strong but source structured, and source-residual residual-dictionary reconstruction-error drift retains a moderate future16 association without source variance. The source-residual residual feature is the best current candidate for a source-robust temporal-dynamics physics proxy, but it remains an in-cohort weak-label scalar audit and needs held-out-source/domain-adapted modeling before any warning claim.

## 2026-05-22 Source-Balanced Residual-Physics Coupling Audit

Added `scripts/tier4_source_balanced_residual_physics_coupling_audit.py` and ran it on Isambard against the source-balanced residual dictionary feature table. The audit asks whether source-normalized residual-dictionary dynamics align with crop-local mask/front/apparent-diffusion proxies, rather than only predicting weak future-drop labels.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_physics_coupling_audit`
- `derived_local/source_balanced_residual_physics_coupling_audit`

Cohort/result snapshot:

- 96 ROI sequences, 48 cycles, 14 source movies.
- Tested 72 residual-dictionary descriptors against 25 crop-local physics proxies under raw, source-residual, within-source-rank, and within-source-z transforms.
- Strong raw and source-residual correlations often involve mask centroid/area summaries, but many of those are not useful future16 targets after source removal.
- For the source-residual primary candidates, the largest coupling is `resdict_pc02_slope` versus `mask_area_fraction_median` (rho 0.521), but the target directions are not aligned.
- The strongest source-residual target-aligned pair is `dictionary_recon_error_last_minus_first` versus `masked_minus_background_mean_slope`: rho 0.373, residual future16 AUC 0.637, physics-proxy future16 AUC 0.628, with both source eta2 values effectively zero after residualization.
- The same residual candidate couples to front-radius slope proxies in magnitude (`front_radius_q60_slope_px_per_norm_time` rho -0.380, `front_radius_q80_slope_px_per_norm_time` rho -0.305), but those are not target-direction aligned in this audit.
- Apparent q70 diffusion proxy coupling is weak (`dictionary_recon_error_last_minus_first` versus apparent q70 diffusion rho -0.184, p=0.183) and should not be interpreted as calibrated diffusion evidence.

Interpretation:

The current source-robust residual-dynamics candidate is most consistent with an optical contrast/reconstruction-error change rather than a clean diffusion-front measurement. It is still useful for triage because the signal survives source residualization and aligns with a crop-local contrast proxy, but the front/diffusion interpretation remains guarded until manual QC and calibrated phase-boundary tracking are available.

## 2026-05-22 Source-Balanced Residual Dictionary Normalized Readout

Added `scripts/tier4_source_balanced_residual_dictionary_normalized_readout.py` and ran it on Isambard against the source-balanced residual dictionary feature table. The audit applies unsupervised within-source transforms and evaluates grouped logistic readouts under leave-cycle and leave-source splits.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_normalized_readout`
- `derived_local/source_balanced_residual_dictionary_normalized_readout`

Cohort/result snapshot:

- 96 ROI sequences, 48 cycles, 14 source movies.
- Feature sets include 72 raw/source-normalized residual-dictionary descriptors, 24 mask/front descriptors, combined residual+mask/front sets, and the single source-residual `dictionary_recon_error_last_minus_first` candidate.
- Future16 leave-source raw residual dictionary remained inverted/failed: AUC 0.375/AP 0.448.
- Future16 leave-source full residual dictionary improved after source residualization to AUC 0.550/AP 0.583 and within-source-rank reached AUC 0.543/AP 0.506.
- The single source-residual `dictionary_recon_error_last_minus_first` readout was strongest: leave-source future16 AUC 0.612/AP 0.613 and leave-cycle future16 AUC 0.627/AP 0.630.
- A 200-permutation null for the best leave-source future16 readout had null AUC p95 0.621, empirical p(AUC)=0.100 and p(AP)=0.070.

Interpretation:

Source normalization partially rescues the held-out-source failure mode, but the improvement concentrates in one reconstruction-error drift feature rather than the full residual dictionary. The permutation null is suggestive but not conventionally significant, so this is a provisional source-robust temporal-dynamics readout for follow-up, not a deployable degradation warning model.

## 2026-05-22 Source-Balanced Residual Candidate Review Packet

Added `scripts/tier4_source_balanced_residual_candidate_review_packet.py` and ran it on Isambard. This converts the source-normalized residual readout and residual-physics coupling evidence into a concrete manual-QC queue for the source-balanced ROI crops without assigning labels.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_candidate_review_packet`
- `derived_local/source_balanced_residual_candidate_review_packet`

Cohort/result snapshot:

- Ranked 96 ROI candidates across 48 cycles and 14 source movies.
- Review tier counts: 8 immediate manual-QC, 11 high-priority, 19 standard-review, and 58 routine candidates.
- Ranking combines source-heldout future16 probability from `dictionary_recon_error_last_minus_first_source_residual`, source-residual reconstruction-error drift, source-residual contrast slope, and front-motion magnitude.
- The top immediate-review candidate is `source_balanced_cycle108_rank6_obj2_12_c2_x10_070723`, score 0.917, source-heldout future16 probability 0.781, future16 weak label 1, with positive source-residual reconstruction-error and contrast-slope evidence.
- Other immediate-review candidates include cycle/source groups 132 (`15_c2_x5_HighCOV_120723`), 145/144 (`16_c2_x10_HighHighCOV_130723`), and 108 (`12_c2_x10_070723`), giving both weak-positive and weak-negative examples for artifact/front review.
- Every row keeps `manual_qc_status=pending` and includes an instruction to review particle identity, stable mask, phase-front plausibility, and artifact risk before treating any automatic feature as physics evidence.

Interpretation:

This packet turns the current AI-derived residual dynamics signal into an actionable lab-in-the-loop queue. It does not improve the model by itself, but it reduces the next bottleneck: selecting which automatic crops should be manually checked before making stronger claims about phase-front movement, diffusion proxies, or degradation modes.

## 2026-05-22 Source-Balanced Residual Temporal Specificity Audit

Added `scripts/tier4_source_balanced_residual_temporal_specificity_audit.py` and ran it on Isambard against the source-balanced residual dictionary feature table plus the full particle abrupt-drop event cycle table. The audit tests whether the source-residual reconstruction-error drift behaves like a future precursor rather than a past/current event marker.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_temporal_specificity_audit`
- `derived_local/source_balanced_residual_temporal_specificity_audit`

Cohort/result snapshot:

- 96 ROI rows, 48 cycles, 14 source movies.
- Full particle abrupt-drop event cycles used for temporal labels: 60, 86, 116, 156.
- Label counts on the source-balanced ROI rows: future8 28, future16 48, past8 10, past16 30, current-event 0.
- Primary source-residual `dictionary_recon_error_last_minus_first` future16 remains positive: AUC 0.637/AP 0.637, rho 0.237, p=0.020.
- The same primary candidate is weak for future8 and is not future-specific there: future8 AUC 0.539 versus past8 fixed-direction AUC 0.630.
- For future16, the primary candidate barely exceeds the past16 control: future16 AUC 0.637 versus past16 fixed-direction AUC 0.625, future-minus-control AUC 0.012.
- A 500 within-source circular-shift null for the primary future16 candidate is strong: null AUC p95 0.521, empirical p(AUC)=0.002 and p(AP)=0.002.
- The most future-specific row overall is raw `masked_minus_background_mean_slope` at future8 AUC 0.821 versus past8 AUC 0.377, but this raw contrast proxy is already known to be strongly source structured.

Interpretation:

The source-residual reconstruction-error drift is temporally ordered relative to the event-cycle sequence, but it is not a clean precursor: it also marks cycles near past events. This downgrades the claim from future-warning evidence to a source-robust degradation-neighborhood dynamics marker. Raw optical contrast has stronger future8 temporal specificity, but source structure keeps it under the acquisition/confounding guardrail.

## 2026-05-22 Source-Balanced Future-Specific Residual Audit

Added `scripts/tier4_source_balanced_future_specific_residual_audit.py` and ran it on Isambard against the source-balanced residual dictionary table plus full particle abrupt-drop event cycles. This audit follows the temporal-specificity result by excluding past-event windows and by comparing grouped future prediction against a past-event-context-only baseline.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_future_specific_residual_audit`
- `derived_local/source_balanced_future_specific_residual_audit`

Cohort/result snapshot:

- 96 ROI rows, 48 cycles, 14 source movies.
- Full abrupt-drop event cycles: 60, 86, 116, 156.
- Label counts: future8 28, future16 48, past8 10, past16 30; no-past8 rows 86 and no-past16 rows 66.
- Primary source-residual `dictionary_recon_error_last_minus_first` future16 drops from AUC 0.637/AP 0.637 on all rows to AUC 0.589/AP 0.708 after excluding past16 rows, and to AUC 0.541/AP 0.329 in pre-first-event rows.
- The strongest clean scalar after excluding past8 remains raw `masked_minus_background_mean_slope`, future8 AUC 0.812/AP 0.713, but this is still a source-structured optical contrast proxy.
- After excluding past16, raw residual dictionary PCs retain clean future16 scalar signal: `resdict_pc01_mean` AUC 0.702/AP 0.783 and `resdict_pc09_slope` AUC 0.695/AP 0.790.
- Grouped models without future-distance leakage show past-event context alone is already strong: leave-cycle future16 AUC 0.755 and leave-source future16 AUC 0.720.
- Adding source-residual mask contrast gives the largest future16 delta over past-event context: leave-cycle delta AUC +0.065 (AUC 0.820) and leave-source delta AUC +0.017 (AUC 0.737).
- Adding the primary source-residual reconstruction-error feature gives a smaller future16 delta: leave-cycle +0.055 and leave-source +0.004, with lower AP than the past-context baseline. Future8 deltas are mostly zero or negative.

Interpretation:

Past-event proximity explains much of the apparent future-warning structure in this source-balanced cohort. Some future16 optical/residual signal remains after removing past16 rows or modeling past-event context, but the deployable precursor claim is still weak: the strongest increments are modest, source-transfer increments are small, and the clean scalar winners are raw/source-structured optical contrast and residual PC features rather than the source-residual reconstruction-error candidate. The useful next direction is to treat these as event-neighborhood degradation state features and expand source-balanced pre-event sampling before claiming forecasting.

## 2026-05-22 Source-Balanced Degradation Mode Audit

Added `scripts/tier4_source_balanced_degradation_mode_audit.py` and ran it on Isambard. The audit clusters source-residualized residual/front/contrast features from the 96 source-balanced particle-region crops, then tests how the resulting unsupervised modes map onto pre-event, post-event, and far-from-event cycle neighborhoods.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_degradation_mode_audit`
- `derived_local/source_balanced_degradation_mode_audit`

Cohort/result snapshot:

- 96 ROI rows, 48 cycles, 14 source movies, and 18 source-residual residual/front/contrast features.
- KMeans selected k=4 by silhouette, but the partition is imbalanced: mode 0 has 70 ROI rows, mode 3 has 23, and modes 1/2 are tiny outlier states with 1 and 2 rows.
- Mode 0 is a broad source-residual front-geometry state with future16 fraction 0.543 and past16 fraction 0.329.
- Mode 3 is a residual-dictionary state with lower future16 fraction 0.391 and more far-from-event/pre-event-8 mixture.
- The strongest enrichment is tiny mode 2 for post_event_16: 2/2 rows versus 8/94 outside, Fisher p=0.010. This should be treated as an outlier review cue, not a stable class.
- Source-local mode transitions are common enough to be useful for review triage: 82 adjacent source/cycle transitions, change fraction 0.244.
- Representative rows include the mode-2 pair at cycle 132 from `15_c2_x5_HighCOV_120723`, plus mode-1 pre-event-16 outlier `source_balanced_cycle142_rank18_obj1_16_c2_x10_HighHighCOV_130723`.

Interpretation:

The source-balanced residual/front feature space organizes degradation-neighborhood states, but it does not yet support a validated degradation-mode taxonomy. The main practical value is manual-QC triage: the broad mode captures common front/dictionary residual state, while tiny temporal-dictionary outlier modes identify specific post-event or pre-event examples worth checking. This reinforces the current synthesis guardrail: use these modes as hypotheses and review candidates only, not calibrated phase boundaries, diffusion coefficients, causal mechanisms, or deployable warning classes.

## 2026-05-22 Source-Balanced Pre-Event Sampling Manifest

Added `scripts/tier4_source_balanced_pre_event_sampling_manifest.py` and ran it on Isambard against the full masked-residual cycle table, abrupt-event cycle list, and HDF5 movies. This is the follow-up to the future-specific residual audit: it creates an event-relative, source-balanced pool for testing true pre-event dynamics separately from current/post-event proximity.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_sampling_manifest`
- `derived_local/source_balanced_pre_event_sampling_manifest`

Cohort/result snapshot:

- Selected and sampled 64 cycles across 14 source movies, with 13 cycle/source pairs not already present in existing video cohorts.
- Reconstructed 3,840 automatic particle-like candidates from sampled HDF5 segments and retained 128 top ROI proposal rows for follow-up particle-only sequence export.
- No selected cycles were missing from the accessible HDF5 data.
- Event cycles used for event-relative bins: 60, 86, 116, and 156.
- Cycle bins after excluding current-event cycles from pre-event bins: near pre-event 1-8 cycles = 16, mid pre-event 9-16 cycles = 11, far pre-event 17-32 cycles = 11, post-event 1-16 cycles = 20, no-near-event controls = 6.
- ROI proposal bins: near pre-event = 32, mid pre-event = 22, far pre-event = 22, post-event = 40, no-near-event controls = 12.
- The first run exposed and then fixed an event-relative binning issue where current-event cycles could be mislabeled as far-pre-event relative to the next event; the committed script now labels current events separately and excludes them from selection.

Interpretation:

This closes the immediate sampling gap raised by the future-specific audit. The project now has a concrete source-balanced, event-relative ROI proposal set for exporting particle-only pre-event/post-event/control videos and for testing whether residual/front/rollout features are genuine precursors rather than event-neighborhood markers. The manifest is still automatic candidate generation, so it supports follow-up modeling and QC prioritization but not validated particle identity, phase-front motion, diffusion, or causal degradation claims.

## 2026-05-22 Source-Balanced Pre-Event ROI Sequence Export and Audit

Exported the pre-event sampling manifest into particle-region crop tensors with `scripts/tier4_export_source_balanced_roi_sequences.py`, preserving event-relative bin metadata, and added `scripts/tier4_source_balanced_pre_event_sequence_audit.py` to measure simple video dynamics across near/mid/far pre-event, post-event, and no-near-event controls.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_sequence_audit`
- `derived_local/source_balanced_pre_event_roi_sequences` contains only compact manifest/summary files; NPZ tensors remain on Isambard.
- `derived_local/source_balanced_pre_event_sequence_audit`

Cohort/result snapshot:

- Exported 128 ROI sequences across 64 cycles and 14 source movies with 0 failures; each sequence has 96 frames from fixed padded particle-region crops.
- Event-relative audit feature rows match the export: 128 rows, 64 cycles, 14 sources, 0 feature failures.
- Bin counts are near pre-event 32 ROI rows, mid pre-event 22, far pre-event 22, post-event 40, and no-near-event controls 12.
- Near-pre-event versus all other bins has a strong spatial video readout: leave-cycle spatial AUC 0.763/AP 0.536 and leave-source spatial AUC 0.759/AP 0.515.
- The top near-pre scalar is raw `spatial_std_slope`, AUC 0.803/AP 0.591, p = 3.14e-07, indicating stronger growth in crop heterogeneity near abrupt-event windows.
- Clean pre16 versus post/control is moderate under leave-cycle all-video readout: AUC 0.723/AP 0.759.
- Broader any-pre versus post/control does not transfer robustly by source: all-video leave-source AUC 0.463/AP 0.645, while leave-cycle AUC is 0.637/AP 0.758.

Interpretation:

The new pre-event tensors make the earlier future-specificity gap actionable. There is a localized near-pre-event spatial/heterogeneity signal that survives held-out-source testing, but it does not generalize as a broad all-pre-event warning axis. The physics-facing hypothesis is therefore narrower: the final 1-8 cycles before abrupt event windows may show increasing particle-crop spatial heterogeneity/front disorder. This remains automatic-crop, event-proximity evidence, not validated phase-boundary motion, diffusion, particle identity, or causal forecasting.

## 2026-05-22 Source-Balanced Pre-Event Rollout/Mask/Event-Relative Readout

Exported the source-balanced pre-event proposal manifest into particle-region crop tensors on Isambard and ran rollout, mask/front, and event-relative readout audits. The NPZ tensors remain remote; compact manifests and feature summaries are mirrored locally.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_roi_sequences`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_sequence_rollout_audit`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_mask_front_audit`
- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_readout_audit`
- `derived_local/source_balanced_pre_event_roi_sequences` (manifest/summary only; NPZ tensors stay remote)
- `derived_local/source_balanced_pre_event_sequence_rollout_audit`
- `derived_local/source_balanced_pre_event_mask_front_audit`
- `derived_local/source_balanced_pre_event_readout_audit`

Cohort/result snapshot:

- Exported 128/128 particle-region crop tensors across 64 cycles and 14 source movies, with 0 export failures.
- Future-label counts on exported sequences: 32 future8 positives and 66 future16 positives.
- Rollout audit top future16 ROI row: `roi_norm_mean_delta_last_minus_first`, AUC 0.628/AP 0.620, but source eta2 0.414.
- Mask/front audit top future16 ROI rows: `roi_norm_mean_delta_last_minus_first` AUC 0.628/AP 0.620 and `masked_minus_background_mean_slope` AUC 0.624/AP 0.655; both remain source structured.
- Event-relative readout clean near-pre-event (1-8 cycles) versus post/control: raw `masked_minus_background_mean_slope` is strong, AUC 0.797/AP 0.749, p=5.48e-06, but source eta2 0.578.
- Best source-residual clean near-pre-event readout: `front_radius_q60_slope_px_per_norm_time`, AUC 0.660/AP 0.665, p=0.083, eta2 approximately zero.
- Clean pre-event 1-16 versus post/control source-residual readout: `front_radius_q80_median_px`, AUC 0.645/AP 0.570, p=0.010, eta2 approximately zero.
- Near-pre versus far-pre source-residual apparent diffusion/front proxy: `apparent_diffusion_q70_px2_per_norm_time`, AUC 0.681/AP 0.648, p=0.077, eta2 approximately zero.

Interpretation:

The event-relative export makes the pre-event question more concrete. Raw optical contrast strongly separates near-pre-event from post/control bins, but it is still source structured. After source residualization, the surviving signal shifts to front-radius/slope and apparent-diffusion-like proxies with moderate AUC and weaker p-values. This is useful physics triage: it suggests pre-event organization in crop-local front geometry, while keeping the claim guarded because masks/fronts are automatic and not calibrated phase boundaries or diffusion coefficients.

## 2026-05-22 Source-Balanced Pre-Event Source-Invariant Mechanism Audit

Added `scripts/tier4_source_balanced_pre_event_source_invariant_audit.py` and ran it on Isambard against the source-balanced pre-event readout feature table. This audit tests whether interpretable feature families preserve event-relative signal under leave-source evaluation after simple source-confound handling.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_source_invariant_audit`
- `derived_local/source_balanced_pre_event_source_invariant_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies, 47 interpretable features.
- Targets: clean near-pre-event versus post/control, and near-pre-event versus far-pre-event.
- Clean near-pre versus post/control is led by the combined physics-front feature family after source-mean residualization: leave-source AUC 0.694/AP 0.660, rho 0.327.
- ROI intensity after source-mean residualization is close behind for clean-pre: AUC 0.681/AP 0.594, so residualized intensity drift remains a guardrail.
- Near-pre versus far-pre is strongest for physics-front features under source-residual handling: leave-source AUC 0.744/AP 0.711, rho 0.416.
- Raw univariate clean-pre is still dominated by `masked_minus_background_mean_slope`, AUC 0.797/AP 0.749/source eta2 0.578.
- The low-source-eta2 model set is not physically convincing for clean-pre; its best row is object-context guardrail at AUC 0.573, so source-invariant signal is not simply solved by dropping high-eta features.

Interpretation:

The source-invariant audit strengthens a narrow physics hypothesis: pre-event organization is most consistently carried by front/mask/apparent-diffusion feature families when source structure is handled in the model. It also exposes an important guardrail: intensity and object-context terms can still perform under some transforms, so these models are mechanism-ranking tools for manual QC and follow-up, not deployable warning models or validated phase-boundary/diffusion measurements.

## 2026-05-22 Source-Balanced Pre-Event Review Packet

Added `scripts/tier4_source_balanced_pre_event_review_packet.py` and ran it on Isambard. This converts the source-invariant/front/diffusion pre-event hypotheses into a concrete manual-QC queue with ranked ROI rows, individual frame-strip PNGs, and a contact sheet.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_review_packet`
- `derived_local/source_balanced_pre_event_review_packet`

Cohort/result snapshot:

- Ranked 128 source-balanced pre-event candidate ROI rows across 64 cycles and 14 source movies.
- Rendered 24 top candidate frame strips plus `pre_event_review_contact_sheet.png` for quick visual screening.
- Event-relative bins in the packet: near pre-event 32 rows, mid pre-event 22, far pre-event 22, post-event 40, and no-near-event controls 12.
- Review reasons: 13 `near_pre_high_source_invariant_front_score`, 19 `near_pre_front_diffusion_review`, 22 `mid_pre_followup_front_diffusion_review`, and 74 context/guardrail rows.
- Top candidate: `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723`, near-pre, score 4.789, source-invariant clean-front probability 0.941, near/far physics probability 0.522, front q60 slope 1.130 px/norm time, apparent q70 diffusion proxy 0.576 um2/norm time.
- The top six rows are all from source `17_c2_x10_HighHighCOV_150723` around cycles 150-152, so source clustering remains an explicit review guardrail rather than a discovery claim.

Interpretation:

This packet makes the most actionable current AI output visual: it tells a reviewer which automatic crops to inspect first for pre-event front/diffusion-like behavior. It does not assign manual labels, validate particle identity, validate phase boundaries, calibrate diffusion, or establish causal precursors. The immediate use is reviewer triage and failure-mode discovery, especially checking whether the source-17 near-pre cluster reflects real particle morphology or an acquisition/source artifact.

## 2026-05-22 Source-Balanced Pre-Event Directionality Audit

Added `scripts/tier4_source_balanced_pre_event_directionality_audit.py` and ran it on Isambard as a ROI-level complement to the cycle-collapsed event-distance trajectory audit. The audit merges pre-event sequence features with rollout/mask/front readouts and compares clean near-pre-event, near-versus-far-pre-event, and post-versus-control readouts while tracking whether features correlate with pre-event and post-event clocks.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_directionality_audit`
- `derived_local/source_balanced_pre_event_directionality_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies, and 57 tested rollout/front/mask/heterogeneity features.
- Reran with 250 source-stratified clock permutations.
- Raw spatial/intensity clocks are strong descriptively but source structured: `spatial_std_slope` has pre-event rho = 0.436 and nominal p = 8.12e-05, but source-stratified permutation p = 0.769.
- Raw `masked_minus_background_mean_slope` also increases as the event approaches: pre-event rho = 0.379, nominal p = 7.46e-04, but source-stratified permutation p = 0.785.
- Physics-facing source-residual clock: `apparent_diffusion_q70_um2_per_norm_time` has pre-event rho = 0.201, source-stratified permutation p = 0.024, post-event rho = -0.120, and near-pre versus far-pre AUC = 0.681/AP = 0.648.
- Equivalent px-scale apparent diffusion proxy has pre-event rho = 0.201 and source-stratified permutation p = 0.032.
- Clean near-pre versus post/control source-residual readout remains `front_radius_q60_slope_px_per_norm_time`: AUC = 0.660/AP = 0.665, pre-clock rho = 0.206, post-clock rho = -0.278.

Interpretation:

The permutation-enabled ROI-level directionality screen separates raw visual clock effects from source-normalized physics proxies. The strongest raw heterogeneity and mask/background clocks do not survive source-stratified clock permutation, while the apparent q70 diffusion-like proxy retains a modest source-residual pre-event ordering and weak post-event washout. This supports pre-event review prioritization and follow-up manual front QC, not causal precursor, calibrated diffusion, or validated phase-boundary claims.

## 2026-05-22 Source-Balanced Pre-Event Event-Distance Trajectory Audit

Added `scripts/tier4_source_balanced_pre_event_trajectory_audit.py` and ran it on Isambard using the source-balanced pre-event readout feature table. This audit collapses the two automatic ROI proposals per sampled cycle into one cycle-level row before testing whether rollout/front/mask/diffusion-like proxies change monotonically as sampled cycles approach abrupt optical event cycles.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_trajectory_audit`
- `derived_local/source_balanced_pre_event_trajectory_audit`

Cohort/result snapshot:

- Input readout table: 128 ROI rows; pre-event subset: 76 ROI rows.
- Cycle-level trajectory table: 38 cycle rows across 10 source movies and 4 event cycles, with 44 tested features.
- Event-relative cycle bins: far pre-event 17-32 cycles = 11 cycle rows, mid pre-event 9-16 cycles = 11, near pre-event 1-8 cycles = 16.
- Complete near/far event-cycle comparisons exist for 2 event cycles after allowing the far/mid/near bins around the same event to come from different source movies.
- Leading source-residual physics-facing trend toward the event is `apparent_diffusion_q70_px2_per_norm_time`: Spearman rho versus event proximity = 0.272, source-stratified permutation p = 0.064, near-minus-far median = 8.259 px^2 per normalized time.
- Related source-residual front trend: `front_radius_q70_slope_px_per_norm_time`, rho = 0.160, source-stratified permutation p = 0.084, near-minus-far median = 0.341 px per normalized time.
- The strongest source-residual rows overall include ROI intensity drift terms with low permutation p-values, but those are not physics-specific and should remain secondary to the front/diffusion proxy rows.

Interpretation:

This adds a stricter pre-event trajectory question on top of the bin-classification audits: are particle-crop front/diffusion proxies progressively organized as an abrupt event approaches? The answer is suggestive but still guarded. The apparent q70 radius-squared proxy and q70 front-slope proxy move in the expected direction after source residualization, but p-values are marginal and the event-distance design is sparse. These results are useful for prioritizing phase-front/diffusion follow-up experiments and manual QC, not for claiming calibrated phase-boundary motion, diffusion coefficients, particle identity, or causal forecasting.

## 2026-05-22 Source-Balanced Pre-Event Matched Counterfactual Audit

Added `scripts/tier4_source_balanced_pre_event_matched_counterfactual_audit.py` and ran it on Isambard. This audit builds baseline/context-nearest counterfactual pairs for near-pre-event ROI rows against far-pre and post/control rows, then tests paired differences in front, mask, apparent-diffusion, and video-dynamics descriptors with 1000 sign-flip permutations.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_matched_counterfactual_audit`
- `derived_local/source_balanced_pre_event_matched_counterfactual_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies.
- Matching context features: cycle/local-cycle indices, object location/area/z-score, crop location, first-frame ROI intensity, and baseline mask area.
- Pair counts: 32 near-vs-far source-penalized global pairs, 32 near-vs-post/control source-penalized global pairs, and 12 same-source near-vs-post/control pairs. Same-source near-vs-far pairs are not available in this sparse event-relative design.
- Strongest near-vs-far matched row is `masked_minus_background_mean_median`: median near-minus-far difference 0.0030, sign-flip p=0.0010.
- Near-vs-far `masked_minus_background_mean_slope` also survives matching: median difference 0.0030, sign-flip p=0.0050.
- Near-vs-far `mask_centroid_path_px` is larger in near-pre rows: median difference 0.658 px, sign-flip p=0.0080.
- The q60 front-slope row is directionally positive but underpowered after matching: median difference 0.988 px/norm time, n=10, sign-flip p=0.124.
- The apparent q70 diffusion proxy does not survive the matched counterfactual screen: median difference -0.204 um2/norm time, n=9, sign-flip p=0.395.

Interpretation:

This audit tightens the pre-event claim. Matched counterfactuals preserve a near-pre optical/mask-contrast signal, but they weaken the apparent-diffusion interpretation and leave front-slope evidence underpowered. The most defensible current statement is that source-balanced near-pre crops show review-worthy mask/contrast and some front-motion hints after observed-context matching; calibrated diffusion and phase-boundary claims still require manual QC and stronger matched same-source coverage.

## 2026-05-22 Pre-Event Source Lattice Coverage Audit

Added `scripts/tier4_pre_event_source_lattice_coverage_audit.py` and ran it on Isambard. This audit checks the raw `exampleParticles` cycle/source lattice, not just the sampled ROI table, to determine whether missing same-source far-pre controls are a sampler artifact or a real data-design limitation.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/pre_event_source_lattice_coverage_audit`
- `derived_local/pre_event_source_lattice_coverage_audit`

Cohort/result snapshot:

- Raw cycle/source lattice: 89 cycle rows across 16 HDF5-backed source movies.
- Event-relative bins: 20 near-pre rows, 14 mid-pre, 14 far-pre, 29 post-event, 8 no-near-event controls, and 4 current-event rows.
- Five raw sources have near-pre rows: `12_c2_x10_070723`, `17_c2_x10_HighHighCOV_150723`, `6_c2_x10_270623_2`, `7_c2_x10_290623`, and `9_c2_x10_010723`.
- Among those near-pre sources, 3 have near+mid-pre coverage, 2 have near+post/control coverage, and 0 have near+far-pre coverage.
- Four separate sources provide far-pre rows: `15_c2_x5_HighCOV_120723`, `16_c2_x10_HighHighCOV_130723`, `4_c2_x10_240623`, and `5_c2_x10_260623`.

Interpretation:

This closes an important design question from the matched and same-source ladder audits. The missing same-source near-vs-far-pre comparison is not latent in the current raw particle cycle index; it is absent from the available source/event lattice. Therefore the defensible designs are same-source near-vs-mid or near-vs-post/control checks where available, and cross-source far-pre controls only with explicit source/acquisition-class matching or source residualization. This prevents overclaiming source-independent pre-event diffusion or phase-boundary behavior from an unavailable same-source far-pre control.

## 2026-05-22 Source-Balanced Pre-Event Same-Source Ladder Audit

Added `scripts/tier4_source_balanced_pre_event_same_source_ladder_audit.py` and ran it on Isambard. This audit addresses the matched-counterfactual limitation that near-pre versus far-pre had no same-source pairs by asking a narrower within-source question: where a source contains near-pre rows plus mid-pre, post-event, or no-near-event controls, do automatic physics descriptors move consistently along that local event-relative ladder?

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_same_source_ladder_audit`
- `derived_local/source_balanced_pre_event_same_source_ladder_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies.
- Same-source ladder coverage is sparse: 5 sources have near-pre rows, 3 have near+mid-pre ladders, 2 have near+post/control ladders, and 0 have near+far-pre ladders.
- Pair counts: 32 near-vs-any-non-near same-source pairs, 20 near-vs-mid-pre same-source pairs, and 12 near-vs-post/control same-source pairs.
- Near-vs-any local controls: `mask_centroid_path_px` median near-control difference 0.784 px, sign-flip p=0.0020.
- Near-vs-any local controls: `front_radius_q60_slope_px_per_norm_time` median difference 1.769 px/norm time, p=0.0060.
- Near-vs-any local controls: `apparent_diffusion_q70_um2_per_norm_time` median difference 0.666 um2/norm time, p=0.0040, but this is conditional on reused same-source local controls and should stay guarded.
- Near-vs-mid-pre same-source pairs are stronger for q70 apparent diffusion and front slope, but only across 3 sources and no same-source far-pre anchors.
- Continuous within-source event-proximity clocks remain weak: top physics clock is q60 front-slope rho 0.109, p=0.417.

Interpretation:

This is a useful guardrail rather than a final mechanism. Strict same-source local pairing preserves near-pre front/mask/apparent-mobility differences when mid-pre or post/control anchors exist, but the event-relative ladder is sparse and lacks same-source far-pre controls. The result supports manual-QC follow-up on same-source near-pre fronts, while the weak continuous clocks and reused controls prevent calibrated diffusion, phase-boundary, or causal precursor claims.

## 2026-05-22 Source-Balanced Pre-Event Radial Kymograph Audit

Added `scripts/tier4_source_balanced_pre_event_radial_kymograph_audit.py` and ran it on Isambard. This audit extracts radial intensity kymographs from the source-balanced pre-event ROI tensors, tracks an automatic radial gradient-front over normalized time, and renders review kymograph PNGs for the top-ranked candidates.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_radial_kymograph_audit`
- `derived_local/source_balanced_pre_event_radial_kymograph_audit`

Cohort/result snapshot:

- Processed 128 ROI crops across 64 cycles and 14 source movies.
- Rendered 32 radial-kymograph PNGs for top review candidates.
- Near-pre versus far-pre is led by `front_radius2_slope_px2_per_norm_time`: AUC 0.705/AP 0.754, median near-minus-far difference 3.455 px^2 per normalized time, p=0.040.
- The paired front-radius slope is similar but weaker: `front_radius_slope_px_per_norm_time` AUC 0.686/AP 0.726, median difference 0.298 px/norm time, p=0.062.
- Clean pre-event versus post/control is led by `kymograph_temporal_energy`: AUC 0.664/AP 0.771, median difference 2.86e-06, p=0.00168, suggesting broader radial profile activity rather than a clean single-front effect.
- The top review candidate `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723` has radial-kymograph front radius2 slope 12.90 px^2/norm time, front radius slope 0.910 px/norm time, but modest front-track R2 0.076.

Interpretation:

This gives the pre-event cohort a more explicit phase-front-style representation than the scalar mask/front summaries. It strengthens the near-pre versus far-pre front-motion hypothesis, especially for radius-squared slope, while also showing why diffusion language remains guarded: front-track fit quality is often modest, clean-pre separation is dominated by temporal kymograph energy, and all tracks are automatic fixed-crop optical proxies. The outputs are best used for manual front QC and candidate selection, not calibrated diffusion or validated phase-boundary claims.

## 2026-05-22 Source-Balanced Pre-Event Physics-Mode Taxonomy

Added `scripts/tier4_source_balanced_pre_event_physics_mode_taxonomy.py` and ran it on Isambard using the source-balanced pre-event directionality feature table. This audit clusters source-residual front, mask, apparent-diffusion, and heterogeneity descriptors to test whether the pre-event signal forms repeatable discrete physics modes rather than only continuous event-distance/readout axes.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_physics_mode_taxonomy`
- `derived_local/source_balanced_pre_event_physics_mode_taxonomy`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies.
- Features used: 32 source-residual front/mask/apparent-diffusion/heterogeneity descriptors.
- K-means selection chooses k=2 as the only stable non-singleton clustering; silhouette is weak at 0.110.
- The two modes are broad and balanced: mode 0 has 60 ROI rows, mode 1 has 68 ROI rows; both contain all 14 sources.
- No strong event-neighborhood enrichment is detected. Best near-pre Fisher row is p=0.689, with near-pre fractions 0.267 versus 0.235 outside.
- Both modes are mixed front/mask/video-heterogeneity states. Representative rows include near-pre, mid-pre, and post-event examples rather than a clean degradation-neighborhood class.
- The most source-confounded raw features before residualization include `temporal_energy_p95`, `mask_centroid_path_px`, `frame_diff_mse_mean`, `masked_minus_background_mean_slope`, and `spatial_std_slope`; residualization drives their source eta2 approximately to zero.

Interpretation:

This is a useful negative result. In the source-balanced pre-event cohort, coarse unsupervised source-residual modes do not recover a repeatable near-pre-event degradation state. The more defensible signal remains continuous: front/diffusion-like clocks and source-invariant feature-family readouts. Mode labels from this audit should be used only as broad review strata, not as validated degradation modes, phase boundaries, diffusion coefficients, or causal warning classes.

## 2026-05-22 Source-Balanced Pre-Event Echem/Front Coupling Audit

Added `scripts/tier4_source_balanced_pre_event_echem_front_coupling_audit.py` and ran it on Isambard. This audit joins the source-balanced pre-event ROI/front/kymograph descriptors to the cycle-level electrochemical regime atlas, then asks which optical proxies are explained by echem/context and which event-bin effects remain after source plus echem/context residualization.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_echem_front_coupling_audit`
- `derived_local/source_balanced_pre_event_echem_front_coupling_audit`

Cohort/result snapshot:

- Joined table: 128 ROI rows, 64 cycles, 14 source movies.
- Conditioning set: 39 cycle-level echem descriptors, 15 acquisition/context descriptors, and 20 front/mask/kymograph targets.
- Strongest raw event-bin result is clean near-pre versus post/control `masked_minus_background_mean_slope`: n=32 versus 52, median difference 0.005637, AUC 0.797, Mann-Whitney p=5.48e-06.
- Raw clean near-pre versus post/control `kymograph_temporal_energy` remains visible: median difference 3.87e-06, AUC 0.717, p=0.000898.
- The strongest echem/optical correlation is `shape_V_mean` versus `kymograph_temporal_energy`: rho=0.592, n=122, p=7.05e-13. Other leading echem correlations also concentrate on kymograph temporal energy and masked-background slope.
- Source+echem residualization weakens the raw near-pre optical separations. The top residual event-bin row is clean near-pre versus post/control `front_radius_slope_px_per_norm_time`: residual median difference 0.210 px/norm time, AUC 0.641, p=0.080.
- The source+echem residual `front_radius2_slope_px2_per_norm_time` row is similar but underpowered: residual median difference 2.616 px^2/norm time, AUC 0.634, p=0.0965.
- Echem/context/source models explain large fractions of some proxy variance, especially `front_gradient_strength_median` (0.943 variance explained) and `mask_centroid_path_px` (0.916), reinforcing that raw ROI readouts are heavily context-structured.

Interpretation:

This audit is an important guardrail. Cycle-level echem descriptors explain substantial structure in kymograph energy and mask/front proxies, and source+echem residualization removes most of the strongest raw clean-pre contrast/mask signal. The remaining residual evidence is a modest front-radius/front-radius-squared slope hint rather than a calibrated diffusion or phase-boundary result. These outputs support manual front QC and better source/echem matched sampling; they argue against presenting the current automatic optical signals as standalone causal precursors or deployable warnings.


## 2026-05-22 Source-Balanced Pre-Event Echem-Matched Residual Audit

Added `scripts/tier4_source_balanced_pre_event_echem_matched_residual_audit.py` and ran it on Isambard. This audit consumes the echem/front joined table, explicitly pairs near-pre ROI rows to control rows using source/acquisition/context plus cycle-level echem descriptors, then tests the source+echem residual front/kymograph outcomes.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_echem_matched_residual_audit`
- `derived_local/source_balanced_pre_event_echem_matched_residual_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies.
- Matching used 54 descriptors: 15 context/acquisition/baseline features and 39 cycle-level echem features.
- Outcome table tests 51 raw, source-context residual, and source+echem residual front/mask/kymograph features.
- Pair counts are 32 for most source-penalized/global comparisons; same-source pairs exist for near-vs-mid (20), near-vs-post/control (12), and near-vs-any-non-near (32), but not for near-vs-far.
- The strongest source+echem residual matched row is same-source near-vs-mid `apparent_diffusion_q70_um2_per_norm_time`: n=14 valid pairs, residual median near-minus-control difference 0.930, positive fraction 0.929, sign-flip p=0.0010.
- Same-source near-vs-mid `front_radius_q70_slope_px_per_norm_time` also remains positive after source+echem conditioning: n=14, residual median difference 1.427 px/norm time, p=0.0030.
- Same-source near-vs-any-non-near `apparent_diffusion_q70_um2_per_norm_time` remains positive: n=16, residual median difference 0.665, p=0.0030.
- The best near-vs-far source+echem residual row is `front_radius2_slope_px2_per_norm_time` under source-penalized echem/context matching: n=20, residual median difference 10.58 px^2/norm time, positive fraction 0.700, sign-flip p=0.0090.
- Near-vs-post/control residual signs can differ for some features, especially q70 front/apparent-diffusion residuals under source-penalized matching, which keeps this as a review/hypothesis-prioritization result rather than a monotone precursor claim.

Interpretation:

This extends the echem/front guardrail into explicit matched residual testing. It supports a review-worthy residual front/apparent-diffusion-like signal in same-source near-vs-mid and near-vs-any controls, while the near-vs-far result still depends on cross-source matching because the raw source lattice lacks same-source far controls. The guarded conclusion is narrower than the raw readouts: source/echem-conditioned front-radius and apparent-diffusion proxies remain promising manual-QC targets, but they are still automatic optical residuals, not calibrated diffusion coefficients, validated phase boundaries, or causal warning models.


## 2026-05-22 Source-Balanced Pre-Event Front-Consensus Audit

Added `scripts/tier4_source_balanced_pre_event_front_consensus_audit.py` and ran it on Isambard. This audit combines the echem/front joined table with the echem-matched residual pairs to test whether the automatic front-motion evidence is internally coherent across q60/q70/q80 slopes, radius/radius-squared fits, monotonicity, gradient coherence, and source+echem residual outward-front proxies.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_front_consensus_audit`
- `derived_local/source_balanced_pre_event_front_consensus_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies.
- The audit used 320 matched pairs from the source/echem-matched residual audit.
- Consensus features include `front_consensus_score`, residual/raw outward-front z-means, q-slope sign fraction, q-slope mean, q-slope coefficient of variation, and a front-quality score.
- Unpaired event-bin consensus remains weak: best near-vs-any-non-near row is `front_quantile_positive_fraction`, median difference 0.333, AUC 0.599, p=0.0716.
- Matched-pair consensus is much stronger. Same-source near-vs-mid `front_consensus_score` has n=20 pairs, median near-minus-control difference 4.946, positive fraction 0.650, sign-flip p=0.0010.
- Same-source near-vs-mid `front_raw_outward_z_mean` has median difference 2.473, positive fraction 0.750, p=0.0010.
- Source-penalized echem/context near-vs-mid `front_consensus_score` has n=32 pairs, median difference 2.428, positive fraction 0.719, p=0.0010.
- Source-penalized echem/context near-vs-mid `front_residual_outward_z_mean` has n=28 pairs, median difference 0.817, p=0.0010.
- Continuous pre-event clock tests remain weak: `front_consensus_score` versus event proximity has rho approximately -0.005, p=0.965, so this is not yet a smooth monotone precursor clock.
- The top consensus-ranked ROI is `source_balanced_cycle38_rank42_obj2_5_c2_x10_260623`, but it is far-pre; the top near-pre examples include cycles 154, 151, and 80. This reinforces that consensus ranking is a manual-QC queue, not an event label.

Interpretation:

The consensus audit sharpens the current physics picture. A broad unpaired classifier does not cleanly separate near-pre rows, and continuous global clocks are weak. However, once matched by source/echem/acquisition context, near-pre rows show stronger coherent outward-front motion relative to mid-pre and non-near controls. The defensible claim is therefore paired, conditional, and review-oriented: source/echem-conditioned front propagation remains a promising automatic proxy for manual phase-front QC, but it is still not calibrated diffusion, validated phase-boundary motion, particle-identity proof, or a causal warning model.

## 2026-05-22 Source-Balanced Pre-Event Echem-Matched Far-Control Audit

Added `scripts/tier4_source_balanced_pre_event_echem_matched_far_control_audit.py` and ran it on Isambard. This audit targets the raw source-lattice gap that no source contains both near-pre and far-pre rows. It uses the next-best control design: nearest far-pre controls matched by source class plus crop/acquisition context, with an echem-augmented variant using cycle-level regime descriptors.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_echem_matched_far_control_audit`
- `derived_local/source_balanced_pre_event_echem_matched_far_control_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies; event bins include 32 near-pre rows and 22 far-pre rows.
- Each matching scheme forms 32 near-vs-far pairs: global echem/context, same-source-class context-only, and same-source-class echem/context.
- Source-class+echem/context matching uses 3 far-control sources; global echem/context uses 4 far-control sources.
- The top paired row overall is same-source-class context-only `front_radius2_slope_px2_per_norm_time`: n=19 valid pairs, median near-far difference 15.95 px^2/norm time, positive fraction 0.842, sign-flip p=0.0010.
- The echem-augmented source-class version keeps the same front-radius-squared direction: n=20 valid pairs, median difference 8.79 px^2/norm time, positive fraction 0.750, p=0.0010.
- The matched front-radius slope also remains positive under source-class+echem/context matching: n=20, median difference 0.675 px/norm time, positive fraction 0.750, p=0.0040.
- Matched kymograph temporal energy remains directionally higher in near-pre rows under source-class+echem/context matching: median difference 2.53e-06, positive fraction 0.625, p=0.0020.
- Mask/contrast signals also persist in some schemes, including global echem/context `masked_minus_background_mean_slope` median difference 0.00264, p=0.0040.

Interpretation:

This strengthens the radial-kymograph near-pre front-motion result against a more specific cross-source far-control design, including echem/context matching. It still does not solve the missing same-source near-vs-far problem: controls are cross-source, only 3-4 far-control sources are used, and controls can be reused. The defensible conclusion is that near-pre front-radius and radius-squared movement remains review-worthy after source-class/echem/context matching, not that calibrated diffusion, particle identity, phase-boundary tracking, or causal precursor forecasting has been proven.

## 2026-05-22 Source-Balanced Pre-Event Consensus Review Queue

Added `scripts/tier4_source_balanced_pre_event_consensus_review_queue.py` and ran it on Isambard. This audit does not create a new physical claim; it consolidates the scattered pre-event evidence into one manual-QC work queue by combining source-invariant review scores, radial-kymograph front motion, source/echem residual front proxies, and matched-control support from the residual and far-control audits.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_consensus_review_queue`
- `derived_local/source_balanced_pre_event_consensus_review_queue`

Cohort/result snapshot:

- Ranked 128 source-balanced pre-event candidates across 64 cycles and 14 source movies.
- Priority tiers: 21 `matched_support_front_qc` rows and 107 routine-review rows.
- The top candidate is `source_balanced_cycle151_rank28_obj1_17_c2_x10_HighHighCOV_150723`: near-pre, consensus score 0.932, matched-positive support count 56, cycle 151, 5 cycles to the next event.
- The previous top review-packet candidate `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723` remains near the top: rank 2, score 0.906, matched-positive support count 55.
- The top 10 candidates are all near-pre rows and concentrate in the HighHighCOV source `17_c2_x10_HighHighCOV_150723` plus supporting near-pre candidates from `9_c2_x10_010723`, which is useful for manual QC but also highlights residual source/local-condition concentration.
- The queue writes a full ranked CSV and a top-40 CSV with ROI IDs, source/cycle/bin metadata, consensus score, prior review score, key front/echem-residual proxies, matched-support counts, support details, and the source ROI tensor path.

Interpretation:

This is an operational bridge from automatic analysis to manual review. It makes the next human-QC step concrete: review the consensus top rows first, especially ranks 1-7 from the near-pre HighHighCOV sequence, while keeping matched-control and source/echem guardrails visible. The queue assigns no accept/reject labels and does not validate particle identity, front masks, calibrated diffusion, phase-boundary tracking, degradation causality, or deployable warnings.

## 2026-05-22 Source-Balanced Pre-Event Front Consensus Audit

Added `scripts/tier4_source_balanced_pre_event_front_consensus_audit.py` and ran it on Isambard. This audit asks whether the pre-event front signal is internally coherent across multiple automatic front proxies rather than depending on a single radius or apparent-diffusion descriptor. It combines raw outward-front slopes, source+echem residual outward-front slopes, radius-quantile sign agreement, monotonicity/gradient quality, and matched-control evidence.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_front_consensus_audit`
- `derived_local/source_balanced_pre_event_front_consensus_audit`

Cohort/result snapshot:

- Input table: 128 ROI rows, 64 cycles, 14 source movies.
- Consensus features tested include `front_consensus_score`, residual/raw outward-front z means, quantile-positive fraction, q-slope mean, q-slope CV, and front-quality score.
- Unpaired event-bin separation is weak: best row is near-vs-any-non-near `front_quantile_positive_fraction`, n=32 versus 96, median difference 0.333, AUC 0.599, p=0.0717.
- Matched consensus evidence is stronger: same-source near-vs-mid `front_consensus_score` has n=20, median near-control difference 4.946, sign-flip p=0.0010.
- Same-source near-vs-mid `front_raw_outward_z_mean` also has median difference 2.473, p=0.0010.
- Source-penalized echem/context near-vs-mid `front_consensus_score` has n=32, median difference 2.428, p=0.0010.
- Same-source near-vs-any-non-near `front_raw_outward_z_mean` remains positive: n=27, median difference 1.074, p=0.0010.
- The top consensus-ranked candidates feed into the consensus review queue rather than creating labels.

Interpretation:

This is a coherence stress test for the front-motion hypothesis. It shows that unpaired event-bin front consensus is not strong enough on its own, while matched same-source near-vs-mid and source/echem-matched comparisons preserve a coherent outward-front signal. The result supports prioritizing front-consensus candidates for manual QC, but it remains automatic optical evidence rather than calibrated diffusion, validated phase-boundary tracking, particle identity, or causal degradation proof.

## 2026-05-22 Source-Balanced Pre-Event Consensus Visual Packet

Added `scripts/tier4_source_balanced_pre_event_consensus_visual_packet.py` and ran it on Isambard. This converts the consensus review queue into a visual manual-QC packet by rendering frame strips, stable-mask overlays, and radial-kymograph/front-trace plots for the top-ranked candidates, plus a contact sheet and a manifest linking visual assets back to ROI IDs and proxy scores.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_consensus_visual_packet`
- `derived_local/source_balanced_pre_event_consensus_visual_packet`

Cohort/result snapshot:

- The queue contains 128 candidates; the packet requested and rendered the top 24.
- Rendered candidates span 8 source movies.
- For each candidate, the packet writes a frame-strip PNG, a stable-mask overlay PNG, and a radial-kymograph/front-trace PNG.
- Created contact sheet `consensus_visual_contact_sheet.png` from the top frame strips.
- Event-bin composition of rendered candidates is 19 near-pre rows, 1 mid-pre row, 1 far-pre row, 2 post-event rows, and 1 no-near-event control row, preserving a small amount of control/context material while focusing manual effort on near-pre candidates.
- Top visual candidate is `source_balanced_cycle151_rank28_obj1_17_c2_x10_HighHighCOV_150723`: consensus rank 1, near-pre, score 0.932, matched-positive support count 56, front radius2 slope 16.47 px^2/norm time, source+echem residual front slope 1.17 px/norm time.
- The prior top review-packet candidate `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723` is rank 2 in the visual packet, score 0.906, matched-positive support count 55.

Interpretation:

This is the practical handoff from automatic AI/physics evidence to manual QC. It does not add a new model claim; it makes the highest-priority candidates inspectable with synchronized scalar evidence, frame strips, mask overlays, and radial kymographs. The packet should be used to decide which ROIs have interpretable particle/front behavior and which are artifacts or ambiguous crops. It assigns no labels and does not validate particle identity, front masks, calibrated diffusion, phase-boundary motion, degradation causality, or deployable warnings.

## 2026-05-22 Source-Balanced Pre-Event Visual Sanity Audit

Added `scripts/tier4_source_balanced_pre_event_visual_sanity_audit.py` and ran it on Isambard. This audit adds a machine-readable QC layer to the consensus visual packet by scoring each rendered candidate from the ROI tensors for stable-mask size/edge contact, frame-to-frame centroid stability, focus/blur proxy, temporal contrast, radial-front trace fit, and front-trace monotonicity.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_visual_sanity_audit`
- `derived_local/source_balanced_pre_event_visual_sanity_audit`

Cohort/result snapshot:

- Audited all 24 rendered consensus visual candidates; all 24 ROI tensors loaded successfully.
- The 24 candidates span 8 source movies.
- Automatic visual sanity flags are 9 `reviewable_auto` and 15 `uncertain_auto`; no rendered candidate was assigned the hard `artifact_risk_auto` flag.
- Median visual sanity score is 0.579.
- Event-bin composition remains 19 near-pre rows, 1 mid-pre row, 1 far-pre row, 2 post-event rows, and 1 no-near-event control row.
- The main source concentrations are `17_c2_x10_HighHighCOV_150723` with 8 candidates and median sanity 0.541, `9_c2_x10_010723` with 5 candidates and median sanity 0.698, and `6_c2_x10_270623_2` with 4 candidates and median sanity 0.632.
- Event-bin tests on the visual sanity score are weak and non-significant: near-pre median 0.590 versus rendered non-near median 0.572, p = 0.679; near-pre versus post/control p = 0.191.
- The top artifact-risk/low-sanity rows are still `uncertain_auto`, not hard rejects, so they should remain in manual review with caution rather than being dropped automatically.

Interpretation:

This is a QC guardrail for the visual packet, not an event classifier or physical mechanism result. The candidates are generally inspectable enough for manual review, but most remain automatically uncertain and source concentration is still visible. Manual QC should prioritize the `reviewable_auto` rows while retaining the uncertainty flags, source metadata, and matched-control evidence when judging whether any ROI has interpretable particle/front behavior.

## 2026-05-22 Source-Balanced Pre-Event Visual QC Modes

Added `scripts/tier4_source_balanced_pre_event_visual_qc_modes.py` and ran it on Isambard. This is a conservative automatic visual/front plausibility layer over the consensus visual packet: it reloads the rendered ROI sequences, recomputes stable masks and radial front traces, scores front-trace fit, direction consistency, mask fraction, temporal SNR, kymograph energy, and artifact risk, then clusters the 24 rendered candidates into review modes.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_visual_qc_modes`
- `derived_local/source_balanced_pre_event_visual_qc_modes`
- `source_balanced_pre_event_visual_qc_modes.csv`
- `source_balanced_pre_event_visual_qc_mode_summary.csv`
- `visual_qc_priority_contact_sheet.png`
- `source_balanced_pre_event_visual_qc_modes_summary.json`

Results:

- Scored all 24 rendered consensus candidates.
- Automatic tiers: 8 `front_plausible_followup`, 16 `routine_or_low_front_plausibility`, and 0 accepted-front/priority labels.
- Mode counts: 23 `low front-plausibility context` and 1 `moderate front-like followup`.
- Top automatic visual-QC candidate is `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723`, consensus rank 2, near-pre-event, visual review score 0.590, front plausibility score 0.262, artifact risk 0.050.
- The previous consensus rank 1 candidate remains high but is downgraded to follow-up rather than priority because the automatically traced front has weak linear fit/direction consistency.

Interpretation: this is a useful negative/guardrail result. The scalar consensus/matched-control evidence still identifies near-pre-event candidates worth reviewing, but the rendered kymograph/front-trace plausibility audit does not yet support automatic acceptance of coherent phase-front motion. Diffusion and phase-boundary claims should therefore remain gated on manual ROI/front QC and calibration. The new contact sheet orders review by visual-QC score so manual inspection can start with the strongest follow-up cases while preserving likely low-plausibility context rows.

## 2026-05-22 Source-Balanced Pre-Event Strict QC-Gated Front Audit

Added `scripts/tier4_source_balanced_pre_event_strict_qc_gated_front_audit.py` and ran it on Isambard. This audit turns the consensus visual packet, visual sanity metrics, and visual QC modes into explicit gate-level review decisions for candidate front motion and diffusion-readout claims.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_strict_qc_gated_front_audit`
- `derived_local/source_balanced_pre_event_strict_qc_gated_front_audit`
- `source_balanced_pre_event_strict_qc_gated_front_candidates.csv`
- `source_balanced_pre_event_strict_qc_gate_counts.csv`
- `strict_qc_front_review_contact_sheet.png`
- `source_balanced_pre_event_strict_qc_gated_front_summary.json`

Results:

- Scored all 24 rendered consensus candidates.
- Gate pass counts: stable mask 24, centroid stability 20, visual sanity 9, visual mode 6, front-trace fit 4, front-direction agreement 11, manual front-review 1, automatic diffusion claim 0.
- Only one candidate survives the automatic manual-front-review gate: `source_balanced_cycle80_rank62_obj2_9_c2_x10_010723`, consensus rank 8, near-pre-event, strict QC score 0.600, visual sanity score 0.761, visual review score 0.506, front-trace r2 0.279, consensus front-radius2 slope 15.449 px^2/norm time.
- Zero candidates pass the automatic diffusion-claim gate.

Interpretation: this is a conservative review-prioritization guardrail. The automatic evidence can nominate one ROI for manual front inspection, but it does not validate particle identity, masks, coherent phase-boundary motion, calibrated diffusion, degradation causality, or deployable warnings. Diffusion coefficient/material transport claims remain blocked until manual ROI/front QC and calibration are done.

## 2026-05-22 Source-Balanced Pre-Event Phase Kinetics Audit

Added `scripts/tier4_source_balanced_pre_event_phase_kinetics_audit.py` and ran it on Isambard. This audit measures particle-region-only optical kinetics on the 128 source-balanced pre-event ROI tensors: stable particle masks with per-frame fallback, masked-minus-background intensity trajectories, q55/q65/q75 phase-fraction slopes, logistic timing summaries, Avrami-style descriptive exponents, matched controls, echem correlations, and source summaries.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_phase_kinetics_audit`
- `derived_local/source_balanced_pre_event_phase_kinetics_audit`
- `source_balanced_pre_event_phase_kinetics_features.csv`
- `source_balanced_pre_event_phase_kinetics_event_tests.csv`
- `source_balanced_pre_event_phase_kinetics_matched_tests.csv`
- `source_balanced_pre_event_phase_kinetics_correlations.csv`
- `source_balanced_pre_event_phase_kinetics_summary.json`

Results:

- Loaded all 128 ROI tensors across 14 sources and tested 56 stable kinetic features after excluding near-zero-denominator variation-to-net ratios.
- Event-bin counts are 32 near-pre, 22 mid-pre, 22 far-pre, 40 post-event, and 12 no-near-event control rows.
- Best event-bin row is near-vs-any-non-near raw `masked_minus_bg_slope`: AUC 0.816, AP 0.634, near-minus-control median difference 0.00376, Mann-Whitney p = 8.99e-08, but source eta2 remains high at 0.504.
- The same masked-minus-background slope is consistent across near-vs-post-control (AUC 0.799, p = 4.80e-06), near-vs-mid-pre (AUC 0.849, p = 1.55e-05), and near-vs-far-pre (AUC 0.825, p = 5.76e-05).
- The best source-residual rows are weaker but still present, for example near-vs-mid-pre `q55_phase_fraction_delta` AUC 0.739, median difference 0.0138, p = 0.00319.
- Top matched-control rows are global echem-context near-vs-far comparisons: `masked_mean_total_variation` median near-control difference 0.00843, p = 0.000500, and `masked_minus_bg_slope` median difference 0.00747, positive fraction 0.781, p = 0.000500.
- Strongest echem correlation is `q55_phase_fraction_total_variation` versus `shape_V_mean`, rho 0.523, p = 6.52e-10.

Interpretation: this is a useful optical phase-kinetics signal that connects particle-region intensity dynamics, event proximity, and echem context. It strengthens the case for masked optical kinetics as a pre-event review feature and a higher-yield manual-QC target than automatic radial front acceptance, but source structure is still substantial and the logistic/Avrami values are descriptive summaries only. It does not provide manual phase labels, validated particle identity, calibrated reaction constants, diffusion coefficients, phase-boundary proof, or causal degradation evidence.

## 2026-05-22 Source-Balanced Pre-Event Multimodal Predictor

Added `scripts/tier4_source_balanced_pre_event_multimodal_predictor.py` and ran it on Isambard. This audit asks whether the new masked phase-kinetics features add source-heldout predictive value beyond existing echem, source/echem-residual front, consensus-review, and visual-QC features. It uses leave-source logistic models over weak event-relative labels and compares feature families rather than treating any model as deployable.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_multimodal_predictor`
- `derived_local/source_balanced_pre_event_multimodal_predictor`

Results:

- Input table has 128 ROI rows, 64 cycles, and 14 sources.
- Feature family sizes: 7 echem-context, 19 front/echem-residual, 5 consensus/visual-QC, 64 phase-kinetics, 31 no-kinetics signal, 93 all-signal, and 6 object-context guardrail features.
- Best clean-pre16-vs-post/control leave-source row is `consensus_visual_qc` with source-confound filtering: n=106, 13 held-out sources, AUC 0.826, AP 0.879, mean/max raw source eta2 0.549/0.818.
- The same target using no-kinetics signal features reaches AUC 0.735/AP 0.777 raw, while all-signal drops to AUC 0.647/AP 0.701, consistent with overfitting or source-structured feature mixing in this small cohort.
- Near-vs-post/control is easiest for consensus/no-kinetics features: `consensus_visual_qc` raw AUC 0.999/AP 0.998 and no-kinetics raw AUC 0.963/AP 0.945, but these are review/consensus features and should not be interpreted as independent prospective warning performance.
- Near-vs-far-pre is also dominated by consensus/echem context: `consensus_visual_qc` raw AUC 0.991/AP 0.994, echem-context source-mean-residual AUC 0.920/AP 0.937.
- Phase kinetics alone are modest under leave-source modeling: clean-pre16-vs-post/control raw AUC 0.548/AP 0.620; near-vs-post/control source-mean-residual AUC 0.594/AP 0.512; near-vs-far-pre source-mean-residual AUC 0.491/AP 0.700.
- However, phase kinetics improve over front/echem-residual features in family-delta checks, for example near-vs-post/control source-mean-residual phase kinetics versus front/echem residuals gives dAUC +0.127 and dAP +0.109; near-vs-far-pre raw gives dAUC +0.131 and dAP +0.149.

Interpretation:

This is a useful model-selection guardrail. Masked phase kinetics add signal relative to automatic front/echem-residual features, but they do not yet make a robust standalone leave-source predictor. The strongest source-heldout discrimination comes from consensus/visual-QC and echem-context features that are partly review-ranking or source/event-bin structured, so they should guide manual QC and experiment selection, not be reported as deployable precursor accuracy. The practical next target is a smaller, regularized manual-QC candidate set that combines consensus/QC ranking with masked kinetics rather than feeding every automatic feature into one model.

## 2026-05-22 Source-Balanced Pre-Event Front/Kinetic Concordance Audit

Added `scripts/tier4_source_balanced_pre_event_front_kinetic_concordance_audit.py` and ran it on Isambard. This audit joins the source-balanced pre-event phase-kinetics features, front-consensus proxies, visual QC scores, and strict front gates to ask whether the same ROI candidates carry both masked optical kinetics and front-like evidence.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_front_kinetic_concordance_audit`
- `derived_local/source_balanced_pre_event_front_kinetic_concordance_audit`
- `source_balanced_pre_event_front_kinetic_concordance_ranked_candidates.csv`
- `source_balanced_pre_event_front_kinetic_concordance_event_tests.csv`
- `source_balanced_pre_event_front_kinetic_concordance_correlations.csv`
- `source_balanced_pre_event_front_kinetic_concordance_summary.json`

Results:

- Ranked 128 ROI rows across 14 sources.
- Tier counts are 99 routine/low-concordance, 13 front-only guardrail, 7 kinetic-only guardrail, 4 near-pre front/kinetic review, 4 non-near front/kinetic concordant, and 1 strict-gate manual-front-review row.
- Top ranked row is `source_balanced_cycle154_rank30_obj2_17_c2_x10_HighHighCOV_150723`, near-pre, score 1.669, kinetic evidence 2.030, front evidence 1.585, tier `near_pre_front_kinetic_review`.
- The strict-QC manual-front candidate `source_balanced_cycle80_rank62_obj2_9_c2_x10_010723` remains in the top set with score 1.124, kinetic evidence 0.672, front evidence 1.413, tier `strict_gate_manual_front_review`.
- Best event-bin row is near-vs-any-non-near raw `kinetic_evidence_score`: AUC 0.770, AP 0.622, median difference 0.956, p = 5.00e-06. The composite concordance score reaches AUC 0.768, p = 3.90e-05.

Interpretation: this audit turns the phase-kinetics/front-QC layers into an actionable review queue while preserving discordance as a guardrail. Near-pre rows with both kinetic and front evidence are good manual-QC targets, but front-only and kinetic-only rows show why automatic phase-boundary or diffusion claims remain blocked without manual inspection and calibration.

## 2026-05-22 Source-Balanced Pre-Event Front/Kinetic Source-Null Audit

Added `scripts/tier4_source_balanced_pre_event_front_kinetic_null_audit.py` and ran it on Isambard. This audit stress-tests the front/kinetic concordance score and component features with raw, source-residual, and within-source-rank transforms, then compares observed AUC/correlation against source-stratified label shuffles so raw near-pre separation is not mistaken for source composition.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_front_kinetic_null_audit`
- `derived_local/source_balanced_pre_event_front_kinetic_null_audit`
- `source_balanced_pre_event_front_kinetic_null_tests.csv`
- `source_balanced_pre_event_front_kinetic_proximity_tests.csv`
- `source_balanced_pre_event_front_kinetic_null_summary.json`

Results:

- Tested 128 ROI rows, 14 sources, and 9 front/kinetic/QC features with 50 source-stratified permutations per test and no bootstrap resampling in the committed fast audit; p-values therefore have coarse resolution and should be treated as directional guardrails.
- Strongest null-tested row is near-vs-mid-pre raw `kinetic_evidence_score`: AUC 0.820, AP 0.889, median difference 0.969, source-stratified permutation p = 0.0196, null AUC median 0.694, null AUC p95 0.775, but only 3 eligible mixed-label sources.
- Near-vs-any-non-near raw `masked_minus_bg_slope` also survives the coarse null: AUC 0.816, AP 0.634, median difference 0.00376, p = 0.0196, null AUC median 0.750, p95 0.801, 5 eligible sources.
- Near-vs-any-non-near raw `front_kinetic_concordance_score` remains positive but weaker: AUC 0.768, AP 0.526, p = 0.0392, null AUC median 0.704, p95 0.741, 5 eligible sources.
- Source-residual versions are much weaker: near-vs-any-non-near `kinetic_evidence_score` source-residual AUC 0.598 and near-vs-post/control `front_kinetic_concordance_score` source-residual AUC 0.591.
- The top event-proximity row is within-source-rank `strict_qc_priority_score`, rho = -0.450, source-stratified permutation p = 0.0196; the sign indicates this strict front-QC score is not a simple closer-to-event clock.

Interpretation: masked optical kinetics are the most robust part of the current front/kinetic evidence under source-aware nulls. The composite concordance score is useful for review prioritization, but the weakened source-residual/rank rows and small mixed-label source counts mean it should not be treated as source-invariant physics, manual front validation, causal degradation proof, or calibrated diffusion evidence.

## 2026-05-22 Source-Balanced Pre-Event Manual-QC Decision Packet

Added `scripts/tier4_source_balanced_pre_event_manual_qc_decision_packet.py` and ran it on Isambard. This packet consolidates front/kinetic concordance, source-stratified null evidence, strict front gates, and rendered visual-asset paths into an operational manual-QC queue with explicit action tiers and review questions.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_manual_qc_decision_packet`
- `derived_local/source_balanced_pre_event_manual_qc_decision_packet`
- `source_balanced_pre_event_manual_qc_decision_queue.csv`
- `source_balanced_pre_event_manual_qc_top40.csv`
- `source_balanced_pre_event_manual_qc_visual_asset_manifest.csv`
- `source_balanced_pre_event_manual_qc_action_summary.csv`
- `source_balanced_pre_event_manual_qc_decision_summary.json`

Results:

- Ranked 128 ROI rows across 14 sources and correctly identified 24 rows with rendered visual assets from the consensus visual packet.
- Action counts are 96 `routine_or_low_concordance`, 13 `review_front_only_guardrail`, 9 `review_kinetic_only_guardrail`, 5 `review_front_and_kinetics_first`, 4 `context_control_concordant`, and 1 `review_strict_front_gate`.
- The top-40 review packet contains all 5 front/kinetic-first rows, the 1 strict-front-gate row, 8 of 9 kinetic-only guardrails, 11 of 13 front-only guardrails, 4 non-near concordant controls, and 11 routine comparators.
- Top decision candidate is `source_balanced_cycle80_rank62_obj2_9_c2_x10_010723`, action `review_strict_front_gate`, decision score 1.181. Its manual question is whether a human can confirm a coherent outward/front-like trace in the strip, overlay, and radial kymograph.
- Next front/kinetic-first candidates include `source_balanced_cycle151_rank28_obj1_17_c2_x10_HighHighCOV_150723`, `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723`, `source_balanced_cycle78_rank61_obj2_9_c2_x10_010723`, and `source_balanced_cycle154_rank30_obj2_17_c2_x10_HighHighCOV_150723`.
- No row passes the automatic diffusion-claim gate; all diffusion interpretation remains blocked pending manual front validation and calibration.

Interpretation: this converts the scattered review artifacts into a compact decision ledger. It is the right next manual-QC handoff: inspect the strict-front candidate first, then the near-pre front/kinetic-first rows, then kinetic-only and front-only guardrails as failure-mode controls. The packet prioritizes review only; it does not create manual labels, validate fronts, calibrate diffusion, or establish source-invariant causal physics.

## 2026-05-22 Source-Balanced Pre-Event Manual-QC Visual Packet

Added `scripts/tier4_source_balanced_pre_event_manual_qc_visual_packet.py` as the reproducible renderer for the manual-QC decision queue and indexed the output in the project synthesis. The script renders direct particle-crop visual evidence for every action tier, rather than only the earlier consensus top candidates.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_manual_qc_visual_packet`
- `derived_local/source_balanced_pre_event_manual_qc_visual_packet`
- `source_balanced_pre_event_manual_qc_visual_assets.csv`
- `manual_qc_visual_contact_sheet.png`
- `source_balanced_pre_event_manual_qc_visual_summary.json`

Results:

- Rendered 40 of 40 requested queue rows from the 128-row manual-QC decision queue.
- The rendered set spans 12 sources and all review tiers: 1 `review_strict_front_gate`, 5 `review_front_and_kinetics_first`, 11 `review_front_only_guardrail`, 8 `review_kinetic_only_guardrail`, 4 `context_control_concordant`, and 11 `routine_or_low_concordance`.
- Event-bin coverage includes 23 near-pre-event rows, 4 mid-pre-event rows, 4 far-pre-event rows, 6 post-event rows, and 3 no-near-event controls.
- Each rendered row has a frame strip, mask overlay, radial kymograph, and contact-sheet entry. The top visual row is still `source_balanced_cycle80_rank62_obj2_9_c2_x10_010723`, the strict-front-gate candidate.

Interpretation: this closes the biggest practical handoff gap for manual QC. The reviewer can now inspect one contact sheet plus per-ROI strip/overlay/kymograph assets for the exact decision queue that controls whether front, phase-boundary, and diffusion-proxy evidence should be trusted. The packet still assigns no labels and does not validate particle identity, front masks, calibrated diffusion, or causality.

## 2026-05-22 Source-Balanced Pre-Event Blinded Manual-QC Workbook

Added `scripts/tier4_source_balanced_pre_event_manual_qc_blind_workbook.py` and ran it on Isambard. This takes the top-40 rendered manual-QC visual packet and creates a randomized reviewer-facing workbook plus a separate hidden key so manual front/phase labels can be collected without exposing event timing, source, cycle, ROI ID, or automatic action tier during scoring.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_manual_qc_blind_workbook`
- `derived_local/source_balanced_pre_event_manual_qc_blind_workbook`
- `source_balanced_pre_event_manual_qc_blinded_workbook.csv`
- `source_balanced_pre_event_manual_qc_blinded_key.csv`
- `source_balanced_pre_event_manual_qc_blinded_review.html`
- `source_balanced_pre_event_manual_qc_rubric.json`
- `source_balanced_pre_event_manual_qc_blind_summary.json`

Results:

- Built 40 blinded review rows from 40 rendered candidates using seed 20260522.
- The reviewer-facing workbook contains image paths and label fields only; validation confirmed it does not include `roi_id`, `source_stem`, `cycleNo`, `event_relative_bin`, or `manual_qc_action_tier`.
- The hidden key preserves 12 sources, event bins (23 near-pre, 4 mid-pre, 4 far-pre, 6 post-event, 3 no-near-event controls), and action tiers (1 strict-front, 5 front/kinetic-first, 8 kinetic-only, 11 front-only, 4 context-control concordant, 11 routine/low-concordance).
- Rubric fields explicitly ask for particle identity, front-like motion, front direction, phase-boundary interpretability, particle-local kinetic signal, artifact/drift risk, diffusion interpretation permission, and final manual decision.

Interpretation: this is the missing bridge from automatic physics proxies to defensible manual validation. It prevents automatic ranking and event proximity from biasing the first scoring pass, while retaining a hidden key for post-label enrichment tests. It still assigns no labels and keeps diffusion claims blocked unless manual review supports particle identity, front mask quality, and front-motion interpretability.


## 2026-05-22 Diffusion Claim Readiness Audit

Added scripts/tier4_diffusion_claim_readiness_audit.py and ran it on Isambard to assemble a single go/no-go ledger for calibrated diffusion claims. The audit joins the calibration metadata, apparent-diffusion calibration bounds, diffusion physics consistency gates, control-balanced diffusion sanity checks, manual-QC gate state, and manual visual-review packet into separate criteria and candidate tables. It is deliberately a readiness/guardrail audit, not a new diffusion estimator.

Remote output:

/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/diffusion_claim_readiness_audit

Local compact mirror:

derived_local/diffusion_claim_readiness_audit

Key outputs:

- diffusion_claim_readiness_criteria.csv
- diffusion_claim_readiness_candidates.csv
- diffusion_claim_readiness_summary.json
- README.md

Results:

- Overall status remains not_ready_for_calibrated_diffusion_claim.
- 14 criteria were evaluated: 1 pass, 5 partial, 7 fail, and 1 blocked pending manual QC.
- 8 hard blockers remain: missing HDF5 spatial-calibration metadata, per-ROI HDF5 timing stability, radius2 fit quality, q70 positive confidence interval, no publication-ready diffusion candidates, no accepted manual-QC labels, no control-balanced publication candidates, and non-significant event/control diffusion separability.
- Candidate ledger contains 60 rows. There is 1 automatic physics-consistent candidate and 0 publication-ready diffusion candidates.
- Top candidate remains cycle78_rank22_obj2 from source 9_c2_x10_010723: it passes most automatic physics gates but remains blocked by q70 positive-CI failure and the publication-ready gate.

Interpretation:

- The current project can responsibly report optical-front/apparent-motion proxy analyses and use the ranked candidate ledger for manual review.
- It should not report calibrated material diffusion coefficients until raw spatial calibration, timestamp semantics, manual particle/front labels, and the control-balanced sanity gates pass.

## 2026-05-22 Source-Balanced Pre-Event Masked Rollout Benchmark

Added `scripts/tier4_source_balanced_pre_event_masked_rollout_benchmark.py` and ran it on Isambard as a particle-region-only held-out-tail benchmark for the 128 source-balanced pre-event ROI sequences. The benchmark derives a stable particle mask only from the history frames, trains/sets baselines on the first 60% of each ROI sequence, and scores the held-out tail inside the particle mask and surrounding context.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_masked_rollout_benchmark`
- `derived_local/source_balanced_pre_event_masked_rollout_benchmark`
- `source_balanced_pre_event_masked_rollout_per_roi.csv`
- `source_balanced_pre_event_masked_rollout_frame_samples.csv`
- `source_balanced_pre_event_masked_rollout_method_summary.csv`
- `source_balanced_pre_event_masked_rollout_event_tests.csv`
- `source_balanced_pre_event_masked_rollout_source_summary.csv`
- `source_balanced_pre_event_masked_rollout_summary.json`

Results:

- Processed 128 of 128 ROI rows with 0 failures across 64 cycles and 14 sources.
- Persistence is still the strongest median particle-region baseline: median particle MSE 0.00149, mean particle MSE 0.00312, and 0 non-persistence wins by the median-gain criterion.
- Velocity, pixel-linear, particle-mean-drift, and radial-profile-trend baselines do not beat persistence overall; their median particle-MSE ratios versus persistence are 15.10, 2.40, 1.31, and 1.32, respectively.
- The strongest event-relative signal is not better rollout but higher held-out particle error near events: `persistence_particle_mse` for near-pre-event versus post/control has AUC 0.662, AP 0.619, median positive-negative 0.000675, and MW p=0.0134.
- Near-pre-event versus any non-near row also shows higher particle/context persistence error ratio and higher persistence particle MSE with raw AUC around 0.62.

Interpretation: the held-out-tail experiment strengthens the earlier modeling conclusion. For these slowly varying particle crops, persistence is a hard baseline and simple trend models are not credible simulators. The useful physics-facing signal is the localized prediction difficulty: near-pre-event particles are harder to persist inside a history-derived particle mask, especially versus post/control rows. This supports using masked rollout residual energy as an instability/readiness descriptor, not as a calibrated phase-boundary tracker or diffusion estimator.

Guardrail: this benchmark uses history-derived automatic particle masks and held-out future frames within ROI crops. It does not provide manual segmentation, calibrated phase-boundary motion, material diffusion coefficients, or causal event evidence.


## 2026-05-22 Residualized Future8 Video-Physics Benchmark

Added `scripts/tier4_residualized_future8_video_physics_benchmark.py` and ran it on Isambard to test whether the short-horizon `future_any_drop_within_8cycles` weak label still contains source-robust optical/video physics after acquisition residualization, cycle balancing, source/source-cohort holdouts, source-stratified permutation, and comparison against echem context.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/residualized_future8_video_physics_benchmark`
- `derived_local/residualized_future8_video_physics_benchmark`
- `residualized_future8_video_physics_metrics.csv`
- `residualized_future8_video_physics_predictions.csv`
- `residualized_future8_video_physics_deltas.csv`
- `residualized_future8_video_physics_summary.json`

Results:

- Evaluated 172 rows across 34 cycles and 12 sources.
- Decision status is `future8_video_physics_status = not_supported_after_controls`.
- Fused video+echem status is `future8_fused_video_echem_incremental_status = not_incremental_over_echem_context`.
- Acquisition context alone remains perfect in the source-cohort split: AUC/AP 1.000/1.000, source-stratified p=0.001996.
- Residualized echem context is moderate but not source-stratified significant: AUC/AP 0.688/0.762, source-stratified p=0.228.
- Strict residualized optical physics fails the source-cohort test: ROC-AUC/AP 0.0046/0.314, source-stratified p=1.000, rho=-0.858. Leave-source residualized optical physics is also weak: ROC-AUC/AP 0.090/0.329.
- Residualized fused video+echem is worse than residualized echem alone in the source-cohort split: AUC/AP 0.590/0.713, source-stratified p=0.896, delta AUC versus echem -0.099.

Interpretation: the current future8 warning label is dominated by acquisition/context and echem structure, not by an independent source-robust optical physics mechanism. Future8 should remain a guardrail/context label, while physics claims should focus on longer-horizon future16, pre-event, manual-QC, and source-invariant tracks.

Guardrail: this benchmark is a falsification/control test. It blocks using future8 video/optical metrics as standalone degradation physics evidence unless future runs pass source/source-cohort holdout, acquisition residualization, cycle balancing, source-stratified permutation, and incremental echem comparisons.


## 2026-05-22 Source-Balanced Pre-Event Observable Forecast

Added `scripts/tier4_source_balanced_pre_event_observable_forecast.py` and ran it on Isambard. This audit forecasts held-out-tail physical observables from the first 60% of each source-balanced pre-event particle crop, then compares prefix-observable models against persistence, context, echem-context, and prefix+echem models under leave-source and leave-cycle splits.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_observable_forecast`
- `derived_local/source_balanced_pre_event_observable_forecast`
- `source_balanced_pre_event_observable_forecast_features.csv`
- `source_balanced_pre_event_observable_forecast_metrics.csv`
- `source_balanced_pre_event_observable_forecast_predictions.csv`
- `source_balanced_pre_event_observable_forecast_incremental.csv`
- `source_balanced_pre_event_observable_forecast_event_diagnostics.csv`
- `source_balanced_pre_event_observable_forecast_summary.json`

Results:

- Evaluated 128 ROI sequences across 64 cycles and 14 sources with 0 failures.
- Best leave-source forecast is `prefix_observables` predicting `tail_contrast_delta`: Spearman rho 0.949, R2 0.954, MAE 0.00320.
- Prefix observables also forecast `tail_particle_mean_delta` under leave-source: rho 0.892, R2 0.933, MAE 0.00324.
- Prefix observables forecast `tail_particle_minus_background_delta` under leave-source: rho 0.817, R2 0.854, MAE 0.00432.
- Front-tail observables are weaker but nonzero: `tail_front_radius_q70_delta` leave-source prefix model has rho 0.532, R2 0.307, MAE 0.345; `tail_front_radius2_slope` has rho 0.412 but negative R2.
- Prefix+echem improves over echem context for some rank correlations, led by `tail_contrast_delta` under leave-source with delta rho 0.616 and delta MAE -0.051.
- Event-relative diagnostics are weak after source-centering: near-pre AUCs range only 0.520-0.549 across tail particle mean, contrast, front radius, radius2 slope, and frame-difference energy.

Interpretation: early particle-crop observables strongly forecast later optical-observable evolution across held-out sources, so observable forecasting is a viable AI task for this dataset. But these forecastable dynamics are not, by themselves, a strong near-pre degradation detector. Use them as physically interpretable rollout/uncertainty descriptors and as inputs to manual-QC review, not as standalone evidence for degradation causality or diffusion.

Guardrail: the audit uses automatic source-balanced particle crops and held-out-tail optical observables. It does not validate particle identity, phase-boundary motion, calibrated diffusion coefficients, or degradation causality.

## 2026-05-22 Source-Balanced Pre-Event Optical-Flow Transport Audit

Added `scripts/tier4_source_balanced_pre_event_optical_flow_transport_audit.py` and ran it on Isambard to estimate apparent particle-local transport from the 128 source-balanced pre-event ROI sequences. The audit uses a history-derived stable particle mask, computes dense Farneback optical flow on normalized ROI crops, and summarizes held-out-tail motion inside the particle mask, boundary band, and context.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_optical_flow_transport_audit`
- `derived_local/source_balanced_pre_event_optical_flow_transport_audit`
- `source_balanced_pre_event_optical_flow_transport_per_roi.csv`
- `source_balanced_pre_event_optical_flow_transport_frame_samples.csv`
- `source_balanced_pre_event_optical_flow_transport_event_tests.csv`
- `source_balanced_pre_event_optical_flow_transport_source_summary.csv`
- `source_balanced_pre_event_optical_flow_transport_summary.json`

Results:

- Processed 128 of 128 ROI rows with 0 failures across 64 cycles and 14 sources.
- Median particle-mask fraction is 0.204, median particle flow magnitude is 7.39e-07, and the median particle/context flow ratio is 87.25. The absolute scale is normalized-image optical flow, not physical velocity.
- The strongest raw event-relative signal is higher near-pre-event radial apparent motion: `abs_radial_flow_mean` for near-pre-event versus any non-near row has AUC 0.756, AP 0.596, median positive-negative 3.38e-07, and MW p=1.54e-05.
- Near-pre-event versus post/control is similar: `abs_radial_flow_mean` AUC 0.760, AP 0.778, median positive-negative 3.37e-07, and MW p=7.05e-05.
- Source residualization weakens but does not erase the lead row: source-residual `abs_radial_flow_mean` near-pre-event versus any non-near row has AUC 0.632, AP 0.399, median positive-negative 1.09e-07, and MW p=0.0253.

Interpretation: near-pre-event particle crops show stronger apparent radial image motion and curl-like flow than non-near rows, consistent with a particle-local instability/readiness proxy. The source-residual drop means this should be treated as a candidate transport descriptor, not a source-invariant mechanism claim. It is useful for ranking manual-QC candidates and for fusing with phase-kinetic/front evidence.

Guardrail: apparent optical-flow transport is computed inside history-derived automatic masks on normalized ROI crops. It is an image-motion proxy only; it is not calibrated particle velocity, phase-boundary velocity, material flux, diffusion, or causal event evidence.

## 2026-05-22 Source-Balanced Pre-Event Transport/Kinetic Fusion Audit

Added `scripts/tier4_source_balanced_pre_event_transport_kinetic_fusion_audit.py` and ran it on Isambard to join the manual-QC decision ledger with the masked rollout benchmark and optical-flow transport audit. The audit builds component scores for transport, rollout difficulty, phase kinetics, front evidence, and QC evidence, then tests fused scores with source-residual variants, source-stratified permutation, and leave-source logistic comparisons.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_transport_kinetic_fusion_audit`
- `derived_local/source_balanced_pre_event_transport_kinetic_fusion_audit`
- `source_balanced_pre_event_transport_kinetic_fusion_per_roi.csv`
- `source_balanced_pre_event_transport_kinetic_fusion_event_tests.csv`
- `source_balanced_pre_event_transport_kinetic_fusion_leave_source_models.csv`
- `source_balanced_pre_event_transport_kinetic_fusion_ranked_candidates.csv`
- `source_balanced_pre_event_transport_kinetic_fusion_summary.json`

Results:

- Joined 128 ROI rows across 64 cycles and 14 sources. Feature sets cover 8 transport, 5 rollout, 14 kinetic/front, 22 transport+kinetic/front, and 27 all-core nonvisual features.
- The best near-pre-event versus any non-near row is `manual_qc_augmented_fusion_score`: AUC 0.785, AP 0.616, MW p=1.49e-06, and source-stratified permutation p=0.00399.
- The same fused score remains strong for near-pre-event versus post/control: AUC 0.798, AP 0.755, MW p=5.02e-06, source-stratified permutation p=0.0259.
- A nonvisual `transport_kinetic_front_fusion_score` also passes source-stratified permutation for near versus any non-near (AUC 0.761, AP 0.615, p=1.03e-05, source-stratified p=0.00399) and near versus post/control (AUC 0.752, AP 0.724, source-stratified p=0.00599).
- Leave-source models are encouraging but limited by class-balanced fold coverage; the best transport-only near-vs-post/control row scores AUC 0.889/AP 0.959 but evaluates only 18 rows from 2 sources. Treat this as exploratory, not a deployable classifier.
- The top fusion review candidate is `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723`, a near-pre-event row already queued as `review_front_and_kinetics_first`, followed by another near-pre row from the same cycle/source and several guardrail/control candidates.

Interpretation: adding optical-flow transport and masked-rollout difficulty to existing phase/front/kinetic evidence improves the near-pre-event ranking signal under source-stratified permutation. The strongest defensible output is now a prioritized manual-review/fusion ledger, not a standalone automatic degradation detector. The source-balanced fusion result supports the idea that near-event particle regions show coupled apparent transport, prediction difficulty, and phase/front evidence before degradation events.

Guardrail: fusion scores join automatic descriptors from history-derived particle ROI crops. They are ranking and hypothesis-generation tools only; they are not manual QC labels, calibrated phase-boundary velocities, material fluxes, diffusion coefficients, or causal degradation proof.

## 2026-05-22 Source-Balanced Transport Mechanism Dossier

Added `scripts/tier4_source_balanced_transport_mechanism_dossier.py` and ran it on Isambard to join the source-balanced optical-flow transport audit, manual-QC decision packet, observable-forecast descriptors, and front/kinetic concordance audit into one candidate-level mechanism review dossier.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_transport_mechanism_dossier`
- `derived_local/source_balanced_transport_mechanism_dossier`
- `source_balanced_transport_mechanism_dossier.csv`
- `source_balanced_transport_mechanism_top40.csv`
- `source_balanced_transport_mechanism_immediate_review.csv`
- `source_balanced_transport_mechanism_source_summary.csv`
- `source_balanced_transport_mechanism_tier_summary.csv`
- `source_balanced_transport_mechanism_summary.json`

Results:

- Ranked 128 ROI rows across 64 cycles and 14 sources; 6 rows land in the immediate-review queue and 0 rows pass the automatic diffusion-claim gate.
- Review tiers are 103 routine/context-control rows, 19 guarded transport-hypothesis rows, 5 priority transport-mechanism rows, and 1 priority manual transport/front row.
- The top mechanism candidate is `source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723`, a near-pre-event row at cycle 152 with `transport_mechanism_score=0.892`, future8/future16 positives, visual assets present, and diffusion claims blocked.
- Top-40 event bins are enriched for near-pre-event rows: 22 near-pre, 7 post-event, 6 far-pre, 3 no-near-event controls, and 2 mid-pre.
- Source `17_c2_x10_HighHighCOV_150723` has the strongest concentration of priority rows: max mechanism score 0.892, 8 near-pre rows, and 4 priority rows. Source `9_c2_x10_010723` contributes two additional priority near-pre rows.

Interpretation: the dossier converts several separate AI/physics proxies into a practical manual-review ledger. The strongest candidates show coupled apparent transport, source-residual transport, front/kinetic concordance, future-observable tail signal, and QC evidence before abrupt optical drops, but the output is still a ranking/hypothesis tool rather than a mechanistic claim.

Guardrail: automatic optical flow, front proxies, and fusion scores are not calibrated phase-boundary velocities, material fluxes, diffusion coefficients, or causal degradation proof. Diffusion and transport-mechanism claims remain blocked until strict gates and manual validation pass.

## 2026-05-22 Source-Balanced Transport Mechanism Falsification Audit

Added `scripts/tier4_source_balanced_transport_mechanism_falsification_audit.py` and ran it on Isambard to pressure-test the transport mechanism dossier with source-stratified permutation, same-source matched near-pre versus control/far/post comparisons, source-median contrasts, and top-k enrichment checks.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_transport_mechanism_falsification_audit`
- `derived_local/source_balanced_transport_mechanism_falsification_audit`
- `source_balanced_transport_mechanism_falsification_event_tests.csv`
- `source_balanced_transport_mechanism_falsification_matched_pairs.csv`
- `source_balanced_transport_mechanism_falsification_pair_tests.csv`
- `source_balanced_transport_mechanism_falsification_source_contrasts.csv`
- `source_balanced_transport_mechanism_falsification_source_tests.csv`
- `source_balanced_transport_mechanism_falsification_topk.csv`
- `source_balanced_transport_mechanism_falsification_summary.json`

Results:

- The mechanism score remains event-local under source-stratified permutation: near-pre versus any non-near row has AUC 0.783, AP 0.654, median near-minus-control score difference 0.217, and source-stratified p=0.00050.
- Near-pre versus post/control remains positive: AUC 0.765, AP 0.746, median score difference 0.186, and source-stratified p=0.00200.
- Same-source nearest-control matching supports the event-local ranking: near-pre rows beat the nearest same-source non-near row in 25 of 32 pairs, median score delta 0.208, sign-flip p=0.00050.
- Same-source post/control matching is smaller but stronger by sign fraction: 11 of 12 pairs have positive near-control deltas, median delta 0.204, sign-flip p=0.00150.
- Source-median contrasts are directionally consistent but underpowered: five sources have both near and non-near rows, all five have positive near-minus-non mechanism-score medians, median delta 0.216, sign-flip p=0.057.
- Top-k enrichment still shows source concentration: top-5 rows are 80% near-pre but 80% from source `17_c2_x10_HighHighCOV_150723`; top-20 rows span 9 sources and are 75% near-pre. Zero top-k rows pass the diffusion-claim gate.

Interpretation: this falsification audit strengthens the practical use of the mechanism dossier as an event-local manual-review ranker. The same-source paired comparisons argue against a purely cross-source artifact, while the top-k and five-source source-median limits keep the conclusion guarded. This supports a particle-local readiness/instability hypothesis, not a calibrated mechanism claim.

Guardrail: the audit tests automatic optical/AI descriptors. It does not establish calibrated transport, phase-boundary velocity, diffusion, or causality; those claims remain blocked pending strict gates and manual validation.

## 2026-05-22 Source-Balanced Expansion Transport/Front Audit

Added `scripts/tier4_source_balanced_expansion_transport_front_audit.py` and ran it on Isambard over the broader `source_balanced_roi_sequences` cohort. This extends the pre-event transport/front analysis from the review-focused pre-event dossier to the 96 expanded source-balanced ROI sequences sampled across more cycles and sources.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_expansion_transport_front_audit`
- `derived_local/source_balanced_expansion_transport_front_audit`
- `source_balanced_expansion_transport_front_features.csv`
- `source_balanced_expansion_transport_front_tests.csv`
- `source_balanced_expansion_transport_front_source_summary.csv`
- `source_balanced_expansion_transport_front_top40.csv`
- `source_balanced_expansion_transport_front_summary.json`

Results:

- Processed 96/96 expanded ROI rows across 48 cycles and 14 sources with Farneback optical flow; future8/future16 positives are 28/48 rows.
- The strongest future8 raw transport rows are radial/flow magnitude features: `abs_radial_flow_mean` AUC 0.727/AP 0.645, `abs_radial_flow_q90` AUC 0.722/AP 0.636. Their source-stratified permutation p-values are only 0.078 and 0.062, so they are suggestive but not strict source-invariant evidence.
- The strongest future16 row is raw `gradient_aligned_flow_slope` with AUC 0.666/AP 0.683, but source-stratified p=0.138. A source-residual `divergence_q90` row is weaker in AUC (0.642/AP 0.723) but passes source-stratified p=0.034.
- The composite `expansion_transport_front_score` is not a strong future16 classifier in raw form (AUC 0.549), although its source-residual permutation p=0.012 flags within-source ranking structure that needs follow-up rather than direct interpretation.
- Top expanded candidates include future-positive rows from sources `12_c2_x10_070723`, `17_c2_x10_HighHighCOV_150723`, and `9_c2_x10_010723`, but also high-scoring future-negative control rows such as `source_balanced_cycle6_rank25_obj1_2_c2_x14_200623`.

Interpretation: apparent transport/front descriptors partially generalize to the broader source-balanced expansion cohort, especially for future8 raw radial-flow features, but the evidence is weaker and more source-sensitive than in the pre-event review dossier. This is useful as a falsification/generalization check: the transport signal is not simply universal across all expanded crops, and high-score future-negative controls should be used during manual QC and model refinement.

Guardrail: expanded-cohort optical flow and front-radius descriptors are automatic fixed-crop image-motion proxies. They are not manual particle labels, calibrated transport velocities, phase-boundary velocities, diffusion coefficients, or causal degradation proof.

## 2026-05-22 Diffusion Unblock Sensitivity Audit

Added `scripts/tier4_diffusion_unblock_sensitivity_audit.py` and ran it on Isambard to quantify which strict-readiness blockers dominate the current apparent-diffusion candidate ledger. This is a what-if sensitivity audit over existing blocker labels; it does not relax production gates or create new diffusion claims.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/diffusion_unblock_sensitivity_audit`
- `derived_local/diffusion_unblock_sensitivity_audit`
- `diffusion_unblock_blocker_table.csv`
- `diffusion_unblock_review_queue.csv`
- `diffusion_unblock_scenario_table.csv`
- `diffusion_unblock_sensitivity_summary.json`

Results:

- The audit covers 60 diffusion-candidate rows and applies three global hard blockers to every row: HDF5 spatial calibration metadata present, control-balanced diffusion sanity candidates, and event/control diffusion separability.
- The dominant blockers are automatic/publication diffusion candidates, control-balanced diffusion sanity candidates, event/control diffusion separability, and HDF5 spatial calibration metadata present at 60/60 candidate rows each; radius2 linear-fit quality blocks 53 rows and q70 positive confidence interval blocks 51 rows.
- Manual QC accepted labels blocks 30 rows, so manual labeling and spatial calibration alone are not sufficient to unblock calibrated diffusion claims.
- Scenario analysis gives 0 eligible candidates under the current guardrails, 0 after only confirming spatial calibration plus accepting manual QC, 0 after also rechecking the publication gate, and 0 eligible with 4 one-blocker-away rows after clearing the external/global blockers.
- Eligibility appears only when external/global blockers plus timing and q70 confidence gates are relaxed: 5 eligible and 17 one-blocker-away rows. The all-blockers-removed sanity upper bound is 46 eligible rows, which is not an allowed claim path.
- The nearest current row is `cycle78_rank22_obj2` from source `9_c2_x10_010723`, with five blockers remaining: automatic/publication diffusion candidates, control-balanced diffusion sanity candidates, event/control diffusion separability, HDF5 spatial calibration metadata present, and q70 positive confidence interval.

Interpretation: calibrated diffusion remains blocked for multiple independent reasons, not just missing manual QC or spatial calibration. The actionable path is to validate raw spatial/timing calibration, perform manual front/QC labeling, and then rerun the q70/radius2/control-balanced gates before making any physical diffusion statement.

Guardrail: this sensitivity audit removes blockers only in explicit what-if scenarios. It does not change the diffusion readiness status, accept manual labels, relax gates in production, or create calibrated diffusion coefficients.

## 2026-05-22 Current Claim Readiness Matrix

Added `scripts/tier4_current_claim_readiness_matrix.py` and ran it on Isambard to consolidate the current source-balanced transport/fusion, mechanism-dossier, falsification, expanded-cohort, rollout, residualized-warning, manual-QC, and diffusion-readiness audits into explicit claim-level wording decisions.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/current_claim_readiness_matrix`
- `derived_local/current_claim_readiness_matrix`
- `current_claim_readiness_matrix.csv`
- `current_claim_readiness_status_counts.csv`
- `current_claim_readiness_summary.json`

Results:

- The matrix audits 7 current claims. Three are supported or operational as guarded internal tools: event-local particle-readiness ranking, transport-mechanism candidate dossier handoff, and next-frame rollout physics descriptors.
- Three claims are blocked or not supported: broad future-drop generalization, calibrated diffusion coefficients, and deployable future warning detection. Phase-boundary/front tracking remains a partial optical proxy only.
- Positive evidence is concentrated in source-aware event-local review ranking: fusion near-vs-any AUC 0.785 with source-stratified p=0.00399; falsification near-vs-any AUC 0.783 with source-stratified p=0.00050; same-source matched-pair median score delta 0.208 with sign-flip p=0.00050.
- Negative evidence remains decisive for stronger claims: diffusion status is `not_ready_for_calibrated_diffusion_claim` with 0 publication-ready candidates; future8 video physics is `not_supported_after_controls`; fused video/echem is `not_incremental_over_echem_context`; expanded future8 transport/front source-stratified p=0.078.

Interpretation: the project currently supports source-aware particle-region manual-review ranking and optical-proxy hypothesis generation. It does not yet support calibrated diffusion, causal mechanism, broad source-robust forecasting, or deployable warning claims.

Guardrail: the matrix is a wording/readiness ledger over existing automatic audits. It adds no manual labels, calibrated spatial metadata, causal validation, or prospective deployment validation.

## 2026-05-22 Source-Heldout Event Rank Transfer Audit

Added `scripts/tier4_source_heldout_event_rank_transfer_audit.py` and ran it on Isambard to test whether particle-region event ranking transfers when each acquisition source is held out. The audit trains only feature orientation and empirical scaling on all other sources, excludes precomputed source-residual features from the learned transfer score, and scores the held-out source without using its labels.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_heldout_event_rank_transfer_audit`
- `derived_local/source_heldout_event_rank_transfer_audit`
- `source_heldout_event_rank_transfer_predictions.csv`
- `source_heldout_event_rank_transfer_folds.csv`
- `source_heldout_event_rank_transfer_feature_weights.csv`
- `source_heldout_event_rank_transfer_score_tests.csv`
- `source_heldout_event_rank_transfer_topk.csv`
- `source_heldout_event_rank_transfer_summary.json`

Results:

- The audit starts from the 128-row source-balanced transport-mechanism dossier with 14 sources and 60 candidate raw automatic descriptors. Five sources have both near-pre and non-near rows and therefore support held-out evaluation.
- The transfer-learned oriented feature score reaches near-pre versus any non-near AUC 0.832 and AP 0.924 over 48 held-out prediction rows, with median near-minus-control score difference 0.237. The source-level sign-flip p-value is 0.125 across the five eligible held-out sources.
- Fixed `transport_mechanism_score` remains slightly higher by pooled AUC at 0.838/AP 0.938, but its top-5 is 80% dominated by source `17_c2_x10_HighHighCOV_150723`.
- The transfer-learned score improves top-k source diversity while preserving high near-pre enrichment: top-5 is 100% near-pre across 3 sources with max source fraction 0.40; top-10 is 100% near-pre across 4 sources; top-20 is 95% near-pre across 4 sources.
- The eligible-source count is still small, so this strengthens the review-ranker claim but does not yet establish a deployable source-transfer detector.

Interpretation: a simple held-out-source orientation/scaling procedure can preserve most of the event-local ranking signal while reducing top-k source concentration. This is useful for building a fair manual-review queue and for separating robust particle-region optical descriptors from source-specific acquisition artifacts.

Guardrail: this is an automatic particle-region ranking diagnostic. It adds no manual labels, calibrated velocities, diffusion coefficients, causal mechanism proof, or prospective deployment validation.

## 2026-05-22 All-Cycle Dataset Coverage Atlas

Added `scripts/tier4_all_cycle_dataset_coverage_atlas.py` and ran it on Isambard to map the complete cycle-level echem/optical ledger against the accumulated ROI/video-sequence cohorts. This directly addresses the project-level coverage question: how much of the Alek_Jiho NMC degradation dataset is represented by the current AI/physics analyses, and which source/cycle windows should be prioritized next.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/all_cycle_dataset_coverage_atlas`
- `derived_local/all_cycle_dataset_coverage_atlas`
- `all_cycle_coverage_table.csv`
- `all_cycle_source_coverage_summary.csv`
- `all_cycle_roi_cohort_coverage_summary.csv`
- `all_cycle_output_coverage_summary.csv`
- `all_cycle_coverage_gap_priority.csv`
- `all_cycle_h5_inventory_enriched.csv`
- `all_cycle_dataset_coverage_summary.json`

Results:

- The atlas covers 89 cycle-level rows across 16 sources and joins them to 33 HDF5 inventory rows.
- It checks 17 ROI/video cohorts and 7 cycle-level outputs. Across the accumulated cohorts, 88/89 cycle rows have at least one tracked ROI/video sequence, and 88/89 have primary ROI-sequence coverage.
- No future16-positive cycle is completely missing tracked ROI-sequence coverage.
- Source `16_c2_x10_HighHighCOV_130723` is the main residual source-level gap: 9/10 cycles have ROI coverage and the uncovered cycle is cycle 137, which is not future8/future16 positive but is an echem/context outlier candidate.
- The highest coverage-priority cycles are already represented but deserve denser/manual-QC review because they combine future-event labels, echem/hazard/context signals, and ROI coverage: cycle 150 from `17_c2_x10_HighHighCOV_150723`, cycle 110 from `12_c2_x10_070723`, and cycles 80/78/82 from `9_c2_x10_010723`.
- Cohort coverage is now explicit: source-balanced pre-event ROI sequences cover 128 rows across 64 cycles and 14 sources; source-balanced expansion ROI sequences cover 96 rows across 48 cycles and 14 sources; balanced-future ROI sequences cover 72 rows across 24 cycles and 9 sources.

Interpretation: the project has moved beyond a small event/reference-only core cohort. The accumulated source-balanced and expansion cohorts now cover nearly the full cycle-level ledger, so the next useful expansion is not blind all-cycle extraction; it is targeted densification/manual QC around high-priority cycles and the single uncovered source/cycle gap.

Guardrail: this atlas audits dataset and ROI-analysis coverage only. It does not extract new ROIs, validate particle identity, train deployment models, or make calibrated diffusion/phase-boundary claims.

## 2026-05-22 Targeted Densification and Manual-QC Plan

Added `scripts/tier4_targeted_densification_qc_plan.py` and ran it on Isambard to convert the all-cycle coverage atlas plus candidate-level transfer, mechanism, manual-QC, and diffusion-blocker ledgers into a concrete action queue for the next project phase.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/targeted_densification_qc_plan`
- `derived_local/targeted_densification_qc_plan`
- `targeted_densification_cycle_plan.csv`
- `targeted_densification_source_plan.csv`
- `targeted_manual_qc_roi_queue.csv`
- `targeted_densification_action_counts.csv`
- `targeted_densification_roi_origin_counts.csv`
- `targeted_densification_qc_summary.json`

Results:

- The plan ranks 89 cycle rows across 16 sources and produces 276 ROI-level action rows.
- Recommended cycle actions are 64 cycles for manual QC of existing visual assets, 13 cycles for diffusion-blocker follow-up, 9 cycles for densifying existing ROI candidates, 2 cycles for lower-priority existing-ROI review, and 1 cycle for extracting ROI candidates for the uncovered cycle.
- The highest-priority sources are `9_c2_x10_010723`, `17_c2_x10_HighHighCOV_150723`, and `12_c2_x10_070723`; they combine future-event labels, source-heldout transfer support, mechanism/front evidence, and existing visual assets.
- The highest-priority cycle is cycle 78 from `9_c2_x10_010723`, driven by the nearest diffusion-unblock candidate `cycle78_rank22_obj2` plus source-balanced visual/manual-QC candidates from the same cycle.
- The top ROI action is `cycle78_rank22_obj2` from the diffusion-unblock queue, followed by visual-asset-backed source-balanced candidates `source_balanced_cycle78_rank61_obj2_9_c2_x10_010723` and related cycle-78 transport/mechanism rows.
- The single uncovered cycle remains cycle 137 from `16_c2_x10_HighHighCOV_130723`; it is an extraction target but lower priority than the existing high-evidence manual-QC/diffusion rows because it is not future8/future16 positive.

Interpretation: the next useful work is no longer blind all-cycle extraction. The plan makes the immediate handoff explicit: manually review the cycle-78/cycle-152/cycle-154/cycle-82/cycle-80 assets first, use cycle 78 to test whether diffusion blockers can be resolved by human front/QC labels, and only then densify low-ROI cycles or extract the one uncovered cycle.

Guardrail: this plan prioritizes ROI densification and manual-QC review using existing automatic ledgers. It does not extract new ROIs, accept labels, relax gates, or create calibrated diffusion/phase-boundary claims.

## 2026-05-22 Pre-Event Temporal Dose-Response Audit

Added `scripts/tier4_pre_event_temporal_dose_response_audit.py` and ran it on Isambard to test whether automatic particle-region transport/front/kinetic descriptors intensify monotonically as the next degradation event approaches. This is a stricter temporal physics check than near-pre versus control ranking because it asks whether descriptor magnitude follows event proximity across near, mid, and far pre-event windows.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/pre_event_temporal_dose_response_audit`
- `derived_local/pre_event_temporal_dose_response_audit`
- `pre_event_temporal_dose_response_feature_tests.csv`
- `pre_event_temporal_dose_response_bin_summary.csv`
- `pre_event_temporal_dose_response_source_summary.csv`
- `pre_event_temporal_dose_response_summary.json`

Results:

- The audit uses 76 pre-event particle ROI rows from 38 cycles and 10 sources, split into distance bins: 1-2 cycles (8 rows), 3-4 cycles (8), 5-8 cycles (16), 9-16 cycles (22), and 17-32 cycles (22).
- Raw pooled correlations with event proximity are positive for review/ranking scores: `qc_review_score` rho 0.417, `transport_mechanism_score` rho 0.404, and `front_kinetic_score` rho 0.354, all with permutation p <= 0.0045.
- After source-centering, the temporal dose-response mostly collapses: `qc_review_score` source-centered rho 0.110 (p=0.345), `transport_mechanism_score` rho 0.011 (p=0.924), and `front_kinetic_score` rho -0.007 (p=0.948).
- Source-level slopes are also weak for the main ranking scores: positive source-slope fraction is 5/9 for `transport_mechanism_score` and `qc_review_score`, and 4/9 for `front_kinetic_score`.
- The most directional source-slope feature is `kinetic_evidence_score`, with positive slopes in 7/9 eligible sources and sign-flip p=0.0625, but its source-centered rank correlation remains weak.

Interpretation: apparent temporal countdown structure exists in pooled pre-event descriptors, but much of it is source/acquisition structured rather than a robust within-source monotonic ramp. The result supports using these descriptors for event-local ranking and review prioritization, but it argues against claiming a calibrated or generally monotonic phase-boundary/diffusion precursor trajectory from the automatic scores alone.

Guardrail: temporal dose-response tests use automatic particle-region optical/front descriptors and event-relative labels. They support precursor-ranking hypotheses only, not causal mechanisms, calibrated phase-boundary velocities, or diffusion coefficients.

## 2026-05-22 Particle Mask History/Fallback Audit

Added `scripts/tier4_particle_mask_history_fallback_audit.py` and ran it on Isambard over the 128-row source-balanced transport-mechanism dossier. This directly tests the drift-correction/blurriness concern by building a history-based particle mask from early frames, comparing each framewise contrast mask to that prior, and flagging frames where low history IoU, centroid displacement, or low sharpness would justify falling back to the prior particle support.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_mask_history_fallback_audit`
- `derived_local/particle_mask_history_fallback_audit`
- `particle_mask_history_fallback_metrics.csv`
- `particle_mask_history_fallback_event_tests.csv`
- `particle_mask_history_fallback_source_summary.csv`
- `particle_mask_history_fallback_high_risk_rois.csv`
- `particle_mask_history_fallback_failures.csv`
- `particle_mask_history_fallback_summary.json`

Results:

- Processed 128/128 source-balanced particle ROI sequences with 0 failures across 14 sources.
- The history-prior mask is broadly stable but not trivial: median fallback frame fraction is 0.292, q90 fallback fraction is 0.575, median history IoU is 0.865, and median q90 centroid jitter is 0.874 px.
- Near-pre rows have a higher pooled fallback fraction than non-near rows by 0.167, but this is not source-robust. For near-vs-any/non-near, fallback fraction has AUC 0.677 but source-stratified p=0.083, while source-residual fallback fraction has AUC 0.520 and p=0.712.
- The mechanism/review scores remain stronger than mask-instability metrics under source-stratified testing: `qc_review_score` AUC 0.831/source-stratified p=0.0015 and `transport_mechanism_score` AUC 0.783/source-stratified p=0.0005 for near/future8 labels.
- High-fallback cases are now explicitly queued, including `source_balanced_cycle72_rank58_obj1_8_c2_x10_300623`, `source_balanced_cycle52_rank49_obj1_6_c2_x10_270623_2`, and `source_balanced_cycle46_rank46_obj2_6_c2_x10_270623_2`.

Interpretation: history-based particle masks are useful as a robustness guardrail for blur/drift-correction artifacts, especially in the high-fallback tail. However, mask fallback/blur metrics do not explain the source-aware event-ranking signal, so the current transport-mechanism score is not simply a mask-instability artifact.

Guardrail: these are automatic cropped-particle mask robustness diagnostics. Fallback flags do not provide manual segmentation labels, validated particle boundaries, phase-boundary velocities, or diffusion coefficients.

## 2026-05-22 Targeted Diffusion Blocker Diagnostic

Added `scripts/tier4_targeted_diffusion_blocker_diagnostic.py` and ran it on Isambard to recheck the diffusion-readiness candidates against threshold-sweep front metrics plus same-source and same-cycle reference percentiles. This is a targeted follow-up to the diffusion readiness/unblock audits: it asks which candidates deserve manual front remeasurement, not whether the project can yet claim calibrated diffusion.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/targeted_diffusion_blocker_diagnostic`
- `derived_local/targeted_diffusion_blocker_diagnostic`
- `targeted_diffusion_candidate_diagnostic.csv`
- `targeted_diffusion_threshold_variants.csv`
- `targeted_diffusion_remeasurement_queue.csv`
- `targeted_diffusion_diagnostic_action_counts.csv`
- `targeted_diffusion_blocker_diagnostic_summary.json`

Results:

- The diagnostic covers 20 targeted diffusion/front candidates and 140 threshold-variant rows.
- Action counts are 16 `fit_quality_blocked_retrack_front`, 2 `guarded_review_only`, 1 `remeasure_q70_ci_and_manual_front_qc`, and 1 `source_context_common_not_candidate_specific`.
- The one nearest diffusion follow-up remains `cycle78_rank22_obj2` from `9_c2_x10_010723`, cycle 78. It is automatic-physics-consistent with 6 gates passing, HDF5 timing stable, positive-D fraction 1.0, q70 apparent D 3.457e-06 um2/s, q70 radius2 fit R2 0.424, max positive-fit R2 0.513, and same-source median-D percentile 0.75.
- Its remaining blocker is not broad candidate discovery; it is specifically q70 positive-CI/manual-front-QC/publication-gate follow-up.
- Most other candidate rows do not provide a diffusion shortcut. They are dominated by weak radius2 fit quality and should be treated as optical-front proxy/retracking tasks, not calibrated transport estimates.

Interpretation: this narrows the diffusion path substantially. If manual effort is available, the first concrete remeasurement target is `cycle78_rank22_obj2`; the rest of the current ledger mainly needs front-boundary retracking or remains guarded proxy evidence. The result does not unblock calibrated diffusion, but it prevents wasting effort on the wrong candidates.

Guardrail: the diagnostic ranks blocker follow-up using existing automatic threshold/front ledgers. It does not accept labels, relax readiness gates, or emit calibrated diffusion coefficients.

## 2026-05-22 History/Fallback Masked Rollout Ablation

Added `scripts/tier4_history_fallback_masked_rollout_ablation.py` and ran it on Isambard to connect the history-based particle mask fallback directly to next-frame prediction and rollout. The audit compares adaptive frame masks, fixed history-prior masks, and hybrid fallback masks, then trains a compact latent-linear predictor using only history-mask particle pixels.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/history_fallback_masked_rollout_ablation`
- `derived_local/history_fallback_masked_rollout_ablation`
- `history_fallback_masked_rollout_ablation_per_roi.csv`
- `history_fallback_masked_rollout_ablation_event_tests.csv`
- `history_fallback_masked_rollout_ablation_source_summary.csv`
- `history_fallback_masked_rollout_ablation_method_summary.csv`
- `history_fallback_masked_rollout_ablation_failures.csv`
- `history_fallback_masked_rollout_ablation_summary.json`

Results:

- Processed 128/128 source-balanced pre-event particle ROI sequences with 0 failures across 64 cycles and 14 sources.
- Median fallback frame fraction is 0.286, consistent with the separate mask-history audit.
- Hybrid fallback masking slightly improves the median one-step prediction residual relative to adaptive masks: median adaptive one-step MSE is 4.126e-05 and median hybrid one-step MSE is 3.673e-05, with median hybrid-minus-adaptive delta 0.
- The particle-only latent-linear rollout improves over persistence on the median ROI: median gain is 0.158 when scored on the history mask and 0.0817 when scored on the hybrid fallback mask. The wide q10-q90 range shows this is a useful baseline but not a universally stable rollout model.
- Event associations remain partly acquisition/source structured. Raw near-pre tests show higher fallback fraction and one-step residuals, for example near-vs-far fallback fraction AUC 0.790 and near-vs-post-control hybrid one-step MSE AUC 0.701, while the most source-residual signal is weaker and should be treated as a robustness cue rather than a physics claim.
- Source-level behavior is explicit: high-fallback sources include `10_c2_x10_030723`, `17_c2_x10_HighHighCOV_150723`, `12_c2_x10_070723`, and `6_c2_x10_270623_2`; latent-linear gains are positive for several future-positive sources but negative for some low-motion/control-dominated sources.

Interpretation: history/fallback masks are not just a post-hoc QC metric; they improve the predictive mask definition for next-frame residual scoring and give a particle-only latent rollout baseline that can beat persistence on the median ROI. However, the event-linked residual structure is still not strong enough to claim a source-invariant precursor model or calibrated phase/diffusion physics.

Guardrail: this is an automatic particle-mask ablation and compact latent-linear rollout benchmark. It uses history/fallback particle support for robustness under drift-correction blur, but it does not provide manual segmentation, validated phase-boundary velocities, or calibrated diffusion coefficients.

## 2026-05-22 Rollout Front/Mode Coupling Audit

Added `scripts/tier4_rollout_front_mode_coupling_audit.py` and ran it on Isambard to connect the history/fallback particle-only rollout residuals to front, phase-kinetic, transport, and unsupervised mode descriptors. This addresses the key interpretability question left by the rollout ablation: whether prediction residuals behave like physically meaningful optical/front changes or only like mask/source artifacts.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_front_mode_coupling_audit`
- `derived_local/rollout_front_mode_coupling_audit`
- `rollout_front_mode_coupling_merged.csv`
- `rollout_front_mode_coupling_feature_tests.csv`
- `rollout_front_mode_coupling_correlations.csv`
- `rollout_front_mode_coupling_mode_summary.csv`
- `rollout_front_mode_coupling_review_queue.csv`
- `rollout_front_mode_coupling_summary.json`

Results:

- Joined 128 source-balanced ROI rows across 64 cycles, 14 sources, and 2 automatic physics modes.
- Source-residual rollout/physics correlations support a real link between particle-only prediction error and optical physics descriptors. The strongest source-residual link is `persistence_rollout_hybrid_mse` versus `observable_tail_score` with Spearman rho 0.421 and p=7.29e-07.
- Additional source-residual links connect rollout residuals to `abs_radial_flow_mean` around rho 0.306 for hybrid rollout MSE and rho 0.292 for history-mask rollout MSE, and to `transport_mechanism_score` with rho 0.301 for hybrid rollout MSE.
- Raw future8/near-pre feature tests still rank the existing multimodal review scores highest: `qc_review_score` AUC 0.831/AP 0.719, `masked_minus_bg_slope` AUC 0.816/AP 0.634, and `transport_mechanism_score` AUC 0.783/AP 0.654.
- The coupled mode summary shows mode 0 has larger median latent-linear gain (0.300) and higher one-step hybrid residual (6.50e-05), while mode 1 has lower latent gain (0.0707) and lower one-step residual (1.05e-05), with broadly similar future-event fractions. This makes the mode split more useful as a residual/physics texture state than as a direct degradation label.
- The review queue now prioritizes rows where rollout residual, latent gain, transport score, and front kinetic score jointly stand out after source centering. It includes known near-pre future-positive rows such as `source_balanced_cycle110_rank12_obj1_12_c2_x10_070723`, `source_balanced_cycle108_rank11_obj1_12_c2_x10_070723`, `source_balanced_cycle152_rank29_obj2_17_c2_x10_HighHighCOV_150723`, and `source_balanced_cycle58_rank52_obj2_7_c2_x10_290623`, plus discordant controls/far-pre rows that are useful for falsification.

Interpretation: particle-only rollout residuals are not just generic prediction errors. After source-centering, they align with observable-tail, radial-flow, and transport descriptors, which supports using rollout residual maps as a physically interpretable review signal. The coupling is still automatic and mixed with mask/phase-state variation, so it strengthens review prioritization rather than unblocking calibrated phase-boundary or diffusion claims.

Guardrail: this audit couples automatic particle-only rollout residuals to automatic front/phase/transport/mode descriptors. It nominates physically interpretable review rows and source-robust associations, but does not validate manual phase boundaries or calibrated diffusion coefficients.

## 2026-05-22 Cycle 78 Diffusion Remeasurement Audit

Added `scripts/tier4_cycle78_diffusion_remeasurement_audit.py` and ran it on Isambard for the current nearest diffusion follow-up candidate, `cycle78_rank22_obj2` from source `9_c2_x10_010723`. This directly follows the targeted diffusion blocker diagnostic by remeasuring the q70 front-radius2 slope with multiple threshold quantiles, central support masks, baseline windows, time windows, and contiguous-block bootstrap confidence intervals.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle78_diffusion_remeasurement_audit`
- `derived_local/cycle78_diffusion_remeasurement_audit`
- `cycle78_diffusion_remeasurement_per_variant.csv`
- `cycle78_diffusion_remeasurement_roi_summary.csv`
- `cycle78_diffusion_remeasurement_context_manifest.csv`
- `cycle78_diffusion_remeasurement_summary.json`
- `plots/cycle78_rank22_obj2_q70_remeasurement.png`

Results:

- The audit covers 5 same-source context ROIs and 1260 threshold/mask/window variants using 200 contiguous-block bootstrap resamples per variant.
- For the target `cycle78_rank22_obj2`, the default full-window q70 estimate is D = 3.457e-06 um2/s, bootstrap p05 = 1.953e-06 and p95 = 5.380e-06 um2/s, so the automatic q70 positive-CI gate passes in this remeasurement packet.
- The target remains the strongest same-source local context row: default q70 D percentile 0.9, q70 R2 percentile 0.9, median D percentile 0.9, positive-CI fraction percentile 0.9.
- Across all 252 variants for the target, median D is 3.288e-06 um2/s, positive-D fraction is 0.738, positive-CI fraction is 0.397, and max radius2 fit R2 is 0.849.
- Neighboring cycle-78 objects do not show the same default q70 positive-CI behavior: `cycle78_rank22_obj1` has default q70 D -2.212e-06 and `cycle78_rank22_obj3` has default q70 D -4.307e-07, both with negative/overlapping bootstrap CIs.
- The later same-source context `cycle84_rank23_obj2` and previous `cycle76_rank8_obj2` also fail the default q70 positive-CI gate, despite some positive-D threshold variants.

Interpretation: this is the first targeted automatic remeasurement that clears the specific q70 positive-CI blocker for the leading cycle-78 diffusion candidate. It strengthens `cycle78_rank22_obj2` as a manual-front/QC priority and suggests the front-radius2 expansion is object-specific rather than a whole-source/cycle artifact. It still does not make a publication-ready diffusion claim because manual front identity, spatial calibration provenance, and broader control-balanced separability remain guardrails.

Guardrail: automatic threshold/mask/window remeasurement only; does not accept manual labels, validate front identity, or create calibrated diffusion coefficients.

## 2026-05-22 Post-Remeasurement Diffusion Gate Audit

Added `scripts/tier4_post_remeasurement_diffusion_gate_audit.py` and ran it on Isambard to propagate the new cycle-78 q70 positive-CI result through the existing diffusion-readiness ledgers. This audit separates a target-specific automatic gate update from a global calibrated-diffusion claim.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/post_remeasurement_diffusion_gate_audit`
- `derived_local/post_remeasurement_diffusion_gate_audit`
- `post_remeasurement_diffusion_candidate_update.csv`
- `post_remeasurement_diffusion_gate_table.csv`
- `post_remeasurement_diffusion_scenario_table.csv`
- `post_remeasurement_diffusion_gate_summary.json`

Results:

- The candidate-specific q70 blocker for `cycle78_rank22_obj2` is removed after the remeasurement packet: pre-blockers were `q70 positive CI; publication-ready gate`, and post-candidate blockers are now only `publication-ready gate`.
- The overall diffusion claim status remains `not_ready_for_calibrated_diffusion_claim`, with 0 publication-ready candidates.
- The q70 global gate is not marked globally passed; it is recorded as `target_specific_pass_global_not_rerun` because only the leading candidate was remeasured, not the full 72-ROI diffusion-readiness cohort.
- Remaining publication blockers are explicit: manual front/QC labels are not accepted, raw spatial calibration metadata is still not located in HDF5/microscope metadata, control-balanced diffusion sanity remains negative, event/control diffusion separability remains weak, the publication-ready gate has not been rerun with accepted labels, and the global per-ROI timing-stability gate remains failed.
- The next handoff is now narrower: use `cycle78_rank22_obj2` as a manual front/QC and calibration-provenance target, not as a broad automatic diffusion claim.

Interpretation: the project now has a concrete target-specific diffusion follow-up success: the q70 positive-CI blocker cleared for `cycle78_rank22_obj2`. The result upgrades the candidate from “blocked by q70 CI” to “publication-gate/manual-QC/calibration blocked,” which is meaningful progress but still below a calibrated material diffusion coefficient claim.

Guardrail: this audit propagates a target-specific automatic q70 CI update through existing readiness gates. It does not rerun global diffusion readiness, accept manual labels, verify raw calibration metadata, or create publication-ready diffusion coefficients.

## 2026-05-22 Cycle 78 Front Identity Review Packet

Added `scripts/tier4_cycle78_front_identity_review_packet.py` and ran it on Isambard as the next handoff after the cycle-78 q70 positive-CI remeasurement. The packet renders the target `cycle78_rank22_obj2` plus same-source/same-cycle context ROIs using the same q70 front definition, and writes a pending manual-review manifest with blank particle/front/diffusion decision fields.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle78_front_identity_review_packet`
- `derived_local/cycle78_front_identity_review_packet`
- `cycle78_front_identity_review_manifest.csv`
- `cycle78_front_identity_review.html`
- `cycle78_front_identity_review_summary.json`
- `panels/*_front_identity_panel.png`

Results:

- The packet contains 5 review rows: the target, two same-cycle neighboring objects, one previous same-source/same-object-rank context row, and one later same-source/same-object-rank context row.
- Only 1/5 rows has a default q70 positive-CI apparent diffusion remeasurement, and that row is the target `cycle78_rank22_obj2`.
- No row is free of automatic front-identity flags.
- The target retains positive q70 diffusion context: q70 D 3.457e-06 um2/s, positive CI true, q70 R2 0.424, and positive-CI fraction 0.397.
- The target also fails automatic front-identity cleanliness: flags are `fragmented_q70_mask;no_dominant_component`, median q70 components is 53, largest-component fraction is 0.420, median q70 area fraction is 0.352, area CV is 0.095, centroid path is 18.75 px, and net centroid displacement is only 0.513 px.
- Edge contact is 0, so the main automatic concern is not crop-boundary leakage; it is fragmented threshold-front support and lack of a dominant coherent front component.

Interpretation: the q70 positive-CI blocker is numerically cleared for the target, but the front-identity review packet shows why calibrated diffusion is still blocked. The apparent expansion is object-specific and worth manual review, yet the q70 mask is fragmented enough that a human needs to decide whether the front is physically interpretable or just a thresholded texture artifact.

Guardrail: this packet assigns no manual labels. It renders review panels and automatic front-identity metrics only; particle identity, front coherence, artifact risk, diffusion interpretability, and final accept/reject decisions remain blank.

## 2026-05-22 Cycle 78 Component Front Retracking Audit

Added `scripts/tier4_cycle78_component_front_retracking_audit.py` and ran it on Isambard to test whether connected-component front repair can resolve the fragmented q70 mask blocker for `cycle78_rank22_obj2`. The audit compares raw q70 masks, largest-component masks, central-component masks, and top-3-component masks for the target plus four same-source context ROIs.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle78_component_front_retracking_audit`
- `derived_local/cycle78_component_front_retracking_audit`
- `cycle78_component_front_retracking_per_strategy.csv`
- `cycle78_component_front_retracking_summary.json`
- `plots/*_component_retracking.png`

Results:

- The audit covers 5 context ROIs and 20 component-strategy rows.
- For the target, the raw q70 mask has radius2 slope 0.144 px2/sample, R2 0.424, centroid path 18.75 px, and median area 2349.5 px.
- The largest-component repair preserves a positive slope but does not improve fit quality: slope rises to 2.582 px2/sample, but R2 drops to 0.127 and centroid path jumps to 222.07 px.
- The central-component and top-3-component repairs also preserve positive slope but have weak fit quality: R2 0.067 and 0.071.
- Context rows show the same pattern: component choices often create large centroid paths and low-R2 slopes, so connected-component repair is not a reliable automatic replacement for manual front tracing.

Interpretation: automatic component repair does not solve the front-identity blocker. The raw fragmented mask remains the most linearly coherent q70 radius2 readout, while component-only masks behave like unstable component switching. This strengthens the conclusion that `cycle78_rank22_obj2` needs manual front tracing/acceptance before diffusion wording, rather than another automatic mask-cleaning shortcut.

Guardrail: this is an automatic connected-component retracking diagnostic. It does not validate front identity, accept manual labels, or create calibrated diffusion coefficients.

## 2026-05-22 Calibration Provenance Evidence Audit

Added `scripts/tier4_calibration_provenance_evidence_audit.py` and ran it on Isambard to make the spatial-calibration blocker explicit across raw HDF5 metadata, project PPTX/XLSX/CSV/text files, and raw movie dimensions. This is a provenance audit only; it does not change any front-tracking values.

Outputs:

- `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/calibration_provenance_evidence_audit`
- `derived_local/calibration_provenance_evidence_audit`
- `calibration_provenance_file_inventory.csv`
- `calibration_provenance_evidence_statements.csv`
- `calibration_provenance_summary.json`

Results:

- The audit inventoried 61 files: 33 raw HDF5 files, 3 presentations, 2 spreadsheets, 12 CSV files, 2 project observation notes, and 9 other text notes.
- All raw movies share spatial dimensions 1200 x 1920 px.
- No raw HDF5 explicit spatial-scale statement was found: 0 raw HDF5 spatial-scale statements and 0 raw near-96 nm/px statements.
- No contradictory spatial-scale statement was found.
- Independent slide evidence supports the provisional scale: `Degradation Paper Outline.pptx` slide 3 states 96 nm pixel size and 180 x 120 um FoV, while FoV divided by the raw movie dimensions gives 93.75 nm/px horizontally and 100 nm/px vertically.
- The highest independent evidence class remains `slide_text`, not raw microscope metadata. The provenance status is `slide_or_project_text_supported_but_raw_metadata_blocked`.

Interpretation: the 96 nm/px scale is plausible and internally compatible with the slide FoV and raw movie dimensions, but it is not yet proven from raw microscope metadata. Calibrated diffusion constants remain blocked by spatial provenance, manual front identity/QC, timing stability, and control-balanced diffusion sanity. Current diffusion-like outputs should continue to be described as apparent optical-front proxies.

Guardrail: this audit strengthens the documentation of the calibration blocker; it does not clear the blocker or create publication-ready diffusion coefficients.
