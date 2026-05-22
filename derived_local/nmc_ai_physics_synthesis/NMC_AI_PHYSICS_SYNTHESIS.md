# NMC AI Physics Synthesis

## Scope

This report consolidates the Alek_Jiho NMC charge/discharge photometry analyses into one auditable view. It is generated from derived outputs on Isambard and should be treated as a synthesis of computational evidence, not as a manual curation substitute.

## Current Evidence Base

- Multi-cycle ROI/echem rows: 52
- Distinct cycles in coupled ROI/echem table: 11
- Event-reference cycles: 4
- Calibrated front-QC ROI rows: 12
- Manual-QC pending front ROIs: 12
- ROI/front QC package candidates: 24
- Control-balanced front QC candidates: 40
- Residual physics mode clusters: 4
- Manual-QC label workbook candidates: 47
- Manual-QC gated accepted fronts: 0
- Prefix feature-importance audit features: 54
- Spatiotemporal degradation graph nodes/edges: 52 / 510
- Phase-kinetics ROI rows/features: 52 / 40
- Calibration metadata HDF5/camera-timing files: 33 / 32
- Calibration claim-risk families/source tables: 11 / 11
- Particle trace cycle rows/drop cycles: 89 / 4
- Particle precursor event/control anchors: 4 / 24
- ROI trace-fusion rows/predictors: 52 / 100
- ROI trace-fusion cycle-null points: 11
- Precursor-informed review candidates: 47
- Precursor visual-bundle candidates/assets: 12 / 12
- Within-cycle echem shape cycles/features: 81 / 48
- Echem-shape-conditioned ROI/front rows/shape PCs: 52 / 6
- Physics-consistency matrix ROI/cycles: 52 / 11
- Cycle state-space rows/clusters: 89 / 4
- Cycle hazard warning evaluated cycles/events: 62 / 4
- Cycle-state ROI bridge rows/cycles: 52 / 11
- Cycle-state mode-frequency bridge cycles/modes: 11 / 4
- Particle-mask stability ROI/frame rows: 52 / 4992
- Masked ROI rollout frame rows: 4992
- Masked rollout cycle-warning ROI cycles/features: 11 / 105
- Masked residual transition ROI/method rows: 156
- Masked residual state-transfer anchor/full cycles: 11 / 89
- Transfer-ranked reconstructed cycles/ROI rows: 12 / 48
- Transfer-ranked masked rollout ROI/frame rows: 48 / 4608
- Transfer-ranked front physics ROI/cycles: 48 / 12
- Transfer-ranked residual transition timing ROI/method rows: 48 / 144
- Multi-cohort future-drop model rows/features: 59 / 44
- Active-learning QC candidates/visual/immediate: 97 / 47 / 4
- Automatic QC triage surrogate candidates/likely/artifact/diffusion-guardrail: 47 / 6 / 10 / 20
- QC decision evidence ledger candidates/action tiers: 47 / {'high_priority_review': 5, 'review_artifact_or_reject_first': 4, 'review_but_diffusion_guarded': 16, 'review_for_possible_accept_first': 3, 'routine_pending_review': 19}
- Balanced future-drop ROI rows/cycles/features: 72 / 24 / 26
- Cross-cohort rollout transfer selected/transfer ROIs: 11 / 48
- Diffusion sanity selected-front/publication candidates: 12 / 0
- Control-balanced high-res front tracking/sanity candidates: 40 / 0
- Weak-label benchmark trainable positives/negatives: 3 / 4
- Control-balanced QC sensitivity robust strata: 6

## Main Findings

- Persistence is the strongest raw next-frame baseline; DMD/velocity/learned residual experiments are most useful as residual and latent descriptors.
- Prefix-only cropped ROI forecasts still rank the front-direction residual class highest: random_forest at prefix 0.500 gives AUC 0.691, but after excluding raw frame-index features the audited permutation null is not significant.
- The 75%-prefix feature-importance audit is descriptive but not independently significant: pooled OOF AUC 0.447, null empirical p=0.714; strongest ablation groups are mean_intensity_trace, frame_texture_level.
- ROI event/control optical differences survive event-reference-cycle centering, especially cumulative normalized change, first-last decorrelation, latent net displacement, high-fraction growth, and ROI mean trend.
- Frame count and protocol-block position strongly couple to ROI dynamics, so echem/protocol context must be a model covariate and a guardrail.
- After residualizing available protocol/echem covariates and event-reference fixed effects, event/control separation remains in ROI mean delta, high-fraction delta, first-last correlation, cumulative change, DMD residual, and latent displacement.
- Cycles 86 and 116 remain the strongest synchronized event-timing regimes; cycles 60 and 156 provide stronger single-particle morphology/latent-movement examples.
- Apparent front tracking currently indicates optical-front contraction/loss more than clean expanding diffusion fronts.
- Threshold sweeps show robust event/control differences in phase-fraction slope, but radius-derived diffusion proxies remain weaker and threshold-sensitive.
- Diffusion proxy sanity audit rejects calibrated-diffusion promotion for the selected high-resolution front set: 0 automatic positive candidates and 0 publication candidates; median selected-front apparent D is -3.647e-04 um2/s and only 0.083 of selected fronts are nonnegative.
- Control-balanced high-resolution front tracking expands this check to 40 ROIs ({'control': 24, 'event': 16}); it still yields 0 automatic positive diffusion candidates and event/control selected-D separation remains non-significant (top p=0.314).
- Calibration metadata audit finds camera-timing datasets in 32 of 33 scanned HDF5 files and no HDF5 pixel-size attributes; sampled timing rows can be sparse segment/cycle timing, while the 96 nm/px scale remains slide-derived pending raw microscope metadata confirmation.
- Calibration claim-risk register audits 11 front/kinetic/diffusion claim families; it classifies diffusion-like values as apparent proxies and keeps manual-QC-gated diffusion/front claims pending.
- Apparent diffusion calibration-bounds audit maps all 72 balanced ROIs to HDF5 timing; ROI elapsed/HDF5 elapsed median ratio is 1.002, q70 median apparent D at 96 nm/px is 4.322e-08 um2/s, and q70 future8 separation is non-significant (top p=0.175).
- Diffusion physics-consistency audit collapses 504 threshold rows to 72 ROI gates: 1 automatic ROI passes the internal physics gate and 0 pass publication-ready diffusion gates; median radius2 fit R2 is only 0.055.
- Cross-modal consensus ranks cycles 86, 116 as synchronized multimodal degradation candidates; the top cycle has 4 modal votes and consensus score 0.813, while the score remains an audit statistic rather than a calibrated probability.
- Echem/optical breakpoint audit tests 84 cycle-level echem/trace features around synchronized cycles [86.0, 116.0]; strongest event-centered shift is state_step_norm_delta_prev over +/-4 cycles (scaled shift -2.263, bootstrap p=0.002).
- Echem-optical regime atlas organizes 89 cycles by charge/discharge asymmetry and dQ/dV-proxy shape; top binary contrast is pos_dq_abs_peak_voltage vs multimodal_outlier_without_trace_drop (median shift 0.050, p=1.453e-04), and top continuous link is shape_dVdt_abs_p95 vs cross_modal_consensus_score (rho=0.617).
- Echem-conditioned optical predictor shows the clearest echem gain for high_cross_modal_consensus_q75 under leave_one_cycle: echem_regime_minus_acquisition changes AUC by 0.113; same-cycle synchronized candidates remain acquisition/context dominated and underpowered.
- Echem-conditioned ROI rollout/front audit joins 72 ROI rows across 24 cycles; strongest leave-cycle echem gain is transferred_masked_residual_signature echem_regime_minus_acquisition with delta Spearman 0.450 and delta R2 0.626.
- Echem-video embedding fusion tests 172 masked-video rows across 34 cycles; top fusion delta is future_any_drop_within_8cycles video_plus_echem_acquisition_minus_echem_regime with delta AUC 0.284 and delta Spearman 0.492.
- Protocol-conditioned front residuals preserve phase-slope sign consistency, but not front-magnitude or diffusion-proxy separability.
- Automatic front-QC sensitivity keeps the positive phase-front residual in 5 strata: all_front_rois, complete_threshold_sweep, q70_phase_ci_excludes_zero, q70_phase_ci_positive, review_panel_selected; review-panel diffusion proxy differences are selection-sensitive and not calibrated transport.
- Protocol-adjusted residual mode taxonomy chooses k=4; its most event-enriched mode is optical_brightening_decorrelating_rollout_hard_front_positive with event fraction 0.846 and Fisher p=0.003.
- A QC review packet prioritizes 30 ROI/front candidates, a control-balanced front package adds 24 control candidates, and the manual-QC label workbook deduplicates these into 47 pending ROI labels.
- Control-balanced QC sensitivity keeps positive phase-front residuals robust in 6 automatic strata, including the balanced selected panel; diffusion-proxy residuals remain non-significant in that balanced panel.
- Manual-QC gated front-effect tests are status `ready_for_manual_labels` with 0 accepted fronts, so no manual-QC-filtered diffusion/front claim is emitted yet.
- Spatiotemporal graph tests show strong same-cycle spatial homophily in front-positive residuals and event-enriched residual modes, but cross-cycle nearest-neighbor front/event labels do not show simple propagation and remain cohort-design sensitive.
- Optical phase-kinetics fits add transition-sharpness and Avrami-style descriptors: event-enriched residual modes have larger q70/q80 transformed-fraction deltas and faster q60/q70 logistic rates, while kinetic fit quality/rates remain strongly coupled to frame count.
- The larger four-particle cycle table shows leakage-conscious early-warning signal for future abrupt drops: any-drop within 8 cycles has mean AUC 0.883 with empirical null p=0.002; synchronized 2+ drops are also detectable but with only two positive cycles.
- Event-aligned precursor windows show lower pre-event capacity/CE and higher cross-particle delta dispersion versus matched non-event anchors; the strongest precursor window test is pre8_to_pre5 capacity_mAh with p=0.002.
- ROI trace-fusion links lagged global particle-trace state to localized front behavior at the ROI-row level: strongest focused context-residual association is trace_lag2_particle_norm_range vs phase_slope_positive_fraction_protocol_residual, rho=0.725, p=2.688e-07.
- Cycle-collapsed ROI trace-fusion null audit reduces 52 ROI rows to 11 cycle points; top surviving collapsed association is trace_lag8_frames_percentile vs mode_review_priority, rho=0.834, empirical p=0.002.
- Precursor-informed ROI review ranks 47 pending manual-QC candidates; the top candidate is cycle156_rank7_obj27 with score 5.527.
- A visual review bundle now packages 12 top precursor-informed ROI candidates; 12 have at least one copied QC/preview asset and a contact sheet for manual inspection.
- Within-cycle echem shape descriptors add raw voltage/current trajectory and dQ/dV-proxy context for 81 observed cycles; strongest ROI association is shape_V_q95 vs mode_review_priority, rho=-0.864, but direct event-cycle shape tests are weak and shape terms remain protocol/capacity guardrails.
- Echem-shape-conditioned residual audit uses 45 shape features compressed to 6 PCs; phase-slope positive-fraction residual remains the strongest event/control readout after shape conditioning (p=0.004), while diffusion residuals remain non-significant and the shape-residual classifier is poor.
- Physics-consistency claim matrix scores 52 ROI rows across front, optical-change, rollout, kinetics, precursor, echem-shape, and mode-taxonomy pillars; 2 rows are cross-modal high priority, but all 52 remain `manual_qc_required_no_physics_claim`.
- Cycle state-space transition audit builds a 4-state cycle manifold from trace plus echem-shape features; PC2 is the strongest future 8-cycle abrupt-drop separator (permutation p=0.016), the shuffled-fold classifier reaches mean AUC 0.781, and stricter temporal holdout reaches AUC 0.779 across 2 usable blocks.
- Rolling-origin cycle hazard warning audit evaluates 62 cycles for future 8-cycle abrupt drops; best AUC is 0.783 with permutation p=0.048, and 8-cycle pre-event warnings hit 0.750 of event cycles.
- Cycle-state to ROI/front bridge links state PC2 to ROI physics-consistency after collapsing repeated ROI rows to 11 cycles: top collapsed test cycle_state_pc2 vs mode_taxonomy_score, rho=0.855, permutation p=0.002.
- Cycle-state mode-frequency bridge tests whether cycle/echem state organizes ROI degradation modes across 11 cycles: best macro model cycle_state_only has MAE 0.261 versus context-only MAE 0.303; compact permutation p=0.381 keeps this as a guarded organization signal.
- Particle-mask stability audit confirms ROI-only crops can be processed with a history-aware particle support guardrail: median fallback fraction 0.000, accepted-area CV 0.042, centroid path 73.607 px; event/control mask instability is not significantly different in the current cohort.
- Masked ROI rollout audit scores held-out predictions only inside accepted particle masks; persistence remains best for 52 of 52 ROIs, while low-rank DMD particle MSE tracks cumulative optical change (top rho=0.637, p=3.909e-07).
- Cycle-collapsed masked-rollout warning audit covers 11 observed ROI cycles; strongest tests align residual jumps with same-cycle abrupt drops (top permutation p=0.014), while future-drop evaluation is underpowered with only 1 positive 8-cycle warning case.
- Masked residual state-transfer warning expands the masked-residual signature from 11 video-backed cycles to 89 cycle-state rows; the transferred score separates future 8-cycle drops (AUC 0.708, permutation p=0.004), but anchor leave-one-cycle transfer is weak (rho=-0.155, p=0.650) and cycle-state PC2 remains the stronger direct future8 baseline (AUC 0.772).
- Transfer-ranked ROI reconstruction converts that state-transfer hypothesis list back into direct video crops: 12 cycles yielded 960 reconstructed components and 48 ROI rows; masked rollout on the exported crops again picks persistence as best for 48 of 48 ROIs, while low-rank DMD particle residuals remain much larger than nonparticle context.
- Transfer-ranked front physics audit links those crops to phase/front proxies across 48 ROIs; strongest ROI-level future8 association is radius2_slope_positive_fraction with median positive-negative 0.857, AUC 0.781, and permutation p=3.999e-04; radius/diffusion-like values remain apparent optical-front proxies only.
- Transfer-ranked residual transition timing gives a stronger temporal residual/phase link than the broader event-control cohort: low_rank_dmd weighted_center_distance_to_transition_frac median distance 0.087 versus null mean 0.250, p=2.000e-04; top future8 timing target is persistence particle_to_nonparticle_mse_ratio_median, AUC 0.832.
- Cross-cohort rollout transfer audit shows the late transfer-ranked crops are a distinct video-dynamics domain: selected-cohort DMD evaluated on transfer-ranked ROIs has median particle MSE 0.020, 3.494x the transfer-internal DMD baseline (p=3.277e-09), while pooled training is close to transfer-internal (1.061x).
- Multi-cohort future-drop model combines selected and transfer-ranked ROI physics features across 59 rows / 44 features; leave-cycle random forest reaches AUC 0.886/AP 0.914 with 40-permutation p=0.024, but leave-cohort transfer is not evaluable because the selected cohort has no positive future8 labels.
- Active-learning QC prioritization merges manual-QC, precursor, weak-model, front, and timing evidence into 97 review candidates; 47 have visual assets and 4 are immediate manual-QC picks, led by cycle116_rank7_obj37. No manual labels are assigned.
- Automatic QC triage surrogate ranks the same pending-review bottleneck without assigning labels: 6 likely interpretable candidates, 10 artifact-risk candidates, and 20 diffusion-guardrail rows; top likely ROI is cycle156_rank7_obj27 and top artifact-risk ROI is cycle156_rank5_obj4.
- QC decision evidence ledger converts the 47 pending labels into explicit reviewer actions without assigning labels: {'high_priority_review': 5, 'review_artifact_or_reject_first': 4, 'review_but_diffusion_guarded': 16, 'review_for_possible_accept_first': 3, 'routine_pending_review': 19}; top possible-accept ROI is cycle156_rank7_obj27, while top artifact/reject-first ROI is cycle156_rank5_obj4.
- Balanced future-drop direct-video audit removes the transfer-ranked class imbalance by sampling 24 cycles and 72 ROI rows with equal weak future8 positives/negatives; leave-cycle logistic_l2 reaches AUC 0.716/AP 0.761, permutation p=0.049. Top positive-associated features are radius2/front-motion proxies and particle-mask rollout residual fractions, still under optical-proxy/manual-QC guardrails.
- Source-balanced ROI expansion attacks the remaining cohort-breadth bottleneck: it samples 48 cycles across 14 source movies, including 41 cycle/source pairs not already in video cohorts, and proposes 96 automatic ROI rows for follow-up sequence export/QC.
- Source-balanced pre-event sampling addresses the future-specificity gap directly: it reconstructs 128 automatic ROI proposals from 64 event-relative cycles across 14 sources, with cycle bins {'far_pre_event_17_32': 11, 'mid_pre_event_9_16': 11, 'near_pre_event_1_8': 16, 'no_near_event_control': 6, 'post_event_1_16': 20} and 13 new cycle/source pairs.
- Source-balanced pre-event sequence audit exports 128 event-relative particle crops and finds near-pre-event spatial video structure under leave-source AUC 0.759/AP 0.515, while broader any-pre transfer remains weak at AUC 0.463.
- Pre-event rollout/mask audits on those crops find top future16 ROI signals in roi_norm_mean_delta_last_minus_first AUC 0.628 and masked_minus_background_mean_slope AUC 0.624; event-relative clean-pre source-residual readout peaks at front_radius_q60_slope_px_per_norm_time AUC 0.660.
- Pre-event event-distance trajectory audit collapses duplicate ROI proposals to 38 cycle rows and tests monotonic approach-to-event physics proxies; the leading source-residual physics trend is apparent_diffusion_q70_px2_per_norm_time with rho 0.272 and source-stratified permutation p=0.064.
- Pre-event directionality audit keeps ROI-level rows and compares pre-event versus post-event clocks with 250 source-stratified clock permutations: physics-facing pre-clock feature apparent_diffusion_q70_um2_per_norm_time has rho 0.201, permutation p=0.024, while the best source-residual clean-pre readout is front_radius_q60_slope_px_per_norm_time at AUC 0.660.
- Pre-event source-invariant audit tests interpretable feature families under leave-source splits; clean-pre is led by physics_front_combo / source_mean_resid_2 at AUC 0.694, while near-vs-far is led by physics_front_combo / source_residual at AUC 0.744.
- Pre-event review packet ranks 128 automatic ROI crops and renders 24 frame strips plus a contact sheet; top candidate is source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723 from near_pre_event_1_8 with review reason near_pre_high_source_invariant_front_score.
- Pre-event matched-counterfactual audit pairs near-pre ROIs against baseline/context-nearest controls; the strongest matched physics row is near_vs_far_pre source_penalized_global masked_minus_background_mean_median with median near-minus-control 0.003 and sign-flip p=9.990e-04, while q70 apparent diffusion remains weak at p=0.395.
- Same-source pre-event ladder audit shows the sampling gap explicitly: 5 sources have near-pre rows, 3 have near+mid ladders, and 0 have near+far ladders; the strongest same-source paired row is near_vs_any_non_near_same_source mask_centroid_path_px with p=0.002, while the global within-source clock remains weak (top rho 0.109).
- Pre-event radial-kymograph audit renders explicit front tracks for 128 ROI crops; near-vs-far is led by front_radius2_slope_px2_per_norm_time at AUC 0.705, median diff 3.455, p=0.040.
- Pre-event physics-mode taxonomy clusters source-residual front/diffusion/heterogeneity features into k=2 broad states but finds no strong near-pre enrichment (best Fisher p=0.689), so continuous front/diffusion clocks remain more informative than coarse modes for this cohort.
- Source-balanced ROI sequence export converts that manifest into 96 particle-region crop tensors across 48 cycles and 14 sources with 0 export failures; the fast rollout audit finds strongest future16 ROI signal in roi_norm_mean_delta_last_minus_first at AUC 0.626, while prediction-error features are highly source-structured.
- A source-balanced mask/front sanity audit adds crop-local particle masks, centroid stability, radial front proxies, and apparent q70 radius-squared slopes across 96 ROI tensors; top future16 mask/front proxy is masked_minus_background_mean_slope at AUC 0.690/AP 0.696, but source eta2 is 0.634.
- A source-residual mask/front audit tests whether those crop-local descriptors survive source structure: best source-residual future16 proxy is front_radius_q80_slope_px_per_norm_time at AUC 0.631/AP 0.634, and best within-source-rank proxy is front_radius_q80_slope_px_per_norm_time at AUC 0.656/AP 0.677.
- A source-balanced residual dictionary learns label-free next-frame residual bases on the same 96 crop tensors; residual_dictionary leave-cycle future16 reaches AUC 0.602/AP 0.581, but leave-source future16 drops to AUC 0.375, marking source transfer as the main failure mode.
- Source-normalizing the source-balanced residual dictionary leaves a source-residual future16 residual-dynamics candidate, dictionary_recon_error_last_minus_first, at AUC 0.637/AP 0.637 with source eta2 1.484e-33; within-source-rank residual PCs are weaker at AUC 0.574.
- The grouped normalized residual-dictionary readout partially rescues held-out-source future16 transfer: raw residual dictionary AUC 0.375 improves to 0.550 after source residualization, while the single dictionary_recon_error_last_minus_first_source_residual readout reaches AUC 0.612/AP 0.613; permutation p=0.100 keeps it provisional.
- Temporal-specificity controls show the same source-residual reconstruction-error drift is temporally ordered but not cleanly precursor-specific: future16 AUC 0.637 beats a within-source shift null (p=0.002) but barely exceeds past16 AUC 0.625; raw masked-minus-background slope is more future8-specific but source structured.
- Future-specific residual controls sharpen that guardrail: after excluding past16 rows, the primary source-residual reconstruction-error feature drops to AUC 0.589, while grouped future16 prediction gains only modestly over past-event context (best delta AUC 0.065 from mask_contrast_source_residual_plus_context).
- Source-balanced degradation-mode audit clusters source-residual residual/front/contrast features into k=4 modes; the strongest enrichment is mode 2 for post_event_16 (fraction 1.000, p=0.010), but tiny outlier modes keep this as review triage rather than a stable taxonomy.
- Source-balanced residual-physics coupling links the best source-residual dictionary candidate to crop-local physics proxies: top target-aligned pair is dictionary_recon_error_last_minus_first vs masked_minus_background_mean_slope with rho 0.373, residual AUC 0.637, and physics AUC 0.628; apparent diffusion coupling remains weak.
- Source-balanced residual candidate review packet converts the residual/readout/coupling evidence into 96 pending manual-QC candidates; 8 are immediate-review, led by source_balanced_cycle108_rank6_obj2_12_c2_x10_070723 with score 0.917.
- Balanced future particle-mask stability audit covers 72 ROIs / 6912 frames; median fallback fraction is 0.000, and the strongest future8 mask-stability contrast is accepted_centroid_max_step_px with p=0.175, so the balanced future signal is not explained by a simple mask-instability split.
- Masked video embedding audit extracts particle-prior self-supervised descriptors across 172 ROI tensors; balanced future leave-cycle AUC/AP is 0.816/0.865 with label-permutation p=0.012, while selected event/control readout is weaker at AUC 0.588.
- Learned residual-CNN embeddings trained label-free for next-frame residual prediction reach future8 leave-cycle AUC 0.849 versus PCA-video 0.569 and handcrafted scalar 0.828; future16 learned_all remains weak at AUC 0.538 versus handcrafted 0.680.
- Residual dictionary embedding learns label-free next-frame residual bases over 172 ROI videos; residual-dictionary future8 AUC is 0.663 with p=0.005, and residual_dictionary_plus_handcrafted reaches AUC 0.771.
- Echem residual-dictionary fusion shows conditioning boosts residual-dictionary future8 AUC to 0.917, while acquisition/context alone reaches 1.000; treat this as context-sensitive representation evidence rather than deployable warning.
- Echem-conditioned residual-dictionary audit converts post-hoc fusion into a split-specific residual objective: conditioned residual dictionary future16 reaches leave-source AUC 0.785 versus raw residual dictionary 0.058 (delta 0.726), while leave-cycle conditioned residual+echem reaches AUC 0.834; future8 remains context dominated.
- Conditioned residual physics atlas makes that objective interpretable: top source-centered physics alignment is leave_source resdict_pc04_mean to temporal_diffusion_proxy_median_um2_per_s (front_phase_diffusion) with rho 0.815; top single residual future16 mode is resdict_pc08_slope at AUC 0.821/AP 0.956.
- Acquisition-residualized video benchmark confirms the context guardrail: future8 acquisition context reaches AUC 1.000, raw all-video reaches 0.756, and context-residualized all-video alone reaches 0.319; future16 raw handcrafted reaches AUC 0.796 but residualized all-video alone is 0.620.
- Acquisition-residualized video/echem warning audit executes the top tournament experiment: leave-cycle future16 residualized video_plus_echem reaches AUC 0.697 versus acquisition-only 0.727, but leave-source residualized AUC falls to 0.512 versus acquisition-only 0.697.
- Source-domain video/echem adaptation partially rescues leave-source future16 transfer: source-centered video_plus_echem reaches AUC 0.737 versus acquisition-only 0.697, while CORAL reaches only 0.420.
- Source-balanced transfer audit shows source-rank/weighting only modestly lifts video+echem future16 AUC to 0.614 versus raw video+echem 0.594, below acquisition context 0.704 and echem source-rank 0.642; source label composition remains the dominant guardrail.
- Source-invariant projection is more promising but still guarded: best video_plus_echem future16 is source_mean_resid_4 at AUC 0.729 versus raw 0.612 and acquisition context 0.745; video-only source-confound filtering reaches AUC 0.770.
- Source-invariant physical-family audit localizes the future16 rescue to normalized heterogeneity and particle-vs-context contrast: norm-heterogeneity source_mean_resid_4 reaches AUC 0.738, contrast source_mean_resid_4 reaches 0.703, while raw embedding alone is 0.462.
- Exact-feature source-invariant audit nominates particle_vs_context_mean_diff_positive_fraction as the strongest univariate future16 descriptor (oriented AUC 0.769, source eta2 0.390); best small transfer set trio::particle_vs_context_mean_diff_positive_fraction+particle_mean_last_minus_first+particle_gradient_diff_q90 reaches leave-source AUC 0.750.
- Exact-feature mechanism consistency audit is a useful falsification check: exact_optical_loss_score predicts future16 with AUC 0.853 but has source eta2 0.513; the primary particle-vs-context descriptor has weak radius-slope linkage after source residualization (rho -0.047).
- Signed optical-loss mechanism audit converts the pattern into interpretable axes: combined_loss_mechanism_axis future16 AUC is 0.989 but source eta2 is 0.556; leave-source all-axis AUC is 0.927, while optical-only AUC is 0.732.
- Signed-loss source robustness audit shows why this remains guarded: combined-axis raw/source-mean/within-source-rank AUCs are 0.989/0.942/0.551; optical-axis raw/source-mean/within-source-rank AUCs are 0.815/0.774/0.514.
- Source-residual echem/optical audit finds low-source-eta residual evidence: echem+optical+front direct AUC 0.809, echem+optical direct AUC 0.774, and leave-source echem+optical residual AUC 0.708.
- Invariant sparse rule discovery finds review-prioritization rules rather than a standalone predictor: best leave-source rule low(particle_std_diff_positive_fraction) covers 27/72 rows with precision 0.889, lift 1.123, and source-positive hits in 6 sources.
- Current-evidence agentic hypothesis tournament ranks the next paper-inspired experiment as Echem-conditioned video residuals are the best longer-horizon weak-label signal with score 0.598.
- Balanced future context/region guardrail shows acquisition/spatial context alone predicts weak future8 labels strongly (best AUC 0.851), while selection-design context is perfect by construction (AUC 1.000); after acquisition-context residualization, the top physics residual is radius2_slope_median_px2_per_s with p=0.447. Treat balanced physics features as review hypotheses, not context-independent degradation detectors.
- Temporal directionality audit supports a precursor interpretation but not a causal claim: balanced ROI physics predicts future8 with logistic_l2 AUC 0.799/AP 0.793, beating circular time-shift labels at empirical p=0.042; reversed labels remain nontrivial (best AUC 0.750) and past8 is underpowered with 3 positives.
- Balanced spatial front-propagation audit builds 414 spatial kNN edges over 72 balanced ROI nodes; nearest next-cycle front descriptors autocorrelate strongly (radius2_slope_median_px2_per_s rho=0.594, p_perm=9.990e-04) and same future8-label homophily is high (0.867), but automatic ROI identity and cycle-level labels keep this as spatial hypothesis ranking.
- Masked residual transition timing finds low-rank DMD residual weighted centers are closer to automatic phase-transition centers than random at borderline strength (empirical p=0.056), but peak-frame timing is not aligned and persistence particle/nonparticle ratios track kinetic rates.
- Weak-label degradation benchmark converts consensus physics/mode/mask evidence into a guarded manifest: 7 trainable weak rows (3 positive / 4 negative), and only 1 leave-reference fold is class-balanced enough for binary evaluation.

## Model Readout

- Strict no-selection-QC random forest: ROC-AUC 0.651, balanced accuracy 0.573.
- Strict no-selection-QC logistic: ROC-AUC 0.625, balanced accuracy 0.562.
- All physics plus QC random forest: ROC-AUC 0.797, balanced accuracy 0.688.
- Protocol-conditioned residual logistic: ROC-AUC 0.672, balanced accuracy 0.682.
- Protocol-conditioned front-residual logistic: ROC-AUC 0.453, balanced accuracy 0.312.

Interpretation: the stricter model is above random but not deployable. QC/acquisition features improve apparent performance and should be treated as leakage-sensitive guardrails.

## Top Within-Reference Optical Separations

- cumulative_abs_norm_change_centered_by_reference: event-control 0.010, p=2.738e-05
- first_last_corr_centered_by_reference: event-control -0.027, p=4.710e-04
- latent_net_displacement_centered_by_reference: event-control 0.846, p=6.186e-04
- high_fraction_delta_centered_by_reference: event-control 0.048, p=6.840e-04
- roi_norm_mean_slope_centered_by_reference: event-control 5.536e-05, p=7.076e-04
- roi_norm_mean_delta_centered_by_reference: event-control 0.010, p=8.084e-04
- low_fraction_delta_centered_by_reference: event-control -0.046, p=9.843e-04
- centroid_path_px_centered_by_reference: event-control -1.512, p=0.003

## Prefix-Only ROI Forecasts

- front_positive_residual_binary prefix_only random_forest f=0.500: AUC 0.691, balanced accuracy 0.472
- front_positive_residual_binary prefix_plus_context random_forest f=0.250: AUC 0.524, balanced accuracy 0.586
- front_positive_residual_binary prefix_only random_forest f=0.250: AUC 0.520, balanced accuracy 0.534
- front_positive_residual_binary prefix_plus_context random_forest f=0.500: AUC 0.506, balanced accuracy 0.464
- front_positive_residual_binary prefix_only logistic_l2 f=0.500: AUC 0.476, balanced accuracy 0.479
- front_positive_residual_binary prefix_only random_forest f=0.750: AUC 0.409, balanced accuracy 0.510
- front_positive_residual_binary prefix_only logistic_l2 f=0.750: AUC 0.407, balanced accuracy 0.466
- front_positive_residual_binary prefix_only logistic_l2 f=0.250: AUC 0.387, balanced accuracy 0.390
- front_positive_residual_binary prefix_plus_context random_forest f=0.750: AUC 0.356, balanced accuracy 0.484
- front_positive_residual_binary prefix_plus_context logistic_l2 f=0.250: AUC 0.316, balanced accuracy 0.210
- Null check is_event_roi f=0.750: observed AUC 0.562, null p95 0.675, empirical p=0.292
- Null check is_event_enriched_mode f=0.750: observed AUC 0.630, null p95 0.715, empirical p=0.189
- Null check front_positive_residual_binary f=0.500: observed AUC 0.476, null p95 0.668, empirical p=0.515
- Regression mode_review_priority prefix_plus_context random_forest f=0.250: MAE ratio vs median baseline 0.951, rho -0.054
- Regression mode_review_priority prefix_plus_context random_forest f=0.750: MAE ratio vs median baseline 0.979, rho 0.027
- Regression mode_review_priority prefix_plus_context random_forest f=0.500: MAE ratio vs median baseline 0.989, rho -0.046
- Regression mode_review_priority prefix_only random_forest f=0.750: MAE ratio vs median baseline 1.208, rho 0.033
- Guardrail: Prefix-only ROI forecasts use cropped particle-region sequences, but the cohort is small and selected around event-reference cycles; treat results as physics-signal triage, not a deployable predictor.

## Prefix ROI Feature Importance

- Rows/features: 52 / 54
- Model/target: logistic_l2 / front_positive_residual_binary
- Pooled OOF AUC: 0.447; balanced accuracy 0.489
- Null check: p95 AUC 0.638, empirical p=0.714
- Group ablation remove mean_intensity_trace: AUC drop 0.171, n_features=18
- Group ablation remove frame_texture_level: AUC drop -0.002, n_features=5
- Group ablation remove stage_drift: AUC drop -0.023, n_features=3
- Group ablation remove temporal_change_energy: AUC drop -0.048, n_features=14
- Group ablation remove bright_dark_fraction: AUC drop -0.074, n_features=14
- Feature permutation stage_prefix_step_max (stage_drift): AUC drop 0.132, positive-fold fraction 0.500
- Feature permutation low_fraction_prefix_std (bright_dark_fraction): AUC drop 0.009, positive-fold fraction 0.250
- Feature permutation average_intensity_prefix_mean (mean_intensity_trace): AUC drop 0.008, positive-fold fraction 0.250
- Feature permutation average_intensity_prefix_last (mean_intensity_trace): AUC drop 0.008, positive-fold fraction 0.250
- Feature permutation roi_norm_mean_prefix_first (mean_intensity_trace): AUC drop 0.007, positive-fold fraction 0.500
- Feature permutation first_q30_threshold (bright_dark_fraction): AUC drop 0.006, positive-fold fraction 0.250
- Guardrail: Feature importance is computed on the small 52-ROI selected cohort with leave-event-reference-cycle-out folds. Treat it as mechanistic triage for early particle-region video signals, not causal proof or a deployable detector.

## Top Protocol-Conditioned Event Effects

- roi_norm_mean_delta_protocol_residual: event-control 0.003, p=4.424e-05
- high_fraction_delta_protocol_residual: event-control 0.015, p=8.238e-05
- low_fraction_delta_protocol_residual: event-control -0.015, p=4.707e-04
- first_last_corr_protocol_residual: event-control -0.018, p=0.019
- cumulative_abs_norm_change_protocol_residual: event-control 0.004, p=0.019
- dmd_minus_persistence_mse_protocol_residual: event-control 4.515e-04, p=0.041
- latent_net_displacement_protocol_residual: event-control 0.377, p=0.043
- low_rank_dmd_mse_protocol_residual: event-control 4.593e-04, p=0.046

## Threshold-Robust Front Readout

- phase_slope_median_per_s: event-control 1.094e-06, p=0.017
- phase_slope_abs_median_per_s: event-control 7.931e-07, p=0.031
- threshold_robust_phase_score: event-control 0.156, p=0.041
- phase_slope_positive_fraction: event-control 0.139, p=0.041
- radius2_slope_median_px2_per_s: event-control 5.002e-04, p=0.259
- diffusion_proxy_median_um2_per_s: event-control 1.152e-06, p=0.259
- diffusion_proxy_abs_median_um2_per_s: event-control -4.722e-07, p=0.680
- threshold_robust_diffusion_score: event-control 0.011, p=0.993
- By-event strongest cases include cycle 156 phase-slope separation and cycle 86 radius/diffusion-proxy differences; all remain apparent optical proxies.

## Protocol-Conditioned Front Effects

- phase_slope_positive_fraction_protocol_residual: event-control 0.064, p=8.252e-04
- phase_slope_negative_fraction_protocol_residual: event-control -0.064, p=8.252e-04
- threshold_robust_phase_score_protocol_residual: event-control 0.090, p=0.393
- diffusion_proxy_iqr_um2_per_s_protocol_residual: event-control -3.988e-07, p=0.404
- radius2_slope_iqr_px2_per_s_protocol_residual: event-control -1.731e-04, p=0.404
- phase_slope_iqr_per_s_protocol_residual: event-control -9.686e-08, p=0.414
- radius2_slope_negative_fraction_protocol_residual: event-control -0.099, p=0.468
- radius2_slope_positive_fraction_protocol_residual: event-control 0.099, p=0.468
- Conditioning preserves phase-slope sign consistency but weakens magnitude and diffusion-proxy effects; front residuals are not a standalone detector.

## Front QC Sensitivity

- all_front_rois: n=52, event/control 24/28, phase-CI-positive fraction 0.500
- complete_threshold_sweep: n=52, event/control 24/28, phase-CI-positive fraction 0.500
- front_quality_top_half: n=26, event/control 20/6, phase-CI-positive fraction 0.577
- front_quality_top_quartile: n=13, event/control 8/5, phase-CI-positive fraction 0.462
- q70_phase_ci_excludes_zero: n=30, event/control 18/12, phase-CI-positive fraction 0.867
- q70_phase_ci_positive: n=26, event/control 18/8, phase-CI-positive fraction 1.000
- review_panel_selected: n=24, event/control 18/6, phase-CI-positive fraction 0.750
- review_panel_not_fragmented: n=7, event/control 7/0, phase-CI-positive fraction 0.857
- review_panel_no_auto_flags: n=5, event/control 5/0, phase-CI-positive fraction 1.000
- all_front_rois phase-sign residual: median event-control 0.047, bootstrap p05 0.034, MW p=8.252e-04, permutation p=0.047
- complete_threshold_sweep phase-sign residual: median event-control 0.047, bootstrap p05 0.034, MW p=8.252e-04, permutation p=0.042
- front_quality_top_half phase-sign residual: median event-control 0.093, bootstrap p05 -0.051, MW p=0.116, permutation p=0.024
- front_quality_top_quartile phase-sign residual: median event-control 0.139, bootstrap p05 -0.038, MW p=0.158, permutation p=0.036
- q70_phase_ci_excludes_zero phase-sign residual: median event-control 0.088, bootstrap p05 0.047, MW p=1.065e-04, permutation p=9.995e-04
- Diffusion-proxy separations do not remain globally robust; review-panel diffusion differences are selection-biased and still require manual QC.
- Control-balanced QC augmentation selects 40 ROIs (16 event / 24 control), with non-fragmented counts {'control': 4, 'event': 6}.
- Control-balanced sensitivity balanced_qc_not_fragmented phase-sign residual: event/control 6/4, median event-control 0.083, bootstrap p05 -0.004, MW p=0.139, permutation p=0.046.
- Control-balanced sensitivity balanced_qc_selected phase-sign residual: event/control 16/24, median event-control 0.091, bootstrap p05 0.047, MW p=0.003, permutation p=0.008.
- Control-balanced robust positive phase-residual strata: all_front_rois, balanced_qc_selected, complete_threshold_sweep, original_qc_selected, q70_phase_ci_excludes_zero, q70_phase_ci_positive.

## Residual Physics Mode Taxonomy

- Selected k=4 with silhouette=0.204 and mean seed-stability ARI=0.935.
- optical_brightening_decorrelating_rollout_hard_front_positive: n=13, event fraction 0.846, p=0.003, cycles 60.0;156.0;62.0;86.0
- optical_loss_rollout_hard: n=5, event fraction 0.400, p=1.000, cycles 157.0;156.0;62.0
- near_baseline_or_context_like: n=24, event fraction 0.333, p=0.102, cycles 86.0;58.0;118.0;116.0
- front_negative_high_apparent_front_proxy: n=10, event fraction 0.300, p=0.309, cycles 88.0;90.0;116.0;60.0
- Guardrail: Mode labels are protocol-adjusted computational hypotheses from automatic ROI candidates; they require QC labels before being treated as biological/material degradation modes.

## Cycle/Region Mode Context

- cycle 60: n=6, event-enriched mode fraction=0.833, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;front_negative_high_apparent_front_proxy
- cycle 156: n=6, event-enriched mode fraction=0.667, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;optical_loss_rollout_hard
- cycle 62: n=4, event-enriched mode fraction=0.250, top modes=optical_brightening_decorrelating_rollout_hard_front_positive;optical_loss_rollout_hard;front_negative_high_apparent_front_proxy
- cycle 158: n=4, event-enriched mode fraction=0.250, top modes=near_baseline_or_context_like;optical_brightening_decorrelating_rollout_hard_front_positive;front_negative_high_apparent_front_proxy
- cycle 86: n=6, event-enriched mode fraction=0.167, top modes=near_baseline_or_context_like;optical_brightening_decorrelating_rollout_hard_front_positive
- cycle 116: n=6, event-enriched mode fraction=0.167, top modes=near_baseline_or_context_like;front_negative_high_apparent_front_proxy;optical_brightening_decorrelating_rollout_hard_front_positive
- region x2_y3: n=15, event-enriched mode fraction=0.467, event fraction=0.600
- region x1_y1: n=3, event-enriched mode fraction=0.333, event fraction=0.333
- region x1_y2: n=13, event-enriched mode fraction=0.308, event fraction=0.308
- region x3_y1: n=15, event-enriched mode fraction=0.067, event fraction=0.400
- Context-only leave-cycle-out classifier: pooled ROC-AUC 0.429, pooled balanced accuracy 0.433; descriptive context is not a standalone detector.

## Spatiotemporal Degradation Graph

- Graph size: 52 ROI nodes and 510 directed nearest-neighbor edges.
- Homophily same_cycle_spatial_knn front_positive_residual_binary: same fraction 0.936, null mean 0.489, empirical p=9.990e-04
- Homophily same_cycle_spatial_knn is_event_roi: same fraction 1.000, null mean 0.473, empirical p=9.990e-04
- Homophily same_cycle_spatial_knn is_event_enriched_mode: same fraction 0.769, null mean 0.647, empirical p=0.006
- Homophily same_reference_spatial_knn is_event_enriched_mode: same fraction 0.673, null mean 0.648, empirical p=0.264
- Homophily next_cycle_spatial_knn is_event_enriched_mode: same fraction 0.602, null mean 0.645, empirical p=0.888
- Homophily previous_cycle_spatial_knn is_event_enriched_mode: same fraction 0.544, null mean 0.628, empirical p=0.975
- Continuous neighbor same_cycle_spatial_knn phase_slope_positive_fraction_protocol_residual: rho 0.696, null p95 0.197, empirical p=9.990e-04
- Continuous neighbor same_reference_spatial_knn threshold_robust_phase_score_protocol_residual: rho 0.275, null p95 0.096, empirical p=0.003
- Continuous neighbor same_cycle_spatial_knn mode_review_priority: rho 0.814, null p95 0.784, empirical p=0.010
- Continuous neighbor next_cycle_spatial_knn mode_review_priority: rho 0.783, null p95 0.776, empirical p=0.024
- Continuous neighbor next_cycle_spatial_knn phase_slope_positive_fraction_protocol_residual: rho -0.219, null p95 0.124, empirical p=0.024
- Continuous neighbor next_cycle_spatial_knn diffusion_proxy_abs_median_um2_per_s_protocol_residual: rho -0.227, null p95 0.061, empirical p=0.036
- Temporal lag is_event_roi: n=30, previous-neighbor metric 0.167
- Temporal lag is_event_enriched_mode: n=30, previous-neighbor metric 0.441
- Temporal lag front_positive_residual_binary: n=30, previous-neighbor metric 0.426
- Temporal lag mode_review_priority: n=30, previous-neighbor metric NA
- Distance gradient is_event_roi: positive-positive median distance 277.440 px vs other 10.419 px, p=2.672e-04
- Distance gradient front_positive_residual_binary: positive-positive median distance 196.291 px vs other 28.496 px, p=0.129
- Distance gradient is_event_enriched_mode: positive-positive median distance 97.506 px vs other 103.716 px, p=0.672
- Guardrail: Spatiotemporal graph tests use automatic ROI coordinates and automatic residual labels on a selected 52-ROI cohort. They test clustering/propagation hypotheses for review prioritization, not causal material degradation mechanisms.

## Phase Kinetics Avrami Audit

- ROI/features: 52 / 40
- event_enriched_mode_vs_other q60_logistic_k_per_s: median difference 2.660e-04, p=0.014
- event_enriched_mode_vs_other q70_fraction_delta: median difference 0.006, p=0.017
- event_enriched_mode_vs_other q70_transformed_fraction_delta: median difference 0.006, p=0.017
- event_vs_control q70_avrami_r2: median difference 0.045, p=0.022
- event_vs_control q80_time_of_max_abs_rate_frac: median difference -0.132, p=0.024
- event_vs_control q60_avrami_r2: median difference 0.025, p=0.028
- event_enriched_mode_vs_other q80_positive_rate_fraction: median difference 0.021, p=0.032
- event_enriched_mode_vs_other q80_time_of_max_abs_rate_frac: median difference -0.200, p=0.034
- Correlation q70_logistic_r2 vs n_frames_percentile: rho -0.801, p=2.796e-12
- Correlation q60_logistic_r2 vs n_frames_percentile: rho -0.776, p=9.625e-11
- Correlation roi_norm_rate_sign_consistency vs n_frames_percentile: rho 0.728, p=9.758e-10
- Correlation roi_norm_max_abs_rate_per_s vs n_frames_percentile: rho 0.701, p=7.281e-09
- Correlation q80_max_abs_rate_per_s vs n_frames_percentile: rho 0.699, p=8.018e-09
- Correlation q60_max_abs_rate_per_s vs n_frames_percentile: rho 0.690, p=1.525e-08
- Correlation q70_max_abs_rate_per_s vs n_frames_percentile: rho 0.686, p=2.043e-08
- Correlation q70_variation_to_net_abs vs n_frames_percentile: rho -0.675, p=4.098e-08
- Guardrail: Kinetic fits are optical phase-fraction proxies from cropped particle ROIs using provisional timing. Avrami/logistic parameters are descriptive and not calibrated reaction constants.

## Calibration Metadata Audit

- HDF5 files/movie datasets/camera timing: 33 / 32 / 32
- HDF5 calibration-like attribute hits: 0
- Median sampled HDF5 timing FPS proxy: 0.099
- PPTX files scanned/calibration hits: 3 / 3
- PPTX hit Degradation Paper Outline.pptx slide 16: We also plot the minimum viable facet size based on the point spread function of the microscope as ~0.2um^2 (dotted line)(check value). ADD COMPARISON TO MANUFACTURER’S EXPECTED FACET SIZES. The consistency of facet size
- PPTX hit Degradation Paper Outline.pptx slide 3: an improved microscope design, iterating on the instrument used by Merryweather et al., that offers 96nm pixel size and 180x120um FoV , compared to XXnm and YYxZZum . Example image of an NMC-811 electrode with 60%(?) act
- PPTX hit Degradation Project.pptx slide 3: and wouldn’t cycle again. Switched to second cell… Cell 2: Cycled from pristine. No bubbles visible FOV: full field of view, including 4 drilled holes on left half, and undrilled holes on right half. Imaging camera expos
- Guardrail: This audit verifies metadata evidence for timebase/spatial calibration. Camera timing is present in sampled HDF5 files, but the sampled timing rows can represent sparse segment/cycle timing rather than true camera frame cadence. Physical pixel-size evidence should be treated as slide-derived unless raw HDF5 attributes or microscope metadata explicitly confirm it.

## Calibration Claim Risk Register

- Claim families/source tables: 11 / 11
- Calibration evidence: 32 HDF5 timing files, 0 HDF5 spatial-calibration attrs, 3 PPTX hits
- event_candidate_fronts: proxy_only - Use as automatic front-candidate ranking only; do not interpret as calibrated diffusion.
- roi_phase_boundary_mobility: optical_front_proxy - Describe as apparent optical boundary mobility in cropped particle ROIs.
- multi_cycle_roi_mobility: optical_front_proxy - Use for event/control morphology comparisons, not transport constants.
- multi_cycle_threshold_robust_fronts: phase_sign_usable_diffusion_proxy_only - Phase-slope sign/fraction trends are usable optical proxies; diffusion numbers are apparent um^2/s proxies pending calibration provenance.
- front_roi_calibration_qc: apparent_um_scale_proxy - Keep units as apparent um^2/s optical-front proxies and pair with QC flags/manual labels.
- phase_kinetics_avrami: descriptive_optical_kinetics_proxy - Report as optical transition-sharpness/rate descriptors, not reaction constants.
- manual_qc_gated_front_effects: publication_gate_pending - No manual-QC accepted front/diffusion claim should be made until this table is populated.
- roi_front_qc_package: review_prioritization_only - Use to assign labels and inspect artifacts; it does not validate diffusion by itself.
- Guardrail: This register audits wording risk, not numerical correctness. Current front/kinetic outputs are strongest as optical particle-region proxies; diffusion-like values remain apparent proxies until spatial calibration, true frame cadence, masks, and manual QC are jointly validated.

## Apparent Diffusion Calibration Bounds

- Threshold rows/ROI/cycles: 504 / 72 / 24
- ROI with HDF5 timing: 72
- Pixel-size assumptions: [0.08, 0.096, 0.12]; default 0.096 um/px
- Median ROI elapsed / HDF5 elapsed ratio: 1.002
- q70 median apparent D at 96 nm/px: 4.322e-08 um2/s; median abs 2.597e-06 um2/s; positive fraction 0.514
- Threshold 0.550: median D 3.899e-07, median abs D 1.489e-06, positive fraction 0.556
- Threshold 0.600: median D 3.868e-07, median abs D 1.672e-06, positive fraction 0.556
- Threshold 0.650: median D 2.259e-07, median abs D 2.087e-06, positive fraction 0.500
- Threshold 0.700: median D 4.320e-08, median abs D 2.597e-06, positive fraction 0.514
- Threshold 0.750: median D 1.058e-07, median abs D 2.909e-06, positive fraction 0.514
- Threshold 0.800: median D 6.785e-07, median abs D 3.799e-06, positive fraction 0.556
- Threshold 0.850: median D 1.032e-06, median abs D 4.461e-06, positive fraction 0.569
- q70 calibration future8 test apparent_D_h5median_px0p08_um2_per_s: median positive-negative -4.398e-07, p=0.175
- q70 calibration future8 test apparent_D_h5median_px0p096_um2_per_s: median positive-negative -6.334e-07, p=0.175
- q70 calibration future8 test apparent_D_h5median_px0p12_um2_per_s: median positive-negative -9.896e-07, p=0.175
- q70 calibration future8 test roi_elapsed_to_h5_median_ratio: median positive-negative 3.021e-04, p=0.189
- q70 calibration future8 test apparent_D_h5_timing_envelope_abs_min_um2_per_s: median positive-negative -1.272e-07, p=0.313
- q70 calibration future8 test apparent_D_h5median_abs_um2_per_s: median positive-negative -1.013e-07, p=0.377
- Source timing 11_c2_x10_050723: dt median 10.039s, max/median 1.124, ROI/H5 elapsed 1.002, median abs D 1.594e-06
- Source timing 12_c2_x10_070723: dt median 10.062s, max/median 1.143, ROI/H5 elapsed 1.000, median abs D 3.175e-06
- Source timing 14_c2_x10_HighCOV_110723: dt median 10.062s, max/median 1.185, ROI/H5 elapsed 1.001, median abs D 1.016e-06
- Source timing 15_c2_x5_HighCOV_120723: dt median 10.039s, max/median 13.753, ROI/H5 elapsed 1.009, median abs D 1.157e-06
- Source timing 16_c2_x10_HighHighCOV_130723: dt median 10.039s, max/median 13.736, ROI/H5 elapsed 1.002, median abs D 7.629e-06
- Source timing 17_c2_x10_HighHighCOV_150723: dt median 10.039s, max/median 13.757, ROI/H5 elapsed 1.002, median abs D 2.543e-06
- Source timing 18_c2_xN_HighHighCOV_170723: dt median 10.062s, max/median 1.157, ROI/H5 elapsed 1.001, median abs D 2.730e-06
- Source timing 5_c2_x10_260623: dt median 10.039s, max/median 13.783, ROI/H5 elapsed 1.002, median abs D 2.373e-06
- Calibration-bound link h5_dt_max_to_median_ratio vs transferred_masked_residual_signature: rho=0.728, p=2.858e-84
- Calibration-bound link apparent_D_h5median_px0p096_um2_per_s vs transferred_masked_residual_signature: rho=-0.257, p=4.700e-09
- Calibration-bound link roi_elapsed_to_h5_median_ratio vs validation_score_recon: rho=0.222, p=5.050e-07
- Calibration-bound link apparent_D_h5median_abs_um2_per_s vs transferred_masked_residual_signature: rho=-0.153, p=5.768e-04
- Guardrail: Apparent diffusion values are recalibrated from HDF5 camera timing and slide-derived pixel-size assumptions. No HDF5 pixel-size attribute was found, and the values remain optical-front proxies, not validated material diffusion coefficients.

## Diffusion Physics Consistency Audit

- ROI/threshold rows/sources: 72 / 504 / 9
- Automatic physics-consistent / publication-ready diffusion candidates: 1 / 0
- Median abs apparent D / positive-D fraction / radius2 fit R2 / threshold sensitivity: 2.229e-06 / 0.714 / 0.055 / 0.798
- Diffusion physics gate gate_all_thresholds_present: 72/72 pass (1.000)
- Diffusion physics gate gate_positive_expansion: 26/72 pass (0.361)
- Diffusion physics gate gate_fit_quality: 7/72 pass (0.097)
- Diffusion physics gate gate_threshold_stability: 63/72 pass (0.875)
- Diffusion physics gate gate_h5_timing_stable: 24/72 pass (0.333)
- Diffusion physics gate gate_low_drift: 72/72 pass (1.000)
- Diffusion physics gate gate_q70_positive_ci: 0/72 pass (0.000)
- Diffusion physics gate automatic_diffusion_physics_consistent: 1/72 pass (0.014)
- Diffusion physics gate publication_ready_diffusion_candidate: 0/72 pass (0.000)
- Physics-consistent candidate cycle78_rank22_obj2: cycle 78, D 4.093e-06, fit R2 0.415, future8=1, future16=1
- Diffusion consistency target future_any_drop_within_16cycles positive_D_fraction: median positive-negative -0.714, AUC 0.705, p=0.013
- Diffusion consistency target future_any_drop_within_16cycles median_apparent_D_um2_per_s: median positive-negative -2.569e-06, AUC 0.701, p=0.018
- Diffusion consistency target future_any_drop_within_16cycles transferred_masked_residual_signature: median positive-negative 0.342, AUC 0.695, p=0.021
- Diffusion consistency target future_any_drop_within_16cycles physics_consistency_score: median positive-negative -0.399, AUC 0.691, p=0.024
- Diffusion consistency target future_any_drop_within_16cycles diffusion_physics_gate_count: median positive-negative 0.000, AUC 0.629, p=0.104
- Diffusion consistency target future_any_drop_within_16cycles threshold_sensitivity_iqr_over_median_abs: median positive-negative 0.248, AUC 0.622, p=0.149
- Diffusion consistency correlation h5_dt_max_to_median_ratio vs transferred_masked_residual_signature: rho=0.728, p=4.405e-13, n=72
- Diffusion consistency correlation physics_consistency_score vs transferred_masked_residual_signature: rho=-0.529, p=1.746e-06, n=72
- Diffusion consistency correlation diffusion_physics_gate_count vs transferred_masked_residual_signature: rho=-0.469, p=3.243e-05, n=72
- Diffusion consistency correlation median_abs_apparent_D_um2_per_s vs validation_score_recon: rho=-0.409, p=3.547e-04, n=72
- Guardrail: Automatic apparent-D candidates must pass positive expansion, radius^2 fit, threshold-stability, timing, drift, q70 CI, and manual-QC gates before any calibrated diffusion claim. This audit is a physics-consistency filter over optical front proxies, not a material diffusion measurement.

## Cross-Modal Degradation Consensus

- Cycles scored/with votes: 89 / 53
- Median consensus score: 0.499; modal vote threshold 0.850
- Consensus class synchronized_multimodal_degradation_candidate: n=2, median score 0.804, modal votes 4.000, event rate 1.000, future8 rate 0.000, median frame percentile 0.045
- Consensus class multimodal_outlier_without_trace_drop: n=4, median score 0.623, modal votes 3.000, event rate 0.000, future8 rate 0.500, median frame percentile 0.770
- Consensus class trace_event_low_modal_support: n=2, median score 0.618, modal votes 1.500, event rate 1.000, future8 rate 0.000, median frame percentile 0.640
- Consensus class low_consensus: n=81, median score 0.485, modal votes 0.753, event rate 0.000, future8 rate 0.222, median frame percentile 0.500
- Consensus cycle 86: synchronized_multimodal_degradation_candidate, score 0.813, votes 4, event=1, future8=0, frame percentile 0.079
- Consensus cycle 116: synchronized_multimodal_degradation_candidate, score 0.795, votes 4, event=1, future8=0, frame percentile 0.011
- Consensus cycle 150: multimodal_outlier_without_trace_drop, score 0.646, votes 3, event=0, future8=1, frame percentile 0.848
- Consensus cycle 151: multimodal_outlier_without_trace_drop, score 0.625, votes 3, event=0, future8=1, frame percentile 0.831
- Consensus cycle 104: multimodal_outlier_without_trace_drop, score 0.621, votes 3, event=0, future8=0, frame percentile 0.079
- Consensus cycle 126: multimodal_outlier_without_trace_drop, score 0.565, votes 3, event=0, future8=0, frame percentile 0.708
- Consensus cycle 76: low_consensus, score 0.710, votes 2, event=0, future8=0, frame percentile 0.202
- Consensus cycle 125: low_consensus, score 0.709, votes 2, event=0, future8=0, frame percentile 0.742
- Consensus contrast cross_modal_consensus_score vs any_abrupt_drop: median positive-negative 0.240, rho=0.298, n=89
- Consensus contrast n_modal_votes vs any_abrupt_drop: median positive-negative 2.000, rho=0.277, n=89
- Consensus contrast low_frame_rank vs any_abrupt_drop: median positive-negative 0.298, rho=0.122, n=89
- Consensus contrast hazard_probability_max vs any_abrupt_drop: median positive-negative -0.024, rho=-0.116, n=62
- Consensus contrast cross_modal_consensus_score vs future_any_drop_within_8cycles: median positive-negative 0.009, rho=0.062, n=89
- Consensus contrast n_modal_votes vs future_any_drop_within_8cycles: median positive-negative 0.000, rho=0.068, n=89
- Guardrail: Consensus score is an audit/ranking statistic across already-derived modalities; it is not a calibrated probability and keeps frame-count/acquisition as an explicit confounder.

## Echem Optical Breakpoint Audit

- Cycles/features/permutations: 89 / 84 / 5000
- Event cycles tested: [86.0, 116.0]
- Event-centered breakpoint state_step_norm_delta_prev +/-4 cycles: scaled shift -2.263, control p95 abs 1.774, empirical p=0.040, bootstrap p=0.002
- Event-centered breakpoint mean_abs_delta_prev_delta_prev +/-4 cycles: scaled shift -1.753, control p95 abs 1.506, empirical p=0.053, bootstrap p=0.003
- Event-centered breakpoint axis_step_delta_prev +/-4 cycles: scaled shift -1.459, control p95 abs 1.073, empirical p=0.053, bootstrap p=0.006
- Event-centered breakpoint all_dq_abs_midV_frac_delta_prev +/-4 cycles: scaled shift 7.882, control p95 abs 10.640, empirical p=0.074, bootstrap p=0.011
- Event-centered breakpoint cycle_state_pc2 +/-8 cycles: scaled shift -1.390, control p95 abs 1.540, empirical p=0.150, bootstrap p=0.014
- Event-centered breakpoint shape_V_std_delta_prev +/-4 cycles: scaled shift -4.038, control p95 abs 4.307, empirical p=0.074, bootstrap p=0.023
- Event-centered breakpoint cycle_state_pc2 +/-12 cycles: scaled shift -1.348, control p95 abs 1.473, empirical p=0.175, bootstrap p=0.023
- Event-centered breakpoint shape_V_mean +/-12 cycles: scaled shift 0.492, control p95 abs 0.866, empirical p=0.125, bootstrap p=0.024
- Echem/trace label link axis_step vs any_abrupt_drop: median positive-negative 2.149, MW p=8.062e-05, rho=0.339
- Echem/trace label link cycle_state_pc1_delta_prev vs any_abrupt_drop: median positive-negative -2.149, MW p=8.062e-05, rho=-0.339
- Echem/trace label link degradation_state_axis_delta_prev vs any_abrupt_drop: median positive-negative 2.149, MW p=8.062e-05, rho=0.339
- Echem/trace label link mean_abs_delta_prev_delta_prev vs any_abrupt_drop: median positive-negative 0.093, MW p=1.087e-04, rho=0.339
- Echem/trace label link state_step_norm_delta_prev vs any_abrupt_drop: median positive-negative 6.141, MW p=2.165e-04, rho=0.332
- Echem/trace label link cycle_state_pc2 vs future_any_drop_within_8cycles: median positive-negative 0.730, MW p=2.322e-04, rho=0.393
- Echem/trace label link frames_percentile vs future_sync2_drop_within_8cycles: median positive-negative -0.362, MW p=9.666e-04, rho=-0.353
- Echem/trace label link n_frames vs future_sync2_drop_within_8cycles: median positive-negative -36.500, MW p=9.666e-04, rho=-0.353
- Event-cycle breakpoint rank cycle 116 all_dq_abs_midV_frac +/-12: rank 1, percentile 1.000, scaled shift -0.822
- Event-cycle breakpoint rank cycle 116 axis_step_rolling5_slope +/-8: rank 1, percentile 1.000, scaled shift 2.171
- Event-cycle breakpoint rank cycle 86 particle_norm_cv_delta_prev +/-8: rank 1, percentile 1.000, scaled shift 1.782
- Event-cycle breakpoint rank cycle 86 shape_I_abs_mean_mA +/-12: rank 1, percentile 0.890, scaled shift -1.125
- Event-cycle breakpoint rank cycle 116 shape_V_mean +/-12: rank 1, percentile 1.000, scaled shift 0.958
- Event-cycle breakpoint rank cycle 116 shape_V_std +/-12: rank 1, percentile 1.000, scaled shift 0.700
- Global echem breakpoint candidate cycle 150 coulombic_efficiency_pct_rolling5_slope: scaled shift -237.256, rank 1, pre/post n=12/2
- Global echem breakpoint candidate cycle 150 coulombic_efficiency_pct_rolling5_slope: scaled shift -237.143, rank 1, pre/post n=8/2
- Global echem breakpoint candidate cycle 150 coulombic_efficiency_pct_rolling5_slope: scaled shift -233.602, rank 1, pre/post n=4/2
- Global echem breakpoint candidate cycle 150 all_dq_abs_entropy_rolling5_slope: scaled shift -194.381, rank 1, pre/post n=12/2
- Global echem breakpoint candidate cycle 150 all_dq_abs_entropy_rolling5_slope: scaled shift -193.995, rank 1, pre/post n=8/2
- Guardrail: Cycle-level echem/optical breakpoint audit uses compact derived cycle features and weak optical event labels. It tests temporal co-occurrence and trajectory shifts, not causality, manual ROI validation, or calibrated material transport.

## Echem Optical Regime Atlas

- Cycles/features: 89 / 44
- Missing echem-shape cycles: 8; extreme-or-missing CE cycles: 10
- Echem PC1 regime pc1_mid: n=29, median cycle 102.000, median consensus 0.542, future8 rate 0.379, extreme/missing CE rate 0.276
- Echem PC1 regime pc1_high: n=30, median cycle 53.000, median consensus 0.527, future8 rate 0.167, extreme/missing CE rate 0.033
- Echem PC1 regime pc1_low: n=30, median cycle 135.000, median consensus 0.429, future8 rate 0.133, extreme/missing CE rate 0.033
- Echem binary contrast pos_dq_abs_peak_voltage vs multimodal_outlier_without_trace_drop: median positive-negative 0.050, p=1.453e-04, n=81
- Echem binary contrast all_dq_abs_peak_voltage vs multimodal_outlier_without_trace_drop: median positive-negative 0.050, p=0.005, n=81
- Echem binary contrast shape_charge_mAh_abs vs future_any_drop_within_8cycles: median positive-negative -0.025, p=0.030, n=81
- Echem binary contrast capacity_fade_from_first_mAh vs future_any_drop_within_8cycles: median positive-negative 0.014, p=0.033, n=81
- Echem binary contrast capacity_fraction_of_first vs future_any_drop_within_8cycles: median positive-negative -0.021, p=0.033, n=81
- Echem binary contrast capacity_mAh vs future_any_drop_within_8cycles: median positive-negative -0.014, p=0.033, n=81
- Echem binary contrast shape_charge_mAh_neg_abs vs future_any_drop_within_8cycles: median positive-negative -0.014, p=0.034, n=81
- Echem binary contrast dqdv_entropy_asymmetry vs future_any_drop_within_8cycles: median positive-negative 0.027, p=0.039, n=81
- Echem optical link shape_dVdt_abs_p95 vs cross_modal_consensus_score: rho=0.617, p=8.505e-10, n=81
- Echem optical link all_dq_abs_midV_frac vs particle_norm_cv: rho=-0.404, p=1.873e-04, n=81
- Echem optical link pos_dq_abs_highV_frac vs particle_norm_cv: rho=0.393, p=2.853e-04, n=81
- Echem optical link all_dq_abs_peak_voltage vs max_abs_delta_prev: rho=-0.394, p=2.994e-04, n=80
- Echem optical link pos_dq_abs_midV_frac vs particle_norm_cv: rho=-0.389, p=3.355e-04, n=81
- Echem optical link neg_dq_abs_peak_frac vs particle_norm_cv: rho=-0.385, p=3.935e-04, n=81
- Echem optical link shape_dVdt_abs_p95 vs n_modal_votes: rho=0.373, p=5.997e-04, n=81
- Echem optical link echem_regime_pc3 vs particle_norm_cv: rho=0.356, p=6.112e-04, n=89
- Echem-optical priority cycle 150: score 0.939, regime pc1_low, consensus 0.646, class multimodal_outlier_without_trace_drop, CE-flag=1
- Echem-optical priority cycle 151: score 0.927, regime pc1_mid, consensus 0.625, class multimodal_outlier_without_trace_drop, CE-flag=1
- Echem-optical priority cycle 126: score 0.799, regime pc1_high, consensus 0.565, class multimodal_outlier_without_trace_drop, CE-flag=1
- Echem-optical priority cycle 116: score 0.792, regime pc1_low, consensus 0.795, class synchronized_multimodal_degradation_candidate, CE-flag=0
- Echem-optical priority cycle 86: score 0.781, regime pc1_mid, consensus 0.813, class synchronized_multimodal_degradation_candidate, CE-flag=0
- Echem-optical priority cycle 112: score 0.777, regime pc1_high, consensus 0.649, class low_consensus, CE-flag=0
- Guardrail: This atlas uses echem shape and dQ/dV-like proxy descriptors to organize optical degradation hypotheses. It is not calibrated dQ/dV, not a mechanistic phase diagram, and does not remove the acquisition/frame-count confounder by itself.

## Echem-Conditioned Optical Predictor

- Cycles/targets: 89 / 7
- Feature set sizes: {'acquisition_context': 7, 'cycle_state_upper_bound': 14, 'echem_plus_acquisition': 56, 'echem_regime': 49}
- Echem feature-set delta leave_one_cycle high_cross_modal_consensus_q75 echem_regime_minus_acquisition: delta AUC 0.113, base 0.686, comparison 0.799
- Echem feature-set delta leave_one_cycle high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC 0.113, base 0.527, comparison 0.640
- Echem feature-set delta leave_one_cycle high_cross_modal_consensus_q75 echem_plus_acquisition_minus_acquisition: delta AUC 0.111, base 0.686, comparison 0.797
- Echem feature-set delta leave_one_cycle high_particle_norm_cv_q75 echem_plus_acquisition_minus_acquisition: delta AUC 0.103, base 0.527, comparison 0.630
- Echem feature-set delta rolling_origin high_particle_norm_cv_q75 echem_regime_minus_acquisition: delta AUC 0.084, base 0.525, comparison 0.609
- Echem feature-set delta rolling_origin high_cross_modal_consensus_q75 echem_regime_minus_acquisition: delta AUC 0.067, base 0.625, comparison 0.692
- Echem feature-set delta rolling_origin high_particle_norm_cv_q75 echem_plus_acquisition_minus_acquisition: delta AUC 0.052, base 0.525, comparison 0.577
- Echem feature-set delta leave_one_cycle high_roi_phase_slope_abs_q75 echem_plus_acquisition_minus_acquisition: delta AUC 0.046, base 0.944, comparison 0.991
- Echem feature-set delta leave_one_cycle high_roi_phase_slope_abs_q75 echem_regime_minus_acquisition: delta AUC 0.037, base 0.944, comparison 0.981
- Echem feature-set delta rolling_origin high_cross_modal_consensus_q75 echem_plus_acquisition_minus_acquisition: delta AUC 0.014, base 0.625, comparison 0.639
- Echem-conditioned metric leave_one_cycle synchronized_multimodal_candidate cycle_state_upper_bound: AUC 0.994, AP 0.833, n=89, positives=2
- Echem-conditioned metric leave_one_cycle high_roi_phase_slope_abs_q75 echem_plus_acquisition: AUC 0.991, AP 0.976, n=24, positives=6
- Echem-conditioned metric leave_one_cycle high_roi_phase_slope_abs_q75 echem_regime: AUC 0.981, AP 0.958, n=24, positives=6
- Echem-conditioned metric leave_one_cycle synchronized_multimodal_candidate acquisition_context: AUC 0.966, AP 0.325, n=89, positives=2
- Echem-conditioned metric leave_one_cycle high_roi_phase_slope_abs_q75 acquisition_context: AUC 0.944, AP 0.735, n=24, positives=6
- Echem-conditioned metric leave_one_cycle high_roi_phase_slope_abs_q75 cycle_state_upper_bound: AUC 0.944, AP 0.735, n=24, positives=6
- Echem-conditioned metric leave_one_cycle high_state_step_norm_q75 cycle_state_upper_bound: AUC 0.899, AP 0.835, n=88, positives=22
- Echem-conditioned metric leave_one_cycle high_cross_modal_consensus_q75 cycle_state_upper_bound: AUC 0.829, AP 0.684, n=89, positives=23
- Echem-conditioned metric leave_one_cycle future_any_drop_within_8cycles cycle_state_upper_bound: AUC 0.808, AP 0.530, n=89, positives=20
- Echem-conditioned metric leave_one_cycle high_cross_modal_consensus_q75 echem_regime: AUC 0.799, AP 0.522, n=89, positives=23
- Guardrail: This is a cycle-level weak-label model comparison. Echem-regime gains show conditional association, not deployable prediction, causal mechanism, calibrated dQ/dV, or validated front/diffusion physics.

## Echem-Conditioned ROI Rollout/Front Audit

- ROI rows/cycles/targets: 72 / 24 / 12
- Feature set sizes: {'acquisition_context': 10, 'echem_plus_acquisition': 59, 'echem_regime': 49}
- ROI echem feature-set delta transferred_masked_residual_signature echem_regime_minus_acquisition: delta Spearman 0.450, delta R2 0.626, base rho -0.086, comparison rho 0.364
- ROI echem feature-set delta phase_slope_positive_fraction echem_plus_acquisition_minus_acquisition: delta Spearman 0.327, delta R2 -18.120, base rho 0.010, comparison rho 0.337
- ROI echem feature-set delta transferred_masked_residual_signature echem_plus_acquisition_minus_acquisition: delta Spearman 0.272, delta R2 1.042, base rho -0.086, comparison rho 0.186
- ROI echem feature-set delta phase_slope_positive_fraction echem_regime_minus_acquisition: delta Spearman 0.223, delta R2 -9.254, base rho 0.010, comparison rho 0.234
- ROI echem feature-set delta threshold_robust_phase_score echem_regime_minus_acquisition: delta Spearman 0.159, delta R2 -3.963, base rho 0.119, comparison rho 0.278
- ROI echem feature-set delta persistence_particle_mse_fraction_of_full_mean echem_plus_acquisition_minus_acquisition: delta Spearman 0.044, delta R2 -0.668, base rho 0.685, comparison rho 0.729
- ROI echem feature-set delta roi_norm_mean_delta_last_minus_first echem_regime_minus_acquisition: delta Spearman 0.038, delta R2 -11.834, base rho 0.018, comparison rho 0.057
- ROI echem feature-set delta low_rank_dmd_particle_to_nonparticle_mse_ratio_mean echem_regime_minus_acquisition: delta Spearman 0.027, delta R2 -6.778, base rho 0.031, comparison rho 0.058
- ROI echem feature-set delta radius2_slope_median_px2_per_s echem_plus_acquisition_minus_acquisition: delta Spearman 0.001, delta R2 -0.338, base rho 0.520, comparison rho 0.522
- ROI echem feature-set delta velocity_particle_mse_fraction_of_full_mean echem_plus_acquisition_minus_acquisition: delta Spearman -0.024, delta R2 -18.776, base rho 0.667, comparison rho 0.643
- ROI leave-cycle metric persistence_particle_mse_fraction_of_full_mean echem_plus_acquisition: rho 0.729, R2 -0.199, MAE 0.292, n=72, cycles=24
- ROI leave-cycle metric persistence_particle_mse_fraction_of_full_mean acquisition_context: rho 0.685, R2 0.469, MAE 0.247, n=72, cycles=24
- ROI leave-cycle metric velocity_particle_mse_fraction_of_full_mean acquisition_context: rho 0.667, R2 0.323, MAE 0.255, n=72, cycles=24
- ROI leave-cycle metric phase_slope_abs_median_per_s acquisition_context: rho 0.655, R2 0.267, MAE 1.243e-06, n=72, cycles=24
- ROI leave-cycle metric velocity_particle_mse_fraction_of_full_mean echem_plus_acquisition: rho 0.643, R2 -18.452, MAE 0.668, n=72, cycles=24
- ROI leave-cycle metric persistence_particle_mse_fraction_of_full_mean echem_regime: rho 0.587, R2 -48.210, MAE 0.971, n=72, cycles=24
- ROI leave-cycle metric velocity_particle_mse_fraction_of_full_mean echem_regime: rho 0.530, R2 -73.147, MAE 1.093, n=72, cycles=24
- ROI leave-cycle metric radius2_slope_median_px2_per_s echem_plus_acquisition: rho 0.522, R2 -0.058, MAE 0.002, n=72, cycles=24
- ROI leave-cycle metric radius2_slope_median_px2_per_s acquisition_context: rho 0.520, R2 0.280, MAE 0.002, n=72, cycles=24
- ROI leave-cycle metric phase_slope_abs_median_per_s echem_regime: rho 0.463, R2 -6.996, MAE 2.134e-06, n=72, cycles=24
- ROI acquisition-residual echem link pos_dq_abs_entropy vs residual transferred_masked_residual_signature: rho=-0.745, p=1.006e-10, n=54
- ROI acquisition-residual echem link all_dq_abs_peak_frac vs residual transferred_masked_residual_signature: rho=0.727, p=4.861e-10, n=54
- ROI acquisition-residual echem link all_dq_abs_entropy vs residual transferred_masked_residual_signature: rho=-0.649, p=1.140e-07, n=54
- ROI acquisition-residual echem link neg_dq_abs_peak_voltage vs residual transferred_masked_residual_signature: rho=0.592, p=2.439e-06, n=54
- ROI acquisition-residual echem link shape_V_range vs residual transferred_masked_residual_signature: rho=-0.574, p=5.604e-06, n=54
- ROI acquisition-residual echem link capacity_mAh vs residual transferred_masked_residual_signature: rho=-0.558, p=1.176e-05, n=54
- ROI acquisition-residual echem link capacity_fade_from_first_mAh vs residual transferred_masked_residual_signature: rho=0.558, p=1.176e-05, n=54
- ROI acquisition-residual echem link capacity_fraction_of_first vs residual transferred_masked_residual_signature: rho=-0.558, p=1.176e-05, n=54
- Guardrail: ROI rows are automatic, clustered by cycle, and front/diffusion variables are proxy measurements; use this as a weak-label explanatory audit, not calibrated electrochemical mechanism proof.

## Echem Video Embedding Fusion Audit

- Embedding rows/cycles: 172 / 34
- Cohort counts: {'balanced_future': 72, 'selected_event_control': 52, 'transfer_ranked': 48}
- Feature set sizes: {'acquisition_context': 24, 'echem_regime': 51, 'video_all': 64, 'video_embedding': 16, 'video_plus_echem': 115, 'video_plus_echem_acquisition': 138, 'video_scalar': 48}
- Echem-video fusion delta classification future_any_drop_within_8cycles video_plus_echem_acquisition_minus_echem_regime: delta AUC 0.284, delta R2 NA, delta rho 0.492
- Echem-video fusion delta classification future_any_drop_within_16cycles video_plus_echem_minus_echem_regime: delta AUC 0.249, delta R2 NA, delta rho 0.351
- Echem-video fusion delta classification future_any_drop_within_16cycles video_plus_echem_acquisition_minus_echem_regime: delta AUC 0.236, delta R2 NA, delta rho 0.332
- Echem-video fusion delta classification future_any_drop_within_16cycles video_all_minus_echem_regime: delta AUC 0.192, delta R2 NA, delta rho 0.270
- Echem-video fusion delta classification future_any_drop_within_8cycles video_all_minus_echem_regime: delta AUC 0.191, delta R2 NA, delta rho 0.331
- Echem-video fusion delta classification future_any_drop_within_8cycles video_plus_echem_minus_echem_regime: delta AUC 0.174, delta R2 NA, delta rho 0.301
- Echem-video fusion delta classification future_any_drop_within_8cycles video_plus_echem_acquisition_minus_video_all: delta AUC 0.093, delta R2 NA, delta rho 0.160
- Echem-video fusion delta classification future_any_drop_within_16cycles video_plus_echem_minus_video_all: delta AUC 0.057, delta R2 NA, delta rho 0.081
- Echem-video fusion delta classification future_any_drop_within_16cycles video_plus_echem_acquisition_minus_video_all: delta AUC 0.044, delta R2 NA, delta rho 0.063
- Echem-video fusion delta classification future_any_drop_within_16cycles video_plus_echem_minus_acquisition_context: delta AUC 0.026, delta R2 NA, delta rho 0.036
- Echem-video classification future_any_drop_within_8cycles acquisition_context: AUC 1.000, AP 1.000, rho 0.866, null p=9.990e-04, n=72
- Echem-video classification future_any_drop_within_8cycles video_plus_echem_acquisition: AUC 0.916, AP 0.957, rho 0.720, null p=9.990e-04, n=72
- Echem-video classification future_any_drop_within_8cycles video_all: AUC 0.823, AP 0.874, rho 0.560, null p=9.990e-04, n=72
- Echem-video classification future_any_drop_within_8cycles video_scalar: AUC 0.816, AP 0.866, rho 0.547, null p=NA, n=72
- Echem-video classification future_any_drop_within_8cycles video_plus_echem: AUC 0.806, AP 0.851, rho 0.529, null p=9.990e-04, n=72
- Echem-video classification future_any_drop_within_16cycles video_plus_echem: AUC 0.754, AP 0.926, rho 0.358, null p=0.002, n=72
- Echem-video classification future_any_drop_within_16cycles video_plus_echem_acquisition: AUC 0.742, AP 0.927, rho 0.340, null p=0.002, n=72
- Echem-video classification future_any_drop_within_16cycles acquisition_context: AUC 0.729, AP 0.934, rho 0.322, null p=0.003, n=72
- Echem-video regression particle_norm_cv video_scalar: rho 0.988, R2 0.979, MAE 0.003, n=172
- Echem-video regression particle_norm_cv video_all: rho 0.979, R2 0.955, MAE 0.004, n=172
- Echem-video regression particle_norm_cv video_plus_echem_acquisition: rho 0.941, R2 0.886, MAE 0.006, n=172
- Echem-video regression particle_norm_cv video_plus_echem: rho 0.932, R2 -0.116, MAE 0.010, n=172
- Echem-video regression persistence_particle_mse_fraction_of_full_mean video_scalar: rho 0.817, R2 0.679, MAE 0.177, n=116
- Echem-video regression persistence_particle_mse_fraction_of_full_mean video_plus_echem: rho 0.765, R2 -15.763, MAE 0.603, n=116
- Echem-video regression velocity_particle_mse_fraction_of_full_mean video_scalar: rho 0.758, R2 0.616, MAE 0.199, n=116
- Echem-video regression cross_modal_consensus_score video_plus_echem_acquisition: rho 0.748, R2 0.150, MAE 0.084, n=172
- Guardrail: This is a grouped weak-label fusion audit over automatic masked-video embeddings, echem descriptors, and automatic ROI physics. It supports representation design and review prioritization, not deployable warning, manual particle/front labels, or calibrated diffusion claims.

## Diffusion Proxy Sanity Audit

- Selected high-resolution front ROIs: 12
- Automatic positive diffusion-proxy candidates: 0
- Publication diffusion candidates after manual-QC gate: 0
- Median selected/threshold apparent D: -3.647e-04 / -6.470e-07 um2/s
- Estimator consensus counts: {'mixed': 6, 'negative': 5, 'positive': 1}
- Gate low_drift_relative_to_motion: 12/12 pass (1.000)
- Gate threshold_nonnegative: 5/12 pass (0.417)
- Gate q70_bootstrap_positive: 2/12 pass (0.167)
- Gate selected_nonnegative: 1/12 pass (0.083)
- Gate selected_fit_good: 1/12 pass (0.083)
- Gate manual_qc_accepted: 0/12 pass (0.000)
- Candidate check cycle116_front1_obj6 (event, cycle 116): selected D -3.345e-04, threshold D -1.021e-06, selected R2 0.215, consensus negative, manual pending
- Candidate check cycle116_front2_obj2 (event, cycle 116): selected D -5.845e-04, threshold D -1.065e-06, selected R2 0.222, consensus negative, manual pending
- Candidate check cycle116_front3_obj9 (event, cycle 116): selected D -6.054e-04, threshold D 2.138e-06, selected R2 0.299, consensus mixed, manual pending
- Candidate check cycle116_front4_obj58 (None, cycle 116): selected D -3.712e-04, threshold D NA, selected R2 0.267, consensus mixed, manual pending
- Candidate check cycle116_front7_obj37 (event, cycle 116): selected D -5.000e-04, threshold D 4.589e-06, selected R2 0.612, consensus mixed, manual pending
- Candidate check cycle116_front8_obj20 (event, cycle 116): selected D 8.568e-05, threshold D -5.566e-06, selected R2 0.091, consensus mixed, manual pending
- Diffusion-sanity link selected_r2 vs mask_instability_score: rho=0.218, p=0.519, n=11
- Diffusion-sanity link selected_r2 vs front_quality_score: rho=-0.209, p=0.537, n=11
- Diffusion-sanity link selected_diffusion_um2_per_s vs threshold_robust_diffusion_score: rho=0.182, p=0.593, n=11
- Diffusion-sanity link drift_to_motion_ratio vs threshold_robust_diffusion_score: rho=0.182, p=0.593, n=11
- Guardrail: This audit checks whether optical radius-squared front proxies behave like calibrated diffusion estimates. Publication diffusion claims require manual QC plus positive, estimator-consistent, low-drift, threshold-robust slopes; otherwise values remain apparent optical-front proxies.

## Control-Balanced Front Tracking And Diffusion Sanity

- High-resolution tracked ROIs: 40
- Cohort counts: {'control': 24, 'event': 16}
- Automatic/publication diffusion candidates: 0 / 0
- Median selected/threshold apparent D: -1.029e-04 / -1.021e-06 um2/s
- Estimator consensus counts: {'negative': 15, 'mixed': 14, 'positive': 11}
- Balanced gate low_drift_relative_to_motion: 40/40 pass (1.000)
- Balanced gate threshold_nonnegative: 16/40 pass (0.400)
- Balanced gate q70_bootstrap_positive: 9/40 pass (0.225)
- Balanced gate selected_nonnegative: 8/40 pass (0.200)
- Balanced gate selected_fit_good: 1/40 pass (0.025)
- Balanced gate manual_qc_accepted: 0/40 pass (0.000)
- Balanced event/control diffusion_proxy_median_um2_per_s: median diff 1.195e-06, p=0.314, n=16/24
- Balanced event/control threshold_robust_diffusion_score: median diff 0.019, p=0.456, n=16/24
- Balanced event/control drift_to_motion_ratio: median diff -0.001, p=0.516, n=16/24
- Balanced event/control selected_diffusion_um2_per_s: median diff -4.076e-05, p=0.858, n=16/24
- Balanced event/control selected_radius2_um2_per_s: median diff -1.630e-04, p=0.858, n=16/24
- Balanced candidate check cycle156_front8_obj10 (event, cycle 156): selected D 5.881e-05, selected R2 0.068, consensus positive, manual pending
- Balanced candidate check cycle157_front2_obj2 (control, cycle 157): selected D 6.320e-06, selected R2 6.553e-04, consensus positive, manual pending
- Balanced candidate check cycle157_front4_obj8 (control, cycle 157): selected D -2.030e-05, selected R2 0.003, consensus positive, manual pending
- Balanced candidate check cycle158_front2_obj1 (control, cycle 158): selected D -3.503e-05, selected R2 0.030, consensus positive, manual pending
- Balanced candidate check cycle58_front4_obj2 (control, cycle 58): selected D -2.425e-04, selected R2 0.046, consensus positive, manual pending
- Guardrail: This audit checks whether optical radius-squared front proxies behave like calibrated diffusion estimates. Publication diffusion claims require manual QC plus positive, estimator-consistent, low-drift, threshold-robust slopes; otherwise values remain apparent optical-front proxies.

## Particle Trace Physics Audit

- Cycle rows/range: 89 rows, cycles 2.000-158.000
- Drop cycles: any=4, synchronized 2+=2, synchronized 3+=2
- Trace-state clustering: k=2, silhouette=0.266
- Future-drop classifier future_any_drop_within_8cycles: folds=5, AUC 0.883, balanced accuracy 0.648
- Future-drop classifier future_sync2_drop_within_8cycles: folds=3, AUC 0.827, balanced accuracy 0.828
- Future-drop null future_any_drop_within_8cycles: observed AUC 0.883, null p95 0.679, empirical p=0.002
- Future-drop null future_sync2_drop_within_8cycles: observed AUC 0.827, null p95 0.744, empirical p=0.010
- Event feature any_abrupt_drop mean_delta_prev: median pos-neg -0.116, p=1.029e-05
- Event feature any_abrupt_drop max_abs_delta_prev: median pos-neg 0.162, p=0.001
- Event feature any_abrupt_drop mean_abs_delta_prev: median pos-neg 0.089, p=0.003
- Event feature any_abrupt_drop particle_norm_mean: median pos-neg -0.096, p=0.059
- Event feature any_abrupt_drop delta_std_across_particles: median pos-neg 0.049, p=0.067
- Event feature any_abrupt_drop frames_percentile: median pos-neg -0.298, p=0.255
- Trace/echem correlation V_max vs particle_norm_mean: rho 0.626, p=4.144e-10
- Trace/echem correlation V_max vs particle_norm_range: rho 0.475, p=7.293e-06
- Trace/echem correlation V_max vs particle_norm_std: rho 0.459, p=1.656e-05
- Trace/echem correlation coulombic_efficiency_pct vs particle_norm_mean: rho -0.456, p=1.909e-05
- Trace/echem correlation capacity_mAh vs particle_norm_mean: rho 0.454, p=2.081e-05
- Trace/echem correlation V_max vs mean_abs_delta_prev: rho -0.250, p=0.025
- Guardrail: This audit uses the larger four-particle cycle intensity table, not video ROI masks. It tests cycle-level photometry/echem physics hypotheses and early-warning signals, but cannot localize phase fronts or validate diffusion without ROI/video QC.

## Particle Event Precursor Atlas

- Anchors: 4 event anchors, 17 candidate controls, 24 matched controls
- Precursor pre8_to_pre5 capacity_mAh min_value: event-control -0.024, p=0.002
- Precursor pre16_to_pre9 coulombic_efficiency_pct min_value: event-control -1.110, p=0.002
- Precursor pre16_to_pre9 coulombic_efficiency_pct mean_value: event-control -0.421, p=0.002
- Precursor pre16_to_pre9 delta_std_across_particles max_value: event-control 0.040, p=0.007
- Precursor pre16_to_pre9 n_frames slope_per_cycle: event-control -2.175, p=0.012
- Precursor pre16_to_pre9 frames_percentile slope_per_cycle: event-control -0.019, p=0.021
- Precursor pre16_to_pre9 coulombic_efficiency_pct max_value: event-control -0.299, p=0.024
- Precursor pre4_to_pre1 particle_norm_mean min_value: event-control 0.047, p=0.027
- Precursor pre8_to_pre5 mean_delta_prev max_value: event-control -0.017, p=0.030
- Precursor pre16_to_pre9 V_max slope_per_cycle: event-control -5.163e-04, p=0.037
- All-window event_cycle max_abs_delta_prev max_value: event-control 0.136, p=0.002
- All-window event_cycle max_abs_delta_prev mean_value: event-control 0.136, p=0.002
- All-window event_cycle max_abs_delta_prev median_value: event-control 0.136, p=0.002
- All-window event_cycle max_abs_delta_prev min_value: event-control 0.136, p=0.002
- All-window pre8_to_pre5 capacity_mAh min_value: event-control -0.024, p=0.002
- Guardrail: Precursor windows are aligned to four detected abrupt-drop cycles and matched non-event anchors from the four-particle cycle table. Results show cycle-level trace precursors for review and hypothesis generation, not localized phase-front motion or calibrated diffusion.

## ROI Trace Fusion Audit

- Fusion rows/predictors: 52 ROI rows, 100 lagged trace/context predictors
- Focused residual association trace_lag2_particle_norm_range vs phase_slope_positive_fraction_protocol_residual: rho=0.725, p=2.688e-07, n=38
- Focused residual association trace_lag2_particle_norm_std vs phase_slope_positive_fraction_protocol_residual: rho=0.725, p=2.688e-07, n=38
- Focused residual association trace_lag4_delta_std_across_particles vs phase_slope_positive_fraction_protocol_residual: rho=-0.705, p=7.719e-07, n=38
- Focused residual association lag16_trace_predprob_future_any_drop_within_8cycles vs phase_slope_positive_fraction_protocol_residual: rho=-0.652, p=9.324e-06, n=38
- Focused residual association trace_lag4_particle_norm_cv vs phase_slope_positive_fraction_protocol_residual: rho=0.652, p=9.324e-06, n=38
- Focused residual association trace_lag2_particle_norm_cv vs phase_slope_positive_fraction_protocol_residual: rho=0.641, p=1.450e-05, n=38
- Focused residual association trace_lag16_frames_percentile vs phase_slope_positive_fraction_protocol_residual: rho=0.600, p=6.805e-05, n=38
- Focused residual association trace_lag16_max_abs_delta_prev vs phase_slope_positive_fraction_protocol_residual: rho=0.600, p=6.805e-05, n=38
- Event-enriched mode precursor test trace_lag16_delta_std_across_particles: median diff -0.007, p=0.005
- Event-enriched mode precursor test trace_lag8_frames_percentile: median diff 0.185, p=0.006
- Event-enriched mode precursor test trace_lag8_n_frames: median diff 17.000, p=0.006
- Event-enriched mode precursor test trace_lag8_future_sync2_drop_within_8cycles: median diff -1.000, p=0.011
- Event-enriched mode precursor test trace_lag16_V_max: median diff 0.002, p=0.013
- Guardrail: Trace lags are cycle-level four-particle/echem summaries attached to selected ROI rows by cycle number. Associations are useful for linking global precursor state to ROI/front outcomes, but rows are not independent within cycle and this does not prove localized causality.

## ROI Trace Fusion Cycle Null

- Cycle-collapsed points: 11 from 52 ROI rows
- Event-reference cycles: 4
- Predictors/permutations: 68 / 10000
- Cycle-collapsed trace_lag8_frames_percentile vs mode_review_priority: rho 0.834, empirical p=0.002, n=11
- Cycle-collapsed trace_lag8_n_frames vs mode_review_priority: rho 0.834, empirical p=0.003, n=11
- Cycle-collapsed lag8_trace_predprob_future_any_drop_within_8cycles vs mode_review_priority: rho -0.827, empirical p=0.003, n=11
- Cycle-collapsed trace_lag16_frames_percentile vs mode_review_priority: rho 0.813, empirical p=0.003, n=11
- Cycle-collapsed trace_lag16_n_frames vs mode_review_priority: rho 0.813, empirical p=0.005, n=11
- Cycle-collapsed lag4_trace_predprob_future_any_drop_within_8cycles vs q70_logistic_k_per_s: rho -0.800, empirical p=0.005, n=11
- Reference-centered trace_lag8_particle_norm_mean vs phase_slope_median_per_s_protocol_residual: rho 0.918, empirical p=2.000e-04, n=11
- Reference-centered trace_lag8_n_frames vs mode_review_priority: rho 0.881, empirical p=8.999e-04, n=11
- Reference-centered trace_lag8_frames_percentile vs mode_review_priority: rho 0.858, empirical p=0.002, n=11
- Reference-centered lag16_trace_predprob_future_any_drop_within_8cycles vs phase_slope_positive_fraction_protocol_residual: rho -0.809, empirical p=0.003, n=11
- Guardrail: This audit collapses repeated ROI rows to one median point per cycle before testing trace/front associations. It is deliberately conservative for the 52-ROI cohort; surviving tests are stronger evidence, while lost tests indicate cycle-clustering sensitivity.

## Precursor-Informed ROI Review

- Review candidates: 47
- Event/control candidates: 23 / 24
- Review tiers: {'high': 12, 'medium': 18, 'routine': 17}
- cycle156_rank7_obj27 (event, cycle 156): score 5.527, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle60_rank5_obj18 (event, cycle 60): score 5.094, tier high, reason event-cycle ROI;high precursor-context cycle;large front-direction residual;kinetic proxy available
- cycle156_rank6_obj3 (event, cycle 156): score 5.086, tier high, reason event-cycle ROI;high precursor-context cycle;kinetic proxy available
- cycle156_rank5_obj4 (event, cycle 156): score 5.054, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle156_rank2_obj2 (event, cycle 156): score 4.926, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle156_rank1_obj1 (event, cycle 156): score 4.834, tier high, reason event-cycle ROI;high precursor-context cycle;kinetic proxy available
- cycle156_rank8_obj10 (event, cycle 156): score 4.741, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;kinetic proxy available
- cycle60_rank3_obj9 (event, cycle 60): score 4.632, tier high, reason event-cycle ROI;high precursor-context cycle;event-enriched residual mode;large front-direction residual;kinetic proxy available
- Guardrail: This is a review-prioritization manifest. It combines automatic ROI/front/mode/kinetic descriptors with cycle-level precursor context to decide what to inspect first; it does not assign manual QC labels or validate diffusion/front claims.

## Precursor Visual Review Bundle

- Ranked candidates included: 12
- Candidates with visual assets: 12
- Contact sheet: /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_review_visual_bundle/top_candidate_contact_sheet.png
- Visual rank 1 cycle156_rank7_obj27 (event, cycle 156): score 5.527, tier high
- Visual rank 2 cycle60_rank5_obj18 (event, cycle 60): score 5.094, tier high
- Visual rank 3 cycle156_rank6_obj3 (event, cycle 156): score 5.086, tier high
- Visual rank 4 cycle156_rank5_obj4 (event, cycle 156): score 5.054, tier high
- Visual rank 5 cycle156_rank2_obj2 (event, cycle 156): score 4.926, tier high
- Visual rank 6 cycle156_rank1_obj1 (event, cycle 156): score 4.834, tier high
- Visual rank 7 cycle156_rank8_obj10 (event, cycle 156): score 4.741, tier high
- Visual rank 8 cycle60_rank3_obj9 (event, cycle 60): score 4.632, tier high
- Guardrail: This bundle copies existing automatic QC/preview assets for manual inspection. It does not create labels, adjudicate particle identity, or validate diffusion/front interpretability.

## Within-Cycle Echem Shape Audit

- Echem shape cycles/features: 81 / 48
- ROI rows joined: 52
- ROI shape correlation shape_V_q95 vs mode_review_priority: rho=-0.864, p=2.718e-12, n=38
- ROI shape correlation neg_dq_abs_peak_frac vs mode_review_priority: rho=0.841, p=3.806e-11, n=38
- ROI shape correlation pos_dq_abs_midV_frac vs mode_review_priority: rho=0.840, p=4.174e-11, n=38
- ROI shape correlation all_dq_abs_entropy vs mode_review_priority: rho=-0.839, p=4.679e-11, n=38
- ROI shape correlation pos_dq_abs_peak_frac vs mode_review_priority: rho=0.839, p=4.679e-11, n=38
- ROI shape correlation pos_dq_abs_highV_frac vs mode_review_priority: rho=-0.839, p=4.787e-11, n=38
- ROI shape correlation pos_dq_abs_entropy vs mode_review_priority: rho=-0.801, p=1.607e-09, n=38
- ROI shape correlation all_dq_abs_peak_frac vs mode_review_priority: rho=0.790, p=3.651e-09, n=38
- Cycle shape correlation shape_charge_mAh_neg_abs vs capacity_mAh: rho=1.000, p=1.253e-149, n=81
- Cycle shape correlation neg_dq_abs_total_mAh vs capacity_mAh: rho=1.000, p=7.534e-126, n=81
- Cycle shape correlation echem_shape_duration_s vs capacity_mAh: rho=0.985, p=1.680e-61, n=81
- Cycle shape correlation shape_charge_mAh_abs vs capacity_mAh: rho=0.985, p=1.680e-61, n=81
- Cycle shape correlation shape_charge_mAh_signed vs coulombic_efficiency_pct: rho=-0.968, p=2.136e-49, n=81
- Cycle shape correlation shape_dIdt_slope vs capacity_mAh: rho=0.934, p=5.846e-37, n=81
- Event shape test neg_dq_abs_peak_voltage: median event-control 0.000, p=0.230
- Event shape test shape_dVdt_abs_p95: median event-control 0.003, p=0.255
- Event shape test neg_dq_abs_total_mAh: median event-control -0.010, p=0.406
- Event shape test shape_charge_mAh_neg_abs: median event-control -0.010, p=0.406
- Guardrail: Within-cycle echem shape features are computed from raw time/potential/current rows for observed particle/ROI cycles. dQ/dV terms are proxy descriptors from current-time integration over voltage bins, not calibrated electrochemical capacity analysis.

## Echem-Shape-Conditioned ROI/Front Effects

- Rows event/control: 24 / 28
- Shape PCA: 45 features, 6 PCs, total explained variance 0.997
- Shape-residual classifier: ROC-AUC 0.469, balanced accuracy 0.448
- Shape-conditioned phase_slope_positive_fraction_protocol_residual: event-control residual median 0.031, p=0.004
- Shape-conditioned high_fraction_delta_protocol_residual: event-control residual median 0.002, p=0.008
- Shape-conditioned q70_transformed_fraction_delta: event-control residual median 0.005, p=0.010
- Shape-conditioned low_fraction_delta_protocol_residual: event-control residual median -0.007, p=0.016
- Shape-conditioned q80_transformed_fraction_delta: event-control residual median 0.003, p=0.021
- Shape-conditioned dmd_minus_persistence_mse_protocol_residual: event-control residual median 4.775e-04, p=0.041
- Shape-conditioned mode_review_priority: event-control residual median 0.135, p=0.300
- Shape-conditioned threshold_robust_phase_score_protocol_residual: event-control residual median -0.013, p=0.393
- Shape context fit mode_review_priority: variance explained 0.839, n=52
- Shape context fit first_last_corr_protocol_residual: variance explained 0.583, n=52
- Shape context fit q60_logistic_k_per_s: variance explained 0.544, n=48
- Shape context fit q70_logistic_k_per_s: variance explained 0.515, n=50
- Shape context fit cumulative_abs_norm_change_protocol_residual: variance explained 0.423, n=52
- Shape context fit latent_net_displacement_protocol_residual: variance explained 0.413, n=52
- Shape PC correlation echem_shape_pc3 vs mode_review_priority: rho=-0.707, p=4.516e-09, n=52
- Shape PC correlation echem_shape_pc6 vs phase_slope_positive_fraction_protocol_residual: rho=0.586, p=4.918e-06, n=52
- Shape PC correlation echem_shape_pc6 vs high_fraction_delta_protocol_residual: rho=0.516, p=9.166e-05, n=52
- Shape PC correlation echem_shape_pc4 vs cumulative_abs_norm_change_protocol_residual: rho=-0.490, p=2.252e-04, n=52
- Guardrail: Echem-shape conditioning uses low-dimensional PCA/ridge covariates on a small automatically selected ROI cohort. Surviving residuals are evidence that optical/front signals are not fully explained by measured within-cycle echem shape, but they are not causal proof or calibrated transport constants.

## Probabilistic Rollout Calibration

- Frame rows / ROI-method rows: 4992 / 156
- ROI / event-reference cycles: 52 / 4
- Near-transition frame fraction: 0.264
- 95% empirical coverage low_rank_dmd all: global 0.921, local 0.921, n=1664
- 95% empirical coverage low_rank_dmd event_roi: global 0.871, local 0.871, n=768
- 95% empirical coverage low_rank_dmd near_transition: global 0.909, local 0.941, n=440
- 95% empirical coverage persistence all: global 0.955, local 0.955, n=1664
- 95% empirical coverage persistence event_roi: global 0.945, local 0.945, n=768
- 95% empirical coverage persistence near_transition: global 0.975, local 0.952, n=440
- 95% empirical coverage velocity all: global 0.942, local 0.942, n=1664
- 95% empirical coverage velocity event_roi: global 0.917, local 0.917, n=768
- 95% empirical coverage velocity near_transition: global 0.952, local 0.959, n=440
- Residual test low_rank_dmd late_minus_early_rollout: median diff 0.006, p=8.875e-29, n=572/572
- Residual test low_rank_dmd event_minus_control: median diff 0.004, p=5.586e-15, n=768/896
- Residual test velocity event_minus_control: median diff 1.569e-04, p=0.110, n=768/896
- Residual test persistence event_minus_control: median diff 4.067e-05, p=0.161, n=768/896
- Residual test velocity late_minus_early_rollout: median diff -5.112e-05, p=0.516, n=572/572
- Calibration/physics link low_rank_dmd mae_max vs cumulative_abs_first_last_first: rho=0.631, p=5.263e-07, n=52
- Calibration/physics link low_rank_dmd mae_mean vs cumulative_abs_first_last_first: rho=0.600, p=2.617e-06, n=52
- Calibration/physics link low_rank_dmd q90_undercoverage_rate vs cumulative_abs_first_last_first: rho=0.586, p=5.079e-06, n=52
- Calibration/physics link persistence mae_max vs q70_transformed_fraction_delta_first: rho=0.569, p=1.078e-05, n=52
- Calibration/physics link low_rank_dmd mae_max vs first_last_corr_first: rho=-0.542, p=3.274e-05, n=52
- Undercoverage priority cycle156_rank7_obj27 low_rank_dmd: q90 undercoverage 1.000, priority 3.843, role event
- Undercoverage priority cycle156_rank5_obj4 low_rank_dmd: q90 undercoverage 0.344, priority 3.699, role event
- Undercoverage priority cycle156_rank7_obj27 velocity: q90 undercoverage 1.000, priority 3.606, role event
- Undercoverage priority cycle156_rank8_obj10 low_rank_dmd: q90 undercoverage 0.125, priority 3.516, role event
- Undercoverage priority cycle156_rank7_obj27 persistence: q90 undercoverage 1.000, priority 3.471, role event
- Undercoverage priority cycle156_rank2_obj2 low_rank_dmd: q90 undercoverage 0.781, priority 3.401, role event
- Guardrail: Empirical residual quantiles are a calibration audit for ROI-only rollout baselines. They are not a generative uncertainty model, not calibrated diffusion, and not manual QC.

## Masked Residual State Transfer Warning

- Anchor/full cycles/permutations: 11 / 89 / 5000
- Signature features: 8
- Anchor leave-one-cycle transfer: rho=-0.155, p=0.650, n=11
- Signature feature low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_max_delta_prev_observed_roi_cycle: median positive-negative 12.290, permutation p=0.013
- Signature feature low_rank_dmd_particle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.009, permutation p=0.014
- Signature feature low_rank_dmd_nonparticle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.002, permutation p=0.024
- Signature feature low_rank_dmd_particle_mse_mean_median_delta_prev_observed_roi_cycle: median positive-negative 0.005, permutation p=0.043
- Signature feature low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_max: median positive-negative 8.364, permutation p=0.045
- Signature feature low_rank_dmd_nonparticle_mse_mean_median_delta_prev_observed_roi_cycle: median positive-negative 7.885e-04, permutation p=0.050
- Transfer target future_any_drop_within_16cycles transferred_masked_residual_signature: median positive-negative 0.631, AUC 0.676, permutation p=3.999e-04, n=40/49
- Transfer target any_abrupt_drop mean_abs_delta_prev: median positive-negative 0.089, AUC 0.946, permutation p=0.001, n=4/84
- Transfer target future_any_drop_within_8cycles transferred_masked_residual_signature: median positive-negative 0.648, AUC 0.708, permutation p=0.004, n=20/69
- Transfer target any_abrupt_drop state_step_norm: median positive-negative 5.544, AUC 0.887, permutation p=0.013, n=4/84
- Transfer target future_sync2_drop_within_8cycles frames_percentile: median positive-negative -0.362, AUC 0.856, permutation p=0.014, n=8/81
- Transfer target future_any_drop_within_8cycles cycle_state_pc2: median positive-negative 0.730, AUC 0.772, permutation p=0.016, n=20/69
- Transfer target future_any_drop_within_16cycles mean_abs_delta_prev: median positive-negative -0.011, AUC 0.652, permutation p=0.020, n=40/48
- Transfer target future_any_drop_within_16cycles state_step_norm: median positive-negative -0.813, AUC 0.668, permutation p=0.021, n=40/48
- Transfer/context link cycle_state_pc6: rho=0.541, p=4.330e-08, n=89
- Transfer/context link axis_step: rho=0.522, p=1.779e-07, n=88
- Transfer/context link coulombic_efficiency_pct: rho=-0.499, p=2.162e-06, n=81
- Transfer/context link cycle_state_pc5: rho=-0.399, p=1.098e-04, n=89
- Transfer/context link cycle_state_pc2: rho=0.354, p=6.631e-04, n=89
- Transfer/context link frames_percentile: rho=0.319, p=0.002, n=89
- Transfer-ranked cycle 150: score 40.972, future8=1, future16=1, abrupt=0
- Transfer-ranked cycle 146: score 3.217, future8=0, future16=1, abrupt=0
- Transfer-ranked cycle 156: score 1.612, future8=0, future16=0, abrupt=1
- Transfer-ranked cycle 151: score 1.439, future8=1, future16=1, abrupt=0
- Transfer-ranked cycle 152: score 1.188, future8=1, future16=1, abrupt=0
- Transfer-ranked cycle 154: score 1.010, future8=1, future16=1, abrupt=0
- Guardrail: Masked residual signature is learned from 11 video-backed ROI cycles and transferred through the cycle-state/echem manifold to 89 cycles; use as hypothesis-ranking evidence, not a deployable warning model or direct video residual measurement for unexported cycles.

## Transfer-Ranked ROI Reconstruction And Masked Rollout

- Sampled cycles/reconstructed candidates/ROI rows: 12 / 960 / 48
- Exported ROI sequences: 48 at 96 frames per ROI
- Masked rollout ROI/frame rows: 48 / 4608
- Best method counts inside particle masks: {'persistence': 48}
- Transfer sampled cycle 150 rank 1: score 40.972, future8=1, future16=1, candidates=80
- Transfer sampled cycle 146 rank 2: score 3.217, future8=0, future16=1, candidates=80
- Transfer sampled cycle 156 rank 3: score 1.612, future8=0, future16=0, candidates=80
- Transfer sampled cycle 151 rank 4: score 1.439, future8=1, future16=1, candidates=80
- Transfer sampled cycle 152 rank 5: score 1.188, future8=1, future16=1, candidates=80
- Transfer sampled cycle 154 rank 6: score 1.010, future8=1, future16=1, candidates=80
- Transfer sampled cycle 153 rank 7: score 0.871, future8=1, future16=1, candidates=80
- Transfer sampled cycle 40 rank 8: score 0.784, future8=0, future16=0, candidates=80
- Transfer sequence cycle 116.0: mean ROI delta -10.252, norm delta -4.391e-04, n=4
- Transfer sequence cycle 146.0: mean ROI delta -3.926, norm delta -2.887e-04, n=4
- Transfer sequence cycle 147.0: mean ROI delta 6.302, norm delta 9.638e-04, n=4
- Transfer sequence cycle 148.0: mean ROI delta 1.052, norm delta 5.269e-04, n=4
- Transfer sequence cycle 150.0: mean ROI delta -4.974, norm delta -1.556e-04, n=4
- Transfer sequence cycle 151.0: mean ROI delta 2.846, norm delta 6.862e-05, n=4
- Transfer sequence cycle 152.0: mean ROI delta -5.496, norm delta -3.611e-04, n=4
- Transfer sequence cycle 153.0: mean ROI delta -13.522, norm delta -7.464e-04, n=4
- Transfer-ranked persistence: particle-MSE median 1.847e-04, nonparticle-MSE median 8.412e-05, particle/nonparticle ratio median 2.155
- Transfer-ranked velocity: particle-MSE median 4.602e-04, nonparticle-MSE median 2.397e-04, particle/nonparticle ratio median 1.827
- Transfer-ranked low_rank_dmd: particle-MSE median 0.006, nonparticle-MSE median 6.980e-04, particle/nonparticle ratio median 8.308
- Transfer-ranked difficult ROI cycle151_rank4_obj3 low_rank_dmd cycle 151: particle MSE 0.014, particle/nonparticle ratio 8.196
- Transfer-ranked difficult ROI cycle153_rank7_obj4 low_rank_dmd cycle 153: particle MSE 0.013, particle/nonparticle ratio 7.656
- Transfer-ranked difficult ROI cycle156_rank3_obj2 low_rank_dmd cycle 156: particle MSE 0.013, particle/nonparticle ratio 4.500
- Transfer-ranked difficult ROI cycle154_rank6_obj2 low_rank_dmd cycle 154: particle MSE 0.012, particle/nonparticle ratio 8.299
- Transfer-ranked difficult ROI cycle151_rank4_obj2 low_rank_dmd cycle 151: particle MSE 0.012, particle/nonparticle ratio 9.524
- Transfer-ranked difficult ROI cycle153_rank7_obj2 low_rank_dmd cycle 153: particle MSE 0.011, particle/nonparticle ratio 12.454
- Transfer-ranked ROI candidate cycle 150 obj 4: validation score 32.064, mean abs z 29.677, future8=1
- Transfer-ranked ROI candidate cycle 150 obj 1: validation score 31.252, mean abs z 25.298, future8=1
- Transfer-ranked ROI candidate cycle 150 obj 2: validation score 30.291, mean abs z 20.694, future8=1
- Transfer-ranked ROI candidate cycle 150 obj 3: validation score 30.065, mean abs z 19.134, future8=1
- Transfer-ranked ROI candidate cycle 146 obj 4: validation score 25.085, mean abs z 13.691, future8=0
- Transfer-ranked ROI candidate cycle 146 obj 3: validation score 24.258, mean abs z 20.728, future8=0
- Guardrail: Automatic transfer-ranked ROI candidates reconstructed from sampled HDF5 cycle segments; compatible with particle-region sequence export, but not manual annotations or validated fronts. Held-out rollout errors are scored inside automatic history-aware particle masks; this is not manual segmentation or a new learned video model.

## Transfer-Ranked Front Physics Audit

- Threshold-front ROI/cycles/quantiles: 48 / 12 / 7
- Front-physics target positives: {'any_abrupt_drop': 8, 'future_any_drop_within_16cycles': 36, 'future_any_drop_within_8cycles': 28}
- Front target future_any_drop_within_8cycles radius2_slope_positive_fraction: median positive-negative 0.857, AUC 0.781, permutation p=3.999e-04, n=28/20
- Front target future_any_drop_within_8cycles radius2_slope_median_px2_per_s: median positive-negative 0.002, AUC 0.846, permutation p=5.999e-04, n=28/20
- Front target future_any_drop_within_8cycles q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 0.002, AUC 0.827, permutation p=7.998e-04, n=28/20
- Front target future_any_drop_within_8cycles diffusion_proxy_median_um2_per_s: median positive-negative 5.147e-06, AUC 0.846, permutation p=9.998e-04, n=28/20
- Front target any_abrupt_drop roi_norm_mean_delta_last_minus_first: median positive-negative 0.008, AUC 0.684, permutation p=9.998e-04, n=8/40
- Front target future_any_drop_within_8cycles persistence_particle_mse_mean: median positive-negative 6.220e-04, AUC 0.945, permutation p=0.002, n=28/20
- Front target future_any_drop_within_8cycles q70_phase_slope_bootstrap_p50: median positive-negative -1.709e-06, AUC 0.759, permutation p=0.003, n=28/20
- Front target future_any_drop_within_8cycles phase_slope_median_per_s: median positive-negative -1.805e-06, AUC 0.770, permutation p=0.004, n=28/20
- Front/residual link roi_norm_mean_delta_last_minus_first vs roi_mean_delta_last_minus_first: rho=0.965, p=1.998e-28, n=48
- Front/residual link persistence_particle_mse_mean vs future_any_drop_within_8cycles: rho=0.760, p=3.886e-10, n=48
- Front/residual link radius2_slope_median_px2_per_s vs persistence_particle_mse_mean: rho=0.724, p=6.169e-09, n=48
- Front/residual link diffusion_proxy_median_um2_per_s vs persistence_particle_mse_mean: rho=0.724, p=6.169e-09, n=48
- Front/residual link q70_radius2_slope_bootstrap_p50_px2_per_s vs persistence_particle_mse_mean: rho=0.703, p=2.481e-08, n=48
- Front/residual link q70_radius2_slope_bootstrap_p50_px2_per_s vs low_rank_dmd_particle_mse_mean: rho=0.679, p=1.156e-07, n=48
- Cycle-collapsed front target future_any_drop_within_8cycles persistence_particle_mse_mean: median positive-negative 5.592e-04, AUC 0.971, permutation p=0.058, n=7/5
- Cycle-collapsed front target future_any_drop_within_16cycles diffusion_proxy_abs_median_um2_per_s: median positive-negative 3.253e-06, AUC 0.926, permutation p=0.063, n=9/3
- Cycle-collapsed front target future_any_drop_within_8cycles threshold_robust_diffusion_score: median positive-negative -0.294, AUC 0.743, permutation p=0.065, n=7/5
- Cycle-collapsed front target future_any_drop_within_8cycles q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 0.003, AUC 0.886, permutation p=0.086, n=7/5
- Cycle-collapsed front target future_any_drop_within_8cycles radius2_slope_median_px2_per_s: median positive-negative 0.003, AUC 0.886, permutation p=0.089, n=7/5
- Cycle-collapsed front target future_any_drop_within_8cycles diffusion_proxy_median_um2_per_s: median positive-negative 6.872e-06, AUC 0.886, permutation p=0.092, n=7/5
- Transfer front cycle 40: phase slope 1.580e-06, apparent D median 1.685e-06, phase score 0.661, n=4
- Transfer front cycle 116: phase slope 3.254e-06, apparent D median -1.678e-06, phase score 0.969, n=4
- Transfer front cycle 146: phase slope 2.641e-06, apparent D median -5.688e-06, phase score 0.609, n=4
- Transfer front cycle 147: phase slope 3.909e-06, apparent D median -6.335e-06, phase score 0.729, n=4
- Transfer front cycle 148: phase slope 3.377e-06, apparent D median -4.354e-06, phase score 0.812, n=4
- Transfer front cycle 150: phase slope 2.471e-06, apparent D median 1.369e-06, phase score 1.010, n=4
- Transfer front cycle 151: phase slope 1.226e-06, apparent D median 4.742e-06, phase score 0.398, n=4
- Transfer front cycle 152: phase slope -1.547e-07, apparent D median 5.194e-06, phase score -0.047, n=4
- Front-physics review ROI cycle151_rank4_obj2: score 3.479, future8=1, phase score 1.260, apparent D 9.838e-06, DMD particle MSE 0.012
- Front-physics review ROI cycle150_rank1_obj3: score 3.104, future8=1, phase score 1.260, apparent D 5.160e-06, DMD particle MSE 0.006
- Front-physics review ROI cycle152_rank5_obj2: score 3.104, future8=1, phase score 0.885, apparent D 1.171e-05, DMD particle MSE 0.011
- Front-physics review ROI cycle156_rank3_obj2: score 2.990, future8=0, phase score 1.198, apparent D 1.925e-06, DMD particle MSE 0.013
- Front-physics review ROI cycle151_rank4_obj3: score 2.833, future8=1, phase score 0.760, apparent D 3.555e-06, DMD particle MSE 0.014
- Front-physics review ROI cycle154_rank6_obj2: score 2.813, future8=1, phase score 0.531, apparent D 1.407e-05, DMD particle MSE 0.012
- Front-physics review ROI cycle146_rank2_obj1: score 2.771, future8=0, phase score 0.885, apparent D -5.805e-06, DMD particle MSE 0.006
- Front-physics review ROI cycle153_rank7_obj4: score 2.760, future8=1, phase score 0.594, apparent D 1.101e-05, DMD particle MSE 0.013
- Guardrail: Transfer-ranked front descriptors are automatic optical phase/radius proxies from ROI crops. Diffusion-like values are apparent front-motion descriptors, not calibrated transport coefficients, and the cohort is warning-ranked rather than an event/control design.

## Transfer-Ranked Residual Transition Timing

- Phase-kinetic ROI/timing rows/permutations: 48 / 144 / 5000
- Timing target positives: {'any_abrupt_drop': 8, 'future_any_drop_within_16cycles': 36, 'future_any_drop_within_8cycles': 28}
- Transfer timing alignment low_rank_dmd weighted_center_distance_to_transition_frac: median distance 0.087, null mean 0.250, empirical p=2.000e-04, n=48
- Transfer timing alignment persistence weighted_center_distance_to_transition_frac: median distance 0.156, null mean 0.252, empirical p=0.004, n=48
- Transfer timing alignment velocity weighted_center_distance_to_transition_frac: median distance 0.170, null mean 0.251, empirical p=0.012, n=48
- Transfer timing alignment low_rank_dmd peak_distance_to_transition_frac: median distance 0.421, null mean 0.439, empirical p=0.400, n=48
- Transfer timing alignment persistence peak_distance_to_transition_frac: median distance 0.318, null mean 0.329, empirical p=0.441, n=48
- Transfer timing alignment velocity peak_distance_to_transition_frac: median distance 0.324, null mean 0.312, empirical p=0.591, n=48
- Transfer timing target future_any_drop_within_8cycles persistence particle_to_nonparticle_mse_ratio_median: median positive-negative 0.648, AUC 0.832, permutation p=3.999e-04, n=28/20
- Transfer timing target future_any_drop_within_16cycles persistence particle_to_nonparticle_mse_ratio_median: median positive-negative 0.645, AUC 0.808, permutation p=7.998e-04, n=36/12
- Transfer timing target future_any_drop_within_8cycles persistence residual_peak_particle_mse: median positive-negative 0.010, AUC 0.879, permutation p=0.002, n=28/20
- Transfer timing target future_any_drop_within_8cycles velocity residual_peak_particle_mse: median positive-negative 0.012, AUC 0.923, permutation p=0.002, n=28/20
- Transfer timing target any_abrupt_drop low_rank_dmd near_transition_residual_fraction: median positive-negative -0.073, AUC 0.772, permutation p=0.002, n=8/40
- Transfer timing target future_any_drop_within_16cycles low_rank_dmd weighted_center_distance_to_transition_frac: median positive-negative -0.199, AUC 0.778, permutation p=0.003, n=36/12
- Transfer timing target future_any_drop_within_8cycles low_rank_dmd weighted_center_distance_to_transition_frac: median positive-negative -0.116, AUC 0.645, permutation p=0.003, n=28/20
- Transfer timing target future_any_drop_within_16cycles low_rank_dmd peak_distance_to_transition_frac: median positive-negative -0.146, AUC 0.736, permutation p=0.004, n=36/12
- Transfer timing/target link velocity residual_peak_particle_mse vs future_any_drop_within_8cycles: rho=0.723, p=6.528e-09, n=48
- Transfer timing/target link persistence residual_peak_particle_mse vs future_any_drop_within_8cycles: rho=0.647, p=6.850e-07, n=48
- Transfer timing/target link velocity residual_peak_particle_mse vs future_any_drop_within_16cycles: rho=0.629, p=1.717e-06, n=48
- Transfer timing/target link velocity particle_to_nonparticle_mse_ratio_median vs future_any_drop_within_16cycles: rho=0.583, p=1.346e-05, n=48
- Transfer timing/target link persistence particle_to_nonparticle_mse_ratio_median vs future_any_drop_within_8cycles: rho=0.567, p=2.610e-05, n=48
- Transfer timing/target link persistence residual_peak_particle_mse vs future_any_drop_within_16cycles: rho=0.542, p=6.981e-05, n=48
- Near-transition transfer ROI cycle152_rank5_obj3 persistence: near residual fraction 0.700, peak MSE 0.011, future8=1
- Near-transition transfer ROI cycle152_rank5_obj4 persistence: near residual fraction 0.693, peak MSE 0.010, future8=1
- Near-transition transfer ROI cycle152_rank5_obj1 persistence: near residual fraction 0.660, peak MSE 0.017, future8=1
- Near-transition transfer ROI cycle152_rank5_obj4 velocity: near residual fraction 0.528, peak MSE 0.010, future8=1
- Near-transition transfer ROI cycle152_rank5_obj3 velocity: near residual fraction 0.504, peak MSE 0.010, future8=1
- Near-transition transfer ROI cycle156_rank3_obj1 persistence: near residual fraction 0.486, peak MSE 7.274e-04, future8=0
- Guardrail: Transfer-ranked transition timing uses automatic phase-fraction kinetics and masked rollout residuals from warning-ranked ROI crops. It tests temporal alignment and future-label association only; it is not manual phase-boundary annotation or calibrated transport.

## Multi-Cohort Future-Drop Weak Model

- ROI rows selected/transfer/features: 11 / 48 / 44
- RF trees/permutation null: 80 / 40
- Multi-cohort OOF logistic_l2: AUC 0.692, AP 0.777, scored 58 rows over 13/13 folds
- Multi-cohort OOF random_forest: AUC 0.886, AP 0.914, scored 58 rows over 13/13 folds
- Multi-cohort permutation null random_forest: observed AUC 0.886, null mean 0.441, p95 0.591, empirical p=0.024
- Multi-cohort feature velocity_particle_mse_fraction_of_full_mean: median positive-negative 0.599, oriented AUC 0.968, p=1.012e-09, n=28/30
- Multi-cohort feature velocity_particle_to_nonparticle_mse_ratio_mean: median positive-negative 1.573, oriented AUC 0.964, p=1.354e-09, n=28/30
- Multi-cohort feature velocity_particle_mse_mean: median positive-negative 0.001, oriented AUC 0.964, p=1.354e-09, n=28/30
- Multi-cohort feature persistence_particle_mse_mean: median positive-negative 6.253e-04, oriented AUC 0.940, p=8.944e-09, n=28/30
- Multi-cohort feature persistence_particle_to_nonparticle_mse_ratio_mean: median positive-negative 1.431, oriented AUC 0.920, p=4.132e-08, n=28/30
- Multi-cohort feature persistence_particle_mse_fraction_of_full_mean: median positive-negative 0.629, oriented AUC 0.901, p=1.640e-07, n=28/30
- Multi-cohort feature phase_slope_positive_fraction: median positive-negative -0.071, oriented AUC 0.750, p=1.377e-05, n=28/30
- Multi-cohort feature radius2_slope_median_px2_per_s: median positive-negative 0.002, oriented AUC 0.832, p=1.467e-05, n=28/30
- Multi-cohort importance logistic_l2 velocity_particle_mse_fraction_of_full_mean: 1.301
- Multi-cohort importance logistic_l2 persistence_particle_mse_fraction_of_full_mean: 0.931
- Multi-cohort importance logistic_l2 velocity_particle_to_nonparticle_mse_ratio_mean: 0.719
- Multi-cohort importance logistic_l2 transferred_masked_residual_signature: 0.703
- Multi-cohort importance logistic_l2 stage_drift_xy_sampled: 0.651
- Multi-cohort importance logistic_l2 object_mean_abs_z: 0.588
- Multi-cohort importance logistic_l2 persistence_particle_to_nonparticle_mse_ratio_mean: 0.527
- Multi-cohort importance logistic_l2 roi_mean_delta_last_minus_first: 0.498
- Leave-cohort selected -> transfer_ranked logistic_l2: status missing_class, train positives/negatives 0/10, test positives/negatives 28/20
- Leave-cohort selected -> transfer_ranked random_forest: status missing_class, train positives/negatives 0/10, test positives/negatives 28/20
- Leave-cohort transfer_ranked -> selected logistic_l2: status missing_class, train positives/negatives 28/20, test positives/negatives 0/10
- Leave-cohort transfer_ranked -> selected random_forest: status missing_class, train positives/negatives 28/20, test positives/negatives 0/10
- Guardrail: Future-drop labels are weak cycle-level labels projected onto ROI rows. Grouped splits reduce cycle leakage, but this is a review-prioritization model, not a deployment-ready degradation detector or manual-QC label set.

## Active-Learning QC Prioritization

- Candidate/visual/immediate rows: 97 / 47 / 4
- Tier counts: {'control_balance_review': 47, 'front_diffusion_guardrail_review': 2, 'immediate_manual_qc': 4, 'model_boundary_case': 8, 'standard_manual_qc': 36}
- Active-QC ROI cycle116_rank7_obj37: rank 1, tier immediate_manual_qc, score 0.463, cycle 116, tags model_boundary_case;control_balance_review;visual_asset_available, model p=0.456
- Active-QC ROI cycle156_rank3_obj2: rank 2, tier model_boundary_case, score 0.462, cycle 156, tags model_boundary_case;control_balance_review, model p=0.529
- Active-QC ROI cycle147_rank11_obj2: rank 3, tier model_boundary_case, score 0.437, cycle 147, tags model_boundary_case;control_balance_review, model p=0.503
- Active-QC ROI cycle86_rank4_obj9: rank 4, tier immediate_manual_qc, score 0.414, cycle 86, tags control_balance_review;visual_asset_available, model p=0.369
- Active-QC ROI cycle151_rank4_obj2: rank 5, tier front_diffusion_guardrail_review, score 0.406, cycle 151, tags high_future_drop_probability;front_diffusion_guardrail_review, model p=0.929
- Active-QC ROI cycle156_rank3_obj1: rank 6, tier model_boundary_case, score 0.396, cycle 156, tags model_boundary_case;control_balance_review, model p=0.428
- Active-QC ROI cycle156_rank3_obj4: rank 7, tier control_balance_review, score 0.391, cycle 156, tags control_balance_review, model p=0.640
- Active-QC ROI cycle146_rank2_obj2: rank 8, tier control_balance_review, score 0.390, cycle 146, tags control_balance_review, model p=0.663
- Active-QC cycle 116: max score 0.463, mean score 0.230, ROI 11, immediate 1
- Active-QC cycle 156: max score 0.462, mean score 0.360, ROI 10, immediate 1
- Active-QC cycle 147: max score 0.437, mean score 0.367, ROI 4, immediate 0
- Active-QC cycle 86: max score 0.414, mean score 0.291, ROI 6, immediate 1
- Active-QC cycle 151: max score 0.406, mean score 0.353, ROI 4, immediate 0
- Active-QC cycle 146: max score 0.390, mean score 0.294, ROI 4, immediate 0
- Active-QC reason control_balance_review: n=54
- Active-QC reason visual_asset_available: n=47
- Active-QC reason high_future_drop_probability: n=19
- Active-QC reason model_boundary_case: n=9
- Active-QC reason general_pending_qc: n=6
- Active-QC reason front_diffusion_guardrail_review: n=4
- Active-QC reason transition_timing_residual_review: n=3
- Guardrail: Active-learning QC ranks pending ROI review candidates from automatic and weak-label evidence only; manual labels, diffusion claims, and deployment decisions remain withheld until human QC.

## Automatic QC Triage Surrogate

- Candidate/manual-status rows: 47 / {'pending': 47}
- Likely/artifact/diffusion-guardrail counts: 6 / 10 / 20
- Auto-QC tier auto_surrogate_likely_interpretable: n=6, median score 0.836, median risk 0.062, diffusion guardrail rate 0.167
- Auto-QC tier standard_review: n=31, median score 0.555, median risk 0.375, diffusion guardrail rate 0.387
- Auto-QC tier auto_surrogate_artifact_risk: n=10, median score 0.463, median risk 0.500, diffusion guardrail rate 0.700
- Auto-QC likely ROI cycle156_rank7_obj27: score 0.919, risk 0.000, cycle 156, role event, reason event_roi;primary_visual_panel;control_balanced_panel;no_auto_flags;high_residual_mode_priority;hard_rollout_mobility
- Auto-QC likely ROI cycle156_rank8_obj10: score 0.843, risk 0.000, cycle 156, role event, reason event_roi;primary_visual_panel;control_balanced_panel;no_auto_flags;hard_rollout_mobility
- Auto-QC likely ROI cycle156_rank2_obj2: score 0.840, risk 0.125, cycle 156, role event, reason event_roi;primary_visual_panel;control_balanced_panel;no_auto_flags;hard_rollout_mobility
- Auto-QC likely ROI cycle156_rank1_obj1: score 0.832, risk 0.125, cycle 156, role event, reason event_roi;primary_visual_panel;control_balanced_panel;no_auto_flags
- Auto-QC likely ROI cycle156_rank6_obj3: score 0.762, risk 0.250, cycle 156, role event, reason event_roi;primary_visual_panel;control_balanced_panel;diffusion_ci_check
- Auto-QC likely ROI cycle157_rank1_obj1: score 0.703, risk 0.000, cycle 157, role control, reason control_roi;control_balanced_panel;no_auto_flags;large_conditioned_front_sign_residual
- Auto-QC artifact-risk ROI cycle156_rank5_obj4: score 0.598, risk 0.625, diffusion guardrail 1, reason event_roi;primary_visual_panel;control_balanced_panel;fragmentation_check;diffusion_ci_check;hard_rollout_mobility
- Auto-QC artifact-risk ROI cycle86_rank6_obj78: score 0.542, risk 0.500, diffusion guardrail 1, reason event_roi;control_balanced_panel;fragmentation_check;diffusion_ci_check
- Auto-QC artifact-risk ROI cycle62_rank2_obj2: score 0.523, risk 0.500, diffusion guardrail 0, reason control_roi;control_balanced_panel;fragmentation_check;large_conditioned_front_sign_residual
- Auto-QC artifact-risk ROI cycle157_rank2_obj2: score 0.486, risk 0.625, diffusion guardrail 1, reason control_roi;control_balanced_panel;fragmentation_check;diffusion_ci_check;high_residual_mode_priority;large_conditioned_front_sign_residual
- Auto-QC artifact-risk ROI cycle86_rank4_obj9: score 0.464, risk 0.500, diffusion guardrail 1, reason event_roi;primary_visual_panel;fragmentation_check;diffusion_ci_check
- Auto-QC artifact-risk ROI cycle158_rank2_obj1: score 0.463, risk 0.500, diffusion guardrail 1, reason control_roi;control_balanced_panel;fragmentation_check;diffusion_ci_check;large_conditioned_front_sign_residual
- Auto-QC contrast surrogate_diffusion_interpretability_score vs auto_diffusion_guardrail: median positive-negative -0.515, p=6.682e-09, n=47
- Auto-QC contrast automatic_artifact_risk_score vs is_artifact_risk: median positive-negative 0.125, p=5.722e-07, n=47
- Auto-QC contrast surrogate_front_mask_score vs is_likely_interpretable: median positive-negative 0.453, p=7.450e-07, n=47
- Auto-QC contrast diffusion_proxy_abs_median_um2_per_s vs auto_diffusion_guardrail: median positive-negative -2.211e-06, p=4.045e-05, n=46
- Auto-QC contrast surrogate_particle_identity_score vs is_likely_interpretable: median positive-negative 0.173, p=4.815e-04, n=47
- Auto-QC contrast automatic_artifact_risk_score vs is_likely_interpretable: median positive-negative -0.312, p=5.139e-04, n=47
- Auto-QC correlation surrogate_particle_identity_score vs automatic_qc_surrogate_score: rho=0.722, p=9.959e-09, n=47
- Auto-QC correlation surrogate_front_mask_score vs automatic_qc_surrogate_score: rho=0.616, p=3.982e-06, n=47
- Auto-QC correlation automatic_artifact_risk_score vs automatic_qc_surrogate_score: rho=-0.575, p=2.407e-05, n=47
- Auto-QC correlation surrogate_particle_identity_score vs automatic_artifact_risk_score: rho=-0.559, p=4.392e-05, n=47
- Auto-QC correlation surrogate_front_mask_score vs automatic_artifact_risk_score: rho=-0.507, p=2.773e-04, n=47
- Auto-QC correlation surrogate_diffusion_interpretability_score vs automatic_qc_surrogate_score: rho=0.483, p=5.824e-04, n=47
- Guardrail: Automatic QC triage surrogate preserves manual_qc_status as pending. It ranks candidate review priority from existing automatic/visual QC diagnostics and must not be treated as manual particle identity, front-mask, or diffusion-interpretable labels.

## QC Decision Evidence Ledger

- Candidate/manual-status rows: 47 / {'pending': 47}
- Decision action counts: {'high_priority_review': 5, 'review_artifact_or_reject_first': 4, 'review_but_diffusion_guarded': 16, 'review_for_possible_accept_first': 3, 'routine_pending_review': 19}
- Visual-asset candidates: 47
- QC possible-accept ROI cycle156_rank7_obj27: rank 1, score 0.948, risk 0.000, cycle 156, reasons auto_likely_interpretable;cross_modal_high_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- QC possible-accept ROI cycle156_rank8_obj10: rank 2, score 0.889, risk 0.000, cycle 156, reasons auto_likely_interpretable;cross_modal_review_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- QC possible-accept ROI cycle156_rank2_obj2: rank 3, score 0.888, risk 0.125, cycle 156, reasons auto_likely_interpretable;cross_modal_high_priority;multi_pillar_support;front_mask_candidate;diffusion_proxy_candidate;visual_assets_available
- QC artifact/reject-first ROI cycle156_rank5_obj4: artifact score 0.920, risk 0.625, cycle 156, reasons artifact_risk_review;cross_modal_review_priority;multi_pillar_support;front_mask_candidate;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- QC artifact/reject-first ROI cycle86_rank4_obj9: artifact score 0.865, risk 0.500, cycle 86, reasons artifact_risk_review;cross_modal_review_priority;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- QC artifact/reject-first ROI cycle157_rank2_obj2: artifact score 0.794, risk 0.625, cycle 157, reasons artifact_risk_review;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- QC artifact/reject-first ROI cycle158_rank2_obj1: artifact score 0.787, risk 0.500, cycle 158, reasons artifact_risk_review;cross_modal_review_priority;multi_pillar_support;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- QC artifact/reject-first ROI cycle86_rank5_obj8: artifact score 0.768, risk 0.500, cycle 86, reasons artifact_risk_review;artifact_guardrail_needed;diffusion_claim_guardrail;visual_assets_available
- QC cycle 156: n=6, possible accept=3, artifact-first=1, max score 0.948
- QC cycle 60: n=6, possible accept=0, artifact-first=1, max score 0.776
- QC cycle 62: n=4, possible accept=0, artifact-first=1, max score 0.733
- QC cycle 157: n=4, possible accept=0, artifact-first=1, max score 0.710
- QC cycle 116: n=5, possible accept=0, artifact-first=0, max score 0.645
- Guardrail: This ledger does not assign manual QC labels. It prioritizes pending particle/front candidates for human review using existing automatic evidence and keeps all physics claims guarded until particle identity, front mask, and calibration checks are manually accepted.

## Source-Balanced ROI Expansion Manifest

- Ranked/selected/sampled cycles: 89 / 48 / 48
- Sources selected: 14
- New cycle/source pairs versus existing video cohorts: 41
- Reconstructed candidates/ROI rows/missing cycles: 2880 / 96 / 0
- Selected label counts: {'future16_positive': 24, 'future8_positive': 14, 'same_cycle_drop': 0}
- Selection reason counts: {'future16_negative': 10, 'future16_positive': 8, 'source_fill': 17, 'source_representative': 13}
- Expansion source 10_c2_x10_030723: selected 2, new 2, future16+ 0, candidates 120
- Expansion source 11_c2_x10_050723: selected 3, new 3, future16+ 2, candidates 180
- Expansion source 12_c2_x10_070723: selected 4, new 2, future16+ 4, candidates 240
- Expansion source 14_c2_x10_HighCOV_110723: selected 4, new 4, future16+ 0, candidates 240
- Expansion source 15_c2_x5_HighCOV_120723: selected 3, new 3, future16+ 0, candidates 180
- Expansion source 16_c2_x10_HighHighCOV_130723: selected 4, new 4, future16+ 3, candidates 240
- Expansion source 17_c2_x10_HighHighCOV_150723: selected 3, new 0, future16+ 3, candidates 180
- Expansion source 2_c2_x14_200623: selected 4, new 4, future16+ 0, candidates 240
- Expansion source 4_c2_x10_240623: selected 4, new 4, future16+ 0, candidates 240
- Expansion source 5_c2_x10_260623: selected 3, new 3, future16+ 0, candidates 180
- Expansion ROI candidate cycle 92 10_c2_x10_030723 rank 1: score 25.929, future16 0, existing cohort False
- Expansion ROI candidate cycle 92 10_c2_x10_030723 rank 2: score 24.775, future16 0, existing cohort False
- Expansion ROI candidate cycle 94 10_c2_x10_030723 rank 1: score 25.611, future16 0, existing cohort False
- Expansion ROI candidate cycle 94 10_c2_x10_030723 rank 2: score 24.428, future16 0, existing cohort False
- Expansion ROI candidate cycle 98 11_c2_x10_050723 rank 2: score 25.035, future16 0, existing cohort False
- Expansion ROI candidate cycle 98 11_c2_x10_050723 rank 1: score 23.986, future16 0, existing cohort False
- Guardrail: Source-balanced expansion candidates are automatic proposals from sampled HDF5 frames. They reduce source/cycle selection bias for follow-up ROI export and manual QC, but do not validate particle identity, fronts, diffusion, or degradation mechanisms.

## Source-Balanced ROI Sequence Export and Rollout Audit

- Exported ROI sequences/cycles/sources/failures: 96 / 48 / 14 / 0
- Crop/output size/samples per ROI: 192 / 96 / 96
- Rollout audit ROI sequences/cycles/sources: 96 / 48 / 14
- Future8/future16 positive sequences: 28 / 48
- Source-balanced ROI feature future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.626, AP 0.589, source eta2 0.365, median pos-neg -2.135e-04
- Source-balanced ROI feature future_any_drop_within_16cycles raw_roi_mean_delta_last_minus_first: AUC 0.606, AP 0.549, source eta2 0.534, median pos-neg -2.669
- Source-balanced ROI feature future_any_drop_within_16cycles roi_norm_mean_positive_step_fraction: AUC 0.602, AP 0.603, source eta2 0.233, median pos-neg 0.005
- Source-balanced ROI feature future_any_drop_within_16cycles stage_drift_xy_recomputed: AUC 0.563, AP 0.590, source eta2 0.335, median pos-neg -0.026
- Source-balanced ROI feature future_any_drop_within_16cycles temporal_energy_late_minus_early: AUC 0.561, AP 0.548, source eta2 0.294, median pos-neg 2.230e-05
- Source-balanced ROI feature future_any_drop_within_16cycles persistence_mse_late_mean: AUC 0.556, AP 0.619, source eta2 0.787, median pos-neg 4.682e-06
- Source-balanced ROI feature future_any_drop_within_16cycles velocity_mse_p95: AUC 0.556, AP 0.616, source eta2 0.717, median pos-neg 7.237e-05
- Source-balanced ROI feature future_any_drop_within_16cycles velocity_minus_persistence_mse: AUC 0.550, AP 0.609, source eta2 0.825, median pos-neg 7.504e-06
- Source-balanced ROI feature future_any_drop_within_16cycles persistence_mse_p95: AUC 0.548, AP 0.614, source eta2 0.727, median pos-neg 1.061e-05
- Source-balanced ROI feature future_any_drop_within_16cycles velocity_mse_mean: AUC 0.540, AP 0.605, source eta2 0.824, median pos-neg 8.163e-06
- Source-balanced cycle feature future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.648, AP 0.594, median pos-neg -2.333e-04
- Source-balanced cycle feature future_any_drop_within_16cycles stage_drift_xy_recomputed: AUC 0.563, AP 0.590, median pos-neg -0.026
- Source-balanced cycle feature future_any_drop_within_16cycles object_mean_abs_z: AUC 0.562, AP 0.559, median pos-neg -0.867
- Source-balanced cycle feature future_any_drop_within_16cycles temporal_energy_late_minus_early: AUC 0.554, AP 0.556, median pos-neg 2.904e-05
- Source-balanced cycle feature future_any_drop_within_16cycles velocity_mse_mean: AUC 0.519, AP 0.604, median pos-neg 5.961e-07
- Source-balanced cycle feature future_any_drop_within_16cycles persistence_mse_mean: AUC 0.502, AP 0.598, median pos-neg 4.526e-07
- Source-balanced rollout source 10_c2_x10_030723: ROI 4, cycles 2, persistence MSE 1.133e-04, future16 seq 0
- Source-balanced rollout source 11_c2_x10_050723: ROI 6, cycles 3, persistence MSE 5.424e-05, future16 seq 4
- Source-balanced rollout source 12_c2_x10_070723: ROI 8, cycles 4, persistence MSE 8.179e-05, future16 seq 8
- Source-balanced rollout source 14_c2_x10_HighCOV_110723: ROI 8, cycles 4, persistence MSE 9.804e-05, future16 seq 0
- Source-balanced rollout source 15_c2_x5_HighCOV_120723: ROI 6, cycles 3, persistence MSE 1.923e-04, future16 seq 0
- Source-balanced rollout source 16_c2_x10_HighHighCOV_130723: ROI 8, cycles 4, persistence MSE 2.279e-04, future16 seq 6
- Sequence guardrail: Source-balanced sequences are fixed padded particle-region crops around automatic reconstructed candidates. They broaden source coverage for model/physics tests, but are not manual particle annotations or validated front labels.
- Rollout guardrail: Source-balanced rollout features are computed from automatic particle-centered crops and weak future labels. They quantify ROI-only temporal prediction difficulty and optical drift/intensity dynamics, not manual QC, causal degradation, or calibrated diffusion.

## Source-Balanced Mask/Front Sanity Audit

- Mask/front ROI sequences/cycles/sources: 96 / 48 / 14
- Future8/future16 positive sequences: 28 / 48
- Assumed output-pixel scale: 0.192 um
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles masked_minus_background_mean_slope: AUC 0.690, AP 0.696, source eta2 0.634, median pos-neg 0.004
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.626, AP 0.589, source eta2 0.365, median pos-neg -2.135e-04
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles front_radius_q80_slope_px_per_norm_time: AUC 0.612, AP 0.616, source eta2 0.104, median pos-neg -0.246
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles front_radius_q60_slope_px_per_norm_time: AUC 0.611, AP 0.649, source eta2 0.370, median pos-neg 0.094
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles mask_area_fraction_median: AUC 0.607, AP 0.606, source eta2 0.317, median pos-neg 0.007
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles mask_base_area_fraction: AUC 0.595, AP 0.610, source eta2 0.243, median pos-neg 0.005
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles front_radius_q60_median_px: AUC 0.578, AP 0.535, source eta2 0.156, median pos-neg -1.500
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles front_radius_q80_median_px: AUC 0.576, AP 0.538, source eta2 0.153, median pos-neg -0.500
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles front_gradient_peak_radius_median_px: AUC 0.564, AP 0.550, source eta2 0.215, median pos-neg 0.500
- Source-balanced mask/front ROI feature future_any_drop_within_16cycles mask_centroid_max_step_px: AUC 0.560, AP 0.618, source eta2 0.695, median pos-neg 0.012
- Source-balanced mask/front cycle feature future_any_drop_within_16cycles masked_minus_background_mean_slope: AUC 0.688, AP 0.703, median pos-neg 0.003
- Source-balanced mask/front cycle feature future_any_drop_within_16cycles mask_base_area_fraction: AUC 0.666, AP 0.652, median pos-neg 0.009
- Source-balanced mask/front cycle feature future_any_drop_within_16cycles front_radius_q80_median_px: AUC 0.661, AP 0.602, median pos-neg -2.000
- Source-balanced mask/front cycle feature future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.648, AP 0.594, median pos-neg -2.333e-04
- Source-balanced mask/front cycle feature future_any_drop_within_16cycles mask_area_fraction_median: AUC 0.632, AP 0.651, median pos-neg 0.010
- Source-balanced mask/front cycle feature future_any_drop_within_16cycles front_radius_q70_median_px: AUC 0.625, AP 0.573, median pos-neg -1.000
- Source-balanced mask/front source 10_c2_x10_030723: ROI 4, cycles 2, q70 radius slope -1.332, future16 seq 0
- Source-balanced mask/front source 11_c2_x10_050723: ROI 6, cycles 3, q70 radius slope 0.389, future16 seq 4
- Source-balanced mask/front source 12_c2_x10_070723: ROI 8, cycles 4, q70 radius slope -0.425, future16 seq 8
- Source-balanced mask/front source 14_c2_x10_HighCOV_110723: ROI 8, cycles 4, q70 radius slope -0.183, future16 seq 0
- Source-balanced mask/front source 15_c2_x5_HighCOV_120723: ROI 6, cycles 3, q70 radius slope 0.186, future16 seq 0
- Source-balanced mask/front source 16_c2_x10_HighHighCOV_130723: ROI 8, cycles 4, q70 radius slope 0.032, future16 seq 6
- Guardrail: Automatic crop-local masks/front radii are source-balanced sanity proxies from resized ROI tensors; they are not manual particle masks, not calibrated fronts, and apparent diffusion uses an approximate 0.192 um/output-pixel scale.

## Source-Balanced Mask/Front Source-Residual Audit

- Rows/features/sources: 96 / 54 / 14
- Best raw future16 feature: masked_minus_background_mean_slope AUC 0.690, source eta2 0.634
- Best source-residual future16 feature: front_radius_q80_slope_px_per_norm_time AUC 0.631, AP 0.634, p=0.122
- Best within-source-rank future16 feature: front_radius_q80_slope_px_per_norm_time AUC 0.656, AP 0.677, p=0.062
- Guardrail: Source residualization tests whether automatic mask/front proxies survive source structure. Passing this audit would still be weak-label, automatic-mask evidence; failing it means the feature is useful mainly for QC/source triage.

## Source-Balanced Residual Dictionary Audit

- ROI sequences/cycles/sources: 96 / 48 / 14
- PCA components/downsample/variance explained: 16 / 2 / 0.522
- Feature set sizes: {'residual_dictionary': 72, 'mask_front_scalar': 27, 'residual_dictionary_plus_mask_front': 99}
- Residual dictionary leave-cycle future16: AUC 0.602, AP 0.581; leave-source future16: AUC 0.375, AP 0.448
- Top residual ROI/cycle future16 scalar: resdict_pc01_mean AUC 0.668, eta2 0.187 / resdict_pc01_mean AUC 0.769, eta2 0.421
- Source-balanced residual dictionary metric cycleNo future_any_drop_within_16cycles residual_dictionary: AUC 0.602, AP 0.581, n=96
- Source-balanced residual dictionary metric cycleNo future_any_drop_within_16cycles mask_front_scalar: AUC 0.593, AP 0.578, n=96
- Source-balanced residual dictionary metric cycleNo future_any_drop_within_16cycles residual_dictionary_plus_mask_front: AUC 0.573, AP 0.555, n=96
- Source-balanced residual dictionary metric source_stem future_any_drop_within_16cycles mask_front_scalar: AUC 0.456, AP 0.466, n=96
- Source-balanced residual dictionary metric source_stem future_any_drop_within_16cycles residual_dictionary: AUC 0.375, AP 0.448, n=96
- Source-balanced residual dictionary metric source_stem future_any_drop_within_16cycles residual_dictionary_plus_mask_front: AUC 0.322, AP 0.396, n=96
- Source-balanced residual dictionary metric cycleNo future_any_drop_within_8cycles mask_front_scalar: AUC 0.641, AP 0.508, n=96
- Source-balanced residual dictionary metric cycleNo future_any_drop_within_8cycles residual_dictionary_plus_mask_front: AUC 0.614, AP 0.437, n=96
- Source-balanced residual dictionary metric cycleNo future_any_drop_within_8cycles residual_dictionary: AUC 0.600, AP 0.438, n=96
- Source-balanced residual dictionary metric source_stem future_any_drop_within_8cycles mask_front_scalar: AUC 0.418, AP 0.255, n=96
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles masked_minus_background_mean_slope: AUC 0.690, AP 0.696, source eta2 0.634
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles resdict_pc01_mean: AUC 0.668, AP 0.670, source eta2 0.187
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles resdict_pc02_slope: AUC 0.668, AP 0.719, source eta2 0.199
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles resdict_pc09_slope: AUC 0.653, AP 0.674, source eta2 0.150
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles resdict_pc03_mean: AUC 0.641, AP 0.666, source eta2 0.329
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles resdict_pc02_last_minus_first: AUC 0.637, AP 0.617, source eta2 0.112
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles resdict_pc09_last_minus_first: AUC 0.629, AP 0.640, source eta2 0.114
- Source-balanced residual dictionary ROI feature future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first: AUC 0.626, AP 0.589, source eta2 0.365
- Source-balanced residual dictionary cycle feature future_any_drop_within_16cycles resdict_pc01_mean: AUC 0.769, AP 0.753, source eta2 0.421
- Source-balanced residual dictionary cycle feature future_any_drop_within_16cycles resdict_pc09_slope: AUC 0.724, AP 0.755, source eta2 0.422
- Source-balanced residual dictionary cycle feature future_any_drop_within_16cycles resdict_pc09_last_minus_first: AUC 0.698, AP 0.698, source eta2 0.316
- Source-balanced residual dictionary cycle feature future_any_drop_within_16cycles resdict_pc07_last_minus_first: AUC 0.689, AP 0.728, source eta2 0.461
- Source-balanced residual dictionary cycle feature future_any_drop_within_16cycles masked_minus_background_mean_slope: AUC 0.688, AP 0.703, source eta2 0.649
- Source-balanced residual dictionary cycle feature future_any_drop_within_16cycles resdict_pc07_slope: AUC 0.682, AP 0.725, source eta2 0.536
- Guardrail: Residual dictionary bases are label-free PCA summaries of automatic source-balanced ROI crops. They are useful for ranking dynamics hypotheses, not a trained deployable predictor or calibrated physics model.

## Source-Balanced Residual Dictionary Source-Residual Audit

- Rows/features/sources: 96 / 102 / 14
- Feature family counts: {'residual_dictionary': 72, 'mask_front_scalar': 24, 'object_reconstruction': 6}
- Best future16 source-residual residual dictionary feature: dictionary_recon_error_last_minus_first AUC 0.637, AP 0.637, eta2 1.484e-33, median pos-neg 2.537e-06
- Best future16 within-source-rank residual dictionary feature: dictionary_recon_error_mse_slope AUC 0.574, AP 0.551, eta2 0.004
- Best future16 source-residual feature overall: dictionary_recon_error_last_minus_first (residual_dictionary) AUC 0.637, AP 0.637
- Guardrail: Source-normalized residual dictionary tests are in-cohort weak-label audits. They can identify source-robust residual dynamics candidates for follow-up, but they do not prove source-transferable prediction or calibrated phase/diffusion physics.

## Source-Balanced Residual Dictionary Normalized Readout

- Rows/cycles/sources: 96 / 48 / 14
- Feature set sizes: {'residual_dictionary_raw': 72, 'residual_dictionary_source_residual': 72, 'residual_dictionary_within_source_z': 72, 'residual_dictionary_within_source_rank': 72, 'mask_front_source_residual': 24, 'residual_dictionary_plus_mask_front_source_residual': 96, 'mask_front_within_source_rank': 24, 'residual_dictionary_plus_mask_front_within_source_rank': 96, 'dictionary_recon_error_last_minus_first_source_residual': 1}
- Future16 leave-source raw residual dictionary: AUC 0.375, AP 0.448
- Future16 leave-source source-residual residual dictionary: AUC 0.550, AP 0.583
- Future16 leave-source within-source-rank residual dictionary: AUC 0.543, AP 0.506
- Best future16 leave-source readout: dictionary_recon_error_last_minus_first_source_residual AUC 0.612, AP 0.613, n=96
- Same single-feature leave-cycle future16 readout: AUC 0.627, AP 0.630
- Permutation null: n=200, AUC p95=0.621, empirical p(AUC)=0.100, empirical p(AP)=0.070
- Guardrail: Source transforms are unsupervised within-source normalizations computed from ROI feature distributions, including held-out source rows without labels. This tests source-normalized readout stability, not a deployable source-transfer warning model.

## Source-Balanced Residual Temporal Specificity Audit

- Rows/cycles/sources: 96 / 48 / 14
- Event cycles and label counts: [60.0, 86.0, 116.0, 156.0] / {'current_any_event': 0, 'future8': 28, 'future16': 48, 'past8': 10, 'past16': 30}
- Primary source-residual reconstruction-error future8: AUC 0.539, past-control AUC 0.630, future-minus-control -0.091
- Primary source-residual reconstruction-error future16: AUC 0.637, AP 0.637, past-control AUC 0.625, future-minus-control 0.012
- Primary future16 within-source shift null: n=500, null AUC p95=0.521, empirical p(AUC)=0.002
- Most future-specific row overall: masked_minus_background_mean_slope / raw / 8 cycles, future AUC 0.821, past-control AUC 0.377, delta 0.444
- Guardrail: Temporal labels come from full particle abrupt-drop cycles but are evaluated on the 96 source-balanced ROI rows. Future-oriented scores use the future-label sign and apply the same sign to past/current controls. This tests temporal specificity, not causality or deployable warning.

## Source-Balanced Future-Specific Residual Audit

- Rows/cycles/sources: 96 / 48 / 14
- Event cycles and label counts: [60.0, 86.0, 116.0, 156.0] / {'current_any_event': 0, 'future8': 28, 'future16': 48, 'past8': 10, 'past16': 30, 'no_past8_rows': 86, 'no_past16_rows': 66}
- Primary source-residual reconstruction-error future16 all rows: AUC 0.637, AP 0.637
- Primary source-residual reconstruction-error future16 excluding past16 rows: AUC 0.589, AP 0.708, n=66
- Primary source-residual reconstruction-error future16 pre-first-event only: AUC 0.541, AP 0.329, n=32
- Top clean scalar row: masked_minus_background_mean_slope / raw / window 8 / exclude_past8 AUC 0.812, AP 0.713
- Best grouped residual delta over past-event context: cycleNo full_future_any_event_within_16cycles mask_contrast_source_residual_plus_context delta AUC 0.065, model AUC 0.820, base AUC 0.755
- Guardrail: Past-event context uses event-cycle labels derived from full particle abrupt-drop cycles. Excluding or modeling past windows tests future specificity on the source-balanced ROI table; it does not prove causality, source transfer, or calibrated phase/diffusion physics.

## Source-Balanced Pre-Event Sampling Manifest

- Selected/sample cycles/sources: 64 / 64 / 14
- New cycle/source pairs vs existing video cohorts: 13 of 64
- Event cycles: [60.0, 86.0, 116.0, 156.0]
- Cycle bins: {'far_pre_event_17_32': 11, 'mid_pre_event_9_16': 11, 'near_pre_event_1_8': 16, 'no_near_event_control': 6, 'post_event_1_16': 20}
- ROI proposal bins: {'far_pre_event_17_32': 22, 'mid_pre_event_9_16': 22, 'near_pre_event_1_8': 32, 'no_near_event_control': 12, 'post_event_1_16': 40}
- Reconstructed candidates / ROI rows / missing cycles: 3840 / 128 / 0
- Guardrail: Pre-event sampling candidates are automatic particle-like proposals from sampled HDF5 frames. They broaden event-relative/source-balanced coverage for follow-up ROI video export and future-specific modeling, but do not validate particle identity, phase fronts, diffusion, or causal degradation mechanisms.
- Source coverage 10_c2_x10_030723: selected=4, near=0, mid=0, far=0, post=4, controls=0, new=0
- Source coverage 11_c2_x10_050723: selected=5, near=0, mid=1, far=0, post=4, controls=0, new=1
- Source coverage 12_c2_x10_070723: selected=5, near=4, mid=1, far=0, post=0, controls=0, new=1
- Source coverage 14_c2_x10_HighCOV_110723: selected=4, near=0, mid=0, far=0, post=4, controls=0, new=3
- Source coverage 15_c2_x5_HighCOV_120723: selected=2, near=0, mid=0, far=2, post=0, controls=0, new=0
- Source coverage 16_c2_x10_HighHighCOV_130723: selected=5, near=0, mid=4, far=1, post=0, controls=0, new=2
- Source coverage 17_c2_x10_HighHighCOV_150723: selected=5, near=4, mid=1, far=0, post=0, controls=0, new=0
- Source coverage 2_c2_x14_200623: selected=5, near=0, mid=0, far=0, post=0, controls=5, new=3

## Source-Balanced Pre-Event ROI Sequence Export

- ROI sequences/cycles/sources/failures: 128 / 64 / 14 / 0
- Future8/future16 sequence positives from event proximity: 32 / 66
- Guardrail: Source-balanced sequences are fixed padded particle-region crops around automatic reconstructed candidates. They broaden source coverage for model/physics tests, but are not manual particle annotations or validated front labels.

## Source-Balanced Pre-Event Sequence Audit

- Feature rows/cycles/sources/failures: 128 / 64 / 14 / 0
- Bin counts: {'far_pre_event_17_32': 22, 'mid_pre_event_9_16': 22, 'near_pre_event_1_8': 32, 'no_near_event_control': 12, 'post_event_1_16': 40}
- Target positives: {'target_any_pre_vs_post_control': 76, 'target_near_pre_vs_rest': 32, 'target_pre16_clean_vs_post_control': 54}
- Near-pre-event spatial readout: leave-cycle AUC 0.763/AP 0.536; leave-source AUC 0.759/AP 0.515
- Clean pre16 vs post/control all-video leave-cycle readout: AUC 0.723/AP 0.759
- Any-pre vs post/control all-video leave-source guardrail: AUC 0.463/AP 0.645
- Top scalar test: target_any_pre_vs_post_control frame_diff_mse_p95 / raw AUC 0.632, AP 0.778, p=0.011
- Guardrail: Event-relative sequence features are computed from automatic fixed particle crops and labels are derived from abrupt-event cycle proximity. Positive readouts indicate pre/post/control optical-dynamics separation for follow-up QC and modeling, not validated precursors, particle identities, phase boundaries, diffusion coefficients, or causal degradation mechanisms.
- Event-relative model target_any_pre_vs_post_control cycleNo all_video: AUC 0.637, AP 0.758, n=128
- Event-relative model target_any_pre_vs_post_control cycleNo spatial: AUC 0.608, AP 0.701, n=128
- Event-relative model target_any_pre_vs_post_control cycleNo dynamics: AUC 0.600, AP 0.736, n=128
- Event-relative model target_any_pre_vs_post_control source_stem spatial: AUC 0.550, AP 0.652, n=128
- Event-relative model target_any_pre_vs_post_control cycleNo intensity: AUC 0.530, AP 0.708, n=128
- Event-relative model target_any_pre_vs_post_control source_stem dynamics: AUC 0.501, AP 0.676, n=128
- Event-relative model target_any_pre_vs_post_control source_stem all_video: AUC 0.463, AP 0.645, n=128
- Event-relative model target_any_pre_vs_post_control source_stem intensity: AUC 0.408, AP 0.595, n=128

## Source-Balanced Pre-Event Rollout, Mask/Front, and Event-Relative Readout

- Rollout audit rows/cycles/sources: 128 / 64 / 14
- Top pre-event rollout future-label row: future_any_drop_within_16cycles roi_norm_mean_delta_last_minus_first AUC 0.628, AP 0.620, source eta2 0.414
- Top pre-event mask/front future-label row: future_any_drop_within_16cycles masked_minus_background_mean_slope AUC 0.624, AP 0.655, source eta2 0.578
- Event-relative bins: {'far_pre_event_17_32': 22, 'mid_pre_event_9_16': 22, 'near_pre_event_1_8': 32, 'no_near_event_control': 12, 'post_event_1_16': 40}
- Best source-residual clean-pre readout: clean_pre_1_8_vs_post_control front_radius_q60_slope_px_per_norm_time AUC 0.660, AP 0.665, p=0.083
- Event-distance trajectory rows/sources/events/features: 38 / 10 / 4 / 44; complete near/far event cycles 2
- Top trajectory physics trend: source_residual apparent_diffusion_q70_px2_per_norm_time rho 0.272, p=0.120, source-stratified permutation p=0.064, near-far median 8.259
- Trajectory guardrail: Cycle-level event-distance trajectories reduce duplicate ROI counting and test monotonic pre-event organization, but automatic crops/masks and sparse full far-mid-near event trajectories mean these are physics triage signals, not calibrated phase boundaries, diffusion coefficients, or causal degradation forecasts.
- Directionality rows/cycles/sources/features: 128 / 64 / 14 / 57
- Top physics-facing pre-event clock feature: source_residual apparent_diffusion_q70_um2_per_norm_time pre rho 0.201, p=0.176, permutation p=0.024, post rho -0.120
- Top pre/post clock asymmetry: raw frame_diff_mse_slope |pre|-|post| rho delta 0.227
- Directionality near-pre vs far-pre: raw spatial_std_slope AUC 0.801, AP 0.853, pre rho 0.436
- Directionality clean-pre source-residual readout: front_radius_q60_slope_px_per_norm_time AUC 0.660, AP 0.665, pre rho 0.206, post rho -0.278
- Directionality guardrail: Temporal directionality uses automatic pre-event ROI crops, weak event-relative bins, and within-source clock permutations. AUC rows remain descriptive readouts. This tests whether optical/front/rollout proxies are ordered around event time; it does not validate causal precursors, particle identity, calibrated phase boundaries, or diffusion coefficients.
- Source-invariant rows/cycles/sources/features: 128 / 64 / 14 / 47
- Source-invariant clean-pre best: physics_front_combo source_mean_resid_2 leave-source AUC 0.694, AP 0.660, rho 0.327
- Source-invariant near-vs-far best: physics_front_combo source_residual leave-source AUC 0.744, AP 0.711, rho 0.416
- Source-invariant low-source-eta clean guardrail: object_context_guardrail source_confound_filter_0.25 AUC 0.573, max eta2 0.270
- Source-invariant guardrail: Leave-source source-invariant pre-event models use automatic ROI features and weak event-relative labels. Source residual/rank transforms are analysis-time normalizations. These results nominate pre-event mechanism families for review; they are not causal precursors, deployable warnings, manual particle identities, calibrated phase boundaries, or diffusion coefficients.
- Review packet candidates/cycles/sources/rendered strips: 128 / 64 / 14 / 24
- Review packet reasons: {'context_or_guardrail_row': 74, 'mid_pre_followup_front_diffusion_review': 22, 'near_pre_front_diffusion_review': 19, 'near_pre_high_source_invariant_front_score': 13}
- Review packet contact sheet: /scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_review_packet/pre_event_review_contact_sheet.png
- Top review candidate: rank 1 source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723 cycle 152 score 4.789, reason near_pre_high_source_invariant_front_score, source-invariant clean probability 0.941
- Review packet guardrail: This packet ranks automatic source-balanced pre-event ROI crops for manual review using weak event-relative labels and source-invariant/front-proxy scores. Frame strips are rendered from automatic fixed crops. No manual labels, particle identities, phase-boundary validation, calibrated diffusion coefficients, or causal precursor claims are assigned.
- Matched-counterfactual rows/cycles/sources/permutations: 128 / 64 / 14 / 1000
- Matched-counterfactual pair counts: {'near_vs_far_pre|source_penalized_global': 32, 'near_vs_post_control|same_source': 12, 'near_vs_post_control|source_penalized_global': 32}; same-source fractions: {'near_vs_far_pre|source_penalized_global': 0.0, 'near_vs_post_control|same_source': 1.0, 'near_vs_post_control|source_penalized_global': 0.375}
- Top matched physics row: near_vs_far_pre source_penalized_global masked_minus_background_mean_median n=32, median near-control diff 0.003, sign-flip p=9.990e-04
- Matched front/diffusion guardrails: q60 front-slope median diff 0.988, p=0.124; q70 apparent diffusion median diff -0.204, p=0.395
- Matched-counterfactual guardrail: Nearest-neighbor matching uses automatic context/baseline descriptors only and controls observed confounding, not unobserved acquisition or particle-identity effects. Pairs reuse controls and all ROI masks/fronts are automatic. Results prioritize manual QC and physics follow-up; they do not validate phase boundaries, calibrated diffusion coefficients, degradation causality, or deployable warnings.
- Same-source ladder rows/cycles/sources/permutations: 128 / 64 / 14 / 1000
- Same-source ladder coverage: {'sources_with_near_far_ladder': 0, 'sources_with_near_mid_ladder': 3, 'sources_with_near_post_ladder': 2, 'sources_with_near_rows': 5}; pair counts: {'near_vs_any_non_near_same_source': 32, 'near_vs_mid_pre_same_source': 20, 'near_vs_post_control_same_source': 12}
- Top same-source paired row: near_vs_any_non_near_same_source mask_centroid_path_px n=32, median near-control diff 0.784, sign-flip p=0.002
- Same-source front/diffusion guardrails: q60 front-slope median diff 1.769, p=0.006; q70 apparent diffusion median diff 0.666, p=0.004; top continuous clock front_radius_q60_slope_px_per_norm_time rho 0.109, p=0.417
- Same-source ladder guardrail: This audit is strictly within-source but opportunistic: sources have incomplete event-relative ladders, pairs reuse controls, and all particle masks/fronts are automatic. It tests whether observed near-pre signals survive same-source local controls; it does not validate particle identity, calibrated diffusion, phase-boundary motion, causality, or deployable warnings.
- Radial-kymograph ROI/cycles/sources/rendered: 128 / 64 / 14 / 32
- Top radial-kymograph near-vs-far row: front_radius2_slope_px2_per_norm_time AUC 0.705, median diff 3.455, p=0.040
- Top radial-kymograph clean-pre row: kymograph_temporal_energy AUC 0.664, median diff 2.863e-06, p=0.002
- Top review-candidate kymograph: source_balanced_cycle152_rank29_obj1_17_c2_x10_HighHighCOV_150723 rank 1 front radius2 slope 12.903, front slope R2 0.076
- Radial-kymograph guardrail: Radial kymographs are automatic fixed-crop optical front proxies from source-balanced ROI tensors. Centroids, thresholds, and gradient fronts are not manual segmentations. Radius-squared slopes are apparent mobility descriptors only, not calibrated diffusion coefficients or validated phase-boundary tracks.
- Physics-mode taxonomy rows/cycles/sources/features/k: 128 / 64 / 14 / 32 / 2
- Top physics mode: mode 0 front_geometry_mask_contrast_video_heterogeneity n=60, near-pre fraction 0.267, post fraction 0.300, loadings temporal_energy_mean;frame_diff_abs_mean;mask_base_area_fraction;front_radius_q60_median_px;mask_area_fraction_median;temporal_energy_p95
- Best mode enrichment row: mode 1 near_pre fraction 0.235 vs outside 0.267, p=0.689
- Physics-mode guardrail: Pre-event physics modes are unsupervised clusters of automatic source-residual ROI features. Mode enrichment and event-clock tests nominate repeatable front/diffusion/heterogeneity states for review; they are not manual degradation labels, calibrated phase boundaries, diffusion coefficients, or causal forecasts.
- Guardrail: Event-relative readouts use automatic ROI crops and weak event-distance bins. They test pre/post/control organization of optical/front/rollout proxies, not manual particle identity, causal mechanism, calibrated phase boundaries, or diffusion coefficients.
- Event-relative readout clean_pre_1_16_vs_post_control raw mask_area_fraction_iqr: AUC 0.683, AP 0.661, p=0.001, eta2 0.384
- Event-relative readout clean_pre_1_16_vs_post_control source_residual front_radius_q80_median_px: AUC 0.645, AP 0.570, p=0.010, eta2 4.321e-33
- Event-relative readout clean_pre_1_16_vs_post_control within_source_rank front_radius_q70_delta_px: AUC 0.593, AP 0.611, p=0.094, eta2 0.004
- Event-relative readout clean_pre_1_32_vs_post_control raw mask_area_fraction_iqr: AUC 0.654, AP 0.700, p=0.003, eta2 0.384
- Event-relative readout clean_pre_1_32_vs_post_control source_residual front_radius_q80_median_px: AUC 0.620, AP 0.660, p=0.021, eta2 4.321e-33
- Event-relative readout clean_pre_1_32_vs_post_control within_source_rank temporal_energy_late_minus_early: AUC 0.566, AP 0.639, p=0.204, eta2 0.002
- Event-relative readout clean_pre_1_8_vs_post_control raw masked_minus_background_mean_slope: AUC 0.797, AP 0.749, p=5.478e-06, eta2 0.578
- Event-relative readout clean_pre_1_8_vs_post_control source_residual front_radius_q60_slope_px_per_norm_time: AUC 0.660, AP 0.665, p=0.083, eta2 3.045e-33
- Event-relative readout clean_pre_1_8_vs_post_control within_source_rank front_radius_q70_slope_px_per_norm_time: AUC 0.592, AP 0.479, p=0.306, eta2 0.024
- Event-relative readout near_pre_vs_far_pre raw masked_minus_background_mean_slope: AUC 0.734, AP 0.815, p=0.004, eta2 0.578
- Event-distance trajectory source_residual apparent_diffusion_q70_px2_per_norm_time: rho 0.272, permutation p=0.064, near-far median 8.259
- Event-distance trajectory source_residual front_radius_q70_slope_px_per_norm_time: rho 0.160, permutation p=0.084, near-far median 0.341
- Event-distance trajectory source_residual apparent_diffusion_q70_um2_per_norm_time: rho 0.272, permutation p=0.100, near-far median 0.304
- Event-distance trajectory event_residual mask_centroid_path_px: rho 0.239, permutation p=0.167, near-far median 1.204
- Event-distance trajectory source_residual front_radius_q60_slope_px_per_norm_time: rho 0.166, permutation p=0.191, near-far median 0.278
- Event-distance trajectory event_residual mask_area_fraction_slope: rho 0.135, permutation p=0.191, near-far median 0.006

## Source-Balanced Degradation Mode Audit

- Rows/cycles/sources/features: 96 / 48 / 14 / 18
- Chosen k and source-mode transitions: k=4, transition count 82, change fraction 0.244
- Strongest event-neighborhood enrichment: mode 2 post_event_16 fraction 1.000 vs outside 0.085, p=0.010
- Degradation mode 0 source_residual_front_geometry_state: n=70, cycles=39, sources=13, future16 fraction 0.543, past16 fraction 0.329, phases {'pre_event_8': 20, 'far_from_event': 19, 'pre_event_16': 18, 'post_event_8': 8, 'post_event_16': 5}
- Degradation mode 1 source_residual_temporal_dictionary_state: n=1, cycles=1, sources=1, future16 fraction 1.000, past16 fraction 0.000, phases {'pre_event_16': 1}
- Degradation mode 2 source_residual_temporal_dictionary_state: n=2, cycles=1, sources=1, future16 fraction 0.000, past16 fraction 1.000, phases {'post_event_16': 2}
- Degradation mode 3 source_residual_temporal_dictionary_state: n=23, cycles=16, sources=10, future16 fraction 0.391, past16 fraction 0.217, phases {'far_from_event': 9, 'pre_event_8': 8, 'post_event_16': 3, 'post_event_8': 2, 'pre_event_16': 1}
- Mode enrichment 2 post_event_16 (event_phase): fraction 1.000 vs outside 0.085, p=0.010, n=2
- Mode enrichment 3 pre_event_16 (event_phase): fraction 0.043 vs outside 0.260, p=0.036, n=23
- Mode enrichment 0 pre_event_16 (event_phase): fraction 0.257 vs outside 0.077, p=0.087, n=70
- Mode enrichment 2 past16 (binary_label): fraction 1.000 vs outside 0.298, p=0.095, n=2
- Mode enrichment 0 post_event_16 (event_phase): fraction 0.071 vs outside 0.192, p=0.128, n=70
- Mode enrichment 1 pre_event_16 (event_phase): fraction 1.000 vs outside 0.200, p=0.208, n=1
- Mode enrichment 0 future16 (binary_label): fraction 0.543 vs outside 0.385, p=0.251, n=70
- Mode enrichment 3 far_from_event (event_phase): fraction 0.391 vs outside 0.260, p=0.293, n=23
- Mode representative 1: source_balanced_cycle142_rank18_obj1_16_c2_x10_HighHighCOV_130723 cycle 142, phase pre_event_16, distance 0.000
- Mode representative 0: source_balanced_cycle92_rank1_obj1_10_c2_x10_030723 cycle 92, phase post_event_8, distance 1.032
- Mode representative 0: source_balanced_cycle92_rank1_obj2_10_c2_x10_030723 cycle 92, phase post_event_8, distance 1.070
- Mode representative 0: source_balanced_cycle102_rank5_obj1_11_c2_x10_050723 cycle 102, phase pre_event_16, distance 1.303
- Mode representative 0: source_balanced_cycle64_rank40_obj1_7_c2_x10_290623 cycle 64, phase post_event_8, distance 1.376
- Mode representative 3: source_balanced_cycle98_rank3_obj2_11_c2_x10_050723 cycle 98, phase post_event_16, distance 1.581
- Guardrail: Modes are unsupervised source-residual clusters of automatic particle-region optical/residual proxies. They organize degradation-state hypotheses and review candidates, but they do not prove calibrated phase boundaries, diffusion coefficients, or causal event mechanisms.

## Source-Balanced Residual-Physics Coupling Audit

- Rows/residual features/physics proxies/sources: 96 / 72 / 25 / 14
- Top source-residual primary-candidate coupling: resdict_pc02_slope vs mask_area_fraction_median rho 0.521, residual AUC 0.515, physics AUC 0.534
- Top source-residual target-aligned coupling: dictionary_recon_error_last_minus_first vs masked_minus_background_mean_slope rho 0.373, residual AUC 0.637, physics AUC 0.628
- Residual-physics coupling raw: residual_energy_p95 vs mask_centroid_max_step_px rho 0.782, residual AUC 0.474, physics AUC 0.560
- Residual-physics coupling source_residual: resdict_pc09_std vs mask_centroid_max_step_px rho 0.796, residual AUC 0.510, physics AUC 0.499
- Residual-physics coupling within_source_rank: residual_energy_mean vs mask_centroid_max_step_px rho 0.643, residual AUC 0.457, physics AUC 0.469
- Residual-physics coupling within_source_z: resdict_pc01_std vs front_radius_q60_median_px rho -0.626, residual AUC 0.482, physics AUC 0.501
- Dictionary recon source-residual coupling to front_radius_q60_slope_px_per_norm_time: rho -0.380, p=0.006, physics AUC 0.542, target aligned=False
- Dictionary recon source-residual coupling to masked_minus_background_mean_slope: rho 0.373, p=1.826e-04, physics AUC 0.628, target aligned=True
- Dictionary recon source-residual coupling to mask_area_fraction_slope: rho -0.361, p=3.056e-04, physics AUC 0.592, target aligned=False
- Dictionary recon source-residual coupling to front_radius_q80_slope_px_per_norm_time: rho -0.305, p=0.033, physics AUC 0.631, target aligned=False
- Dictionary recon source-residual coupling to front_radius_q70_slope_px_per_norm_time: rho -0.262, p=0.055, physics AUC 0.504, target aligned=False
- Dictionary recon source-residual coupling to mask_centroid_drift_px: rho -0.231, p=0.024, physics AUC 0.580, target aligned=False
- Guardrail: Residual-physics coupling is a source-normalized correlation audit over automatic crop-local optical proxies. It can prioritize follow-up mechanisms, but it does not calibrate diffusion coefficients or prove phase-boundary physics without manual/QC and physical scale validation.

## Source-Balanced Residual Candidate Review Packet

- Candidates/sources/cycles: 96 / 14 / 48
- Review tier counts: {'routine': 58, 'standard_review': 19, 'high_priority': 11, 'immediate_manual_qc': 8}
- Top candidate: source_balanced_cycle108_rank6_obj2_12_c2_x10_070723 score 0.917, source 12_c2_x10_070723, cycle 108, status pending
- Residual candidate review rank 1: source_balanced_cycle108_rank6_obj2_12_c2_x10_070723 score 0.917, prob 0.781, tier immediate_manual_qc, status pending
- Residual candidate review rank 2: source_balanced_cycle132_rank14_obj2_15_c2_x5_HighCOV_120723 score 0.904, prob 0.828, tier immediate_manual_qc, status pending
- Residual candidate review rank 3: source_balanced_cycle145_rank20_obj2_16_c2_x10_HighHighCOV_130723 score 0.887, prob 0.585, tier immediate_manual_qc, status pending
- Residual candidate review rank 4: source_balanced_cycle132_rank14_obj1_15_c2_x5_HighCOV_120723 score 0.878, prob 0.773, tier immediate_manual_qc, status pending
- Residual candidate review rank 5: source_balanced_cycle108_rank6_obj1_12_c2_x10_070723 score 0.875, prob 0.747, tier immediate_manual_qc, status pending
- Residual candidate review rank 6: source_balanced_cycle145_rank20_obj1_16_c2_x10_HighHighCOV_130723 score 0.853, prob 0.592, tier immediate_manual_qc, status pending
- Residual candidate review rank 7: source_balanced_cycle144_rank19_obj2_16_c2_x10_HighHighCOV_130723 score 0.848, prob 0.584, tier immediate_manual_qc, status pending
- Residual candidate review rank 8: source_balanced_cycle144_rank19_obj1_16_c2_x10_HighHighCOV_130723 score 0.840, prob 0.586, tier immediate_manual_qc, status pending
- Residual candidate source 12_c2_x10_070723: n=8, max score 0.917, future16 rate 1.000
- Residual candidate source 15_c2_x5_HighCOV_120723: n=6, max score 0.904, future16 rate 0.000
- Residual candidate source 16_c2_x10_HighHighCOV_130723: n=8, max score 0.887, future16 rate 0.750
- Residual candidate source 17_c2_x10_HighHighCOV_150723: n=6, max score 0.797, future16 rate 1.000
- Residual candidate source 8_c2_x10_300623: n=8, max score 0.769, future16 rate 0.750
- Residual candidate source 14_c2_x10_HighCOV_110723: n=8, max score 0.727, future16 rate 0.000
- Guardrail: This packet ranks automatic source-balanced ROI crops for human review. It keeps manual_qc_status pending and does not assign particle identity, front validity, diffusion validity, or degradation labels.

## Balanced Future-Drop Direct-Video ROI Audit

- Reconstructed cycles/candidates/ROI rows: 24 / 1920 / 72
- Exported ROI sequences: 72
- Physics ROI/cycles/features: 72 / 24 / 26
- Label counts: [{'cohort_role': 'future8_negative', 'future_any_drop_within_8cycles': 0, 'n': 36}, {'cohort_role': 'future8_positive', 'future_any_drop_within_8cycles': 1, 'n': 36}]
- Masked rollout best-method counts: {'persistence': 72}
- Mask stability ROI/frames: 72 / 6912; overall {'median_fallback_frame_fraction': 0.0, 'median_accepted_area_cv': 0.07390347879875156, 'median_centroid_path_px': 133.39177039942166, 'median_mask_instability_score': 1.4906788486948246}
- Mask stability role summary: [{'cohort_role': 'future8_negative', 'n_roi': 36, 'fallback_frame_fraction': 0.0, 'accepted_area_cv': 0.07014676858327966, 'accepted_centroid_path_px': 132.55635556011174, 'mask_instability_score': 1.4676753658649946}, {'cohort_role': 'future8_positive', 'n_roi': 36, 'fallback_frame_fraction': 0.0, 'accepted_area_cv': 0.07785297724433327, 'accepted_centroid_path_px': 135.94103030938612, 'mask_instability_score': 1.5007457893873566}]
- Context-only / physics-only / physics+acquisition best AUC: 0.851 / 0.716 / 0.887
- Design-context best AUC: 1.000 (includes selection metadata such as balanced rank/subrole and warning score)
- Balanced future OOF logistic_l2: AUC 0.716, AP 0.761, scored 72 rows (36/36 pos/neg)
- Balanced future OOF random_forest: AUC 0.673, AP 0.736, scored 72 rows (36/36 pos/neg)
- Balanced future permutation null logistic_l2: observed AUC 0.716, null mean 0.489, p95 0.701, empirical p=0.049
- Balanced future ROI feature radius2_slope_median_px2_per_s: median positive-negative 0.001, oriented AUC 0.717, permutation p=0.024, n=36/36
- Balanced future ROI feature diffusion_proxy_median_um2_per_s: median positive-negative 2.622e-06, oriented AUC 0.717, permutation p=0.024, n=36/36
- Balanced future ROI feature q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 0.001, oriented AUC 0.710, permutation p=0.024, n=36/36
- Balanced future ROI feature persistence_particle_mse_fraction_of_full_mean: median positive-negative 0.282, oriented AUC 0.709, permutation p=0.024, n=36/36
- Balanced future ROI feature velocity_particle_mse_fraction_of_full_mean: median positive-negative 0.345, oriented AUC 0.704, permutation p=0.024, n=36/36
- Balanced future ROI feature velocity_particle_to_nonparticle_mse_ratio_mean: median positive-negative 0.495, oriented AUC 0.672, permutation p=0.024, n=36/36
- Balanced future ROI feature persistence_particle_to_nonparticle_mse_ratio_mean: median positive-negative 0.447, oriented AUC 0.673, permutation p=0.049, n=36/36
- Balanced future ROI feature transferred_masked_residual_signature: median positive-negative 0.416, oriented AUC 0.681, permutation p=0.073, n=36/36
- Balanced future cycle feature persistence_particle_mse_fraction_of_full_mean: median positive-negative 0.246, oriented AUC 0.729, permutation p=0.049, n=12/12
- Balanced future cycle feature radius2_slope_median_px2_per_s: median positive-negative 8.206e-04, oriented AUC 0.722, permutation p=0.049, n=12/12
- Balanced future cycle feature persistence_particle_to_nonparticle_mse_ratio_mean: median positive-negative 0.520, oriented AUC 0.694, permutation p=0.073, n=12/12
- Balanced future cycle feature q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative 8.303e-04, oriented AUC 0.729, permutation p=0.098, n=12/12
- Balanced future cycle feature velocity_particle_mse_fraction_of_full_mean: median positive-negative 0.278, oriented AUC 0.715, permutation p=0.122, n=12/12
- Balanced future cycle feature transferred_masked_residual_signature: median positive-negative 0.416, oriented AUC 0.681, permutation p=0.122, n=12/12
- Balanced future mask-stability accepted_centroid_max_step_px: future8 positive-negative 2.153, MW p=0.175, n=36/36
- Balanced future mask-stability accepted_area_cv: future8 positive-negative 0.008, MW p=0.248, n=36/36
- Balanced future mask-stability accepted_centroid_path_px: future8 positive-negative 3.385, MW p=0.395, n=36/36
- Balanced future mask-stability candidate_area_cv: future8 positive-negative 0.005, MW p=0.414, n=36/36
- Balanced future mask-stability mask_instability_score: future8 positive-negative 0.033, MW p=0.461, n=36/36
- Balanced future mask-stability low_area_fraction: future8 positive-negative 0.000, MW p=0.569, n=36/36
- Balanced future context OOF acquisition_context_only logistic_l2: AUC 0.823, AP 0.846
- Balanced future context OOF acquisition_context_only random_forest: AUC 0.851, AP 0.888
- Balanced future context OOF design_context_only logistic_l2: AUC 1.000, AP 1.000
- Balanced future context OOF design_context_only random_forest: AUC 1.000, AP 1.000
- Balanced future context OOF physics_only logistic_l2: AUC 0.711, AP 0.683
- Balanced future context OOF physics_only random_forest: AUC 0.689, AP 0.690
- Balanced future context OOF physics_plus_acquisition_context logistic_l2: AUC 0.887, AP 0.896
- Balanced future context OOF physics_plus_acquisition_context random_forest: AUC 0.786, AP 0.841
- Balanced future context OOF physics_plus_design_context logistic_l2: AUC 1.000, AP 1.000
- Balanced future context OOF physics_plus_design_context random_forest: AUC 1.000, AP 1.000
- Masked video embedding OOF future_any_drop_within_8cycles: AUC 0.816, AP 0.865, scored 72 rows (36/36 pos/neg)
- Masked video embedding OOF event_vs_control: AUC 0.588, AP 0.530, scored 52 rows (24/28 pos/neg)
- Masked video embedding future8 permutation null: observed AUC 0.816, null mean 0.469, p95 0.641, empirical p=0.012
- Masked video embedding feature particle_std_last_minus_first: median positive-negative 0.007, oriented AUC 0.891, MW p=1.168e-08
- Masked video embedding feature particle_std_slope: median positive-negative 0.009, oriented AUC 0.884, MW p=2.107e-08
- Masked video embedding feature particle_gradient_slope: median positive-negative 0.002, oriented AUC 0.813, MW p=4.951e-06
- Masked video embedding feature particle_gradient_last_minus_first: median positive-negative 0.001, oriented AUC 0.792, MW p=2.123e-05
- Masked video embedding feature particle_std_diff_q10: median positive-negative -0.001, oriented AUC 0.775, MW p=5.944e-05
- Masked video embedding feature particle_vs_context_mean_last_minus_first: median positive-negative 0.002, oriented AUC 0.773, MW p=6.856e-05
- Masked video embedding cluster 0: n=33, future8 positive fraction 0.235, prototype transfer_ranked::cycle147_rank11_obj4
- Masked video embedding cluster 1: n=46, future8 positive fraction 0.364, prototype selected_event_control::cycle60_rank1_obj1
- Masked video embedding cluster 2: n=11, future8 positive fraction 0.000, prototype selected_event_control::cycle156_rank8_obj10
- Masked video embedding cluster 3: n=15, future8 positive fraction 0.857, prototype transfer_ranked::cycle151_rank4_obj1
- Masked video embedding cluster 4: n=32, future8 positive fraction 0.429, prototype transfer_ranked::cycle40_rank8_obj4

## Learned Video Residual Embedding Audit

- Rows/cycles: 172 / 34
- Cohorts: {'balanced_future': 72, 'selected_event_control': 52, 'transfer_ranked': 48}
- Training: {'best_val_loss': 0.0004373242554720491, 'channels': 12, 'downsample': 2, 'final_train_loss': 0.0003976869063100163, 'n_epochs_run': 14, 'n_pairs_used': 7000, 'pair_stride': 2}
- Feature set sizes: {'handcrafted_scalar': 55, 'learned_all': 105, 'learned_latent': 96, 'learned_plus_handcrafted': 160, 'learned_residual': 9, 'pca_video': 16}
- Future8 learned_all / PCA-video / handcrafted AUC: 0.849 / 0.569 / 0.828
- Future16 learned_all / handcrafted AUC: 0.538 / 0.680
- Learned-residual classification future_any_drop_within_8cycles learned_all: AUC 0.849, AP 0.872, p=9.990e-04, n=72
- Learned-residual classification future_any_drop_within_8cycles learned_plus_handcrafted: AUC 0.843, AP 0.871, p=9.990e-04, n=72
- Learned-residual classification future_any_drop_within_8cycles learned_latent: AUC 0.829, AP 0.850, p=9.990e-04, n=72
- Learned-residual classification future_any_drop_within_8cycles handcrafted_scalar: AUC 0.828, AP 0.867, p=9.990e-04, n=72
- Learned-residual classification future_any_drop_within_8cycles learned_residual: AUC 0.690, AP 0.790, p=NA, n=72
- Learned-residual classification future_any_drop_within_16cycles handcrafted_scalar: AUC 0.680, AP 0.910, p=0.019, n=72
- Learned-residual classification future_any_drop_within_16cycles learned_plus_handcrafted: AUC 0.601, AP 0.864, p=0.124, n=72
- Learned-residual classification future_any_drop_within_16cycles learned_latent: AUC 0.593, AP 0.865, p=0.144, n=72
- Learned-residual delta future_any_drop_within_8cycles learned_all_minus_pca_video: delta AUC 0.279, delta rho 0.484, delta R2 NA
- Learned-residual delta future_any_drop_within_8cycles learned_plus_handcrafted_minus_pca_video: delta AUC 0.273, delta rho 0.473, delta R2 NA
- Learned-residual delta future_any_drop_within_8cycles learned_latent_minus_pca_video: delta AUC 0.259, delta rho 0.449, delta R2 NA
- Learned-residual delta future_any_drop_within_16cycles learned_plus_handcrafted_minus_pca_video: delta AUC 0.089, delta rho 0.125, delta R2 NA
- Learned-residual delta future_any_drop_within_16cycles learned_latent_minus_pca_video: delta AUC 0.081, delta rho 0.114, delta R2 NA
- Learned-residual delta future_any_drop_within_16cycles learned_all_minus_pca_video: delta AUC 0.026, delta rho 0.036, delta R2 NA
- Learned-residual delta future_any_drop_within_8cycles learned_all_minus_handcrafted_scalar: delta AUC 0.021, delta rho 0.036, delta R2 NA
- Learned-residual delta future_any_drop_within_8cycles learned_plus_handcrafted_minus_handcrafted_scalar: delta AUC 0.015, delta rho 0.025, delta R2 NA
- Learned-residual regression true_residual_energy_mean learned_residual: R2 0.999, rho 0.999, n=172
- Learned-residual regression learned_residual_mse_mean learned_residual: R2 0.999, rho 0.999, n=172
- Learned-residual regression true_residual_energy_mean learned_all: R2 0.996, rho 0.990, n=172
- Learned-residual regression learned_residual_mse_mean learned_all: R2 0.996, rho 0.990, n=172
- Guardrail: The residual CNN is trained label-free on automatic ROI crops and evaluated only through weak cycle labels. Learned embeddings support representation design and review prioritization, not deployable prediction, manual particle/front labels, or calibrated diffusion.

## Residual Dictionary Embedding Audit

- Rows/cycles: 172 / 34
- Cohorts: {'balanced_future': 72, 'selected_event_control': 52, 'transfer_ranked': 48}
- Dictionary: {'downsample': 4, 'explained_variance_ratio_sum': 0.5991450548171997, 'n_residual_samples': 800, 'rank': 8, 'stride': 2}
- Feature set sizes: {'handcrafted_scalar': 55, 'pca_video': 16, 'residual_dictionary': 39, 'residual_dictionary_plus_handcrafted': 94}
- Residual-dictionary classification future_any_drop_within_8cycles handcrafted_scalar: AUC 0.825, AP 0.865, p=0.005, n=72
- Residual-dictionary classification future_any_drop_within_8cycles residual_dictionary_plus_handcrafted: AUC 0.771, AP 0.832, p=0.005, n=72
- Residual-dictionary classification future_any_drop_within_8cycles residual_dictionary: AUC 0.663, AP 0.693, p=0.005, n=72
- Residual-dictionary classification future_any_drop_within_16cycles handcrafted_scalar: AUC 0.662, AP 0.901, p=0.040, n=72
- Residual-dictionary classification future_any_drop_within_8cycles pca_video: AUC 0.566, AP 0.620, p=0.169, n=72
- Residual-dictionary classification future_any_drop_within_16cycles residual_dictionary_plus_handcrafted: AUC 0.527, AP 0.849, p=0.393, n=72
- Residual-dictionary delta future_any_drop_within_8cycles residual_dictionary_plus_handcrafted_minus_pca_video: delta AUC 0.204, delta rho 0.354
- Residual-dictionary delta future_any_drop_within_8cycles residual_dictionary_minus_pca_video: delta AUC 0.096, delta rho 0.167
- Residual-dictionary delta future_any_drop_within_16cycles residual_dictionary_plus_handcrafted_minus_pca_video: delta AUC 0.021, delta rho 0.030
- Residual-dictionary delta future_any_drop_within_16cycles residual_dictionary_minus_pca_video: delta AUC -0.034, delta rho -0.048
- Residual-dictionary delta future_any_drop_within_8cycles residual_dictionary_plus_handcrafted_minus_handcrafted_scalar: delta AUC -0.054, delta rho -0.094
- Residual-dictionary delta future_any_drop_within_16cycles residual_dictionary_plus_handcrafted_minus_handcrafted_scalar: delta AUC -0.135, delta rho -0.189
- Residual-dictionary regression dictionary_recon_error_mse_mean residual_dictionary: R2 0.998, rho 0.999, n=172
- Residual-dictionary regression residual_energy_mean residual_dictionary: R2 0.998, rho 0.998, n=172
- Residual-dictionary regression residual_energy_mean residual_dictionary_plus_handcrafted: R2 0.992, rho 0.992, n=172
- Residual-dictionary regression dictionary_recon_error_mse_mean residual_dictionary_plus_handcrafted: R2 0.990, rho 0.988, n=172
- Guardrail: The residual dictionary is label-free and uses automatic ROI crops. It is a fast temporal-residual representation audit for model design and review prioritization, not a deployable detector, manual front label, or calibrated diffusion measurement.

## Echem Residual-Dictionary Fusion Audit

- Rows/cycles: 172 / 34
- Feature set sizes: {'acquisition_context': 11, 'echem_regime': 75, 'handcrafted_plus_echem': 148, 'handcrafted_scalar': 62, 'pca_video': 16, 'pca_video_plus_echem': 102, 'residual_dictionary': 39, 'residual_dictionary_handcrafted_echem': 187, 'residual_dictionary_plus_echem': 125, 'residual_dictionary_plus_handcrafted': 101}
- Echem-resdict classification future_any_drop_within_8cycles acquisition_context: AUC 1.000, AP 1.000, p=0.002, n=72
- Echem-resdict classification future_any_drop_within_8cycles residual_dictionary_plus_echem: AUC 0.917, AP 0.958, p=0.002, n=72
- Echem-resdict classification future_any_drop_within_8cycles residual_dictionary_handcrafted_echem: AUC 0.917, AP 0.958, p=0.002, n=72
- Echem-resdict classification future_any_drop_within_8cycles pca_video_plus_echem: AUC 0.917, AP 0.958, p=0.002, n=72
- Echem-resdict classification future_any_drop_within_8cycles handcrafted_plus_echem: AUC 0.917, AP 0.958, p=0.002, n=72
- Echem-resdict classification future_any_drop_within_8cycles handcrafted_scalar: AUC 0.829, AP 0.876, p=0.002, n=72
- Echem-resdict classification future_any_drop_within_8cycles residual_dictionary_plus_handcrafted: AUC 0.796, AP 0.823, p=NA, n=72
- Echem-resdict classification future_any_drop_within_16cycles handcrafted_plus_echem: AUC 0.781, AP 0.945, p=0.004, n=72
- Echem-resdict delta future_any_drop_within_8cycles pca_video_plus_echem_minus_pca_video: delta AUC 0.347, delta rho 0.601, delta R2 NA
- Echem-resdict delta future_any_drop_within_8cycles residual_dictionary_plus_echem_minus_residual_dictionary: delta AUC 0.248, delta rho 0.430, delta R2 NA
- Echem-resdict delta future_any_drop_within_16cycles residual_dictionary_handcrafted_echem_minus_residual_dictionary_plus_handcrafted: delta AUC 0.230, delta rho 0.324, delta R2 NA
- Echem-resdict delta future_any_drop_within_16cycles residual_dictionary_plus_echem_minus_residual_dictionary: delta AUC 0.196, delta rho 0.276, delta R2 NA
- Echem-resdict delta future_any_drop_within_16cycles handcrafted_plus_echem_minus_handcrafted_scalar: delta AUC 0.179, delta rho 0.252, delta R2 NA
- Echem-resdict delta future_any_drop_within_8cycles residual_dictionary_plus_echem_minus_echem_regime: delta AUC 0.174, delta rho 0.300, delta R2 NA
- Echem-resdict delta future_any_drop_within_16cycles pca_video_plus_echem_minus_pca_video: delta AUC 0.143, delta rho 0.201, delta R2 NA
- Echem-resdict delta future_any_drop_within_8cycles residual_dictionary_handcrafted_echem_minus_residual_dictionary_plus_handcrafted: delta AUC 0.121, delta rho 0.210, delta R2 NA
- Echem-resdict regression residual_energy_mean residual_dictionary: R2 0.992, rho 0.994, n=172
- Echem-resdict regression particle_norm_cv handcrafted_scalar: R2 0.963, rho 0.980, n=172
- Echem-resdict regression particle_norm_cv residual_dictionary_plus_handcrafted: R2 0.961, rho 0.978, n=172
- Echem-resdict regression residual_energy_mean residual_dictionary_plus_handcrafted: R2 0.979, rho 0.973, n=172
- Guardrail: Fusion uses weak cycle labels, automatic ROI crops, and echem/acquisition covariates. It tests representation conditioning and review prioritization, not deployable warning, manual particle/front labels, causal echem mechanism, or calibrated diffusion.

## Echem-Conditioned Residual Dictionary

- Rows/cycles/sources: 172 / 34 / 12
- Residual/context features: 39 / 119
- Feature set sizes: {'leave_cycle': {'conditioned_residual_dictionary': 39, 'conditioned_residual_plus_echem_context': 158, 'conditioned_residual_plus_handcrafted_echem': 216, 'echem_context': 119, 'handcrafted_plus_echem_context': 177, 'raw_residual_plus_echem_context': 158, 'residual_dictionary_raw': 39}, 'leave_source': {'conditioned_residual_dictionary': 39, 'conditioned_residual_plus_echem_context': 158, 'conditioned_residual_plus_handcrafted_echem': 216, 'echem_context': 119, 'handcrafted_plus_echem_context': 177, 'raw_residual_plus_echem_context': 158, 'residual_dictionary_raw': 39}}
- Future16 conditioned residual dictionary leave-cycle/leave-source AUC: 0.695 / 0.785
- Future16 conditioned residual+echem leave-cycle/leave-source AUC: 0.834 / 0.651
- Echem-conditioned resdict leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem: AUC 0.848, AP 0.961, p=0.003, n=72
- Echem-conditioned resdict leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_echem_context: AUC 0.834, AP 0.955, p=0.003, n=72
- Echem-conditioned resdict leave_cycle future_any_drop_within_16cycles handcrafted_plus_echem_context: AUC 0.821, AP 0.956, p=NA, n=72
- Echem-conditioned resdict leave_cycle future_any_drop_within_16cycles echem_context: AUC 0.806, AP 0.951, p=0.003, n=72
- Echem-conditioned resdict leave_cycle future_any_drop_within_16cycles raw_residual_plus_echem_context: AUC 0.746, AP 0.935, p=NA, n=72
- Echem-conditioned resdict leave_cycle future_any_drop_within_16cycles conditioned_residual_dictionary: AUC 0.695, AP 0.902, p=0.020, n=72
- Echem-conditioned resdict leave_cycle future_any_drop_within_16cycles residual_dictionary_raw: AUC 0.481, AP 0.855, p=0.561, n=72
- Echem-conditioned resdict leave_source future_any_drop_within_16cycles conditioned_residual_dictionary: AUC 0.785, AP 0.943, p=0.003, n=72
- Echem-conditioned resdict leave_source future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem: AUC 0.701, AP 0.913, p=0.010, n=72
- Echem-conditioned resdict leave_source future_any_drop_within_16cycles conditioned_residual_plus_echem_context: AUC 0.651, AP 0.891, p=0.056, n=72
- Echem-conditioned resdict delta leave_source future_any_drop_within_16cycles conditioned_residual_dictionary_minus_residual_dictionary_raw: delta AUC 0.726, delta AP 0.312, delta rho 1.022
- Echem-conditioned resdict delta leave_cycle future_any_drop_within_16cycles conditioned_residual_dictionary_minus_residual_dictionary_raw: delta AUC 0.214, delta AP 0.047, delta rho 0.301
- Echem-conditioned resdict delta leave_source future_any_drop_within_16cycles conditioned_residual_plus_echem_context_minus_raw_residual_plus_echem_context: delta AUC 0.157, delta AP 0.074, delta rho 0.221
- Echem-conditioned resdict delta leave_source future_any_drop_within_8cycles conditioned_residual_dictionary_minus_residual_dictionary_raw: delta AUC 0.145, delta AP 0.069, delta rho 0.251
- Echem-conditioned resdict delta leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_echem_context_minus_raw_residual_plus_echem_context: delta AUC 0.088, delta AP 0.020, delta rho 0.123
- Echem-conditioned resdict delta leave_source future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem_minus_handcrafted_plus_echem_context: delta AUC 0.081, delta AP 0.036, delta rho 0.114
- Echem-conditioned resdict delta leave_source future_any_drop_within_16cycles conditioned_residual_plus_handcrafted_echem_minus_conditioned_residual_plus_echem_context: delta AUC 0.049, delta AP 0.022, delta rho 0.069
- Echem-conditioned resdict delta leave_cycle future_any_drop_within_16cycles conditioned_residual_plus_echem_context_minus_echem_context: delta AUC 0.028, delta AP 0.004, delta rho 0.039
- Context fit leave_cycle resdict_pc03_std: R2 0.614, rho 0.783, residual std 0.054
- Context fit leave_cycle resdict_pc04_std: R2 0.102, rho 0.788, residual std 0.078
- Context fit leave_cycle resdict_pc01_std: R2 -0.029, rho 0.647, residual std 0.159
- Context fit leave_cycle resdict_pc03_mean: R2 -0.232, rho 0.111, residual std 0.001
- Context fit leave_cycle dictionary_recon_energy_mean: R2 -0.357, rho 0.546, residual std 2.972e-04
- Context fit leave_cycle resdict_pc05_last_minus_first: R2 -0.542, rho 0.135, residual std 0.008
- Guardrail: Echem-conditioned residual dictionary features are split-specific residuals from echem/acquisition predictions of label-free residual bases. They test whether video residual modes add signal beyond measured context, not deployable warning, manual QC, causal mechanism, or calibrated diffusion.

## Conditioned Residual Physics Atlas

- Rows/cycles/sources: 172 / 34 / 12
- Physics descriptor columns screened: 212
- Conditioned residual modes per split: {'leave_cycle': 39, 'leave_source': 39}
- Source-centered atlas leave_source resdict_pc04_mean vs temporal_diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho 0.815, raw rho 0.592
- Source-centered atlas leave_source resdict_pc04_mean vs diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho 0.815, raw rho 0.592
- Source-centered atlas leave_source resdict_pc04_mean vs roi_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho 0.815, raw rho 0.592
- Source-centered atlas leave_source resdict_pc04_mean vs temporal_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho 0.815, raw rho 0.592
- Source-centered atlas leave_source resdict_pc04_mean vs radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho 0.815, raw rho 0.592
- Source-centered atlas leave_source resdict_pc04_last_minus_first vs q70_radius2_slope_bootstrap_p50_px2_per_s (front_phase_diffusion): centered rho -0.815, raw rho -0.559
- Source-centered atlas leave_source resdict_pc04_last_minus_first vs temporal_diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho -0.810, raw rho -0.576
- Source-centered atlas leave_source resdict_pc04_last_minus_first vs diffusion_proxy_median_um2_per_s (front_phase_diffusion): centered rho -0.810, raw rho -0.576
- Source-centered atlas leave_source resdict_pc04_last_minus_first vs roi_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho -0.810, raw rho -0.576
- Source-centered atlas leave_source resdict_pc04_last_minus_first vs temporal_radius2_slope_median_px2_per_s (front_phase_diffusion): centered rho -0.810, raw rho -0.576
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc08_slope: AUC 0.821, AP 0.956, source eta2 0.096
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc08_last_minus_first: AUC 0.801, AP 0.951, source eta2 0.075
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc04_last_minus_first: AUC 0.800, AP 0.950, source eta2 0.058
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc04_slope: AUC 0.793, AP 0.942, source eta2 0.066
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc04_mean: AUC 0.786, AP 0.930, source eta2 0.067
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc07_mean: AUC 0.743, AP 0.895, source eta2 0.252
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc06_last_minus_first: AUC 0.742, AP 0.919, source eta2 0.092
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc01_last_minus_first: AUC 0.713, AP 0.924, source eta2 0.044
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc06_slope: AUC 0.706, AP 0.911, source eta2 0.071
- Residual-mode target leave_cycle future_any_drop_within_16cycles resdict_pc07_last_minus_first: AUC 0.701, AP 0.917, source eta2 0.072
- Atlas category leave_cycle echem_state: max centered |rho| 0.764, strong centered pairs 1489
- Atlas category leave_cycle degradation_trace: max centered |rho| 0.676, strong centered pairs 71
- Atlas category leave_cycle rollout_prediction: max centered |rho| 0.589, strong centered pairs 70
- Atlas category leave_cycle front_phase_diffusion: max centered |rho| 0.500, strong centered pairs 70
- Atlas category leave_cycle particle_optical: max centered |rho| 0.440, strong centered pairs 67
- Atlas category leave_cycle mask_qc: max centered |rho| 0.428, strong centered pairs 40
- Atlas category leave_source front_phase_diffusion: max centered |rho| 0.815, strong centered pairs 315
- Atlas category leave_source echem_state: max centered |rho| 0.769, strong centered pairs 1296
- Guardrail: Conditioned residual modes are split-specific, label-free video residual features; source-centered correlations reduce source/acquisition confounding but do not prove causal physics, calibrated diffusion, or deployable warning performance.

## Acquisition-Residualized Video Physics Benchmark

- Rows/cycles: 172 / 34
- Feature group sizes: {'acquisition_context': 27, 'all_video_raw': 144, 'context_plus_all_video': 171, 'context_plus_handcrafted_particle': 116, 'context_plus_residual_dictionary': 66, 'context_plus_video_pca': 43, 'echem_context': 107, 'echem_context_plus_all_video': 251, 'handcrafted_particle_raw': 89, 'residual_dictionary_raw': 39, 'video_pca_raw': 16}
- Future8 context/raw all-video/residualized all-video AUC: 1.000 / 0.756 / 0.319
- Future16 context/raw handcrafted/residualized all-video AUC: 0.716 / 0.796 / 0.620
- Acquisition-residualized metric future_any_drop_within_8cycles acquisition_context: AUC 1.000, AP 1.000, cycle-block p=0.002, n=72
- Acquisition-residualized metric future_any_drop_within_8cycles context_plus_residual_dictionary: AUC 1.000, AP 1.000, cycle-block p=NA, n=72
- Acquisition-residualized metric future_any_drop_within_8cycles context_plus_video_pca: AUC 1.000, AP 1.000, cycle-block p=NA, n=72
- Acquisition-residualized metric future_any_drop_within_8cycles residualized_residual_dictionary_plus_context_logit: AUC 1.000, AP 1.000, cycle-block p=0.002, n=72
- Acquisition-residualized metric future_any_drop_within_8cycles residualized_video_pca_plus_context_logit: AUC 1.000, AP 1.000, cycle-block p=0.002, n=72
- Acquisition-residualized metric future_any_drop_within_8cycles context_plus_all_video: AUC 0.990, AP 0.991, cycle-block p=0.002, n=72
- Acquisition-residualized metric future_any_drop_within_8cycles context_plus_handcrafted_particle: AUC 0.986, AP 0.988, cycle-block p=NA, n=72
- Acquisition-residualized metric future_any_drop_within_8cycles residualized_all_video_plus_context_logit: AUC 0.922, AP 0.959, cycle-block p=0.002, n=72
- Acquisition-residualized delta future_any_drop_within_16cycles echem_context_plus_all_video_minus_echem_context: delta AUC 0.097, delta AP 0.024, delta rho 0.136
- Acquisition-residualized delta future_any_drop_within_16cycles handcrafted_particle_raw_minus_acquisition_context: delta AUC 0.081, delta AP -1.233e-04, delta rho 0.113
- Acquisition-residualized delta future_any_drop_within_16cycles context_plus_handcrafted_particle_minus_acquisition_context: delta AUC 0.021, delta AP -0.020, delta rho 0.029
- Acquisition-residualized delta future_any_drop_within_16cycles all_video_raw_minus_acquisition_context: delta AUC 0.018, delta AP -0.018, delta rho 0.024
- Acquisition-residualized delta future_any_drop_within_8cycles context_plus_residual_dictionary_minus_acquisition_context: delta AUC 0.000, delta AP 0.000, delta rho -6.692e-04
- Acquisition-residualized delta future_any_drop_within_8cycles context_plus_video_pca_minus_acquisition_context: delta AUC 0.000, delta AP 0.000, delta rho -6.692e-04
- Acquisition-residualized delta future_any_drop_within_8cycles residualized_residual_dictionary_plus_context_logit_minus_acquisition_context: delta AUC 0.000, delta AP 0.000, delta rho -6.692e-04
- Acquisition-residualized delta future_any_drop_within_8cycles residualized_video_pca_plus_context_logit_minus_acquisition_context: delta AUC 0.000, delta AP 0.000, delta rho -6.692e-04
- Context-residual feature handcrafted_particle roi_threshold_robust_diffusion_score vs future_any_drop_within_16cycles: |rho|=0.501, direction-free AUC 0.821
- Context-residual feature handcrafted_particle threshold_robust_diffusion_score vs future_any_drop_within_16cycles: |rho|=0.501, direction-free AUC 0.821
- Context-residual feature all_video roi_threshold_robust_diffusion_score vs future_any_drop_within_16cycles: |rho|=0.501, direction-free AUC 0.821
- Context-residual feature all_video threshold_robust_diffusion_score vs future_any_drop_within_16cycles: |rho|=0.501, direction-free AUC 0.821
- Context-residual feature handcrafted_particle mask_low_area_fraction vs future_any_drop_within_8cycles: |rho|=0.477, direction-free AUC 0.596
- Context-residual feature all_video mask_low_area_fraction vs future_any_drop_within_8cycles: |rho|=0.477, direction-free AUC 0.596
- Guardrail: This is a weak-label, leave-one-cycle benchmark over automatically selected ROI embeddings. A strong acquisition-context score is treated as design/context structure, not a deployable warning model. Residualized video scores test whether particle-region video descriptors add signal after context conditioning.

## Acquisition-Residualized Video/Echem Warning Audit

- Rows/cycles/sources: 172 / 34 / 12
- Feature set sizes: {'acquisition_context': 40, 'echem_regime': 57, 'video_all': 64, 'video_embedding': 16, 'video_plus_echem': 121, 'video_scalar': 48}
- Leave-cycle future16 residualized video+echem: AUC 0.697, AP 0.903, p=0.016; acquisition-only AUC 0.727
- Leave-source future16 residualized video+echem: AUC 0.512, AP 0.860, p=0.433; acquisition-only AUC 0.697
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo video_plus_echem_raw_minus_echem_regime_raw: delta AUC 0.099, delta rho 0.140
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo video_plus_echem_raw_minus_video_all_raw: delta AUC 0.055, delta rho 0.077
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo video_plus_echem_acquisition_residualized_minus_echem_regime_raw: delta AUC 0.044, delta rho 0.062
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo video_plus_echem_raw_minus_acquisition_context_raw: delta AUC 0.025, delta rho 0.035
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo video_plus_echem_acquisition_residualized_minus_video_all_raw: delta AUC 1.110e-16, delta rho 8.917e-06
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo video_plus_echem_acquisition_residualized_minus_acquisition_context_raw: delta AUC -0.030, delta rho -0.043
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo echem_regime_acquisition_residualized_minus_acquisition_context_raw: delta AUC -0.041, delta rho -0.058
- Acquisition-residualized video/echem delta future_any_drop_within_16cycles cycleNo video_all_acquisition_residualized_minus_acquisition_context_raw: delta AUC -0.258, delta rho -0.364
- Guardrail: This audit residualizes candidate video/echem features against acquisition/context inside each held-out fold. It tests context-resistant weak-label signal for prioritizing next analyses, not deployable warning, causal mechanism, manual QC labels, or calibrated diffusion.

## Source-Domain Video/Echem Adaptation Audit

- Rows/cycles/sources: 172 / 34 / 12
- Feature set sizes: {'acquisition_context': 40, 'echem_regime': 82, 'video_all': 64, 'video_plus_echem': 146}
- Best leave-source method: video_plus_echem source_centered AUC 0.737, AP 0.931, p=0.006
- Acquisition-only / source-centered video+echem / CORAL video+echem AUC: 0.697 / 0.737 / 0.420
- Source-domain delta video_plus_echem source_centered: delta AUC 0.040, delta AP 0.009, delta rho 0.056
- Source-domain delta acquisition_context raw: delta AUC 0.000, delta AP 0.000, delta rho 0.000
- Source-domain delta echem_regime source_centered: delta AUC -0.044, delta AP -0.006, delta rho -0.062
- Source-domain delta video_plus_echem raw: delta AUC -0.102, delta AP -0.054, delta rho -0.143
- Source-domain delta video_plus_echem acq_resid: delta AUC -0.102, delta AP -0.047, delta rho -0.143
- Source-domain delta echem_regime acq_resid: delta AUC -0.115, delta AP -0.050, delta rho -0.161
- Source-domain delta video_all acq_resid: delta AUC -0.120, delta AP -0.066, delta rho -0.169
- Source-domain delta video_all source_centered: delta AUC -0.174, delta AP -0.068, delta rho -0.245
- Source 17_c2_x10_HighHighCOV_150723: rows/cycles 30/10, future16 fraction 1.000, mean feature z-shift 0.405
- Source 9_c2_x10_010723: rows/cycles 9/3, future16 fraction 1.000, mean feature z-shift 0.618
- Source 12_c2_x10_070723: rows/cycles 6/2, future16 fraction 1.000, mean feature z-shift 0.559
- Source 11_c2_x10_050723: rows/cycles 3/1, future16 fraction 1.000, mean feature z-shift 0.696
- Source 16_c2_x10_HighHighCOV_130723: rows/cycles 9/3, future16 fraction 0.667, mean feature z-shift 0.533
- Source 5_c2_x10_260623: rows/cycles 6/2, future16 fraction 0.500, mean feature z-shift 0.769
- Guardrail: This is an unlabeled target-source adaptation audit over weak labels and automatic ROI descriptors. Source centering/CORAL use held-out source feature distributions but not held-out labels. Results diagnose domain shift; they are not deployable warnings or causal physics proof.

## Source-Balanced Video/Echem Transfer Audit

- Rows/cycles/sources: 172 / 34 / 12
- Feature set sizes: {'acquisition_context': 28, 'echem_regime': 55, 'video_all': 64, 'video_plus_echem': 119}
- Future16 source-rank weighted video+echem: AUC 0.614, AP 0.868, p=0.068
- Future16 acquisition raw / echem source-rank / video+echem source-rank AUC: 0.704 / 0.642 / 0.614
- Source-balanced delta future_any_drop_within_16cycles video_plus_echem_source_rank_weighted_minus_video_all_raw_unweighted: delta AUC 0.160, delta rho 0.225
- Source-balanced delta future_any_drop_within_16cycles video_plus_echem_source_rank_weighted_minus_video_plus_echem_raw_unweighted: delta AUC 0.020, delta rho 0.028
- Source-balanced delta future_any_drop_within_16cycles video_plus_echem_source_rank_unweighted_minus_video_plus_echem_raw_unweighted: delta AUC 0.020, delta rho 0.028
- Source-balanced delta future_any_drop_within_16cycles video_plus_echem_raw_weighted_minus_video_plus_echem_raw_unweighted: delta AUC 0.000, delta rho 0.000
- Source-balanced delta future_any_drop_within_16cycles video_plus_echem_source_rank_weighted_minus_echem_regime_raw_unweighted: delta AUC -0.018, delta rho -0.025
- Source-balanced delta future_any_drop_within_16cycles video_plus_echem_source_rank_weighted_minus_acquisition_context_raw_unweighted: delta AUC -0.090, delta rho -0.127
- Source-balanced delta future_any_drop_within_8cycles video_plus_echem_raw_weighted_minus_video_plus_echem_raw_unweighted: delta AUC 0.000, delta rho 0.000
- Source-balanced delta future_any_drop_within_8cycles video_plus_echem_source_rank_weighted_minus_echem_regime_raw_unweighted: delta AUC -0.105, delta rho -0.182
- Source 17_c2_x10_HighHighCOV_150723: rows/cycles 66/10, future16 labeled/positive/negative 30/30/0, future16 fraction 1.000
- Source 18_c2_xN_HighHighCOV_170723: rows/cycles 21/3, future16 labeled/positive/negative 3/0/3, future16 fraction 0.000
- Source 10_c2_x10_030723: rows/cycles 14/3, future16 labeled/positive/negative 0/0/0, future16 fraction NA
- Source 13_c2_x6_100723: rows/cycles 14/2, future16 labeled/positive/negative 0/0/0, future16 fraction NA
- Source 7_c2_x10_290623: rows/cycles 14/3, future16 labeled/positive/negative 0/0/0, future16 fraction NA
- Source 5_c2_x10_260623: rows/cycles 10/2, future16 labeled/positive/negative 6/3/3, future16 fraction 0.500
- Guardrail: Leave-source labels are highly source-composition imbalanced. Source weighting and within-source rank normalization test domain robustness and review-prioritization only; they do not validate source-transferable warning, causal degradation mechanisms, manual QC labels, or calibrated diffusion.

## Source-Invariant Video/Echem Transfer Audit

- Rows/cycles/sources: 172 / 34 / 12
- Methods: ['raw', 'source_mean_resid_1', 'source_mean_resid_2', 'source_mean_resid_4', 'source_mean_resid_8', 'source_confound_filter_0.10', 'source_confound_filter_0.25', 'source_confound_filter_0.50']
- Future16 acquisition raw / video+echem raw / best video+echem invariant AUC: 0.745 / 0.612 / 0.729
- Best video+echem method: source_mean_resid_4 AP 0.927, p=0.004
- Best video-only future16 method: source_confound_filter_0.50 AUC 0.770, AP 0.919, p=0.004
- Best video+echem future8 invariant method: source_mean_resid_1 AUC 0.784, AP 0.832
- Source-invariant delta future_any_drop_within_16cycles video_all_source_confound_filter_0.50_minus_same_raw: delta AUC 0.281, delta AP 0.104, delta rho 0.395
- Source-invariant delta future_any_drop_within_16cycles video_all_source_mean_resid_2_minus_same_raw: delta AUC 0.240, delta AP 0.107, delta rho 0.337
- Source-invariant delta future_any_drop_within_16cycles video_all_source_mean_resid_8_minus_same_raw: delta AUC 0.129, delta AP 0.032, delta rho 0.181
- Source-invariant delta future_any_drop_within_16cycles video_plus_echem_source_mean_resid_4_minus_same_raw: delta AUC 0.117, delta AP 0.069, delta rho 0.165
- Source-invariant delta future_any_drop_within_16cycles video_plus_echem_source_mean_resid_8_minus_same_raw: delta AUC 0.106, delta AP 0.060, delta rho 0.150
- Source-invariant delta future_any_drop_within_16cycles video_all_source_mean_resid_1_minus_same_raw: delta AUC 0.095, delta AP 0.040, delta rho 0.133
- Source-invariant delta future_any_drop_within_16cycles video_plus_echem_source_confound_filter_0.50_minus_same_raw: delta AUC 0.091, delta AP 0.041, delta rho 0.128
- Source-invariant delta future_any_drop_within_16cycles video_all_source_mean_resid_4_minus_same_raw: delta AUC 0.090, delta AP 0.035, delta rho 0.127
- Source 17_c2_x10_HighHighCOV_150723: rows/cycles 66/10, future16 fraction 1.000, mean video/echem z-shift 0.589
- Source 16_c2_x10_HighHighCOV_130723: rows/cycles 9/3, future16 fraction 0.667, mean video/echem z-shift 0.598
- Source 9_c2_x10_010723: rows/cycles 9/3, future16 fraction 1.000, mean video/echem z-shift 0.376
- Source 5_c2_x10_260623: rows/cycles 10/2, future16 fraction 0.500, mean video/echem z-shift 0.509
- Source 12_c2_x10_070723: rows/cycles 6/2, future16 fraction 1.000, mean video/echem z-shift 0.344
- Source 18_c2_xN_HighHighCOV_170723: rows/cycles 21/3, future16 fraction 0.000, mean video/echem z-shift 0.449
- Guardrail: Source-invariant projections and source-confound filters are trained without held-out source labels, but labels remain source-composition imbalanced. These results test robustness for review prioritization only; they do not validate source-transferable warning, causal degradation mechanisms, manual QC labels, or calibrated diffusion.

## Source-Invariant Physical Family Audit

- Rows/cycles/sources: 172 / 34 / 12
- Feature family sizes: {'all_video': 64, 'handcrafted_particle': 48, 'norm_heterogeneity': 8, 'particle_gradient': 10, 'particle_intensity': 20, 'particle_vs_context': 10, 'video_embedding': 16}
- Best future16 family/method: all_video source_confound_filter_0.50 AUC 0.770, AP 0.919, p=0.002
- Norm heterogeneity source_mean_resid_4: AUC 0.738, AP 0.917, p=0.002
- Particle-vs-context source_mean_resid_4: AUC 0.703, AP 0.903, p=0.010
- Raw video embedding future16: AUC 0.462, AP 0.753; best future8 family/method all_video source_mean_resid_2 AUC 0.846
- Family delta future_any_drop_within_16cycles norm_heterogeneity_source_mean_resid_4_minus_same_raw: delta AUC 0.433, delta AP 0.244, delta rho 0.609
- Family delta future_any_drop_within_16cycles particle_gradient_source_confound_filter_0.50_minus_same_raw: delta AUC 0.407, delta AP 0.164, delta rho 0.573
- Family delta future_any_drop_within_16cycles norm_heterogeneity_source_mean_resid_2_minus_same_raw: delta AUC 0.329, delta AP 0.169, delta rho 0.462
- Family delta future_any_drop_within_16cycles particle_vs_context_source_mean_resid_4_minus_same_raw: delta AUC 0.289, delta AP 0.158, delta rho 0.406
- Family delta future_any_drop_within_16cycles all_video_source_confound_filter_0.50_minus_same_raw: delta AUC 0.281, delta AP 0.104, delta rho 0.395
- Family delta future_any_drop_within_16cycles norm_heterogeneity_source_mean_resid_4_minus_all_video_raw: delta AUC 0.249, delta AP 0.103, delta rho 0.351
- Family delta future_any_drop_within_16cycles all_video_source_mean_resid_2_minus_same_raw: delta AUC 0.240, delta AP 0.107, delta rho 0.337
- Family delta future_any_drop_within_16cycles particle_vs_context_source_mean_resid_2_minus_same_raw: delta AUC 0.239, delta AP 0.147, delta rho 0.336
- Source-confounded feature all_video particle_norm_mean: eta2 0.932
- Source-confounded feature handcrafted_particle particle_norm_mean: eta2 0.932
- Source-confounded feature norm_heterogeneity particle_norm_mean: eta2 0.932
- Source-confounded feature handcrafted_particle particle_norm_max: eta2 0.874
- Source-confounded feature all_video particle_norm_max: eta2 0.874
- Source-confounded feature norm_heterogeneity particle_norm_max: eta2 0.874
- Source-confounded feature all_video particle_norm_min: eta2 0.840
- Source-confounded feature handcrafted_particle particle_norm_min: eta2 0.840
- Guardrail: Physical family readouts use automatic particle-region descriptors and weak future labels under leave-source splits. They identify candidate physics families for review prioritization only; source/outcome imbalance, automatic masks, missing manual QC, and uncalibrated optical-front diffusion remain guardrails.

## Source-Invariant Interpretable Feature Audit

- Rows/cycles/sources: 72 / 24 / 9
- Feature family sizes: {'norm_heterogeneity': 8, 'particle_gradient': 10, 'particle_intensity': 20, 'particle_vs_context': 10}
- Top univariate descriptor: particle_vs_context_mean_diff_positive_fraction (particle_vs_context), orientation lower_in_positive, AUC 0.769, AP 0.904, eta2 0.390
- Top single-feature transfer set: single::particle_vs_context_mean_diff_positive_fraction raw AUC 0.727, AP 0.880, p=0.008
- Top small-combo transfer set: trio::particle_vs_context_mean_diff_positive_fraction+particle_mean_last_minus_first+particle_gradient_diff_q90 raw AUC 0.750, AP 0.905, p=0.002
- Exact feature particle_vs_context particle_vs_context_mean_diff_positive_fraction: oriented AUC 0.769, direction lower_in_positive, eta2 0.390, median pos-neg -0.053
- Exact feature particle_intensity particle_std_diff_positive_fraction: oriented AUC 0.723, direction lower_in_positive, eta2 0.184, median pos-neg -0.011
- Exact feature particle_vs_context particle_vs_context_mean_last_minus_first: oriented AUC 0.709, direction higher_in_positive, eta2 0.773, median pos-neg 0.001
- Exact feature particle_intensity particle_mean_diff_positive_fraction: oriented AUC 0.707, direction lower_in_positive, eta2 0.306, median pos-neg -0.021
- Exact feature particle_intensity particle_mean_last_minus_first: oriented AUC 0.683, direction higher_in_positive, eta2 0.710, median pos-neg 0.002
- Exact feature particle_vs_context particle_vs_context_mean_slope: oriented AUC 0.683, direction higher_in_positive, eta2 0.800, median pos-neg 0.002
- Exact feature norm_heterogeneity particle_norm_mean: oriented AUC 0.674, direction higher_in_positive, eta2 0.868, median pos-neg 0.010
- Exact feature particle_intensity particle_mean_slope: oriented AUC 0.667, direction higher_in_positive, eta2 0.731, median pos-neg 0.002
- Feature set trio::particle_vs_context_mean_diff_positive_fraction+particle_mean_last_minus_first+particle_gradient_diff_q90 raw: n_features 3, AUC 0.750, AP 0.905, p=0.002
- Feature set trio::particle_vs_context_mean_diff_positive_fraction+particle_mean_last_minus_first+particle_gradient_diff_abs_mean raw: n_features 3, AUC 0.747, AP 0.905, p=0.002
- Feature set pair::particle_vs_context_mean_diff_positive_fraction+particle_prior_area_fraction raw: n_features 2, AUC 0.746, AP 0.897, p=0.004
- Feature set single::particle_vs_context_mean_diff_positive_fraction raw: n_features 1, AUC 0.727, AP 0.880, p=0.008
- Feature set single::particle_vs_context_mean_diff_positive_fraction source_confound_filter_0.50: n_features 1, AUC 0.727, AP 0.880, p=0.008
- Feature set pair::particle_vs_context_mean_diff_positive_fraction+particle_norm_mean source_confound_filter_0.50: n_features 2, AUC 0.727, AP 0.880, p=0.008
- Feature set pair::particle_vs_context_mean_diff_positive_fraction+particle_norm_max source_confound_filter_0.50: n_features 2, AUC 0.727, AP 0.880, p=0.008
- Feature set pair::particle_vs_context_mean_diff_positive_fraction+particle_norm_min source_confound_filter_0.50: n_features 2, AUC 0.727, AP 0.880, p=0.008
- Guardrail: Exact-feature readouts are automatic particle-region descriptors evaluated against weak future16 labels under leave-source splits. They are hypothesis-prioritization signals only; source imbalance, acquisition context, automatic masks, and absent manual QC still block mechanistic claims.

## Exact Feature Mechanism Consistency Audit

- Rows/cycles/sources: 172 / 34 / 12
- Exact optical-loss composite future16 metric: AUC 0.853, AP 0.960, median positive-negative 0.730, source eta2 0.513
- Primary low context-change metric: AUC 0.769, AP 0.904, source eta2 0.317
- Front-contraction composite future16 metric: AUC 0.648, AP 0.888, p=0.081
- Composite exact-loss vs radius2 slope: raw rho 0.355, source-residual rho 0.404; primary descriptor vs radius2 source-residual rho -0.047
- Mechanism correlation exact_optical_loss_score vs radius2_slope_median_px2_per_s: rho 0.355, p=0.002, source-resid rho 0.404, source-resid p=4.309e-04
- Mechanism correlation exact_optical_loss_score vs front_contraction_score: rho -0.136, p=0.254, source-resid rho -0.398, source-resid p=5.333e-04
- Mechanism correlation exact_optical_loss_score vs positive_D_fraction: rho -0.126, p=0.291, source-resid rho 0.313, source-resid p=0.008
- Mechanism correlation exact_low_context_change_score vs radius2_slope_median_px2_per_s: rho 0.121, p=0.311, source-resid rho 0.049, source-resid p=0.682
- Mechanism correlation particle_vs_context_mean_diff_positive_fraction vs radius2_slope_median_px2_per_s: rho -0.121, p=0.311, source-resid rho -0.047, source-resid p=0.693
- Mechanism correlation exact_low_context_change_score vs positive_D_fraction: rho -0.071, p=0.553, source-resid rho 0.109, source-resid p=0.364
- Mechanism correlation particle_vs_context_mean_diff_positive_fraction vs positive_D_fraction: rho 0.071, p=0.553, source-resid rho -0.108, source-resid p=0.365
- Mechanism correlation exact_low_context_change_score vs front_contraction_score: rho -0.049, p=0.681, source-resid rho -0.078, source-resid p=0.516
- High exact-loss stratum shift exact_optical_loss_score: median high-low 1.322, p=1.449e-15
- High exact-loss stratum shift particle_vs_context_mean_diff_positive_fraction: median high-low -0.074, p=1.147e-12
- High exact-loss stratum shift exact_low_context_change_score: median high-low 1.498, p=1.147e-12
- High exact-loss stratum shift particle_std_diff_positive_fraction: median high-low -0.084, p=1.676e-11
- High exact-loss stratum shift particle_gradient_diff_q90: median high-low 0.001, p=8.784e-09
- High exact-loss stratum shift transferred_masked_residual_signature: median high-low 0.614, p=5.290e-05
- High exact-loss stratum shift radius2_slope_median_px2_per_s: median high-low 0.003, p=9.685e-04
- High exact-loss stratum shift diffusion_proxy_median_um2_per_s: median high-low 5.932e-06, p=9.685e-04
- Guardrail: Mechanism consistency joins automatic particle descriptors to automatic front, rollout, echem, and diffusion-proxy outputs. It tests whether the exact feature behaves like optical loss/contraction, but all front/diffusion quantities remain apparent proxies pending manual QC and calibration.

## Signed Optical-Loss Mechanism Audit

- Rows/eval rows/cycles/sources: 172 / 72 / 24 / 9
- Axis inputs: {'echem_degraded_state_axis': ['capacity_fraction_of_first', 'shape_charge_mAh_neg_abs', 'shape_V_q10', 'shape_V_q05', 'neg_dq_abs_lowV_frac', 'dqdv_integral_asymmetry', 'cycle_state_pc3', 'cycle_state_pc4'], 'front_contraction_axis': ['radius2_slope_median_px2_per_s', 'radius2_slope_positive_fraction', 'diffusion_proxy_median_um2_per_s', 'phase_slope_positive_fraction', 'threshold_robust_diffusion_score'], 'rollout_difficulty_axis': ['transferred_masked_residual_signature', 'persistence_particle_mse_fraction_of_full_mean', 'low_rank_dmd_particle_mse_fraction_of_full_mean', 'velocity_particle_mse_fraction_of_full_mean', 'object_mean_abs_z'], 'signed_optical_loss_axis': ['particle_vs_context_mean_diff_positive_fraction', 'particle_std_diff_positive_fraction', 'particle_mean_diff_positive_fraction', 'particle_gradient_diff_positive_fraction', 'roi_norm_mean_delta_last_minus_first', 'particle_prior_area_fraction']}
- Future16 combined/optical/echem axis AUCs: 0.989 / 0.815 / 0.821; source eta2 combined/optical 0.556 / 0.684
- Best future16 leave-source axis model: all_loss_mechanism_axes AUC 0.927, AP 0.984, rho 0.601
- Best future8 leave-source axis model: optical_plus_front AUC 0.797, AP 0.694, rho 0.515
- Signed-loss axis future_any_drop_within_16cycles combined_loss_mechanism_axis: AUC 0.989, median positive-negative 0.575, p=6.795e-09, eta2 0.556
- Signed-loss axis future_any_drop_within_16cycles echem_degraded_state_axis: AUC 0.821, median positive-negative 0.849, p=1.434e-04, eta2 0.055
- Signed-loss axis future_any_drop_within_16cycles signed_optical_loss_axis: AUC 0.815, median positive-negative 0.809, p=1.915e-04, eta2 0.684
- Signed-loss axis future_any_drop_within_16cycles rollout_difficulty_axis: AUC 0.632, median positive-negative 0.204, p=0.120, eta2 0.147
- Signed-loss axis future_any_drop_within_16cycles front_contraction_axis: AUC 0.558, median positive-negative 0.039, p=0.497, eta2 0.292
- Mechanism mode rollout_difficulty_dominant: n=3, cycles=1, sources=1, future8/future16 1.000/1.000, median residual 40.972
- Mechanism mode echem_degraded_state_dominant: n=39, cycles=14, sources=6, future8/future16 0.718/0.795, median residual 0.558
- Mechanism mode front_contraction_dominant: n=30, cycles=11, sources=7, future8/future16 0.167/0.767, median residual 0.487
- Top signed-loss candidate cycle150_rank13_obj3: cycle 150, source 17_c2_x10_HighHighCOV_150723, combined axis 1.174, future8/future16 1/1
- Top signed-loss candidate cycle150_rank13_obj2: cycle 150, source 17_c2_x10_HighHighCOV_150723, combined axis 1.136, future8/future16 1/1
- Top signed-loss candidate cycle150_rank13_obj1: cycle 150, source 17_c2_x10_HighHighCOV_150723, combined axis 1.102, future8/future16 1/1
- Top signed-loss candidate cycle154_rank16_obj2: cycle 154, source 17_c2_x10_HighHighCOV_150723, combined axis 0.527, future8/future16 1/1
- Top signed-loss candidate cycle104_rank6_obj2: cycle 104, source 11_c2_x10_050723, combined axis 0.462, future8/future16 0/1
- Top signed-loss candidate cycle146_rank1_obj3: cycle 146, source 17_c2_x10_HighHighCOV_150723, combined axis 0.457, future8/future16 0/1
- Guardrail: Signed optical-loss axes are computed from automatic ROI/video/echem descriptors and weak future labels. They support mechanism triage for optical loss/contraction versus front expansion, but they do not validate manual particle identity, front masks, calibrated diffusion, or deployable warnings.

## Signed-Loss Source Robustness Audit

- Rows/future16 labeled/cycles/sources: 172 / 72 / 34 / 12
- Combined axis raw/source-mean/within-source-rank AUC: 0.989 / 0.942 / 0.551
- Optical axis raw/source-mean/within-source-rank AUC: 0.815 / 0.774 / 0.514
- Echem degraded source-residual AUC/AP: 0.716 / 0.915
- Optical within-source rank global/within-source permutation p: 0.448 / 0.036
- Robustness combined_loss_mechanism_axis raw: AUC 0.989, AP 0.997, source eta2 0.556, balanced AUC mean 0.989, within-source p 0.003
- Robustness combined_loss_mechanism_axis source_mean_only: AUC 0.942, AP 0.980, source eta2 1.000, balanced AUC mean 0.967, within-source p 1.000
- Robustness combined_loss_mechanism_axis within_source_centered_z: AUC 0.632, AP 0.866, source eta2 0.019, balanced AUC mean 0.603, within-source p 0.003
- Robustness combined_loss_mechanism_axis source_residual: AUC 0.558, AP 0.832, source eta2 0.104, balanced AUC mean 0.515, within-source p 0.003
- Robustness combined_loss_mechanism_axis within_source_rank: AUC 0.551, AP 0.813, source eta2 0.053, balanced AUC mean 0.552, within-source p 0.003
- Robustness echem_degraded_state_axis raw: AUC 0.821, AP 0.939, source eta2 0.055, balanced AUC mean 0.793, within-source p 0.003
- Robustness echem_degraded_state_axis source_residual: AUC 0.716, AP 0.915, source eta2 0.007, balanced AUC mean 0.553, within-source p 0.003
- Robustness echem_degraded_state_axis within_source_centered_z: AUC 0.674, AP 0.882, source eta2 0.059, balanced AUC mean 0.557, within-source p 0.003
- Robustness echem_degraded_state_axis within_source_rank: AUC 0.489, AP 0.818, source eta2 0.102, balanced AUC mean 0.511, within-source p 1.000
- Robustness echem_degraded_state_axis source_mean_only: AUC 0.384, AP 0.807, source eta2 1.000, balanced AUC mean 0.178, within-source p 1.000
- Robustness signed_optical_loss_axis raw: AUC 0.815, AP 0.940, source eta2 0.684, balanced AUC mean 0.866, within-source p 0.008
- Robustness signed_optical_loss_axis source_mean_only: AUC 0.774, AP 0.923, source eta2 1.000, balanced AUC mean 0.838, within-source p 1.000
- Source influence front_contraction_axis drop 18_c2_xN_HighHighCOV_170723: full AUC 0.558, drop-source AUC 0.461, delta -0.097
- Source influence signed_optical_loss_axis drop 15_c2_x5_HighCOV_120723: full AUC 0.815, drop-source AUC 0.769, delta -0.046
- Source influence signed_optical_loss_axis drop 18_c2_xN_HighHighCOV_170723: full AUC 0.815, drop-source AUC 0.769, delta -0.046
- Source influence echem_degraded_state_axis drop 5_c2_x10_260623: full AUC 0.821, drop-source AUC 0.778, delta -0.043
- Source influence signed_optical_loss_axis drop 16_c2_x10_HighHighCOV_130723: full AUC 0.815, drop-source AUC 0.775, delta -0.041
- Source influence echem_degraded_state_axis drop 15_c2_x5_HighCOV_120723: full AUC 0.821, drop-source AUC 0.789, delta -0.032
- Guardrail: Source robustness transforms distinguish within-source signal from source-level composition. High raw AUC with high source-mean-only AUC or weak within-source permutation evidence should be treated as source/context-sensitive review evidence, not a source-independent detector.

## Echem/Optical Source-Residual Audit

- Rows/future16 labeled/cycles/sources: 172 / 72 / 34 / 12
- Direct future16 echem / optical / echem+optical residual AUCs: 0.716 / 0.656 / 0.774
- Direct echem+optical+front residual AUC/AP/source eta2: 0.809 / 0.951 / 0.006
- Leave-source echem+optical residual model AUC/AP: 0.708 / 0.922
- Direct source-residual set raw_axes_guardrail: AUC 0.989, AP 0.997, source eta2 0.556, p=2.291e-11
- Direct source-residual set source_mean_only_guardrail: AUC 0.942, AP 0.980, source eta2 1.000, p=7.825e-10
- Direct source-residual set echem_optical_front_residual: AUC 0.809, AP 0.951, source eta2 0.006, p=1.329e-04
- Direct source-residual set echem_plus_optical_source_residual: AUC 0.774, AP 0.943, source eta2 0.002, p=8.148e-04
- Direct source-residual set echem_source_residual: AUC 0.716, AP 0.915, source eta2 0.007, p=0.009
- Direct source-residual set optical_source_residual: AUC 0.656, AP 0.899, source eta2 0.050, p=0.064
- Direct source-residual set echem_plus_optical_centered_z: AUC 0.628, AP 0.863, source eta2 0.021, p=0.130
- Direct source-residual set all_source_residual_axes: AUC 0.604, AP 0.878, source eta2 0.030, p=0.222
- Leave-source residual model raw_axes_guardrail: AUC 0.926, AP 0.983, rho 0.600
- Leave-source residual model echem_plus_optical_source_residual: AUC 0.708, AP 0.922, rho 0.292
- Leave-source residual model optical_source_residual: AUC 0.611, AP 0.882, rho 0.156
- Leave-source residual model source_mean_only_guardrail: AUC 0.584, AP 0.863, rho 0.123
- Leave-source residual model echem_source_residual: AUC 0.558, AP 0.880, rho 0.082
- Leave-source residual model echem_optical_front_residual: AUC 0.538, AP 0.868, rho 0.053
- Leave-source residual model front_source_residual: AUC 0.505, AP 0.864, rho 0.007
- Leave-source residual model echem_plus_optical_centered_z: AUC 0.473, AP 0.774, rho -0.039
- Residual rule future_any_drop_within_8cycles echem_ge_q50: precision 0.750, recall 0.750, lift 1.500, source-positive hits 3
- Residual rule future_any_drop_within_8cycles echem_ge_q50: precision 0.750, recall 0.750, lift 1.500, source-positive hits 3
- Residual rule future_any_drop_within_8cycles echem_ge_q50: precision 0.750, recall 0.750, lift 1.500, source-positive hits 3
- Residual rule future_any_drop_within_8cycles echem_ge_q50: precision 0.750, recall 0.750, lift 1.500, source-positive hits 3
- Residual rule future_any_drop_within_8cycles echem_ge_q50_OR_optical_ge_q75: precision 0.667, recall 0.833, lift 1.333, source-positive hits 3
- Residual rule future_any_drop_within_8cycles echem_ge_q50_OR_optical_ge_q70: precision 0.646, recall 0.861, lift 1.292, source-positive hits 3
- Guardrail: Source-residual and within-source-rank transforms use unlabeled source distribution information and weak future labels. They test whether echem state contextualizes optical loss after source normalization; they are not deployable warning models or causal mechanism proof.

## Invariant Physics Rule Discovery

- Rows/eval rows/cycles/sources: 172 / 72 / 24 / 9
- Target/positive rate/candidate rules: future_any_drop_within_16cycles / 0.792 / 171
- Best rule: low(particle_std_diff_positive_fraction) with precision 0.889, recall 0.421, lift 1.123, binary AUC 0.611, Fisher p=0.099
- Best-rule source support: hits in 7 sources, positive hits in 6 sources, max feature source eta2 0.184
- Rule low(particle_std_diff_positive_fraction): covered 27/72, precision 0.889, recall 0.421, lift 1.123, positive-source hits 6
- Rule low(particle_vs_context_mean_diff_positive_fraction): covered 21/72, precision 0.905, recall 0.333, lift 1.143, positive-source hits 5
- Rule low(shape_V_q10): covered 18/72, precision 1.000, recall 0.316, lift 1.263, positive-source hits 4
- Rule low(shape_dVdt_slope): covered 18/72, precision 1.000, recall 0.316, lift 1.263, positive-source hits 4
- Rule low(shape_V_q10) AND low(shape_dVdt_slope): covered 18/72, precision 1.000, recall 0.316, lift 1.263, positive-source hits 4
- Rule low(particle_vs_context_mean_diff_positive_fraction) AND low(particle_std_diff_positive_fraction): covered 13/72, precision 0.923, recall 0.211, lift 1.166, positive-source hits 5
- Rule low(shape_V_q05): covered 15/72, precision 1.000, recall 0.263, lift 1.263, positive-source hits 4
- Rule low(neg_dq_abs_total_mAh): covered 15/72, precision 1.000, recall 0.263, lift 1.263, positive-source hits 4
- Oriented feature low(capacity_fraction_of_first): AUC 0.893, AP 0.965, p=3.874e-05, source eta2 0.519
- Oriented feature low(shape_charge_mAh_neg_abs): AUC 0.893, AP 0.965, p=3.874e-05, source eta2 0.519
- Oriented feature high(neg_dq_abs_lowV_frac): AUC 0.875, AP 0.971, p=8.588e-05, source eta2 0.122
- Oriented feature low(all_dq_abs_total_mAh): AUC 0.875, AP 0.961, p=8.588e-05, source eta2 0.547
- Oriented feature low(neg_dq_abs_total_mAh): AUC 0.875, AP 0.961, p=8.588e-05, source eta2 0.507
- Oriented feature low(shape_charge_mAh_abs): AUC 0.875, AP 0.961, p=8.588e-05, source eta2 0.825
- Oriented feature low(pos_dq_abs_total_mAh): AUC 0.857, AP 0.963, p=1.841e-04, source eta2 0.858
- Oriented feature low(shape_V_q10): AUC 0.839, AP 0.961, p=3.818e-04, source eta2 0.196
- Guardrail: Rules use automatic ROI masks, weak future labels, and data-derived thresholds under leave-source evaluation. They are sparse review-prioritization hypotheses only; source/outcome imbalance, acquisition coupling, missing manual QC, and uncalibrated diffusion/front proxies remain hard claim limits.

## Agentic Current Hypothesis Tournament

- Hypotheses ranked: 8
- Paper-inspired roles: {'robin': 'lab-in-the-loop agents that generate, analyze, and update hypotheses', 'co_scientist': 'asynchronous generation, critique, tournament-style ranking, and refinement', 'empirical_software': 'guarded code/execution workflow with explicit evaluation criteria'}
- Rank 1 Echem-conditioned video residuals are the best longer-horizon weak-label signal: score 0.598; next experiment: Run an acquisition-residualized video+echem future16 audit with leave-source and leave-cycle splits.
- Rank 2 Short-horizon future8 labels are acquisition/context dominated: score 0.594; next experiment: Build a residualized future8 benchmark that removes acquisition/context signal before testing video physics features.
- Rank 3 Manual QC on the top cycle-156 panel is the highest-yield lab-in-the-loop step: score 0.557; next experiment: Complete the cycle-156 manual QC mini-batch and rerun manual-QC-gated front/echem effects.
- Next spec 1: script tier4_acquisition_residualized_video_echem_warning.py with success evidence: Require the video+echem gain to persist under cycle-balanced acquisition residualization and source-cohort holdouts.
- Next spec 2: script tier4_residualized_future8_video_physics_benchmark.py with success evidence: A future8 model must remain above null after acquisition, frame-count, cohort-role, and source-movie residualization.
- Next spec 3: script tier4_manual_qc_gated_front_effects.py with success evidence: If manual review rejects the possible-accept candidates or accepts the artifact foil, current automatic physics support is over-weighted.
- Guardrail: This tournament ranks next analyses from existing automatic evidence. It does not create manual QC labels, does not validate a deployable degradation detector, and does not license calibrated diffusion claims.
- Acquisition-context residual feature radius2_slope_median_px2_per_s: median positive-negative 9.596e-05, AUC 0.552, MW p=0.447
- Acquisition-context residual feature diffusion_proxy_median_um2_per_s: median positive-negative 2.211e-07, AUC 0.552, MW p=0.447
- Acquisition-context residual feature q70_radius2_slope_bootstrap_p50_px2_per_s: median positive-negative -4.025e-05, AUC 0.552, MW p=0.454
- Acquisition-context residual feature threshold_robust_diffusion_score: median positive-negative -0.002, AUC 0.549, MW p=0.481
- Acquisition-context residual feature phase_slope_positive_fraction: median positive-negative 0.086, AUC 0.543, MW p=0.532
- Acquisition-context residual feature threshold_robust_phase_score: median positive-negative -0.002, AUC 0.522, MW p=0.748
- Spatial region test y_bin: chi2=0.333, p=0.846, regions=3
- Spatial region test xy_region: chi2=1.511, p=0.982, regions=8
- Spatial region test x_bin: chi2=0.000, p=1.000, regions=3
- Temporal directionality future8 model: logistic_l2 AUC 0.799, AP 0.793; shift-null mean AUC 0.593, p95 0.775, empirical p=0.042
- Temporal directionality reversed/past8 guardrails: reversed best AUC 0.750; past8 positives 3 and evaluable AUC NA
- Temporal future8 feature q70_radius2_slope_bootstrap_p95_px2_per_s: AUC 0.731, median positive-negative 0.002, MW p=7.434e-04
- Temporal future8 feature diffusion_proxy_median_um2_per_s: AUC 0.717, median positive-negative 2.622e-06, MW p=0.002
- Temporal future8 feature radius2_slope_median_px2_per_s: AUC 0.717, median positive-negative 0.001, MW p=0.002
- Temporal future8 feature default_q70_diffusion_proxy_um2_per_s: AUC 0.712, median positive-negative 2.430e-06, MW p=0.002
- Temporal future8 feature q70_radius2_slope_bootstrap_p50_px2_per_s: AUC 0.710, median positive-negative 0.001, MW p=0.002
- Temporal future8 feature persistence_particle_mse_fraction_of_full_mean: AUC 0.709, median positive-negative 0.282, MW p=0.002
- Temporal timing correlation diffusion_proxy_iqr_um2_per_s vs cycles_since_previous_drop: rho=0.644, p=5.400e-09
- Temporal timing correlation radius2_slope_iqr_px2_per_s vs cycles_since_previous_drop: rho=0.644, p=5.400e-09
- Temporal timing correlation velocity_particle_mse_fraction_of_full_mean vs cycles_since_previous_drop: rho=0.518, p=8.539e-06
- Temporal timing correlation persistence_particle_mse_fraction_of_full_mean vs cycles_to_next_drop: rho=-0.493, p=1.665e-05
- Temporal timing correlation diffusion_proxy_abs_median_um2_per_s vs cycles_since_previous_drop: rho=0.492, p=2.671e-05
- Apparent diffusion q70 apparent_D_h5median_px0p08_um2_per_s: median future8 positive-negative -4.398e-07, positive fractions 0.472/0.556, MW p=0.175
- Apparent diffusion q70 apparent_D_h5median_px0p096_um2_per_s: median future8 positive-negative -6.334e-07, positive fractions 0.472/0.556, MW p=0.175
- Apparent diffusion q70 apparent_D_h5median_px0p12_um2_per_s: median future8 positive-negative -9.896e-07, positive fractions 0.472/0.556, MW p=0.175
- Apparent diffusion q70 roi_elapsed_to_h5_median_ratio: median future8 positive-negative 3.021e-04, positive fractions 1.000/1.000, MW p=0.189
- Apparent diffusion q70 apparent_D_h5_timing_envelope_abs_min_um2_per_s: median future8 positive-negative -1.272e-07, positive fractions 1.000/1.000, MW p=0.313
- Apparent diffusion calibration correlation h5_dt_max_to_median_ratio vs transferred_masked_residual_signature: rho=0.728, p=2.858e-84, n=504
- Apparent diffusion calibration correlation apparent_D_h5median_px0p096_um2_per_s vs transferred_masked_residual_signature: rho=-0.257, p=4.700e-09, n=504
- Apparent diffusion calibration correlation roi_elapsed_to_h5_median_ratio vs validation_score_recon: rho=0.222, p=5.050e-07, n=504
- Apparent diffusion calibration correlation apparent_D_h5median_abs_um2_per_s vs transferred_masked_residual_signature: rho=-0.153, p=5.768e-04, n=504
- Apparent diffusion timing guardrail: ROI/HDF5 elapsed median ratio 1.002, max source dt max/median ratio 13.783, q70 positive-D fraction 0.514
- Apparent diffusion guardrail: Apparent diffusion values are recalibrated from HDF5 camera timing and slide-derived pixel-size assumptions. No HDF5 pixel-size attribute was found, and the values remain optical-front proxies, not validated material diffusion coefficients.
- Balanced spatial propagation graph: nodes 72, edges 414, edge counts {'next_observed_cycle_spatial_knn': 135, 'previous_observed_cycle_spatial_knn': 135, 'same_cycle_spatial_knn': 144}
- Spatial propagation homophily next_observed_cycle_spatial_knn: same future8 0.867 vs null mean 0.548, p=9.990e-04
- Spatial propagation homophily previous_observed_cycle_spatial_knn: same future8 0.867 vs null mean 0.549, p=9.990e-04
- Spatial propagation homophily same_cycle_spatial_knn: same future8 1.000 vs null mean 0.501, p=9.990e-04
- Spatial propagation autocorr next_observed_cycle_spatial_knn radius2_slope_median_px2_per_s: rho=0.594, p=9.990e-04
- Spatial propagation autocorr next_observed_cycle_spatial_knn diffusion_proxy_median_um2_per_s: rho=0.594, p=9.990e-04
- Spatial propagation autocorr next_observed_cycle_spatial_knn q70_radius2_slope_bootstrap_p50_px2_per_s: rho=0.599, p=9.990e-04
- Spatial propagation autocorr next_observed_cycle_spatial_knn q70_radius2_slope_bootstrap_p95_px2_per_s: rho=0.581, p=9.990e-04
- Spatial propagation autocorr next_observed_cycle_spatial_knn phase_slope_positive_fraction: rho=0.622, p=9.990e-04
- Spatial propagation autocorr next_observed_cycle_spatial_knn threshold_robust_phase_score: rho=0.420, p=9.990e-04
- Spatial lag feature-to-next-label phase_slope_positive_fraction: AUC 0.682, null p=9.990e-04
- Spatial lag feature-to-next-label mask_instability_score: AUC 0.563, null p=0.279
- Spatial lag feature-to-next-label radius2_slope_median_px2_per_s: AUC 0.558, null p=0.293
- Spatial lag feature-to-next-label diffusion_proxy_median_um2_per_s: AUC 0.558, null p=0.327
- Spatial distance gradient same_cycle_spatial_knn: both-positive minus other 9.863 um, p=0.339
- Spatial distance gradient next_observed_cycle_spatial_knn: both-positive minus other -5.562 um, p=0.831
- Spatial distance gradient previous_observed_cycle_spatial_knn: both-positive minus other -5.562 um, p=0.831
- Spatial propagation guardrail: Balanced spatial front propagation audit uses automatic ROI coordinates and reconstructed candidates, not tracked particle identities. Edges test spatial/temporal coherence for hypothesis ranking, not causal propagation.
- Temporal guardrail: Temporal directionality audit compares ROI physics against future, past, reversed, and circularly shifted weak degradation labels. It uses automatic ROI/front descriptors and weak cycle-level abrupt-drop labels.
- Context guardrail: Context/region audit tests whether balanced future8 ROI signal is explainable by acquisition context or spatial position. It still uses weak cycle labels and automatic ROI candidates.
- Front-script guardrail: Diffusion values are apparent optical-front proxies computed as slope(radius^2) * pixel_size_um^2 / 4. They are not validated Li diffusion coefficients and require manual front QC.
- Audit guardrail: Balanced direct-video ROI audit uses weak cycle-level future8 labels projected to automatic particle-region candidates. Cycle-group splits and cycle-collapsed tests reduce leakage, but this is still review-prioritization evidence, not manual QC or deployable detection.

## Cross-Cohort Rollout Transfer

- ROI cohorts selected/transfer-ranked: 11 / 48
- Low-rank rank/train fraction: 16 / 0.670
- Transfer model selected_internal on selected: median particle MSE 0.002, DMD/persistence 22.808, particle/nonparticle ratio 5.416, internal ratio NA, p=NA
- Transfer model pooled on selected: median particle MSE 0.006, DMD/persistence 40.246, particle/nonparticle ratio 4.473, internal ratio 3.521, p=0.115
- Transfer model transfer_ranked_internal on selected: median particle MSE 0.016, DMD/persistence 102.029, particle/nonparticle ratio 5.186, internal ratio 10.285, p=8.113e-04
- Transfer model transfer_ranked_internal on transfer_ranked: median particle MSE 0.006, DMD/persistence 20.622, particle/nonparticle ratio 8.308, internal ratio NA, p=NA
- Transfer model pooled on transfer_ranked: median particle MSE 0.006, DMD/persistence 21.376, particle/nonparticle ratio 8.449, internal ratio 1.061, p=0.674
- Transfer model selected_internal on transfer_ranked: median particle MSE 0.020, DMD/persistence 65.440, particle/nonparticle ratio 7.493, internal ratio 3.494, p=3.277e-09
- Transfer/error link transfer_ranked_internal on transfer_ranked cycleNo vs particle_mse_mean: rho=0.569, p=2.493e-05, n=48
- Transfer/error link pooled on transfer_ranked cycleNo vs particle_mse_mean: rho=0.511, p=2.092e-04, n=48
- Transfer/error link selected_internal on selected cycleNo vs particle_mse_mean: rho=-0.866, p=5.670e-04, n=11
- Transfer/error link selected_internal on transfer_ranked validation_score_first vs dmd_particle_mse_ratio_vs_persistence: rho=0.408, p=0.004, n=48
- Transfer/error link transfer_ranked_internal on selected validation_score_first vs particle_mse_mean: rho=-0.782, p=0.004, n=11
- Transfer/error link selected_internal on transfer_ranked validation_score_first vs particle_mse_mean: rho=0.381, p=0.008, n=48
- Transfer-ranked hard ROI cycle116_rank12_obj2 via selected_internal: particle MSE 0.054, DMD/persistence 844.134, cycle 116
- Transfer-ranked hard ROI cycle152_rank5_obj3 via selected_internal: particle MSE 0.045, DMD/persistence 84.097, cycle 152
- Transfer-ranked hard ROI cycle40_rank8_obj3 via selected_internal: particle MSE 0.043, DMD/persistence 443.572, cycle 40
- Transfer-ranked hard ROI cycle152_rank5_obj1 via selected_internal: particle MSE 0.040, DMD/persistence 46.942, cycle 152
- Transfer-ranked hard ROI cycle150_rank1_obj4 via selected_internal: particle MSE 0.040, DMD/persistence 325.487, cycle 150
- Transfer-ranked hard ROI cycle154_rank6_obj3 via selected_internal: particle MSE 0.039, DMD/persistence 104.473, cycle 154
- Guardrail: Cross-cohort low-rank rollout transfer compares interpretable linear dynamics across automatic ROI cohorts. It is evidence about video-domain generalization and difficult particle-local dynamics, not manual QC, calibrated diffusion, or a deployable learned video predictor.

## Masked Residual Transition Timing

- ROI/method rows/permutations: 156 / 5000
- ROI count: 52
- Alignment low_rank_dmd weighted_center_distance_to_transition_frac: median distance 0.196, null mean 0.250, empirical p=0.056, n=52
- Alignment low_rank_dmd peak_distance_to_transition_frac: median distance 0.453, null mean 0.419, empirical p=0.714, n=52
- Alignment velocity weighted_center_distance_to_transition_frac: median distance 0.279, null mean 0.250, empirical p=0.798, n=52
- Alignment persistence peak_distance_to_transition_frac: median distance 0.346, null mean 0.299, empirical p=0.828, n=52
- Alignment velocity peak_distance_to_transition_frac: median distance 0.357, null mean 0.299, empirical p=0.886, n=52
- Alignment persistence weighted_center_distance_to_transition_frac: median distance 0.319, null mean 0.253, empirical p=0.966, n=52
- Event/control timing persistence near_transition_residual_fraction: median event-control 0.069, p=0.068, n=24/28
- Event/control timing low_rank_dmd peak_distance_to_transition_frac: median event-control 0.122, p=0.183, n=24/28
- Event/control timing velocity near_transition_residual_fraction: median event-control 0.053, p=0.209, n=24/28
- Event/control timing low_rank_dmd near_minus_far_particle_mse_median: median event-control -5.138e-04, p=0.317, n=24/28
- Event/control timing low_rank_dmd particle_to_nonparticle_mse_ratio_median: median event-control -0.444, p=0.354, n=24/28
- Event/control timing low_rank_dmd weighted_center_distance_to_transition_frac: median event-control 0.042, p=0.393, n=24/28
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q70_max_abs_rate_per_s: rho=0.518, p=8.420e-05, n=52
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q80_transformed_fraction_delta: rho=0.490, p=2.286e-04, n=52
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q80_max_abs_rate_per_s: rho=0.489, p=2.310e-04, n=52
- Timing/kinetics link low_rank_dmd near_minus_far_particle_mse_median vs q70_transformed_fraction_delta: rho=-0.474, p=3.847e-04, n=52
- Timing/kinetics link persistence particle_to_nonparticle_mse_ratio_median vs q70_transformed_fraction_delta: rho=0.438, p=0.001, n=52
- Timing/kinetics link low_rank_dmd near_minus_far_particle_mse_median vs q80_transformed_fraction_delta: rho=-0.421, p=0.002, n=52
- Near-transition residual ROI cycle158_rank2_obj1 velocity (control, cycle 158): near fraction 0.860, peak distance 0.021, weighted distance 0.048
- Near-transition residual ROI cycle158_rank2_obj1 persistence (control, cycle 158): near fraction 0.773, peak distance 0.021, weighted distance 0.046
- Near-transition residual ROI cycle86_rank4_obj9 persistence (event, cycle 86): near fraction 0.696, peak distance 0.044, weighted distance 0.006
- Near-transition residual ROI cycle86_rank8_obj17 persistence (event, cycle 86): near fraction 0.645, peak distance 0.054, weighted distance 0.008
- Near-transition residual ROI cycle86_rank5_obj8 persistence (event, cycle 86): near fraction 0.626, peak distance 0.044, weighted distance 0.001
- Near-transition residual ROI cycle86_rank3_obj5 persistence (event, cycle 86): near fraction 0.597, peak distance 0.033, weighted distance 0.015
- Guardrail: Automatic phase-kinetic transition timing and masked rollout residual timing audit; not manual front annotation or calibrated transport.

## Masked Rollout Cycle Warning

- ROI cycles/features/permutations: 11 / 105 / 5000
- Target positive counts: {'any_abrupt_drop': 4, 'synchronized_drop_2plus': 2, 'future_any_drop_within_4cycles': 1, 'future_any_drop_within_8cycles': 1, 'future_any_drop_within_16cycles': 1, 'future_sync2_drop_within_4cycles': 0, 'future_sync2_drop_within_8cycles': 0, 'future_sync2_drop_within_16cycles': 0}
- Target test any_abrupt_drop low_rank_dmd_particle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.009, MW p=0.038, permutation p=0.014, n=4/6
- Target test any_abrupt_drop low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_max_delta_prev_observed_roi_cycle: median positive-negative 12.290, MW p=0.019, permutation p=0.016, n=4/6
- Target test synchronized_drop_2plus low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_median_delta_prev_observed_roi_cycle: median positive-negative -2.591, MW p=0.044, permutation p=0.022, n=2/8
- Target test synchronized_drop_2plus low_rank_dmd_particle_mse_fraction_of_full_mean_median_delta_prev_observed_roi_cycle: median positive-negative -0.425, MW p=0.044, permutation p=0.024, n=2/8
- Target test any_abrupt_drop low_rank_dmd_nonparticle_mse_mean_max_delta_prev_observed_roi_cycle: median positive-negative 0.002, MW p=0.038, permutation p=0.024, n=4/6
- Target test synchronized_drop_2plus low_rank_dmd_particle_mse_fraction_of_full_mean_median: median positive-negative -0.430, MW p=0.073, permutation p=0.036, n=2/9
- Cycle context link persistence_particle_mse_fraction_of_full_mean_max vs cycle_state_pc2: rho=0.955, p=4.989e-06, permutation p=2.000e-04, n=11
- Cycle context link persistence_particle_mse_fraction_of_full_mean_median_rolling2_mean vs frames_percentile: rho=0.927, p=4.052e-05, permutation p=2.000e-04, n=11
- Cycle context link persistence_particle_mse_fraction_of_full_mean_median vs cycle_state_pc2: rho=0.909, p=1.056e-04, permutation p=3.999e-04, n=11
- Cycle context link persistence_particle_to_nonparticle_mse_ratio_mean_median_rolling2_mean vs frames_percentile: rho=0.890, p=2.380e-04, permutation p=3.999e-04, n=11
- Cycle context link persistence_particle_to_nonparticle_mse_ratio_mean_median vs cycle_state_pc2: rho=0.900, p=1.600e-04, permutation p=5.999e-04, n=11
- Cycle context link persistence_particle_mse_fraction_of_full_mean_max vs frames_percentile: rho=0.904, p=1.332e-04, permutation p=7.998e-04, n=11
- Warning-ranked cycle 60: score 3.091, abrupt_drop=1, future8=0, n_roi=6
- Warning-ranked cycle 156: score 3.091, abrupt_drop=1, future8=0, n_roi=6
- Warning-ranked cycle 62: score 2.818, abrupt_drop=0, future8=0, n_roi=4
- Warning-ranked cycle 157: score 2.727, abrupt_drop=0, future8=0, n_roi=4
- Warning-ranked cycle 158: score 2.545, abrupt_drop=0, future8=0, n_roi=4
- Warning-ranked cycle 86: score 2.273, abrupt_drop=1, future8=0, n_roi=6
- Guardrail: Cycle-level audit from selected masked ROI rollout residuals; small selected cycle set, no deployable warning model.

## Masked ROI Rollout Audit

- ROI/frame metric rows: 52 / 4992
- Best method counts inside particle masks: {'persistence': 52}
- DMD spectral radius: 1.002
- persistence: particle-MSE median 1.146e-04, nonparticle-MSE median 7.408e-05, particle/nonparticle ratio median 1.461
- velocity: particle-MSE median 2.595e-04, nonparticle-MSE median 2.040e-04, particle/nonparticle ratio median 1.317
- low_rank_dmd: particle-MSE median 0.007, nonparticle-MSE median 0.001, particle/nonparticle ratio median 5.233
- Masked rollout event/control low_rank_dmd particle_mse_mean: median event-control 0.003, p=0.015
- Masked rollout event/control low_rank_dmd particle_mae_mean: median event-control 0.013, p=0.015
- Masked rollout event/control low_rank_dmd particle_mse_ratio_vs_persistence: median event-control 16.640, p=0.026
- Masked rollout event/control low_rank_dmd nonparticle_mse_mean: median event-control 3.952e-04, p=0.048
- Masked rollout event/control low_rank_dmd particle_to_nonparticle_mse_ratio_mean: median event-control -0.505, p=0.364
- Masked rollout event/control velocity particle_mse_fraction_of_full_mean: median event-control 0.023, p=0.640
- Masked rollout/physics link persistence particle_mse_fraction_of_full_mean vs cumulative_abs_first_last: rho=0.637, p=3.909e-07, n=52
- Masked rollout/physics link low_rank_dmd particle_mse_mean vs cumulative_abs_first_last: rho=0.603, p=2.273e-06, n=52
- Masked rollout/physics link persistence particle_mse_fraction_of_full_mean vs first_last_corr: rho=-0.585, p=5.151e-06, n=52
- Masked rollout/physics link persistence particle_to_nonparticle_mse_ratio_mean vs cumulative_abs_first_last: rho=0.567, p=1.170e-05, n=52
- Masked rollout/physics link low_rank_dmd particle_to_nonparticle_mse_ratio_mean vs high_fraction_slope_per_s: rho=-0.555, p=1.995e-05, n=52
- Masked rollout/physics link low_rank_dmd particle_mse_mean vs first_last_corr: rho=-0.544, p=3.088e-05, n=52
- Particle-rollout difficult ROI cycle156_rank7_obj27 low_rank_dmd (event, cycle 156): particle MSE 0.022, particle/nonparticle ratio 6.368
- Particle-rollout difficult ROI cycle86_rank8_obj17 low_rank_dmd (event, cycle 86): particle MSE 0.018, particle/nonparticle ratio 3.990
- Particle-rollout difficult ROI cycle157_rank2_obj2 low_rank_dmd (control, cycle 157): particle MSE 0.016, particle/nonparticle ratio 8.877
- Particle-rollout difficult ROI cycle156_rank2_obj2 low_rank_dmd (event, cycle 156): particle MSE 0.016, particle/nonparticle ratio 3.066
- Particle-rollout difficult ROI cycle62_rank3_obj9 low_rank_dmd (control, cycle 62): particle MSE 0.015, particle/nonparticle ratio 8.474
- Guardrail: Held-out rollout errors are scored inside automatic history-aware particle masks; this is not manual segmentation or a new learned video model.

## Particle Mask Stability Audit

- ROI/frame rows: 52 / 4992
- Overall median fallback fraction: 0.000
- Overall median accepted-area CV: 0.042
- Overall median centroid path: 73.607 px
- Overall median instability score: 0.809
- control masks: n=28, fallback median 0.000, area-CV median 0.036, centroid-path median 73.607 px
- event masks: n=24, fallback median 0.000, area-CV median 0.045, centroid-path median 68.920 px
- Event/control mask test fallback_frame_fraction: median event-control 0.000, p=0.298
- Event/control mask test centroid_jump_fraction: median event-control 0.000, p=0.298
- Event/control mask test accepted_area_cv: median event-control 0.008, p=0.435
- Event/control mask test accepted_area_fraction_iqr: median event-control 0.002, p=0.576
- Event/control mask test candidate_area_cv: median event-control 0.002, p=0.790
- Event/control mask test accepted_centroid_path_px: median event-control -4.687, p=0.847
- Mask/physics link accepted_centroid_path_px vs high_fraction_slope_per_s: rho=-0.370, p=0.007, n=52
- Mask/physics link mask_instability_score vs high_fraction_slope_per_s: rho=-0.353, p=0.010, n=52
- Mask/physics link accepted_area_fraction_iqr vs high_fraction_slope_per_s: rho=-0.341, p=0.013, n=52
- Mask/physics link accepted_area_fraction_iqr vs evidence_score: rho=-0.269, p=0.054, n=52
- Mask/physics link accepted_area_fraction_iqr vs mean_drop_frac: rho=-0.250, p=0.074, n=52
- High-instability ROI cycle86_rank8_obj17 (event, cycle 86): score 2.228, area-CV 0.228, centroid path 275.841 px
- High-instability ROI cycle156_rank2_obj2 (event, cycle 156): score 2.155, area-CV 0.155, centroid path 197.337 px
- High-instability ROI cycle86_rank6_obj78 (event, cycle 86): score 2.117, area-CV 0.117, centroid path 263.636 px
- High-instability ROI cycle88_rank2_obj4 (control, cycle 88): score 2.115, area-CV 0.115, centroid path 301.115 px
- High-instability ROI cycle116_rank2_obj2 (event, cycle 116): score 2.113, area-CV 0.113, centroid path 290.833 px
- Guardrail: guardrail audit of particle-region mask stability, not manual segmentation

## Physics Consistency Claim Matrix

- ROI/cycles: 52 / 11
- Tier counts: {'cross_modal_high_priority': 2, 'cross_modal_review_priority': 8, 'discordant_guardrail': 14, 'front_kinetic_consistent': 4, 'rollout_mode_consistent': 3, 'routine_or_low_consistency': 21}
- Claim readiness: {'manual_qc_required_no_physics_claim': 52}
- Manual-QC accepted rows: 0
- Rank 1 cycle156_rank7_obj27 (event, cycle 156): score 5.790, support 6, tier cross_modal_high_priority
- Rank 2 cycle156_rank5_obj4 (event, cycle 156): score 5.503, support 6, tier cross_modal_review_priority
- Rank 3 cycle156_rank2_obj2 (event, cycle 156): score 5.486, support 6, tier cross_modal_high_priority
- Rank 4 cycle158_rank2_obj1 (control, cycle 158): score 4.968, support 6, tier cross_modal_review_priority
- Rank 5 cycle156_rank8_obj10 (event, cycle 156): score 3.925, support 4, tier cross_modal_review_priority
- Rank 6 cycle158_rank4_obj8 (control, cycle 158): score 3.656, support 5, tier cross_modal_review_priority
- Rank 7 cycle158_rank6_obj3 (control, cycle 158): score 3.146, support 3, tier front_kinetic_consistent
- Rank 8 cycle156_rank6_obj3 (event, cycle 156): score 2.681, support 3, tier front_kinetic_consistent
- Event/control pillar test optical_change_score: median event-control 0.574, MW p=0.001, permutation p=0.002
- Event/control pillar test mode_taxonomy_score: median event-control 0.644, MW p=0.060, permutation p=0.011
- Event/control pillar test front_direction_score: median event-control 0.531, MW p=0.010, permutation p=0.041
- Event/control pillar test rollout_residual_score: median event-control 0.354, MW p=0.023, permutation p=0.056
- Event/control pillar test physics_consistency_score: median event-control 0.775, MW p=0.071, permutation p=0.239
- Event/control pillar test precursor_context_score: median event-control 0.159, MW p=0.425, permutation p=0.396
- Guardrail: This matrix is a multimodal consistency and review-prioritization audit. It does not assign manual QC labels and does not validate calibrated diffusion or material degradation mechanisms.

## Cycle State-Space Transition Audit

- Cycle rows/features: 89 / 107
- Echem-shape cycles joined: 81
- Chosen state clusters: 4 with silhouette 0.634
- Degradation axis oriented to mean_abs_delta_prev with rho 0.326
- Future-drop classifier: shuffled-fold AUC 0.781, balanced accuracy 0.731
- Expanding temporal holdout: AUC 0.779, balanced accuracy 0.645, evaluated blocks 2 / 4, purge 8 cycles
- Future-drop test cycle_state_pc2: positive-negative median 0.730, permutation p=0.016, MW p=2.322e-04
- Future-drop test mean_abs_delta_prev: positive-negative median -0.012, permutation p=0.080, MW p=0.036
- Future-drop test state_step_norm: positive-negative median -0.694, permutation p=0.101, MW p=0.097
- Future-drop test degradation_state_axis: positive-negative median -2.269, permutation p=0.184, MW p=0.295
- Future-drop test cycle_state_pc1: positive-negative median 2.269, permutation p=0.199, MW p=0.295
- Future-drop test capacity_mAh: positive-negative median -0.014, permutation p=0.205, MW p=0.033
- State correlation degradation_state_axis vs cycleNo: rho=-0.738, p=1.594e-16, n=89
- State correlation cycle_state_pc1 vs cycleNo: rho=0.738, p=1.594e-16, n=89
- State correlation degradation_state_axis vs capacity_mAh: rho=-0.710, p=1.133e-13, n=81
- State correlation cycle_state_pc1 vs capacity_mAh: rho=0.710, p=1.133e-13, n=81
- State correlation degradation_state_axis vs frames_percentile: rho=-0.641, p=1.360e-11, n=89
- State 1: n=58, cycles 2-158, future8 rate=0.259
- State 2: n=1, cycles 150-150, future8 rate=1.000
- State 0: n=29, cycles 112-149, future8 rate=0.138
- State 3: n=1, cycles 126-126, future8 rate=0.000
- Transition 0->2: n=1, next future8 rate=1.000, step norm=29.647
- Transition 2->1: n=1, next future8 rate=1.000, step norm=28.533
- Transition 1->0: n=1, next future8 rate=1.000, step norm=8.756
- Transition 1->1: n=56, next future8 rate=0.250, step norm=1.833
- Guardrail: Cycle state-space clusters use four-particle trace summaries and echem-shape descriptors at cycle resolution. They are degradation-state hypotheses and early-warning covariates, not localized ROI/front validation or calibrated diffusion measurements.

## Cycle Hazard Warning Audit

- Target/events: future_any_drop_within_8cycles / 4 event cycles [60.0, 86.0, 116.0, 156.0]
- Rolling-origin purge/min-train/permutation nulls: 8 / 24 / 20
- Feature set particle_trace_echem_with_acquisition: AUC 0.783, AP 0.573, Brier 0.313, balanced accuracy 0.789, n=62, positives=16
- Feature set particle_trace_echem_no_acquisition: AUC 0.742, AP 0.490, Brier 0.393, balanced accuracy 0.705, n=62, positives=16
- Feature set combined_with_cycle_state: AUC 0.739, AP 0.485, Brier 0.382, balanced accuracy 0.705, n=62, positives=16
- Feature set particle_trace: AUC 0.637, AP 0.402, Brier 0.290, balanced accuracy 0.579, n=62, positives=16
- Feature set echem_shape_cycle: AUC 0.306, AP 0.192, Brier 0.484, balanced accuracy 0.326, n=62, positives=16
- Null check: observed AUC 0.783, null p95 0.581, empirical p=0.048
- Lead-time 4 cycles: hit rate 0.500, median max probability 0.993, median lead 4.000 cycles
- Lead-time 8 cycles: hit rate 0.750, median max probability 1.000, median lead 6.000 cycles
- Lead-time 16 cycles: hit rate 0.750, median max probability 1.000, median lead 6.000 cycles
- Ablation remove particle_trace: AUC 0.492, drop vs best 0.291
- Ablation remove echem_shape: AUC 0.712, drop vs best 0.071
- Ablation remove acquisition_context: AUC 0.742, drop vs best 0.041
- Ablation remove cycle_echem: AUC 0.780, drop vs best 0.003
- Warning probability link cycle_state_pc2: rho=0.456, p=1.958e-04, n=62
- Warning probability link mean_abs_delta_prev: rho=-0.258, p=0.043, n=62
- Warning probability link shape_V_q95: rho=-0.113, p=0.414, n=54
- Warning probability link capacity_mAh: rho=-0.108, p=0.439, n=54
- Warning probability link cycle_state_pc1: rho=-0.081, p=0.532, n=62
- Guardrail: Rolling-origin cycle-level warning audit using particle trace/echem/state descriptors. It tests early-warning covariates, not localized ROI fronts, manual degradation labels, or calibrated diffusion.

## Cycle State To ROI/Front Bridge

- ROI rows/cycles joined: 52 / 11
- Predictors/targets: 9 / 13
- Row bridge cycle_state_pc2 vs physics_consistency_score: rho=0.702, permutation p=4.998e-04, n=52
- Row bridge cycle_state_pc2 vs kinetic_transition_score: rho=0.695, permutation p=4.998e-04, n=52
- Row bridge cycle_state_pc2 vs precursor_context_score: rho=0.687, permutation p=4.998e-04, n=47
- Row bridge cycle_state_pc3 vs kinetic_transition_score: rho=0.598, permutation p=4.998e-04, n=52
- Row bridge cycle_state_cluster vs kinetic_transition_score: rho=0.595, permutation p=4.998e-04, n=52
- Cycle-collapsed bridge cycle_state_pc2 vs mode_taxonomy_score: rho=0.855, permutation p=0.002, n=11
- Cycle-collapsed bridge cycle_state_pc2 vs physics_consistency_score: rho=0.836, permutation p=0.002, n=11
- Cycle-collapsed bridge cycle_state_pc2 vs kinetic_transition_score: rho=0.764, permutation p=0.009, n=11
- Cycle-collapsed bridge cycle_state_pc3 vs mode_taxonomy_score: rho=0.709, permutation p=0.021, n=11
- Cycle-collapsed bridge cycle_state_pc3 vs kinetic_transition_score: rho=0.682, permutation p=0.026, n=11
- Reference-centered bridge axis_step_ref_centered vs precursor_context_score_ref_centered: rho=0.790, permutation p=4.998e-04, n=47
- Reference-centered bridge state_step_norm_ref_centered vs precursor_context_score_ref_centered: rho=0.712, permutation p=4.998e-04, n=47
- Reference-centered bridge axis_step_ref_centered vs optical_change_score_ref_centered: rho=0.549, permutation p=4.998e-04, n=52
- Reference-centered bridge axis_step_ref_centered vs front_direction_score_ref_centered: rho=0.451, permutation p=4.998e-04, n=52
- Cycle-state cluster 1: ROI n=42, cycles=9, cross-modal priority fraction=0.238
- Cycle-state cluster 0: ROI n=10, cycles=2, cross-modal priority fraction=0.000
- Guardrail: Cycle-state to ROI/front bridge joins cycle-level state coordinates to selected automatic ROI rows. Row-level associations are not independent within cycle; reference-centered and cycle-collapsed tests are the stricter evidence. This does not create manual QC labels or calibrated diffusion claims.

## Cycle-State Mode-Frequency Bridge

- Cycles/ROI rows/mode targets: 11 / 52 / 4
- Best macro model: cycle_state_only MAE 0.261; context-only MAE 0.303; reduction 0.043
- Mode-frequency model cycle_state_only -> macro_mode_fraction: MAE 0.261, R2 NA, rho NA
- Mode-frequency model context_only -> macro_mode_fraction: MAE 0.303, R2 NA, rho NA
- Mode-frequency model cycle_state_plus_context -> macro_mode_fraction: MAE 0.348, R2 NA, rho NA
- Mode-frequency model cycle_state_echem_context -> macro_mode_fraction: MAE 0.395, R2 NA, rho NA
- Mode-frequency model echem_only -> macro_mode_fraction: MAE 0.396, R2 NA, rho NA
- Mode-frequency model echem_plus_context -> macro_mode_fraction: MAE 0.409, R2 NA, rho NA
- Mode-frequency model cycle_state_only -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.312, R2 -1.251, rho -0.196
- Mode-frequency model context_only -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.328, R2 -1.272, rho -0.182
- Mode-frequency model cycle_state_plus_context -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.390, R2 -2.075, rho -0.225
- Mode-frequency model echem_only -> mode_fraction__front_negative_high_apparent_front_proxy: MAE 0.603, R2 -8.539, rho -0.569
- Mode-frequency null cycle_state_only: observed macro MAE 0.261, null mean 0.290, p=0.381
- Mode-frequency null echem_plus_context: observed macro MAE 0.409, null mean 0.435, p=0.381
- Mode-frequency null context_only: observed macro MAE 0.303, null mean 0.343, p=0.429
- Mode-frequency null cycle_state_echem_context: observed macro MAE 0.395, null mean 0.391, p=0.524
- Mode-frequency null cycle_state_plus_context: observed macro MAE 0.348, null mean 0.344, p=0.667
- Mode-frequency null echem_only: observed macro MAE 0.396, null mean 0.377, p=0.714
- Cycle-state cluster 1: cycles=9, ROI=42, median cycle=88.000
- Cycle-state cluster 0: cycles=2, ROI=10, median cycle=117.000
- Guardrail: Cycle-state mode-frequency bridge predicts automatic ROI mode composition at cycle resolution from cycle/echem descriptors. It is a degradation-mode organization audit, not manual QC, causal proof, or calibrated diffusion validation.

## Top ROI/Echem Or Protocol Couplings

- n_frames_percentile vs latent_net_displacement: rho=0.776, p=1.340e-11, n=52
- n_frames_percentile vs cumulative_abs_norm_change: rho=0.756, p=9.501e-11, n=52
- n_frames_percentile vs first_last_corr: rho=-0.700, p=7.504e-09, n=52
- n_frames_percentile vs latent_path_length: rho=0.697, p=9.707e-09, n=52
- n_frames_percentile vs latent_mean_step: rho=0.697, p=9.707e-09, n=52
- cycles_from_block_start vs latent_mean_step: rho=0.609, p=1.635e-06, n=52
- cycles_from_block_start vs latent_path_length: rho=0.609, p=1.635e-06, n=52
- n_frames_percentile vs dmd_minus_persistence_mse: rho=0.595, p=3.280e-06, n=52

## Highest-Priority ROI Candidates

- cycle62_rank3_obj9 (control, cycle 62): rollout/mobility score 3.058, latent path 14.524, first-last corr 0.824
- cycle156_rank5_obj4 (event, cycle 156): rollout/mobility score 3.038, latent path 19.837, first-last corr 0.878
- cycle156_rank7_obj27 (event, cycle 156): rollout/mobility score 3.019, latent path 22.511, first-last corr 0.853
- cycle156_rank2_obj2 (event, cycle 156): rollout/mobility score 2.865, latent path 19.329, first-last corr 0.911
- cycle60_rank3_obj9 (event, cycle 60): rollout/mobility score 2.846, latent path 22.630, first-last corr 0.869
- cycle60_rank6_obj26 (event, cycle 60): rollout/mobility score 2.615, latent path 16.147, first-last corr 0.834
- cycle60_rank4_obj5 (event, cycle 60): rollout/mobility score 2.596, latent path 14.736, first-last corr 0.844
- cycle156_rank8_obj10 (event, cycle 156): rollout/mobility score 2.519, latent path 19.657, first-last corr 0.902
- cycle86_rank4_obj9 (event, cycle 86): rollout/mobility score 2.327, latent path 18.205, first-last corr 0.976
- cycle60_rank2_obj2 (event, cycle 60): rollout/mobility score 2.308, latent path 19.447, first-last corr 0.904

## Residual-Mode Review Priorities

- cycle60_rank3_obj9 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.673
- cycle60_rank6_obj26 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.596
- cycle60_rank4_obj5 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.481
- cycle62_rank3_obj9 (control, cycle 62): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.308
- cycle156_rank7_obj27 (event, cycle 156): optical_brightening_decorrelating_rollout_hard_front_positive, priority 3.260
- cycle157_rank2_obj2 (control, cycle 157): optical_loss_rollout_hard, priority 3.106
- cycle156_rank5_obj4 (event, cycle 156): optical_brightening_decorrelating_rollout_hard_front_positive, priority 2.933
- cycle62_rank4_obj1 (control, cycle 62): optical_loss_rollout_hard, priority 2.923
- cycle60_rank2_obj2 (event, cycle 60): optical_brightening_decorrelating_rollout_hard_front_positive, priority 2.923
- cycle58_rank3_obj9 (control, cycle 58): near_baseline_or_context_like, priority 2.913

## Manual QC Label Workbook

- Deduplicated ROI candidates: 47
- Role counts: {'control': 24, 'event': 23}
- Priority tiers: {'high': 12, 'medium': 17, 'routine': 18}
- Manual-QC status counts: {'pending': 47}
- Guardrail: This workbook is a manual-label template. It deduplicates review candidates and preserves pending status, but it does not assign accept/reject labels or validate diffusion.

## Weak-Label Degradation Benchmark

- ROI rows: 52
- Trainable weak-label rows: 7
- Positive/negative weak labels: 3 / 4
- Label counts: {'review_control_uncertain': 19, 'review_positive_uncertain': 15, 'review_uncertain': 11, 'weak_event_enriched_front_mode': 3, 'weak_low_consistency_control': 4}
- Leave-reference usable binary folds: 1 / 4
- Weak positive cycle156_rank7_obj27 (event, cycle 156): physics score 5.790, mode optical_brightening_decorrelating_rollout_hard_front_positive
- Weak positive cycle156_rank8_obj10 (event, cycle 156): physics score 3.925, mode optical_brightening_decorrelating_rollout_hard_front_positive
- Weak positive cycle60_rank6_obj26 (event, cycle 60): physics score 1.683, mode optical_brightening_decorrelating_rollout_hard_front_positive
- Weak negative cycle118_rank2_obj2 (control, cycle 118): physics score -3.012, mode near_baseline_or_context_like
- Weak negative cycle90_rank3_obj4 (control, cycle 90): physics score -1.988, mode front_negative_high_apparent_front_proxy
- Weak negative cycle58_rank1_obj1 (control, cycle 58): physics score -1.885, mode near_baseline_or_context_like
- Weak negative cycle90_rank4_obj6 (control, cycle 90): physics score -1.493, mode front_negative_high_apparent_front_proxy
- Split guardrail fold 2 holdout 86: weak_label_class_missing_in_train_or_test (0 positive / 2 negative test)
- Split guardrail fold 3 holdout 116: weak_label_class_missing_in_train_or_test (0 positive / 1 negative test)
- Split guardrail fold 4 holdout 156: weak_label_class_missing_in_train_or_test (2 positive / 0 negative test)
- Guardrail: This benchmark contains weak consensus labels for model development and review prioritization only. It is not a manual-QC label set and must not be used to claim validated degradation modes or calibrated diffusion.

## Manual-QC Gated Front Effects

- Status: ready_for_manual_labels
- Joined ROI rows: 52
- Accepted front-effect rows: 0
- Accepted diffusion-interpretable rows: 0
- Guardrail: Only manually accepted particle/front labels are eligible for gated front-effect tests. Current pending labels intentionally produce no accepted-front physics claims.

## Control-Balanced QC Sensitivity

- Robust positive phase-residual strata: all_front_rois, balanced_qc_selected, complete_threshold_sweep, original_qc_selected, q70_phase_ci_excludes_zero, q70_phase_ci_positive
- Guardrail: Control-balanced QC sensitivity compares the original event-heavy review panel with the augmented event/control-balanced panel. It audits selection bias in phase-front directionality, but remains an automatic candidate review and not a manual accept/reject QC result.

## Completion Audit

- Implement paper-inspired agentic workflows in separate Isambard folders: implemented. Evidence: agentic_research outputs plus derived tier1/tier2/tier3 experiment folders were created on Isambard and compact outputs synced locally. Limitation: The synthesis script summarizes the outputs but does not rerun the original literature analysis.
- Focus on Alek_Jiho NMC degradation dataset on Isambard: implemented. Evidence: Synthesis reads Isambard derived directory with 52 ROI rows, 11 cycles, and 4 event-reference cycles. Limitation: The current multi-cycle ROI cohort is selected around event/reference cycles, not every raw video in the full dataset.
- Next-frame prediction and rollout: implemented_with_guardrail. Evidence: Persistence, velocity, low-rank DMD, PCA latent trajectories, PCA-ridge, residual-CNN guardrails, and prefix-only ROI forecasts were run. Persistence is best across raw pixel rollouts: True; best prefix classifier target is front_positive_residual_binary with AUC 0.691. Limitation: Learned/full rollout models do not yet beat persistence robustly; use residuals, latent paths, and prefix forecasts as physics descriptors rather than claiming superior pixel prediction.
- Select and guard particle-region-only ROIs: implemented_with_guardrail. Evidence: Model inputs use cropped ROI tensors and the particle-mask stability audit covers 52 ROI rows / 4992 frames; median fallback fraction is 0.000. Limitation: The audit uses automatic contrast/history masks and is not a manual segmentation of each particle boundary.
- Track phase-boundary movement: implemented_as_proxy. Evidence: Front/phase mobility descriptors, selected-front tracking, and threshold-robust sweeps exist; threshold sweep covers 52 ROI rows. Limitation: Front masks are automatic; after protocol/echem conditioning, front-direction sign consistency survives more strongly than front-magnitude metrics and is robust in 5 automatic QC strata.
- Extract diffusion coefficients: partial_proxy_only. Evidence: Provisional 0.096 um/px apparent diffusion proxies were computed and stress-tested across 7 thresholds with bootstrap slopes; the stricter diffusion sanity audit finds 0 automatic positive candidates and 0 publication candidates; the control-balanced high-resolution rerun tracks 40 ROIs and still finds 0 publication candidates. Limitation: Global threshold-robust phase slopes separate event/control ROIs, but QC-stratified diffusion proxies are inconsistent, conditioned diffusion-proxy residuals remain non-significant, and selected-front radius-squared slopes fail sign/fit/manual-QC gates.
- Identify degradation modes: implemented_as_hypothesis_ranking. Evidence: Joint physics/rollout/echem mode tables exist, residual taxonomy found 4 protocol-adjusted modes, and cycle/region context maps them across 11 cycles and 7 coarse regions. Limitation: Modes are unsupervised/automatic and tied to the selected ROI cohort; residual taxonomy silhouette is modest and cycle/region context is descriptive.
- Correlate degradation with cycles, particle regions, and echem/protocol context: implemented_with_guardrail. Evidence: Multi-cycle ROI echem coupling found strong frame-count/protocol correlations; protocol-conditioned residual tests still show event/control optical shifts and phase-front sign consistency. Limitation: Residualization reduces but does not eliminate confounding, and front residual classifiers are not deployable with 52 automatic ROIs.
- Keep objectives and observations updated: implemented. Evidence: OBJECTIVES_OBSERVATIONS.md contains chronological experiment summaries and guardrails. Limitation: It is long and narrative; the new synthesis is the compact index.
- Keep GitHub updated: implemented_with_verification. Evidence: Scoped analysis scripts, compact derived outputs, and observations are committed and pushed after each completed increment. Limitation: The synthesis report records workflow state at generation time; final push status should still be checked with git status and git log.

## Recommended Next Experiments

1. Manual QC the ROI/front package panels and update `manual_qc_status` with accepted/rejected labels.
2. Expand the ROI cohort across more cycles after QC to reduce event-reference and protocol confounding.
3. Recompute apparent diffusion/front-motion proxies only on QC-accepted fronts with confirmed spatial/time calibration.
4. Convert the top ROI candidates into a labeled degradation-mode benchmark for future self-supervised video models.
5. Grow protocol-conditioned residual models after adding more QC-accepted cycles and particle regions.

## Guardrail

The current outputs support a physics-extraction workflow and ranked hypotheses. They do not yet support a claim of calibrated diffusion coefficients or a deployable automated degradation detector.
