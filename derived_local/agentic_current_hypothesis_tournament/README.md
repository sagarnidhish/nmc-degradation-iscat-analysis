# Agentic Current Hypothesis Tournament

Deterministic co-scientist-style tournament over the current NMC photometry evidence.
It uses generator, skeptic, and ranker roles without making unverified physical claims.

## Inputs

- `echem_video_embedding_fusion_audit`: 14 classification rows; future16 video+echem AUC 0.754
- `residual_dictionary_embedding_audit`: 8 classification rows; residual future8 AUC 0.663
- `qc_decision_evidence_ledger`: 47 pending candidates; 3 possible accept-first
- `apparent_diffusion_calibration_bounds`: used as diffusion-claim guardrail
- `temporal_directionality_physics_audit`: used as phase-front directionality guardrail
- `balanced_spatial_front_propagation_audit`: used for spatial-lag hypothesis
- `cycle_state_space_transition_audit`: used for cycle-state hypothesis
- `prefix_roi_forecast`: used for early warning/review prioritization hypothesis
- `nmc_ai_physics_synthesis`: available

## Top Hypotheses

### 1. Echem-conditioned video residuals are the best longer-horizon weak-label signal

- Score: 0.598
- Agent: `multimodal_evidence_agent`
- Claim: Longer-horizon future-drop signal is strongest when optical video descriptors are conditioned by echem regime.
- Falsification: Require the video+echem gain to persist under cycle-balanced acquisition residualization and source-cohort holdouts.
- Next experiment: Run an acquisition-residualized video+echem future16 audit with leave-source and leave-cycle splits.
- Suggested script: `tier4_acquisition_residualized_video_echem_warning.py`

Evidence:
- future16 video+echem AUC 0.754 versus echem-only AUC 0.505
- future16 video+echem empirical p 0.002
- Masked residual signatures also become interpretable when echem features are included in prior fusion audits.

### 2. Short-horizon future8 labels are acquisition/context dominated

- Score: 0.594
- Agent: `skeptical_confound_agent`
- Claim: The future8 weak label is partly a cohort-design or acquisition proxy, so it should not drive mechanistic claims alone.
- Falsification: A future8 model must remain above null after acquisition, frame-count, cohort-role, and source-movie residualization.
- Next experiment: Build a residualized future8 benchmark that removes acquisition/context signal before testing video physics features.
- Suggested script: `tier4_residualized_future8_video_physics_benchmark.py`

Evidence:
- future8 acquisition/context AUC 1.000 versus video-all AUC 0.823
- Prior context audits repeatedly flag frame count and protocol position as central confounders.
- This explains why high future8 performance can coexist with weak physical interpretability.

### 3. Manual QC on the top cycle-156 panel is the highest-yield lab-in-the-loop step

- Score: 0.557
- Agent: `lab_in_loop_qc_agent`
- Claim: The next human intervention should be a small targeted cycle-156 review because it maximizes cross-modal support and artifact contrast.
- Falsification: If manual review rejects the possible-accept candidates or accepts the artifact foil, current automatic physics support is over-weighted.
- Next experiment: Complete the cycle-156 manual QC mini-batch and rerun manual-QC-gated front/echem effects.
- Suggested script: `tier4_manual_qc_gated_front_effects.py`

Evidence:
- QC ledger has 47 pending candidates, 3 possible-accept-first and 4 artifact/reject-first candidates.
- Cycle 156 contains all current possible-accept-first candidates plus a strong artifact-risk foil.
- This mirrors a lab-in-the-loop discovery loop: rank candidates, inspect, feed labels back into models.

### 4. Cycle-state transitions can organize ROI degradation modes

- Score: 0.556
- Agent: `state_space_agent`
- Claim: Cycle-level electrochemical state transitions may explain which ROI physics modes appear at each cycle.
- Falsification: Cycle-state modes must predict held-out ROI mode frequencies better than cycle number and acquisition context alone.
- Next experiment: Join cycle-state transition features to ROI mode counts and run leave-cycle mode-frequency prediction.
- Suggested script: `tier4_cycle_state_mode_frequency_bridge.py`

Evidence:
- Cycle-state guardrail: Cycle state-space clusters use four-particle trace summaries and echem-shape descriptors at cycle resolution. They are degradation-state hypotheses and early-warning covariates, not localized ROI/front validation or calibrated diffusion measurements.
- Echem regime, future-drop, and ROI physics audits repeatedly point to cycle-conditioned behavior.
- This provides a compact bridge between cell-level echem state and ROI-level optical heterogeneity.

### 5. Generic video embeddings underperform physics-aware temporal descriptors

- Score: 0.547
- Agent: `representation_skeptic_agent`
- Claim: Current label-free residual/PCA embeddings help triage but do not replace hand-built particle and mask dynamics.
- Falsification: Show that a learned objective beats handcrafted features under leave-cycle splits without acquisition leakage.
- Next experiment: Train or approximate an echem/context-conditioned residual objective and compare it against handcrafted scalars.
- Suggested script: `tier4_echem_conditioned_residual_dictionary.py`

Evidence:
- residual-dictionary future8 AUC 0.663 versus raw PCA AUC 0.566
- residual+handcrafted AUC 0.771 versus handcrafted-only AUC 0.825
- This points toward physics-aware or echem-conditioned objectives rather than larger generic embeddings.

### 6. Prefix traces are useful for review prioritization before event completion

- Score: 0.535
- Agent: `early_warning_agent`
- Claim: Partial ROI traces may rank candidate events early enough for review, but need stricter leakage and cycle nulls.
- Falsification: Prefix performance must survive reference-cycle centering and label-time shift nulls.
- Next experiment: Rerun prefix forecasting with manual-QC queue labels and stronger shift-null controls.
- Suggested script: `tier4_prefix_qc_queue_forecast.py`

Evidence:
- Best prefix classification target: front_positive_residual_binary with ROC AUC 0.691
- Prefix models already exist, making this a low-cost route to an online warning-style analysis.
- The result should be framed as review prioritization until manual QC labels exist.

### 7. Spatial propagation should be tested as graph lag, not just ROI-wise classification

- Score: 0.534
- Agent: `spatial_graph_agent`
- Claim: If phase fronts are physically coordinated, nearby ROI/front events should show lagged graph structure beyond cycle-level labels.
- Falsification: Shuffle cycle labels and spatial edges; true propagation should weaken under both nulls while retaining local lag consistency.
- Next experiment: Expand graph lag tests with edge-shuffled and cycle-shuffled nulls plus echem-conditioned edge weights.
- Suggested script: `tier4_echem_conditioned_spatial_lag_nulls.py`

Evidence:
- Spatial propagation guardrail: Balanced spatial front propagation audit uses automatic ROI coordinates and reconstructed candidates, not tracked particle identities. Edges test spatial/temporal coherence for hypothesis ranking, not causal propagation.
- The existing graph outputs provide nodes and edges, but final interpretation still hinges on lag robustness and source-cycle balance.
- This is closer to a physics mechanism than a high-AUC weak-label classifier.

### 8. Phase-front directionality is stronger than calibrated diffusion

- Score: 0.517
- Agent: `physics_guardrail_agent`
- Claim: The current defensible physics claim is optical phase/front movement, not calibrated lithium diffusion.
- Falsification: Require stable front signs, pixel/time calibration, and accepted particle/front masks before reporting diffusion-like coefficients.
- Next experiment: Run a manual-QC-gated apparent-diffusion bounds update only on accepted particle/front masks.
- Suggested script: `tier4_manual_qc_gated_diffusion_bounds.py`

Evidence:
- Diffusion/calibration guardrail: Apparent diffusion values are recalibrated from HDF5 camera timing and slide-derived pixel-size assumptions. No HDF5 pixel-size attribute was found, and the values remain optical-front proxies, not validated material diffusion coefficients.
- Temporal directionality guardrail: Temporal directionality audit compares ROI physics against future, past, reversed, and circularly shifted weak degradation labels. It uses automatic ROI/front descriptors and weak cycle-level abrupt-drop labels.
- Several audits find front-direction or phase-fraction signal while diffusion magnitudes remain calibration- and mask-sensitive.

## Guardrail

This tournament ranks next analyses from existing automatic evidence. It does not create manual QC labels, does not validate a deployable degradation detector, and does not license calibrated diffusion claims.
