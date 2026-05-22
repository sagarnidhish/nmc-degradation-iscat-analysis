# Agentic AI Literature Implementation Plan

Paper-driven implementation notes for applying the May 2026 Nature agentic-AI papers to the Alek/Jiho NMC charge-photometry project.

## Papers Reviewed

- Ghareeb et al., "A multi-agent system for automating scientific discovery", Nature 2026, DOI: `10.1038/s41586-026-10652-y`.
- Gottweis et al., "Accelerating scientific discovery with Co-Scientist", Nature 2026, DOI: `10.1038/s41586-026-10644-y`.
- Aygun et al., "An AI system to help scientists write expert-level empirical software", Nature 2026, DOI: `10.1038/s41586-026-10658-6`.

## Bottom Line

The directly implementable value is not a new black-box battery model. It is a disciplined research operating system around the existing photometry/echem analyses:

1. generate competing degradation and artifact hypotheses,
2. score them against current evidence and skeptical controls,
3. automatically propose small auditable analysis jobs,
4. optimize those jobs against explicit quality metrics,
5. require human QC labels before making physical claims.

The project already contains partial versions of this: `agentic_current_hypothesis_tournament`, calibration claim guardrails, QC prioritization, and synthesis summaries. The next step is to make these loops benchmark-driven and closed-loop rather than one-off reports.

## Technique Mapping

| Paper technique | Project adaptation | Implementation status | Priority |
| --- | --- | --- | --- |
| Multi-agent literature/data loop from Robin | Use generator, data analyst, skeptic, and lab-QC roles to update degradation-event hypotheses after each new audit | Partly present in `tier4_agentic_current_hypothesis_tournament.py`; extend with evidence ingestion and automatic next-job manifests | High |
| Co-Scientist hypothesis tournament | Rank hypotheses such as synchronized degradation, frame-count artifact, echem-conditioned optical residuals, and spatial propagation with explicit falsification tests | Present as deterministic tournament; improve with source-balanced and leave-source score cards | High |
| ERA tree-search/code-search | Search over analysis variants, feature families, null controls, and model settings using a project metric such as source-held-out AUC minus artifact leakage | Not yet implemented as a reusable runner | High |
| Test-time compute scaling | Spend more compute on the highest-value unresolved hypotheses, not on generic larger models | Manual at present through Isambard scripts | Medium |
| Internal critique/refinement | Add skeptic checks for acquisition leakage, source/movie leakage, frame-count confounding, weak-label leakage, and calibration overclaim | Partly present through risk register and source-balanced audits | High |
| Lab-in-the-loop validation | Treat manual ROI/front labels as the decisive feedback signal for accepting or rejecting automated hypotheses | QC packet exists; need labels populated and fed back into downstream scripts | High |
| Generated empirical software with metrics | Let code generation produce candidate analysis stubs, but only accept scripts that pass compile, schema, null-model, and leakage tests | Initial guarded-code-generation folder exists; needs automated acceptance tests | Medium |

## Concrete Modules To Build Next

### 1. Benchmark-Driven Analysis Search

Create `tier4_agentic_metric_search.py`.

Inputs:

- source-balanced ROI sequence manifest,
- echem-conditioned residual dictionary outputs,
- source robustness audit outputs,
- synthesis summary JSON.

Search space:

- feature family: residual dictionary, ROI physics, echem shape, video embedding, fused features,
- split: leave-cycle, leave-source, source-balanced bootstrap,
- residualization: none, frame-count, protocol/context, source/movie,
- target: future8, future16, synchronized-event, manual-QC queue.

Primary metric:

- `scientific_score = heldout_auc + 0.5 * heldout_ap - leakage_penalty - calibration_claim_penalty`.

Acceptance rule:

- A candidate is promotable only if it beats a matched null and does not rely on source/acquisition features alone.

### 2. Closed-Loop Hypothesis Ledger

Extend the current hypothesis tournament into a ledger with one row per claim:

- claim,
- evidence tables used,
- strongest support,
- strongest counter-evidence,
- required falsification test,
- next script,
- status: `proposed`, `under_test`, `supported`, `weakened`, `blocked_by_qc`.

This makes the project closer to Robin/Co-Scientist without pretending to automate physical interpretation.

### 3. Human-QC Feedback Hook

Require each front/particle physical claim to pass through:

- automatic ROI/front candidate,
- review packet,
- manual label,
- manual-QC-gated reanalysis,
- synthesis update.

This is the key guardrail because the papers still depend on human experimental validation, and our analogous validation is manual spatial/ROI review plus echem sanity checks.

### 4. Guarded Code-Generation Acceptance Tests

Generated scripts should be rejected unless they satisfy:

- `python -m py_compile`,
- input paths exist or are declared optional,
- output schema is documented in README,
- grouped split is used for predictive claims,
- at least one null or leakage control is included,
- calibrated diffusion/material claims are not emitted unless calibration/QC prerequisites are present.

## Techniques To Avoid For Now

- Fully autonomous scientific claims: current labels are weak and some signals are acquisition-context dominated.
- Bigger generic video embeddings as the first response: previous audits show physics-aware/echem-conditioned descriptors are more useful than generic embeddings alone.
- Calibrated diffusion constants from current front proxies: spatial calibration and manual front validation remain incomplete.
- Literature-only battery mechanism claims: the present project strength is direct video/echem evidence, not broad literature synthesis.

## Immediate Recommendation

Implement the high-priority ERA-style metric search first. It will convert the current collection of scripts into a measurable search loop and will tell us which analysis variants survive source, cycle, and acquisition controls. The second priority is the closed-loop hypothesis ledger, because it prevents agentic output from becoming untracked narrative.

## Guardrail

These papers support a more systematic AI-assisted research workflow. They do not justify treating an AI-generated hypothesis, a high weak-label AUC, or an automatically detected optical front as a validated battery mechanism without source-balanced controls, manual ROI/front QC, and electrochemical consistency checks.
