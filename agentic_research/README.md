# Agentic Research Workflows For Alek/Jiho NMC Degradation

This directory implements separate, auditable AI-inspired workflows for the
Alek/Jiho NMC degradation dataset on Isambard. The workflows are deliberately
bounded: they read raw data and derived summaries, write new outputs under
`agentic_research_outputs/`, and do not mutate raw HDF5/CSV inputs.

## Workflows

| Folder | Paper-inspired idea | Purpose |
|---|---|---|
| `01_era_experiment_search` | ERA-style metric-driven empirical search | Rank next computational experiments against current evidence and a composite physics score. |
| `02_hypothesis_tournament` | Co-Scientist-style hypothesis tournament | Score, critique, and prioritize degradation/transport hypotheses with explicit evidence and missing tests. |
| `03_closed_loop_analysis` | Robin-style closed-loop analysis | Summarize current outputs, update observations, and produce a next-action queue. |
| `04_guarded_code_generation` | Guarded empirical software generation | Generate reviewable Slurm/script stubs for high-value experiments with smoke commands and safety checks. |
| `05_agentic_metric_search` | ERA-style benchmark search over existing result tables | Rank real feature/split/control variants by scientific score with leakage and source-control penalties. |
| `06_closed_loop_hypothesis_ledger` | Co-Scientist/Robin-style claim ledger | Track claim status, support, counter-evidence, falsification tests, and next scripts. |
| `07_manual_qc_feedback_hook` | Lab-in-the-loop feedback | Convert pending manual ROI/front QC into explicit feedback gates for downstream physics claims. |
| `08_guarded_code_acceptance` | Empirical-software acceptance gates | Compile and scan generated scripts for split/null/schema/destructive-operation guardrails. |

## Isambard Usage

From `/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg`:

```bash
bash agentic_research/run_all_agentic_smoke.sh
```

Outputs are written to:

```text
/scratch/u6hp/nsagar.u6hp/Alek_Jiho/agentic_research_outputs/
```

For full HPC submission, use:

```bash
sbatch agentic_research/slurm/submit_agentic_research.sh
```

## Scientific Guardrails

- Treat optical intensity as a state/degradation proxy, not calibrated Li concentration.
- Treat same-cycle particle events as grouped observations, not independent samples.
- Use shuffled-cycle, shuffled-particle, protocol-local, and artifact checks before making claims.
- Use apparent transport/degradation coefficients only after spatial calibration and visual front validation.
