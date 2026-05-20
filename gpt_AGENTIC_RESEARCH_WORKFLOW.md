# Agentic AI Research Workflow for NMC Charge-Photometry HDF5 Videos

This document is intended to be read by Claude Code, Codex CLI, or another coding agent running in a terminal. It describes the target repository, implementation plan, safety rules, and first scientific milestones for building an agent-assisted research workflow around `.hdf5/.h5` charge-photometry videos of NMC battery particles during cycling.

## 0. Mission

Build a reproducible, human-in-the-loop, agent-assisted research system that can:

1. inspect and validate HDF5 charge-photometry video datasets;
2. preprocess videos robustly;
3. segment and track particles;
4. extract scientifically meaningful optical/electrochemical features;
5. propose and test hypotheses about NMC degradation;
6. submit parallel jobs to a supercomputer through Slurm/Snakemake where available;
7. produce auditable experiment reports suitable for later publication.

The system must help discover scientifically interesting effects, but every scientific claim must be backed by logged code, configurations, metrics, plots, statistical tests, controls, and clear limitations.

## 1. Non-negotiable rules for coding agents

- Raw data are read-only. Never delete, overwrite, move, rechunk, or mutate raw HDF5 files.
- All derived data must be written under `derived/`, `experiments/`, or another explicitly configured output root.
- Every experiment must be reproducible from a config file and a git commit.
- Every full experiment must have a small dry run first.
- Never run unbounded jobs on the login node.
- Never submit large HPC jobs without an explicit config and a clear expected output directory.
- Never claim a scientific discovery from a single uncorrected analysis.
- Always include artifact controls: drift, illumination, segmentation stability, background/fiducial regions if available, and shuffled-label or shuffled-cycle negative controls.
- Treat frames as repeated measurements, not independent samples.
- Treat particles from the same video/cell as grouped data.
- Separate discovery from validation whenever enough videos/cells are available.
- Any agent-generated conclusion must include a skeptical review section.

## 2. Recommended stack

### Local machine

- Claude Code and/or Codex CLI for repo-level coding.
- Git for code versioning.
- LangGraph for durable multi-agent orchestration after the baseline pipeline exists.
- MLflow for experiment tracking.
- DVC or DataLad for large-data manifests and data provenance.
- Papermill for parameterized notebook reports.
- Jupyter and napari for interactive visual QC.

### HPC / supercomputer

- Slurm for job submission, if available.
- Snakemake as the first workflow engine.
- Optional: Nextflow if the lab already uses it.
- Optional: Dask or Ray only after the simpler Slurm/Snakemake design is working.
- Optional: Apptainer/Singularity if the cluster prefers containers.

### Scientific Python stack

- `numpy`, `scipy`, `pandas`, `h5py`, `zarr`, `dask`, `xarray`
- `scikit-image`, `opencv-python-headless`, `trackpy`
- `matplotlib`, `plotly` or `bokeh` for diagnostics
- `napari` for interactive multidimensional video inspection
- `torch` only where needed for deep segmentation or models
- `cupy`, `cucim`, or `kornia` only where GPU image processing is worthwhile
- `pybamm` for mechanistic battery-model comparison
- `statsmodels`, `scikit-learn`, `pymc` or `numpyro` for statistics

## 3. Check HPC policy before implementation

Before installing AI tools or submitting jobs, check:

1. Is outbound internet/API access allowed from login nodes?
2. Is outbound internet/API access allowed from compute nodes?
3. Are Claude Code, Codex CLI, or similar AI tools permitted on the cluster?
4. Are unpublished raw data allowed to appear in prompts sent to external model providers?
5. Is Apptainer/Singularity available?
6. Is the scheduler Slurm, PBS, LSF, or something else?
7. Which GPU partitions and GPU types are available?
8. Are job arrays allowed?
9. What are login-node CPU/time/memory limits?
10. Is MPI-enabled HDF5 installed?

Default design: run agents locally, keep raw data on the HPC, and interact with the cluster by Git/rsync/SSH/Sbatch. Do not require AI tools to run on the cluster.

## 4. Target repository layout

Create or adapt the repository to this structure:

```text
nmc-charge-photometry/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── environment/
│   ├── environment.yml
│   └── requirements.txt
├── configs/
│   ├── dataset.yaml
│   ├── preprocessing.yaml
│   ├── segmentation.yaml
│   ├── features.yaml
│   ├── hpc.yaml
│   └── hypotheses.yaml
├── data_manifest/
│   ├── raw_files.csv
│   └── hdf5_schema_reports/
├── src/
│   └── cpvideo/
│       ├── __init__.py
│       ├── io.py
│       ├── inspect_hdf5.py
│       ├── validate.py
│       ├── preprocess.py
│       ├── register.py
│       ├── segment.py
│       ├── track.py
│       ├── features.py
│       ├── controls.py
│       ├── models.py
│       ├── stats.py
│       ├── viz.py
│       └── reporting.py
├── workflows/
│   ├── Snakefile
│   └── rules/
│       ├── inspect.smk
│       ├── preprocess.smk
│       ├── segment.smk
│       ├── features.smk
│       ├── controls.smk
│       └── report.smk
├── slurm/
│   ├── inspect_job.sh
│   ├── preprocess_job.sh
│   ├── feature_job.sh
│   └── model_fit_job.sh
├── notebooks/
│   ├── 00_qc_viewer.ipynb
│   ├── 01_baseline_features.ipynb
│   └── 02_hypothesis_report.ipynb
├── experiments/
├── derived/
├── reports/
├── tests/
└── docs/
    ├── scientific_hypotheses.md
    ├── data_contract.md
    └── validation_protocol.md
```

## 5. Create `AGENTS.md`

Create this file at repo root. It should be short enough for coding agents to read repeatedly.

```markdown
# AGENTS.md

This repo analyzes NMC battery charge-photometry HDF5 videos.

## Safety

Raw data are read-only. Never delete, overwrite, rechunk, move, or mutate raw HDF5 files.
Do not run expensive jobs on login nodes. Use small dry runs before full jobs.
Write all outputs to configured output directories under `derived/`, `experiments/`, or `reports/`.

## Scientific validity

No scientific claim is valid unless backed by:
- config file;
- git commit hash;
- raw-data manifest ID;
- environment record;
- code path;
- metrics JSON/CSV;
- QC plots;
- statistical test summary;
- negative controls;
- skeptical review.

## Required controls

Every major result should test, where applicable:
- drift/registration sensitivity;
- illumination correction sensitivity;
- segmentation-mask stability;
- background/fiducial-region behavior;
- shuffled-cycle or shuffled-label negative control;
- robustness across videos/cells;
- robustness to particle size/brightness/position covariates.

## Coding rules

Prefer modular scripts under `src/cpvideo/`.
All scripts must accept a config path and output directory.
Do not hard-code raw data paths.
Add tests for I/O, shape conventions, metadata parsing, and feature extraction.
Keep notebooks for reporting and QC, not as the only implementation.

## HPC rules

Use Slurm/Snakemake for full runs.
Every submitted job must write logs and a machine-readable status/metrics file.
No unbounded loops. No silent overwrites.
```

## 6. Data contract

Create `docs/data_contract.md` describing the expected dataset and conventions.

Minimum required metadata per raw file:

- file path;
- file size;
- HDF5 keys;
- video dataset path;
- shape;
- dtype;
- chunking;
- compression;
- frame axis;
- y/x axes;
- time axis or frame rate;
- cycle labels if available;
- charge/discharge labels if available;
- current/voltage/capacity data if available;
- microscope metadata if available;
- missing/corrupt frame report;
- notes on illumination/focus/drift.

Expected canonical in-memory video convention:

```python
video.shape == (T, Y, X)              # single-channel video
# or
video.shape == (T, C, Y, X)           # multi-channel video
```

If raw files use another convention, convert only in derived outputs or lazy adapters; never mutate raw files.

## 7. First scripts to implement

### 7.1 HDF5 inspector

Implement:

```bash
python -m cpvideo.inspect_hdf5 \
  --input /path/to/file.h5 \
  --output data_manifest/hdf5_schema_reports/file.json
```

Requirements:

- Recursively list groups/datasets.
- For each dataset, record shape, dtype, chunks, compression, attrs, and approximate size.
- Identify candidate video datasets by dimensionality and size.
- Identify possible metadata arrays: time, voltage, current, capacity, cycle, state, frame index.
- Output JSON.
- Never load full video into memory during inspection.

### 7.2 Raw-file manifest builder

Implement:

```bash
python -m cpvideo.io build-manifest \
  --raw-root /path/to/raw \
  --pattern "*.h5" \
  --output data_manifest/raw_files.csv
```

Record path, size, modification time, checksum if cheap enough, schema report path, and notes.

### 7.3 Small video sampler

Implement:

```bash
python -m cpvideo.io sample-video \
  --input /path/to/file.h5 \
  --dataset /hdf5/path/to/video \
  --frames 0 10 20 30 \
  --output derived/samples/file_sample.npz
```

This is for local debugging and napari QC. Avoid copying large data by default.

### 7.4 Preprocessing baseline

Implement:

```bash
python -m cpvideo.preprocess \
  --config configs/preprocessing.yaml \
  --input /path/to/file.h5 \
  --output derived/preprocessed/file.zarr
```

Initial preprocessing steps:

1. optional dark/flat correction;
2. robust frame normalization;
3. illumination correction;
4. registration/drift correction;
5. bad-frame detection;
6. QC metrics and plots.

### 7.5 Segmentation baseline

Implement:

```bash
python -m cpvideo.segment \
  --config configs/segmentation.yaml \
  --input derived/preprocessed/file.zarr \
  --output derived/segmentation/file_masks.zarr \
  --summary reports/segmentation/file_summary.json
```

Start with classical methods:

- thresholding;
- background subtraction;
- watershed;
- connected components;
- morphology filters;
- minimum/maximum particle size constraints.

Optional later variants:

- ilastik export/import;
- SAM2;
- Cellpose-SAM;
- custom U-Net.

### 7.6 Feature extraction

Implement:

```bash
python -m cpvideo.features \
  --config configs/features.yaml \
  --video derived/preprocessed/file.zarr \
  --masks derived/segmentation/file_masks.zarr \
  --metadata data_manifest/file_metadata.json \
  --output experiments/EXP_ID/features/file_features.parquet
```

Feature table should include:

- file ID;
- particle ID;
- frame/time/cycle;
- charge/discharge state;
- x/y centroid;
- area;
- perimeter;
- eccentricity;
- orientation;
- whole-particle mean/median intensity;
- edge intensity;
- core intensity;
- edge-core difference;
- angular-sector response if feasible;
- local background intensity;
- registration residual;
- illumination correction diagnostics;
- QC flags.

### 7.7 Controls

Implement:

```bash
python -m cpvideo.controls \
  --config configs/hypotheses.yaml \
  --features experiments/EXP_ID/features/all_features.parquet \
  --output experiments/EXP_ID/controls/
```

Controls to support:

- shuffled cycle labels;
- shuffled particle IDs;
- spatial permutation;
- background-region analysis;
- no-registration variant;
- alternate illumination-correction variant;
- alternate segmentation variant.

### 7.8 Statistical models

Implement:

```bash
python -m cpvideo.stats \
  --config configs/hypotheses.yaml \
  --features experiments/EXP_ID/features/all_features.parquet \
  --controls experiments/EXP_ID/controls/ \
  --output experiments/EXP_ID/stats/
```

Start with transparent models:

- grouped train/test split by video/cell;
- robust regression;
- mixed-effects regression where possible;
- bootstrap confidence intervals grouped by particle/video;
- FDR correction across many hypotheses/features;
- simple predictive models with heldout videos/cells.

Avoid frame-level leakage.

## 8. First scientific hypotheses

Create `configs/hypotheses.yaml` with these initial hypotheses.

```yaml
hypotheses:
  - id: H001_flux_asymmetry_growth
    claim: "Ion-flux asymmetry increases with cycle number in a subset of NMC particles."
    observable: "Angular or radial optical response asymmetry during charge/discharge."
    prediction: "Asymmetry metrics increase with cycle index and survive illumination/drift controls."
    falsification:
      - "Effect appears equally in background/fiducial regions."
      - "Effect vanishes after illumination correction variants."
      - "Effect is reproduced by shuffled cycle labels."

  - id: H002_edge_core_predicts_degradation
    claim: "Early-cycle edge-core optical gradients predict later-cycle irreversible optical offsets."
    observable: "Edge-core intensity difference and later irreversible offset."
    prediction: "Particles with stronger early edge-core gradients have larger late-cycle residual offsets or slower relaxation."
    falsification:
      - "Prediction disappears after controlling for particle size, brightness, position, and video/cell."
      - "Prediction is matched by shuffled particle or cycle controls."

  - id: H003_charge_discharge_nonreciprocity
    claim: "Charge/discharge optical trajectories become increasingly non-reciprocal with aging."
    observable: "Difference between charge-path and discharge-path optical signal at matched voltage/capacity/time proxy."
    prediction: "Hysteresis/non-reciprocity grows with cycle index."
    falsification:
      - "Non-reciprocity tracks global illumination/focus drift."
      - "Effect is not robust to registration or normalization variants."

  - id: H004_spatial_gradient_degradation
    claim: "Electrode-scale spatial position predicts particle-level degradation signatures."
    observable: "Spatial maps of response lag, hysteresis, irreversible offset, or dropout probability."
    prediction: "Certain electrode regions exhibit earlier or stronger degradation metrics."
    falsification:
      - "Spatial pattern matches illumination residuals."
      - "Spatial pattern vanishes across videos/cells."

  - id: H005_morphology_kinetics
    claim: "Particle morphology predicts local optical kinetic response."
    observable: "Area, eccentricity, orientation, roughness, neighbor density vs response time/hysteresis/edge-core gradient."
    prediction: "Morphology remains predictive after brightness, position, and video/cell controls."
    falsification:
      - "Effect is not robust across segmentation methods."
      - "Effect disappears after controlling for SNR or particle size."
```

## 9. Experiment directory standard

Every experiment must use this structure:

```text
experiments/EXP-YYYYMMDD-HYPOTHESIS-SHORTNAME-vNN/
├── hypothesis.json
├── config.yaml
├── git_commit.txt
├── data_manifest.json
├── environment.txt
├── slurm_job_ids.txt
├── logs/
├── features/
├── controls/
├── stats/
├── figures/
├── metrics.json
├── qc_summary.md
├── skeptic_review.md
└── conclusion.md
```

Each `conclusion.md` must contain:

```markdown
# Conclusion

## Claim tested

## Data used

## Methods

## Main result

## Controls

## Robustness checks

## Failure modes / limitations

## Skeptical review

## Verdict

One of: reject, inconclusive, promising, validation-ready.
```

## 10. Minimal Snakemake workflow

Implement `workflows/Snakefile` with rules for:

1. inspect HDF5;
2. preprocess;
3. segment;
4. extract features;
5. run controls;
6. run stats;
7. make report.

Skeleton:

```python
configfile: "configs/dataset.yaml"

rule all:
    input:
        expand("reports/{sample}/summary.html", sample=config["samples"])

rule inspect:
    input:
        h5=lambda wildcards: config["files"][wildcards.sample]
    output:
        "data_manifest/hdf5_schema_reports/{sample}.json"
    shell:
        "python -m cpvideo.inspect_hdf5 --input {input.h5} --output {output}"

rule preprocess:
    input:
        h5=lambda wildcards: config["files"][wildcards.sample],
        schema="data_manifest/hdf5_schema_reports/{sample}.json"
    output:
        directory("derived/preprocessed/{sample}.zarr")
    shell:
        "python -m cpvideo.preprocess --config configs/preprocessing.yaml --input {input.h5} --schema {input.schema} --output {output}"

rule segment:
    input:
        video=directory("derived/preprocessed/{sample}.zarr")
    output:
        masks=directory("derived/segmentation/{sample}_masks.zarr"),
        summary="reports/segmentation/{sample}_summary.json"
    shell:
        "python -m cpvideo.segment --config configs/segmentation.yaml --input {input.video} --output {output.masks} --summary {output.summary}"

rule features:
    input:
        video=directory("derived/preprocessed/{sample}.zarr"),
        masks=directory("derived/segmentation/{sample}_masks.zarr")
    output:
        "experiments/{exp}/features/{sample}_features.parquet"
    shell:
        "python -m cpvideo.features --config experiments/{wildcards.exp}/config.yaml --video {input.video} --masks {input.masks} --output {output}"
```

Adapt to the actual repository and HDF5 schema.

## 11. Slurm job template

Create `slurm/feature_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cp_features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/cp_features_%A_%a.out
#SBATCH --error=logs/cp_features_%A_%a.err

set -euo pipefail

source ~/.bashrc || true
conda activate cpvideo || mamba activate cpvideo

python -m cpvideo.features \
  --config "$CONFIG_PATH" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "$OUTPUT_ROOT"
```

Do not assume the partition name or GPU request format is correct. Inspect the target cluster first.

## 12. Agent architecture to implement later

After the deterministic pipeline works, implement a LangGraph workflow with these roles.

### Supervisor / PI agent

- chooses the next step;
- enforces human approval gates;
- refuses untracked or unsafe experiments;
- summarizes status.

### LiteratureScout

Outputs structured notes:

```json
{
  "paper_or_source": "...",
  "mechanism": "...",
  "observable_prediction": "...",
  "analysis_idea": "...",
  "controls": ["..."],
  "risk": "..."
}
```

### HypothesisGenerator

Outputs:

```json
{
  "hypothesis_id": "H002",
  "claim": "...",
  "mechanism": "...",
  "observable_prediction": "...",
  "required_data": "...",
  "analysis_method": "...",
  "confounders": ["..."],
  "falsification_tests": ["..."],
  "minimum_evidence": "..."
}
```

### MethodDesigner

Creates analysis variants and configs.

### CodingAgent

Writes modular code and tests.

### HPCExecutionAgent

Submits and monitors jobs. It may run:

- `snakemake --dry-run`
- small sample jobs;
- full jobs only after config exists and user/PI approval is recorded.

### StatsSkeptic

Attempts to reject claims by checking:

- leakage;
- grouped dependence;
- multiple testing;
- segmentation instability;
- optical artifacts;
- confounds by brightness/size/position/video/cell;
- whether negative controls fail appropriately.

### ReportAgent

Writes evidence summaries, not overclaims.

## 13. First milestone

Build a minimal end-to-end test for:

> Early-cycle edge-core optical gradients predict later-cycle irreversible optical offsets.

Required outputs:

1. HDF5 schema report for at least one file.
2. Preprocessed sample video.
3. Particle segmentation masks.
4. Per-particle feature table.
5. Edge/core traces for selected particles.
6. Definition of early-cycle feature and late-cycle degradation metric.
7. Baseline regression or predictive model.
8. Grouped or particle-aware uncertainty estimate.
9. Shuffled-cycle or shuffled-particle negative control.
10. QC report with drift/illumination/segmentation caveats.

## 14. Suggested first commands for coding agents

Run these in order, adapting paths as needed.

```bash
# 1. Inspect current repo
pwd
ls -la
find . -maxdepth 2 -type f | sort | sed 's#^./##' | head -200

# 2. Create structure
mkdir -p configs data_manifest/hdf5_schema_reports src/cpvideo workflows/rules slurm notebooks experiments derived reports tests docs environment

# 3. Create Python package skeleton
touch src/cpvideo/__init__.py

# 4. Add AGENTS.md from this spec

# 5. Implement inspect_hdf5 first
python -m cpvideo.inspect_hdf5 --help

# 6. Add tests for inspect_hdf5 using a tiny synthetic HDF5 file
pytest -q

# 7. Run inspector on one real file only after config/raw path is known
```

## 15. Implementation order

1. `AGENTS.md`
2. `docs/data_contract.md`
3. package skeleton
4. HDF5 inspector
5. raw manifest builder
6. sample video extraction
7. napari QC notebook
8. preprocessing baseline
9. segmentation baseline
10. feature extraction
11. controls
12. stats
13. Snakemake workflow
14. Slurm templates
15. LangGraph orchestration

Do not implement LangGraph first. The deterministic pipeline must exist before agent orchestration is useful.

## 16. Definition of done for phase 1

Phase 1 is complete when a coding agent can run one command that, on a small sample, produces:

- schema report;
- preprocessed video or frame stack;
- segmentation masks;
- particle features;
- QC plots;
- one hypothesis test;
- one negative control;
- one markdown report.

Example target command:

```bash
python -m cpvideo.run_phase1 \
  --config configs/phase1.yaml \
  --sample-id SAMPLE001 \
  --output experiments/EXP-phase1-smoke
```

The report must say whether the result is `reject`, `inconclusive`, `promising`, or `validation-ready`.

## 17. Key scientific caution

The agentic system should generate hypotheses and scale analysis, but it must not lower the evidentiary bar. Optical battery videos are vulnerable to drift, focus variation, illumination artifacts, segmentation errors, repeated-measure leakage, and post-hoc multiple testing. The publication value will come from robust controls and careful mechanistic interpretation, not from the fact that an AI agent suggested the analysis.
