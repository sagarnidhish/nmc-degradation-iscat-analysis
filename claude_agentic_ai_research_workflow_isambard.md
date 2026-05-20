# Multi-Agent Agentic AI for Scientific Research: Complete Setup Guide for Isambard HPC

## Executive Summary

This guide covers setting up a multi-agent agentic AI system for automated research workflows, specifically for analyzing operando optical microscopy video data of NMC battery degradation on the Isambard supercomputer. The system leverages Claude Code, Codex CLI, and open-source orchestration frameworks to run parallel hypothesis generation, experiment design, and data analysis agents.

**Key insight:** Isambard's ARM64 (aarch64) architecture requires running AI agents on your local x86_64 machine, with compute jobs submitted remotely via SLURM-MCP. This is actually cleaner than traditional setups.

---

## Part 1: Tool Landscape (2025-2026)

### 1.1 Primary Agent Tools

#### Claude Code (Anthropic)
- **Architecture:** Terminal-first CLI, also in IDE extensions (VS Code, JetBrains) and desktop app
- **Models:** Claude Opus 4.6 and Claude Sonnet 4.6
- **Context:** Up to 1M tokens (Opus beta)
- **Strengths:** Deep reasoning, long-context codebase understanding, native git worktree support for parallel agents
- **New feature:** Agent Teams (research preview) — multiple Claude Code sessions coordinated by a lead session
- **Cost:** Fixed subscription ($20/month base, higher tiers for more compute)
- **Best for:** Complex scientific code, long-horizon reasoning, publishable work requiring deep analysis

#### Codex CLI (OpenAI)
- **Architecture:** Open source (Apache 2.0), terminal-based, also cloud agent available
- **Models:** GPT-5.3-Codex, GPT-5.4 (can use via ChatGPT subscription)
- **Context:** 256K tokens default, 1M with GPT-5.4
- **Strengths:** Token efficiency (4x fewer tokens than Claude Code), open source, OS-level sandboxing (Seatbelt on macOS, Landlock on Linux)
- **Cost:** Cheap ($20/month via ChatGPT Plus if using cloud agent, or pay-per-token via API)
- **Best for:** Parallel variation runs, cost-sensitive workflows, multi-model flexibility

#### Open Source Alternatives
- **opencode:** Most flexible, supports Claude/OpenAI/Google/local models in the same tool, TUI workflow
- **Aider:** Oldest mature option, excellent for git workflows, good for iterative coding
- **Goose:** Rising alternative, lighter weight

### 1.2 Multi-Agent Orchestration

#### Claude Code Native Features
- Git worktree support built-in (`claude --worktree feature-name`)
- Each worktree gets isolated checkout + branch
- Desktop app auto-creates worktree per session
- New Agent Teams feature: lead session assigns subtasks, tracks changes, coordinates across parallel sessions

#### Framework Options
- **Composio's agent-orchestrator:** Agent-agnostic (Claude Code, Codex, Aider), manages git worktrees, auto-fixes CI failures, handles merge conflicts
- **ccswarm:** Claude Code-specific orchestrator, uses git worktree isolation + message-bus coordination
- **AutoGen (Microsoft):** General-purpose multi-agent framework, best for role-based teams
- **CrewAI:** Easiest learning curve, good for "scientist crew" with specialized agent roles (literature reviewer, hypothesis generator, code writer, critic)
- **LangGraph:** Maximum flexibility, good for complex workflows with branching logic

#### Scientific-Specific Frameworks
- **Sakana AI Scientist v2:** Fully autonomous end-to-end research (hypothesis generation → experiment design → data analysis → LaTeX paper). Open source (MIT), but output quality is "unmotivated undergraduate rushing deadline" — use for ideas, not rigor.
- **VibeCodeHPC (2025):** Multi-LLM agent system specializing in HPC programming, uses Claude Code backend, auto-tunes performance

### 1.3 Comparison Table

| Tool | Model(s) | Best For | Deployment | Cost | Notes |
|------|----------|----------|------------|------|-------|
| Claude Code | Claude Opus/Sonnet 4.6 | Complex reasoning, long-context | Local terminal, IDE | $20+/mo | Deep scientific code, native worktrees |
| Codex CLI | GPT-5.x | Cheap parallel runs | Local terminal, cloud agent | $20/mo or pay-per-token | Token efficient, open source |
| opencode | Any model | Maximum flexibility | Terminal TUI | Free (open source) | Switch models per task, no lock-in |
| AutoGen | Any LLM | Multi-agent coordination | Python framework | Free (open source) | Good for role-based teams, async patterns |
| CrewAI | Any LLM | "Scientist crew" simulation | Python framework | Free (open source) | Easiest for role-based agents (literature, hypothesis, code, critic) |
| Sakana AI Scientist v2 | Claude (configurable) | Autonomous research pipelines | Python, self-contained | Free (open source) | End-to-end automation, but lower rigor |

---

## Part 2: Isambard-Specific Architecture & Constraints

### 2.1 Hardware & Architecture Reality

**Isambard-AI Phase 1 & 2 (GPU systems):**
- **CPU:** NVIDIA Grace CPU (ARM64 / aarch64 architecture)
- **GPU:** NVIDIA H100 Tensor Core (4 per node)
- **Memory:** 460 GB CPU + 384 GB GPU per node
- **Interconnect:** Slingshot 11 (200 Gbps per NIC)

**Critical constraint:** Isambard uses ARM64 (aarch64), not x86_64. Most AI tools are distributed as x86_64 binaries.

**Isambard 3 MACS (Multi-Architecture):** Also available as alternative — has x86_64 nodes (Milan, Genoa, Sapphire Rapids) with some GPU support (A100, H100, MI100). Smaller system.

**BlueCrystal 5:** x86_64-based, smaller GPU allocation.

### 2.2 The Architecture Mismatch Problem

- Claude Code: x86_64 binary distribution, cannot run natively on aarch64
- Codex CLI: Also primarily x86_64, though open source (could recompile for aarch64 in theory, but non-trivial)
- Python packages: Most have aarch64 wheels (NumPy, scikit-image, h5py, Dask, PyTorch, etc.), but special care needed for some

**Solution:** Run agents on your local x86_64 machine; submit compute jobs to Isambard via SSH + MCP.

### 2.3 Recommended Architecture

```
┌─────────────────────────────────────────┐
│  Your Laptop/Workstation (x86_64)       │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ Claude Code or Codex CLI         │  │
│  │ (Agent orchestration)            │  │
│  └──────────────────────────────────┘  │
│           │                             │
│           │ SSH + MCP                   │
│           │                             │
└───────────┼─────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│  Isambard Login Node (aarch64)          │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ slurm-mcp Server                 │  │
│  │ (wraps sbatch, squeue, file I/O) │  │
│  └──────────────────────────────────┘  │
│           │                             │
│           │ SLURM                       │
│           │                             │
└───────────┼─────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│  Isambard Compute Nodes (aarch64 GPUs)  │
│                                         │
│  Your Python analysis scripts           │
│  (h5py, Dask, SAM2, xarray, etc.)      │
│  Run on H100s                           │
└─────────────────────────────────────────┘
```

**Why this works:**
- Agent logic stays on your machine (where Claude Code/Codex run natively)
- Pure-Python analysis code runs on Isambard (architecture-agnostic NumPy/PyTorch)
- Git worktree coordination happens locally
- SLURM job submission/monitoring is transparent via MCP

---

## Part 3: Step-by-Step Setup

### 3.1 Local Machine Setup (Your Laptop)

#### 3.1.1 Install Claude Code

```bash
# On macOS
brew install anthropics/cask/claude-code

# On Linux (Debian/Ubuntu)
curl -fsSL https://docs.claude.com/downloads/install.sh | bash

# Or use the desktop app
# Download from https://claude.ai/download
```

Verify installation:
```bash
claude --version
```

#### 3.1.2 Install Codex CLI (Optional, for parallel runs)

```bash
npm install -g @openai/codex
```

Or use the Rust version:
```bash
cargo install openai/codex-cli
```

#### 3.1.3 Configure SSH Key-Based Auth to Isambard

Generate SSH key (if you don't have one):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/isambard_key -N ""
```

Add to your Isambard account via the BriCS portal or helpdesk.

Test SSH connection:
```bash
ssh -i ~/.ssh/isambard_key your_username@login.isambard.ac.uk
# You should get a shell without password prompt
```

#### 3.1.4 Install Local Python Environment (for local testing)

```bash
python3 -m venv ~/.virtualenvs/research
source ~/.virtualenvs/research/bin/activate
pip install h5py numpy scipy scikit-image opencv-python torch torchvision dask xarray zarr
```

### 3.2 Isambard Login Node Setup

SSH into Isambard login node:
```bash
ssh -i ~/.ssh/isambard_key your_username@login.isambard.ac.uk
```

#### 3.2.1 Install slurm-mcp (MCP server for SLURM)

On the login node:
```bash
cd $HOME
git clone https://github.com/dongwookim-ml/slurm-mcp.git
cd slurm-mcp
# Follow installation instructions (likely Python-based)
pip install -e .
```

Verify SLURM is accessible:
```bash
sbatch --version
squeue --help
```

#### 3.2.2 Set Up Python Environment on Isambard

Isambard uses aarch64, so packages must be built for that architecture. Install Miniforge:

```bash
cd $HOME
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -o Miniforge3-installer.sh
bash Miniforge3-installer.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/bin/activate
```

Create a conda environment for analysis:
```bash
conda create -n research python=3.11
conda activate research
conda install -c conda-forge h5py numpy scipy scikit-image opencv dask xarray zarr pytorch::pytorch pytorch::pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Important:** Check conda-forge for aarch64 compatibility at https://anaconda.org (filter platform to `linux-aarch64`).

#### 3.2.3 Data Directory Setup

Create directories for data and results:
```bash
mkdir -p $WORK/nmc_photometry/{data,results,analysis_scripts}
cd $WORK/nmc_photometry
```

Copy your HDF5 files to `$WORK/nmc_photometry/data/`.

#### 3.2.4 Create Initial CLAUDE.md and AGENTS.md

In your project directory on Isambard (`$WORK/nmc_photometry`), create `CLAUDE.md`:

```markdown
# NMC Battery Photometry Analysis

## Project Goals
- Analyze operando optical scattering microscopy video data of NMC battery cycling
- Identify kinetically-induced lithium heterogeneities and degradation mechanisms
- Generate publishable scientific findings on single-particle degradation

## Data Structure
- Located in: `$WORK/nmc_photometry/data/`
- Format: HDF5 (.h5 or .hdf5)
- Content: Video frames (time × height × width), electrochemical curve, cycle metadata
- Size: [Your data size]
- Processing: Convert to Zarr for parallel access

## Known Science (from literature)
- Merryweather et al. (2022): Kinetically-induced Li heterogeneities in NMC particles
  - Upon delithiation: Li-poor periphery, Li-rich core
  - At end of discharge: Li-rich surface prevents complete lithiation
- Recent (2025): Single-crystal NMC governed by mechanical failure modes (Argonne/UChicago)
- Phase transitions: H2 → H3 in high-Ni materials with distinct optical signatures

## Research Questions (Priority Order)
1. **Cycle-resolved heterogeneity evolution:** Do particles develop memory across cycles?
2. **Pre-failure optical signatures:** Can we predict particle failure 10-100 cycles in advance?
3. **Population statistics:** Distribution of heterogeneity modes across electrode?
4. **Phase-transition kinetics:** Sub-particle resolution tracking of H2→H3 transition?

## Workflow
1. Data preprocessing: Convert HDF5 → Zarr, create per-particle intensity time-series
2. Segmentation: Use SAM3 for particle tracking across video frames
3. Feature extraction: Optical flow + image registration for deformation tracking
4. Analysis: Heterogeneity metrics, crack detection, phase transition identification
5. Hypothesis testing: Correlation analysis, statistical validation

## Success Criteria
- All analyses reproducible with fixed random seeds
- Test suite with known synthetic particles passes
- Each claim backed by re-runnable script with ground-truth validation
- Ready for publication

## Notes
- Do not modify shared files (package-lock.json, migrations) unless explicitly tasked
- Always write RESULTS.md in each worktree with summary of findings
- Update CHANGELOG.md with progress, failed approaches, and why they didn't work
```

### 3.3 Configure Claude Code to Use slurm-mcp

On your local machine, configure Claude Code to connect to Isambard's slurm-mcp server.

Create/edit `~/.claude/mcp-config.json` (or use Claude Code's settings UI):

```json
{
  "mcpServers": {
    "slurm-isambard": {
      "command": "ssh",
      "args": [
        "-i", "/Users/yourname/.ssh/isambard_key",
        "your_username@login.isambard.ac.uk",
        "slurm-mcp"
      ],
      "env": {
        "SLURM_MCP_HOME_DIR": "/home/your_username",
        "SLURM_MCP_DATA_DIR": "/work/your_project/nmc_photometry/data",
        "SLURM_MCP_SCRATCH_DIR": "/work/your_project/nmc_photometry/results",
        "SLURM_MCP_HOME_QUOTA_GB": "1000"
      }
    }
  }
}
```

Verify the connection:
```bash
claude --chat "Test SLURM connection"
# In Claude, ask: "@slurm squeue" to list current jobs
```

### 3.4 Set Up Git Worktrees for Parallel Agents (Local)

Initialize a local git repo for your analysis project (on your laptop):

```bash
mkdir -p ~/research/nmc-analysis
cd ~/research/nmc-analysis
git init
git config user.email "your_email@example.com"
git config user.name "Your Name"

# Create initial files
cat > CLAUDE.md << 'EOF'
# NMC Photometry Analysis - Agent Orchestration
[Copy content from 3.2.4 above]
EOF

cat > CHANGELOG.md << 'EOF'
# Project Changelog

## Session 1: Initial Setup
- Goal: Establish baseline analysis pipeline
- Status: In progress
EOF

git add .
git commit -m "Initial project setup"

# Create .gitignore
cat > .gitignore << 'EOF'
.claude/
.env
.env.local
*.pyc
__pycache__/
*.h5
*.hdf5
.DS_Store
EOF

git add .gitignore
git commit -m "Add gitignore"
```

### 3.5 Data Engineering: Convert HDF5 to Zarr

Before agents start, prepare data for parallel processing.

Create a script `prepare_data.py` on Isambard:

```python
import h5py
import zarr
import numpy as np
import dask.array as da
from pathlib import Path

def convert_hdf5_to_zarr(input_hdf5, output_zarr, chunk_size_mb=100):
    """
    Convert HDF5 video dataset to Zarr for parallel access.
    
    Args:
        input_hdf5: Path to .h5 file
        output_zarr: Path where .zarr will be created
        chunk_size_mb: Target chunk size in MB
    """
    with h5py.File(input_hdf5, 'r') as h5f:
        # List datasets
        print(f"Datasets in {input_hdf5}:")
        h5f.visit(lambda name: print(f"  {name}"))
        
        # Assuming video data is in 'video' or 'frames' dataset
        video_key = 'video' if 'video' in h5f else 'frames'
        video = h5f[video_key]
        
        shape = video.shape
        dtype = video.dtype
        
        # Calculate chunk size (aim for ~100MB chunks)
        bytes_per_element = np.dtype(dtype).itemsize
        elements_per_chunk = (chunk_size_mb * 1024 * 1024) // bytes_per_element
        
        # Chunk along time and spatial dimensions
        chunk_time = max(1, elements_per_chunk // (shape[1] * shape[2]))
        chunks = (chunk_time, shape[1], shape[2])
        
        print(f"Shape: {shape}, Chunks: {chunks}")
        
        # Create Zarr array
        root = zarr.open(str(output_zarr), mode='w')
        z = root.create_dataset(
            video_key,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=zarr.Blosc(cname='zstd', clevel=5)
        )
        
        # Copy data in chunks
        print(f"Converting {shape[0]} frames...")
        for i in range(0, shape[0], chunk_time):
            end_i = min(i + chunk_time, shape[0])
            z[i:end_i] = video[i:end_i]
            print(f"  Frame {end_i}/{shape[0]}")
    
    print(f"✓ Converted to {output_zarr}")

if __name__ == "__main__":
    # Adjust these paths
    input_file = "/work/your_project/nmc_photometry/data/nmc_cycle_data.h5"
    output_zarr = "/work/your_project/nmc_photometry/data/nmc_cycle_data.zarr"
    
    convert_hdf5_to_zarr(input_file, output_zarr)
```

Run it once on Isambard:
```bash
cd $WORK/nmc_photometry
python prepare_data.py
```

---

## Part 4: The Multi-Agent Workflow

### 4.1 Single-Agent Baseline (Week 1)

Start with Claude Code locally, working against remote Isambard compute via MCP.

```bash
cd ~/research/nmc-analysis
claude
```

In the Claude Code session:

```
> You are a scientific data analysis assistant helping analyze NMC battery degradation.
> Read CLAUDE.md for context.
> 
> Task 1: Write a Python script that:
> 1. Loads video frames from Zarr at /work/your_project/nmc_photometry/data/nmc_cycle_data.zarr
> 2. Performs frame-by-frame intensity statistics (mean, std, min, max per frame)
> 3. Saves results to CSV
> 4. Creates a matplotlib plot of intensity evolution
>
> The script should be:
> - Reproducible (fixed random seed where needed)
> - Well-tested (include validation against a known synthetic particle)
> - Documented for future agents to understand
>
> Write the code, test it with sbatch on Isambard, and commit to git.
```

Claude Code will:
1. Write `intensity_analysis.py`
2. Test locally if possible, or write a SLURM batch script
3. Submit to Isambard via `sbatch` (using slurm-mcp)
4. Monitor job status
5. Collect results
6. Commit to git with meaningful message

### 4.2 Parallel Hypothesis Variants (Week 2-3)

Once baseline works, spawn parallel agents exploring different hypotheses.

```bash
cd ~/research/nmc-analysis

# Create worktrees for different hypotheses
claude --worktree hyp-heterogeneity-memory
claude --worktree hyp-crack-precursors
claude --worktree hyp-phase-transition-kinetics
```

Each worktree runs in its own branch. In each session:

```
> Read CLAUDE.md and /work/your_project/nmc_photometry/data/zarr.
> 
> Hypothesis: [Specific hypothesis from CLAUDE.md]
> 
> Write a Python analysis pipeline that:
> 1. Segments individual particles using SAM3
> 2. Extracts heterogeneity metric (e.g., core-periphery intensity ratio)
> 3. Tracks this metric across multiple cycles
> 4. Tests the hypothesis (e.g., does this metric predict cycle lifetime?)
> 5. Writes results to RESULTS.md in this worktree
> 
> Include:
> - Clear success criteria
> - Test on 5 representative particles
> - Plots showing the findings
> - Statistical validation
```

### 4.3 Orchestrator / Merger Agent (Week 4)

Create a coordinator session (or human review):

```bash
claude  # Main orchestrator session
```

```
> Review all parallel worktrees in git:
> - hyp-heterogeneity-memory: [summary from RESULTS.md]
> - hyp-crack-precursors: [summary from RESULTS.md]
> - hyp-phase-transition-kinetics: [summary from RESULTS.md]
>
> Compare their findings:
> 1. Which hypothesis has strongest evidence?
> 2. Are findings contradictory or complementary?
> 3. What new questions emerge?
> 4. Which results should be merged into main branch?
> 5. What new parallel experiments would be valuable?
>
> Update CHANGELOG.md with synthesis and next steps.
> Create new worktrees for round 2 based on findings.
```

Merge winning branches back to main:
```bash
git merge hyp-heterogeneity-memory
git merge hyp-crack-precursors
# Delete non-winning branches
```

### 4.4 Hypothesis Generation Layer (When Ready)

Once you have a few rounds of solid baseline results, add a "scientist crew":

Create `scientist_crew.py`:

```python
from crewai import Agent, Task, Crew

# Initialize agents
literature_agent = Agent(
    name="Literature Reviewer",
    role="Search and synthesize recent battery degradation literature",
    goal="Find emerging hypotheses about NMC single-particle failure mechanisms",
    backstory="Expert in battery materials science with access to recent preprints",
    tools=[web_search, arxiv_search],  # These would be actual tools
)

hypothesis_generator = Agent(
    name="Hypothesis Generator",
    role="Generate novel research hypotheses",
    goal="Propose testable hypotheses from literature + your existing data",
    backstory="Creative scientist who combines domain knowledge with data",
)

experiment_designer = Agent(
    name="Experiment Designer",
    role="Design Python analysis pipelines",
    goal="Turn hypotheses into executable analysis plans",
    backstory="Expert at translating ideas into code",
)

critic = Agent(
    name="Critical Reviewer",
    role="Evaluate hypotheses and findings for rigor",
    goal="Ensure all claims are scientifically sound and reproducible",
    backstory="Rigorous scientist who values reproducibility and statistical validity",
)

# Define tasks
literature_task = Task(
    description="Find 5 recent papers on NMC single-particle degradation. Summarize novel mechanisms.",
    agent=literature_agent,
)

hypothesis_task = Task(
    description="""Given the literature and our existing data on Li heterogeneity and cracking,
    propose 3 novel, testable hypotheses about NMC degradation that we haven't explored yet.""",
    agent=hypothesis_generator,
)

design_task = Task(
    description="For each hypothesis, outline a Python analysis pipeline: inputs, processing, outputs, success metrics.",
    agent=experiment_designer,
)

critique_task = Task(
    description="Evaluate each proposed hypothesis and pipeline for scientific rigor, testability, and feasibility.",
    agent=critic,
)

# Create crew
crew = Crew(
    agents=[literature_agent, hypothesis_generator, experiment_designer, critic],
    tasks=[literature_task, hypothesis_task, design_task, critique_task],
    verbose=True,
)

result = crew.kickoff()
print(result)
```

Run this periodically (e.g., weekly) to generate new ideas:
```bash
python scientist_crew.py > hypothesis_ideas.txt
```

Then manually (or via Claude) turn top ideas into new worktrees.

---

## Part 5: Domain-Specific Guidance: NMC Photometry Analysis

### 5.1 What's Already Known (Don't Rediscover)

From Merryweather et al. (Nature 2021, Cambridge):
- Operando optical scattering microscopy can track Li-ion dynamics in single NMC particles
- Two kinetically-induced heterogeneities observed:
  1. On delithiation (charge): rapid diffusivity rise → Li-poor core, Li-rich periphery
  2. On end of discharge: slow diffusion at full lithiation → Li-rich surface

From recent work (2025):
- Uniaxial mechanical failure is dominant degradation mode in single-crystal NMC (UChicago)
- Crack propagation follows predictable patterns
- Phase transitions (H2 → H3 in Ni-rich) have characteristic optical signatures

**Don't waste agent time re-measuring these.** Assume they're ground truth and build on them.

### 5.2 High-Probability Research Directions

#### Direction 1: Cycle-Resolved Heterogeneity Evolution (Highest Confidence)

**The gap:** Previous work measured heterogeneity within individual cycles. Nobody has tracked *whether the heterogeneity pattern itself changes* across many cycles.

**Agent task:**
```python
# For each cycle (1, 5, 10, 25, 50, 100):
#   - Segment particles
#   - Measure heterogeneity metric H(t) = core_intensity / periphery_intensity
#   - Fit to diffusion model to extract diffusivity D
#   - Track D evolution: does it change? saturate? oscillate?
# 
# Hypothesis: Particles develop "memory" — heterogeneity patterns repeat cyclically,
# suggesting cyclic strain/stress fields
```

**Success metric:** Can you predict cycle N heterogeneity from cycles 1-10?

#### Direction 2: Pre-Failure Optical Signatures (Medium Confidence)

**The gap:** Which particles will fragment or become electrically isolated? Can you see it coming?

**Agent task:**
```python
# Ground truth: particles that died (became disconnected) vs. those that survived
# (You need to identify these, e.g., from electrochemical data post-mortem)
#
# For each particle:
#   - Extract 30+ features: intensity variance, heterogeneity, crack area, 
#     deformation gradient, phase transition presence, etc.
#   - Test each feature's predictive power for cycle-to-failure
#   - Use logistic regression or random forest to rank feature importance
#
# Hypothesis: A combination of optical signatures can predict failure 
# 10-50 cycles in advance
```

**Success metric:** AUC > 0.8 on hold-out test set. Feature importance makes physical sense.

#### Direction 3: Population-Level Statistics (High Novelty)

**The gap:** Most papers report 2-5 particles. You likely have thousands in your videos.

**Agent task:**
```python
# Process entire electrode (all particles in field of view)
#   - Segment all particles
#   - Measure heterogeneity distribution at each cycle
#   - Fit to distribution (e.g., bimodal? log-normal?)
#   - Track distribution evolution
#
# Questions:
#   - Are there 2 populations: "fast" and "slow" degraders?
#   - Are these spatially correlated? (suggests local electrolyte heterogeneity)
#   - Does population structure change with cycle number?
```

**Success metric:** Statistical evidence for population structure. Spatial autocorrelation analysis.

#### Direction 4: Phase Transition Kinetics (Medium-High Confidence)

**The gap:** H2 ↔ H3 transition is known, but sub-particle resolution dynamics are not.

**Agent task:**
```python
# Known fact: H3 phase has different optical reflectivity than H2
# (This is from spectroscopy, but may show in intensity changes)
#
# For high-Ni particles (Ni > 0.7):
#   - Divide particle into radial zones (center, middle, edge)
#   - Track intensity signature changes → infer phase transition front motion
#   - Measure transition speed, directionality
#   - Correlate with lithiation state (from electrochemical curve)
#
# Hypothesis: Phase transition initiates at surface and propagates inward
# (standard) or vice versa (anomalous)?
```

**Success metric:** Quantitative measurement of transition front velocity, validation against literature.

### 5.3 Tools & Techniques for NMC Video Analysis

#### Segmentation: SAM 2 / SAM 3
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

model = build_sam2(config_file="configs/sam2.1_hiera_l.yaml", checkpoint="checkpoints/sam2.1_hiera_large.pt")
predictor = SAM2ImagePredictor(model)

# For video: use SAM2VideoPredictor for temporal consistency
# Specify particles in frame 1, let it track across frames
# Memory mechanism (SAM3) handles occlusions & re-appearances
```

**Why:** Zero-shot (no training), handles occlusions, maintains particle identity across frames.

#### Deformation Tracking: Optical Flow + Image Registration
```python
import cv2
import numpy as np
from scipy.ndimage import shift

# Optical flow between consecutive frames
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Image registration to correct for stage drift, cell movement
# Use skimage.registration.phase_cross_correlation
shift_amount = phase_cross_correlation(frame1, frame2)

# Per-particle deformation: track corner points in ROI
corners = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.01, minDistance=10)
# Track across frames
```

**Why:** Captures mechanical strain, cracks, heterogeneous deformation.

#### Crack Detection: GREAT2 / GRAPES Toolkit Approach
```python
# Analyze grey-level intensity changes in individual particles
# Cracks appear as darker voxels (partial volume effect in microscopy)

# For each particle ROI:
intensity_trend = []
for frame in video:
    roi = extract_particle_roi(frame, particle_id)
    mean_intensity = roi.mean()
    intensity_trend.append(mean_intensity)

# Detect sudden drops (crack initiation) or oscillations (crack movement)
```

**Why:** Simple, fast, and validated in literature (Heenan et al., PMC 2025).

#### Heterogeneity Metric: Core vs. Periphery
```python
def measure_heterogeneity(particle_roi):
    """
    Measure Li heterogeneity using intensity ratio (correlates with Li content).
    
    High intensity = reduced conduction band electron population = high Li content.
    """
    # Separate core from periphery
    h, w = particle_roi.shape
    center = (h//2, w//2)
    radius = min(h, w) // 2
    
    # Create masks
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center[0])**2 + (xx - center[1])**2)
    
    core_mask = dist < radius * 0.4
    periphery_mask = (dist >= radius * 0.4) & (dist < radius * 0.9)
    
    core_intensity = particle_roi[core_mask].mean()
    periphery_intensity = particle_roi[periphery_mask].mean()
    
    heterogeneity = core_intensity / (periphery_intensity + 1e-6)
    
    return heterogeneity
```

**Why:** Directly measures what Merryweather observed; correlates with Li concentration.

### 5.4 Validation Against Ground Truth

**Critical:** Every agent claim should be testable against synthetic data.

Create synthetic particles with known properties:

```python
def create_synthetic_particle(shape=(256, 256), 
                             core_li=0.8, periphery_li=0.3, 
                             noise_level=0.01):
    """
    Create synthetic particle with known lithiation heterogeneity.
    """
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    center = (shape[0]//2, shape[1]//2)
    dist = np.sqrt((yy - center[0])**2 + (xx - center[1])**2)
    radius = min(shape) // 2
    
    # Radial heterogeneity gradient
    li_content = periphery_li + (core_li - periphery_li) * (1 - dist / radius)
    li_content = np.clip(li_content, 0, 1)
    
    # Intensity correlates with Li: high Li → high intensity
    intensity = 100 + 150 * li_content
    
    # Add noise
    intensity += np.random.normal(0, noise_level * 255, shape)
    intensity = np.clip(intensity, 0, 255).astype(np.uint8)
    
    return intensity

# Test your analysis
synthetic = create_synthetic_particle(core_li=0.8, periphery_li=0.3)
measured_heterogeneity = measure_heterogeneity(synthetic)
expected_heterogeneity = 0.8 / 0.3  # ≈ 2.67

assert abs(measured_heterogeneity - expected_heterogeneity) < 0.1, \
    f"Heterogeneity measurement failed: {measured_heterogeneity} vs {expected_heterogeneity}"
```

Every agent script should include `test_on_synthetic_particles()` function.

---

## Part 6: Reproducibility & Publication Discipline

### 6.1 The Test Oracle Approach

From Anthropic's scientific computing guide: every autonomous analysis needs a "test oracle" — a way for the agent to verify its own work.

For NMC analysis:

```python
# test_oracle.py
import numpy as np
from analysis_module import measure_heterogeneity, detect_cracks, segment_particles

class TestOracle:
    """Validates analysis correctness on synthetic and semi-synthetic data."""
    
    def test_heterogeneity_on_synthetic(self):
        """Test heterogeneity measurement against known synthetic particles."""
        # Test 1: Uniform particle (no heterogeneity)
        uniform = np.ones((256, 256)) * 100
        h = measure_heterogeneity(uniform)
        assert abs(h - 1.0) < 0.05, f"Uniform heterogeneity should be ~1.0, got {h}"
        
        # Test 2: Known heterogeneity
        synthetic = create_synthetic_particle(core_li=0.8, periphery_li=0.3)
        h = measure_heterogeneity(synthetic)
        expected = 0.8 / 0.3
        assert abs(h - expected) / expected < 0.1, f"Heterogeneity mismatch: {h} vs {expected}"
        
    def test_crack_detection_on_synthetic(self):
        """Test crack detection on synthetic cracks."""
        # Create particle with known crack
        particle = np.ones((256, 256)) * 100
        # Add crack (darker region)
        particle[100:110, :] = 50
        
        cracks_detected = detect_cracks(particle)
        assert len(cracks_detected) > 0, "Should detect synthetic crack"
        
    def test_segmentation_on_known_particles(self):
        """Test particle segmentation on annotated real data (if available)."""
        # If you have a few manually-annotated frames, use them
        frame = load_annotated_frame()
        manual_mask = load_ground_truth_mask()
        
        predicted_mask = segment_particles(frame)[0]
        iou = intersection_over_union(predicted_mask, manual_mask)
        assert iou > 0.7, f"Segmentation IOU too low: {iou}"

def run_test_oracle():
    """Run full validation suite. Agents call this before committing results."""
    oracle = TestOracle()
    tests = [
        oracle.test_heterogeneity_on_synthetic,
        oracle.test_crack_detection_on_synthetic,
        oracle.test_segmentation_on_known_particles,
    ]
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            return False
    
    print("\n✓ All tests passed. Ready to run on real data.")
    return True

if __name__ == "__main__":
    success = run_test_oracle()
    exit(0 if success else 1)
```

Add to every agent task:

```
> Before submitting results:
> 1. Run `python test_oracle.py`
> 2. All tests must pass
> 3. Commit with message: "Results validated against test oracle"
```

### 6.2 Documenting AI Involvement

Journals increasingly require disclosure of AI usage. Standard format:

> **AI Assistance Disclosure:** 
> 
> This research used Claude Code (Anthropic) and Codex CLI (OpenAI) as AI assistants for:
> - Code generation and debugging: Python analysis scripts for video processing, statistical analysis, and visualization
> - Hypothesis generation: Initial brainstorming of research directions based on literature
> - Literature synthesis: Summarizing relevant background from existing publications
> 
> All scientific claims and interpretations were validated by the authors through manual review, ground-truth testing, and statistical validation. The AI tools did not contribute to the core interpretation or conclusions beyond code implementation. All results are reproducible via the provided scripts.

### 6.3 Reproducibility Checklist

Before submitting a paper:

- [ ] All scripts include fixed random seeds (`np.random.seed(42)`, `torch.manual_seed(42)`)
- [ ] Every result can be regenerated by running a single command
- [ ] Synthetic test cases pass (test_oracle.py)
- [ ] README.md explains data structure, how to run analysis, expected outputs
- [ ] CHANGELOG.md documents all attempts (including failures) and why they didn't work
- [ ] Environment files (environment.yml, requirements.txt) specify exact versions
- [ ] Scripts handle edge cases (missing frames, bad segmentations, etc.)
- [ ] Statistical claims include effect sizes, confidence intervals, p-values
- [ ] Data (or links to data) are publicly available or supplementary

---

## Part 7: Timeline & Execution Plan

### Week 1: Local Setup & Baseline
- [ ] Install Claude Code, configure for Isambard via slurm-mcp
- [ ] Test SSH connection, create conda environment on Isambard
- [ ] Convert HDF5 → Zarr
- [ ] Write intensity analysis baseline script, submit via SLURM
- [ ] Create git repo, first commit

### Week 2-3: Single-Agent Deepening
- [ ] Claude Code writes segmentation pipeline (SAM3)
- [ ] Add optical flow + image registration
- [ ] Measure heterogeneity, crack metrics per particle
- [ ] Generate plots, summary statistics
- [ ] Update CHANGELOG.md with findings

### Week 4: Parallel Hypothesis Exploration
- [ ] Create 3 worktrees (heterogeneity evolution, pre-failure signatures, population stats)
- [ ] Each worktree implements hypothesis-specific pipeline
- [ ] Run test_oracle.py in each
- [ ] Collect RESULTS.md from each

### Week 5: Synthesis & Second Round
- [ ] Orchestrator reviews all findings
- [ ] Merge winning branches
- [ ] Spawn next round of focused hypotheses
- [ ] Iterate

### Week 6-8: Publication Preparation
- [ ] Add hypothesis generation layer (CrewAI crew)
- [ ] Explore 2-3 entirely new ideas
- [ ] Statistical validation, effect sizes
- [ ] Write paper outline
- [ ] Prepare supplementary code/data

---

## Part 8: Common Pitfalls & Solutions

### Pitfall 1: Agent hallucinations about data structure
**Solution:** Write test_oracle.py that validates data assumptions.

### Pitfall 2: Jobs fail silently on Isambard, nobody notices
**Solution:** Agent should parse sbatch output, check `sacct`, read stderr, retry if needed.

```python
import subprocess
import time

def submit_and_monitor_job(script_path, timeout_hours=24):
    """Submit SLURM job and wait for completion."""
    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")
    
    job_id = result.stdout.split()[-1]
    print(f"Submitted job {job_id}")
    
    # Poll job status
    start_time = time.time()
    while time.time() - start_time < timeout_hours * 3600:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--parsable2"],
            capture_output=True,
            text=True
        )
        
        status = result.stdout.strip().split('\n')[-1]  # Last line is actual status
        
        if "COMPLETED" in status:
            print(f"Job {job_id} completed")
            return job_id
        elif "FAILED" in status or "CANCELLED" in status:
            # Get error logs
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True
            )
            raise RuntimeError(f"Job {job_id} failed: {result.stdout}")
        
        print(f"Job {job_id} status: {status}, waiting...")
        time.sleep(30)
    
    raise TimeoutError(f"Job {job_id} timed out after {timeout_hours} hours")
```

### Pitfall 3: HDF5 I/O bottleneck
**Solution:** Convert to Zarr once, read from Zarr forever (or use kerchunk for parallel HDF5 access).

### Pitfall 4: Multiple agents modifying package-lock.json simultaneously
**Solution:** Each worktree gets its own Python environment, don't install during agent run.

### Pitfall 5: Agent generates low-quality visualizations
**Solution:** Specify matplotlib style, color schemes, plot requirements upfront.

```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

def plot_heterogeneity_evolution(cycles, heterogeneity_metrics, particle_id):
    """
    Standard plot for heterogeneity over cycles.
    
    Requirements:
    - x-axis: cycle number (log scale if >100 cycles)
    - y-axis: heterogeneity metric (ratio or percentage)
    - Include error bars (std across particles of similar size)
    - Include horizontal line at initial value
    - Title includes particle ID and observation type
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [m['mean'] for m in heterogeneity_metrics]
    stds = [m['std'] for m in heterogeneity_metrics]
    
    ax.errorbar(cycles, means, yerr=stds, fmt='o-', capsize=5, label='Measured')
    ax.axhline(y=means[0], color='r', linestyle='--', label='Initial value')
    
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Heterogeneity (core/periphery intensity ratio)', fontsize=12)
    ax.set_title(f'Li Heterogeneity Evolution: Particle {particle_id}', fontsize=14)
    ax.set_xscale('log' if max(cycles) > 100 else 'linear')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

---

## Part 9: Additional Resources & References

### Tool Documentation
- Claude Code: https://code.claude.com/docs/
- Codex CLI: https://github.com/openai/codex-cli (open source)
- slurm-mcp: https://github.com/dongwookim-ml/slurm-mcp
- Isambard Docs: https://docs.isambard.ac.uk/
- SAM2: https://github.com/facebookresearch/sam2

### Scientific Background (NMC)
- Merryweather et al. (2021): "Operando optical tracking of single-particle ion dynamics in batteries," Nature 594, 522-528
- Xu et al. (2022): "Operando visualization of kinetically induced lithium heterogeneities in single-particle layered Ni-rich cathodes," Joule
- Heenan et al. (2025): "Demonstrating Faster Multi‐Label Grey‐Level Analysis for Crack Detection," PMC
- UChicago 2025: "Understanding Degradation in Single-Crystalline Ni-Rich Li-Ion Battery Cathodes," Chemical Reviews

### Multi-Agent Frameworks
- AutoGen: https://microsoft.github.io/autogen/
- CrewAI: https://github.com/joaomdmoura/crewai
- LangGraph: https://github.com/langchain-ai/langgraph

### Data Processing
- Dask: https://docs.dask.org/
- Xarray: https://docs.xarray.dev/
- Zarr: https://zarr-specs.readthedocs.io/

---

## Summary

You now have:

1. **Tooling** — Claude Code + Codex CLI + orchestration frameworks
2. **HPC architecture** — Agents local (x86_64), compute remote (aarch64) via SSH+MCP
3. **Step-by-step setup** — From local Python to Isambard integration
4. **Domain-specific guidance** — NMC photometry analysis techniques, high-probability research directions
5. **Reproducibility discipline** — Test oracles, synthetic validation, publication checklist

**Next step:** Follow Section 3 (Step-by-Step Setup) in order. Start a Claude Code session and ask it to help implement each step. Use this document as context (paste it into the chat or reference it).

Good luck with your research!
