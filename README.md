# NMC Battery Degradation — Alek_Jiho iSCAT Analysis

Operando interferometric scattering (iSCAT) microscopy of NMC811 cathode particles
during galvanostatic cycling, with simultaneous GAMRY potentiostat electrochemistry.

## Scientific goals

1. Quantify per-particle optical degradation rates and correlate with cell-level capacity fade
2. Distinguish crack-like (discrete) vs gradual degradation kinetics
3. Detect spatial heterogeneity among neighbouring particles
4. Predict remaining useful life from early-cycle optical signatures
5. Identify which optical features precede dQ/dV peak shifts (electro-optical coupling)

## Dataset (Isambard: `/scratch/<account>/<username>/Alek_Jiho/`)

| Session | Size | Cycles | Notes |
|---|---|---|---|
| NMC_degradation_1_180523 | 16 GB | ~20 | C/20 formation + C/2 |
| NMC_degradation_2_240523 | 30 GB | ~20 | OCV + C/20 + C/2 |
| NMC_degradation_3_160623_Halfthedata | 674 GB | ~158 | Main degradation run; HighCOV sessions |

Each HDF5 contains: `movie` (N×1200×1920 uint16), `average_intensity`, `camera_timing`,
`potentiostat_value` (V, I, charge, time), `echem_dataframe`, `stage_position`.

**Never load the `movie` dataset directly** — up to 60 GB per file.

## Scripts

```
scripts/
  tier1_h5_inventory.py          # metadata scan all HDF5
  tier1_echem_eda.py             # capacity fade, dQ/dV heatmap
  tier1_particle_intensity_eda.py # particle traces, crack detection
  tier1_background_drift.py      # illumination drift per session
  submit_tier1.sh                # SLURM: 1 GPU, 2h

  tier2_rul_regressor.py         # RUL prediction: MLP + GBR, session-holdout CV
  tier2_spatial_heterogeneity.py # 1D CNN encoder + HDBSCAN + Moran's I
  tier2_crack_hazard.py          # discrete-time survival model
  tier2_optical_echem_coupling.py # lagged Spearman + CCA
  submit_tier2.sh                # SLURM: 1 GPU, 4h

  tier3_ts2vec_pretrain.py       # TS2Vec contrastive pretraining on traces
  tier3_multimodal_transformer.py # TFT: optical+echem -> fade acceleration forecast
  tier3_crack_precursor_detector.py # 1D CNN weak supervision + curriculum training
  submit_tier3.sh                # SLURM: 2 GPU, 10h
```

## Outputs (Isambard: `derived/`)

```
derived/
  h5_inventory.csv
  echem_per_cycle.csv
  particle_intensity_normalized.csv
  plots/
  rul/           metrics.json, feature_importance.csv
  spatial/       cluster_assignments.csv, morans_i.json
  hazard/        hazard_predictions.csv, hazard_results.json
  coupling/      lag_correlations.csv, results.json
  ts2vec/        encoder_weights.pt, trace_embeddings.npy
  transformer/   model_weights.pt, predictions.csv
  crack_detector/ model.pt, results.json
  logs/
```

## Running on Isambard

```bash
# Transfer scripts
scp -r scripts/ u6hp.aip2.isambard:/scratch/<account>/<username>/Alek_Jiho/

# Submit in order (each tier depends on previous tier's derived/ outputs)
ssh u6hp.aip2.isambard "cd /scratch/<account>/<username>/Alek_Jiho && sbatch scripts/submit_tier1.sh"
# After Tier 1 completes:
ssh u6hp.aip2.isambard "cd /scratch/<account>/<username>/Alek_Jiho && sbatch scripts/submit_tier2.sh"
# After Tier 2 completes:
ssh u6hp.aip2.isambard "cd /scratch/<account>/<username>/Alek_Jiho && sbatch scripts/submit_tier3.sh"
```

## Environment

- Isambard-AI Phase 2, GH200 (H100 96GB HBM), aarch64
- `walrus-env` conda: torch 2.5.1+cu124, h5py, scikit-image, pandas, scipy, sklearn
- **fp32 only** — no fp16/autocast (GH200 NaN issue)
- SLURM account: `<slurm-account>`

## Physics hypotheses being tested

1. **Cracking kinetics**: discrete intensity drops accelerate after cycle 80–100 (Poisson mixed model)
2. **Surface reconstruction**: monotonic drift co-evolves with dQ/dV broadening with a lag
3. **Spatial heterogeneity**: adjacent particles degrade at different rates (Moran's I)
4. **Early warning**: optical CoV in cycles 1–40 predicts fade onset (Spearman + regression)
5. **HighCOV = mechanical failure**: higher crack hazard ratio vs standard sessions (Cox model)
