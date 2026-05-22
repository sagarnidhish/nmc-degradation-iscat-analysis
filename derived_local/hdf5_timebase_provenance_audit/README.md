# HDF5 Timebase Provenance Audit

Source-level and ROI-level audit of camera timing used to express front and apparent-diffusion proxies per second.

- q70 ROI rows: 72
- Sources: 9
- Strict timebase sources: 5
- Pause-heavy sources: 4
- ROI/HDF5 elapsed-aligned rows: 72 / 72
- Median ROI/HDF5 elapsed ratio: 1.001587
- Max source dt max/median ratio: 13.783
- Timebase status: mixed_timebase_pause_heavy_sources_present

## Interpretation

ROI elapsed times align closely with HDF5 median frame spacing, but several source files contain large camera-timing gaps relative to their 10 s median frame interval. Per-second apparent front/diffusion values are usable as median-timebase proxies, while calibrated diffusion claims should either exclude pause-heavy sources or model dropped/paused acquisition intervals explicitly.

## Outputs

- `hdf5_timebase_source_summary.csv`
- `hdf5_timebase_roi_q70_table.csv`
- `hdf5_timebase_scenario_table.csv`
- `hdf5_timebase_correlations.csv`
- `hdf5_timebase_provenance_summary.json`
