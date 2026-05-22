# Calibration Provenance Evidence Audit

Metadata/text provenance audit for the assumed 96 nm/px optical scale and timing context.

- Files inventoried: 61
- HDF5 files scanned: 33
- Raw HDF5 explicit spatial-scale statements: 0
- Explicit or inferred near-96 nm/px statements: 21
- Contradictory scale statements: 0
- Highest evidence class: slide_text
- Provenance status: slide_or_project_text_supported_but_raw_metadata_blocked

## Interpretation

The 96 nm/px assumption is supported only by slide/project-text evidence and by FoV divided by raw movie dimensions being approximately compatible with that scale. No raw HDF5 attribute or microscope metadata statement was found, so calibrated diffusion constants remain blocked; use apparent optical-front units.

## Outputs

- `calibration_provenance_file_inventory.csv`
- `calibration_provenance_evidence_statements.csv`
- `calibration_provenance_summary.json`
