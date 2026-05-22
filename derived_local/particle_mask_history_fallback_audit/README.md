# Particle Mask History/Fallback Audit

Audits whether framewise particle masks are stable or would need history-based fallback under blur/drift artifacts.

- Input rows: 128
- Processed rows: 128
- Median fallback frame fraction: 0.292
- Median history IoU: 0.865

## Guardrail

History-based masks are an automatic robustness audit for cropped particle ROIs. Fallback flags indicate possible blur/drift mask instability; they are not manual segmentation labels or physical phase-boundary measurements.
