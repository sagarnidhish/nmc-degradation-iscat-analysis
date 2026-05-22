# Source-Balanced Residual Temporal Specificity Audit

Rows/cycles/sources: 96 / 48 / 14
Event cycles: [60.0, 86.0, 116.0, 156.0]
Label counts: {'current_any_event': 0, 'future8': 28, 'future16': 48, 'past8': 10, 'past16': 30}

## Primary Candidate
- window 8: future AUC=0.539, past AUC=0.630, current AUC=nan, future-control delta=-0.091
- window 16: future AUC=0.637, past AUC=0.625, current AUC=nan, future-control delta=0.012

## Shift Null
- future16 source-residual dictionary_recon_error_last_minus_first: observed AUC=0.637, null p95=0.521, empirical p=0.002

## Guardrail
Temporal labels come from full particle abrupt-drop cycles but are evaluated on the 96 source-balanced ROI rows. Future-oriented scores use the future-label sign and apply the same sign to past/current controls. This tests temporal specificity, not causality or deployable warning.
