# Objectives And Observations (Addendum)

Date: 2026-05-25

## Experiment Result

Context-length sweep for the source-balanced next-frame baseline.

## Verified Metrics

- `context=2`: test `MAE median=0.0075888`, `MAE p90=0.0138592`, `MSE median=1.00848e-04`
- `context=4`: test `MAE median=0.0077136`, `MAE p90=0.0148134`, `MSE median=1.06319e-04`
- `context=8`: test `MAE median=0.0087020`, `MAE p90=0.0108151`, `MSE median=1.33766e-04`

## Observation

The shortest history window is best on median error, while the longest history improves the 90th-percentile tail. That suggests the crop dynamics are not gaining much from long memory in the current representation, but there is still a subset of harder sequences where longer context helps stabilize the worst cases.

## Follow-Up

This makes the next physics-aware step more obvious:

- keep short-context next-frame prediction as the control model;
- attach physics-facing heads or auxiliary losses for phase-fraction slope, front-radius slope, and rollout-energy growth;
- compare those against the bridge audit that joins rollout residuals to phase-kinetics descriptors.

