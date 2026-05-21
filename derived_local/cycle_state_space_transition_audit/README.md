# Cycle State-Space Transition Audit

Cycle-level degradation-state audit using four-particle photometry traces joined to within-cycle echem-shape descriptors.

- Cycles scored: 89
- Features used: 107
- Joined echem-shape cycles: 81
- Chosen state clusters: 4 (silhouette 0.634)
- Degradation axis oriented to: mean_abs_delta_prev (rho 0.32562542378621273)
- Future-drop classifier mean AUC: 0.781; balanced accuracy 0.731

## Top Future-Drop Associations

- cycle_state_pc2: positive-negative median 0.730, permutation p=0.0162
- mean_abs_delta_prev: positive-negative median -0.012, permutation p=0.08
- state_step_norm: positive-negative median -0.694, permutation p=0.101
- degradation_state_axis: positive-negative median -2.269, permutation p=0.184
- cycle_state_pc1: positive-negative median 2.269, permutation p=0.199
- capacity_mAh: positive-negative median -0.014, permutation p=0.205

## Top State Clusters

- state 1: n=58, cycles 2-158, future8 rate=0.25862068965517243
- state 2: n=1, cycles 150-150, future8 rate=1.0
- state 0: n=29, cycles 112-149, future8 rate=0.13793103448275862
- state 3: n=1, cycles 126-126, future8 rate=0.0

## Interpretation

Cycle state-space clusters use four-particle trace summaries and echem-shape descriptors at cycle resolution. They are degradation-state hypotheses and early-warning covariates, not localized ROI/front validation or calibrated diffusion measurements.
