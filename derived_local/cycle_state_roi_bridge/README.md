# Cycle State to ROI/Front Bridge

Joins cycle-level state-space coordinates to ROI/front physics-consistency and echem-shape-conditioned residual targets.

- ROI rows joined: 52
- Cycles represented: 11
- Cycle-state predictors: 9
- ROI/front targets: 13

## Strongest Associations

- row cycle_state_pc2 vs physics_consistency_score: rho=0.702, permutation p=0.0005, n=52
- row cycle_state_pc2 vs kinetic_transition_score: rho=0.695, permutation p=0.0005, n=52
- row cycle_state_pc2 vs precursor_context_score: rho=0.687, permutation p=0.0005, n=47
- row cycle_state_pc3 vs kinetic_transition_score: rho=0.598, permutation p=0.0005, n=52
- row cycle_state_cluster vs kinetic_transition_score: rho=0.595, permutation p=0.0005, n=52

## Stricter Checks

- cycle-collapsed cycle_state_pc2 vs mode_taxonomy_score: rho=0.855, permutation p=0.002, n=11
- cycle-collapsed cycle_state_pc2 vs physics_consistency_score: rho=0.836, permutation p=0.002, n=11
- cycle-collapsed cycle_state_pc2 vs kinetic_transition_score: rho=0.764, permutation p=0.0095, n=11
- cycle-collapsed cycle_state_pc3 vs mode_taxonomy_score: rho=0.709, permutation p=0.021, n=11
- cycle-collapsed cycle_state_pc3 vs kinetic_transition_score: rho=0.682, permutation p=0.0265, n=11

## Interpretation

Cycle-state to ROI/front bridge joins cycle-level state coordinates to selected automatic ROI rows. Row-level associations are not independent within cycle; reference-centered and cycle-collapsed tests are the stricter evidence. This does not create manual QC labels or calibrated diffusion claims.
