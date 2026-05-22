# Source-Invariant Physical Family Audit

Decomposes source-invariant leave-source video signal into particle-region feature families.

- Rows: 172
- Cycles: 34
- Sources: 12
- Feature families: {'particle_intensity': 20, 'particle_vs_context': 10, 'particle_gradient': 10, 'norm_heterogeneity': 8, 'video_embedding': 16, 'handcrafted_particle': 48, 'all_video': 64}
- Methods: ['raw', 'source_mean_resid_2', 'source_mean_resid_4', 'source_confound_filter_0.50']

Guardrail: Physical family readouts use automatic particle-region descriptors and weak future labels under leave-source splits. They identify candidate physics families for review prioritization only; source/outcome imbalance, automatic masks, missing manual QC, and uncalibrated optical-front diffusion remain guardrails.
