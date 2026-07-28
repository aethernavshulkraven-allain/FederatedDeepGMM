# Study A v2 off-hours demo setup — 27 July 2026

This directory is the materialized demo-only setup for
`experiments/eicu_study_a_v2_offhours/protocol_v2.md`.

Current materialized facts:

- 2,031 admissions;
- 179 hospital clients;
- fixed split sizes 1,420 Train / 306 Dev / 305 Test;
- off-hours rate 0.55;
- 18 certified scenario artifacts (three structural functions for tuning seed
  11 and final seeds 101–105);
- 47-dimensional response and critic inputs;
- 36 launchable tuning rows;
- one two-round CPU federated canary completed without divergence;
- federated batch size 4 produced 2–5 local batches for every client (zero
  disguised single/full-client batches).

The scenario artifacts are under
`fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth_offhours_v2_demo`.
The canary is under
`results/eicu_study_a_v2_offhours_canary_20260727`.
The centralized two-iteration canary is under
`results/eicu_study_a_v2_offhours_central_canary_20260727`.
The superseded batch-32 canary is archived under
`results/_failed/20260727-study-a-v2-batch32-canary`.

The final 105-row manifest does not exist yet by design. Run the 36 tuning
rows, select hyperparameters from Validation only, then materialize the final
manifest. Do not copy v1 hyperparameters or select using Test MSE.

This is an eICU demo engineering artifact and cannot support a full-eICU or
clinical claim.
