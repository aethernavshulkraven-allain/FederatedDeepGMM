# eICU Study A demo campaign — 2026-07-27

This is the complete 105-row Study A engineering campaign on the credential-free
eICU demo release. It is intentionally limited to pipeline verification and is
not a paper result, full-eICU result, or clinical-effect analysis.

The campaign uses the five frozen confirmatory seed pairs and all required roles:

- 30 uniform-client federated confirmation runs;
- 45 hospital-aware centralized baseline runs; and
- 30 paired sample-size aggregation-ablation runs.

Hyperparameter tuning is deliberately skipped. All federated method/scenario
combinations use the preregistered defaults in `fixed_hyperparameters.json`;
centralized methods use the defaults recorded in `freeze_record.json`. No Test
metric was used to choose a configuration. Checkpoints are still selected within
each run using equal-client validation structural MSE.

The canonical full-eICU validator rejects demo scenarios by default. This campaign
must therefore be validated with the explicit `--allow-demo` flag, which changes
only the data-scope gate and does not relax matrix, pairing, checksum, checkpoint,
metric, or artifact checks.
