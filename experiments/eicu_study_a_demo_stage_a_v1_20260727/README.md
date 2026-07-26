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

## Completion status

The campaign completed on 2026-07-27:

- 30 / 30 confirmatory federated runs completed;
- 45 / 45 centralized-baseline runs completed;
- 30 / 30 aggregation-ablation runs completed;
- all 105 runs recorded finite metrics and `diverged: false`; and
- strict post-run validation passed with zero blocking errors and zero warnings.

The exact generated federated YAML files, launcher ledgers, reconciliation report,
and post-run validation reports are stored beside this README. The trained
artifacts are under
`results/eicu_study_a_demo_stage_a_v1_20260727`.

The initial federated result serializer omitted eight manifest provenance fields
from `effective_config.json`. No learned parameters, checkpoints, predictions,
curves, or metric values were affected. The missing metadata was reconciled from
the frozen manifest after identity and checksum checks. Copies of all 120 original
JSON files are preserved under
`results/_failed/20260727-demo-study-a-provenance-reconciliation`.

## Demo limitations

Every scenario retains only nine rows after the real-instrument eligibility gate:
four Train, two Dev, and three Test. Only two clients occur in Dev, and no
non-Test client has the five rows needed for adjusted first-stage certification.
Accordingly, these outputs validate orchestration and artifact contracts only.
They must not be reported as Study A scientific results or used for clinical
claims.
