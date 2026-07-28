# eICU Study A demo campaign — 2026-07-27

This is the complete 105-row Study A engineering campaign on the credential-free
eICU demo release. It is intentionally limited to pipeline verification and is
not a paper result, full-eICU result, or clinical-effect analysis.

**Status: archived.** `superseded_by`: Study A v2,
`experiments/eicu_study_a_v2_offhours/protocol_v2.md`. v2 supersedes this
campaign's cohort and instrument for new runs; it does not alter or invalidate
the archived pipeline-validation artifacts recorded here. This campaign cannot
rank FedGDA against FedOGDA — see "Why this campaign cannot rank methods"
below.

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

## Why this campaign cannot rank methods

Beyond the cohort collapse above, two further reasons independently prevent
this campaign from ranking FedGDA against FedOGDA at any sample size:

- **No hyperparameter tuning was performed.** Every entry in
  `fixed_hyperparameters.json` carries `learning_rate_status:
  "preregistered_fixed_no_tuning"`, and all six method/scenario combinations
  are pinned at `learning_rate: 0.001` (`server_learning_rate: 1.0`). A
  comparison of untuned methods run at identical, arbitrarily chosen learning
  rates is not a method ranking — a different fixed rate could reverse it.
- **Only 30 of the 105 runs were federated confirmations.** Per
  `completion_record.json`'s `execution` block, the 105 completed runs split
  into `confirmatory_completed: 30`, `centralized_baseline_completed: 45`, and
  `aggregation_ablation_completed: 30`. The centralized-baseline and
  aggregation-ablation runs exercise other parts of the pipeline; they are not
  additional FedGDA-vs-FedOGDA comparisons, so the federated method comparison
  this campaign can speak to rests on 30 runs, not 105.

Either limitation alone would rule out a method ranking; together with the
cohort collapse, this campaign is pipeline verification, not a paper result.
