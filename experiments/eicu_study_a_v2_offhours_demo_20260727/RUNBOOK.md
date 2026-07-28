# Study A v2 demo runbook

Run all commands from `/home/arnav22103/FederatedDeepGMM` with:

```bash
P=/home/arnav22103/miniconda3/envs/fedgmm/bin/python
CAMPAIGN=experiments/eicu_study_a_v2_offhours_demo_20260727
SCENARIOS=fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth_offhours_v2_demo
RESULTS=results/eicu_study_a_v2_offhours_demo
```

The current machine has no visible CUDA device, so the frozen demo manifests
set `using_gpu=false`. `--gpu-ids 0` is only a launcher slot label in these
commands. Start with `--max-parallel 2`; raise to 4 only if memory and load
remain comfortable.

## Rebuild the frozen demo inputs

```bash
$P scripts/prepare_eicu_study_a_v2_cohort.py \
  --eicu-root physionet.org/files/eicu-crd-demo/2.0.1 \
  --out "$CAMPAIGN" \
  --scenario-scope demo

$P scripts/prepare_eicu_study_a_v2_scenarios.py \
  --cohort "$CAMPAIGN/cohort.csv" \
  --out "$SCENARIOS" \
  --scenario-seeds 11,101,102,103,104,105 \
  --scenario-scope demo
```

Do not rerun these commands after tuning has begun unless deliberately
creating a new version. The current scenarios are checksummed artifacts.

## Tuning

The tuning manifest and generated dry-run configs already exist. To recreate:

```bash
$P scripts/prepare_eicu_study_a_v2_manifest.py \
  --stage tuning \
  --scenario-dir "$SCENARIOS" \
  --output-root "$RESULTS" \
  --out "$CAMPAIGN/tuning_manifest.csv"

$P scripts/run_manifest.py \
  --manifest "$CAMPAIGN/tuning_manifest.csv" \
  --config-dir "$CAMPAIGN/generated_configs_tuning" \
  --output-root "$RESULTS" \
  --gpu-ids 0 \
  --max-parallel 2 \
  --resume-skip-completed \
  --keep-going \
  --results-json "$CAMPAIGN/tuning_run_results.json"
```

Select from Validation only:

```bash
$P scripts/select_eicu_study_a_v2_tuning.py \
  --manifest "$CAMPAIGN/tuning_manifest.json" \
  --out "$CAMPAIGN/selected_hyperparameters.json"
```

Do not proceed if any of the six `(g0, method)` selections is missing.

## Freeze and validate the 105-row final campaign

```bash
$P scripts/prepare_eicu_study_a_v2_manifest.py \
  --stage final \
  --scenario-dir "$SCENARIOS" \
  --output-root "$RESULTS" \
  --selected-hyperparameters "$CAMPAIGN/selected_hyperparameters.json" \
  --out "$CAMPAIGN/final_manifest.csv"

$P scripts/validate_eicu_study_a_campaign.py \
  --manifest "$CAMPAIGN/final_manifest.csv" \
  --contract experiments/eicu_study_a_v2_offhours/validation_contract_demo.json \
  --phase prelaunch \
  --allow-demo \
  --out "$CAMPAIGN/prelaunch_validation"
```

Expected final matrix: 30 confirmatory federated, 45 centralized, 30
aggregation ablations, 105 total. The validator must report zero blocking
errors and zero warnings.

## Final training

Federated rows:

```bash
$P scripts/run_manifest.py \
  --manifest "$CAMPAIGN/final_manifest.csv" \
  --config-dir "$CAMPAIGN/generated_configs_final" \
  --output-root "$RESULTS" \
  --gpu-ids 0 \
  --max-parallel 2 \
  --resume-skip-completed \
  --keep-going \
  --results-json "$CAMPAIGN/final_federated_run_results.json"
```

Centralized rows:

```bash
$P scripts/run_eicu_study_a_v2_centralized.py \
  --manifest "$CAMPAIGN/final_manifest.csv" \
  --output-root "$RESULTS" \
  --results-json "$CAMPAIGN/final_centralized_run_results.json"
```

Never pass an overwrite flag for a completed run unless its original directory
has first been archived under `results/_failed/<timestamp>/`.

## Post-run validation and effect metrics

```bash
$P scripts/validate_eicu_study_a_campaign.py \
  --manifest "$CAMPAIGN/final_manifest.csv" \
  --contract experiments/eicu_study_a_v2_offhours/validation_contract_demo.json \
  --results-root "$RESULTS" \
  --phase postrun \
  --allow-demo \
  --out "$CAMPAIGN/postrun_validation"

$P scripts/materialize_eicu_study_a_v2_effect_metrics.py \
  --manifest "$CAMPAIGN/final_manifest.csv" \
  --results-root "$RESULTS" \
  --keep-going \
  --ledger "$CAMPAIGN/effect_metric_materialization.json"
```

The effect evaluator uses the frozen continuous-treatment contrast
`g(1,W)-g(0,W)` and does not change checkpoint selection.

## Full eICU release

Before building a full-eICU campaign, run:

```bash
$P scripts/preflight_eicu_release.py \
  --eicu-root /path/to/eicu-crd/2.0 \
  --out experiments/eicu_full_data_preflight/audits/study-a-v2 \
  --require-full \
  --count-patient-rows \
  --checksum
```

Then build into new full-release cohort/scenario/result directories and derive
a validation contract whose dataset matches that scenario directory. Do not
overwrite this demo setup or relabel demo outputs as full-eICU evidence.
