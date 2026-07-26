# eICU Study A campaign validation

`scripts/validate_eicu_study_a_campaign.py` is a standalone, read-only
validator for Study A campaign manifests and result artifacts. It does not
import the scenario generator, trainer, centralized-baseline runner, or
campaign orchestrator. Campaign expectations and JSON field aliases are
declared in `default_contract.json`.

Study A is an extension study with no published numeric target. A passing
report is a protocol/completeness result, not a claim that any paper number was
reproduced.

## Usage

Prelaunch validation:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python \
  scripts/validate_eicu_study_a_campaign.py \
  --manifest experiments/eicu_study_a/manifest.csv \
  --contract experiments/eicu_study_a_validation/default_contract.json \
  --config-dir experiments/eicu_study_a/generated_configs \
  --scenario-root experiments/eicu_study_a/scenarios \
  --results-root results/eicu_study_a \
  --phase prelaunch \
  --out experiments/eicu_study_a/validation/prelaunch
```

Postrun validation uses the same inputs with `--phase postrun`. `--out` is
optional. When it is omitted, the validator writes nothing. When present, the
only writes are:

- `eicu_study_a_validation.json`
- `eicu_study_a_validation.md`

The command exits zero only when `launchable` (prelaunch) or `reportable`
(postrun) is true. Protocol violations are blocking errors. Advisory findings
are kept in the separate `warnings` array.

## Default fixed matrix

| Role | g0 | Seeds | Methods | Aggregation | Rows |
|---|---|---|---|---|---:|
| `confirmatory` | linear, interaction, mlp | 0–4 | fedgda_s, fedogda_s | uniform_clients | 30 |
| `centralized_baseline` | linear, interaction, mlp | 0–4 | gda, sgda, oadam | none | 45 |
| `aggregation_ablation` | linear, interaction, mlp | 0–4 | fedgda_s, fedogda_s | sample_size | 30 |

The fixed final/ablation total is 105 rows. `tuning` and `smoke` are declared
optional, variable-size roles. They still inherit validation-only selection,
equal-client primary selection, no Test-MSE selection, and role-specific
aggregation/label restrictions.

## Canonical manifest schema

The default contract maps canonical names to CSV columns under
`manifest.fields`. The integrated campaign may rename columns by changing
those mappings; validator code need not change.

Required canonical fields are:

```text
run_id, role, dataset, scenario, g0, method, seed,
aggregation_weighting, objective_mode, alignment_label,
primary_selection_metric, test_mse_used_for_selection, selection_source,
scenario_checksum, scenario_scope, study_claim, config_path,
scenario_metadata_path, result_path, input_dim, instrument_dim
```

`output_root` is optional in the CSV. The declared result root is resolved in
this order:

1. `--results-root`;
2. the row's `output_root`;
3. `paths.results_root` in the contract.

`result_path` is interpreted relative to that root unless it is absolute. The
resolved path must remain inside the root and must be unique.

Paths in `config_path` and `scenario_metadata_path` are interpreted relative
to their matching CLI root when supplied, otherwise relative to the manifest
directory.

## Config and scenario metadata assumptions

JSON lookup aliases are declared under `json_paths`. The first matching path
is used, so flat and selected nested schemas are both supported.

Each scenario metadata JSON object must supply:

```text
eligible_client_provenance
g0
seed
dimensions.input_dim
dimensions.instrument_dim
dimensions.outcome_dim
scenario_checksum
is_demo
scenario_scope
```

The provenance object is required and intentionally opaque to the validator;
the campaign owns its internal client-ID/count/hash schema. If metadata also
provides `artifact_path`, the validator recomputes the artifact SHA-256 and
compares it with `scenario_checksum`. Without `artifact_path`, it still checks
that the same declared checksum appears in the manifest, config, metadata,
paired methods, and paired roles.

Fixed full-eICU roles require `scenario_scope=full_eicu` and `is_demo=false`.
Confirmatory configs are scanned recursively for contract-declared unresolved
tuning placeholder patterns.

Aggregation ablations are matched to confirmatory rows by g0, seed, and
method. Every common manifest column and every flattened config field must be
identical except the paths explicitly listed in
`manifest.ablation_allowed_differences` and
`manifest.config_ablation_allowed_differences`. This is how the contract
enforces a weighting-only ablation without coupling the validator to a
generator.

## Result schema assumptions

Each completed result directory must contain:

```text
effective_config.json
metrics.json
mse_by_round.csv
predictions.npz
```

It must also contain one best-checkpoint name and one final-checkpoint name
from the alternative groups declared in `postrun.required_file_alternatives`.
Confirmatory runs additionally require one declared per-client evaluation
artifact.

The default `metrics.json` paths require:

- run/method/seed/scenario provenance;
- equal-client validation selection metric and validation-only source;
- identical `best_validation_round` and `selected_round`;
- an attestation that Test MSE was computed/reported after selection;
- primary equal-client validation and Test MSE;
- secondary sample-weighted validation and Test MSE;
- `test_mse_at_best_validation`;
- `diverged` and non-finite evidence;
- primary/non-primary marking and the alignment label.

For finite, non-diverged runs these metrics must be numeric and finite.
`mse_by_round.csv` must expose the contract-declared round and equal-client
validation columns; its finite minimum must agree with the selected round and
reported best validation MSE.

`diverged=true` is accepted only when `metrics.json` contains an actually
non-finite numeric value or explicitly records
`divergence_evidence.nonfinite_parameters=true`. A poor but finite MSE is not
divergence. A `nonfinite_metrics=true` assertion is checked against the numeric
contents of `metrics.json`.

The validator checks only NPZ/checkpoint existence. It does not deserialize
model checkpoints or prediction arrays, keeping the validator standard-library
only and independent of training frameworks.

## Contract reconciliation checklist

Before integrating the eventual campaign, reconcile these declarative choices
with its emitted schema:

- CSV column names and whether `result_path` is root-relative;
- exact role, alignment-label, scenario-scope, objective, and method strings;
- whether configs are flat or nested, and their JSON aliases;
- scenario metadata dimension/checksum paths and optional artifact path;
- checkpoint and per-client artifact filenames;
- metrics JSON paths, especially selection timing, primary/secondary metrics,
  divergence evidence, and non-primary ablation marking;
- `mse_by_round.csv` column names;
- all legitimate weighting-only ablation differences.

Change the contract for intentional schema differences. Do not weaken
validation in the Python script to accommodate accidental protocol drift.
