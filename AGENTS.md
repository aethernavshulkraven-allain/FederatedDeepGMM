# AGENTS.md

Guidance for Codex agents working in `/home/arnav22103/FederatedDeepGMM`.

## Environment

- Use the `fedgmm` Conda environment.
- Preferred Python executable:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python
```

- Prefer `rg` / `rg --files` for searching.
- Inspect files before editing them.
- Do not use destructive git commands. The worktree contains local, uncommitted experiment work.

## Result Safety

- Preserve `results/_golden`.
- Archive failed or superseded runs under `results/_failed/<timestamp>/` rather than overwriting or deleting them.
- Treat existing result folders as scientific artifacts unless the user explicitly says otherwise.
- Do not claim paper-match status without checking `paper_ctxt.md`, `experiments/reproduction_targets.csv`, and the current manifest/result metadata.

## Scientific Rules

- Hyperparameter tuning must be validation-driven.
- Do not select a config using Test MSE.
- Report `test_mse_at_best_validation` only after the validation-selected config is fixed.
- `diverged: true` means numerical failure such as NaN/inf model parameters or metrics, not merely worse MSE.
- Current known paper-alignment caveat: synthetic data certification found that current author-code data is reproducible, but not verified as paper-aligned.

## Current FedOGDA-S Tuning Pilot

Purpose: tune `fedogda_s` critic multiplier and weight decay for stochastic low-dimensional runs, focusing on stability/oscillation blindspots rather than assuming FedOGDA must beat FedGDA everywhere.

Pilot scope:

```text
method: fedogda_s
datasets: abs, step, linear
alpha: 0.5
seeds: 0, 1, 2
critic_multiplier: 2, 5, 10, 20
weight_decay: 0.001, 0.01, 0.03, 0.1
total runs: 144
```

Pilot files:

```text
manifest: experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/manifest.csv
manifest json: experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/manifest.json
setup summary: experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/setup_summary.json
generated configs: experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/generated_configs/
output root: results/rerun_protocol_v1_tuning/fedogda_s_pilot_alpha0p5
```

The full dry-run passed with `launchable: 144`, `shown: 144`, and `skipped_unlaunchable: 0`.

Launch command:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_manifest.py --manifest experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/manifest.csv --config-dir experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/generated_configs --output-root results/rerun_protocol_v1_tuning/fedogda_s_pilot_alpha0p5 --gpu-ids 0,1 --max-parallel 1 --resume-skip-completed --results-json experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/run_results.json
```

Important: pass `--output-root` explicitly when using `scripts/run_manifest.py`; do not rely on the manifest `output_root` column to control launcher output.

Selection rule for this pilot:

- For each dataset at alpha `0.5`, select `critic_multiplier + weight_decay` by lowest mean `best_validation_mse` across seeds.
- Tie-break by mean `last_50_val_mse_std`.
- Tie-break again by mean `final_vs_best_validation_gap`.
- After selection, report `test_mse_at_best_validation`.

## Useful Context Files

- `todo-arnav.md`
- `handoff-ctxt-17jun.md`
- `paper_ctxt.md`
- `experiments/reproduction_targets_README.md`
- `experiments/reproduction_targets.csv`
