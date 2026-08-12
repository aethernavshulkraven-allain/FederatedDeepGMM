# FedDeepGMM

Run federated experiments from `fedgmm/sp_decentralized_mnist_lr_example` with
the nested FedML `sp` backend. Client-local work can remain serial, use one
persistent process per GPU, or use several persistent processes on one GPU.
The coordinator always retains sampling, aggregation, optimizer history,
FedEG barriers, evaluation, checkpointing, and result generation.

## Client execution modes

Use logical CUDA device IDs in YAML. Multi-GPU execution uses one worker per
listed device:

```yaml
train_args:
  client_execution_mode: multi_gpu_processes
  enable_multiprocessing: false
  multiprocessing_num_workers: 4
  multiprocessing_gpu_ids: [0, 1, 2, 3]
```

True same-GPU multiprocessing deliberately assigns multiple spawned processes
to one logical device:

```yaml
train_args:
  client_execution_mode: multiprocessingsinglegpu
  enable_multiprocessing: false
  multiprocessingsinglegpu_num_workers: 2
  multiprocessingsinglegpu_gpu_id: 0

device_args:
  gpu_id: 0
```

Run an isolated physical GPU as follows:

```bash
cd fedgmm/sp_decentralized_mnist_lr_example
CUDA_VISIBLE_DEVICES=2 python main.py --cf <reviewed-config.yaml>
```

Here shell ID `2` selects the physical GPU; PyTorch remaps it to logical
`cuda:0`, so both YAML GPU IDs must be `0`. If `CUDA_VISIBLE_DEVICES` is
omitted, `main.py` exposes the repository default `0,1,2,3`. A shell-supplied
value is preserved. Both process modes fall back to `sp` when fewer than two
workers are selected. The legacy `enable_multiprocessing: true` selects
`multi_gpu_processes` only when `client_execution_mode` is absent.

Before GPU validation, use `nvidia-smi` to confirm the selected devices are
isolated. Deterministic methods must match SP by `torch.equal` for every `g`
and `f` checkpoint tensor, not merely by rounded metrics.

## Auxiliary regression

Auxiliary regression is disabled by default because its learned state is not
used by federated GMM updates, model selection, metrics, predictions, centralized
runs, or eICU evaluation. Disabled checkpoints retain `reg_state_dict: null` and
`state.regression: null` for schema compatibility. Legacy experiments can opt in:

```yaml
train_args:
  auxiliary_regression: true
  auxiliary_regression_epochs: 2
```


## Known full-suite errors

The latest `python -m unittest discover -s tests -p 'test_*.py'` run executed 411 tests: 405 passed and six errored for existing repository test-environment or fixture reasons, not multiprocessing behavior.

- `test_experiment_utils`: collection fails because `scripts.run_abs_smoke` is shadowed or unresolved after test-path mutation.
- `test_preflight_full_gradient`: collection fails because `scripts.preflight_full_gradient` is shadowed or unresolved.
- `test_real_image_abs_pipeline`: collection fails because `scripts.analyze_real_image_abs_tuning` is shadowed or unresolved.
- `V1RealTreeTests.test_funnel_reaches_nine_rows_three_hospitals`
- `V1RealTreeTests.test_independent_scan_matches_completion_record_claim`
- `V1RealTreeTests.test_phase_is_final_complete`

The three `V1RealTreeTests` errors originate in `scripts/report_eicu_study_a_status.py`: the expected final-split record is absent, so `final_split` is `None` before `split_sizes` is read. Keep these limitations visible until test import isolation and the eICU v1 fixture tree are repaired.
