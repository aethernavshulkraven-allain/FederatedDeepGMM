# Single-GPU Multiprocessing Tutorial

This repository can run independent federated client updates concurrently in separate processes on one GPU. Sampling, aggregation, server updates, evaluation, and checkpointing remain in the coordinator, so multiprocessing changes execution only—not the federated algorithm.

## Prerequisites

Run commands from the repository root with the project’s Python environment active. Confirm that PyTorch detects CUDA and that the intended GPU is idle:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

Use an isolated GPU. An out-of-memory error on a GPU occupied by another process is not a valid worker-capacity result.

## Reviewed Manifest

Use `experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv`. It contains short ABS validation rows for FedGDA, FedOGDA, FedEG, and FedZO-EG in deterministic and stochastic forms. Every row selects single-GPU multiprocessing with four workers, logical GPU `0`, and auxiliary regression disabled.

The CSV manifest is the user-facing configuration. `scripts/run_manifest.py` generates YAML internally and invokes `main.py`; users should not create or launch those generated files directly. This smoke manifest verifies execution and is not a substitute for a study's approved hyperparameter manifest.

## Preview and Run

Preview one experiment before using GPU time:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_manifest.py \
  --manifest experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv \
  --config-dir /tmp/feddeepgmm_single_gpu_configs \
  --output-root results/multiprocessing_single_gpu_smoke \
  --gpu-ids 0 \
  --max-parallel 1 \
  --only method=fedgda_s \
  --dry-run
```

After reviewing the preview, run it:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_manifest.py \
  --manifest experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv \
  --config-dir /tmp/feddeepgmm_single_gpu_configs \
  --output-root results/multiprocessing_single_gpu_smoke \
  --gpu-ids 0 \
  --max-parallel 1 \
  --only method=fedgda_s \
  --resume-skip-completed \
  --results-json experiments/multiprocessing_single_gpu/smoke_launcher_results.json
```

Physical GPU `2` is remapped to logical GPU `0`, so the manifest and `--gpu-ids` both use `0`. Change the method filter to `fedgda_d`, `fedogda_s`, `fedogda_d`, `fed_eg_s`, `fed_eg_d`, `fed_zo_eg_s`, or `fed_zo_eg_d`. Omit `--only` to run all rows sequentially.

`--max-parallel` controls complete experiments launched concurrently. `multiprocessingsinglegpu_num_workers` controls client processes inside one experiment. Keep `--max-parallel 1` when validating one physical GPU.

## Where the Multiprocessing Code Lives

The two modes share one implementation path:

- `fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/multiprocess_client.py` contains persistent spawned workers, CPU/GPU tensor transfer, serialized epoch replay, ordered result collection, worker-error propagation, CUDA cleanup, and executor shutdown. `MultiprocessClientExecutor` assigns one logical GPU per worker for multi-GPU mode. `SingleGPUMultiprocessClientExecutor` reuses it while assigning every worker the same logical GPU.
- `fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py` selects `sp`, `multi_gpu_processes`, or `multiprocessingsinglegpu`; validates GPU IDs and worker counts; materializes client batches in coordinator order; submits primary and FedEG correction tasks; restores results to the coordinator device; and preserves aggregation order.
- `fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/client.py` exposes the standard local `train` path and the `train_zo` correction path used by FedZO-EG workers. The corresponding zeroth-order trainer implementation is in `fedgmm/sp_decentralized_mnist_lr_example/fedml/ml/trainer/my_model_trainer_classification.py`.

Configuration and launch integration are in `scripts/run_manifest.py`, `scripts/run_abs_smoke.py`, and `fedgmm/sp_decentralized_mnist_lr_example/fedml/config/simulation_sp/fedml_config.yaml`. The reviewed user-facing example is `experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv`.

Regression coverage is in `tests/test_multiprocess_client.py` for routing, GPU assignment, worker limits, and ordered results; `tests/test_multiprocessing_failures.py` for fallback, validation, generated configuration, and failure propagation; and `tests/test_fedeg_variants.py` for FedEG/FedZO-EG two-phase behavior. Deterministic equivalence also relies on seed/model initialization handling in `fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py` and `fedgmm/sp_decentralized_mnist_lr_example/fedml/model/model_hub.py`.

## Choose a Worker Count

Start with `2` or `4`, then profile the real workload. On the validated CIFAR10-X test with 10 sampled clients, one round, two local epochs, and batch size 256, worker counts `2`, `4`, `6`, `8`, and `10` all completed on an RTX A6000. Four workers was fastest; more workers increased memory and overhead. Capacity depends on the model, batch size, and GPU. To change the count, make a reviewed copy of the manifest and edit `multiprocessingsinglegpu_num_workers`; do not alter a recorded study manifest in place.

## Deterministic Full-Batch Recommendation

Use SP for deterministic full-batch FEMNIST-X, FEMNIST-XZ, and CIFAR10-X runs on one GPU. Matched 20-round FedGDA-D benchmarks with 10/10 clients and three local steps found both two-worker and four-worker same-GPU multiprocessing slower than SP:

| Dataset | SP | MP2 | MP4 |
|---|---:|---:|---:|
| FEMNIST-X | 3:10 | 3:20 | 3:20 |
| FEMNIST-XZ | 6:15 | 6:40 | 6:45 |
| CIFAR10-X | 5:50 | 6:15 | 6:20 |

All MP checkpoints, predictions, and numerical metrics matched their SP baselines exactly; every `g` and `f` tensor passed `torch.equal`. The slowdown comes from process, serialization, and memory overhead while full-batch training already saturates the GPU. Use separate SP experiments on separate GPUs for deterministic campaign throughput. Reproducible manifests are under `experiments/multiprocessing_benchmark/`.

## Outputs and Verification

Results are written beneath the configured `common_args.output_dir`, grouped by dataset, method, seed, and run ID. Check:

- `metrics.json` for `run_status: "completed"` and `nonfinite_first_round: null`.
- `checkpoints/final.pt` and selection checkpoints for saved `g`/`f` state.
- `predictions.npz` for final and best-validation predictions.

For deterministic FedGDA, FedOGDA, or FedEG validation, compare the SP and multiprocessing runs with `torch.equal` for every `g_state_dict` and `f_state_dict` tensor. Matching rounded metrics alone is insufficient.

## Troubleshooting

- **Invalid GPU ID:** after `CUDA_VISIBLE_DEVICES=<physical-id>`, use logical GPU `0` in the manifest and `--gpu-ids 0`.
- **Unexpected serial execution:** select at least two workers and verify CUDA is available.
- **OOM:** check `nvidia-smi` for unrelated processes, then reduce workers or batch size.
- **Worker failure:** inspect the traceback; the coordinator reports the worker and client task that failed and shuts down the pool.
