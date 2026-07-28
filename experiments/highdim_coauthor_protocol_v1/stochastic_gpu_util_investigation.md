# Stochastic GPU Utilization Investigation

Generated: `2026-07-18 22:28:51`.

Scope: high-dimensional stochastic final-run path for `FedGDA-S` and `FedOGDA-S`. This report is validation-safe: it summarizes runtime/profiling artifacts only and does not use Test MSE for hyperparameter selection.

## Current Production Timing

Planned stochastic final runs from manifests: `180`.
Artifact status: completed `17`, partial/running `1`, pending `162`.
Completed 1500-round runtime seconds: min `1749.6`, median `1905.7`, max `2789.3`.

| alpha | dataset | method | completed | partial/running | pending |
|---:|---|---|---:|---:|---:|
| 0.1 | cifar10_x | fedgda_s | 0 | 0 | 5 |
| 0.1 | cifar10_x | fedogda_s | 0 | 0 | 5 |
| 0.1 | cifar10_xz | fedgda_s | 0 | 0 | 5 |
| 0.1 | cifar10_xz | fedogda_s | 0 | 0 | 5 |
| 0.1 | cifar10_z | fedgda_s | 0 | 0 | 5 |
| 0.1 | cifar10_z | fedogda_s | 0 | 0 | 5 |
| 0.1 | femnist_x | fedgda_s | 0 | 0 | 5 |
| 0.1 | femnist_x | fedogda_s | 0 | 0 | 5 |
| 0.1 | femnist_xz | fedgda_s | 0 | 0 | 5 |
| 0.1 | femnist_xz | fedogda_s | 0 | 0 | 5 |
| 0.1 | femnist_z | fedgda_s | 0 | 0 | 5 |
| 0.1 | femnist_z | fedogda_s | 0 | 0 | 5 |
| 0.5 | cifar10_x | fedgda_s | 5 | 0 | 0 |
| 0.5 | cifar10_x | fedogda_s | 3 | 0 | 2 |
| 0.5 | cifar10_xz | fedgda_s | 0 | 0 | 5 |
| 0.5 | cifar10_xz | fedogda_s | 0 | 0 | 5 |
| 0.5 | cifar10_z | fedgda_s | 0 | 0 | 5 |
| 0.5 | cifar10_z | fedogda_s | 0 | 0 | 5 |
| 0.5 | femnist_x | fedgda_s | 0 | 0 | 5 |
| 0.5 | femnist_x | fedogda_s | 0 | 0 | 5 |
| 0.5 | femnist_xz | fedgda_s | 0 | 0 | 5 |
| 0.5 | femnist_xz | fedogda_s | 0 | 0 | 5 |
| 0.5 | femnist_z | fedgda_s | 0 | 0 | 5 |
| 0.5 | femnist_z | fedogda_s | 0 | 0 | 5 |
| 1 | cifar10_x | fedgda_s | 5 | 0 | 0 |
| 1 | cifar10_x | fedogda_s | 4 | 1 | 0 |
| 1 | cifar10_xz | fedgda_s | 0 | 0 | 5 |
| 1 | cifar10_xz | fedogda_s | 0 | 0 | 5 |
| 1 | cifar10_z | fedgda_s | 0 | 0 | 5 |
| 1 | cifar10_z | fedogda_s | 0 | 0 | 5 |
| 1 | femnist_x | fedgda_s | 0 | 0 | 5 |
| 1 | femnist_x | fedogda_s | 0 | 0 | 5 |
| 1 | femnist_xz | fedgda_s | 0 | 0 | 5 |
| 1 | femnist_xz | fedogda_s | 0 | 0 | 5 |
| 1 | femnist_z | fedgda_s | 0 | 0 | 5 |
| 1 | femnist_z | fedogda_s | 0 | 0 | 5 |

## Profiling Matrix

Profiling summaries found: `8`.

### Runtime, GPU, CPU

| phase | dataset | method | rounds | wall min | wall sec/round | loop sec/round | wall rounds/sec | avg GPU | max GPU | CPU util | max mem GB | explained |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auxreg_skip50 | cifar10_x | fedgda_s | 50 | 3.4 | 4.106 | 0.967 | 0.2436 | 44.9% | 100.0% | 109.4% | 8.5 | 100.0% |
| baseline50 | cifar10_x | fedgda_s | 50 | 3.4 | 4.075 | 1.874 | 0.2454 | 37.5% | 100.0% | 905.5% | 17.4 | 100.0% |
| baseline50 | cifar10_x | fedogda_s | 50 | 3.5 | 4.155 | 1.937 | 0.2407 | 35.8% | 100.0% | 883.5% | 18.5 | 100.0% |
| baseline50 | cifar10_xz | fedgda_s | 50 | 5.6 | 6.691 | 2.601 | 0.1494 | 39.9% | 100.0% | 588.1% | 21.3 | 100.0% |
| baseline50 | cifar10_xz | fedogda_s | 50 | 5.7 | 6.797 | 2.627 | 0.1471 | 39.5% | 100.0% | 583.1% | 23.6 | 100.0% |
| baseline50 | cifar10_z | fedgda_s | 50 | 2.8 | 3.371 | 1.169 | 0.2967 | 40.5% | 100.0% | 100.5% | 22.1 | 100.0% |
| baseline50 | cifar10_z | fedogda_s | 50 | 2.8 | 3.347 | 1.133 | 0.2988 | 41.4% | 100.0% | 101.6% | 22.9 | 100.0% |
| optimized5_smoke | cifar10_x | fedgda_s | 5 | 0.1 | 0.928 | 0.559 | 1.0775 | 34.1% | 99.0% | 205.8% | 20.9 | 100.0% |

### Phase Breakdown

| phase | dataset | method | setup/model selection | round loop | GMM train | aux reg train | eval | state collect | reg aggregate | CSV write |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| auxreg_skip50 | cifar10_x | fedgda_s | 154.6s (75.3%) | 48.3s (23.5%) | 9.0s (4.4%) | NA | 19.9s (9.7%) | 0.4s (0.2%) | NA | 0.0s (0.0%) |
| baseline50 | cifar10_x | fedgda_s | 108.0s (53.0%) | 93.7s (46.0%) | 10.0s (4.9%) | 31.7s (15.6%) | 17.5s (8.6%) | 8.2s (4.0%) | 6.8s (3.3%) | 0.0s (0.0%) |
| baseline50 | cifar10_x | fedogda_s | 108.9s (52.4%) | 96.9s (46.6%) | 10.2s (4.9%) | 31.7s (15.2%) | 19.1s (9.2%) | 8.0s (3.8%) | 7.6s (3.7%) | 0.0s (0.0%) |
| baseline50 | cifar10_xz | fedgda_s | 202.0s (60.4%) | 130.0s (38.9%) | 12.8s (3.8%) | 30.7s (9.2%) | 31.5s (9.4%) | 8.6s (2.6%) | 8.0s (2.4%) | 0.0s (0.0%) |
| baseline50 | cifar10_xz | fedogda_s | 205.8s (60.6%) | 131.4s (38.7%) | 14.2s (4.2%) | 33.2s (9.8%) | 31.3s (9.2%) | 8.4s (2.5%) | 7.7s (2.3%) | 0.0s (0.0%) |
| baseline50 | cifar10_z | fedgda_s | 108.8s (64.6%) | 58.5s (34.7%) | 9.9s (5.9%) | 10.2s (6.0%) | 19.3s (11.4%) | 0.4s (0.3%) | 0.0s (0.0%) | 0.0s (0.0%) |
| baseline50 | cifar10_z | fedogda_s | 109.2s (65.2%) | 56.6s (33.8%) | 9.3s (5.6%) | 9.0s (5.4%) | 19.4s (11.6%) | 0.4s (0.2%) | 0.0s (0.0%) | 0.0s (0.0%) |
| optimized5_smoke | cifar10_x | fedgda_s | NA | 2.8s (60.2%) | 1.1s (23.4%) | NA | 1.6s (33.4%) | 0.0s (0.0%) | NA | 0.0s (0.0%) |

### CUDA Visibility

| phase | dataset | method | cuda available | current device | current device name | CUDA_VISIBLE_DEVICES | NVIDIA_VISIBLE_DEVICES | torch device count |
|---|---|---|---:|---:|---|---|---|---:|
| auxreg_skip50 | cifar10_x | fedgda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |
| baseline50 | cifar10_x | fedgda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |
| baseline50 | cifar10_x | fedogda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |
| baseline50 | cifar10_xz | fedgda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |
| baseline50 | cifar10_xz | fedogda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |
| baseline50 | cifar10_z | fedgda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |
| baseline50 | cifar10_z | fedogda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |
| optimized5_smoke | cifar10_x | fedgda_s | True | 0 | NVIDIA H100 NVL | 0,1,2,3 | 0 | 2 |

### Nested Epoch Evidence

The auxiliary regression trainer recorded this loop shape:
- `configured_epochs=3; outer_epochs=3; inner_epochs=3; effective_passes=9`

## Queue-Time Estimates

Projection assumptions: baseline CIFAR profiles are scaled as `fixed setup + 1500 * measured round-loop/50`; FEMNIST is shown only as a same-cost placeholder until FEMNIST profiles are run.

| estimate | 1 GPU | 2 GPUs | note |
|---|---:|---:|---|
| CIFAR 90-run subset, current path | 74.4 h | 37.2 h | 3 CIFAR scenarios x 2 methods x 5 seeds x 3 alphas |
| Full 180-run plan if FEMNIST similar | 148.9 h | 74.4 h | rough; FEMNIST needs its own profile |
| Remaining full plan at current completion fraction | 134.8 h | 67.4 h | uses manifest completed count |
| Full plan, bypass model selection after validation | 141.9 h | 70.9 h | requires equivalence/sign-off that final configs are already fixed |
| Full plan, remove measured aux-reg train+reg aggregate | 104.7 h | 52.4 h | lower-bound estimate; requires equivalence validation |

## Diagnostic Comparisons

| diagnostic phase | dataset | method | baseline wall min | diagnostic wall min | baseline loop sec/round | diagnostic loop sec/round | best val MSE | test MSE at best val | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| auxreg_skip50 | cifar10_x | fedgda_s | 3.4 | 3.4 | 1.874 | 0.967 | 0.1486 -> 0.1518 (+0.0031) | 0.1492 -> 0.1525 (+0.0034) | diagnostic-only; compare metrics across seeds before adopting |
| optimized5_smoke | cifar10_x | fedgda_s | 3.4 | 0.1 | 1.874 | 0.559 | 0.1486 -> 0.2175 (+0.0688) | 0.1492 -> 0.2222 (+0.0730) | diagnostic-only; compare metrics across seeds before adopting |

## Recommendations

Ranked recommendation list:

1. Implemented safe operational fix: run one independent manifest worker per available broker GPU, with disjoint output roots and `--resume-skip-completed`.
2. Implemented protocol-preserving code cleanup: append round CSV rows by default, avoid repeated eval-target CPU copies, avoid unused eval-history state copies, keep auxiliary-regression state on device by default, refresh global state explicitly after aggregation, and make DataLoader workers/pinned memory configurable.
3. Implemented bug fix requiring coauthor awareness: auxiliary regression now treats `auxiliary_regression_epochs=epochs` as exactly that many local passes instead of the previous nested `epochs x epochs` loop.
4. Requires equivalence validation: disable auxiliary regression entirely for final GMM runs if 3-seed diagnostics confirm `g`/`f` validation and test metrics are unchanged.
5. Requires equivalence validation/sign-off: bypass repeated model-selection setup in final runs if the validation-selected architecture/hyperparameters are already fixed; this uses validation MSE as the internal per-round proxy because critic history is absent.
6. Professor sign-off required: fewer validation rounds, fewer communication rounds, changed client sampling, changed batch semantics, or precision changes such as float32/TF32.

Baseline average GPU utilization range is `35.8%` to `41.4%`, so there is clear underfill/headroom on H100.
Baseline model-selection cost ranges from `108.0s` to `205.8s` per run before federated rounds start.
Baseline auxiliary regression training cost ranges from `9.0s` to `33.2s` per 50-round profile.

## Implemented Usage

Implemented runtime controls now flow through `scripts/run_manifest.py` and are recorded in `effective_config.json`/`metrics.json`.

Profile the fast-path candidate without touching production outputs:

```bash
gpurun -g 1 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_highdim_stochastic_gpu_profile.py --phase optimized50 --rounds 50 --datasets cifar10_x --methods fedgda_s --disable-aux-reg --skip-model-selection --disable-periodic-checkpoints
```

Launch production only after equivalence/sign-off, using explicit flags so generated YAMLs show the protocol changes:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_manifest.py --manifest experiments/highdim_coauthor_protocol_v1/alpha1/final_manifest_stochastic.csv --config-dir experiments/highdim_coauthor_protocol_v1/alpha1/generated_configs_final_fast --output-root results/rerun_protocol_v1_real_images_abs_alpha1_fast --gpu-ids 0,1 --max-parallel 2 --resume-skip-completed --disable-auxiliary-regression --skip-model-selection --override-periodic-checkpoint-interval 0 --results-json experiments/highdim_coauthor_protocol_v1/alpha1/final_stochastic_fast_launcher_results.json
```

## Artifact Locations

- Profile root: `results/_profiling/highdim_stochastic_gpu_util`
- Summary CSV: `experiments/highdim_coauthor_protocol_v1/stochastic_gpu_util_profile_summary.csv`
- Report: `experiments/highdim_coauthor_protocol_v1/stochastic_gpu_util_investigation.md`
