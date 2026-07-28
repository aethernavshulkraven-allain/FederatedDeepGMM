# alpha0p5 stochastic — budget slice, launched 2026-07-25

## What is running

A 22-run slice of `final_manifest_stochastic.csv`, launched under the `gpurun` broker
inside tmux session **`fedgmm_highdim`**.

| | |
|---|---|
| manifest | `budget_slice_20260725.csv` |
| output root | `results/rerun_protocol_v1_real_images_abs_alpha0p5` |
| runs | 22 |
| estimated cost | 16.9 GPU-h worst case, ~11.7 GPU-h typical |
| quota at launch | 24.8 of 48 GPU-h remaining |
| launcher | `scripts/launch_highdim_slice_gpurun.sh` |
| log | `logs/gpurun/highdim_alpha0p5_cifar_slice_20260725_212128.log` |
| per-run results | `budget_slice_run_results_<stamp>.json` |

## Why this slice

The full remaining queue is ~161 runs x ~0.53 GPU-h = ~85 GPU-h, far past a 48 GPU-h
weekly quota. A `(dataset, alpha)` cell is only interpretable once **both** methods have
all five seeds — the deliverable is the matched-seed FedGDA-vs-FedOGDA comparison — so
the slice takes whole cells, cheapest-to-complete first, rather than an arbitrary prefix.

Selected (completes the whole CIFAR-10 family at alpha=0.5):

| cell | already done | to run |
|---|---|---|
| `cifar10_x` | 8 | 2 |
| `cifar10_xz` | 0 | 10 |
| `cifar10_z` | 0 | 10 |

Deferred for lack of budget: `femnist_x`, `femnist_xz`, `femnist_z` (7.7 GPU-h each).

## Status when it was launched

Both H100s were held by another user (`himanshus`, two long-lived vLLM 70B servers,
`guaranteed` class, 6.8 h in). The job was **queued at position 1** and starts
automatically when a GPU frees. It may sit in the queue for hours.

## Checking on it

```bash
tmux attach -t fedgmm_highdim          # live view (Ctrl-B then D to detach)
gpurun --status                        # budget + queue position
tail -f logs/gpurun/highdim_alpha0p5_cifar_slice_20260725_212128.log
```

Completed-run count:

```bash
find results/rerun_protocol_v1_real_images_abs_alpha0p5 -name metrics.json | wc -l
```

Expect 19 when all 22 new runs land on top of the 8 pre-existing `cifar10_x` runs
(8 + 22 = 30 total for the CIFAR family).

## Safety notes

- `--resume-skip-completed` is set, so re-launching after any interruption is free and
  never redoes finished runs.
- `--keep-going` is set: one failure does not abort the queue. Check the results JSON for
  `failed_process` / `failed_validation` entries afterwards.
- The launcher **refuses to start** unless exactly one GPU is visible, guarding against
  a run landing on another user's GPU.

## Related fix

`fedgmm/sp_decentralized_mnist_lr_example/main.py` previously set
`CUDA_VISIBLE_DEVICES=0,1,2,3` unconditionally at import, which would have overwritten the
broker's allocation and let a job run on GPU 0 while another user held it. It now only
sets that default when the variable is unset.
