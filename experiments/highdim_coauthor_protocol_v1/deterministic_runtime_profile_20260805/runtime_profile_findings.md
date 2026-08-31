# Deterministic Runtime Profile — Findings

Generated: 2026-08-05. Diagnostic campaign, not a tuning or production queue.

Answers two questions against the protocol-v2 deterministic arm: does disabling
auxiliary regression reduce runtime, and can the projected ETA be reduced
further. Supersedes the runtime numbers in `highdim_full_handoff_20260805.md`
S17; the scientific content of that handoff is unchanged.

Configuration: 1000/1000 clients, `batch_size=0`, 3 local steps, `fedgda_d`,
`lr=0.03`, `critic_multiplier=3`, alpha 0.5, seed 0, 6 rounds, legacy objective,
sample-size weighting, corrected non-finite metric logging.

## 1. Auxiliary regression does reduce runtime — on image scenarios only

Paired aux-on / aux-off runs, identical in every other field:

| Scenario | aux ON | aux OFF | Saved | % of training time |
|---|---:|---:|---:|---:|
| `femnist_z` | 177.7 s | 177.5 s | 0.2 s | **0.2%** |
| `femnist_x` | 215.3 s | 175.7 s | 39.5 s | **25.0%** |
| `cifar10_xz` | 394.4 s | 359.7 s | 34.7 s | **15.4%** |

The handoff (S12) records "little runtime improvement because the 1000-client
GMM loop dominates." That conclusion was measured **only on `femnist_z`**, the
single scenario where the structural model `g` is a scalar MLP. It does not
generalize: once `g` is a CNN, auxiliary regression is 15–25% of training time.

This changes no science. Exact-equality of aux-on vs aux-off is already proven
(S12, four checks, bit-identical), and protocol v2 already sets
`auxiliary_regression=false`. What it changes is the **cost model**: any
projection derived from aux-on runs overstates protocol-v2 cost.

## 2. Refreshed ETA: ~2.7x lower than the handoff estimate

The handoff's ~1780–1820 GPU-h descends from the nine original deterministic
candidates (median 161.7 min / 150 rounds). Those ran with auxiliary regression
on **and** with the accidental nested `3 x 3 = 9` auxiliary loop, both of which
are absent from protocol v2. The estimate cannot be rescaled — it has to be
rebuilt from measured aux-off cost.

Measured per-round cost (aux off), setup excluded:

| Scenario | setup | s/round | basis |
|---|---:|---:|---|
| `femnist_z` | 55.8 s | 19.09 | measured |
| `femnist_x` | 57.1 s | 18.99 | measured |
| `femnist_xz` | 57 s | 24.37 | interpolated |
| `cifar10_z` | 169 s | 24.37 | interpolated |
| `cifar10_x` | 169 s | 24.37 | interpolated |
| `cifar10_xz` | 169.0 s | 29.74 | measured |

`femnist_z` and `femnist_x` cost the same per round despite opposite scenario
shapes, confirming the driver is *how many CNNs the cell carries* (one CNN plus
one MLP vs two CNNs), not which side is the image. The three unmeasured
scenarios are interpolated between the measured extremes on that basis and are
**not** measured — refresh them before committing to a schedule.

| | Handoff S17 | Measured |
|---|---:|---:|
| Tuning (72 runs @150 rounds) | — | 72.7 GPU-h |
| Finals (180 runs @500 rounds) | ~1600 GPU-h | 592.4 GPU-h |
| **Combined** | **~1800 GPU-h (37–38 wk)** | **665.1 GPU-h (13.9 wk)** |

Bracketing every scenario at the measured cheap and dear extremes gives
**535.8 – 844.6 GPU-h (11.2 – 17.6 quota weeks)** at 48 GPU-h/week.

## 3. No remaining overhead to trim

Per-round phase shares at the v2 config:

| Phase | `femnist_z` | `femnist_x` | `cifar10_xz` |
|---|---:|---:|---:|
| Client loop | 96.8% | 96.8% | 96.6% |
| — of which gradient math | 83.3% | 83.4% | 85.0% |
| Aggregation | 1.0% | 1.0% | 0.9% |
| Evaluation | 0.8% | 0.8% | 1.4% |
| Everything else | 1.4% | 1.4% | 1.1% |

Aggregation, evaluation, checkpointing, CSV writes and state copies together are
under 4%. Config-level knobs (`--skip-gmm-eval`, checkpoint interval, dataloader
workers/pinning) all target that 4% and are not worth changing.

The cost is real computation spread across 1000 sequential tiny jobs: about
5.3 ms/step (`femnist_x`) and 8.4 ms/step (`cifar10_xz`) for roughly 20 samples,
far above the FLOPs involved and dominated by kernel-launch overhead. GPU
telemetry agrees — `cifar10_xz` averages 42.8% utilization (median 34.5%),
`femnist_x` 66.0% (median 85.0%).

Further speedup therefore requires changing *how* clients are run:

1. **Parallelize the client loop** (`torch.func` vmap / `stack_module_state`).
   The clients are independent given the global iterate, so this is the only
   large win available. Not bit-exact — needs an equivalence gate like S12's —
   plus padding for unequal client sizes and `BatchNorm1d` handling.
2. **Remove the 1000 per-client model deepcopies** (`fedavg_api.py:543`) and
   accumulate the weighted sum incrementally. Bit-exact if accumulation keeps
   client order, and cuts resident memory from ~17 GB to roughly one model.
   No single-run speedup; it unlocks (3).
3. **More concurrent runs per GPU.** Free and numerics-neutral, but currently
   memory-capped: `cifar10_xz` peaked at 64 GB, `femnist_x` at 34.6 GB. Viable
   only after (2).
4. **float64 to float32.** Not recommended: a protocol change, and these kernels
   are launch-bound so the realised gain would be well under the nominal 2x.

## 4. Consistency with the existing campaigns

These deterministic numbers were checked against work already in the repository.

**Against the 180 completed stochastic finals.** The stochastic arm differs only
in participation (10 of 1000 clients instead of 1000) and horizon, so the
deterministic per-client cost should reproduce stochastic per-round runtime.
Taking measured per-client GMM cost, adding the measured auxiliary overhead
(stochastic production ran aux **on**), and scaling to 10 clients:

| Scenario | stochastic measured | predicted | ratio |
|---|---:|---:|---:|
| `femnist_z` | 0.394 s/round | 0.338 | 1.17x |
| `femnist_x` | 0.423 s/round | 0.397 | 1.07x |
| `cifar10_xz` | 0.893 s/round | 0.756 | 1.18x |

Agreement to 7–18%, with the prediction low in every case — as expected, since
the stochastic path also pays client sampling, `_record_stochastic_batching`,
and carries the mixed 18/162 implementation vintage.

**Against the nine original deterministic candidates.**

| Scenario | old (aux on + 9x loop) | new (aux off) |
|---|---:|---:|
| `femnist_z` | 19.1–27.2 s/round | **19.09** |
| `femnist_x` | 64.7–140.4 s/round | 18.99 |

`femnist_z` matches the fastest old run to three significant figures. That is
the strongest available check: `femnist_z` is precisely the scenario where the
auxiliary model is free, so old and new *must* agree — and they do. `femnist_x`
is 3.4–7.4x slower in the old runs, which is where the auxiliary CNN and the 9x
loop applied.

**Against the stochastic GPU-utilization investigation.** That study measured
39.5–39.9% average H100 utilization for `cifar10_xz`; this one measures 42.8%.
Its `auxreg_skip50` probe also raised `cifar10_x` utilization from ~36–38% to
44.9%, independently corroborating that auxiliary regression is a real workload
component rather than bookkeeping.

**Mechanism, confirmed in `model_hub.py`.** The auxiliary model always mirrors
the structural model `g`:

| Scenario | g | f | auxiliary | aux cost |
|---|---|---|---|---|
| `femnist_z` | MLPModel | DefaultCNN | **MLPModel** | free |
| `cifar10_z` | MLPModel | CIFAR10CNN | **MLPModel** | free (predicted) |
| `femnist_x` | DefaultCNN | MLPModel | **DefaultCNN** | costly |
| `femnist_xz` | DefaultCNN | DefaultCNN | **DefaultCNN** | costly (predicted) |
| `cifar10_x` | CIFAR10CNN | MLPModel | **CIFAR10CNN** | costly (predicted) |
| `cifar10_xz` | CIFAR10CNN | CIFAR10CNN | **CIFAR10CNN** | costly |

So auxiliary regression is free exactly when the structural input `x` is scalar,
and costs a full extra CNN pass whenever `x` is an image. S12 tested the single
scenario family where it is free, which is why its conclusion did not generalise.

**One inconsistency, and it matters.** The old candidates show a 1.4–2.2x spread
across configurations that are compute-identical — learning rate does not change
cost, yet `femnist_z` ranges 19.1–27.2 s/round and `femnist_x` 64.7–140.4. That
spread points to GPU contention during the original queue. It is a second,
independent reason not to base an ETA on those runtimes, and it is why the
*minimum* of each old scenario tracks the clean measurement best.

## 5. Launch trap — manifest columns are silently dropped

`alpha0p5/tuning_manifest_deterministic.csv` predates these columns:

```text
auxiliary_regression        auxiliary_regression_epochs
append_round_csv            periodic_checkpoint_interval
log_test_mse_by_round       test_mse_used_for_selection
selection_metric_source     objective_mode
aggregation_weighting
```

Writing new rows against that header with `csv.DictWriter` drops any of these
fields without error, and the run falls back to defaults — **auxiliary
regression on**, periodic checkpoints every 200 rounds. This was hit during this
diagnostic: the first pass declared `auxiliary_regression=False` and ran aux-on.

`prepare_highdim_deterministic_learning_gate.py` appends them explicitly via its
`EXTRA_FIELDS` tuple. Any protocol-v2 prepare script built from the same source
must do the same, and the generated YAML must be checked before launch:

```bash
grep -H "auxiliary_regression:" <config-dir>/*/*/*/*.yaml
```

Otherwise the campaign silently runs the wrong configuration and costs 15–25%
more than budgeted on every image cell.

## 6. Reproduce

```bash
python scripts/prepare_highdim_deterministic_runtime_profile.py
gpurun -g 2 bash scripts/launch_highdim_deterministic_runtime_profile.sh
python scripts/analyze_highdim_deterministic_runtime_profile.py \
  --output experiments/highdim_coauthor_protocol_v1/\
deterministic_runtime_profile_20260805/runtime_profile_report.json
```

Artifacts: `results/highdim_deterministic_runtime_profile_20260805/`
(`unprofiled/` both aux arms, `profiled/` aux-off only),
`results/_profiling/highdim_deterministic_runtime_profile_20260805/`.
Total cost ~1.5 GPU-h. Stub directories from a killed launcher job were archived
to `results/_failed/20260805_215659/`.

Profiled totals carry a CUDA synchronize on entry and exit of every span, so
they are used only for the relative shape of the breakdown; all timing and
projection figures come from the unprofiled pass.
