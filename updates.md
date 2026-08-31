# High-Dimensional Experiment Update

## Executive summary

1. The stochastic FedGDA-S/FedOGDA-S campaign is complete: all 72 tuning runs
   and all 180 five-seed final runs have been audited.
2. The original deterministic learning-rate grid was too conservative. A
   validation-only learning gate showed that both FedGDA-D and FedOGDA-D can
   learn when larger learning rates and smaller critic multipliers are used.
3. A revised deterministic tuning protocol has been proposed, but no revised
   full tuning queue or five-seed final matrix has been launched.
4. Runtime is the main practical constraint. The simulator processes 1,000
   clients serially in every deterministic round.
5. Measured timing now replaces the earlier rough estimate. The deterministic
   campaign is projected at approximately `665` GPU-hours, or about `14` quota
   weeks, rather than the previously quoted `1,780–1,820` GPU-hours and `37–38`
   weeks. The earlier figure was derived from runs that predate two runtime
   corrections and therefore overstated the cost by roughly `2.7×`.
6. Disabling the auxiliary regression saves `15–25%` of training time whenever
   the structural model is a CNN. Our earlier reading that it made little
   difference was measured only on `femnist_z` and does not generalise.

## Current status

| Work item | Status |
|---|---|
| Stochastic seed-0 tuning | Complete: `72/72` |
| Stochastic five-seed finals | Complete: `180/180` |
| Stochastic final aggregate cells | Complete: `36/36` |
| Deterministic auxiliary-regression equivalence check | Complete: `4/4` |
| Deterministic learning gate | Complete: `24/24` |
| Deterministic multi-seed confirmation | Complete: `6/6` new runs |
| Original deterministic tuning grid | `9/72` complete; superseded and must not be resumed unchanged |
| Deterministic runtime diagnostic | Complete: `6/6` unprofiled runs, `3/3` profiled runs |
| Revised deterministic full tuning | **Proposed, not launched** |
| Deterministic five-seed finals | **Not started: `0/180`** |
| High-dimensional centralized comparison | **Not implemented or launched** |

## 1. Completed stochastic study

The completed study compares FedGDA-S and FedOGDA-S across:

```text
3 heterogeneity levels × 6 image scenarios × 2 methods × 5 seeds
= 180 final runs
```

Hyperparameters and checkpoints were selected using validation MSE only. Test
MSE was read only after the configuration and best-validation checkpoint were
fixed.

Main findings:

- There is no clear monotonic relationship between the tested heterogeneity
  levels and Test MSE.
- Neither method dominates broadly: FedGDA-S has lower mean Test MSE in `10/18`
  scenario/alpha comparisons, while FedOGDA-S has lower mean in `8/18`.
- Best-validation checkpoints can perform well, but the final iterates are often
  unstable for both methods, especially when the structural model is a CNN.
- Transient non-finite metrics occurred in `29/90` FedOGDA-S runs and `0/90`
  FedGDA-S runs. None of those rows selected a best-validation checkpoint, but
  the finding is important for numerical-stability reporting.

The appropriate scientific claim is therefore about performance at the
validation-selected checkpoint, not last-iterate convergence or broad OGDA
superiority.

## 2. Why the deterministic study was paused

The original deterministic grid used learning rates `{0.001, 0.003}` with a
critic multiplier of `10`. Its validation MSE stayed near `0.235`, approximately
the constant-predictor baseline for the diagnostic scenario.

Instead of spending the projected full-campaign budget on a poorly configured
method, we paused the queue and tested whether deterministic training was
fundamentally failing or simply undertuned.

## 3. Deterministic learning gate

The learning gate used the cheapest representative setting:

```text
scenario: femnist_z
alpha: 0.5
seed: 0
rounds: 150
methods: FedGDA-D and FedOGDA-D
learning rates: {0.001, 0.003, 0.01, 0.03}
critic multipliers: {1, 3, 10}
```

The gate showed that deterministic training can learn. For example, FedGDA-D
with learning rate `0.03` and critic multiplier `3` achieved validation MSE
`0.0465`, compared with the approximately `0.235` constant baseline.

The nominal FedOGDA-D `0.03/cm3` result was not accepted because it contained
transient non-finite objective metrics. The strongest fully finite FedOGDA-D
candidate was `0.03/cm1`.

## 4. Three-seed confirmation

We ran the clean/provisional candidates on seeds `1` and `2` and combined them
with the seed-0 gate result:

| Candidate | Mean best-validation MSE ± sample SD, seeds `0–2` |
|---|---:|
| FedGDA-D, `0.03/cm3` | **`0.0624 ± 0.0145`** |
| FedOGDA-D, `0.03/cm1` | **`0.0978 ± 0.0064`** |
| FedOGDA-D, `0.01/cm3` | `0.1860 ± 0.0134` |

All six new runs completed 150 rounds without detected non-finite metrics.
FedOGDA-D `0.03/cm1` had consistent best checkpoints but unstable final
iterates, so checkpoint selection remains essential.

These results establish feasibility only for `femnist_z`, alpha `0.5`, and
seeds `0–2`. They do not yet validate transfer to CNN structural models,
CIFAR-10, other alpha values, 500 rounds, or all five final seeds.

## 5. Supporting implementation checks

- Disabling the separate auxiliary regression produced exactly identical
  per-round curves, final structural/critic parameters, best checkpoints, and
  selection metrics for both deterministic methods. It can therefore be
  disabled without changing the deterministic GMM trajectory.
- **Correction on its runtime effect.** We previously reported that disabling
  the auxiliary regression gave little runtime benefit. That measurement was
  taken only on `femnist_z`, the one scenario whose structural model is a small
  scalar network. Paired timing runs across three scenarios show the saving
  depends on the scenario:

  | Scenario | Auxiliary on | Auxiliary off | Training time saved |
  |---|---:|---:|---:|
  | `femnist_z` | `177.7` s | `177.5` s | `0.2%` |
  | `femnist_x` | `215.3` s | `175.7` s | **`25.0%`** |
  | `cifar10_xz` | `394.4` s | `359.7` s | **`15.4%`** |

  Once the structural model is a CNN, the auxiliary regression accounts for
  `15–25%` of training time. This changes no scientific result — the exact
  equivalence above still holds, and protocol v2 already disables it — but it
  does mean any cost estimate taken from auxiliary-enabled runs is too high.
- The numerical logger previously checked model parameters but not all reported
  metrics. It has been corrected so NaN/Inf model states or metrics mark a run
  as diverged. This changes failure detection, not the optimization updates.
- **Launch hazard found during the diagnostic.** The alpha-0.5 deterministic
  source manifest predates several configuration columns, including
  `auxiliary_regression` and `periodic_checkpoint_interval`. Generating new rows
  against that header silently discards those fields without any error, and the
  run then falls back to its defaults, which enable the auxiliary regression.
  This occurred during the diagnostic: the first pass declared the auxiliary
  regression disabled and in fact ran with it enabled. Any protocol-v2
  preparation script built from the same source must append these columns
  explicitly, as the learning-gate script does, and the generated configuration
  files must be checked before launch. Left unchecked, the campaign would run
  the wrong configuration and cost `15–25%` more than budgeted on every image
  scenario.

## 6. Proposed deterministic protocol v2

For every `(alpha, scenario, method)` cell, the proposed seed-0 candidates are:

| Method | Validation candidates |
|---|---|
| FedGDA-D | `0.01/cm3`, `0.03/cm3` |
| FedOGDA-D | `0.01/cm3`, `0.03/cm1` |

The scientific task, datasets, model architectures, client participation, and
final reporting target remain unchanged. The main amendment is the
validation-informed learning-rate/critic-multiplier grid.

Selection safeguards:

1. Reject candidates with any non-finite model state or reported metric.
2. Require meaningful improvement over the scenario-specific constant
   predictor.
3. Select configurations and checkpoints using validation MSE only.
4. If the winning candidate reaches its best validation value at round `140+`,
   extend both candidates in that cell before fixing the selection.
5. Read Test MSE only after selection is fixed.

The base tuning matrix contains 72 runs. Four exact `femnist_z`, alpha-0.5
seed-0 configurations can be reused from the learning gate, leaving 68 proposed
tuning runs. Once all 36 method/scenario/alpha selections are fixed, the final
study would run five seeds per cell:

```text
3 alphas × 6 scenarios × 2 methods × 5 seeds = 180 final runs
```

This protocol is still awaiting review and has not been launched.

## 7. Runtime status

### 7.1 Revised projection

The runtime diagnostic is complete. It measured `femnist_z`, `femnist_x`, and
`cifar10_xz` at the protocol-v2 configuration, with auxiliary regression both
enabled and disabled. The projection below replaces the earlier rough estimate.

| Stage | Earlier rough estimate | Measured projection |
|---|---:|---:|
| Revised seed-0 tuning (`72` runs, 150 rounds) | `180–220` GPU-hours | `73` GPU-hours |
| Five-seed deterministic finals (`180` runs, 500 rounds) | approximately `1,600` GPU-hours | `592` GPU-hours |
| Combined | approximately `1,780–1,820` GPU-hours | approximately `665` GPU-hours |

At a 48 GPU-hour weekly quota this is roughly **`14` quota weeks rather than
`37–38`**. Bracketing every scenario at the cheapest and most expensive measured
per-round cost gives a range of `536–845` GPU-hours, or `11–18` quota weeks.

The earlier figure was not wrong for the runs it described. It was extrapolated
from the nine original deterministic candidates, which ran with the auxiliary
regression enabled *and* with an accidental nested auxiliary loop that performed
nine passes per client per round instead of three. Neither applies to protocol
v2, so the estimate had to be rebuilt from measured cost rather than rescaled.

Measured per-round cost, auxiliary regression disabled:

| Scenario | Setup | Seconds per round | Basis |
|---|---:|---:|---|
| `femnist_z` | `55.8` s | `19.09` | measured |
| `femnist_x` | `57.1` s | `18.99` | measured |
| `femnist_xz` | `57` s | `24.37` | interpolated |
| `cifar10_z` | `169` s | `24.37` | interpolated |
| `cifar10_x` | `169` s | `24.37` | interpolated |
| `cifar10_xz` | `169.0` s | `29.74` | measured |

`femnist_z` and `femnist_x` cost essentially the same per round despite having
opposite structures, which indicates the driver is how many convolutional
networks a cell carries rather than which side the image appears on. The three
unmeasured scenarios are interpolated on that basis. They account for roughly
half the projected total and should be measured before the schedule is fixed.

### 7.2 Where the time goes

| Component | `femnist_z` | `femnist_x` | `cifar10_xz` |
|---|---:|---:|---:|
| Client loop | `96.8%` | `96.8%` | `96.6%` |
| — of which gradient computation | `83.3%` | `83.4%` | `85.0%` |
| Aggregation | `1.0%` | `1.0%` | `0.9%` |
| Evaluation | `0.8%` | `0.8%` | `1.4%` |
| Everything else | `1.4%` | `1.4%` | `1.1%` |

Aggregation, evaluation, checkpointing, per-round logging and parameter copying
together account for under `4%` of a round. There is no meaningful overhead left
to remove, and configuration-level options that target these stages will not
change the campaign cost.

The remaining cost is genuine computation spread across 1,000 sequential small
jobs: roughly `5.3` ms per local step on `femnist_x` and `8.4` ms on
`cifar10_xz` for about 20 samples each. This is far above the arithmetic
involved and is dominated by per-operation launch overhead. GPU utilisation
confirms the pattern, averaging `42.8%` on `cifar10_xz`.

### 7.3 Options for further speedup

Any further gain requires changing how clients are executed, not trimming waste:

1. Running the client loop in parallel rather than sequentially. The clients are
   independent given the global iterate, so this is the only large gain
   available. It would not be bit-identical to current results and would need
   its own equivalence check, plus handling for unequal client sizes and for
   batch normalisation.
2. Removing the per-client model copies and accumulating the aggregate
   incrementally. This is bit-identical if client order is preserved, and cuts
   resident memory from roughly `17` GB to about one model. It does not speed up
   a single run; it makes option 3 possible.
3. Running more independent jobs per GPU. This needs no code change and does not
   affect results, but memory currently prevents it: `cifar10_xz` peaked at
   `64` GB. It becomes viable only after option 2.
4. Reducing numerical precision is **not** recommended. It would be a protocol
   change, and because the work is launch-bound the realised gain would be well
   below the nominal factor.

Options 1 and 2 are implementation projects and should be scheduled as such, not
attempted inside a production launch.

## 8. Decisions requested from co-authors

1. Approve or revise the proposed deterministic candidate grid.
2. Decide whether the `femnist_z` evidence is sufficient to begin the broader
   tuning matrix or whether an image-scenario transfer gate should be reviewed
   first.
3. Approve use of the legacy objective and sample-size aggregation for
   continuity with the completed stochastic study, with explicit disclosure of
   the associated implementation caveats; otherwise define a separate campaign.
4. Decide whether the full 500-round, five-seed deterministic matrix is
   affordable at the measured `665` GPU-hours, or about `14` quota weeks. This
   is the decision the runtime diagnostic was run to inform.
5. Decide whether to measure the three interpolated scenarios (`femnist_xz`,
   `cifar10_z`, `cifar10_x`) before fixing a schedule. They carry roughly half
   the projected total and are currently estimated rather than measured; the
   check costs well under one GPU-hour.
6. Review the centralized comparison as a separate design task; it is not yet
   launch-ready.

## Important interpretation notes

- The current image data are certified as reproducible from the author-code
  pipeline, but not verified as an exact match to every paper-era detail.
- The campaign studies alpha values `{0.1, 0.5, 1.0}`; this is a controlled
  extension, not an exact reproduction of the paper's alpha-0.3 setting.
- Client heterogeneity is quantity skew rather than class-label skew.
- The partitioner assigns the integer-allocation remainder to the last client;
  sample-size weighting can amplify this artifact, so it must be disclosed.
- The stochastic final table contains 18 runs from before the production
  runtime fix and 162 runs after it. This provenance split is documented and
  should remain visible in review.
- Existing results and interrupted artifacts are preserved as scientific
  records; no completed result is overwritten.
- The runtime diagnostic runs six communication rounds per scenario. They exist
  only to measure cost and carry no scientific selection meaning; no candidate,
  checkpoint or Test MSE from them enters any reported table.
- The revised runtime projection assumes the protocol-v2 configuration with the
  auxiliary regression disabled. If a launch enables it, the image scenarios
  will cost `15–25%` more than projected.

## Reference artifacts

- Full handoff: `experiments/highdim_coauthor_protocol_v1/highdim_full_handoff_20260805.md`
- Stochastic summary: `experiments/highdim_coauthor_protocol_v1/stochastic_writeup.md`
- Stochastic final index: `experiments/highdim_coauthor_protocol_v1/stochastic_final_artifact_index.csv`
- Deterministic learning gate: `experiments/highdim_coauthor_protocol_v1/deterministic_learning_gate_20260802/`
- Deterministic multi-seed confirmation: `experiments/highdim_coauthor_protocol_v1/deterministic_multiseed_validation_20260803/`
- Runtime diagnostic: `experiments/highdim_coauthor_protocol_v1/deterministic_runtime_profile_20260805/`
- Runtime findings and method: `experiments/highdim_coauthor_protocol_v1/deterministic_runtime_profile_20260805/runtime_profile_findings.md`
- Runtime figures, machine-readable: `experiments/highdim_coauthor_protocol_v1/deterministic_runtime_profile_20260805/runtime_profile_report.json`



feddeepgmm
adverserialdeepgmm