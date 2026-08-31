# Complete High-Dimensional Experiment Handoff

Generated: `2026-08-05T21:07:26+05:30`

Repository:

```text
/home/arnav22103/FederatedDeepGMM
```

Python environment:

```text
/home/arnav22103/miniconda3/envs/fedgmm/bin/python
```

This document consolidates the full high-dimensional experiment thread: the
original design, stochastic tuning and five-seed finals, GPU-utilization work,
result migrations, scientific review, paper integration, deterministic audit,
deterministic learning gate, multi-seed validation, and the current launch
decision. It distinguishes completed production results from diagnostics and
from proposals that have not been launched.

## 1. Current State At A Glance

| Work item | Current state |
|---|---|
| Stochastic seed-0 tuning | Complete: `72/72` |
| Stochastic five-seed finals | Complete: `180/180` |
| Stochastic final aggregate cells | Complete: `36/36`, five seeds each |
| Stochastic stability probe | Complete diagnostic: `16/16`; not part of the final table |
| Original deterministic tuning grid | `9/72` canonical candidates complete; grid superseded by the learning-gate findings |
| Deterministic auxiliary equivalence | Complete: `4/4` short runs, exact equality passed |
| Deterministic learning gate | Complete: `24/24` at `femnist_z`, alpha `0.5`, seed `0` |
| Deterministic multi-seed confirmation | Complete: `6/6` for seeds `1,2`, combined with gate seed `0` |
| Deterministic five-seed finals | Not started: `0/180` |
| High-dimensional centralized tuning/finals | Not implemented or launched |
| Active high-dimensional job | None |

Scheduler snapshot at generation time:

```text
GPU 0: idle
GPU 1: idle
Queue: empty
Weekly quota remaining: 38.4 GPU-hours
```

No `run_manifest.py`, high-dimensional launcher, or high-dimensional training
process is active. The only listed tmux session is unrelated (`iecu-0`).

**Do not resume the original deterministic queue unchanged.** The old learning
rate grid was shown to be inadequate, and the latest revised deterministic
matrix is still a proposal rather than a generated or approved production
manifest.

## 2. Scientific Objective And Data

The objective is to estimate the structural response

```text
g0(x) = abs(x)
```

with federated DeepGMM when the structural input `x`, the instrument `z`, or
both are represented by real images.

The scalar data-generating process is:

```text
C ~ Normal(0, 1)
Z ~ Uniform(-3, 3)
epsilon_x, epsilon_y ~ Normal(0, 0.1^2)
X = Z + C + epsilon_x
Y = abs(X) + 2 C + epsilon_y
```

A scalar `v` is mapped to an image label using:

```text
q(v) = round(clip(1.5 v + 5, 0, 9))
```

An image with label `q(v)` is then sampled from digits-only Federated EMNIST or
CIFAR-10. The campaign uses six scenarios:

| Scenario | Structural input `x` | Instrument `z` | Structural model `g` | Critic model `f` |
|---|---|---|---|---|
| `femnist_x` | FEMNIST image | scalar | `DefaultCNN` | `MLPModel` |
| `femnist_z` | scalar | FEMNIST image | `MLPModel` | `DefaultCNN` |
| `femnist_xz` | FEMNIST image | FEMNIST image | `DefaultCNN` | `DefaultCNN` |
| `cifar10_x` | CIFAR-10 image | scalar | `CIFAR10CNN` | `MLPModel` |
| `cifar10_z` | scalar | CIFAR-10 image | `MLPModel` | `CIFAR10CNN` |
| `cifar10_xz` | CIFAR-10 image | CIFAR-10 image | `CIFAR10CNN` | `CIFAR10CNN` |

Fixed data properties:

| Property | Value |
|---|---:|
| Scenario generation seed | `527` |
| Training examples | `20,000` |
| Validation examples | `10,000` |
| Test examples | `10,000` |
| FEMNIST source | TFF Federated EMNIST, `only_digits=True` |
| FEMNIST shape | `1 x 28 x 28` |
| CIFAR-10 shape | `3 x 32 x 32` |
| Model/scenario precision | `torch.float64` |
| Response scaling | Train-outcome mean and standard deviation |

Data provenance and certification:

```text
experiments/rerun_protocol_v1_real_images_abs_alpha0p5/data_generation_manifest.json
experiments/rerun_protocol_v1_real_images_abs_alpha0p5/data_certification.md
```

The saved files passed the available shape, hash, finite-value,
response-semantic, and split-isolation checks. This certifies reproducibility of
the current author-code data. It does not by itself prove exact alignment with
every paper-era implementation detail.

## 3. Heterogeneity And Important Partition Caveats

Train, validation, and test splits are independently quantity-partitioned over
`N=1000` clients:

```text
p ~ Dirichlet(alpha * 1_N), alpha in {0.1, 0.5, 1.0}
```

At least five examples are assigned to every client. This is quantity skew, not
class-label skew. Lower alpha means more unequal client sizes.

The implementation floors the Dirichlet allocations and assigns the full
rounding remainder to the last client. A reviewer simulation found that client
999 receives roughly `241`, `480`, and `519` training samples on average at
alpha `0.1`, `0.5`, and `1.0`, respectively, versus a global average of `20`.
With sample-size aggregation this client can carry about `2.4-2.6%` of a
full-participation deterministic update. This artifact is shared with the
completed stochastic campaign and must be disclosed rather than silently
changed mid-campaign.

The scalar input used by scalar-`x` scenarios is the affine representation
`1.5 X + 5`, not raw `X`. It is invertible and not expected to change the
estimand, but it is another implementation detail reviewers should know.

## 4. Original Coauthor Design And The Corrected Batch Size

An early handoff inherited a halted campaign. The coauthor then explicitly
corrected the stochastic configuration after an earlier batch-size mistake:

```text
1500 communication rounds
batch_size = 256
10 of 1000 clients per round
3 local steps/epochs
six high-dimensional scenarios
tune on seed 0, then run seeds 0,1,2,3,4
```

The intended full comparison was:

| Arm | Methods | Alpha values | Final rounds |
|---|---|---|---:|
| Federated stochastic | FedGDA-S/FedSGDA and FedOGDA-S | `0.1, 0.5, 1.0` | `1500` |
| Federated deterministic | FedGDA-D and FedOGDA-D | `0.1, 0.5, 1.0` | `500` |
| Centralized at the alpha-0.5 comparison | GDA, SGDA, OAdam | `0.5` comparison block | not finalized |

The original nominal count was:

```text
federated tuning:    144
centralized tuning:   36
federated finals:    360
centralized finals:   90
total:               630 runs
```

The original design documents were `high_dim_exp.md` and `high_dim_doe_2.md`.
They were used and edited earlier in the thread, including Markdown rendering
repairs, but neither file is present anywhere under `/home/arnav22103` in the
current filesystem snapshot. Current review must therefore use this handoff,
the protocol manifests, and the result metadata rather than assume those two
files are still available.

An early audit also established that federated manifest weight decay is not
connected to the GMM `g/f` optimizer factories. Twenty-two completed
same-learning-rate/different-weight-decay pairs had identical validation
histories. Weight decay only affected the auxiliary regression Adam optimizer.
The campaign consequently tuned structural learning rate rather than treating
the compatibility weight-decay field as an effective GMM hyperparameter.

## 5. Chronological Execution History

### 5.1 Inherited halted state, 13 July 2026

The initial handoff recorded:

- deterministic tuning job `196` was stopped by the user;
- stochastic tuning job `197` ended with wrapper return code `120`;
- alpha-0.5 had `9/24` deterministic and `24/24` stochastic candidates at the
  artifact level;
- interrupted artifacts were preserved under `results/_failed/`, including:
  `results/_failed/20260713_221959/highdim_interrupted_tuning/` and
  `results/_failed/20260713_235533/highdim_halted_by_user/`;
- no final or centralized run had started.

The return-code-120 condition was a wrapper/process-accounting problem. Artifact
revalidation, not the wrapper code, was used to determine whether a run was
scientifically complete.

### 5.2 Stochastic tuning completion

The remaining alpha `0.1` and alpha `1.0` seed-0 tuning candidates were run via
`gpurun`; job `250` was one of the recorded alpha-1 continuations. All three
alpha blocks were ultimately revalidated as `24/24` complete:

```text
alpha0p1: 24/24
alpha0p5: 24/24
alpha1:   24/24
total:    72/72
```

There was an important communication correction at this point: these were
seed-0 tuning candidates, not the requested five-seed final campaign. The user
correctly challenged the earlier wording that could be read as saying the full
experiment was complete.

### 5.3 Five-seed stochastic finals

The final matrix was launched and continued from already valid runs. Existing
results were preserved. Interrupted partial output was not overwritten.

The original final path was slow, so the remaining queue was migrated twice:

| Source recorded in final index | Count | Notes |
|---|---:|---|
| `old_original` | 19 | Preserved existing complete artifacts |
| `safe_speedup_v1` | 84 | Continuation launched as two processes on two GPUs, job `319` |
| `safe_speedup_v2` | 77 | Remaining continuation, one preemptible GPU, job `324` |
| Total | 180 | Complete |

One v1 partial run was preserved and rerun from scratch in v2 because the
available resume path did not safely restore all best-validation and optimizer
history. No completed result was deleted.

The provenance label and actual implementation vintage differ for one row. The
true split is `18` pre-runtime-fix runs and `162` post-fix runs, even though the
index source labels are `19/84/77`. Three cells are entirely pre-fix:
alpha-0.5 `cifar10_x/FedGDA-S` and both alpha-1.0 `cifar10_x` methods. One cell,
alpha-0.5 `cifar10_x/FedOGDA-S`, mixes three pre-fix and two post-fix seeds.
This implementation-vintage split is documented in the stochastic operational
handoff and must remain in review.

### 5.4 Post-final stability probe

A separate diagnostic tested whether lower structural LR, lower critic
multiplier, or lower server LR removed the severe late-run instability:

```text
cells:
  cifar10_x / FedGDA-S
  cifar10_xz / FedOGDA-S
alpha: 0.1
seed: 0
rounds: 1500
learning_rate: {0.001, 0.003}
critic_multiplier: {1, 3}
server_learning_rate: {1.0, 1.5}
total: 16 runs
```

Job `344` completed `16/16`. The predeclared success criterion was a
final-to-best validation ratio at most `1.5` and a best round after `500`.
No candidate met it; every best round was at or before `365`. The probe is a
diagnostic only and did not replace or select entries in the five-seed table.

### 5.5 Deterministic audit and learning gate

The nine completed original deterministic candidates were reviewed before
spending the projected full-campaign budget. They performed approximately like
a constant structural predictor, so the full queue was paused and a targeted
learning gate was approved.

The gate used the cheapest problematic cell:

```text
scenario: femnist_z
alpha: 0.5
seed: 0
methods: FedGDA-D, FedOGDA-D
learning rates: {0.001, 0.003, 0.01, 0.03}
critic multipliers: {1, 3, 10}
rounds: 150
total: 24 runs
```

An auxiliary-regression equivalence stage ran first. The complete gate was run
on both GPUs as chat-recorded job `426`. It used about `20.11 GPU-hours` for the
24 main runs, with a median of about `50.0 minutes` per run, or about ten hours
wall time with two concurrent GPUs.

### 5.6 Corrected numerical logging and multi-seed confirmation

The gate exposed transient non-finite objective/moment metrics in three OGDA
configurations. The old `finite` field checked only model parameters. The logger
was changed so all structural, moment, objective, and GMM evaluation metrics
must also be finite.

Three clean/provisional candidates were then run for seeds `1` and `2` and
combined with seed `0`. A two-GPU request was initially canceled because the
scheduler needed an atomic two-GPU allocation; the six-run confirmation was
relaunched on GPU 0 as chat-recorded job `434`. It completed in about `4.86`
GPU-hours, roughly five hours wall time.

No deterministic production final has been launched after this confirmation.

## 6. Stochastic Experiment Design

### 6.1 Methods and final shape

| Report label | Repository method | Local optimizer |
|---|---|---|
| FedGDA-S / FedSGDA | `fedgda_s` | momentum-free SGD for both players |
| FedOGDA-S | `fedogda_s` | OGDA for both players |

Final configuration:

| Parameter | Value |
|---|---:|
| Alphas | `0.1, 0.5, 1.0` |
| Scenarios | `6` |
| Methods | `2` |
| Seeds | `0,1,2,3,4` |
| Total clients | `1000` |
| Clients per round | `10` |
| Participation | `1%` |
| Batch size | `256` |
| Local epochs | `3` |
| Communication rounds | `1500` |
| Server learning rate | `1.5` |
| Gradient clip norm | `1.0` |
| Critic multiplier | `10` |
| Objective | legacy `OptimalMomentObjective`, `lambda_1=0.1` |
| Aggregation | sample-size weighted |

Because the average client has only about 20 samples, batch size 256 means most
selected clients use one local full-client batch. The stochastic-vs-
deterministic contrast therefore does not isolate minibatch noise. The major
cross-arm differences are `1%` versus `100%` participation, different horizons,
and different learning-rate grids.

The manifest `weight_decay=0.05` field is not applied to the structural or
critic optimizers. It affected the separate auxiliary regression model. The
paper correctly describes effective `g/f` weight decay as zero.

This is not a logistic-regression experiment. Scalar inputs use a one-hidden-
layer MLP and image inputs use the CNNs listed in Section 2.

The legacy objective used in all completed stochastic finals is:

```text
epsilon = g_theta(X) - Y
M(theta, tau) = E[f_tau(Z) * epsilon]
R(theta, tau) = E[f_tau(Z)^2 * epsilon^2]

g minimizes:  M
f minimizes: -M + 0.1 R
```

The regularizer uses the live structural model. A newer paper-aligned mode uses
a frozen previous global structural iterate and `lambda_1=1/4`; it was not used
for this completed table. Changing only future rows to that mode would break
implementation continuity and must be a separate campaign.

### 6.2 Tuning and selection

Tuning used seed `0`, 150 rounds, and:

```text
learning_rate in {0.003, 0.01}
critic_multiplier = 10
server_learning_rate = 1.5
gradient_clip_norm = 1.0
```

Each of the `3 x 6 x 2 = 36` alpha/scenario/method cells had two candidates,
giving `72` tuning runs. Selection was validation-only:

1. Exclude numerical divergence.
2. Lowest `best_validation_mse`.
3. Lower last-50 validation-MSE standard deviation.
4. Smaller final-minus-best validation gap.
5. Lower learning rate.

Test MSE was read only after the validation-selected configuration and
checkpoint were fixed. Each selected configuration was then run for all five
seeds, giving `180` finals.

Validation-selected structural learning rates:

| Alpha | Scenario | FedGDA-S LR | FedOGDA-S LR |
|---:|---|---:|---:|
| 0.1 | `cifar10_x` | 0.01 | 0.003 |
| 0.1 | `cifar10_xz` | 0.01 | 0.01 |
| 0.1 | `cifar10_z` | 0.01 | 0.003 |
| 0.1 | `femnist_x` | 0.01 | 0.003 |
| 0.1 | `femnist_xz` | 0.003 | 0.003 |
| 0.1 | `femnist_z` | 0.01 | 0.01 |
| 0.5 | `cifar10_x` | 0.003 | 0.01 |
| 0.5 | `cifar10_xz` | 0.01 | 0.01 |
| 0.5 | `cifar10_z` | 0.01 | 0.01 |
| 0.5 | `femnist_x` | 0.01 | 0.01 |
| 0.5 | `femnist_xz` | 0.01 | 0.003 |
| 0.5 | `femnist_z` | 0.01 | 0.01 |
| 1.0 | `cifar10_x` | 0.01 | 0.003 |
| 1.0 | `cifar10_xz` | 0.01 | 0.01 |
| 1.0 | `cifar10_z` | 0.01 | 0.003 |
| 1.0 | `femnist_x` | 0.01 | 0.01 |
| 1.0 | `femnist_xz` | 0.003 | 0.003 |
| 1.0 | `femnist_z` | 0.01 | 0.01 |

### 6.3 Canonical stochastic paths

Design and selection:

```text
scripts/prepare_highdim_coauthor_protocol.py
scripts/analyze_highdim_coauthor_tuning.py
scripts/materialize_highdim_stochastic_finals.py
experiments/highdim_coauthor_protocol_v1/alpha0p1/
experiments/highdim_coauthor_protocol_v1/alpha0p5/
experiments/highdim_coauthor_protocol_v1/alpha1/
```

Execution and audit:

```text
scripts/run_manifest.py
scripts/run_highdim_coauthor_tuning_queue.py
scripts/run_highdim_coauthor_final_queue.py
scripts/audit_highdim_stochastic_finals.py
experiments/highdim_coauthor_protocol_v1/stochastic_final_artifact_index.csv
experiments/highdim_coauthor_protocol_v1/stochastic_final_aggregate_summary.csv
experiments/highdim_coauthor_protocol_v1/stochastic_runs_handoff_20260720.md
experiments/highdim_coauthor_protocol_v1/stochastic_writeup.md
```

The 180-row artifact index is the source of truth for exact result directories.
Final artifacts are intentionally spread over old, v1, and v2 roots. Do not
infer completeness from any one result root.

Exact tuning and migration records:

```text
experiments/highdim_coauthor_protocol_v1/alpha0p1/tuning_manifest_stochastic.csv
experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_stochastic.csv
experiments/highdim_coauthor_protocol_v1/alpha1/tuning_manifest_stochastic.csv
experiments/highdim_coauthor_protocol_v1/alpha0p1/selected_configs_stochastic.csv
experiments/highdim_coauthor_protocol_v1/alpha0p5/selected_configs_stochastic.csv
experiments/highdim_coauthor_protocol_v1/alpha1/selected_configs_stochastic.csv
experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_123539/
experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_233531/
```

### 6.4 Mechanical audit, rechecked 5 August 2026

The latest raw re-audit found:

```text
index rows: 180
unique run IDs: 180
alpha/scenario/method cells: 36
seeds per cell: 0,1,2,3,4
mse_by_round.csv rows per run: exactly 1500
missing required artifacts: 0
```

Every run has:

```text
effective_config.json
metrics.json
mse_by_round.csv
predictions.npz
checkpoints/best_validation.pt
checkpoints/final.pt
```

The indexed runtimes sum to `48.74 GPU-hours`; the overall median is `13.24`
minutes, with a `9.18-46.49` minute range. Runtime comparisons across alpha are
not scientifically meaningful where implementation vintages differ.

Stale/noisy status files include
`final_stochastic_queue_summary.json`, `alpha*/completion_status.json`, and the
alpha-1 original launcher result with return code `120`. Use the final index,
aggregate CSV, revalidation JSON files, and migration summaries instead.

## 7. Stochastic Aggregate Results

Metric: mean and sample standard deviation of
`test_mse_at_best_validation` over five seeds.

| alpha | scenario | method | n | Test MSE at best validation | runtime median min |
|---:|---|---|---:|---:|---:|
| 0.1 | `cifar10_x` | FedGDA-S | 5 | 0.1588 +/- 0.0125 | 13.4 |
| 0.1 | `cifar10_x` | FedOGDA-S | 5 | 0.1730 +/- 0.0268 | 13.9 |
| 0.1 | `cifar10_xz` | FedGDA-S | 5 | 0.1679 +/- 0.0235 | 22.6 |
| 0.1 | `cifar10_xz` | FedOGDA-S | 5 | 0.1621 +/- 0.0104 | 22.7 |
| 0.1 | `cifar10_z` | FedGDA-S | 5 | 0.0449 +/- 0.0147 | 12.5 |
| 0.1 | `cifar10_z` | FedOGDA-S | 5 | 0.0791 +/- 0.0240 | 12.9 |
| 0.1 | `femnist_x` | FedGDA-S | 5 | 0.1590 +/- 0.0186 | 10.3 |
| 0.1 | `femnist_x` | FedOGDA-S | 5 | 0.1541 +/- 0.0115 | 11.1 |
| 0.1 | `femnist_xz` | FedGDA-S | 5 | 0.1554 +/- 0.0165 | 16.4 |
| 0.1 | `femnist_xz` | FedOGDA-S | 5 | 0.1519 +/- 0.0133 | 16.2 |
| 0.1 | `femnist_z` | FedGDA-S | 5 | 0.0118 +/- 0.0021 | 9.4 |
| 0.1 | `femnist_z` | FedOGDA-S | 5 | 0.0103 +/- 0.0011 | 9.9 |
| 0.5 | `cifar10_x` | FedGDA-S | 5 | 0.1890 +/- 0.0561 | 31.8 |
| 0.5 | `cifar10_x` | FedOGDA-S | 5 | 0.1648 +/- 0.0178 | 32.1 |
| 0.5 | `cifar10_xz` | FedGDA-S | 5 | 0.1641 +/- 0.0150 | 22.3 |
| 0.5 | `cifar10_xz` | FedOGDA-S | 5 | 0.1644 +/- 0.0095 | 22.4 |
| 0.5 | `cifar10_z` | FedGDA-S | 5 | 0.0553 +/- 0.0080 | 12.4 |
| 0.5 | `cifar10_z` | FedOGDA-S | 5 | 0.0646 +/- 0.0054 | 12.9 |
| 0.5 | `femnist_x` | FedGDA-S | 5 | 0.1695 +/- 0.0252 | 10.5 |
| 0.5 | `femnist_x` | FedOGDA-S | 5 | 0.1365 +/- 0.0076 | 10.8 |
| 0.5 | `femnist_xz` | FedGDA-S | 5 | 0.1729 +/- 0.0281 | 16.3 |
| 0.5 | `femnist_xz` | FedOGDA-S | 5 | 0.1433 +/- 0.0124 | 16.4 |
| 0.5 | `femnist_z` | FedGDA-S | 5 | 0.0130 +/- 0.0023 | 9.6 |
| 0.5 | `femnist_z` | FedOGDA-S | 5 | 0.0175 +/- 0.0033 | 10.4 |
| 1.0 | `cifar10_x` | FedGDA-S | 5 | 0.1579 +/- 0.0121 | 31.8 |
| 1.0 | `cifar10_x` | FedOGDA-S | 5 | 0.1728 +/- 0.0304 | 30.6 |
| 1.0 | `cifar10_xz` | FedGDA-S | 5 | 0.1656 +/- 0.0149 | 21.9 |
| 1.0 | `cifar10_xz` | FedOGDA-S | 5 | 0.1657 +/- 0.0089 | 22.0 |
| 1.0 | `cifar10_z` | FedGDA-S | 5 | 0.0513 +/- 0.0092 | 12.1 |
| 1.0 | `cifar10_z` | FedOGDA-S | 5 | 0.0876 +/- 0.0251 | 12.7 |
| 1.0 | `femnist_x` | FedGDA-S | 5 | 0.1586 +/- 0.0126 | 10.1 |
| 1.0 | `femnist_x` | FedOGDA-S | 5 | 0.1375 +/- 0.0133 | 10.8 |
| 1.0 | `femnist_xz` | FedGDA-S | 5 | 0.1444 +/- 0.0092 | 16.3 |
| 1.0 | `femnist_xz` | FedOGDA-S | 5 | 0.1489 +/- 0.0108 | 16.7 |
| 1.0 | `femnist_z` | FedGDA-S | 5 | 0.0147 +/- 0.0034 | 9.4 |
| 1.0 | `femnist_z` | FedOGDA-S | 5 | 0.0174 +/- 0.0036 | 10.3 |

## 8. Stochastic Scientific Findings

### 8.1 Heterogeneity trend is not established

Pooled means at alpha `0.1`, `0.5`, and `1.0` are `0.1190`, `0.1212`, and
`0.1185`. None of the 12 scenario/method series has the expected monotone
ordering `alpha=0.1 > 0.5 > 1.0`. Per-cell retuning, best-validation
checkpointing, the remainder artifact, and the quantity-skew-only construction
are all relevant when interpreting this null trend.

### 8.2 Neither method dominates broadly

FedGDA-S has lower mean MSE in `10/18` alpha/scenario cells; FedOGDA-S has lower
mean in `8/18`. FedGDA-S's strongest advantage is concentrated in `cifar10_z`.
FedOGDA-S is often better on `femnist_x` and `femnist_xz`.

FedOGDA-S has lower average within-cell seed standard deviation, `0.0131`
versus `0.0158`, but it does not have a smaller final-to-best gap. Median
final/best test-MSE ratios are `7.6` for FedOGDA-S and `7.0` for FedGDA-S;
their 90th percentiles are `24.5` and `17.2`, respectively.

### 8.3 Both methods have unstable last iterates

In image-`x` and image-`xz` scenarios, the best checkpoint is commonly in the
first few dozen rounds, followed by a high-amplitude, non-periodic tail. Of 90
runs per method, `83` FedGDA-S and `81` FedOGDA-S runs finish above twice their
best-checkpoint test MSE. Forty runs per method finish above test MSE `1.0`.

Concrete example, `cifar10_x/FedGDA-S/seed0/alpha0.1`:

| Displayed round | 1 | 14 best | 30 | 50 | 100 | 300 | 600 | 1000 | 1500 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation MSE | 0.191 | 0.164 | 0.453 | 0.804 | 1.013 | 5.098 | 2.728 | 2.577 | 3.013 |

From displayed round 200 onward, this run has mean validation MSE `3.356`,
sample standard deviation `2.084`, minimum `0.877`, and maximum `22.722`.
Tail direction-reversal rates range from about `56-69%` for FedGDA-S and
`56-81%` for FedOGDA-S across image-`x/xz` runs. These curves do not support a
claim of last-iterate convergence.

The severe behavior is scenario-specific. The `*_z` cases usually improve much
later and operate at a much smaller absolute MSE scale. This is mechanistically
reasonable because `g` is only a scalar MLP in those scenarios; the CNN burden
is on the critic.

### 8.4 FedOGDA-S transient NaN finding

A fresh scan of every round in all 180 finals exactly reproduced:

```text
FedGDA-S affected runs:   0/90
FedOGDA-S affected runs: 29/90
bad rows:                167
```

Breakdown:

| Method/scenario | Affected runs | NaN rows |
|---|---:|---:|
| FedOGDA-S `cifar10_x` | 8 | 11 |
| FedOGDA-S `cifar10_xz` | 14 | 146 |
| FedOGDA-S `femnist_x` | 5 | 8 |
| FedOGDA-S `femnist_xz` | 2 | 2 |

Both `train_mse` and `val_mse` are NaN on each affected row. All 167 rows are
mislabeled `finite=True` by the old logger. No selected best-validation round is
one of the bad rows. A new checkpoint audit loaded the best and final
checkpoints for all 29 affected runs (`58` checkpoint files) and found every
stored tensor finite. Every affected run also returns to finite metrics by its
final row.

The OGDA implementation follows the standard optimistic update:

```text
theta <- theta - lr * (2 grad_t - grad_(t-1))
```

Optimizer history is reset at the start of each local client call and gradients
are clipped before the optimistic combination. Therefore clipping bounds each
input gradient but not the combined direction, which can be as large as three
times the SGD bound when consecutive gradients oppose one another. A tenfold
critic LR, noisy 10-of-1000 sampling, and a CIFAR CNN critic make this a credible
mechanism for the concentration in `cifar10_xz`.

The code trace establishes that the NaNs are not caused by a malformed OGDA
formula or non-finite stored weights. The exact causal contribution of each
factor would still require an intervention study, so the paper appropriately
calls this a plausible mechanism rather than a formal causal proof.

The reported best-validation metrics remain mechanically protected because
`NaN < best` is false, so a NaN validation row cannot update the best
checkpoint. This does not make the numerical instability unimportant; it means
checkpoint accuracy and last-iterate robustness are different reported
properties.

## 9. GPU Utilization Investigation And Speedups

Primary report and profiling artifacts:

```text
experiments/highdim_coauthor_protocol_v1/stochastic_gpu_util_investigation.md
experiments/highdim_coauthor_protocol_v1/stochastic_gpu_util_profile_summary.csv
results/_profiling/highdim_stochastic_gpu_util/
scripts/run_highdim_stochastic_gpu_profile.py
scripts/analyze_highdim_stochastic_gpu_util.py
```

The completion table near the top of the profiling Markdown is a historical
18 July snapshot, not current campaign status. Its runtime measurements remain
useful; the canonical 180-row final index governs current completion.

The 50-round CIFAR baselines showed average H100 utilization of only
`35.8-41.4%`, despite peaks at 100%. Model-selection/setup consumed `108-206`
seconds before training. The principal underfill sources were:

- ten clients simulated serially within a process;
- very small per-client workloads relative to an H100;
- CPU orchestration and state movement;
- `num_workers=0` and synchronous loading;
- full train and validation evaluation every round;
- auxiliary regression and its state aggregation;
- repeated output/checkpoint work;
- an accidental nested `3 x 3 = 9` auxiliary-regression loop.

Implemented runtime controls include append-only per-round CSV output, reduced
repeated evaluation/state copies, explicit global refresh, device-resident
auxiliary state, configurable DataLoader workers and pinning, periodic
checkpoint controls, model-selection/GMM-evaluation controls for diagnostics,
and exact auxiliary pass counting.

The production continuation deliberately kept the scientific path conservative:

```text
model selection: enabled
GMM evaluation: enabled
auxiliary regression: enabled
validation every round: unchanged
client sampling/batch semantics: unchanged
precision: unchanged
```

It adopted append-only CSV, periodic checkpoints every 200 rounds, and exactly
three auxiliary passes instead of nine. It also used one independent process
per available GPU. The mixed 18/162 implementation vintage is consequently a
real review caveat, even though the auxiliary model does not enter the GMM
objective or reported predictions.

Disabling auxiliary regression was not adopted for stochastic production. The
profile-only 50-round comparison changed best validation by about `0.003`, and
stochastic RNG coupling prevented a bit-exact equivalence claim at that point.

## 10. Paper Integration

The current AAAI source is:

```text
latex/fedgmm/aaai2027.tex
```

The high-dimensional stochastic edits are already present at:

```text
latex/fedgmm/sections/experiments.tex:84
latex/fedgmm/sections/experiments.tex:105
latex/fedgmm/sections/experiments.tex:107
latex/fedgmm/sections/experiments.tex:109
latex/fedgmm/sections/experiments.tex:111
```

These lines contain the DGP/image mapping, model/method definitions,
Dirichlet/data details, stochastic configuration, validation-only tuning, and a
short last-iterate caveat.

The complete table and interpretation are in the appendix input:

```text
latex/fedgmm/sections/additional_experiements.tex:16
latex/fedgmm/sections/additional_experiements.tex:50
latex/fedgmm/sections/additional_experiements.tex:52
latex/fedgmm/sections/additional_experiements.tex:54
latex/fedgmm/sections/additional_experiements.tex:56
```

`aaai2027.tex` includes this file at line 312 under "Benchmark Considerations
and Additional Experiments." The appendix text covers the flat heterogeneity
trend, 10/18 versus 8/18 cell split, last-iterate blowup, and 29/90 OGDA NaN
finding.

The `latex/` directory is ignored by git (`.gitignore:60`). These paper edits do
not appear in `git status` and are not protected by a normal repository commit.
They need independent backup/version control. Also review the legacy
`figures/scenarios_short.pdf` still included by `sections/experiments.tex`; the
new numeric table is in the appendix, while the main figure may encode older
values.

## 11. Original Deterministic Protocol And Why It Was Paused

The original deterministic arm was:

| Parameter | Value |
|---|---:|
| Methods | `fedgda_d`, `fedogda_d` |
| Alphas | `0.1, 0.5, 1.0` |
| Scenarios | `6` |
| Total/participating clients | `1000/1000` |
| Batch size | `0`, one full local-client batch |
| Local steps | `3` |
| Tuning seed/rounds | seed `0`, `150` rounds |
| Original LR grid | `{0.001, 0.003}` |
| Original critic multiplier | `10` |
| Server LR / clip | `1.5 / 1.0` |
| Final seeds/rounds | `0-4`, `500` rounds |

`batch_size=0` is not one pooled global batch. Each round loops over all 1000
clients, and every client takes three full-local-batch GMM steps. The current
simulator therefore performs 3000 serial client GMM steps per round, plus full
train/validation evaluation.

For FedGDA-D, the server update is:

```text
delta_t = average_local_theta - theta_t
theta_(t+1) = theta_t + 1.5 * delta_t
```

FedOGDA-D uses OGDA locally and optimistic server deltas:

```text
local: theta <- theta - lr * (2 grad_t - grad_(t-1))
round 0: theta_(t+1) = theta_t + 1.5 * delta_t
later:   theta_(t+1) = theta_t + 1.5 * (2 delta_t - delta_(t-1))
```

Local OGDA history spans the three local steps but is cleared at the beginning
of each client/round call. Server delta history crosses rounds but restarts when
a run is resumed. Aggregation is sample-size weighted. Gradients are clipped to
norm `1.0` before the SGD or OGDA optimizer update.

Nine canonical original tuning candidates completed, all at alpha `0.5`:

| Scenario/method/LR | Best val | Best round | Final val | Runtime min |
|---|---:|---:|---:|---:|
| `femnist_x` FedGDA-D 0.001 | 0.180548 | 15 | 0.295779 | 161.7 |
| `femnist_x` FedGDA-D 0.003 | 0.180558 | 10 | 0.739076 | 350.9 |
| `femnist_x` FedOGDA-D 0.001 | 0.175204 | 28 | 0.319942 | 304.1 |
| `femnist_x` FedOGDA-D 0.003 | 0.178901 | 16 | 0.563882 | 300.2 |
| `femnist_xz` FedGDA-D 0.001 | 0.204649 | 13 | 0.458553 | 288.7 |
| `femnist_z` FedGDA-D 0.001 | 0.235324 | 30 | 0.397179 | 47.7 |
| `femnist_z` FedGDA-D 0.003 | 0.236051 | 10 | 0.248899 | 50.7 |
| `femnist_z` FedOGDA-D 0.001 | 0.235336 | 30 | 0.318233 | 68.0 |
| `femnist_z` FedOGDA-D 0.003 | 0.233017 | 133 | 0.234812 | 59.1 |

Constant-predictor validation MSE is about `0.1824` for image-`x/xz` scenarios
and `0.2352` for scalar-`x` (`*_z`) scenarios. The best of the nine improved
only `3.9%` over its constant baseline and four were worse. This made the
projected `~1600 GPU-hour` final campaign scientifically unjustified.

Additional deterministic review caveats:

- the client-remainder artifact is especially consequential under full
  participation and sample-size weighting;
- image CNNs contain `BatchNorm1d(10)`, and BatchNorm buffers are aggregated
  with tiny unequal local batches, a plausible full-participation failure mode;
- OGDA local history resets each client call and server history restarts on
  resume;
- the original manifests implicitly used legacy objective mode and sample-size
  weighting;
- the campaign has 10k validation/test samples, while an older paper registry
  lists 20k/20k;
- a constant-predictor column should accompany future tables.

## 12. Deterministic Auxiliary Equivalence

Before the learning gate, auxiliary regression was enabled and disabled in
paired ten-round runs for both methods. The checker compared:

- complete per-round curves;
- final `g` and `f` state dictionaries;
- best-validation `g` and `f` checkpoints;
- selection metrics.

All comparisons were exactly equal, not merely close:

```text
FedGDA-D:  passed exact equality
FedOGDA-D: passed exact equality
```

For deterministic runs, disabling auxiliary regression is therefore approved
by this evidence. It produced little runtime improvement because the 1000-client
GMM loop dominates. The four short runs used `0.279 GPU-hours` total.

Paths:

```text
scripts/check_highdim_deterministic_aux_equivalence.py
experiments/highdim_coauthor_protocol_v1/deterministic_learning_gate_20260802/equivalence_report.json
results/highdim_deterministic_learning_gate_20260802/equivalence/
```

## 13. Deterministic Learning-Gate Results

The gate proved that deterministic training can learn and that the original LR
grid was the principal configuration failure.

| Method | LR | Critic mult | Best val | Best round | Final val | Non-finite metric rows |
|---|---:|---:|---:|---:|---:|---:|
| FedGDA-D | 0.001 | 1 | 0.235432 | 30 | 11.550877 | 0 |
| FedGDA-D | 0.001 | 3 | 0.235372 | 30 | 2.081153 | 0 |
| FedGDA-D | 0.001 | 10 | 0.235324 | 30 | 0.397179 | 0 |
| FedGDA-D | 0.003 | 1 | 0.237484 | 9 | 0.248934 | 0 |
| FedGDA-D | 0.003 | 3 | 0.235347 | 78 | 0.254101 | 0 |
| FedGDA-D | 0.003 | 10 | 0.236051 | 10 | 0.248899 | 0 |
| FedGDA-D | 0.010 | 1 | 0.236187 | 149 | 0.236187 | 0 |
| FedGDA-D | 0.010 | 3 | 0.175857 | 148 | 0.178783 | 0 |
| FedGDA-D | 0.010 | 10 | 0.146297 | 149 | 0.146297 | 0 |
| FedGDA-D | 0.030 | 1 | 0.093874 | 149 | 0.093874 | 0 |
| FedGDA-D | 0.030 | 3 | **0.046502** | 131 | 0.062915 | 0 |
| FedGDA-D | 0.030 | 10 | 0.054882 | 146 | 0.056160 | 0 |
| FedOGDA-D | 0.001 | 1 | 0.235419 | 30 | 8.618582 | 0 |
| FedOGDA-D | 0.001 | 3 | 0.235350 | 30 | 0.722683 | 0 |
| FedOGDA-D | 0.001 | 10 | 0.235336 | 30 | 0.318233 | 0 |
| FedOGDA-D | 0.003 | 1 | 0.237349 | 9 | 0.316115 | 0 |
| FedOGDA-D | 0.003 | 3 | 0.237341 | 9 | 0.267623 | 0 |
| FedOGDA-D | 0.003 | 10 | 0.233017 | 133 | 0.234812 | 0 |
| FedOGDA-D | 0.010 | 1 | 0.194386 | 149 | 0.194386 | 0 |
| FedOGDA-D | 0.010 | 3 | 0.170699 | 148 | 0.212318 | 0 |
| FedOGDA-D | 0.010 | 10 | 0.164020 | 147 | 0.169310 | 3 |
| FedOGDA-D | 0.030 | 1 | **0.090483** | 136 | 1.214808 | 0 |
| FedOGDA-D | 0.030 | 3 | 0.065564 | 138 | 0.269225 | 3 |
| FedOGDA-D | 0.030 | 10 | 0.215833 | 140 | 0.423618 | 33 |

The nominal OGDA validation winner, `0.03/cm3`, is not accepted as clean
because it has three non-finite objective/moment rows. The clean provisional
OGDA candidate is `0.03/cm1`; `0.01/cm3` is a less accurate but milder-tail
fallback. FedGDA-D `0.03/cm3` is the strongest clean gate candidate.

Paths:

```text
scripts/prepare_highdim_deterministic_learning_gate.py
scripts/launch_highdim_deterministic_learning_gate.sh
experiments/highdim_coauthor_protocol_v1/deterministic_learning_gate_20260802/
results/highdim_deterministic_learning_gate_20260802/gate/
```

## 14. Non-Finite Metric Logging Fix

The gate launcher originally reported all 24 as passed because model parameters
remained finite. Raw curves exposed non-finite moment/objective values. The
current uncommitted fix adds `metric_values_are_finite` and changes per-round
`finite`/`diverged` to require both finite model state and finite metrics.

Modified files:

```text
fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py
fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py
tests/test_experiment_utils.py
```

This changes failure detection and logging, not optimization updates. The
current verification ran `92` related standard-library unit tests successfully.
These changes are still uncommitted and must be preserved before any new run.

## 15. Deterministic Multi-Seed Confirmation

Seeds `1` and `2` were run for three candidates and combined with gate seed `0`.
All six new runs completed 150 rounds with the corrected logger, no divergence,
and no non-finite metric rows.

| Candidate | Seed-0/1/2 best-val mean +/- sample std | Interpretation |
|---|---:|---|
| FedGDA-D `lr=.03`, `cm=3` | **0.06238 +/- 0.01446** | Strongest clean candidate |
| FedOGDA-D `lr=.03`, `cm=1` | **0.09778 +/- 0.00645** | Consistent best checkpoint, unstable final iterate |
| FedOGDA-D `lr=.01`, `cm=3` | 0.18599 +/- 0.01342 | Less accurate, milder late deterioration |

Per-seed detail:

| Candidate | Seed | Best val | Best round | Final val | Test at best val | Final test |
|---|---:|---:|---:|---:|---:|---:|
| GDA `.03/cm3` | 0 | 0.046502 | 131 | 0.062915 | 0.046851 | 0.062066 |
| GDA `.03/cm3` | 1 | 0.065864 | 148 | 0.073395 | 0.066111 | 0.073602 |
| GDA `.03/cm3` | 2 | 0.074785 | 145 | 0.084243 | 0.075351 | 0.084424 |
| OGDA `.03/cm1` | 0 | 0.090483 | 136 | 1.214808 | 0.090115 | 1.170757 |
| OGDA `.03/cm1` | 1 | 0.100127 | 140 | 0.126620 | 0.101014 | 0.127699 |
| OGDA `.03/cm1` | 2 | 0.102725 | 139 | 0.382308 | 0.103970 | 0.372639 |
| OGDA `.01/cm3` | 0 | 0.170699 | 148 | 0.212318 | 0.171895 | 0.215230 |
| OGDA `.01/cm3` | 1 | 0.191459 | 143 | 0.205823 | 0.192951 | 0.205406 |
| OGDA `.01/cm3` | 2 | 0.195819 | 148 | 0.229227 | 0.197495 | 0.232521 |

These data show viability only for `femnist_z`, alpha `0.5`, over seeds `0-2`.
They do not validate transfer to image-`x/xz`, CIFAR-10, other alphas, 500
rounds, or all five final seeds.

Paths:

```text
scripts/prepare_highdim_deterministic_multiseed_validation.py
scripts/launch_highdim_deterministic_multiseed_validation.sh
experiments/highdim_coauthor_protocol_v1/deterministic_multiseed_validation_20260803/
results/highdim_deterministic_multiseed_validation_20260803/
```

## 16. Latest Proposed Deterministic Protocol V2

This is the latest discussed path, not an approved or generated production
manifest.

For every one of the 18 alpha/scenario cells, tune seed `0` for 150 rounds with
two candidates per method:

```text
FedGDA-D:
  lr=.01, critic_multiplier=3
  lr=.03, critic_multiplier=3

FedOGDA-D:
  lr=.01, critic_multiplier=3
  lr=.03, critic_multiplier=1
```

Keep fixed:

```text
1000/1000 clients
batch_size=0
3 local steps
auxiliary_regression=false
server_learning_rate=1.5
gradient_clip_norm=1.0
objective_mode=legacy
objective lambda_1=0.1
aggregation_weighting=sample_size
validation-only selection
corrected non-finite metric logging
```

The four alpha-0.5 `femnist_z` seed-0 candidates already exist in the learning
gate and can be reused. That leaves `68` revised seed-0 tuning runs.

Proposed safeguards:

1. Reject any candidate with non-finite model state or metric values.
2. Require meaningful improvement over the scenario's constant predictor.
3. Select using validation only; read Test MSE afterward.
4. If a winning candidate's best round is `>=140`, extend both candidates in
   that cell before fixing the selection because the 150-round boundary is
   active.
5. After the 36 cell/method selections are fixed, run all five seeds
   `{0,1,2,3,4}` for 500 rounds, giving `180` deterministic finals.

The answer to "are we doing all seeds?" is therefore:

- tuning uses only seed `0`;
- the completed seed `1,2` runs were a diagnostic confirmation;
- the actual final matrix still requires all five seeds `0-4`.

The revised grid is a protocol-v2 change informed by validation evidence. It is
not the original `{.001,.003}/cm10` design and must be documented as such.

## 17. Runtime Projection And Feasibility

Observed deterministic costs:

```text
old 9 candidates: 47.7-350.9 min/run, median 161.7 min
new femnist_z gate: about 50 min/run
new six-run confirmation: about 4.86 GPU-hours total
```

The latest rough projection for the 68 revised seed-0 tuning runs is
`180-220 GPU-hours`, or about `90-110` continuous hours if two independent
processes scale cleanly. Under a guaranteed `48 GPU-hour/week` quota this alone
is roughly four to five quota weeks. Image-`x/xz` runs are much slower than the
cheap `femnist_z` gate, so this estimate must be refreshed from an actual
representative transfer run before launch.

The original projection for 180 finals at 500 rounds is about `1600 GPU-hours`.
That is about `800` continuous two-GPU wall hours, roughly 33 days with perfect
availability, or about 33 quota weeks at 48 GPU-hours/week. Boundary extensions
would add cost.

Combined revised deterministic tuning and finals are therefore roughly
`1780-1820 GPU-hours`, or around `37-38` quota weeks. The single-process
1000-client simulator remains the dominant operational problem. The stochastic
speedups do not make this full deterministic matrix inexpensive.

Two independent jobs can use two H100s, but the deterministic path is strongly
CPU-orchestration heavy. Concurrency should be benchmarked for throughput and
host-memory pressure; requesting two GPUs atomically can also delay queue start.

## 18. Centralized Arm Status

The original plan included GDA, SGDA, and OAdam centralized references for the
six scenarios in the alpha-0.5 comparison, with five seeds, giving 90 finals
plus a nominal 36-run tuning grid.

No high-dimensional centralized manifest, pooled image-capable runner,
selection procedure, or launch result exists under
`experiments/highdim_coauthor_protocol_v1/`. The protocol summary records only
the planned count. The centralized arm is not launch-ready and must not be
reported as pending jobs that can simply be resumed.

## 19. Code Paths For Review

Core execution:

```text
scripts/run_manifest.py
fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py
fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/client.py
fedgmm/sp_decentralized_mnist_lr_example/fedml/ml/trainer/my_model_trainer_classification.py
```

Models and optimizer:

```text
fedgmm/sp_decentralized_mnist_lr_example/models/mlp_model.py
fedgmm/sp_decentralized_mnist_lr_example/models/cnn_models.py
fedgmm/sp_decentralized_mnist_lr_example/optimizers/ogda.py
```

Data and partitioning:

```text
fedgmm/sp_decentralized_mnist_lr_example/fedml/data/data_loader.py
fedgmm/sp_decentralized_mnist_lr_example/fedml/data/MNIST/data_loader.py
fedgmm/sp_decentralized_mnist_lr_example/fedml/data/cifar10/efficient_loader.py
fedgmm/sp_decentralized_mnist_lr_example/fedml/model/model_hub.py
```

Experiment utilities and objective:

```text
fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py
fedgmm/sp_decentralized_mnist_lr_example/game_objectives/simple_moment_objective.py
```

Stochastic stability probe:

```text
scripts/prepare_stability_probe_v1.py
experiments/highdim_coauthor_protocol_v1/stability_probe_v1_20260722/
results/stability_probe_v1_20260722/
```

## 20. Current Worktree State

The following high-dimensional work is currently modified or untracked and
must not be discarded:

```text
M  fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py
M  fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py
M  tests/test_experiment_utils.py
?? experiments/highdim_coauthor_protocol_v1/deterministic_learning_gate_20260802/
?? experiments/highdim_coauthor_protocol_v1/deterministic_multiseed_validation_20260803/
?? experiments/highdim_coauthor_protocol_v1/highdim_remaining_prelaunch_review.md
?? scripts/check_highdim_deterministic_aux_equivalence.py
?? scripts/launch_highdim_deterministic_learning_gate.sh
?? scripts/launch_highdim_deterministic_multiseed_validation.sh
?? scripts/prepare_highdim_deterministic_learning_gate.py
?? scripts/prepare_highdim_deterministic_multiseed_validation.py
```

This handoff file is also newly added. The result directories are scientific
artifacts even when ignored by git. Preserve `results/_golden` and archive any
future failed/superseded run under `results/_failed/<timestamp>/` rather than
deleting or overwriting it.

## 21. Decisions Still Requiring Review

1. Approve or revise the proposed deterministic protocol-v2 candidate grid.
2. Decide whether validation evidence from only `femnist_z` is enough to launch
   all scenario cells, or require a small image-`x/xz` transfer gate first.
3. Decide whether to continue the legacy objective for comparability or run a
   separate paper-aligned objective campaign. Do not mix objective modes within
   one table.
4. Approve continued sample-size weighting with explicit disclosure of the
   last-client remainder artifact, or define a separate equal-client campaign.
5. Decide how to address BatchNorm under 1000 tiny full-participation clients.
   Changing it would be a scientific protocol change, not a runtime-only fix.
6. Approve the boundary-extension rule for 150-round tuning winners.
7. Decide whether the 500-round, five-seed deterministic matrix is affordable
   in the current simulator and quota, or whether implementation work/resources
   must precede it.
8. Design and smoke-test the centralized high-dimensional arm separately.

## 22. Safe Next Actions

Before any production launch:

1. Preserve or commit the current non-finite logging fix, deterministic scripts,
   manifests, and this handoff.
2. Obtain reviewer approval for the protocol-v2 grid and the legacy
   objective/sample-size-weighting continuity choices.
3. Materialize a new deterministic-v2 manifest in a new campaign directory;
   never edit the old manifests or overwrite old results in place.
4. Reuse only the four exact alpha-0.5 `femnist_z` gate configurations that
   match the approved v2 rows.
5. Dry-run the complete manifest and verify validation-only metadata,
   `auxiliary_regression=false`, corrected finiteness checks, output roots, and
   expected cartesian coverage.
6. Benchmark at least one CNN-`g`/CNN-`f` v2 row before estimating or launching
   the whole queue.
7. Launch independent processes per GPU only after the concurrency benchmark,
   with `--resume-skip-completed`, `--keep-going`, explicit output roots, and a
   results JSON.
8. Audit raw artifacts after every quota slice; do not infer completion from a
   wrapper return code.

## 23. Verification Commands

Scheduler:

```bash
gpurun --status
```

Canonical stochastic final audit:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python \
  scripts/audit_highdim_stochastic_finals.py
```

Original protocol tuning status, which counts the 72 stochastic candidates and
the nine old deterministic candidates but not the later gate:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python \
  scripts/analyze_highdim_coauthor_tuning.py --status-only
```

The canonical stochastic audit should report 180 valid finals and 36 aggregate
rows. The original tuning analyzer should currently show 24 stochastic rows
complete in each alpha block and only nine canonical deterministic rows at
alpha 0.5. The deterministic gate and multi-seed campaigns are intentionally
separate from that original analyzer.

Current focused unit-test command (`92` tests passed on 5 August 2026):

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python -m unittest \
  tests/test_experiment_utils.py \
  tests/test_highdim_coauthor_protocol.py \
  tests/test_paper_aligned_objective.py \
  tests/test_real_image_abs_manifest.py \
  tests/test_run_manifest_seed_fields.py
```

`pytest` is not installed in the `fedgmm` environment; use the command above
unless the environment is intentionally changed.

## 24. Bottom Line

The high-dimensional stochastic campaign is complete, preserved, auditable,
and already integrated into the paper source. Its proper scientific statement
is validation-selected checkpoint performance, not last-iterate convergence.
It does not show a monotone heterogeneity trend or broad OGDA dominance, and it
does show serious last-iterate instability for both methods plus transient NaN
metrics unique to FedOGDA-S in 29/90 runs.

The deterministic campaign is not complete. The learning gate rescued it from
an inadequate low-LR grid and established that deterministic learning is viable
on one cheap cell. The multi-seed check supports FedGDA-D `.03/cm3` and a clean
FedOGDA-D `.03/cm1` checkpoint candidate there, but also shows that OGDA's last
iterate remains unstable. Scaling those findings to all scenarios and all five
seeds is still a proposed, expensive protocol-v2 campaign. No such production
queue is currently running.
