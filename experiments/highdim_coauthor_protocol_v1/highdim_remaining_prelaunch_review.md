# High-Dimensional Remaining Runs: Pre-Launch Review Protocol

Status: **draft for co-author/professor review; do not launch yet**  
Prepared: 2026-08-01  
Campaign: `highdim_coauthor_protocol_v1`

## 1. Purpose and scientific objective

The completed high-dimensional stochastic study evaluated `FedGDA-S` and
`FedOGDA-S` under partial participation and stochastic local updates. The next
proposed stage has two distinct objectives:

1. **Deterministic federated comparison.** Compare `FedGDA-D` and
   `FedOGDA-D` when every client participates and each client computes local
   full-batch updates. This isolates optimizer dynamics from minibatch noise
   and client-sampling noise while retaining heterogeneous client datasets and
   multiple local steps.
2. **Centralized reference comparison at alpha 0.5.** Compare the federated
   methods with centralized DeepGMM `GDA`, `SGDA`, and `OAdam` on the same
   saved high-dimensional scenarios. This requires a separate implementation
   and tuning protocol that is not yet materialized.

The main estimand is the structural function

```text
g0(x) = abs(x).
```

The primary reported metric is structural Test MSE at the checkpoint selected
by minimum validation structural MSE. Hyperparameters and checkpoints must be
selected without using Test MSE.

This campaign should be described as a controlled high-dimensional extension,
not an exact reproduction of every published-paper setting. The paper's
high-dimensional reference rows use `alpha=0.3`, and several optimization
details needed for exact reproduction are not specified in the paper. The
current campaign instead studies `alpha in {0.1, 0.5, 1.0}` under a fully
specified protocol.

## 2. Scenario matrix

All methods consume the same six certified saved scenario files.

| Scenario | Structural input `x` | Instrument `z` | Structural model `g` | Critic model `f` |
|---|---|---|---|---|
| `femnist_x` | FEMNIST image | scalar | EMNIST CNN | width-20 MLP |
| `femnist_z` | scalar | FEMNIST image | width-20 MLP | EMNIST CNN |
| `femnist_xz` | FEMNIST image | FEMNIST image | EMNIST CNN | EMNIST CNN |
| `cifar10_x` | CIFAR-10 image | scalar | CIFAR-10 CNN | width-20 MLP |
| `cifar10_z` | scalar | CIFAR-10 image | width-20 MLP | CIFAR-10 CNN |
| `cifar10_xz` | CIFAR-10 image | CIFAR-10 image | CIFAR-10 CNN | CIFAR-10 CNN |

`femnist` in the run names refers to the digits-only TensorFlow Federated
FEMNIST source used to construct the saved scenarios. It is not the older
torchvision EMNIST proxy present in some legacy source files.

### Cross-regime design

| Arm | Methods | Alphas | Participation | Local batch | Local steps | Final horizon | State |
|---|---|---|---:|---:|---:|---:|---|
| Federated stochastic | `FedGDA-S`, `FedOGDA-S` | `0.1, 0.5, 1.0` | `10/1000` (1%) | `256` | `3` | `1500` rounds | complete, 180/180 |
| Federated deterministic | `FedGDA-D`, `FedOGDA-D` | `0.1, 0.5, 1.0` | `1000/1000` (100%) | full local batch | `3` | `500` rounds | proposed, 0/180 finals |
| Centralized reference | `GDA`, `SGDA`, `OAdam` | alpha-0.5 comparison block | no clients | full batch for GDA; proposed `256` for SGDA/OAdam | to approve | to approve | not implemented |

`FedGDA-S` is the repository name for momentum-free federated stochastic GDA,
also referred to as `FedSGDA` in earlier experiment discussions. The
stochastic arm used learning-rate candidates `{0.003, 0.01}`; the proposed
deterministic arm uses `{0.001, 0.003}`.

## 3. Data-generating process

The scalar one-instrument AGMM process is

```text
C ~ Normal(0, 1)
Z ~ Uniform(-3, 3)
epsilon_x, epsilon_y ~ Normal(0, 0.1^2)
X = Z + C + epsilon_x
Y = abs(X) + 2 C + epsilon_y
```

When `X` or `Z` is represented by an image, the scalar value `v` is mapped to
a digit/class index

```text
q(v) = round(clip(1.5 v + 5, 0, 9)),
```

and an image with label `q(v)` is sampled from the corresponding source. For
an image-valued `X`, the structural target is evaluated at the quantized latent
value `(q(X)-5)/1.5`, so the saved ground-truth response remains auditable.

Fixed data properties:

| Property | Value |
|---|---:|
| Scenario-generation seed | `527` |
| Training examples | `20,000` |
| Validation examples | `10,000` |
| Test examples | `10,000` |
| FEMNIST source | TFF Federated EMNIST, `only_digits=True` |
| CIFAR source | `torchvision.datasets.CIFAR10` |
| FEMNIST image size | `1 x 28 x 28` |
| CIFAR-10 image size | `3 x 32 x 32` |
| Response normalization | Train-outcome mean and standard deviation |

`Y` and `g0(X)` are standardized using the training-outcome mean and standard
deviation. The saved image pools are content-disjoint across Train, Validation,
and Test. All six NPZ files passed shape, hash, finite-value, response-semantic,
and split-isolation certification. The source of truth is
`experiments/rerun_protocol_v1_real_images_abs_alpha0p5/data_generation_manifest.json`;
the certification report is
`experiments/rerun_protocol_v1_real_images_abs_alpha0p5/data_certification.md`.

## 4. Federated heterogeneity and seeds

For each run, Train, Validation, and Test are partitioned independently across
`N=1000` clients. For each split,

```text
p ~ Dirichlet(alpha * 1_N),
```

then a random permutation of the split is allocated according to `p`, with at
least five examples assigned to every client. This induces **quantity skew**,
not class-label skew. The values under study are

```text
alpha in {0.1, 0.5, 1.0}.
```

Lower alpha produces more unequal client sample counts. Run seeds
`{0,1,2,3,4}` control model initialization and the run-time random partition;
the underlying saved DGP and image-mapped examples remain fixed. In the
deterministic regime all clients participate, so there is no client-sampling
randomness after partitioning.

## 5. Model architecture and precision

The scalar model is a one-hidden-layer MLP:

```text
1 -> 20 -> 1, with LeakyReLU after the hidden layer.
```

The image models have two convolution/LeakyReLU/MaxPool blocks followed by
fully connected layers of widths `200`, `10`, and `1`. The EMNIST CNN uses
channels `[20, 50]`; the CIFAR-10 CNN uses channels `[32, 64]`. All models and
scenario tensors are trained in double precision (`torch.float64`).

The implementation also instantiates a separate auxiliary regression model.
That model is trained against `Y`, but its parameters are not used by the GMM
objective, structural predictions, validation checkpoint selection, or Test
MSE. Whether to disable this auxiliary path is a runtime decision discussed in
Section 12.

## 6. Objective actually implemented

The existing stochastic results and the nine completed deterministic tuning
runs use the implementation's legacy objective. With

```text
epsilon = g_theta(X) - Y,
M(theta, tau) = E[f_tau(Z) * epsilon],
R(theta, tau) = E[f_tau(Z)^2 * epsilon^2],
```

the structural player minimizes `M`, and the critic minimizes

```text
-M + lambda_1 R,  with lambda_1 = 0.1.
```

This legacy regularizer uses the **live** structural model. The code also now
supports a paper-aligned alternative that freezes the previous global
structural iterate inside the regularizer and fixes `lambda_1=1/4`. The current
high-dimensional manifests do not name this switch, so they resolve to
`objective_mode=legacy`.

**Proposed continuity choice:** explicitly set `objective_mode=legacy` for all
remaining runs so that deterministic and centralized results are comparable to
the completed stochastic results and the nine reusable deterministic tuning
artifacts.

**Required reviewer decision:** approve this as an implementation-continuity
experiment, or switch to the paper-aligned objective. Switching only the new
runs would invalidate direct comparisons with the completed stochastic table;
a paper-aligned campaign should therefore be treated as a separate experiment.

## 7. Aggregation actually implemented

The present high-dimensional manifests also omit the aggregation switch, so
the launcher resolves to legacy FedAvg sample-size weighting:

```text
weight_i = n_i / sum_j n_j.
```

This is what the completed stochastic runs and completed deterministic tuning
runs used. The code now also supports equal client weighting (`1/K`), which
matches a federated objective written as an unweighted average over clients.

**Proposed continuity choice:** explicitly set
`aggregation_weighting=sample_size` for this campaign. This preserves
comparability but must be disclosed because it gives larger clients more
influence under a quantity-skew partition.

## 8. Deterministic federated methods

### FedGDA-D

At communication round `t`, every client starts from the same global `g` and
`f`, performs three local full-client-batch GDA steps, and returns its local
models. The server forms the sample-size-weighted average and applies

```text
theta_(t+1) = theta_t + beta * (average_local_theta - theta_t),
beta = 1.5.
```

Both local optimizers are momentum-free SGD. The structural learning rate is
the selected `eta`; the critic learning rate is `10 eta`.

### FedOGDA-D

FedOGDA-D uses OGDA for both local players:

```text
theta <- theta - eta * (2 grad_t - grad_(t-1)).
```

Local OGDA history is cleared at the start of every client/communication-round
call, so the three local steps share optimistic history but that local history
does not cross communication rounds. The server separately retains the
previous aggregated model delta and applies

```text
round 0: theta_(t+1) = theta_t + beta * delta_t
later:   theta_(t+1) = theta_t + beta * (2 delta_t - delta_(t-1)),
beta = 1.5.
```

For both methods, the gradient norm for each player is clipped to `1.0` before
the optimizer step. This clips each current gradient, not the combined OGDA
extrapolation. No weight decay is applied to either GMM optimizer. The
manifest's deterministic `weight_decay=0.001` is a fixed compatibility field
only and must not be described as an applied regularizer.

## 9. Meaning of deterministic and local work

The deterministic configuration is

| Parameter | Value |
|---|---:|
| Total clients | `1000` |
| Clients per round | `1000` |
| Participation ratio | `100%` |
| Manifest batch size | `0` |
| Local epochs/steps | `3` |

`batch_size=0` does not form one global batch. The loader first partitions the
20,000 training examples, then combines each client's examples into one local
batch. Therefore each client performs three local GMM steps per communication
round. With 1000 clients, one round performs 3000 serial client GMM steps in
the current single-process simulator. A client has about 20 training examples
on average, although the Dirichlet allocation makes this highly unequal.

The implementation evaluates the global model on all 20,000 training examples
and all 10,000 validation examples after every communication round. Test MSE is
not evaluated per round; it is evaluated during finalization at the already
selected validation checkpoint and at the final iterate.

## 10. Validation-only tuning protocol

Tuning is independent for every
`(alpha, scenario, deterministic method)` cell.

| Tuning parameter | Value |
|---|---:|
| Tuning seed | `0` |
| Current tuning horizon | `150` communication rounds |
| Structural LR grid | `{0.001, 0.003}` |
| Critic LR grid | `{0.01, 0.03}` via fixed `10x` multiplier |
| Server LR | `1.5` |
| Gradient clip norm | `1.0` |
| Objective lambda | `0.1` in legacy mode |
| Effective weight decay | `0` for `g` and `f` |
| Candidates per cell | `2` |

The nominal deterministic tuning count is

```text
3 alphas x 6 scenarios x 2 methods x 2 learning rates = 72 runs.
```

Selection is performed after excluding invalid/non-finite candidates:

1. lowest `best_validation_mse`;
2. lower standard deviation of validation MSE over the last 50 rounds;
3. smaller `final_validation_mse - best_validation_mse`;
4. lower structural learning rate.

The selection code fixes every choice before reading
`test_mse_at_best_validation`. Test MSE may be reported for audit after the
configuration is selected, but it cannot alter the selection.

### Tuning-horizon concern

The 150-round tuning horizon needs explicit review. Earlier guidance was that
200 rounds may be insufficient and trends may settle only after roughly 500
rounds. In the nine completed candidates, one FedOGDA-D candidate reaches its
best validation value at round index `133`, close to the 150-round boundary.
Selecting a learning rate using 150 rounds and then running it for 500 rounds
can miss a later reversal in the candidate ranking.

Recommended gate: extend both learning-rate candidates to 500 rounds for a
small validation-only set containing the boundary case and the slowest CNN
case. If the selected learning rate changes or the best round remains near the
horizon, extend deterministic tuning to 500 rounds consistently. Otherwise,
retain the 150-round tuning rule and document the diagnostic.

## 11. Final deterministic protocol and reporting

After validation-only tuning, the selected learning rate for each
`(alpha, scenario, method)` cell is fixed and reused for all five seeds.

| Final-run parameter | Value |
|---|---:|
| Communication rounds | `500` |
| Seeds | `{0,1,2,3,4}` |
| Alphas | `{0.1,0.5,1.0}` |
| Scenarios | `6` |
| Methods | `FedGDA-D`, `FedOGDA-D` |
| Final federated runs | `180` |

Round 200 is a validation-curve diagnostic only. Runs should not be stopped or
selected using Test MSE. For each seed, the checkpoint is the minimum
validation-MSE checkpoint within that run. The primary result is

```text
mean(test_mse_at_best_validation) +/- sample standard deviation over 5 seeds.
```

The review should also require reporting:

- validation-selected round distribution;
- final-iterate Test MSE and final/best gap;
- last-50-round validation standard deviation and range;
- any non-finite metric rows or model states;
- per-run runtime and the effective configuration.

## 12. Runtime behavior and proposed optimization gate

Nine of the 72 deterministic tuning candidates are already complete and
preserved. They are all at `alpha=0.5`:

- all four `femnist_x` candidates;
- all four `femnist_z` candidates;
- `femnist_xz/FedGDA-D/lr=0.001`.

Their 150-round runtimes range from **47.7 minutes to 350.9 minutes**, with a
median of **161.7 minutes** and a combined cost of **27.2 GPU-hours**. The
remaining deterministic tuning count is `63`.

A naive projection using the observed median is about **170 GPU-hours for the
remaining tuning alone**. Scaling the same median approximately by `500/150`
would put the 180 final runs near **1600 GPU-hours**. These are rough upper-bound
projections from the old execution path, not reliable optimized estimates, but
they show that the unchanged queue is infeasible under the current weekly
budget of 48 GPU-hours. At review time, 44.9 GPU-hours remain and both H100s are
idle.

The completed deterministic runs predate several runtime fixes. In particular,
the old auxiliary regression loop performed `3 x 3 = 9` passes instead of the
intended three, and the old artifact path did more repeated serialization. The
GMM `g/f` trajectory is logically separate from the auxiliary regression
model, but reuse of these nine artifacts should still be supported by an
explicit equivalence audit.

### Required isolated benchmark before production launch

Use non-production output roots and profile representative full-participation
runs before launching the queue:

1. Compare the current baseline with `auxiliary_regression=false` on a short,
   deterministic seed-0 run. Require matching `g/f` validation trajectories
   and checkpoints within deterministic floating-point tolerance.
2. Profile at least one scalar-`g` case and one CNN-`g`/CNN-`f` case to measure
   client training, auxiliary regression, aggregation/state copying, full
   evaluation, and checkpoint I/O separately.
3. Compare one process on one GPU with two independent processes on the two
   GPUs. Full-participation runs are CPU-orchestration heavy, so two concurrent
   runs should be adopted only if total throughput improves without changing
   results or exhausting host memory.
4. Re-estimate the complete tuning and final campaign from measured optimized
   seconds per round before launch.

Candidate operational settings to validate are:

| Setting | Current default | Proposed production setting |
|---|---|---|
| Auxiliary regression | enabled, 3 passes | disable after equivalence proof |
| Per-round CSV | append | keep append |
| Legacy global outputs | disabled in generated configs | keep disabled |
| Periodic checkpoints | every 200 rounds | retain only if resume need justifies cost |
| DataLoader workers | `0` | profile; full-batch local tensors may not benefit |
| Pinned memory | `false` | profile only; do not assume a benefit |
| Model selection setup | enabled (`100/60`, batch `200`) | keep unless equivalence is established |

The current `finite` field in federated `mse_by_round.csv` checks model-state
finiteness but not the metric values in that row. Before future runs, either
fix it to include finite train/validation metrics or make the post-run audit
independently reject every non-finite metric row. This is a logging/audit issue,
not a reason to alter already reported best-validation values.

## 13. Centralized comparison: proposed scope, not launch-ready

The intended centralized comparison at `alpha=0.5` is

```text
6 scenarios x 3 methods x 5 seeds = 90 final runs.
```

The methods are:

| Label | Update regime | Proposed batch semantics |
|---|---|---|
| DeepGMM-GDA | deterministic GDA | pooled full batch |
| DeepGMM-SGDA | stochastic GDA | pooled minibatch, nominally 256 |
| DeepGMM-OAdam | optimistic Adam | pooled minibatch, nominally 256 |

Alpha has no direct meaning for a pooled centralized run; `alpha=0.5` denotes
the table/comparison block, not a centralized client partition. The centralized
runs should use the same saved Train/Validation/Test scenarios, the same
CNN/MLP mapping, double precision, objective mode, gradient clipping, and
validation-only checkpoint rule.

This block is **not currently launch-ready**:

- there is no high-dimensional centralized manifest or result tree;
- `scripts/run_centralized_lowdim.py` rejects the six image datasets and builds
  only MLP models, so it cannot be used unchanged;
- a centralized learning-rate grid has not been approved;
- the iteration budget and fair relationship to federated local work have not
  been approved;
- OAdam's optimizer settings, including its default betas `(0.5, 0.9)`, need to
  be frozen explicitly;
- the centralized tuning-run count is therefore not yet defined.

Recommended centralized design work after the federated decisions are fixed:

1. implement an image-capable centralized runner using the same model factory
   and saved scenario files as the federated runner;
2. define method-specific validation-only LR grids at seed 0;
3. define one common iteration-budget principle before observing Test MSE;
4. run smoke tests for all three model mappings (`x`, `z`, `xz`);
5. materialize tuning and final manifests with isolated result roots;
6. report the same validation-selected Test MSE and last-iterate diagnostics as
   the federated methods.

## 14. Current run accounting

| Block | Planned | Complete | Remaining | Launch readiness |
|---|---:|---:|---:|---|
| Stochastic tuning | `72` | `72` | `0` | complete |
| Stochastic federated finals | `180` | `180` | `0` | complete |
| Deterministic tuning | `72` | `9` | `63` | blocked on review/runtime gate |
| Deterministic federated finals | `180` | `0` | `180` | requires completed tuning |
| Centralized finals | `90` | `0` | `90` | design and implementation missing |
| Centralized tuning | not defined | `0` | not defined | design missing |

Thus, there are `333` concretely enumerated runs remaining, plus an undefined
number of centralized tuning runs.

## 15. Result preservation and artifact requirements

Existing result folders are scientific artifacts and must not be overwritten.
The launcher must use `--resume-skip-completed`; invalid or superseded partial
runs must be moved under `results/_failed/<timestamp>/` rather than deleted.
`results/_golden` must remain untouched.

Every accepted run must contain at least:

```text
effective_config.json
metrics.json
mse_by_round.csv
predictions.npz
checkpoints/best_validation.pt
checkpoints/final.pt
```

The effective configuration must explicitly record `objective_mode`,
`aggregation_weighting`, auxiliary-regression policy, learning rates, client
counts, local epochs, batch mode, round budget, alpha, seed, precision-relevant
settings, and output identity. The currently generated deterministic YAML files
are stale relative to these newer fields and must be regenerated after review;
their omission currently falls back to legacy defaults.

## 16. Decisions requested before launch

| Decision | Recommended choice for continuity | Consequence |
|---|---|---|
| Scientific framing | controlled extension, not exact paper reproduction | accurately reflects alpha grid and unspecified paper details |
| Objective | explicit `legacy`, `lambda_1=0.1` | comparable with completed stochastic runs; not the newer frozen-theta objective |
| Aggregation | explicit `sample_size` | comparable with existing runs; differs from uniform-client theoretical averaging |
| Tuning horizon | run 500-round boundary/slow-case diagnostic first | determines whether all tuning must be extended |
| Auxiliary regression | disable only after trajectory equivalence proof | likely large speedup without changing GMM outputs |
| Reuse nine completed candidates | reuse only after provenance/equivalence audit | preserves 27.2 GPU-hours of work |
| Parallel GPUs | adopt only after one-vs-two-process throughput benchmark | avoids CPU/memory contention masquerading as GPU speedup |
| Non-finite audit | fix metric finiteness or independently audit every row | prevents transient NaNs from being mislabeled finite |
| Centralized grid/budget | approve before implementation/launch | centralized count and runtime are otherwise undefined |

## 17. Proposed execution order after approval

1. Freeze the reviewed objective, aggregation, tuning-horizon, and reporting
   decisions in manifests and generated configs.
2. Run isolated deterministic runtime/equivalence benchmarks and produce a
   measured queue-time estimate.
3. Audit whether the nine completed deterministic candidates are reusable.
4. Complete deterministic seed-0 tuning without inspecting Test MSE.
5. Materialize selected deterministic final manifests and run seeds 0-4.
6. Audit artifacts and aggregate mean plus sample standard deviation.
7. Implement, smoke-test, tune, and run the separately approved centralized
   high-dimensional block.

No production run should begin until Sections 12 and 16 are resolved.

## 18. Primary code and protocol references

- Protocol generator: `scripts/prepare_highdim_coauthor_protocol.py`
- Tuning queue: `scripts/run_highdim_coauthor_tuning_queue.py`
- Validation selector: `scripts/analyze_highdim_coauthor_tuning.py`
- Final queue: `scripts/run_highdim_coauthor_final_queue.py`
- Manifest launcher: `scripts/run_manifest.py`
- Federated orchestration: `fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py`
- Local GMM training: `fedgmm/sp_decentralized_mnist_lr_example/fedml/ml/trainer/my_model_trainer_classification.py`
- Objective definitions: `fedgmm/sp_decentralized_mnist_lr_example/game_objectives/simple_moment_objective.py`
- Model mapping: `fedgmm/sp_decentralized_mnist_lr_example/fedml/model/model_hub.py`
- Partition loader: `fedgmm/sp_decentralized_mnist_lr_example/fedml/data/cifar10/efficient_loader.py`
- Certified data report: `experiments/rerun_protocol_v1_real_images_abs_alpha0p5/data_certification.md`
- Current campaign summary: `experiments/highdim_coauthor_protocol_v1/protocol_summary.json`
- Completed stochastic audit/writeup: `experiments/highdim_coauthor_protocol_v1/stochastic_writeup.md`
- Paper target registry: `experiments/reproduction_targets.csv`
