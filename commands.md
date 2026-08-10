# Fork Migration and Cleanup Commands

Use this document as the task specification for the coding agent working on
the fork of the Federated DeepGMM repository. Execute the phases in order. Do
not skip validation gates or delete files merely because their purpose is not
immediately obvious.

## Objective

Bring the fork to functional parity with the validated implementation in
`fedgmm/sp_decentralized_mnist_lr_example/` by:

1. Porting exact two-phase `fed_eg` and forward-only `fed_zo_eg` support.
2. Porting configurable, persistent multi-GPU client multiprocessing.
3. Preserving the mathematical behavior of FedGDA/SGD, FedOGDA, FedEG, and
   FedZO-EG.
4. Verifying SP and MP correctness on toy and image datasets.
5. Inventorying, archiving, and removing genuinely unused files without
   deleting training inputs or experiment results.
6. Updating `AGENTS.md` and call-path documentation after the executable path
   is verified.

## Non-negotiable safety rules

- Start with read-only inspection. Do not delete or overwrite anything during
  the pull, dependency audit, or comparison phases.
- Preserve all existing user changes and untracked experiment artifacts.
- Before every deletion, provide an explicit path list, dependency evidence,
  total size, and recovery/archive plan, then obtain user approval.
- Archive CSV and compact NPY results separately before cleanup. Never treat
  generated NPZ datasets as result arrays.
- Never recreate raw image-sized plotting arrays for CIFAR/FEMNIST X or XZ.
  Save the compact scenario coordinate `w`, predictions, and ground truth.
- Do not terminate or share GPUs occupied by another user without explicit
  permission.
- After every `AGENTS.md` edit, show its diff and ask the user to approve it.
- Do not present the generic vendored MPI FedAvg implementation as this
  project’s multiprocessing implementation. The supported launcher remains
  `main.py` with `backend: sp`; client multiprocessing is internal to
  `FedAvgAPI`.

## Phase 1: Pull and establish the baseline

Run from the fork’s repository root:

```bash
git status --short --branch
git remote -v
git branch --show-current
git fetch --all --prune
git pull --ff-only
git status --short --branch
```

If `git pull --ff-only` cannot proceed because of local changes or divergent
history, stop and report the exact condition. Do not stash, reset, rebase, or
discard changes without the user’s approval.

Record repository size and largest paths before changing anything:

```bash
du -sh .
du -h --max-depth=3 . | sort -h | tail -n 100
find . -type f -printf '%s %p\n' | sort -n | tail -n 100
```

Locate instructions, launchers, configurations, datasets, imports, and result
artifacts:

```bash
find .. -name AGENTS.md -print
rg -n "FedAvgAPI|client_optimizer|fed_eg|fed_zo_eg|train_gmm_zo|OGDA|CustomSGD" .
rg -n "cifar10_(x|z|xz)|femnist_(x|z|xz)|mnist_(x|z|xz)" .
rg --files | rg '\.(csv|npy|npz|pt|pth|log|png|pdf)$'
```

Read every applicable `AGENTS.md` before editing. Confirm the supported
launcher and active nested experiment directory through imports and runtime
dispatch; do not assume the fork has the same layout.

## Phase 2: Produce a dependency and artifact inventory

Create an inventory document, for example
`.codex/experiment_file_inventory.md`, with these categories:

- Required launch and configuration files.
- Active server, client, trainer, model, objective, optimizer, scenario,
  loader, plotting, and model-selection dependencies.
- Dataset generators and generated NPZ training inputs.
- CIFAR/MNIST/FEMNIST source datasets used by generators or loaders.
- Federated and centralized result CSVs.
- Compact result NPYs.
- Checkpoints, plots, and logs.
- Reference-only code.
- Candidate removable files with proof that no active import, config, script,
  or documented workflow uses them.

Use import tracing and text searches, not filenames alone. Pay special
attention to eager imports in the nested FedML dispatcher: apparently unused
packages may still be required at module import time.

Do not remove root or nested `fedml/`, `models/`, dataset directories, CSVs,
NPYs, checkpoints, or archives during this phase.

## Phase 3: Port FedEG and FedZO-EG

Use the validated implementation in the source working repository as the
behavioral reference. Compare at least these files:

```text
fedml/simulation/sp/fedavg/fedavg_api.py
fedml/simulation/sp/fedavg/client.py
fedml/ml/trainer/my_model_trainer_classification.py
fedml/config/simulation_sp/fedml_config.yaml
optimizers/Customsgd.py
optimizers/ogda.py
optimizers/optimizer_factory.py
game_objectives/simple_moment_objective.py
```

Required optimizer names in YAML:

- `sgd`: FedGDA-style local simultaneous descent/ascent.
- `ogda`: local OGDA plus the existing server optimistic-delta branch.
- `fed_eg`: exact two-phase client correction.
- `fed_zo_eg`: exact predictor phase followed by SPSA forward-only correction.

Do not silently introduce aliases such as `fedgda` or `fedogda`. Unknown
optimizer values must not silently select a different algorithm; ideally add
explicit validation.

FedEG must retain this barrier sequence:

```text
sample clients once
  -> all predictor local updates from server state z_t
  -> weighted predictor aggregation
  -> construct global look-ahead state
  -> the same sampled clients run correction from the look-ahead
  -> weighted correction aggregation
  -> anchor the corrector update at z_t, not the look-ahead
```

`fed_zo_eg` must use the same phase structure, but its correction calls
`Client.train_zo()` / `ModelTrainerCLS.train_gmm_zo()`. Preserve:

- `zo_mu > 0`.
- `zo_num_directions >= 1`.
- Independent Rademacher directions for `g` and `f`.
- Two objective forward evaluations per direction.
- Restoration of unperturbed parameters before applying the update.
- Gradient-estimate averaging, clipping, and the configured learning rates.

Do not alter critic multipliers, gradient signs, clipping, server learning
rates, predictor/corrector anchoring, optimizer-state clearing, client sample
weights, or aggregation equations while porting.

Required YAML keys include:

```yaml
client_optimizer: fed_eg  # or fed_zo_eg
server_learning_rate: 1.5
eg_predictor_server_lr: null
eg_corrector_server_lr: null
zo_mu: 0.001
zo_num_directions: 1
```

## Phase 4: Port multi-GPU client multiprocessing

Port or recreate the validated worker module:

```text
fedml/simulation/sp/fedavg/multiprocess_client.py
```

Integrate it only at the independent client-update boundary in `FedAvgAPI`.
The coordinator must retain:

- Client sampling.
- Global state and sampled-client order.
- Weighted aggregation.
- Server learning-rate application.
- OGDA previous-delta history.
- FedEG/FedZO-EG predictor and corrector barriers.
- Evaluation, early stopping, CSV output, checkpoints, NPY output, and plots.

Each persistent spawned worker must:

1. Own one logical CUDA GPU.
2. Receive CPU batches and CPU global state dictionaries.
3. Move its task to its assigned GPU.
4. Call the unchanged `Client.train()`, `Client.train_reg()`, or
   `Client.train_zo()` method.
5. Detach and return CPU state dictionaries.

The coordinator must restore results to sampled-client order, move GMM states
to its device, and invoke the existing aggregators. Do not aggregate in
workers. Use the `spawn` multiprocessing context for CUDA safety. Keep workers
persistent across rounds, make shutdown idempotent, and avoid blocking forever
when a worker fails with a full task queue.

Configuration belongs under `train_args`:

```yaml
enable_multiprocessing: true
multiprocessing_num_workers: 4
multiprocessing_gpu_ids: [0, 1, 2, 3]
```

GPU IDs are logical indices visible to the Python process. Use one worker per
GPU. For three or two available GPUs, shorten both the list and count. With
fewer than two selected workers, log the reason and use the original SP path.
Also ensure `device_args.gpu_id` names an available coordinator GPU. Do not
start multiple CUDA workers on a single GPU merely to increase process count.

Keep phase-level timing logs for:

- SP primary client phase.
- MP primary client phase and worker count.
- SP correction client phase.
- MP correction client phase and worker count.

The first MP phase includes worker startup. Evaluate speed from later rounds.

## Phase 5: Correct compact image result serialization

Before running CIFAR/FEMNIST X or XZ, inspect final sorting and serialization.
Predictions must still be computed from image-valued `x`, but plotting and
saved curve coordinates must use the scalar scenario coordinate `w` whenever
`x` is image-shaped. Sort predictions and targets with the same one-dimensional
index. Validate that the coordinate has one scalar per observation.

Expected output for 10,000 test observations:

```text
results_<dataset>_<optimizer>_x.npy             shape (10000, 1)
results_<dataset>_<optimizer>_y_pred*.npy       shape (10000, 1)
results_<dataset>_<optimizer>_y_true.npy        shape (10000, 1)
```

Each float64 file should be approximately 80 KB, not tens of gigabytes.

## Phase 6: Validation gates

First run syntax and focused tests:

```bash
cd fedgmm/sp_decentralized_mnist_lr_example
python -m compileall \
  main.py \
  fedml/simulation/sp/fedavg/fedavg_api.py \
  fedml/simulation/sp/fedavg/multiprocess_client.py \
  fedml/ml/trainer/my_model_trainer_classification.py \
  scenarios learning optimizers
pytest -q tests/test_multiprocess_client.py
```

If `pytest` is unavailable, report that dependency instead of claiming the
suite passed. A direct invocation of test functions is only a temporary
diagnostic, not a substitute for the documented pytest run.

Run matched SP and MP smoke configurations. Use copied immutable YAMLs and
unique run names. At minimum validate:

- `step` with `sgd`.
- `step` with `ogda`.
- `step` with `fed_eg`.
- `step` with `fed_zo_eg`.
- One CIFAR X/XZ case.
- One FEMNIST X/XZ case.
- A four-worker launch on GPUs `[0, 1, 2, 3]` when those GPUs are free.
- A one-GPU configuration that logs and takes the SP fallback.

For deterministic methods, compare final MSE and prediction arrays. FedGDA,
FedOGDA, and FedEG should be bit-for-bit equal under matched deterministic
conditions. For FedZO-EG, verify successful two-phase execution, finite
metrics, seed behavior, and comparable distributions; do not demand bitwise
equality across worker schedules.

Before using GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
  --format=csv,noheader
```

Do not launch MP workers on GPUs occupied by another user.

Benchmark at least five rounds. Exclude the first MP round when estimating
steady-state performance because it includes worker startup. Report model
selection, first-round, median steady-round, evaluation, and total wall time.
Estimate 1,000 rounds with:

```text
SP = model_selection + 1000 * median_steady_sp_round
MP = model_selection + worker_startup + 1000 * median_steady_mp_round
```

Do not promise a speedup if local clients are too small for transfer overhead
to be amortized.

## Phase 7: Archive results before cleanup

Create a dated archive outside generated dataset directories. Keep categories
separate:

```text
.codex/results_archive/<date>/csv/
.codex/results_archive/<date>/npy/
.codex/results_archive/<date>/configs/
.codex/results_archive/<date>/plots/
.codex/results_archive/<date>/checksums/
```

Copy, do not move, existing CSVs and compact NPYs first. Record SHA-256 checks:

```bash
find csv -type f -name '*.csv' -print0 | sort -z | xargs -0 sha256sum
find . -maxdepth 1 -type f -name 'results_*.npy' -print0 | \
  sort -z | xargs -0 sha256sum
```

Inspect NPY shape/dtype with memory mapping before classifying it. Never load a
multi-gigabyte array fully into RAM merely to inspect its header.

## Phase 8: Safe cleanup procedure

Prepare, but do not execute, a cleanup proposal containing:

- Exact path.
- Tracked/untracked/ignored status.
- Size.
- Why it is unused.
- Searches/import traces proving it is unused.
- Whether it is archived or reproducible.
- Exact proposed removal command.

Potential categories include duplicate dataset caches, obsolete installers,
stale logs, Python bytecode caches, superseded raw image result arrays, and
framework subsystems unreachable after lazy-import refactoring. A category is
not automatically safe; validate every concrete path.

Before deleting dataset files, confirm they are not shared by any of:

- `cifar10_x`, `cifar10_z`, `cifar10_xz`.
- `mnist_x`, `mnist_z`, `mnist_xz`.
- `femnist_x`, `femnist_z`, `femnist_xz`.
- Dataset generators.
- Loader fallbacks or download/extraction paths.

Do not remove the nested FedML tree merely because only a few experiment files
are edited. First make eager dispatcher imports lazy and then run the supported
SP/MP DeepGMM smoke suite. Retain generic MPI/NCCL files only if the fork still
intends to support those paths; document the decision.

After the user approves the exact removal list, delete only those explicit
paths. Re-run imports and smoke tests after each logical cleanup batch instead
of deleting everything at once.

## Phase 9: Final documentation and handoff

Update `AGENTS.md` with the verified launcher, algorithms, multiprocessing
configuration, compact result contract, validation commands, and cleanup
constraints. Update `.codex/fedavg_api_call_tree.md` whenever executable
dispatch changes. Show the complete `AGENTS.md` diff and explicitly ask the
user to approve or request corrections.

The final report must include:

- Files added and modified.
- Algorithm invariants preserved.
- SP/MP equivalence results.
- Stochastic FedZO-EG validation results.
- GPU mapping used.
- First-round and steady-state timings.
- Image output shapes and sizes.
- Archived artifacts and checksum locations.
- Deleted paths and reclaimed disk space.
- Remaining limitations, especially serial model selection and centralized
  DeepGMM work.

## Centralized runs

Do not apply federated client multiprocessing to centralized DeepGMM. A
centralized run has no independent clients to distribute. Use batching and
data-loader workers first; consider DDP only after profiling proves a single
GPU is saturated. Keep centralized GDA/OAdam/SGDA work in a separate entry
point and result namespace.
