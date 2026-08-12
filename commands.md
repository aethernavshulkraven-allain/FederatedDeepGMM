# Updated-Fork Reconciliation, Multiprocessing, and Cleanup Guide

Use this as the task specification for the agent working on the updated fork.
The fork contains important work added after the validated multiprocessing
implementation, so reconcile it file by file rather than replacing it.

## Repositories and observed baseline

- Target to modify: `/mnt/disk1/geetika/FedDeepGMM`
- Validated reference: `/mnt/disk1/geetika/project`
- Target branch observed 2026-08-10: `experimentsrerun`
- Target HEAD observed 2026-08-10: `38436db` (`cleanup`)

These hashes are audit records, not instructions to reset. If the remote has
advanced, inspect and use its new clean HEAD.

## Objective

1. Understand and test the fork before editing it.
2. Preserve its eICU Study A work, centralized GDA/SGDA/OAdam baselines,
   manifests, gates, plots, provenance, and campaign tooling.
3. Port exact `fed_eg` and `fed_zo_eg` behavior where it is still missing.
4. Port persistent multi-GPU client execution without changing FedGDA,
   FedOGDA, FedEG, or FedZO-EG algorithm logic.
5. Preserve compact CIFAR/FEMNIST result serialization.
6. Build a new dependency inventory before proposing cleanup. Never reuse the
   previous repository's deletion list without revalidation.

## Safety rules

- Work in `/mnt/disk1/geetika/FedDeepGMM`; keep the reference read-only.
- Begin with a read-only audit. Do not copy, cherry-pick, edit, or delete until
  the compatibility report has been shown to the user.
- Never replace `fedavg_api.py`, `client.py`, the trainer, `main.py`, YAMLs, or
  centralized scripts wholesale. Reconcile behavior manually.
- Preserve tracked and untracked artifacts. Never stash, reset, clean, rebase,
  or discard changes without explicit approval.
- Do not cherry-pick cleanup commits from the reference repository.
- Preserve CSV/JSON reports, compact NPYs, manifests, resolved configs,
  validation contracts, plots, and provenance separately from generated data.
- Before deletion, show exact paths, sizes, Git status, dependency evidence,
  archive/recovery plan, and expected freed space; then obtain approval.
- After every `AGENTS.md` edit, show its complete diff and explicitly ask the
  user to approve it or request corrections.

## Phase 1: Audit the updated fork without modifying it

```bash
cd /mnt/disk1/geetika/FedDeepGMM
git status --short --branch
git remote -v
git branch --show-current
git log --oneline --decorate -n 30
find .. -name AGENTS.md -print
```

The observed checkout has no `AGENTS.md`: commit `38436db` removed it after
ignore rules for local instructions were added. Record this; do not silently
recreate or force-add an ignored instruction file.

If the worktree is clean:

```bash
git fetch --all --prune
git pull --ff-only
git status --short --branch
```

If fast-forward pull cannot proceed, report why and stop. Do not reset or
force-checkout. Inspect fork-specific work and disk usage:

```bash
git show --stat --summary 2c37c69
git show --stat --summary 38436db
rg --files scripts tests experiments | sort
rg -n "eICU|Study A|centralized|OAdam|SGDA|preflight|validation|manifest" \
  scripts tests experiments fedgmm/sp_decentralized_mnist_lr_example
du -sh .
du -h --max-depth=3 . | sort -h | tail -n 100
find . -type f -printf '%s %p\n' | sort -n | tail -n 100
```

### Required compatibility report

Produce this table with exact paths and evidence:

| Area | Updated fork | Validated reference | Decision |
|---|---|---|---|
| FL launcher/dispatch | call chain | call chain | keep/reconcile |
| FedGDA/SGD | code and labels | code | preserve/reconcile |
| FedOGDA | client/server state | code | preserve/reconcile |
| FedEG | present/missing | code | keep/port manually |
| FedZO-EG | present/missing | code | keep/port manually |
| Client multiprocessing | present/missing | code | keep/port manually |
| Image serialization | shapes/coordinate | compact behavior | preserve/fix |
| Centralized low-dimensional | runner/tests/results | reference | protect fork |
| eICU Study A | runner/tests/contracts | reference | protect fork |
| Dataset loaders/generators | dependencies | dependencies | reconcile |
| Cleanup candidates | fork evidence | old inventory only | re-audit |

Show this report and ask the user to review it before implementation.

## Phase 2: Compare algorithm-sensitive files

Use these roots for read-only diffs:

```text
TARGET=/mnt/disk1/geetika/FedDeepGMM/fedgmm/sp_decentralized_mnist_lr_example
REFERENCE=/mnt/disk1/geetika/project/fedgmm/sp_decentralized_mnist_lr_example
```

Compare at least:

```text
main.py
fedml/config/simulation_sp/fedml_config.yaml
fedml/simulation/sp/fedavg/fedavg_api.py
fedml/simulation/sp/fedavg/client.py
fedml/ml/trainer/my_model_trainer_classification.py
scenarios/abstract_scenario.py
game_objectives/simple_moment_objective.py
optimizers/Customsgd.py
optimizers/ogda.py
optimizers/oadam.py
optimizers/optimizer_factory.py
```

Protect and test the fork-owned centralized paths:

```text
scripts/run_centralized_lowdim.py
scripts/run_eicu_centralized_baselines.py
scripts/run_eicu_study_a_v2_centralized.py
scripts/validate_centralized_run.py
scripts/preflight_eicu_release.py
tests/test_run_centralized_lowdim_eicu.py
tests/test_run_centralized_lowdim_eicu_gate4.py
tests/test_run_eicu_centralized_baselines.py
```

Search symbols across both repositories; do not assume equal line numbers:

```bash
rg -n "fed_eg|fed_zo_eg|train_gmm_zo|train_zo|zo_mu|zo_num_directions" \
  /mnt/disk1/geetika/FedDeepGMM /mnt/disk1/geetika/project
rg -n "enable_multiprocessing|multiprocessing_num_workers|multiprocessing_gpu_ids|MultiprocessClient" \
  /mnt/disk1/geetika/FedDeepGMM /mnt/disk1/geetika/project
```

The initial audit found OGDA in the fork but no executable `fed_eg`,
`fed_zo_eg`, `train_gmm_zo`, or client-multiprocessing symbols. Reconfirm after
pulling. Inspect reference commits for intent, but never blindly cherry-pick or
copy an entire algorithm-sensitive file.

## Phase 3: Establish a protected baseline

Before editing, run the fork's focused tests and smallest documented
preflights. Discover exact flags from `--help`, tests, manifests, and reports.
Capture:

- Import/compile status for the active FL path.
- Low-dimensional centralized GDA, SGDA, and OAdam smoke status.
- eICU centralized/federated unit and preflight status.
- One existing federated toy smoke.
- Current output schemas and golden-smoke checksums.

Start with:

```bash
python -m compileall \
  fedgmm/sp_decentralized_mnist_lr_example/main.py \
  fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg \
  fedgmm/sp_decentralized_mnist_lr_example/fedml/ml/trainer/my_model_trainer_classification.py \
  scripts tests
pytest -q \
  tests/test_run_centralized_lowdim_eicu.py \
  tests/test_run_centralized_lowdim_eicu_gate4.py \
  tests/test_run_eicu_centralized_baselines.py
```

Add tests referenced by the fork's reports. If a dependency is unavailable,
report it; do not claim an incomplete suite passed.

## Phase 4: Reconcile FedEG and FedZO-EG

Port focused behavior into current fork files while preserving fork logic.
Explicit YAML values should be `sgd`, `ogda`, `fed_eg`, and `fed_zo_eg`.
Unknown values should fail instead of silently falling back to SGD.

Preserve this barrier sequence:

```text
sample clients once
  -> all predictor updates from server state z_t
  -> weighted predictor aggregation and look-ahead
  -> same clients correct from the look-ahead
  -> weighted correction aggregation anchored at z_t
```

Do not alter critic signs or learning-rate multipliers, clipping, sample
weights, server learning rates, optimizer state, or aggregation. For FedZO-EG,
preserve positive `zo_mu`, one or more directions, independent Rademacher
directions for `g` and `f`, two forward evaluations per direction, parameter
restoration, and estimate averaging. Use copied immutable YAMLs and unique
result namespaces.

## Phase 5: Reconcile persistent multi-GPU client execution

Add validated worker behavior to the active nested SP path, normally via:

```text
fedml/simulation/sp/fedavg/multiprocess_client.py
fedml/simulation/sp/fedavg/fedavg_api.py
```

Keep `backend: sp`. Parallelize only independent client-local updates. The
coordinator retains sampling, order, aggregation, server learning rates, OGDA
history, EG barriers, evaluation, checkpointing, and output generation.

Use persistent spawned processes, one per logical GPU. Workers receive CPU
batches/states, move a task to their GPU, invoke unchanged client methods, and
return detached CPU states. Restore results to sampled-client order.

```yaml
enable_multiprocessing: true
multiprocessing_num_workers: 4
multiprocessing_gpu_ids: [0, 1, 2, 3]
```

Use only assigned GPUs. Fewer than two workers must log and fall back to the
serial path. Do not apply this client pool to centralized runs; centralized
multi-GPU/DDP is a separate, profiled project.


### Phase 5A: Add true multiprocessing within one GPU

After the Phase 1 through 3 review gate and while reconciling Phase 5, add the
validated same-GPU process mode. Do not port the separate thread/CUDA-stream
executor or describe stream concurrency as multiprocessing.

First make process-worker determinism explicit. Every spawned CUDA worker must
set deterministic cuDNN behavior equivalent to the coordinator:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Preserve input task order when collecting results. For deterministic methods,
the later validation must compare every returned `g` and `f` checkpoint tensor
against SP with `torch.equal`, not only compare rounded metrics.

Add explicit `client_execution_mode` routing for:

```text
sp
multi_gpu_processes
multiprocessingsinglegpu
```

Retain backward compatibility with `enable_multiprocessing` only when
`client_execution_mode` is absent. Reject unknown explicit mode names. If the
selected worker count is less than two, log the reason and use the original SP
path.

Implement `multiprocessingsinglegpu` with persistent processes created through
the CUDA-safe `spawn` context. Reuse a common process executor where practical,
but deliberately assign the same logical CUDA GPU ID to every same-GPU worker.
Each worker must own an isolated trainer, optimizer state, CUDA context, and at
most one active client task. Cap the worker count at
`client_num_per_round`. Require the worker GPU ID to equal the coordinator GPU
ID and validate it against `torch.cuda.device_count()`.

Do not move algorithm-sensitive coordinator work into the executor. Sampling,
sampled-client order, weighted aggregation, server learning rates, OGDA
previous-delta history, FedEG/FedZO-EG phase barriers, evaluation,
checkpointing, and result generation remain coordinator-owned.

At the end of each worker task, convert returned states to detached CPU data,
synchronize the assigned device, drop per-client references, and release
unused CUDA cache before publishing task completion. Log worker PIDs and their
logical GPU assignments. Keep shutdown idempotent and preserve persistent
workers across communication rounds.

Before modifying the auxiliary direct-regression path, audit whether its model,
training, or aggregation contributes to any fork result: `g`, `f`, MSE,
objective metrics, CSV/JSON, checkpoints, NPY predictions, or plots. If it is
genuinely unused, remove its FedAvg client training and aggregation and let the
supported FedAvg trainer/model list contain only `g` and `f`, while preserving
the legacy third model for any other coordinator that still uses it. If the
fork uses regression anywhere, do not remove or serialize it differently;
report the dependency and reconcile the process task contract around it.

Before changing `CUDA_VISIBLE_DEVICES`, inspect launcher behavior and verify
whether a shell-supplied value is currently overwritten. Change it only if
that issue exists. The corrected launcher must establish environment defaults
before importing CUDA-aware libraries, preserve a user-supplied value, and
default to the repository's intended GPU list when the variable is absent.
Verify physical-to-logical remapping: when only physical GPU 2 is exposed with
`CUDA_VISIBLE_DEVICES=2`, Python sees it as logical `cuda:0`, so both the
coordinator and same-GPU worker configuration use GPU ID 0.

Do not require a dedicated YAML file merely for this mode. Instead, update the
fork's `README.md` with the exact YAML keys and executable commands for running
true same-GPU multiprocessing. Include at least one isolated physical-GPU
example and explain logical remapping, for example:

```yaml
train_args:
  client_execution_mode: multiprocessingsinglegpu
  enable_multiprocessing: false
  multiprocessingsinglegpu_num_workers: 2
  multiprocessingsinglegpu_gpu_id: 0

device_args:
  gpu_id: 0
```

```bash
cd fedgmm/sp_decentralized_mnist_lr_example
CUDA_VISIBLE_DEVICES=2 python main.py --cf <reviewed-config.yaml>
```

State clearly that `2` selects the physical GPU in the shell and `0` selects
the remapped logical device inside Python/YAML. Also document behavior when
`CUDA_VISIBLE_DEVICES` is omitted.

Once implementation and documentation are complete, explicitly remind the
user that these follow-up gates are ready but require their review/timing:

1. Run focused unit tests for routing, worker-count capping, GPU mismatch,
   deterministic cuDNN settings, ordered CPU results, synchronization, cache
   release, and safe shutdown. Use `python -m pytest` so the repository package
   wins over any site-installed FedML package.
2. Run deterministic SP-versus-`multiprocessingsinglegpu` equivalence on
   CIFAR10-X and compare all `g`/`f` tensors, metrics, predictions, and complete
   checkpoint hashes.
3. Validate only on an isolated GPU. Check `nvidia-smi` first and do not call an
   OOM a worker-capacity limit if unrelated processes occupy GPU memory.
4. When the user confirms a GPU is free, benchmark CIFAR10-X with 10 sampled
   clients, one communication round, two local epochs, batch size 256, and
   worker counts `2, 4, 6, 8`. Record peak memory, utilization, client-phase
   time, finite outputs, and the first isolated-GPU OOM; stop increasing after
   that genuine failure.

After all changes, update every affected document to match executable behavior,
including `README.md`, fork-local `AGENTS.md`, configuration guidance,
validation instructions, and the static call tree. Remove stale claims rather
than leaving contradictory modes or GPU mappings. Show the complete
`AGENTS.md` diff and explicitly ask the user to approve it or request
corrections.

## Phase 6: Preserve compact image outputs

For CIFAR/FEMNIST X and XZ, predictions use image-valued `x`, but saved curve
coordinates must use one scalar `w` or sample ID per observation. Apply one
1-D sort index to coordinate, prediction, and truth.

For 10,000 float64 observations, each compact array should be about 80 KB:

```text
results_<dataset>_<optimizer>_x.npy       shape (10000, 1)
results_<dataset>_<optimizer>_y_pred*.npy shape (10000, 1)
results_<dataset>_<optimizer>_y_true.npy  shape (10000, 1)
```

Inspect huge NPY candidates with memory mapping, not full loads.

## Phase 7: Regression and equivalence gates

After compile/unit tests, use copied configs and unique output names for:

- `step` with `sgd`, `ogda`, `fed_eg`, and `fed_zo_eg`.
- One CIFAR X/XZ and one FEMNIST X/XZ case.
- Four workers on `[0, 1, 2, 3]` when free.
- One worker proving serial fallback.
- All protected centralized/eICU regression gates.

Check GPUs first:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
  --format=csv,noheader
```

Matched deterministic SP/MP FedGDA, FedOGDA, and FedEG should produce the same
predictions and MSE. FedZO-EG is stochastic: check finite metrics, seed
behavior, and comparable distributions instead of demanding bitwise identity.

Benchmark at least five rounds. Exclude worker startup from steady-state timing
and report model-selection, first-round, median later-round, evaluation, and
total times. Do not promise a speedup before measurement.

## Phase 8: Build a new cleanup inventory

Only after all functionality passes, inventory:

- FL and centralized launchers and complete import/call dependencies.
- eICU inputs, contracts, manifests, provenance, gates, and reports.
- Scenarios, generators, source inputs, and generated NPZs.
- Shared CIFAR10 and FEMNIST/MNIST X/Z/XZ dependencies.
- CSV, JSON, NPY, plots, checkpoints, configs, logs, and checksums.
- Candidate removals with path-specific evidence.

Treat `experiments/`, `scripts/`, and `tests/` as active until their
reproduction relationships are disproved. The large research history added in
`2c37c69` may be intentional provenance and is not removable based on size.

Archive categories separately before cleanup:

```text
.codex/results_archive/<date>/csv/
.codex/results_archive/<date>/json/
.codex/results_archive/<date>/npy/
.codex/results_archive/<date>/configs/
.codex/results_archive/<date>/plots/
.codex/results_archive/<date>/checksums/
```

Do not remove nested FedML based on filenames. Eager imports can make unrelated
modules runtime dependencies. Show the keep/archive/remove map and ask for
approval. After approval, remove only listed paths in small batches and rerun
relevant FL, centralized, and eICU gates after each batch.

## Phase 9: Documentation and staged Git handoff

After behavior is verified:

1. Update fork-local instructions and the executable call tree.
2. Show each `AGENTS.md` diff and ask for review.
3. Separate implementation, tests/configs, docs, and cleanup commits.
4. Inspect staged contents and large files before each commit.
5. Push only after the user approves the sequence.

The final report must list modified files, preserved fork features, algorithm
invariants, tests, SP/MP equivalence, GPU mapping, timings, output shapes/sizes,
archives, deleted paths, reclaimed space, and remaining limitations.

## Immediate stop point

In the next agent session, perform only Phases 1 through 3. Deliver the
compatibility report and baseline test evidence, then ask the user to review.
Do not port code or delete files until that review is complete.
