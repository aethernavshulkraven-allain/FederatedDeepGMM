# Repository Guidelines

## Scope and Supported Entry Point

The working experiment is `fedgmm/sp_decentralized_mnist_lr_example/`. Run all Federated DeepGMM experiments from that directory through `main.py`; it calls `fedml.init()`, loads the configured data and model, constructs `FedMLRunner`, and dispatches the SP-backend coordinator to `fedml/simulation/sp/fedavg/fedavg_api.py::FedAvgAPI`. That coordinator can execute client-local work serially, through spawned processes across multiple GPUs or within one GPU, or concurrently on CUDA streams within one GPU.

```bash
cd fedgmm/sp_decentralized_mnist_lr_example
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cf fedml/config/simulation_sp/fedml_config.yaml
```

For a detached run, add `nohup` and redirect stdout/stderr to a distinct log. `main.py` preserves a shell-supplied `CUDA_VISIBLE_DEVICES` and otherwise defaults to `0,1,2,3`. `Toy_Example/` is an independent two-variable FedGDA/FedOGDA demonstration, not the DeepGMM execution path.

## Project Structure

- `main.py`: supported FL launcher.
- `fedml/config/simulation_sp/fedml_config.yaml`: active experiment configuration. Use copied YAML files for simultaneous or comparison runs.
- `fedml/simulation/sp/fedavg/fedavg_api.py`: server orchestration, client sampling, aggregation, model selection, evaluation, CSV logging, and checkpoints.
- `fedml/simulation/sp/fedavg/multiprocess_client.py`: persistent spawned GPU workers for multi-GPU and single-GPU multiprocessing.
- `fedml/simulation/sp/fedavg/single_gpu_client.py`: isolated trainer slots and CUDA streams for concurrent client-local updates within one process and GPU.
- `fedml/simulation/sp/fedavg/client.py` and `fedml/ml/trainer/my_model_trainer_classification.py`: client dataset assignment and local DeepGMM updates.
- `scenarios/`: data-generating processes for low-dimensional, MNIST, EMNIST, and CIFAR variants.
- `generate_zoo_data.py`, `generate_mnist_data.py`, `generate_emnist_data.py`, and `generate_cifar10_data.py`: materialize scenario data under `data/`.
- `models/`, `game_objectives/`, `model_selection/`, `learning/`, and `optimizers/`: DeepGMM models, moment objective, selection/evaluation logic, centralized-style learning primitives, and optimizer implementations.
- `csv/`, `checkpoints/`, plots, `.npy` files, and logs: generated experiment artifacts.
- `.codex/fedavg_api_call_tree.md`: static reference for the active FL call path; keep it synchronized with executable path changes.

The obsolete root-level `fedml/` and `models/` fragments were removed after
import-resolution verification. All supported imports must resolve from the
nested experiment implementation. Avoid broad changes inside that vendored
FedML tree when a focused experiment-level change is sufficient.

## Dataset Generation Workflow

Dataset behavior originates in `scenarios/`. `AbstractScenario.setup()` creates train/dev/test splits, `Standardizer` normalizes `y` and the true structural function `g`, and `AbstractScenario.to_file()` writes an NPZ archive. The FedML synthetic loader later reads these artifacts and partitions the training data among configured clients.

Run generators from the experiment directory:

```bash
python generate_zoo_data.py
python generate_mnist_data.py
python generate_emnist_data.py
python generate_cifar10_data.py
```

`generate_zoo_data.py` creates `linear`, `abs`, `sin`, and `step` using `AGMMZoo` with seed 527 and writes `data/zoo/<function>.npz`. The image generators create X, Z, and XZ variants under `data/<dataset>/main.npz`; their current structural function is `abs`. MNIST, EMNIST, or CIFAR source data must already be available where its scenario expects it. To add or change a dataset, modify the relevant scenario first, update its generator, regenerate the artifact, and then select the exact dataset name in YAML. Do not hand-edit generated NPZ files.

## Federated DeepGMM Configuration and Algorithms

Keep `training_type: simulation`, `backend: sp`, and the federated optimizer expected by `SimulatorSingleProcess`. In YAML, select one active dataset block and verify at least `dataset`, `data_cache_dir`, `model`, `client_num_in_total`, `client_num_per_round`, `comm_round`, `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `client_optimizer`, and `server_learning_rate` where applicable. Low-dimensional names are `linear`, `abs`, `sin`, and `step`; high-dimensional names include `mnist_x`, `mnist_z`, `mnist_xz`, `femnist_x`, `femnist_z`, `femnist_xz`, `cifar10_x`, `cifar10_z`, and `cifar10_xz` when their data exists.

The active code recognizes these `client_optimizer` values:

- `sgd`: local simultaneous gradient descent/ascent using `CustomSGD`; this is the FedGDA-style baseline.
- `ogda`: local optimistic gradient descent/ascent using `OGDA`; this is the FedOGDA-style path and also activates the server OGDA branch.
- `fed_eg`: local `CustomSGD` followed by an exact second client phase at the server look-ahead point.
- `fed_zo_eg`: local `CustomSGD` followed by a forward-only SPSA second phase through `Client.train_zo()`.

Do not put conceptual labels such as `fedgda` or `fedogda` in YAML unless code support is added; unmatched values fall through to `CustomSGD` and can silently run a different algorithm. In `FedAvgAPI`, the critic learning rate is a dataset-dependent multiplier of `learning_rate` (currently 10 for toy data, 5 for Z/XZ image variants, and 3 for X image variants). Treat changes to those multipliers as algorithm changes, not routine YAML tuning.

For `fed_eg` and `fed_zo_eg`, keep `server_learning_rate` explicit. `fed_zo_eg` also uses the YAML ZO parameters (currently named `zo_mu` and `zo_num_directions`; inspect the implementation before adding or renaming seed/epsilon fields), and each direction costs two forward objective evaluations. Use a separate YAML, log, CSV name, and checkpoint namespace for comparisons so runs do not overwrite one another.

Treat deterministic and stochastic methods as separate reproducible runs. The
synthetic loader maps `batch_size <= 0` to deterministic/full-batch training
and `batch_size > 0` to stochastic/mini-batch training. Do not infer the mode
from the algorithm label. Encode it in the YAML filename, output namespace,
seed, and plot label (for example `fed_gda_full_batch_abs` versus
`fed_sgda_minibatch_abs`).

The required FL matrix is every requested algorithm and both batch modes over
`linear`, `abs`, `sin`, `step`, `cifar10_x`, `cifar10_z`,
`cifar10_xz`, `femnist_x`, `femnist_z`, and `femnist_xz`. Generate
one reviewed YAML per matrix cell or use a launcher that emits immutable
resolved configs. Never mutate the shared YAML while another run is active.

The FL flow is:

1. `main.py` parses `--cf`, chooses the device, loads scenario data, and creates the model list.
2. `FedMLRunner` selects simulation/SP; `SimulatorSingleProcess` constructs `FedAvgAPI`.
3. `FedAvgAPI` selects clients, assigns their local partitions, and calls client DeepGMM training for `g` and critic `f`.
4. The server aggregates parameters/deltas and applies the algorithm-specific OGDA or extragradient phase.
5. It evaluates the global structural model and writes `csv/<client_optimizer>_<dataset>newtrial.csv`, periodic `checkpoints/<client_optimizer>_<dataset>_round_<round>.pt`, and final result arrays/plots when enabled.

### Federated Client Execution Modes

Keep `backend: sp`; client concurrency is internal to `FedAvgAPI`, not the
generic vendored MPI FedAvg implementation. Select one explicit mode inside
`train_args`. The legacy `enable_multiprocessing` switch remains backward
compatible when `client_execution_mode` is omitted.

Serial client execution:

```yaml
client_execution_mode: sp
```

Persistent multi-GPU processes:

```yaml
client_execution_mode: multi_gpu_processes
enable_multiprocessing: true
multiprocessing_num_workers: 4
multiprocessing_gpu_ids: [0, 1, 2, 3]
```

GPU IDs are logical CUDA indices visible to the Python process. Use one worker
per GPU and list only GPUs assigned to the run. For two or three available
GPUs, shorten both the list and worker count. With fewer than two selected
workers, the implementation logs a warning and uses the original serial client
path.

True multiprocessing within one GPU:

```yaml
client_execution_mode: multiprocessingsinglegpu
enable_multiprocessing: false
multiprocessingsinglegpu_num_workers: 2
multiprocessingsinglegpu_gpu_id: 0
```

This mode creates persistent spawned Python processes and assigns every worker
the same logical CUDA GPU. Each worker owns an isolated trainer, optimizer
state, CUDA context, and one client task at a time. The GPU ID must match
`device_args.gpu_id`; the worker count is capped at `client_num_per_round`, and
fewer than two workers falls back to SP. When exposing one physical GPU
(for example `CUDA_VISIBLE_DEVICES=3`), CUDA remaps it to logical GPU 0, so
set both `device_args.gpu_id: 0` and
`multiprocessingsinglegpu_gpu_id: 0` in the copied run YAML. Start with two
workers, then benchmark two, four, and six because every process adds
CUDA-context and model memory.
The implementation works with standard CUDA and is compatible with an
externally managed NVIDIA MPS service, but it never starts, configures, or stops
MPS itself.

Single-process CUDA-stream concurrency on one GPU (not multiprocessing):

```yaml
client_execution_mode: single_gpu_streams
enable_multiprocessing: false
single_gpu_num_slots: 4
single_gpu_id: 0
```

`single_gpu_id` must match `device_args.gpu_id`, and the shell should expose
only that assigned GPU when isolation is required. With fewer than two slots,
the coordinator logs a warning and falls back to SP. This mode uses a Python
thread pool only to submit independent work to separate CUDA streams; all client
updates remain in one OS process and one CUDA context, so call it stream
concurrency rather than multiprocessing. Benchmark two, four, and six slots for
each high-dimensional workload because memory use and kernel contention are
workload dependent.

`fedml/simulation/sp/fedavg/multiprocess_client.py` owns persistent spawned
workers. `MultiprocessClientExecutor` assigns the configured GPU list, while
`SingleGPUMultiprocessClientExecutor` deliberately repeats one GPU ID for all
workers. Each worker applies deterministic cuDNN settings, moves CPU task
payloads to its device, calls the unchanged client methods, converts results
to CPU, synchronizes, and releases unused CUDA cache before notifying the
coordinator. Preserve that completion barrier so server evaluation never races
worker kernels. Each persistent worker still retains its trainer and CUDA
context, so isolate the selected GPU from unrelated jobs and ensure the combined
worker memory fits before increasing the worker count.

`fedml/simulation/sp/fedavg/single_gpu_client.py` owns one deep-copied `g`/`f`
trainer and one CUDA stream per slot. It schedules DeepGMM client work in
bounded waves with at most one active task per slot, synchronizes each wave,
and restores results to sampled-client order. Trainer and optimizer state are
never shared between concurrent slots. The supported FedAvg task contract has
no auxiliary direct-regression phase: an audit confirmed that baseline did not
feed MSE, checkpoints, CSVs, NPY predictions, or plots, so its per-client
training and aggregation were removed from SP, multi-GPU, and single-GPU
execution. The shared model factory constructs the legacy third model only for
other vendored coordinators; supported FedAvg runs return an empty third entry,
and `main.py`/`FedAvgAPI` retain only `g` and `f` before constructing trainers
or executors.

All concurrent modes reuse the same coordinator task contract. The coordinator
retains client sampling, sampled-client order, weighted aggregation, server
learning rates, OGDA previous-delta history, FedEG/FedZO-EG predictor/corrector
barriers, evaluation, checkpointing, and plotting. Do not move any of those
operations into executors. Treat changes to ordering, barriers, optimizer-state
clearing, or CPU/GPU conversion as algorithm-sensitive.

For deterministic methods, compare every `g` and `f` checkpoint tensor, metric,
and final prediction against SP. The deterministic CIFAR10-X FedGDA checks were
bit-for-bit equal across SP, four-GPU multiprocessing, and single-GPU stream
execution; two-, four-, and six-stream schedules also matched the SP algorithm
state in the earlier equivalence runs. FedZO-EG is stochastic and requires
seed-aware distributional validation rather than assumed bitwise equality
across schedules.

After removing the unused direct-regression path, a fresh deterministic
CIFAR10-X FedGDA check compared SP with four single-GPU streams using seed 0,
four clients, one full-batch local epoch, and one communication round. All 15
`g` tensors and all four `f` tensors matched with `torch.equal`, maximum
absolute difference was zero, checkpoint MSE was identical, and the complete
checkpoint files had the same SHA-256 hash. The client phases measured 0.899 s
for SP and 4.903 s for four streams in this single trial; treat this only as a
correctness smoke test, not a slot-count performance benchmark.

A fresh isolated-GPU test also compared the same deterministic SP reference
with `multiprocessingsinglegpu`, using two spawned workers for four clients.
Both workers were observed as distinct PIDs on logical GPU 0. All 19 `g` and
`f` checkpoint tensors matched with `torch.equal`, maximum absolute difference
was zero, both checkpoint MSE values were 0.20649527556843192, the final test
MSE was 0.2488411918148422, and both complete checkpoints had SHA-256
`9f1ccbc5fe8b8b515c10b3d2805fcaf2c0e2f4dc90a8dbf6e8301ea86ca2bab7`.
The client phase took 10.241 s in this correctness run. A prior attempt while
an unrelated Ray job occupied about 22 GiB correctly failed with CUDA OOM; run
this mode only on an isolated GPU with enough memory for every worker.

Do not reuse the earlier two-, four-, and six-slot timing numbers to select a
default: those measurements included the now-removed auxiliary regression
work. Re-benchmark the g/f-only path for CIFAR10 X/XZ and FEMNIST before
choosing a slot count. The 60-epoch model-selection phase remains serial and
can dominate short end-to-end runs.

These executors accelerate only independent federated client updates. Do not
apply them to centralized DeepGMM, which has no independent client dimension.
Optimize centralized runs with batching, data-loader workers, and later DDP
only when one GPU is demonstrably saturated.

## Centralized DeepGMM Workflow

There is currently no complete, runnable centralized DeepGMM command or YAML switch in this repository. `fedml/centralized/centralized_trainer.py` is a generic cross-entropy classification trainer and must not be presented as centralized DeepGMM. The repository does contain reusable centralized DeepGMM pieces: scenario NPZ loading, `OptimalMomentObjective`, candidate `g`/`f` models, `GradientDescentLearningDevF` and `SGDLearningDevF`, `OptimizerFactory`, `CustomSGD`/`SGDA`, `OGDA`, and `OAdam`. `OAdam` and `SGDA` are not currently wired into an executable centralized runner.

When implementing centralized DeepGMM, add a dedicated experiment-level entry point (for example `centralized_main.py`) rather than routing through federated `FedMLRunner`. It should:

1. Load one generated scenario archive with the matching scenario class or `AbstractScenario(filename)`, then convert it to tensors, flatten when required, and move it to the selected device.
2. Construct the same dataset-appropriate candidate `g` and `f` models and `OptimalMomentObjective` used by the federated path.
3. Pool the entire training split; do not partition it into clients or perform server aggregation.
4. Construct separate optimizers for minimizer `g` and maximizer/critic `f`. For GDA/SGDA use `CustomSGD` or `SGDA` with the ascent direction handled exactly once; for OAdam use separate `OAdam` instances. Preserve distinct `g` and critic learning rates.
5. Run alternating/simultaneous saddle-point updates through a centralized learner, evaluate on dev data for selection/early stopping, and report test MSE plus the moment/psi criterion.
6. Save outputs under a distinct `centralized_*` namespace containing dataset, optimizer, seed, learning rates, and iteration count so they cannot collide with FL artifacts.

Before claiming centralized support is complete, add a small YAML/config schema and a smoke test for each requested optimizer. Verify gradient signs carefully: `OptimalMomentObjective.calc_objective()` returns the objectives used for `g` and `f`, while `SGDA(maximize=True)` or a negated critic loss are alternative ascent mechanisms and must not both be applied. Do not reuse `learning/learning_dev_f.py` blindly: its current `fit()` creates hard-coded `CustomSGD(lr=0.3)` optimizers, so it needs refactoring to honor supplied optimizer factories before it can provide fair GDA/OAdam/SGDA comparisons.

The required centralized matrix is GDA and OAdam, each in deterministic
full-batch and stochastic mini-batch form, across the same ten datasets. If
SGDA is reported separately, define it as the stochastic GDA configuration
rather than silently treating it as a third optimizer. Store the resolved
configuration with every result.

## Plotting and Result Contract

Every run must produce a metrics CSV and compact plotting arrays containing the
evaluation coordinate or sample ID, prediction, and ground truth. Toy curve
plots must overlay sorted `x`, true `g(x)`, and all requested federated and
centralized estimates for sine, absolute, linear, and step. CIFAR/FEMNIST X and
XZ runs must not dump flattened images as plotting coordinates; use the
scenario's compact latent/evaluation coordinate or sample IDs. Keep CSV and NPY
results in separate result subdirectories from generated NPZ datasets.

`curve_plot.py` is currently hard-coded, partly commented, and incomplete
(including no active sine comparison). Refactor it into a parameterized
plotting entry point before using it for the full matrix. Plot filenames must
include scope (federated/centralized/comparison), dataset, algorithm, batch
mode, and seed.

## File Dependency and Cleanup Inventory

The detailed keep/archive/remove map is
`.codex/experiment_file_inventory.md`. Consult and update it before deleting
anything. In particular, preserve all CSVs and compact NPY result arrays in a
separate archive. The two approximately 46 GB raw-X FEMNIST outputs were
removed with user approval after confirming they were neither training inputs
nor compact curve-fitting results; do not recreate image-sized plotting arrays.

For slimming the nested FedML package, follow
`.codex/nested_fedml_slimming_plan.md`. Do not bulk-delete nested FedML
subsystems until its eager dispatch imports have been made lazy and both SP
DeepGMM and the retained MPI FedAvg path pass smoke tests.

## Validation and Coding Conventions

There is no repository-wide acceptance suite. At minimum, run:

```bash
python -m compileall main.py fedml/simulation/sp/fedavg/fedavg_api.py fedml/simulation/sp/fedavg/multiprocess_client.py fedml/simulation/sp/fedavg/single_gpu_client.py fedml/ml/trainer/my_model_trainer_classification.py scenarios learning optimizers
python -m pytest -q tests/test_multiprocess_client.py
```

For algorithm or configuration changes, run the smallest relevant smoke experiment, check for NaNs/divergence, and verify the expected CSV/checkpoint/result names. Add deterministic `test_*.py` tests for new centralized plumbing or optimizer behavior.

Follow PEP 8 with four-space indentation, `snake_case` for functions/variables/YAML keys, `PascalCase` for classes, and `UPPER_CASE` for constants. Prefer explicit configuration over new hard-coded hyperparameters. Keep imports grouped, comments accurate, and diffs focused.

## Git and Artifact Safety

Before pulling or editing, run `git status --short --branch`. Preserve unrelated tracked changes and all untracked logs, YAML variants, CSVs, plots, checkpoints, and result arrays unless the user explicitly asks to replace or remove them. Generated artifacts can represent long-running experiments. Avoid committing large regenerated data, checkpoints, logs, archives, secrets, or machine-specific paths unless they are intentional deliverables.

Use focused, imperative commit subjects naming the affected dataset or optimizer. Pull requests should document the algorithm/config change, exact generation and run commands, seeds and hyperparameters, summarized metrics, and relevant plots.

After every edit to this `AGENTS.md`, show the user the diff and explicitly ask them to review and approve or request corrections.
