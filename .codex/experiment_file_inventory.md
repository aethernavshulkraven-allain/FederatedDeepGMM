# Experiment File and Dependency Inventory

This inventory is scoped to Federated and centralized DeepGMM experiments on
`linear`, `abs`, `sin`, `step`, CIFAR X/Z/XZ, and FEMNIST X/Z/XZ. It is
a cleanup plan, not deletion authorization. Re-run the inventory after code or
configuration changes because the nested FedML package uses dynamic dispatch.

## Required: Shared Experiment Core

- `main.py` -> `fedml/__init__.py`, `fedml/arguments.py`,
  `fedml/device/device.py`, `fedml/data/data_loader.py`,
  `fedml/model/model_hub.py`, `fedml/runner.py`, and
  `fedml/simulation/simulator.py`.
- `fedml/config/simulation_sp/*.yaml` -> parsed by `fedml.init()`; each
  reproducible algorithm/batch-mode run needs its own reviewed YAML.
- `fedml/simulation/sp/fedavg/fedavg_api.py` ->
  `fedml/simulation/sp/fedavg/client.py`,
  `fedml/ml/trainer/trainer_creator.py`,
  `fedml/ml/trainer/my_model_trainer_classification.py`,
  `model_selection_class.py`, `model_selection/`, `game_objectives/`,
  `optimizers/`, and `plotting.py`.
- `models/` and the relevant branches in `fedml/model/model_hub.py` ->
  candidate structural model `g`, critic `f`, and regression model creation.
- `scenarios/abstract_scenario.py` and `scenarios/toy_scenarios.py` ->
  shared NPZ serialization, standardization, and toy structural functions.

## Required: Dataset-Specific

- Toy: `generate_zoo_data.py`, `scenarios/toy_scenarios.py`, and
  `data/zoo/{linear,abs,sin,step}.npz`.
- CIFAR: `generate_cifar10_data.py`, `scenarios/cifar10_scenario.py`,
  `datasets/cifar-10-batches-py/` (or one canonical equivalent source), and
  `data/cifar10_{x,z,xz}/main.npz`.
- FEMNIST: `generate_emnist_data.py`, `scenarios/emnist_scenario.py`, the
  EMNIST raw source actually consumed by that scenario, and
  `data/femnist_{x,z,xz}/main.npz`.
- MNIST files are optional for the current requested matrix. Retain
  `generate_mnist_data.py`, `scenarios/mnist_scenarios.py`, and
  `data/mnist_{x,z,xz}/` only if MNIST experiments remain in scope.

Generated NPZ files can be recreated from the scenario and source dataset.
They are runtime inputs and should be kept for immediate reruns, or archived
when disk space is more important than avoiding regeneration.

## Required: Federated Algorithm Variants

- Deterministic/full-batch and stochastic/mini-batch use the same FL code.
  `fedml/data/data_loader.py` interprets `batch_size <= 0` as full batch and
  `batch_size > 0` as mini-batch.
- FedGDA/S-GDA: `client_optimizer: sgd` -> `CustomSGD`.
- FedOGDA/S-OGDA: `client_optimizer: ogda` -> `optimizers/ogda.py`, plus
  the OGDA server branch in `fedavg_api.py`.
- FedEG: `client_optimizer: fed_eg` -> the exact second phase in
  `fedavg_api.py`.
- FedZO-EG: `client_optimizer: fed_zo_eg` ->
  `Client.train_zo()` and `ModelTrainerCLS.train_gmm_zo()`.

The literal YAML values above are required; conceptual names such as
`fedgda` and `fedogda` are not currently parsed.

## Required: Centralized Algorithms (After Wiring)

No runnable centralized DeepGMM entry point exists yet. A new
`centralized_main.py` and centralized YAML schema are required. They should
depend on:

- `scenarios/abstract_scenario.py` and the same generated NPZ data;
- dataset-appropriate models from `models/`;
- `game_objectives/simple_moment_objective.py`;
- `optimizers/optimizer_factory.py`;
- `optimizers/Customsgd.py` for GDA/SGDA;
- `optimizers/oadam.py` for OAdam;
- optionally `learning/learning_dev_f.py`, only after removing its hard-coded
  `CustomSGD(lr=0.3)` construction and honoring supplied optimizers.

Deterministic centralized runs use the full training split per update.
Stochastic centralized runs use shuffled mini-batches. Both must use separate
`g` and `f` optimizer instances and learning rates, with critic ascent
applied exactly once.

`fedml/centralized/centralized_trainer.py` is not required for DeepGMM; it is
a generic cross-entropy classifier.

## Required: Results and Curve Fitting

Preserve CSV and NPY results separately from source and generated datasets:

- Metrics: `csv/*.csv`.
- Compact curve data:
  `results_<dataset>_<algorithm>_{x,y_pred*,y_true}.npy`.
- Checkpoints: `checkpoints/*.pt`; keep final/best checkpoints and archive or
  remove redundant periodic checkpoints after confirming reproducibility.
- Plots: PNG/PDF outputs and the script/config/CSV/NPY inputs that created them.

`curve_plot.py` currently has hard-coded filenames and only actively plots an
absolute FedEG comparison. Its older toy comparison code is partly commented
and does not cover sine. It must be replaced or refactored into a parameterized
plotter before claiming complete matrix support.

For toy functions, plot sorted scalar `x`, ground-truth `g(x)`, and every
predicted curve. For CIFAR/FEMNIST X and XZ variants, never save flattened image
tensors as curve x-coordinates. Save a compact evaluation coordinate/latent
`w`, sample ID, ground truth, and prediction. Z variants can use the
scenario's compact scalar evaluation coordinate where available.

## High-Confidence Cleanup Candidates

These are reproducible caches, archives, duplicate inputs, or outputs that are
not required to execute the requested experiments:

- All 79 `__pycache__/` directories and all `*.pyc` files (about 3.3 MB).
- `git-lfs-linux-amd64-v3.6.1.tar.gz` after confirming Git LFS is installed.
- Old `nohup.out` and `*.log` files after preserving needed run metadata.
- Duplicate downloaded archives after extraction and checksum verification,
  including duplicate CIFAR tarballs and EMNIST `gzip.zip`.
- One of the duplicate CIFAR raw trees:
  `data/cifar-10-batches-py/` or `datasets/cifar-10-batches-py/`, after
  verifying which path `scenarios/cifar10_scenario.py` reads.
- Superseded plots that are reproducible from preserved results.
- Redundant intermediate checkpoints after retaining the selected/best model.
- `results_femnist_x_sgd_x.npy` and
  `results_femnist_xz_sgd_x.npy` (about 46 GB each). These ignored files
  serialize image-sized X values and are not needed when compact plotting
  coordinates and prediction/ground-truth arrays are preserved.

The two raw-X arrays, Git LFS installer archive, Python bytecode caches, and
obsolete `nohup.out` files were removed with user approval on 2026-08-07.

Dataset verification on 2026-08-07 established that CIFAR generation reads
`datasets/cifar-10-batches-py/`, while the experiment reads generated
`data/cifar10_{x,z,xz}/main.npz`. The identical duplicate tree and tarball
under `data/` were removed. EMNIST generation reads extracted
`datasets/EMNIST/raw/emnist-digits-*` files, while experiments read generated
`data/femnist_{x,z,xz}/main.npz`; the unused `gzip.zip` was removed.
MNIST experiments read generated `data/mnist_{x,z,xz}/main.npz`; no raw
`datasets/MNIST/` tree currently exists, so regeneration will download it.

## Candidate Code Removal: Verify Before Deleting

The requested runs use simulation/SP/FedAvg only. The following nested FedML
subsystems are outside that runtime scope and are candidates for removal from a
slim experiment checkout:

- `fedml/api/`, `cli/`, `computing/`, `cross_cloud/`,
  `cross_device/`, `cross_silo/`, `distributed/`, `fa/`, `mlops/`,
  `scalellm/`, `serving/`, `train/`, and `workflow/`.
- `fedml/simulation/mpi/`, `fedml/simulation/nccl/`, and SP algorithms
  other than the active `fedavg/` implementation.
- Unused dataset loaders under `fedml/data/` and unused model families under
  `fedml/model/`.
- The root-level `fedml/` and `models/` fragments were removed on
  2026-08-07 after the supported launcher resolved every relevant import to
  the nested experiment copies.
- `Toy_Example/` if the standalone bilinear demonstration is no longer
  needed.
- `session.md`, old hard-coded plotting scripts, and historical output files
  after their useful information is migrated.

Do not bulk-delete these directories directly. The local `fedml/__init__.py`
and model/data hubs may import modules dynamically or at import time. First
create a slim copy or branch, run import tracing and the complete smoke matrix,
then delete one group at a time.

## Cleanup Order

1. Copy all CSVs and compact NPY prediction/ground-truth results to a dated
   results archive with a manifest and checksums.
2. Preserve reviewed YAML files, seeds, logs needed for provenance, best
   checkpoints, and final plots.
3. Remove caches and confirmed redundant archives.
4. Remove the two 46 GB raw-X result arrays after explicit approval.
5. Deduplicate raw CIFAR/EMNIST sources only after generator-path verification.
6. Slim vendored FedML code on a separate branch and validate every requested
   dataset/algorithm/batch-mode combination before merging.
