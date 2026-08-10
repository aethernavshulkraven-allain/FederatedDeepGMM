# Nested FedML Slimming Plan

Scope: `fedgmm/sp_decentralized_mnist_lr_example/fedml/`.

Goal: retain current single-process Federated DeepGMM, planned centralized
DeepGMM, and enough MPI infrastructure to build multiprocess FedAvg and the
other DeepGMM methods. This is a deletion plan, not deletion authorization.

The nested FedML tree is about 54 MiB. Roughly 45 MiB is an unused packaged
FederatedEMNIST HDF5 file; most remaining code is small. Cleanup is primarily
for correctness and maintainability rather than major disk recovery.

## Why Direct Bulk Deletion Is Not Yet Safe

Runtime tracing after importing the supported launcher loaded 285 FedML
modules. Most were loaded only because six dispatch files use eager imports:

1. `fedml/__init__.py` imports API, MLOps, serving, cross-silo/cloud/device,
   and every launch surface.
2. `fedml/core/__init__.py` imports contribution, distributed flow, DP,
   security, and aggregation.
3. `fedml/data/data_loader.py` imports many unrelated dataset loaders.
4. `fedml/model/model_hub.py` imports every generic model family.
5. `fedml/ml/trainer/trainer_creator.py` imports NLP/tag trainers.
6. `fedml/simulation/simulator.py` imports every SP or MPI algorithm before
   checking the configured optimizer.

These files must be converted to branch-local/lazy imports before deleting
their unused dependencies.

## Retain: Current Single-Process Federated DeepGMM

### FedML entry and dispatch

- `fedml/__init__.py`
- `fedml/arguments.py`
- `fedml/constants.py`
- `fedml/runner.py`
- `fedml/launch_simulation.py`
- `fedml/config/simulation_sp/`
- `fedml/simulation/__init__.py`
- `fedml/simulation/simulator.py`
- all of `fedml/simulation/sp/fedavg/`

### Device and data

- `fedml/device/__init__.py`
- `fedml/device/device.py`
- `fedml/data/__init__.py`
- `fedml/data/data_loader.py`
- `fedml/data/MNIST/__init__.py`
- `fedml/data/MNIST/data_loader.py`

The custom loader uses generated scenario NPZ files for toy, CIFAR X/Z/XZ,
FEMNIST X/Z/XZ, and MNIST X/Z/XZ. Generic torchvision CIFAR and packaged
FederatedEMNIST loaders are not the runtime inputs for these experiments.

### Models and training

- `fedml/model/__init__.py`
- `fedml/model/model_hub.py`
- `fedml/ml/__init__.py`
- `fedml/ml/trainer/__init__.py`
- `fedml/ml/trainer/trainer_creator.py`
- `fedml/ml/trainer/my_model_trainer_classification.py`
- `fedml/core/alg_frame/client_trainer.py`
- `fedml/core/alg_frame/server_aggregator.py`
- `fedml/core/dp/` until DP references are deliberately removed from the
  active trainer
- `fedml/utils/debugging.py` and package marker files required by imports

Actual DeepGMM model classes are in the experiment-level `models/` directory,
not generic `fedml/model/cv/`: `models/mlp_model.py` supplies toy/numeric
models and `models/cnn_models.py` supplies MNIST/FEMNIST/CIFAR CNNs.

## Retain: Planned Centralized DeepGMM

The planned centralized runner should live at experiment level and reuse:

- the retained NPZ loader;
- experiment-level `models/`;
- experiment-level `scenarios/`, `game_objectives/`, `learning/`,
  `model_selection/`, and `optimizers/`;
- PyTorch directly.

`fedml/centralized/centralized_trainer.py` is a cross-entropy classification
trainer and is not a centralized DeepGMM dependency. It can be removed.

## Retain: Planned Multiprocess/MPI FedAvg

Retain these as a reference and integration base:

- `fedml/config/simulaton_mpi/` (rename the typo when implementing);
- `fedml/device/gpu_mapping_mpi.py`;
- `fedml/ml/engine/`;
- `fedml/ml/aggregator/`;
- all files in `fedml/simulation/mpi/fedavg/`;
- `fedml/core/distributed/fedml_comm_manager.py`;
- `fedml/core/distributed/communication/{__init__.py,base_com_manager.py,constants.py,message.py,observer.py,utils.py}`;
- `fedml/core/distributed/communication/mpi/`;
- the minimum MLOps stubs used directly by the MPI communication manager;
- `fedml/core/security/` and `fedml/core/dp/` while the stock MPI FedAvg
  code imports attacker, defender, and DP singletons.

The stock MPI FedAvg trainer is generic and does not yet implement this
repository's two-model DeepGMM updates. It is retained as transport/process
scaffolding, not as a ready multiprocess implementation.

Before pruning other MPI algorithms, refactor `SimulatorMPI` so only the
selected algorithm is imported. Then retain `simulation/mpi/fedavg/` and
remove the unrelated MPI algorithms.

## Safe Removal After Lazy-Import Refactor

### Product and deployment surfaces

- `fedml/api/`
- `fedml/cli/`
- `fedml/computing/`
- `fedml/cross_cloud/`
- `fedml/cross_device/`
- `fedml/cross_silo/`
- `fedml/fa/`
- `fedml/mlops/`
- `fedml/scalellm/`
- `fedml/serving/`
- `fedml/train/`
- `fedml/workflow/`
- `fedml/launch_cross_cloud.py`
- `fedml/launch_cross_device.py`
- `fedml/launch_cross_silo_hi.py`
- `fedml/launch_cross_silo_horizontal.py`
- `fedml/launch_serving.py`

Removing `api/`, `computing/`, and MLOps code requires first simplifying
`fedml/__init__.py`, `data_loader.py`, and MPI logging/config references.

### Unused training platforms and SP algorithms

- all of `fedml/simulation/nccl/`;
- under `fedml/simulation/sp/`, everything except `fedavg/`;
- `fedml/distributed/` (the top-level legacy package, not
  `fedml/core/distributed/`);
- `fedml/centralized/`.

First change `SimulatorSingleProcess` to import only `sp/fedavg` when
`federated_optimizer: FedAvg`.

### Unused MPI algorithms

After making `SimulatorMPI` lazy, remove:

- `simulation/mpi/async_fedavg/`
- `simulation/mpi/base_framework/`
- `simulation/mpi/classical_vertical_fl/`
- `simulation/mpi/decentralized_framework/`
- `simulation/mpi/fedavg_seq/`
- `simulation/mpi/fedgan/`
- `simulation/mpi/fedgkt/`
- `simulation/mpi/fednas/`
- `simulation/mpi/fednova/`
- `simulation/mpi/fedopt/`
- `simulation/mpi/fedopt_seq/`
- `simulation/mpi/fedprox/`
- `simulation/mpi/fedseg/`
- `simulation/mpi/split_nn/`

Retain only `simulation/mpi/fedavg/` for the planned multiprocess work.

### Unused dataset packages

After replacing top-level imports in `data/data_loader.py` with imports inside
the selected dataset branch, remove:

- `data/AutonomousDriving/`
- `data/FeTS2021/`
- `data/FederatedEMNIST/`, including the unused ~45 MiB
  `datasets/fed_emnist_test.h5`
- `data/ImageNet/`
- `data/Landmarks/`
- `data/NUS_WIDE/`
- `data/UCI/`
- `data/cifar10/`
- `data/cifar100/`
- `data/cinic10/`
- `data/edge_case_examples/`
- `data/fed_cifar100/`
- `data/fed_shakespeare/`
- `data/fednlp/`
- `data/lending_club_loan/`
- `data/reddit/`
- `data/shakespeare/`
- `data/stackoverflow/`
- `data/stackoverflow_lr/`
- `data/stackoverflow_nwp/`
- `data/synthetic_0_0/`
- `data/synthetic_0.5_0.5/`
- `data/synthetic_1_1/`
- `data/utils/`
- `data/data_loader_cross_silo.py`
- `data/file_operation.py` after removing its wildcard import

Retain only `data/data_loader.py` and `data/MNIST/` for the requested
generated scenario datasets. Despite its name, that loader reads all custom
toy/image NPZ variants.

### Unused generic models

After making `model_hub.py` import only the classes selected by the requested
dataset branch, remove:

- `model/cv/`
- `model/finance/`
- `model/linear/`
- `model/mobile/`
- `model/nlp/`

The requested models come from experiment-level `models/cnn_models.py` and
`models/mlp_model.py`. If plain dataset name `mnist` remains supported,
retain `model/linear/`; the requested X/Z/XZ matrix does not need it.

### Unused trainers and generic aggregation

After making `trainer_creator.py` lazy, remove:

- `ml/trainer/my_model_trainer_nwp.py`
- `ml/trainer/my_model_trainer_tag_prediction.py`
- MIME, FedNova, FedProx, SCAFFOLD, FedDyn, and duplicate trainer files

Retain `ml/aggregator/` for planned MPI FedAvg and
`my_model_trainer_classification.py` for current DeepGMM.

### Core modules

After simplifying `core/__init__.py`, current SP DeepGMM does not need:

- `core/contribution/`
- `core/data/`
- `core/fhe/`
- `core/mpc/`
- `core/schedule/`

Retain `core/alg_frame/`, `core/dp/`, `core/security/`, and the selected
`core/distributed/` MPI subset until multiprocess DeepGMM is implemented and
its actual imports are known. Non-MPI communication transports (MQTT, S3,
gRPC, TRPC, Web3, Theta) can be removed after
`core/distributed/communication/__init__.py` and the communication factory
are made MPI-only.

## Files Requiring Modification Before Cleanup

The slimming implementation should first patch:

1. `fedml/__init__.py`: expose only init/config, device, data, model,
   `FedMLRunner`, and simulation.
2. `fedml/core/__init__.py`: stop eagerly importing contribution, flow,
   security, DP, and generic aggregation; import explicit classes at use sites.
3. `fedml/data/__init__.py`: stop importing cross-silo splitting.
4. `fedml/data/data_loader.py`: move every loader import into its dataset
   branch and keep only the custom NPZ branch.
5. `fedml/model/model_hub.py`: remove generic top-level model imports and
   keep experiment model imports.
6. `fedml/ml/trainer/trainer_creator.py`: import only the selected trainer.
7. `fedml/simulation/simulator.py`: lazy-load only the selected backend and
   optimizer.
8. MPI FedAvg files: replace or isolate MLOps, security, and DP dependencies
   that are not wanted in multiprocess DeepGMM.

## Validation Gates Before Any Deletion

1. Save a Git patch/commit or create a cleanup branch.
2. Import `main.py` with runtime tracing; confirm unrelated API/MLOps/model/
   dataset modules no longer load.
3. Run syntax compilation without leaving caches.
4. Load all toy/CIFAR/FEMNIST/MNIST X/Z/XZ NPZ files.
5. Run one-round SP smoke tests for `sgd`, `ogda`, `fed_eg`, and
   `fed_zo_eg`, in full-batch and mini-batch modes where feasible.
6. Import and launch a minimal MPI FedAvg job with two local processes.
7. Add centralized GDA/OAdam smoke tests once that runner exists.
8. Delete one subsystem group at a time and repeat the relevant checks.

## Recommended Deletion Order

1. Unused packaged dataset files (largest immediate saving).
2. Product/deployment surfaces after simplifying `fedml/__init__.py`.
3. Unused generic models and loaders after lazy dispatch.
4. Unused SP and MPI algorithms after lazy simulator dispatch.
5. Non-MPI communication transports and MLOps/security/DP only after the
   multiprocess design no longer imports them.

Do not delete the entire nested `fedml/` package or the experiment-level
`models/`, `scenarios/`, `game_objectives/`, `optimizers/`,
`model_selection/`, or `learning/` directories.
