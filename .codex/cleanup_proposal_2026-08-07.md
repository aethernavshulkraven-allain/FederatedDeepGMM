# Cleanup Proposal — 2026-08-07

This file records the completed high-confidence cleanup. A total of
99,273,385,669 bytes (92.46 GiB) was removed in two verified phases.

## Removed After Dataset Dependency Verification

| Bytes | Path or pattern | Reason |
|---:|---|---|
| 170,498,071 | `fedgmm/sp_decentralized_mnist_lr_example/data/cifar-10-python.tar.gz` | Duplicate download outside the scenario's canonical `datasets/` path; SHA-256 matches the retained archive. |
| 186,214,114 | `fedgmm/sp_decentralized_mnist_lr_example/data/cifar-10-batches-py/` | Duplicate extracted CIFAR tree; the scenario reads `datasets/`. |
| 561,753,746 | `fedgmm/sp_decentralized_mnist_lr_example/datasets/EMNIST/raw/gzip.zip` | Download archive is unnecessary after successful extraction; retain extracted EMNIST files. |

The two extracted CIFAR trees compared identically. After removal,
`torchvision.datasets.CIFAR10(..., download=False)` loaded 50,000 training
and 10,000 test examples from the retained `datasets/` tree.
`torchvision.datasets.EMNIST(split="digits", download=False)` loaded 240,000
training and 40,000 test examples from the retained extracted files.

## Removed After Result and Cache Approval

| Bytes | Path or pattern | Reason |
|---:|---|---|
| 49,172,480,128 | `fedgmm/sp_decentralized_mnist_lr_example/results_femnist_x_sgd_x.npy` | Raw image-X output, not a compact curve result or training input. |
| 49,172,480,128 | `fedgmm/sp_decentralized_mnist_lr_example/results_femnist_xz_sgd_x.npy` | Raw image-X output, not a compact curve result or training input. |
| 5,258,253 | `fedgmm/sp_decentralized_mnist_lr_example/git-lfs-linux-amd64-v3.6.1.tar.gz` | Installer archive, not an experiment dependency. |
| 3,315,018 | all 357 `*.pyc` files and their empty `__pycache__/` directories | Reproducible Python bytecode cache. |
| 1,385,483 | `fedgmm/sp_decentralized_mnist_lr_example/nohup.out` | Obsolete historical launcher log. |
| 728 | root `nohup.out` | Empty/minimal historical launcher output. |

The retained CIFAR archive is
`fedgmm/sp_decentralized_mnist_lr_example/datasets/cifar-10-python.tar.gz`;
both copies have SHA-256
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.

## Explicitly Retained

- Every CSV result, with a separate archive and member manifest.
- Every NPY smaller than 1 GiB, with a separate archive and member manifest.
- All current YAML run configurations.
- All PNG/PDF plots.
- Generated toy/CIFAR/FEMNIST NPZ inputs needed for immediate reruns.
- Generated MNIST X/Z/XZ NPZ inputs. No raw `datasets/MNIST/` tree is
  currently present; regeneration will download it through torchvision.
- Canonical raw CIFAR data under `datasets/`.
- Extracted EMNIST data under `datasets/EMNIST/`.
- All checkpoints until a best-checkpoint retention rule is approved.
- All source code until the slim-copy smoke matrix passes.

## Phase 2: Do Not Remove Yet

Potentially large savings remain in duplicate or unused vendored FedML modules,
MNIST data, old checkpoints, old plots, CSV variants, and source archives.
Those require a clean slim branch plus import and experiment smoke tests.
They are intentionally excluded from Phase 1.

## Post-Cleanup Verification

- All four safety-archive SHA-256 checks pass.
- All nine CIFAR/FEMNIST/MNIST X/Z/XZ generated NPZ inputs remain present.
- Canonical CIFAR and extracted EMNIST raw datasets remain present.
- No `*.pyc` files remain under the experiment.
- The experiment directory is now approximately 5.2 GiB.

## Root Duplicate Cleanup

The incomplete root-level `fedml/` and `models/` trees (five tracked files,
about 124 KiB) were subsequently removed. Before and after removal, import
checks from the supported experiment directory resolved `fedml`, its data
loader and model hub, the SP FedAvg API, and `models.cnn_models` to
`fedgmm/sp_decentralized_mnist_lr_example/`. The nested implementations
remain intact.
