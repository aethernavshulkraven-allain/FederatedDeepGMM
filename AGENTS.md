# Repository Guidelines

## Project Structure & Module Organization

The primary experiment lives in `fedgmm/sp_decentralized_mnist_lr_example/`. The only supported program path is `fedml/simulation/sp/fedavg/fedavg_api.py`; make and validate training changes through its `FedAvgAPI`. The `scenarios/`, `game_objectives/`, `learning/`, `model_selection/`, `models/`, and `optimizers/` directories separate supporting concerns. Runtime settings are in `fedml/config/simulation_sp/fedml_config.yaml`. Dataset generators (`generate_zoo_data.py`, `generate_mnist_data.py`, and related scripts) sit at the experiment root; generated data, CSV results, checkpoints, and plots belong in their existing artifact directories.

`Toy_Example/` is reference material, not a supported execution path. Root-level `fedml/` and `models/`, plus the nested experiment `fedml/` tree, contain framework and model code; avoid broad refactors there when changing one experiment.

## Build, Test, and Development Commands

There is no package build step or pinned environment file. Use a virtual environment with PyTorch, NumPy, Matplotlib, and YAML/FedML dependencies.

```bash
cd fedgmm/sp_decentralized_mnist_lr_example
python generate_zoo_data.py
python -m compileall fedml/simulation/sp/fedavg/fedavg_api.py
```

The generator prepares low-dimensional `step`, `abs`, and `linear` data; `compileall` checks syntax. Run the required program only through `fedml/simulation/sp/fedavg/fedavg_api.py` in the project environment; do not substitute `main.py`, `Toy_Example/example.py`, or another FedML API. This module defines `FedAvgAPI` and is consumed by the simulation framework rather than providing a standalone CLI.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, variables, and YAML keys; use `PascalCase` for classes and `UPPER_CASE` for constants. Keep optimizer implementations in `optimizers/` and dataset-specific setup in `scenarios/`, but integrate them through `FedAvgAPI`. Prefer explicit configuration over embedding new hyperparameters in training logic. No formatter or linter is configured, so keep imports grouped, comments current, and diffs focused.

## Testing Guidelines

There is no repository-wide automated test suite or coverage threshold. For algorithm changes, run `compileall`, exercise the smallest relevant case through `fedavg_api.py`, and verify losses and generated CSV/plot output. Add deterministic `test_*.py` files near the affected package (or under a new `tests/` directory) and use `pytest` for new unit tests. Do not treat tests inside the vendored FedML tree as the project’s acceptance suite.

## Commit & Pull Request Guidelines

History uses short, imperative, experiment-oriented subjects (for example, `Experiments for abs function`). Keep commits focused and name the dataset or optimizer affected. Pull requests should explain the algorithm/configuration change, list exact reproduction commands, and summarize metrics. Link relevant issues and attach plots for behavior changes. Avoid committing large regenerated artifacts, checkpoints, archives, API keys, or machine-specific paths unless they are intentional deliverables.
