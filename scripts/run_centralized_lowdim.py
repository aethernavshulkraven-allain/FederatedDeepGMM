#!/usr/bin/env python3
"""Run true centralized low-dimensional DeepGMM baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Every model here is a tiny 1-2 hidden-layer MLP on at most a few thousand
# rows; PyTorch's CPU default is one BLAS thread per core, which on a shared
# multi-user machine means one process can grab dozens of cores for matmuls
# too small to benefit from that parallelism (measured: ~29 cores per process,
# almost pure thread-contention overhead, no speedup). Must be set before
# numpy/torch are imported -- the BLAS thread pool is initialized at import.
_DEFAULT_CPU_THREADS = "4"
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_env_var, _DEFAULT_CPU_THREADS)

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from experiment_utils import (  # noqa: E402
    json_safe,
    predictions_for_split,
    prepare_run_dir,
    save_predictions_npz,
    state_is_finite,
    structural_mse,
    structural_mse_from_predictions,
    write_effective_config,
    write_metrics,
    write_test_mse_by_round,
)
from game_objectives.simple_moment_objective import OptimalMomentObjective  # noqa: E402
from models.mlp_model import MLPModel  # noqa: E402
from optimizers.Customsgd import CustomSGD  # noqa: E402
from optimizers.oadam import OAdam  # noqa: E402
from scenarios.abstract_scenario import AbstractScenario  # noqa: E402


DATASET_ALIASES = {
    "absolute": "abs",
    "sine": "sin",
}
LOWDIM_DATASETS = {"abs", "step", "linear", "sin"}
# eicu_semisynth is not a single scenario file (unlike the zoo datasets): the
# actual file is data/eicu_semisynth/<scenario_name>.npz, so which scenario to
# load is a separate --scenario-name argument, not encoded in --dataset.
EICU_DATASETS = {"eicu_semisynth"}
METHODS = {"gda", "sgda", "oadam"}

# Matches model_hub.py's eicu* branch; centralized baselines must use the same
# architecture as the federated runs they are compared against.
EICU_HIDDEN_WIDTHS = [64, 64]
ZOO_HIDDEN_WIDTHS = [20, 20]


def canonical_dataset(dataset: str) -> str:
    value = DATASET_ALIASES.get(dataset.lower(), dataset.lower())
    if value not in LOWDIM_DATASETS and value not in EICU_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if args.no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.gpu_id is not None:
        return torch.device(f"cuda:{int(args.gpu_id)}")
    return torch.device("cuda")


def move_split(split: Any, device: torch.device) -> Any:
    for name in ("x", "z", "y", "g", "w"):
        tensor = getattr(split, name, None)
        if torch.is_tensor(tensor):
            setattr(split, name, tensor.to(device).double())
    return split


def load_pooled_splits(
    dataset: str, data_dir: Path, device: torch.device, scenario_name: str | None = None
) -> tuple[Any, Any, Any, Path]:
    if dataset in EICU_DATASETS:
        if not scenario_name:
            raise ValueError(f"--scenario-name is required for --dataset {dataset!r}")
        scenario_path = data_dir / dataset / f"{scenario_name}.npz"
    else:
        scenario_path = data_dir / "zoo" / f"{dataset}.npz"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario file: {scenario_path}")
    scenario = AbstractScenario(filename=str(scenario_path))
    scenario.to_tensor()
    train = move_split(scenario.get_dataset("train"), device)
    val = move_split(scenario.get_dataset("dev"), device)
    test = move_split(scenario.get_dataset("test"), device)
    return train, val, test, scenario_path


def build_models(
    device: torch.device, input_dim_g: int, input_dim_f: int, hidden_widths: list[int]
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Dimensions are derived from the loaded data (train.x/.z widths), not
    hardcoded -- the zoo scenarios are always 1-D treatment / 2-D instrument,
    but eicu_semisynth packs [D, W] / [Z, W] and its width depends on the
    cohort's covariate count.
    """
    g = MLPModel(input_dim=input_dim_g, layer_widths=hidden_widths, activation=nn.LeakyReLU).double()
    f = MLPModel(input_dim=input_dim_f, layer_widths=hidden_widths, activation=nn.LeakyReLU).double()
    g.initialize()
    f.initialize()
    return g.to(device), f.to(device)


def build_optimizer(method: str, model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if method in {"gda", "sgda"}:
        return CustomSGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    if method == "oadam":
        return OAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported method {method!r}")


def choose_batch_indices(split: Any, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    n = int(split.y.shape[0])
    if batch_size <= 0 or batch_size >= n:
        return torch.arange(n, device=split.y.device)
    perm = torch.randperm(n, generator=generator, device=split.y.device)
    return perm[:batch_size]


def capture_state(g: torch.nn.Module, f: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "g_state_dict": {key: value.detach().cpu().clone() for key, value in g.state_dict().items()},
        "f_state_dict": {key: value.detach().cpu().clone() for key, value in f.state_dict().items()},
    }


def load_state(g: torch.nn.Module, f: torch.nn.Module, state: dict[str, dict[str, torch.Tensor]]) -> None:
    g.load_state_dict(copy.deepcopy(state["g_state_dict"]))
    f.load_state_dict(copy.deepcopy(state["f_state_dict"]))


def finite_state_and_metrics(g: torch.nn.Module, f: torch.nn.Module, values: list[float]) -> bool:
    return (
        state_is_finite(g.state_dict())
        and state_is_finite(f.state_dict())
        and all(math.isfinite(float(value)) for value in values)
    )


def write_mse_csv(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = run_dir / "mse_by_round.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "train_mse", "val_mse"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "round": row["round"],
                "train_mse": row["train_mse"],
                "val_mse": row["val_mse"],
            })
    return path


def save_checkpoint(
    path: Path,
    round_idx: int,
    checkpoint_type: str,
    state: dict[str, dict[str, torch.Tensor]],
    metrics: dict[str, Any],
    effective_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "round": int(round_idx),
        "checkpoint_type": checkpoint_type,
        "state": {
            "g": state["g_state_dict"],
            "f": state["f_state_dict"],
        },
        "g_state_dict": state["g_state_dict"],
        "f_state_dict": state["f_state_dict"],
        "metrics": json_safe(metrics),
        "effective_config": json_safe(effective_config),
    }, path)


def default_batch_size(method: str, raw_batch_size: int | None) -> int:
    if raw_batch_size is not None:
        return int(raw_batch_size)
    if method == "gda":
        return 0
    return 256


def default_output_dir(dataset: str, method: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "centralized_lowdim_v1" / dataset / method / f"seed_{seed}"


def build_effective_config(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    batch_size: int,
    output_dir: Path,
    device: torch.device,
    scenario_path: Path,
    input_dim_g: int,
    input_dim_f: int,
    hidden_widths: list[int],
) -> dict[str, Any]:
    mode = "deterministic" if batch_size <= 0 else "stochastic"
    return {
        "training_scope": "centralized",
        "method": method,
        "algorithm": method,
        "variant": method,
        "mode": mode,
        "dataset": dataset,
        "seed": int(args.seed),
        "random_seed": int(args.seed),
        "iterations": int(args.iterations),
        "batch_size": int(batch_size),
        "g_learning_rate": float(args.g_lr),
        "f_learning_rate": float(args.f_lr),
        "learning_rate": float(args.g_lr),
        "weight_decay": float(args.weight_decay),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "test_mse_logged_by_round": bool(args.log_test_mse_by_round),
        "log_test_mse_by_round": bool(args.log_test_mse_by_round),
        "uses_clients": False,
        "uses_fedavg_aggregation": False,
        "uses_client_sampling": False,
        "uses_server_learning_rate_aggregation": False,
        "objective": "OptimalMomentObjective",
        "input_dim_g": int(input_dim_g),
        "input_dim_f": int(input_dim_f),
        "model": f"MLPModel(g:{input_dim_g}->{hidden_widths}, f:{input_dim_f}->{hidden_widths})",
        "data_path": str(scenario_path.relative_to(REPO_ROOT) if scenario_path.is_relative_to(REPO_ROOT) else scenario_path),
        "output_dir": str(output_dir),
        "run_id": str(args.run_id or f"centralized_{dataset}_{method}_seed{args.seed}"),
        "device": str(device),
        "overwrite": bool(args.overwrite),
    }


def run(args: argparse.Namespace) -> Path:
    method = args.method.lower()
    if method not in METHODS:
        raise ValueError(f"Unsupported method {args.method!r}")
    dataset = canonical_dataset(args.dataset)
    batch_size = default_batch_size(method, args.batch_size)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(dataset, method, args.seed)
    output_dir = output_dir.resolve()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if method == "gda" and batch_size > 0:
        raise ValueError("gda is deterministic; use --batch-size 0")
    if method in {"sgda", "oadam"} and batch_size <= 0:
        raise ValueError(f"{method} requires a positive minibatch --batch-size")

    prepare_run_dir(str(output_dir), overwrite=args.overwrite)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args)
    data_dir = Path(args.data_dir).resolve()
    train, val, test, scenario_path = load_pooled_splits(
        dataset, data_dir, device, scenario_name=getattr(args, "scenario_name", None)
    )
    input_dim_g = int(train.x.shape[1])
    input_dim_f = int(train.z.shape[1])
    hidden_widths = EICU_HIDDEN_WIDTHS if dataset in EICU_DATASETS else ZOO_HIDDEN_WIDTHS
    g, f = build_models(device, input_dim_g, input_dim_f, hidden_widths)
    objective = OptimalMomentObjective()
    g_optimizer = build_optimizer(method, g, args.g_lr, args.weight_decay)
    f_optimizer = build_optimizer(method, f, args.f_lr, args.weight_decay)
    effective_config = build_effective_config(
        args, dataset, method, batch_size, output_dir, device, scenario_path,
        input_dim_g, input_dim_f, hidden_widths,
    )
    write_effective_config(str(output_dir), effective_config)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    mse_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_validation_mse = float("inf")
    best_validation_round = -1
    diverged = False
    finite_history = True
    start_time = time.time()
    final_round = -1

    for round_idx in range(int(args.iterations)):
        final_round = round_idx
        g.train()
        f.train()
        idx = choose_batch_indices(train, batch_size, generator)
        x_batch = train.x[idx]
        z_batch = train.z[idx]
        y_batch = train.y[idx]

        g_obj, f_obj = objective.calc_objective(g, f, x_batch, z_batch, y_batch)
        g_optimizer.zero_grad()
        f_optimizer.zero_grad()
        g_obj.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(g.parameters(), float(args.gradient_clip_norm))
        g_optimizer.step()

        f_optimizer.zero_grad()
        f_obj.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), float(args.gradient_clip_norm))
        f_optimizer.step()

        train_mse = float(structural_mse(g, train))
        val_mse = float(structural_mse(g, val))
        row_finite = finite_state_and_metrics(g, f, [train_mse, val_mse])
        if not row_finite:
            diverged = True
            finite_history = False
        mse_rows.append({
            "round": round_idx,
            "train_mse": train_mse,
            "val_mse": val_mse,
        })
        write_mse_csv(output_dir, mse_rows)

        if args.log_test_mse_by_round:
            test_mse = float(structural_mse(g, test))
            test_finite = math.isfinite(test_mse)
            test_rows.append({
                "round": round_idx,
                "test_mse": test_mse,
                "finite": test_finite,
                "diverged": diverged or not test_finite,
            })
            write_test_mse_by_round(str(output_dir), test_rows)
            if not test_finite:
                finite_history = False
                diverged = True

        if row_finite and val_mse < best_validation_mse:
            best_validation_mse = val_mse
            best_validation_round = round_idx
            best_state = capture_state(g, f)
            save_checkpoint(
                output_dir / "checkpoints" / "best_validation.pt",
                round_idx,
                "best_validation",
                best_state,
                {"train_mse": train_mse, "val_mse": val_mse},
                effective_config,
            )
        if diverged:
            break

    final_state = capture_state(g, f)
    final_train_mse = float(structural_mse(g, train))
    final_val_mse = float(structural_mse(g, val))
    final_test_mse = float(structural_mse(g, test))
    save_checkpoint(
        output_dir / "checkpoints" / "final.pt",
        final_round,
        "final",
        final_state,
        {
            "train_mse": final_train_mse,
            "val_mse": final_val_mse,
            "test_mse": final_test_mse,
        },
        effective_config,
    )

    if best_state is None:
        best_state = copy.deepcopy(final_state)
        best_validation_round = final_round
        best_validation_mse = final_val_mse
        save_checkpoint(
            output_dir / "checkpoints" / "best_validation.pt",
            final_round,
            "best_validation",
            best_state,
            {"train_mse": final_train_mse, "val_mse": final_val_mse},
            effective_config,
        )

    final_prediction = predictions_for_split(g, test)
    load_state(g, f, best_state)
    best_prediction = predictions_for_split(g, test)
    test_mse_at_best_validation = float(structural_mse_from_predictions(best_prediction, test.g))
    save_predictions_npz(str(output_dir), test, best_prediction, final_prediction, effective_config)

    finite_history = finite_history and all(
        math.isfinite(float(row["train_mse"])) and math.isfinite(float(row["val_mse"]))
        for row in mse_rows
    )
    metrics = {
        "train_mse_final": final_train_mse,
        "val_mse_final": final_val_mse,
        "test_mse_final": final_test_mse,
        "final_train_mse": final_train_mse,
        "final_validation_mse": final_val_mse,
        "final_test_mse": final_test_mse,
        "best_validation_round": int(best_validation_round),
        "best_validation_mse": float(best_validation_mse),
        "test_mse_at_best_validation": test_mse_at_best_validation,
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "test_mse_logged_by_round": bool(args.log_test_mse_by_round),
        "diverged": bool(diverged),
        "finite_history": bool(finite_history),
        "completed_iterations": len(mse_rows),
        "runtime_seconds": time.time() - start_time,
    }
    write_metrics(str(output_dir), metrics)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        help="One of abs, step, linear, sin, or eicu_semisynth (with --scenario-name)",
    )
    parser.add_argument(
        "--scenario-name", default=None,
        help="Required for --dataset eicu_semisynth, e.g. linear_seed0",
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None, help="0 means full-batch. Defaults: gda=0, sgda/oadam=256.")
    parser.add_argument("--g-lr", type=float, default=0.001)
    parser.add_argument("--f-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--data-dir", default=str(EXAMPLE_ROOT / "data"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--log-test-mse-by-round", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = run(args)
    print(f"centralized run complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
