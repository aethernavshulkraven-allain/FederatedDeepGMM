#!/usr/bin/env python3
"""Queue federated runs from a rerun protocol manifest.

This launcher is intentionally conservative:

* centralized rows are skipped until their runner is verified,
* real launches require concrete learning-rate and weight-decay values,
* dry-runs generate YAML configs and print commands without training,
* completed runs can be skipped after artifact validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "manifest.csv"
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "generated_configs"
DEFAULT_MAIN = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example" / "main.py"
FEDML_LOG_DIR_ENV = "FEDGMM_FEDML_LOG_DIR"
FEDML_TRACE_DIR_ENV = "FEDGMM_FEDML_TRACE_DIR"

SUPPORTED_FEDERATED_METHODS = {"fedgda_d", "fedgda_s", "fedogda_d", "fedogda_s"}
EXPECTED_ARTIFACTS = (
    "effective_config.json",
    "mse_by_round.csv",
    "metrics.json",
    "predictions.npz",
    os.path.join("checkpoints", "best_validation.pt"),
    os.path.join("checkpoints", "final.pt"),
)


class ManifestLaunchError(Exception):
    pass


@dataclass
class Job:
    row: dict[str, str]
    config: dict[str, Any]
    config_path: Path
    run_dir: Path
    command: list[str]
    env: dict[str, str]
    gpu_id: int


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).strip().lower() == "na"


def _as_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ManifestLaunchError(f"{field} must be numeric, got {value!r}")
    if not math.isfinite(number):
        raise ManifestLaunchError(f"{field} must be finite, got {value!r}")
    return number


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ManifestLaunchError(f"{field} must be an integer, got {value!r}")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_only(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ManifestLaunchError(f"--only must be KEY=VALUE, got {value!r}")
        key, expected = value.split("=", 1)
        if not key or expected == "":
            raise ManifestLaunchError(f"--only must be KEY=VALUE, got {value!r}")
        filters[key] = expected
    return filters


def _row_matches_filters(row: dict[str, str], filters: dict[str, str]) -> bool:
    return all(str(row.get(key, "")) == expected for key, expected in filters.items())


def select_rows(rows: list[dict[str, str]], filters: dict[str, str], include_blocked: bool) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("training_scope") != "federated":
            continue
        if row.get("method") not in SUPPORTED_FEDERATED_METHODS:
            continue
        if not include_blocked and str(row.get("run_status", "")).lower() == "blocked":
            continue
        if not include_blocked and "blocked" in str(row.get("implementation_status", "")).lower():
            continue
        if not _row_matches_filters(row, filters):
            continue
        selected.append(row)
    return selected


def _config_value(row: dict[str, str], key: str, default: Any = None) -> Any:
    value = row.get(key)
    if _blank(value):
        return default
    return value


def _resolve_learning_rate(row: dict[str, str], default_learning_rate: float | None) -> float:
    value = _config_value(row, "learning_rate")
    if value is None:
        if default_learning_rate is None:
            raise ManifestLaunchError(f"{row['run_id']} is missing learning_rate")
        return float(default_learning_rate)
    return _as_float(value, "learning_rate")


def _resolve_weight_decay(row: dict[str, str], default_weight_decay: float | None) -> float:
    value = _config_value(row, "weight_decay")
    if value is None:
        if default_weight_decay is None:
            raise ManifestLaunchError(f"{row['run_id']} is missing weight_decay")
        return float(default_weight_decay)
    return _as_float(value, "weight_decay")


def _run_dir(output_root: Path, row: dict[str, str]) -> Path:
    return (
        output_root
        / row["dataset"]
        / row["method"]
        / f"seed_{int(row['seed'])}"
        / row["run_id"]
    )


def _log_dir(row: dict[str, str]) -> Path:
    return REPO_ROOT / "logs" / "fedml" / row["method"] / row["run_id"]


def _trace_dir(row: dict[str, str]) -> Path:
    return REPO_ROOT / "logs" / "fedml_trace" / row["method"] / row["run_id"]


def build_config(
    row: dict[str, str],
    *,
    output_root: Path,
    gpu_id: int,
    default_learning_rate: float | None,
    default_weight_decay: float | None,
    override_comm_round: int | None,
    override_epochs: int | None,
    override_simple_model_selection_epochs: int | None,
    override_f_history_model_selection_epochs: int | None,
    override_model_selection_batch_size: int | None,
    override_model_selection_max_samples: int | None,
    override_skip_model_selection: bool | None,
    override_skip_gmm_eval: bool | None,
    override_auxiliary_regression: bool | None,
    override_auxiliary_regression_epochs: int | None,
    override_append_round_csv: bool | None,
    override_periodic_checkpoint_interval: int | None,
    override_dataloader_num_workers: int | None,
    override_dataloader_pin_memory: bool | None,
) -> dict[str, Any]:
    learning_rate = _resolve_learning_rate(row, default_learning_rate)
    weight_decay = _resolve_weight_decay(row, default_weight_decay)
    test_mse_used_for_selection = _truthy(_config_value(row, "test_mse_used_for_selection", False))
    selection_metric_source = str(_config_value(row, "selection_metric_source", "validation"))
    if test_mse_used_for_selection:
        raise ManifestLaunchError(f"{row['run_id']} attempts to use Test MSE for selection")
    if selection_metric_source.lower() != "validation":
        raise ManifestLaunchError(f"{row['run_id']} selection_metric_source must be validation")
    comm_round = int(override_comm_round) if override_comm_round is not None else _as_int(row["comm_round"], "comm_round")
    epochs = int(override_epochs) if override_epochs is not None else _as_int(row["epochs"], "epochs")
    skip_model_selection = (
        _truthy(_config_value(row, "skip_model_selection", False))
        if override_skip_model_selection is None
        else bool(override_skip_model_selection)
    )
    if override_skip_gmm_eval is not None:
        skip_gmm_eval = bool(override_skip_gmm_eval)
    elif skip_model_selection:
        skip_gmm_eval = True
    else:
        skip_gmm_eval = _truthy(_config_value(row, "skip_gmm_eval", False))
    if skip_model_selection:
        skip_gmm_eval = True
    auxiliary_regression = (
        _truthy(_config_value(row, "auxiliary_regression", True))
        if override_auxiliary_regression is None
        else bool(override_auxiliary_regression)
    )
    auxiliary_regression_epochs_default = epochs if auxiliary_regression else 0
    auxiliary_regression_epochs = (
        int(override_auxiliary_regression_epochs)
        if override_auxiliary_regression_epochs is not None
        else _as_int(
            _config_value(row, "auxiliary_regression_epochs", auxiliary_regression_epochs_default),
            "auxiliary_regression_epochs",
        )
    )
    append_round_csv = (
        _truthy(_config_value(row, "append_round_csv", True))
        if override_append_round_csv is None
        else bool(override_append_round_csv)
    )
    periodic_checkpoint_interval = (
        int(override_periodic_checkpoint_interval)
        if override_periodic_checkpoint_interval is not None
        else _as_int(_config_value(row, "periodic_checkpoint_interval", 200), "periodic_checkpoint_interval")
    )
    dataloader_num_workers = (
        int(override_dataloader_num_workers)
        if override_dataloader_num_workers is not None
        else _as_int(_config_value(row, "dataloader_num_workers", 0), "dataloader_num_workers")
    )
    if dataloader_num_workers < 0:
        raise ManifestLaunchError("dataloader_num_workers must be >= 0")
    if auxiliary_regression_epochs < 0:
        raise ManifestLaunchError("auxiliary_regression_epochs must be >= 0")
    if periodic_checkpoint_interval < 0:
        raise ManifestLaunchError("periodic_checkpoint_interval must be >= 0")
    gmm_eval_proxy = (
        "negative_val_mse"
        if skip_gmm_eval
        else str(_config_value(row, "gmm_eval_proxy", "approx_psi"))
    )
    dataloader_pin_memory = (
        _truthy(_config_value(row, "dataloader_pin_memory", False))
        if override_dataloader_pin_memory is None
        else bool(override_dataloader_pin_memory)
    )
    return {
        "dataset": row["dataset"],
        "model": _config_value(row, "model", "lr"),
        "federated_optimizer": _config_value(row, "federated_optimizer", "FedAvg"),
        "client_id_list": "[]",
        "client_num_in_total": _as_int(row["client_num_in_total"], "client_num_in_total"),
        "client_num_per_round": _as_int(row["client_num_per_round"], "client_num_per_round"),
        "comm_round": comm_round,
        "epochs": epochs,
        "batch_size": _as_int(row["batch_size"], "batch_size"),
        "client_optimizer": row["client_optimizer"],
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "critic_multiplier": _as_float(_config_value(row, "critic_multiplier", 10.0), "critic_multiplier"),
        "server_learning_rate": _as_float(_config_value(row, "server_learning_rate", 1.5), "server_learning_rate"),
        "objective_lambda_1": _as_float(_config_value(row, "objective_lambda_1", 0.1), "objective_lambda_1"),
        "gradient_clip_norm": _as_float(_config_value(row, "gradient_clip_norm", 1.0), "gradient_clip_norm"),
        "simple_model_selection_epochs": (
            int(override_simple_model_selection_epochs)
            if override_simple_model_selection_epochs is not None
            else _as_int(_config_value(row, "simple_model_selection_epochs", 100), "simple_model_selection_epochs")
        ),
        "f_history_model_selection_epochs": (
            int(override_f_history_model_selection_epochs)
            if override_f_history_model_selection_epochs is not None
            else _as_int(_config_value(row, "f_history_model_selection_epochs", 60), "f_history_model_selection_epochs")
        ),
        "model_selection_batch_size": (
            int(override_model_selection_batch_size)
            if override_model_selection_batch_size is not None
            else _as_int(_config_value(row, "model_selection_batch_size", 200), "model_selection_batch_size")
        ),
        "model_selection_max_samples": (
            int(override_model_selection_max_samples)
            if override_model_selection_max_samples is not None
            else _as_int(_config_value(row, "model_selection_max_samples", 0), "model_selection_max_samples")
        ),
        "log_test_mse_by_round": _truthy(_config_value(row, "log_test_mse_by_round", False)),
        "test_mse_used_for_selection": test_mse_used_for_selection,
        "selection_metric_source": selection_metric_source,
        "skip_model_selection": skip_model_selection,
        "skip_gmm_eval": skip_gmm_eval,
        "gmm_eval_proxy": gmm_eval_proxy,
        "auxiliary_regression": auxiliary_regression,
        "auxiliary_regression_epochs": auxiliary_regression_epochs,
        "auxiliary_regression_state_device": str(_config_value(row, "auxiliary_regression_state_device", "device")),
        "append_round_csv": append_round_csv,
        "periodic_checkpoint_interval": periodic_checkpoint_interval,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        "frequency_of_the_test": 1,
        "random_seed": _as_int(row["seed"], "seed"),
        "partition_method": _config_value(row, "partition_method", "hetero"),
        "partition_alpha": _as_float(row["partition_alpha"], "partition_alpha"),
        "data_cache_dir": _config_value(row, "data_cache_dir", "data"),
        "run_id": row["run_id"],
        "output_dir": str(output_root),
        "using_gpu": _truthy(_config_value(row, "using_gpu", True)),
        "gpu_id": int(gpu_id),
        "enable_legacy_outputs": False,
        "overwrite": False,
        "require_multibatch_stochastic": False,
        "variant": row["method"],
        # eicu_semisynth-specific; absent/default for every other row, so this
        # is a pure addition with no effect on existing (non-eicu) manifests.
        "scenario_name": _config_value(row, "scenario_name", ""),
        "objective_mode": _config_value(row, "objective_mode", "legacy"),
        "aggregation_weighting": _config_value(row, "aggregation_weighting", "sample_size"),
        "input_dim_g": _as_int(_config_value(row, "input_dim_g", 0), "input_dim_g"),
        "input_dim_f": _as_int(_config_value(row, "input_dim_f", 0), "input_dim_f"),
        # protocol_v1.md S7.1: scenario_seed (which DGP/scenario artifact) and
        # optimizer_seed (this run's random_seed) must be recorded separately
        # even though row["seed"] continues to mean "optimizer seed" for path
        #/backward-compatibility purposes (_run_dir, the seed-mismatch check
        # below, and every pre-Study-A manifest already key off row["seed"]).
        # Absent scenario_seed defaults to row["seed"] so non-Study-A
        # manifests keep today's conflated-but-consistent behavior exactly.
        "scenario_seed": _as_int(_config_value(row, "scenario_seed", row["seed"]), "scenario_seed"),
        "optimizer_seed": _as_int(row["seed"], "seed"),
        "seed_pair_id": _config_value(row, "seed_pair_id", ""),
        "campaign_role": _config_value(row, "campaign_role", ""),
        "scenario_checksum": _config_value(row, "scenario_checksum", ""),
        "protocol_version": _config_value(row, "protocol_version", ""),
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = {
        "common_args": {
            "training_type": "simulation",
            "using_mlops": False,
            "random_seed": config["random_seed"],
            "run_id": config["run_id"],
            "config_version": "release",
            "output_dir": config["output_dir"],
            "overwrite": config.get("overwrite", False),
            **({"protocol_version": config["protocol_version"]} if config.get("protocol_version") else {}),
        },
        "data_args": {
            "dataset": config["dataset"],
            "data_cache_dir": config["data_cache_dir"],
            "partition_method": config["partition_method"],
            "partition_alpha": config["partition_alpha"],
            "dataloader_num_workers": config["dataloader_num_workers"],
            "dataloader_pin_memory": config["dataloader_pin_memory"],
            **({"scenario_name": config["scenario_name"]} if config.get("scenario_name") else {}),
            **({"scenario_checksum": config["scenario_checksum"]} if config.get("scenario_checksum") else {}),
        },
        "model_args": {
            "model": config["model"],
            **({"input_dim_g": config["input_dim_g"]} if config.get("input_dim_g") else {}),
            **({"input_dim_f": config["input_dim_f"]} if config.get("input_dim_f") else {}),
        },
        "train_args": {
            "federated_optimizer": config["federated_optimizer"],
            "client_id_list": config["client_id_list"],
            "client_num_in_total": config["client_num_in_total"],
            "client_num_per_round": config["client_num_per_round"],
            "comm_round": config["comm_round"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "client_optimizer": config["client_optimizer"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "critic_multiplier": config["critic_multiplier"],
            "server_learning_rate": config["server_learning_rate"],
            "objective_lambda_1": config["objective_lambda_1"],
            "gradient_clip_norm": config["gradient_clip_norm"],
            "simple_model_selection_epochs": config["simple_model_selection_epochs"],
            "f_history_model_selection_epochs": config["f_history_model_selection_epochs"],
            "model_selection_batch_size": config["model_selection_batch_size"],
            "model_selection_max_samples": config["model_selection_max_samples"],
            "enable_legacy_outputs": config["enable_legacy_outputs"],
            "require_multibatch_stochastic": config["require_multibatch_stochastic"],
            "log_test_mse_by_round": config["log_test_mse_by_round"],
            "test_mse_used_for_selection": config["test_mse_used_for_selection"],
            "selection_metric_source": config["selection_metric_source"],
            "skip_model_selection": config["skip_model_selection"],
            "skip_gmm_eval": config["skip_gmm_eval"],
            "gmm_eval_proxy": config["gmm_eval_proxy"],
            "auxiliary_regression": config["auxiliary_regression"],
            "auxiliary_regression_epochs": config["auxiliary_regression_epochs"],
            "auxiliary_regression_state_device": config["auxiliary_regression_state_device"],
            "append_round_csv": config["append_round_csv"],
            "periodic_checkpoint_interval": config["periodic_checkpoint_interval"],
            "objective_mode": config["objective_mode"],
            "aggregation_weighting": config["aggregation_weighting"],
            "scenario_seed": config["scenario_seed"],
            "optimizer_seed": config["optimizer_seed"],
            **({"seed_pair_id": config["seed_pair_id"]} if config.get("seed_pair_id") else {}),
            **({"campaign_role": config["campaign_role"]} if config.get("campaign_role") else {}),
        },
        "validation_args": {
            "frequency_of_the_test": config["frequency_of_the_test"],
        },
        "device_args": {
            "using_gpu": config["using_gpu"],
            "gpu_id": config["gpu_id"],
        },
        "comm_args": {
            "backend": "sp",
        },
        "tracking_args": {
            "enable_tracking": False,
            "enable_wandb": False,
            "run_name": f"{config['dataset']}_{config['variant']}_seed_{config['random_seed']}_{config['run_id']}",
        },
    }
    with path.open("w") as f:
        for section, values in sections.items():
            f.write(f"{section}:\n")
            for key, value in values.items():
                f.write(f"  {key}: {_yaml_scalar(value)}\n")
            f.write("\n")


def _json_numbers_are_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def validate_artifacts(run_dir: Path, row: dict[str, str]) -> dict[str, Any]:
    missing = [name for name in EXPECTED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise ManifestLaunchError(f"missing artifacts: {', '.join(missing)}")
    with (run_dir / "effective_config.json").open("r") as f:
        config = json.load(f)
    with (run_dir / "metrics.json").open("r") as f:
        metrics = json.load(f)
    if config.get("dataset") != row["dataset"]:
        raise ManifestLaunchError(f"dataset mismatch: {config.get('dataset')} != {row['dataset']}")
    if config.get("variant") != row["method"]:
        raise ManifestLaunchError(f"variant mismatch: {config.get('variant')} != {row['method']}")
    if int(config.get("random_seed")) != int(row["seed"]):
        raise ManifestLaunchError(f"seed mismatch: {config.get('random_seed')} != {row['seed']}")
    if str(config.get("run_id")) != row["run_id"]:
        raise ManifestLaunchError(f"run_id mismatch: {config.get('run_id')} != {row['run_id']}")
    if not _json_numbers_are_finite(metrics):
        raise ManifestLaunchError("metrics contain non-finite numeric values")
    if bool(metrics.get("diverged", False)):
        raise ManifestLaunchError("metrics.diverged is true")
    if bool(config.get("test_mse_used_for_selection", False)):
        raise ManifestLaunchError("config.test_mse_used_for_selection must be false")
    if str(config.get("selection_metric_source", "validation")).lower() != "validation":
        raise ManifestLaunchError("config.selection_metric_source must be validation")
    if bool(config.get("log_test_mse_by_round", False)):
        test_curve_path = run_dir / "test_mse_by_round.csv"
        if not test_curve_path.exists():
            raise ManifestLaunchError("missing test_mse_by_round.csv")
        with test_curve_path.open("r", newline="") as f:
            test_rows = list(csv.DictReader(f))
        if len(test_rows) != int(config.get("comm_round")):
            raise ManifestLaunchError(
                f"test_mse_by_round row count {len(test_rows)} != comm_round {config.get('comm_round')}"
            )
        if bool(metrics.get("test_mse_used_for_selection", False)):
            raise ManifestLaunchError("metrics.test_mse_used_for_selection must be false")
        if str(metrics.get("selection_metric_source", "validation")).lower() != "validation":
            raise ManifestLaunchError("metrics.selection_metric_source must be validation")
        for index, test_row in enumerate(test_rows):
            try:
                test_mse = float(test_row.get("test_mse"))
            except (TypeError, ValueError):
                raise ManifestLaunchError(f"test_mse_by_round[{index}].test_mse is not numeric")
            if not math.isfinite(test_mse):
                raise ManifestLaunchError(f"test_mse_by_round[{index}].test_mse is not finite")
            if str(test_row.get("finite")) != "True":
                raise ManifestLaunchError(f"test_mse_by_round[{index}].finite is not True")
            if str(test_row.get("diverged")) != "False":
                raise ManifestLaunchError(f"test_mse_by_round[{index}].diverged is not False")
    for key in ("test_mse_at_best_validation", "final_test_mse"):
        if key in metrics and not math.isfinite(float(metrics[key])):
            raise ManifestLaunchError(f"{key} is not finite")
    return {
        "run_dir": str(run_dir),
        "test_mse_at_best_validation": metrics.get("test_mse_at_best_validation"),
        "final_test_mse": metrics.get("final_test_mse"),
        "best_validation_round": metrics.get("best_validation_round"),
    }


def has_completed_artifacts(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in EXPECTED_ARTIFACTS)


def has_any_artifacts(run_dir: Path) -> bool:
    return run_dir.exists() and any(run_dir.iterdir())


def assign_job_gpu(job: Job, gpu_id: int) -> None:
    """Assign a physical GPU to a job immediately before launch.

    GPU assignment must happen after resume/skip decisions. Otherwise a skipped
    job can consume a round-robin slot and two live jobs can accidentally land
    on the same GPU.
    """

    job.gpu_id = int(gpu_id)
    job.config["gpu_id"] = int(gpu_id)
    write_config(job.config_path, job.config)


def build_jobs(
    rows: list[dict[str, str]],
    *,
    python_executable: str,
    main_path: Path,
    config_dir: Path,
    output_root: Path,
    gpu_ids: list[int],
    default_learning_rate: float | None,
    default_weight_decay: float | None,
    override_comm_round: int | None,
    override_epochs: int | None,
    override_simple_model_selection_epochs: int | None,
    override_f_history_model_selection_epochs: int | None,
    override_model_selection_batch_size: int | None,
    override_model_selection_max_samples: int | None,
    override_skip_model_selection: bool | None,
    override_skip_gmm_eval: bool | None,
    override_auxiliary_regression: bool | None,
    override_auxiliary_regression_epochs: int | None,
    override_append_round_csv: bool | None,
    override_periodic_checkpoint_interval: int | None,
    override_dataloader_num_workers: int | None,
    override_dataloader_pin_memory: bool | None,
) -> tuple[list[Job], list[dict[str, str]]]:
    jobs: list[Job] = []
    skipped: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        try:
            config = build_config(
                row,
                output_root=output_root,
                gpu_id=gpu_id,
                default_learning_rate=default_learning_rate,
                default_weight_decay=default_weight_decay,
                override_comm_round=override_comm_round,
                override_epochs=override_epochs,
                override_simple_model_selection_epochs=override_simple_model_selection_epochs,
                override_f_history_model_selection_epochs=override_f_history_model_selection_epochs,
                override_model_selection_batch_size=override_model_selection_batch_size,
                override_model_selection_max_samples=override_model_selection_max_samples,
                override_skip_model_selection=override_skip_model_selection,
                override_skip_gmm_eval=override_skip_gmm_eval,
                override_auxiliary_regression=override_auxiliary_regression,
                override_auxiliary_regression_epochs=override_auxiliary_regression_epochs,
                override_append_round_csv=override_append_round_csv,
                override_periodic_checkpoint_interval=override_periodic_checkpoint_interval,
                override_dataloader_num_workers=override_dataloader_num_workers,
                override_dataloader_pin_memory=override_dataloader_pin_memory,
            )
        except ManifestLaunchError as exc:
            skipped.append({"run_id": row.get("run_id", ""), "reason": str(exc)})
            continue
        config_path = config_dir / row["dataset"] / row["method"] / f"seed_{int(row['seed'])}" / f"{row['run_id']}.yaml"
        write_config(config_path, config)
        run_dir = _run_dir(output_root, row)
        command = [python_executable, str(main_path), "--cf", str(config_path)]
        env = dict(os.environ)
        env[FEDML_LOG_DIR_ENV] = str(_log_dir(row))
        env[FEDML_TRACE_DIR_ENV] = str(_trace_dir(row))
        jobs.append(Job(row=row, config=config, config_path=config_path, run_dir=run_dir, command=command, env=env, gpu_id=gpu_id))
    return jobs, skipped


def dry_run(jobs: list[Job], skipped: list[dict[str, str]], limit: int | None) -> None:
    shown = jobs if limit is None else jobs[:limit]
    for job in shown:
        print(f"{job.row['run_id']} -> {job.run_dir}")
        print(f"gpu_id={job.gpu_id}")
        print(f"config={job.config_path}")
        print(" ".join(job.command))
    if skipped:
        print("Skipped unlaunchable rows:")
        for item in skipped[:20]:
            print(f"  {item['run_id']}: {item['reason']}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")
    print(json.dumps({
        "dry_run": True,
        "launchable": len(jobs),
        "skipped_unlaunchable": len(skipped),
        "shown": len(shown),
    }, indent=2, sort_keys=True))


def run_jobs(
    jobs: list[Job],
    *,
    gpu_ids: list[int],
    max_parallel: int,
    resume_skip_completed: bool,
    overwrite_incomplete: bool,
    stop_on_failure: bool,
) -> list[dict[str, Any]]:
    pending = list(jobs)
    active: list[tuple[Job, subprocess.Popen[str]]] = []
    results: list[dict[str, Any]] = []
    stop_starting = False
    parallel_limit = min(int(max_parallel), len(gpu_ids))

    while pending or active:
        while pending and len(active) < parallel_limit and not stop_starting:
            active_gpu_ids = {job.gpu_id for job, _ in active}
            free_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in active_gpu_ids]
            if not free_gpu_ids:
                break
            job = pending.pop(0)
            if resume_skip_completed and has_completed_artifacts(job.run_dir):
                try:
                    validation = validate_artifacts(job.run_dir, job.row)
                    results.append({"run_id": job.row["run_id"], "status": "skipped_completed", **validation})
                    print(f"SKIP completed {job.row['run_id']}")
                    continue
                except ManifestLaunchError as exc:
                    print(f"Existing artifacts invalid for {job.row['run_id']}: {exc}; rerunning")
            if has_any_artifacts(job.run_dir):
                if not overwrite_incomplete:
                    results.append({
                        "run_id": job.row["run_id"],
                        "status": "failed_incomplete_artifacts",
                        "error": "run directory contains incomplete artifacts; pass --overwrite-incomplete to rerun",
                        "run_dir": str(job.run_dir),
                    })
                    print(f"FAIL {job.row['run_id']} incomplete artifacts require --overwrite-incomplete")
                    if stop_on_failure:
                        stop_starting = True
                    continue
                job.config["overwrite"] = True
            job.run_dir.parent.mkdir(parents=True, exist_ok=True)
            _log_dir(job.row).mkdir(parents=True, exist_ok=True)
            _trace_dir(job.row).mkdir(parents=True, exist_ok=True)
            assign_job_gpu(job, free_gpu_ids[0])
            print(f"START {job.row['run_id']} gpu={job.gpu_id}")
            process = subprocess.Popen(
                job.command,
                cwd=str(REPO_ROOT),
                env=job.env,
                text=True,
            )
            active.append((job, process))

        still_active: list[tuple[Job, subprocess.Popen[str]]] = []
        for job, process in active:
            returncode = process.poll()
            if returncode is None:
                still_active.append((job, process))
                continue
            if returncode != 0:
                results.append({
                    "run_id": job.row["run_id"],
                    "status": "failed_process",
                    "returncode": returncode,
                    "run_dir": str(job.run_dir),
                })
                print(f"FAIL {job.row['run_id']} returncode={returncode}")
                if stop_on_failure:
                    stop_starting = True
                continue
            try:
                validation = validate_artifacts(job.run_dir, job.row)
                results.append({"run_id": job.row["run_id"], "status": "passed", **validation})
                print(f"PASS {job.row['run_id']}")
            except ManifestLaunchError as exc:
                results.append({
                    "run_id": job.row["run_id"],
                    "status": "failed_validation",
                    "error": str(exc),
                    "run_dir": str(job.run_dir),
                })
                print(f"VALIDATION FAIL {job.row['run_id']}: {exc}")
                if stop_on_failure:
                    stop_starting = True
        active = still_active
        if stop_starting and not active:
            for job in pending:
                results.append({
                    "run_id": job.row["run_id"],
                    "status": "not_started_after_failure",
                    "run_dir": str(job.run_dir),
                })
            pending = []
        if active:
            time.sleep(5)

    return results


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue federated runs from a protocol manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--main", default=str(DEFAULT_MAIN))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-root", default="results/rerun_protocol_v1")
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--only", action="append", default=[], help="Filter rows with KEY=VALUE; repeatable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-skip-completed", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--default-learning-rate", type=float, default=None)
    parser.add_argument("--default-weight-decay", type=float, default=None)
    parser.add_argument("--override-comm-round", type=int, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-simple-model-selection-epochs", type=int, default=None)
    parser.add_argument("--override-f-history-model-selection-epochs", type=int, default=None)
    parser.add_argument("--override-model-selection-batch-size", type=int, default=None)
    parser.add_argument("--override-model-selection-max-samples", type=int, default=None)
    parser.add_argument("--skip-model-selection", dest="override_skip_model_selection", action="store_true", default=None)
    parser.add_argument("--skip-gmm-eval", dest="override_skip_gmm_eval", action="store_true", default=None)
    parser.add_argument("--disable-auxiliary-regression", dest="override_auxiliary_regression", action="store_false", default=None)
    parser.add_argument("--enable-auxiliary-regression", dest="override_auxiliary_regression", action="store_true")
    parser.add_argument("--override-auxiliary-regression-epochs", type=int, default=None)
    parser.add_argument("--rewrite-round-csv", dest="override_append_round_csv", action="store_false", default=None)
    parser.add_argument("--override-periodic-checkpoint-interval", type=int, default=None)
    parser.add_argument("--override-dataloader-num-workers", type=int, default=None)
    parser.add_argument("--dataloader-pin-memory", dest="override_dataloader_pin_memory", action="store_true", default=None)
    parser.add_argument("--results-json", default="experiments/rerun_protocol_v1/launcher_results.json")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    parser.add_argument("--keep-going", dest="stop_on_failure", action="store_false")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    main_path = Path(args.main)
    config_dir = Path(args.config_dir)
    output_root = Path(args.output_root)
    results_json = Path(args.results_json)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not main_path.is_absolute():
        main_path = REPO_ROOT / main_path
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if not results_json.is_absolute():
        results_json = REPO_ROOT / results_json

    gpu_ids = [int(item.strip()) for item in str(args.gpu_ids).split(",") if item.strip() != ""]
    if not gpu_ids:
        raise SystemExit("--gpu-ids must contain at least one GPU id")
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")

    filters = _parse_only(args.only)
    selected = select_rows(_load_rows(manifest_path), filters, bool(args.include_blocked))
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    jobs, skipped = build_jobs(
        selected,
        python_executable=args.python,
        main_path=main_path,
        config_dir=config_dir,
        output_root=output_root,
        gpu_ids=gpu_ids,
        default_learning_rate=args.default_learning_rate,
        default_weight_decay=args.default_weight_decay,
        override_comm_round=args.override_comm_round,
        override_epochs=args.override_epochs,
        override_simple_model_selection_epochs=args.override_simple_model_selection_epochs,
        override_f_history_model_selection_epochs=args.override_f_history_model_selection_epochs,
        override_model_selection_batch_size=args.override_model_selection_batch_size,
        override_model_selection_max_samples=args.override_model_selection_max_samples,
        override_skip_model_selection=args.override_skip_model_selection,
        override_skip_gmm_eval=args.override_skip_gmm_eval,
        override_auxiliary_regression=args.override_auxiliary_regression,
        override_auxiliary_regression_epochs=args.override_auxiliary_regression_epochs,
        override_append_round_csv=args.override_append_round_csv,
        override_periodic_checkpoint_interval=args.override_periodic_checkpoint_interval,
        override_dataloader_num_workers=args.override_dataloader_num_workers,
        override_dataloader_pin_memory=args.override_dataloader_pin_memory,
    )
    if args.dry_run:
        dry_run(jobs, skipped, None)
        return 0
    if skipped:
        print("Refusing real launch because selected rows are missing required hyperparameters:", file=sys.stderr)
        for item in skipped[:20]:
            print(f"  {item['run_id']}: {item['reason']}", file=sys.stderr)
        return 1
    if not jobs:
        print("No launchable jobs selected.", file=sys.stderr)
        return 1
    results = run_jobs(
        jobs,
        gpu_ids=gpu_ids,
        max_parallel=int(args.max_parallel),
        resume_skip_completed=bool(args.resume_skip_completed),
        overwrite_incomplete=bool(args.overwrite_incomplete),
        stop_on_failure=bool(args.stop_on_failure),
    )
    _write_results(results_json, results)
    failed = [row for row in results if row.get("status") not in {"passed", "skipped_completed"}]
    print(json.dumps({
        "jobs": len(jobs),
        "results_json": str(results_json.relative_to(REPO_ROOT)),
        "passed": sum(row.get("status") == "passed" for row in results),
        "skipped_completed": sum(row.get("status") == "skipped_completed" for row in results),
        "failed": len(failed),
    }, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
