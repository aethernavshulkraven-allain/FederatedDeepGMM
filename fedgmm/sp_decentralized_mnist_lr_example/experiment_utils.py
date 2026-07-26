import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import nullcontext


ZOO_DATASETS = {"linear", "abs", "sin", "step", "zoo"}
# "uniform_clients" matches the paper's federated objective U = (1/N) sum_i U^i:
# every client participating in a round gets weight 1/K_t. "sample_size" is the
# legacy FedAvg-style weighting (weight_i = n_i / sum_j n_j) used by every result
# in the repo before this option existed; it is kept, not removed, so those
# results stay reproducible. Absence of the key defaults to "sample_size" for the
# same reason.
AGGREGATION_WEIGHTING_CHOICES = ("uniform_clients", "sample_size")
DEFAULT_AGGREGATION_WEIGHTING = "sample_size"
# "paper_aligned" uses a frozen previous-global-iterate theta~ and a fixed
# lambda=1/4 in the critic regularizer (see PaperAlignedMomentObjective).
# "legacy" is today's OptimalMomentObjective, completely unmodified: reuses
# the live theta for the regularizer and a tunable objective_lambda_1
# (default 0.1). Absence of the key defaults to "legacy" so every existing
# result stays reproducible.
OBJECTIVE_MODE_CHOICES = ("legacy", "paper_aligned")
DEFAULT_OBJECTIVE_MODE = "legacy"
PAPER_ALIGNED_LAMBDA = 0.25
PROFILE_ENV_VAR = "FEDGMM_PROFILE_RUNTIME"
PROFILE_ROOT_ENV_VAR = "FEDGMM_PROFILE_ROOT"
PROFILE_TELEMETRY_ENV_VAR = "FEDGMM_PROFILE_GPU_TELEMETRY"
PROFILE_BATCHES_ENV_VAR = "FEDGMM_PROFILE_BATCHES"
PROFILE_INTERVAL_ENV_VAR = "FEDGMM_PROFILE_GPU_INTERVAL_SECONDS"
PROFILE_SKIP_AUX_REG_ENV_VAR = "FEDGMM_PROFILE_SKIP_AUX_REG"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PROFILE_ROOT = os.path.join(
    REPO_ROOT,
    "results",
    "_profiling",
    "highdim_stochastic_gpu_util",
)
PREDICTION_OUTPUT_KEYS = ("best_validation_prediction", "final_prediction")
RUN_ARTIFACT_NAMES = {
    "effective_config.json",
    "mse_by_round.csv",
    "metrics.json",
    "predictions.npz",
    "test_mse_by_round.csv",
}
EFFECTIVE_CONFIG_FIELDS = (
    "dataset",
    "algorithm",
    "variant",
    "mode",
    "model",
    "federated_optimizer",
    "client_optimizer",
    "learning_rate",
    "g_learning_rate",
    "critic_multiplier",
    "f_learning_rate",
    "server_learning_rate",
    "objective_lambda_1",
    "gradient_clip_norm",
    "weight_decay",
    "batch_size",
    "require_multibatch_stochastic",
    "epochs",
    "local_epochs",
    "comm_round",
    "client_num_in_total",
    "client_num_per_round",
    "partition_method",
    "partition_alpha",
    "data_cache_dir",
    "random_seed",
    "run_id",
    "output_dir",
    "using_gpu",
    "gpu_id",
    "frequency_of_the_test",
    "eval_freq",
    "print_freq",
    "simple_model_selection_epochs",
    "f_history_model_selection_epochs",
    "model_selection_batch_size",
    "model_selection_max_samples",
    "skip_model_selection",
    "skip_gmm_eval",
    "gmm_eval_proxy",
    "auxiliary_regression",
    "auxiliary_regression_epochs",
    "auxiliary_regression_state_device",
    "append_round_csv",
    "periodic_checkpoint_interval",
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "log_test_mse_by_round",
    "test_mse_logged_by_round",
    "test_mse_used_for_selection",
    "selection_metric_source",
    "aggregation_weighting",
    "objective_mode",
    "input_dim_g",
    "input_dim_f",
    "enable_legacy_outputs",
    "enable_legacy_plot",
    "overwrite",
)


def compute_client_weights(sample_counts, mode):
    """Per-participating-client aggregation weights, summing to 1.0.

    ``uniform_clients``: every client gets ``1/K`` regardless of its sample
    count, where ``K = len(sample_counts)`` is however many clients actually
    participated this round (so this is correct under partial participation
    too, not just full participation).

    ``sample_size``: ``weight_i = n_i / sum_j n_j`` — the legacy FedAvg
    weighting. Mathematically identical to the pre-existing inline computation
    it replaces, so results in this mode are bit-for-bit reproducible.
    """
    counts = [int(c) for c in sample_counts]
    if not counts:
        raise ValueError("compute_client_weights requires at least one client")
    if mode == "uniform_clients":
        k = len(counts)
        return [1.0 / k] * k
    if mode == "sample_size":
        total = sum(counts)
        if total <= 0:
            raise ValueError(
                "sample_size aggregation requires a positive total sample count, "
                f"got counts={counts}"
            )
        return [c / total for c in counts]
    raise ValueError(
        f"aggregation_weighting must be one of {AGGREGATION_WEIGHTING_CHOICES}, "
        f"got {mode!r}"
    )


def weighted_average_state_dicts(state_dicts, weights):
    """Weighted sum of same-keyed mappings (torch state dicts or plain numbers).

    This is the single function both FedGDA's direct aggregation and FedOGDA's
    pseudogradient/delta aggregation rely on (the delta is computed as
    ``weighted_average(...) - theta_t``), so fixing the weighting here fixes
    both optimizers identically rather than needing two separate patches.
    """
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"state_dicts and weights must have the same length, got "
            f"{len(state_dicts)} and {len(weights)}"
        )
    if not state_dicts:
        raise ValueError("weighted_average_state_dicts requires at least one entry")

    result = {}
    for key in state_dicts[0].keys():
        acc = None
        for state_dict, weight in zip(state_dicts, weights):
            term = state_dict[key] * weight
            acc = term if acc is None else acc + term
        result[key] = acc
    return result


def default_critic_multiplier(dataset):
    multiplier = 20.0
    if dataset in ZOO_DATASETS:
        multiplier = 10.0
    if dataset in {
        "mnist_z",
        "femnist_z",
        "mnist_xz",
        "femnist_xz",
        "cifar10_z",
        "cifar10_xz",
        "cifar_xz",
    }:
        multiplier = 5.0
    elif dataset in {"mnist_x", "femnist_x", "cifar10_x", "cifar_x"}:
        multiplier = 3.0
    return multiplier


def get_effective_config(args):
    dataset = getattr(args, "dataset", "unknown")
    client_optimizer = getattr(args, "client_optimizer", "sgd")
    optimizer_name = str(client_optimizer).lower()
    algorithm = "fedogda" if optimizer_name == "ogda" else "fedgda"
    batch_size = getattr(args, "batch_size", 0)
    mode = "d" if batch_size is None or int(batch_size) <= 0 else "s"
    seed = int(getattr(args, "random_seed", 0))
    run_id = str(getattr(args, "run_id", "0"))
    output_dir = getattr(args, "output_dir", "results")
    variant = f"{algorithm}_{mode}"
    critic_multiplier = float(
        getattr(args, "critic_multiplier", default_critic_multiplier(dataset))
    )
    server_learning_rate = float(getattr(args, "server_learning_rate", 1.5))
    objective_lambda_1 = float(getattr(args, "objective_lambda_1", 0.1))
    aggregation_weighting = str(
        getattr(args, "aggregation_weighting", DEFAULT_AGGREGATION_WEIGHTING)
    )
    objective_mode = str(getattr(args, "objective_mode", DEFAULT_OBJECTIVE_MODE))

    learning_rate = float(getattr(args, "learning_rate", 0.0))
    log_test_mse_by_round = bool(getattr(args, "log_test_mse_by_round", False))
    test_mse_used_for_selection = bool(getattr(args, "test_mse_used_for_selection", False))
    selection_metric_source = str(getattr(args, "selection_metric_source", "validation"))
    auxiliary_regression = bool(getattr(args, "auxiliary_regression", True))
    auxiliary_regression_epochs = int(
        getattr(args, "auxiliary_regression_epochs", int(getattr(args, "epochs", 0)))
    )
    skip_model_selection = bool(getattr(args, "skip_model_selection", False))
    skip_gmm_eval = bool(getattr(args, "skip_gmm_eval", skip_model_selection))
    if skip_model_selection:
        skip_gmm_eval = True
    gmm_eval_proxy = "negative_val_mse" if skip_gmm_eval else "approx_psi"
    if not skip_model_selection:
        gmm_eval_proxy = str(getattr(args, "gmm_eval_proxy", gmm_eval_proxy))
    return json_safe({
        "dataset": dataset,
        "algorithm": algorithm,
        "variant": variant,
        "mode": "deterministic" if mode == "d" else "stochastic",
        "model": getattr(args, "model", ""),
        "federated_optimizer": getattr(args, "federated_optimizer", ""),
        "client_optimizer": client_optimizer,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "require_multibatch_stochastic": bool(
            getattr(args, "require_multibatch_stochastic", False)
        ),
        "epochs": int(getattr(args, "epochs", 0)),
        "local_epochs": int(getattr(args, "epochs", 0)),
        "comm_round": int(getattr(args, "comm_round", 0)),
        "client_num_in_total": int(getattr(args, "client_num_in_total", 0)),
        "client_num_per_round": int(getattr(args, "client_num_per_round", 0)),
        "frequency_of_the_test": int(getattr(args, "frequency_of_the_test", 1)),
        "eval_freq": int(getattr(args, "eval_freq", 1)),
        "print_freq": int(getattr(args, "print_freq", 1)),
        "simple_model_selection_epochs": int(getattr(args, "simple_model_selection_epochs", 100)),
        "f_history_model_selection_epochs": int(getattr(args, "f_history_model_selection_epochs", 60)),
        "model_selection_batch_size": int(getattr(args, "model_selection_batch_size", 200)),
        "model_selection_max_samples": int(getattr(args, "model_selection_max_samples", 0)),
        "skip_model_selection": skip_model_selection,
        "skip_gmm_eval": skip_gmm_eval,
        "gmm_eval_proxy": gmm_eval_proxy,
        "auxiliary_regression": auxiliary_regression,
        "auxiliary_regression_epochs": auxiliary_regression_epochs,
        "auxiliary_regression_state_device": str(getattr(args, "auxiliary_regression_state_device", "device")),
        "append_round_csv": bool(getattr(args, "append_round_csv", True)),
        "periodic_checkpoint_interval": int(getattr(args, "periodic_checkpoint_interval", 200)),
        "dataloader_num_workers": int(getattr(args, "dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(getattr(args, "dataloader_pin_memory", False)),
        "log_test_mse_by_round": log_test_mse_by_round,
        "test_mse_logged_by_round": log_test_mse_by_round,
        "test_mse_used_for_selection": test_mse_used_for_selection,
        "selection_metric_source": selection_metric_source,
        "g_learning_rate": learning_rate,
        "critic_multiplier": critic_multiplier,
        "f_learning_rate": critic_multiplier * learning_rate,
        "server_learning_rate": server_learning_rate,
        "objective_lambda_1": objective_lambda_1,
        "aggregation_weighting": aggregation_weighting,
        "objective_mode": objective_mode,
        "input_dim_g": int(getattr(args, "input_dim_g", 0)),
        "input_dim_f": int(getattr(args, "input_dim_f", 0)),
        "random_seed": seed,
        "run_id": run_id,
        "output_dir": output_dir,
        "partition_method": getattr(args, "partition_method", ""),
        "partition_alpha": float(getattr(args, "partition_alpha", 0.0)),
        "data_cache_dir": getattr(args, "data_cache_dir", ""),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "gradient_clip_norm": float(getattr(args, "gradient_clip_norm", 1.0)),
        "using_gpu": bool(getattr(args, "using_gpu", False)),
        "gpu_id": int(getattr(args, "gpu_id", 0)),
        "enable_legacy_outputs": bool(getattr(args, "enable_legacy_outputs", True)),
        "enable_legacy_plot": bool(getattr(args, "enable_legacy_plot", False)),
        "overwrite": bool(getattr(args, "overwrite", False)),
    })


def build_run_dir(output_dir, dataset, variant, seed, run_id):
    return os.path.join(str(output_dir), str(dataset), str(variant), f"seed_{int(seed)}", str(run_id))


def run_dir_from_config(config):
    return build_run_dir(
        config["output_dir"],
        config["dataset"],
        config["variant"],
        config["random_seed"],
        config["run_id"],
    )


def ensure_run_dirs(run_dir):
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)


def env_truthy(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


class _NoOpProfileSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ProfileSpan:
    def __init__(self, profiler, phase, round_idx=None, client_id=None, detail=""):
        self.profiler = profiler
        self.phase = phase
        self.round_idx = round_idx
        self.client_id = client_id
        self.detail = detail
        self.start_wall = None
        self.start_cpu = None

    def __enter__(self):
        self.profiler._synchronize_cuda()
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.profiler._synchronize_cuda()
        detail = self.detail
        if exc_type is not None:
            detail = f"{detail}; exception={exc_type.__name__}" if detail else f"exception={exc_type.__name__}"
        self.profiler.record(
            self.phase,
            time.perf_counter() - self.start_wall,
            process_cpu_seconds=time.process_time() - self.start_cpu,
            round_idx=self.round_idx,
            client_id=self.client_id,
            detail=detail,
        )
        return False


class RuntimeProfiler:
    """Low-overhead opt-in profiler for high-dimensional stochastic diagnostics."""

    RUNTIME_FIELDS = (
        "timestamp_epoch",
        "elapsed_since_profile_start",
        "phase",
        "round",
        "client_id",
        "duration_seconds",
        "process_cpu_seconds",
        "detail",
    )
    TELEMETRY_FIELDS = (
        "timestamp_epoch",
        "elapsed_since_profile_start",
        "gpu_index",
        "gpu_name",
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "memory_used_mb",
        "memory_total_mb",
        "power_draw_w",
        "error",
    )
    TOP_LEVEL_PHASES = {
        "data_load",
        "model_create",
        "runner_init",
        "runner_run",
        "model_to_device",
        "global_tensor_to_device",
        "model_selection",
        "setup_clients",
        "training_total",
        "round_total",
        "finalization",
    }

    def __init__(self, enabled, profile_dir=None, effective_config=None):
        self.enabled = bool(enabled)
        self.profile_dir = os.path.abspath(profile_dir) if profile_dir else None
        self.effective_config = effective_config or {}
        self.runtime_path = os.path.join(self.profile_dir, "runtime_profile.csv") if self.profile_dir else None
        self.telemetry_path = os.path.join(self.profile_dir, "gpu_telemetry.csv") if self.profile_dir else None
        self.environment_path = os.path.join(self.profile_dir, "profile_environment.json") if self.profile_dir else None
        self.summary_path = os.path.join(self.profile_dir, "profile_summary.json") if self.profile_dir else None
        self.start_wall = time.time()
        self.start_perf = time.perf_counter()
        self.start_process_cpu = time.process_time()
        self._lock = threading.Lock()
        self._telemetry_stop = threading.Event()
        self._telemetry_thread = None
        self._telemetry_samples = []
        self._phase_totals = defaultdict(float)
        self._phase_cpu_totals = defaultdict(float)
        self._phase_counts = defaultdict(int)
        self._recorded_once = set()
        self.profile_batches = env_truthy(PROFILE_BATCHES_ENV_VAR, default=False)
        if self.enabled and self.profile_dir:
            os.makedirs(self.profile_dir, exist_ok=True)
            self._write_csv_header(self.runtime_path, self.RUNTIME_FIELDS)
            self._write_csv_header(self.telemetry_path, self.TELEMETRY_FIELDS)

    def __deepcopy__(self, memo):
        return self

    @classmethod
    def disabled(cls):
        return cls(False)

    @classmethod
    def from_config(cls, effective_config):
        if not env_truthy(PROFILE_ENV_VAR, default=False):
            return cls.disabled()
        profile_root = os.path.abspath(os.environ.get(PROFILE_ROOT_ENV_VAR, DEFAULT_PROFILE_ROOT))
        profile_dir = build_run_dir(
            profile_root,
            effective_config.get("dataset", "unknown"),
            effective_config.get("variant", "unknown"),
            int(effective_config.get("random_seed", 0)),
            effective_config.get("run_id", "unknown"),
        )
        return cls(True, profile_dir=profile_dir, effective_config=dict(effective_config))

    def configure(self, effective_config, actual_run_dir=None):
        if not self.enabled:
            return self
        self.effective_config = dict(effective_config)
        self.actual_run_dir = os.path.abspath(str(actual_run_dir)) if actual_run_dir else None
        return self

    def _write_csv_header(self, path, fields):
        if not path or os.path.exists(path):
            return
        with open(path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    def _elapsed(self):
        return time.perf_counter() - self.start_perf

    def _synchronize_cuda(self):
        if not self.enabled:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

    def span(self, phase, round_idx=None, client_id=None, detail=""):
        if not self.enabled:
            return _NoOpProfileSpan()
        return _ProfileSpan(self, phase, round_idx=round_idx, client_id=client_id, detail=detail)

    def record(self, phase, duration_seconds=0.0, process_cpu_seconds=0.0, round_idx=None, client_id=None, detail=""):
        if not self.enabled:
            return
        row = {
            "timestamp_epoch": time.time(),
            "elapsed_since_profile_start": self._elapsed(),
            "phase": phase,
            "round": "" if round_idx is None else int(round_idx),
            "client_id": "" if client_id is None else int(client_id),
            "duration_seconds": float(duration_seconds),
            "process_cpu_seconds": float(process_cpu_seconds),
            "detail": detail,
        }
        with self._lock:
            self._write_csv_header(self.runtime_path, self.RUNTIME_FIELDS)
            with open(self.runtime_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.RUNTIME_FIELDS)
                writer.writerow(row)
            self._phase_totals[phase] += float(duration_seconds)
            self._phase_cpu_totals[phase] += float(process_cpu_seconds)
            self._phase_counts[phase] += 1

    def record_once(self, key, phase, detail=""):
        if not self.enabled:
            return
        with self._lock:
            if key in self._recorded_once:
                return
            self._recorded_once.add(key)
        self.record(phase, duration_seconds=0.0, detail=detail)

    def record_environment(self, effective_config=None, actual_run_dir=None, device=None):
        if not self.enabled:
            return
        if effective_config is not None:
            self.effective_config = dict(effective_config)
        if actual_run_dir is not None:
            self.actual_run_dir = os.path.abspath(str(actual_run_dir))
        torch_info = {}
        try:
            import torch
            torch_info = {
                "torch_version": getattr(torch, "__version__", "unknown"),
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else int(torch.cuda.device_count()),
            }
            if torch.cuda.is_available():
                current = torch.cuda.current_device()
                torch_info.update({
                    "cuda_current_device": int(current),
                    "cuda_current_device_name": torch.cuda.get_device_name(current),
                    "cuda_device_names": [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ],
                })
        except Exception as exc:
            torch_info = {"torch_probe_error": repr(exc)}
        payload = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "profile_dir": self.profile_dir,
            "actual_run_dir": getattr(self, "actual_run_dir", None),
            "device": str(device) if device is not None else None,
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                "FEDGMM_PROFILE_RUNTIME": os.environ.get(PROFILE_ENV_VAR),
                "FEDGMM_PROFILE_ROOT": os.environ.get(PROFILE_ROOT_ENV_VAR),
                "FEDGMM_PROFILE_BATCHES": os.environ.get(PROFILE_BATCHES_ENV_VAR),
                "FEDGMM_PROFILE_SKIP_AUX_REG": os.environ.get(PROFILE_SKIP_AUX_REG_ENV_VAR),
            },
            "effective_config": json_safe(self.effective_config),
            "torch": torch_info,
        }
        with open(self.environment_path, "w") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def start_telemetry(self):
        if not self.enabled or not env_truthy(PROFILE_TELEMETRY_ENV_VAR, default=True):
            return
        if self._telemetry_thread is not None:
            return
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, name="fedgmm-gpu-telemetry", daemon=True)
        self._telemetry_thread.start()

    def _telemetry_loop(self):
        interval = _parse_float(os.environ.get(PROFILE_INTERVAL_ENV_VAR)) or 1.0
        interval = max(0.2, float(interval))
        while not self._telemetry_stop.is_set():
            self._sample_gpu_telemetry()
            self._telemetry_stop.wait(interval)

    def _sample_gpu_telemetry(self):
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=3, check=False)
            if completed.returncode != 0:
                self._write_telemetry_error(completed.stderr.strip() or completed.stdout.strip() or f"returncode={completed.returncode}")
                return
            for line in completed.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 7:
                    self._write_telemetry_error(f"unparseable nvidia-smi row: {line}")
                    continue
                row = {
                    "timestamp_epoch": time.time(),
                    "elapsed_since_profile_start": self._elapsed(),
                    "gpu_index": parts[0],
                    "gpu_name": parts[1],
                    "utilization_gpu_pct": _parse_float(parts[2]),
                    "utilization_memory_pct": _parse_float(parts[3]),
                    "memory_used_mb": _parse_float(parts[4]),
                    "memory_total_mb": _parse_float(parts[5]),
                    "power_draw_w": _parse_float(parts[6]),
                    "error": "",
                }
                self._append_telemetry(row)
        except Exception as exc:
            self._write_telemetry_error(repr(exc))

    def _write_telemetry_error(self, error):
        row = {
            "timestamp_epoch": time.time(),
            "elapsed_since_profile_start": self._elapsed(),
            "gpu_index": "",
            "gpu_name": "",
            "utilization_gpu_pct": "",
            "utilization_memory_pct": "",
            "memory_used_mb": "",
            "memory_total_mb": "",
            "power_draw_w": "",
            "error": error,
        }
        self._append_telemetry(row)

    def _append_telemetry(self, row):
        with self._lock:
            self._write_csv_header(self.telemetry_path, self.TELEMETRY_FIELDS)
            with open(self.telemetry_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.TELEMETRY_FIELDS)
                writer.writerow(row)
            self._telemetry_samples.append(row)

    def stop(self, extra=None):
        if not self.enabled:
            return
        self._telemetry_stop.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=5)
            self._telemetry_thread = None
        self.write_summary(extra=extra or {})

    def write_summary(self, extra=None):
        if not self.enabled:
            return
        wall = self._elapsed()
        phase_totals = dict(sorted(self._phase_totals.items()))
        phase_cpu_totals = dict(sorted(self._phase_cpu_totals.items()))
        process_cpu = time.process_time() - self.start_process_cpu
        top_phase_total = sum(
            total for phase, total in phase_totals.items()
            if phase in self.TOP_LEVEL_PHASES
        )
        gpu_util = [
            float(row["utilization_gpu_pct"])
            for row in self._telemetry_samples
            if row.get("utilization_gpu_pct") not in ("", None)
        ]
        gpu_memory = [
            float(row["memory_used_mb"])
            for row in self._telemetry_samples
            if row.get("memory_used_mb") not in ("", None)
        ]
        power = [
            float(row["power_draw_w"])
            for row in self._telemetry_samples
            if row.get("power_draw_w") not in ("", None)
        ]
        completed_rounds = int(self._phase_counts.get("round_total", 0) or 0)
        round_total = float(phase_totals.get("round_total", 0.0) or 0.0)
        summary = {
            "profile_dir": self.profile_dir,
            "actual_run_dir": getattr(self, "actual_run_dir", None),
            "wall_clock_profile_seconds": wall,
            "process_cpu_seconds_total": process_cpu,
            "process_cpu_utilization_pct": process_cpu / wall * 100 if wall > 0 else None,
            "completed_rounds": completed_rounds,
            "seconds_per_round_wall_clock": wall / completed_rounds if completed_rounds else None,
            "rounds_per_second_wall_clock": completed_rounds / wall if completed_rounds and wall > 0 else None,
            "seconds_per_round_training_loop": round_total / completed_rounds if completed_rounds and round_total > 0 else None,
            "rounds_per_second_training_loop": completed_rounds / round_total if completed_rounds and round_total > 0 else None,
            "phase_totals_seconds": phase_totals,
            "phase_process_cpu_seconds": phase_cpu_totals,
            "phase_counts": dict(sorted(self._phase_counts.items())),
            "top_phases_by_total_seconds": sorted(
                phase_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20],
            "top_level_explained_fraction_of_wall_clock": min(1.0, top_phase_total / wall) if wall > 0 else None,
            "gpu_telemetry_sample_count": len(self._telemetry_samples),
            "gpu_utilization_avg_pct": sum(gpu_util) / len(gpu_util) if gpu_util else None,
            "gpu_utilization_max_pct": max(gpu_util) if gpu_util else None,
            "gpu_memory_used_avg_mb": sum(gpu_memory) / len(gpu_memory) if gpu_memory else None,
            "gpu_memory_used_max_mb": max(gpu_memory) if gpu_memory else None,
            "gpu_power_avg_w": sum(power) / len(power) if power else None,
            "gpu_power_max_w": max(power) if power else None,
            "extra": json_safe(extra or {}),
        }
        with open(self.summary_path, "w") as handle:
            json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")


def exact_sample_count(indices_or_subset):
    return int(len(indices_or_subset))


def client_indices_for_round(random_seed, round_idx, client_num_in_total, client_num_per_round):
    if client_num_in_total == client_num_per_round:
        return list(range(client_num_in_total))
    num_clients = min(client_num_per_round, client_num_in_total)
    rng = random.Random(f"{int(random_seed)}:{int(round_idx)}")
    return rng.sample(range(client_num_in_total), num_clients)


def has_run_artifacts(run_dir):
    if not os.path.isdir(run_dir):
        return False
    for artifact_name in RUN_ARTIFACT_NAMES:
        if os.path.exists(os.path.join(run_dir, artifact_name)):
            return True
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    return os.path.isdir(checkpoint_dir) and bool(os.listdir(checkpoint_dir))


def prepare_run_dir(run_dir, overwrite=False, dry_run=False):
    if dry_run:
        return
    normalized = os.path.abspath(run_dir)
    protected = os.path.join(os.sep, "results", "_golden")
    if protected in normalized:
        raise ValueError(f"Refusing to overwrite protected golden output directory: {run_dir}")
    if has_run_artifacts(run_dir):
        if not overwrite:
            raise FileExistsError(
                f"Run directory already contains artifacts: {run_dir}. "
                "Use --overwrite or set overwrite: true to replace it."
            )
        shutil.rmtree(run_dir)


def _flatten_numbers(value):
    if value is None:
        return []
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            value = reshape(-1)
        except TypeError:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return out
    return [float(value)]


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe(item())
    if isinstance(value, dict):
        return {str(key): json_safe(item_value) for key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item_value) for item_value in value]
    return str(value)


def format_no_valid_model_selection_error(context):
    context = context or {}
    fields = {
        "dataset": context.get("dataset", "unknown"),
        "seed": context.get("random_seed", "unknown"),
        "simple_model_selection_epochs": context.get("simple_model_selection_epochs", "unknown"),
        "f_history_model_selection_epochs": context.get("f_history_model_selection_epochs", "unknown"),
        "model_selection_batch_size": context.get("model_selection_batch_size", "unknown"),
        "learning_rate": context.get("learning_rate", "unknown"),
        "critic_multiplier": context.get("critic_multiplier", "unknown"),
        "f_learning_rate": context.get("f_learning_rate", "unknown"),
        "client_optimizer": context.get("client_optimizer", "unknown"),
        "best_score": context.get("best_score", "unknown"),
    }
    details = ", ".join(f"{key}={value}" for key, value in fields.items())
    return (
        "No valid model-selection candidate was selected before federated training started; "
        f"{details}; federated training never started"
    )


def structural_mse_from_predictions(prediction, true_g):
    pred_values = _flatten_numbers(prediction)
    true_values = _flatten_numbers(true_g)
    if len(pred_values) != len(true_values):
        raise ValueError(
            f"Prediction/target length mismatch: {len(pred_values)} != {len(true_values)}"
        )
    if not pred_values:
        raise ValueError("Cannot compute structural MSE on an empty split")
    total = 0.0
    for pred, true in zip(pred_values, true_values):
        total += (pred - true) ** 2
    return total / len(pred_values)


def moment_violation(f_values, y_values, g_values):
    """Held-out moment-restriction violation: ``|| E[f(Z,X) * (Y - g(D,X))] ||^2``.

    Real eICU data has no ground-truth ``g0``, so this is the only recovery
    diagnostic available for Study B; Study A tracks it alongside structural
    MSE (which it can compute, since g0 is known there) as a second, g0-free
    checkpoint criterion. Squaring the mean removes any dependence on which
    epsilon sign convention (``Y-g`` vs ``g-Y``) a caller's objective uses, so
    this is safe to compute the same way regardless of ``objective_mode``.
    """
    f_flat = _flatten_numbers(f_values)
    y_flat = _flatten_numbers(y_values)
    g_flat = _flatten_numbers(g_values)
    if not (len(f_flat) == len(y_flat) == len(g_flat)):
        raise ValueError(
            f"f/y/g length mismatch: {len(f_flat)}, {len(y_flat)}, {len(g_flat)}"
        )
    if not f_flat:
        raise ValueError("Cannot compute moment violation on an empty split")
    moment = sum(f * (y - g) for f, y, g in zip(f_flat, y_flat, g_flat)) / len(f_flat)
    return moment ** 2


def structural_mse(model, split, batch_size=1024):
    if not hasattr(split, "x") or not hasattr(split, "g"):
        raise ValueError("Split must expose x and ground-truth structural g")
    prediction = predictions_for_split(model, split, batch_size=batch_size)
    return structural_mse_from_predictions(prediction, split.g)


def predictions_for_split(model, split, batch_size=1024):
    if not hasattr(split, "x") or not hasattr(split, "g"):
        raise ValueError("Split must expose x and ground-truth structural g")
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    if callable(eval_method):
        eval_method()
    try:
        import torch
        no_grad = torch.no_grad()
    except Exception:
        no_grad = nullcontext()
    shape = getattr(split.x, "shape", None)
    num_samples = int(shape[0]) if shape is not None else len(split.x)
    if batch_size is None or int(batch_size) <= 0:
        batch_size = num_samples
    predictions = []
    with no_grad:
        for start in range(0, num_samples, int(batch_size)):
            predictions.append(model(split.x[start : start + int(batch_size)]))
    if not predictions:
        raise ValueError("Cannot predict an empty split")
    try:
        import torch
        prediction = torch.cat(predictions, dim=0)
    except Exception:
        import numpy as np
        prediction = np.concatenate(predictions, axis=0)
    if callable(train_method) and was_training:
        train_method()
    return prediction


def state_is_finite(state_dict):
    try:
        import torch
    except Exception:
        torch = None
    for value in state_dict.values():
        if torch is not None and torch.is_tensor(value):
            if not bool(torch.isfinite(value).all().item()):
                return False
            continue
        for number in _flatten_numbers(value):
            if not math.isfinite(number):
                return False
    return True


def write_effective_config(run_dir, config):
    ensure_run_dirs(run_dir)
    path = os.path.join(run_dir, "effective_config.json")
    with open(path, "w") as f:
        json.dump(json_safe(config), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_mse_by_round(run_dir, rows):
    path = os.path.join(run_dir, "mse_by_round.csv")
    fieldnames = [
        "round",
        "train_mse",
        "val_mse",
        "train_moment_violation",
        "val_moment_violation",
        "gmm_train_objective",
        "gmm_val_objective",
        "gmm_eval",
        "finite",
        "diverged",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def append_mse_by_round(run_dir, row):
    path = os.path.join(run_dir, "mse_by_round.csv")
    fieldnames = [
        "round",
        "train_mse",
        "val_mse",
        "train_moment_violation",
        "val_moment_violation",
        "gmm_train_objective",
        "gmm_val_objective",
        "gmm_eval",
        "finite",
        "diverged",
    ]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_aggregation_weights_by_round(run_dir, rows):
    path = os.path.join(run_dir, "aggregation_weights_by_round.csv")
    fieldnames = [
        "round",
        "aggregation_weighting",
        "k_participating",
        "weight_sum",
        "client_weights_json",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def append_aggregation_weights_by_round(run_dir, row):
    path = os.path.join(run_dir, "aggregation_weights_by_round.csv")
    fieldnames = [
        "round",
        "aggregation_weighting",
        "k_participating",
        "weight_sum",
        "client_weights_json",
    ]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_test_mse_by_round(run_dir, rows):
    path = os.path.join(run_dir, "test_mse_by_round.csv")
    fieldnames = [
        "round",
        "test_mse",
        "finite",
        "diverged",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def append_test_mse_by_round(run_dir, row):
    path = os.path.join(run_dir, "test_mse_by_round.csv")
    fieldnames = [
        "round",
        "test_mse",
        "finite",
        "diverged",
    ]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_metrics(run_dir, metrics):
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(json_safe(metrics), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_batching_summary(run_dir, rows):
    path = os.path.join(run_dir, "batching_summary.json")
    with open(path, "w") as f:
        json.dump(json_safe(rows), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _to_numpy(value):
    import numpy as np

    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return numpy_method()
    return np.asarray(value)


def save_predictions_npz(
    run_dir,
    test_split,
    best_prediction,
    final_prediction,
    metadata,
):
    import numpy as np

    x = _to_numpy(test_split.x)
    true_g = _to_numpy(test_split.g)
    best_pred = _to_numpy(best_prediction)
    final_pred = _to_numpy(final_prediction)
    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        indices = np.argsort(x.reshape(x.shape[0], -1)[:, 0])
    else:
        # Images have many feature axes.  Flattening argsort results would
        # create one index per pixel rather than one per sample.
        indices = np.arange(x.shape[0])
    path = os.path.join(run_dir, "predictions.npz")
    np.savez(
        path,
        x=x[indices],
        true_g=true_g[indices],
        **{
            PREDICTION_OUTPUT_KEYS[0]: best_pred[indices],
            PREDICTION_OUTPUT_KEYS[1]: final_pred[indices],
        },
        algorithm=np.asarray(str(metadata.get("algorithm", ""))),
        variant=np.asarray(str(metadata.get("variant", ""))),
        seed=np.asarray(int(metadata.get("random_seed", 0))),
        run_id=np.asarray(str(metadata.get("run_id", ""))),
    )
    return path


def now_seconds():
    return time.time()


def expand_smoke_manifest(manifest):
    base = dict(manifest.get("base", {}))
    variants = manifest.get("variants", [])
    expanded = []
    for variant in variants:
        item = dict(base)
        item.update(variant)
        optimizer = item.get("client_optimizer", "sgd")
        batch_size = int(item.get("batch_size", 0))
        algorithm = "fedogda" if str(optimizer).lower() == "ogda" else "fedgda"
        suffix = "d" if batch_size <= 0 else "s"
        item["variant"] = variant.get("variant", f"{algorithm}_{suffix}")
        expanded.append(item)
    return expanded
