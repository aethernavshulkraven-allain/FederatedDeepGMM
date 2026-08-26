"""Tests for scripts/aggregate_highdim_deterministic_finals_post_bn_20260826.py.

Covers the row-count/unresolved-blocking guards with cheap fixtures, plus one
full 180-trajectory happy path proving test metrics unlock correctly once
every planned trajectory is resolved.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import aggregate_highdim_deterministic_finals_post_bn_20260826 as aggregate_finals  # noqa: E402

COMM_ROUND = 500
DATASETS = ["femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz"]
METHODS = ("fedgda_d", "fedogda_d")
ALPHAS = (0.1, 0.5, 1.0)
SEEDS = (0, 1, 2, 3, 4)
FIELDNAMES = [
    "run_id", "dataset", "method", "seed", "alpha", "learning_rate",
    "critic_multiplier", "client_optimizer", "comm_round", "final_result_dir",
    "server_buffer_policy", "reused", "source_stage", "source_run_id",
]


def _write_run(run_dir: Path, run_id: str, *, dataset="fixture", method="fedgda_d",
                seed=0, terminal: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "run_id": run_id,
        "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
        "random_seed": seed, "comm_round": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False, "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": terminal, "best_gmm_eval": 1.0, "best_validation_mse": 0.1,
        "run_status": "completed", "rounds_completed": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": 400 if terminal else None,
        "nonfinite_diagnostics": ([{"round": 400}] if terminal else []),
        "test_mse_at_best_validation": 0.2, "final_test_mse": 0.25, "best_validation_round": 10,
        "g_bn_min_running_var": 0.01, "f_bn_min_running_var": 0.02,
    }))
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "train_mse", "val_mse", "primary_val_mse", "equal_client_val_mse",
            "train_moment_violation", "val_moment_violation", "gmm_train_objective",
            "gmm_val_objective", "gmm_eval", "g_bn_min_running_var", "f_bn_min_running_var",
            "finite", "diverged",
        ])
        for i in range(COMM_ROUND):
            row_finite = "True"
            row_diverged = "False"
            if terminal and i == 400:
                row_finite = "False"
            writer.writerow([
                i, 0.1, 0.1, 0.1, "", 0.1, 0.1, 0.1, 0.1, 1.0, 0.01, 0.02,
                row_finite, row_diverged,
            ])


class AggregateGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_manifest(self, rows) -> Path:
        manifest_path = self.tmp / "finals_manifest.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def test_wrong_row_count_rejected(self):
        manifest_path = self._write_manifest([{
            "run_id": "only_one", "dataset": "d", "method": "fedgda_d", "seed": "0",
            "alpha": "0.5", "client_optimizer": "sgd",
            "final_result_dir": "x", "reused": "False",
        }])
        with self.assertRaisesRegex(ValueError, "exactly 180 rows"):
            aggregate_finals.aggregate(manifest_path)

    def test_unresolved_trajectory_locks_the_whole_report(self):
        rows = []
        for i in range(180):
            run_dir = self.tmp / f"run_{i}"
            run_id = f"run_{i}"
            if i == 0:
                pass  # deliberately never written -- unresolved
            else:
                _write_run(run_dir, run_id, dataset="d", method="fedgda_d")
            rows.append({
                "run_id": run_id, "dataset": "d", "method": "fedgda_d", "seed": "0",
                "alpha": "0.5", "client_optimizer": "sgd",
                "final_result_dir": str(run_dir), "reused": "False",
            })
        manifest_path = self._write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "not yet auditably resolved"):
            aggregate_finals.aggregate(manifest_path)


class AggregateFullMatrixTest(unittest.TestCase):
    def test_full_180_matrix_unlocks_test_metrics_and_summarizes_by_cell_alpha(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            rows = []
            terminal_run_id = None
            for dataset in DATASETS:
                for method in METHODS:
                    for alpha in ALPHAS:
                        for seed in SEEDS:
                            run_id = f"final_{dataset}_{method}_a{alpha}_s{seed}"
                            run_dir = tmp / "results" / run_id
                            terminal = (
                                terminal_run_id is None and dataset == DATASETS[0]
                                and method == METHODS[0] and alpha == 0.5 and seed == 0
                            )
                            if terminal:
                                terminal_run_id = run_id
                            _write_run(
                                run_dir, run_id, dataset=dataset, method=method,
                                seed=seed, terminal=terminal,
                            )
                            rows.append({
                                "run_id": run_id, "dataset": dataset, "method": method,
                                "seed": str(seed), "alpha": f"{alpha:g}",
                                "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
                                "final_result_dir": str(run_dir), "reused": "False",
                            })
            manifest_path = tmp / "finals_manifest.csv"
            with manifest_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            result = aggregate_finals.aggregate(manifest_path)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["trajectories"]), 180)
            self.assertEqual(len(result["cross_seed_summary"]), 36)  # 12 cells x 3 alphas

            # The one deliberately-terminal trajectory is reported, not hidden.
            terminal_entries = [
                t for t in result["trajectories"] if t["run_id"] == terminal_run_id
            ]
            self.assertEqual(len(terminal_entries), 1)
            self.assertTrue(terminal_entries[0]["terminal_ineligible"])
            self.assertIsNone(terminal_entries[0]["final_test_mse"])

            # A stable trajectory reports real test metrics.
            stable_entries = [
                t for t in result["trajectories"]
                if t["run_id"] != terminal_run_id
            ]
            self.assertTrue(all(t["final_test_mse"] == 0.25 for t in stable_entries))

            key = f"{DATASETS[0]}|{METHODS[0]}|0.5"
            self.assertEqual(result["cross_seed_summary"][key]["seeds_terminal"], 1)
            self.assertEqual(result["cross_seed_summary"][key]["seeds_stable"], 4)


if __name__ == "__main__":
    unittest.main()
