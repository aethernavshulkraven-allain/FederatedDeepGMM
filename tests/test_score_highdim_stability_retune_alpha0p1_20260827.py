"""Tests for scripts/score_highdim_stability_retune_alpha0p1_20260827.py --
picks a winner per retuned cell using the same frozen last-50-mean Psi rule
as the corrected screen, over however many cells were actually retuned
(unlike the screen's fixed 12)."""

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

import score_highdim_stability_retune_alpha0p1_20260827 as scorer  # noqa: E402

COMM_ROUND = scorer.RETUNE_COMM_ROUND


def write_run(run_dir: Path, *, dataset, method, run_id, gmm_eval, val_mse,
              learning_rate=0.01, critic_multiplier=5.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": dataset, "variant": method, "run_id": run_id,
        "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
        "random_seed": 0, "comm_round": COMM_ROUND,
        "learning_rate": learning_rate, "critic_multiplier": critic_multiplier,
        "server_buffer_policy": "direct_client_aggregate",
        "test_mse_used_for_selection": False, "selection_metric_source": "validation",
        "log_test_mse_by_round": False,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": False, "best_gmm_eval": gmm_eval, "best_validation_mse": val_mse,
        "run_status": "completed", "rounds_completed": COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "nonfinite_first_round": None, "nonfinite_diagnostics": [],
        "g_bn_min_running_var": 0.01, "f_bn_min_running_var": 0.01,
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
            writer.writerow([
                i, val_mse, val_mse, val_mse, "", 0.2, 0.2, 0.1, 0.1,
                gmm_eval, 0.01, 0.01, "True", "False",
            ])


class ScoreRetuneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.results_root = self.tmp / "results"
        self.rows = []

    def _add_row(self, *, dataset, method, lr, cm, gmm_eval, val_mse):
        run_id = f"det_stability_retune_alpha0p1_{dataset}_{method}_seed0_lr{lr}_cm{cm}"
        run_dir = self.results_root / dataset / method / "seed_0" / run_id
        write_run(
            run_dir, dataset=dataset, method=method, run_id=run_id, gmm_eval=gmm_eval, val_mse=val_mse,
            learning_rate=lr, critic_multiplier=cm,
        )
        self.rows.append({
            "run_id": run_id, "dataset": dataset, "method": method, "seed": "0",
            "alpha": "0.1", "learning_rate": str(lr), "critic_multiplier": str(cm),
            "client_optimizer": "sgd" if method == "fedgda_d" else "ogda",
            "comm_round": str(COMM_ROUND), "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })

    def _write_manifest(self) -> Path:
        manifest_path = self.tmp / "retune_manifest.csv"
        fieldnames = list(self.rows[0].keys())
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest_path

    def test_picks_the_highest_last50_psi_per_cell(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.4)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.03, cm=20, gmm_eval=0.5, val_mse=0.3)
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        self.assertEqual(result["status"], "complete")
        cell = result["cells"]["femnist_z|fedgda_d"]
        self.assertEqual(cell["winner"]["lr"], 0.02)
        self.assertEqual(cell["winner"]["cm"], 10.0)
        self.assertEqual(cell["eligible_candidates"], 3)

    def test_two_independent_cells_scored_separately(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.4)
        self._add_row(dataset="cifar10_x", method="fedogda_d", lr=0.001, cm=5, gmm_eval=5.0, val_mse=0.1)
        self._add_row(dataset="cifar10_x", method="fedogda_d", lr=0.003, cm=10, gmm_eval=1.0, val_mse=0.2)
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        self.assertEqual(len(result["cells"]), 2)
        self.assertEqual(result["cells"]["femnist_z|fedgda_d"]["winner"]["lr"], 0.02)
        self.assertEqual(result["cells"]["cifar10_x|fedogda_d"]["winner"]["lr"], 0.001)

    def test_exact_psi_tie_raises(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=2.0, val_mse=0.5)
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.02, cm=10, gmm_eval=2.0, val_mse=0.5)
        manifest_path = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "unresolved exact Psi tie"):
            scorer.score_retune(manifest_path)

    def test_terminal_candidate_excluded_from_ranking(self):
        self._add_row(dataset="femnist_z", method="fedgda_d", lr=0.01, cm=5, gmm_eval=1.0, val_mse=0.5)
        # A diverged candidate: overwrite its curve with a nonfinite entry.
        run_id = f"det_stability_retune_alpha0p1_femnist_z_fedgda_d_seed0_lr0.02_cm10"
        run_dir = self.results_root / "femnist_z" / "fedgda_d" / "seed_0" / run_id
        write_run(
            run_dir, dataset="femnist_z", method="fedgda_d", run_id=run_id, gmm_eval=99.0, val_mse=0.01,
            learning_rate=0.02, critic_multiplier=10.0,
        )
        with (run_dir / "metrics.json").open() as handle:
            metrics = json.load(handle)
        metrics["diverged"] = True
        metrics["nonfinite_first_round"] = 100
        metrics["nonfinite_diagnostics"] = [{"round": 100}]
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        self.rows.append({
            "run_id": run_id, "dataset": "femnist_z", "method": "fedgda_d", "seed": "0",
            "alpha": "0.1", "learning_rate": "0.02", "critic_multiplier": "10",
            "client_optimizer": "sgd", "comm_round": str(COMM_ROUND), "final_result_dir": str(run_dir),
            "server_buffer_policy": "direct_client_aggregate",
        })
        manifest_path = self._write_manifest()
        result = scorer.score_retune(manifest_path)
        cell = result["cells"]["femnist_z|fedgda_d"]
        # The diverged candidate (gmm_eval=99.0) must not win despite its
        # huge diagnostic value -- it's excluded as terminal, not ranked.
        self.assertEqual(cell["winner"]["lr"], 0.01)
        self.assertEqual(cell["terminal_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
