"""Tests for the pretraining_failure.json classification path (closeout plan
Phase 1 SS4.2): validate_pretraining_failure_artifact, validate_artifacts'
dispatch to it, and check_manifest_stage_complete.py's cross-check. A bare
nonzero return code with no valid artifact must stay unexplained; only a
structurally valid, consistent artifact may classify a run as
terminal_pretraining_ineligible."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_manifest  # noqa: E402
import check_manifest_stage_complete as stage_complete  # noqa: E402


def _row(run_id: str = "pretrain_fail_fixture") -> dict[str, str]:
    return {
        "run_id": run_id,
        "dataset": "femnist_x",
        "method": "fedgda_d",
        "seed": "0",
        "client_optimizer": "sgd",
        "comm_round": "150",
        "learning_rate": "0.333333",
        "critic_multiplier": "10",
        "server_buffer_policy": "direct_client_aggregate",
    }


def _valid_payload(run_id: str, **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "effective_config_checksum": "deadbeef" * 8,
        "failure_phase": "model_selection",
        "federated_rounds_started": 0,
        "model_selection_epochs_attempted": 60,
        "best_model_selection_score": None,
        "per_epoch_finite_status": [
            {"epoch": 0, "epsilon_dev_finite": True, "f_of_z_dev_finite": True},
        ],
        "first_nonfinite_epoch": None,
        "terminal_reason": "No valid model-selection candidate was selected",
        "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: ...",
        "stdout_sha256": None,
        "stderr_sha256": None,
        "hash_bundle_id": None,
    }
    payload.update(overrides)
    return payload


class ValidatePretrainingFailureArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(__import__("shutil").rmtree, self.tmp, ignore_errors=True)
        self.row = _row()

    def _write(self, run_dir: Path, payload: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "effective_config.json").write_text(json.dumps({"run_id": self.row["run_id"]}))
        (run_dir / "pretraining_failure.json").write_text(json.dumps(payload))

    def test_valid_artifact_classifies_as_terminal_pretraining_ineligible(self):
        run_dir = self.tmp / "run"
        self._write(run_dir, _valid_payload(self.row["run_id"]))
        result = run_manifest.validate_pretraining_failure_artifact(run_dir, self.row)
        self.assertTrue(result["terminal_ineligible"])
        self.assertTrue(result["terminal_pretraining_ineligible"])
        self.assertIn("No valid model-selection candidate", result["terminal_reason"])

    def test_validate_artifacts_dispatches_to_pretraining_branch(self):
        # This is the exact scenario that used to raise "missing artifacts"
        # unconditionally: only effective_config.json + pretraining_failure.json
        # exist, none of the round-curve/checkpoint/prediction artifacts do.
        run_dir = self.tmp / "run"
        self._write(run_dir, _valid_payload(self.row["run_id"]))
        result = run_manifest.validate_artifacts(run_dir, self.row)
        self.assertTrue(result["terminal_pretraining_ineligible"])

    def test_successful_run_reports_terminal_pretraining_ineligible_false(self):
        # A normal completed run's validate_artifacts result must carry the
        # new key too, defaulted False, so callers can check it uniformly.
        row = {**self.row, "comm_round": "2"}
        run_dir = self.tmp / "run"
        run_dir.mkdir(parents=True)
        config = {
            "dataset": row["dataset"], "variant": row["method"],
            "run_id": row["run_id"], "client_optimizer": row["client_optimizer"],
            "random_seed": 0, "comm_round": 2, "server_buffer_policy": "direct_client_aggregate",
            "learning_rate": float(row["learning_rate"]),
            "critic_multiplier": float(row["critic_multiplier"]),
            "test_mse_used_for_selection": False, "selection_metric_source": "validation",
            "log_test_mse_by_round": False,
        }
        metrics = {
            "run_status": "completed", "rounds_completed": 2, "diverged": False,
            "failure_reason": None, "server_buffer_policy": "direct_client_aggregate",
            "nonfinite_first_round": None, "nonfinite_diagnostics": [],
            "test_mse_at_best_validation": 0.1, "final_test_mse": 0.1, "best_validation_round": 0,
            "g_bn_min_running_var": 0.1, "f_bn_min_running_var": 0.1,
        }
        (run_dir / "effective_config.json").write_text(json.dumps(config))
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        import csv
        fieldnames = [
            "round", "train_mse", "val_mse", "primary_val_mse", "equal_client_val_mse",
            "train_moment_violation", "val_moment_violation", "gmm_train_objective",
            "gmm_val_objective", "gmm_eval", "g_bn_min_running_var", "f_bn_min_running_var",
            "finite", "diverged",
        ]
        with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(2):
                writer.writerow({
                    "round": i, "train_mse": 0.1, "val_mse": 0.1, "primary_val_mse": 0.1,
                    "equal_client_val_mse": "", "train_moment_violation": 0.1,
                    "val_moment_violation": 0.1, "gmm_train_objective": 0.1,
                    "gmm_val_objective": 0.1, "gmm_eval": 0.1, "g_bn_min_running_var": 0.1,
                    "f_bn_min_running_var": 0.1, "finite": "True", "diverged": "False",
                })
        (run_dir / "predictions.npz").write_bytes(b"fake")
        (run_dir / "checkpoints").mkdir()
        (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
        (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
        result = run_manifest.validate_artifacts(run_dir, row)
        self.assertFalse(result["terminal_ineligible"])
        self.assertFalse(result["terminal_pretraining_ineligible"])

    def test_run_id_mismatch_rejected(self):
        run_dir = self.tmp / "run"
        self._write(run_dir, _valid_payload("some_other_run_id"))
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "run_id"):
            run_manifest.validate_pretraining_failure_artifact(run_dir, self.row)

    def test_nonzero_rounds_started_rejected(self):
        run_dir = self.tmp / "run"
        self._write(run_dir, _valid_payload(self.row["run_id"], federated_rounds_started=1))
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "federated_rounds_started"):
            run_manifest.validate_pretraining_failure_artifact(run_dir, self.row)

    def test_wrong_failure_phase_rejected(self):
        run_dir = self.tmp / "run"
        self._write(run_dir, _valid_payload(self.row["run_id"], failure_phase="federated_training"))
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "failure_phase"):
            run_manifest.validate_pretraining_failure_artifact(run_dir, self.row)

    def test_blank_terminal_reason_rejected(self):
        run_dir = self.tmp / "run"
        self._write(run_dir, _valid_payload(self.row["run_id"], terminal_reason=""))
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "terminal_reason"):
            run_manifest.validate_pretraining_failure_artifact(run_dir, self.row)

    def test_missing_traceback_rejected(self):
        run_dir = self.tmp / "run"
        payload = _valid_payload(self.row["run_id"])
        del payload["traceback"]
        self._write(run_dir, payload)
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "traceback"):
            run_manifest.validate_pretraining_failure_artifact(run_dir, self.row)

    def test_wrong_schema_version_rejected(self):
        run_dir = self.tmp / "run"
        self._write(run_dir, _valid_payload(self.row["run_id"], schema_version=2))
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "schema_version"):
            run_manifest.validate_pretraining_failure_artifact(run_dir, self.row)

    def test_generic_return_code_with_no_artifact_is_not_reclassified(self):
        # No pretraining_failure.json at all -- validate_artifacts must take
        # the normal path and raise "missing artifacts", exactly like today,
        # so a bare nonzero exit stays an unexplained process failure.
        run_dir = self.tmp / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "effective_config.json").write_text(json.dumps({"run_id": self.row["run_id"]}))
        with self.assertRaisesRegex(run_manifest.ManifestLaunchError, "missing artifacts"):
            run_manifest.validate_artifacts(run_dir, self.row)


class StageCompleteCrossCheckTest(unittest.TestCase):
    """check_manifest_stage_complete.py --validate-artifacts must reconcile a
    launcher-recorded terminal_ineligible status against a
    terminal_pretraining_ineligible validate_artifacts() result without
    reporting a mismatch."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(__import__("shutil").rmtree, self.tmp, ignore_errors=True)

    def test_pretraining_ineligible_row_passes_the_cross_check(self):
        row = _row()
        run_dir = self.tmp / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "effective_config.json").write_text(json.dumps({"run_id": row["run_id"]}))
        (run_dir / "pretraining_failure.json").write_text(
            json.dumps(_valid_payload(row["run_id"]))
        )
        manifest_row = {**row, "final_result_dir": str(run_dir)}
        manifest_path = self.tmp / "manifest.csv"
        import csv
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_row.keys()))
            writer.writeheader()
            writer.writerow(manifest_row)
        results_path = self.tmp / "results.json"
        results_path.write_text(json.dumps([
            {"run_id": row["run_id"], "status": "terminal_ineligible", "run_dir": str(run_dir)}
        ]))
        summary = stage_complete.check_stage(
            manifest_path, results_path, validate_stage_artifacts=True,
        )
        self.assertEqual(summary["terminal_ineligible"], 1)


if __name__ == "__main__":
    unittest.main()
