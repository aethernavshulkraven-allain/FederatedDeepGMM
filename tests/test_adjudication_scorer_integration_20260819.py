"""End-to-end integration test for scripts/score_highdim_adjudication_20260819.py:
exercises the real disk-reading path (load_seed_result, resolve_new_run_dir,
resolve_reused_run_dir, load_finals_index, build_cell_candidates,
score_manifest) against a small fixture tree that mimics the real
directory layout -- not just the pure scoring logic already covered by
tests/test_adjudication_scorer_20260819.py's synthetic Candidate objects.

This is the "wiring" the review feedback asked to freeze before real
adjudication results exist: does the CLI actually find, parse, and score
real completed/reused runs correctly.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import score_highdim_adjudication_20260819 as scorer  # noqa: E402


def write_run(run_dir: Path, *, comm_round: int, psi_values: list[float],
              mse_values: list[float], diverged: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "predictions.npz").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "dataset": "fake_ds", "variant": "fedgda_d", "comm_round": comm_round,
        "random_seed": 0, "run_id": run_dir.name,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "diverged": diverged, "best_gmm_eval": max(psi_values), "best_validation_mse": min(mse_values),
    }))
    with (run_dir / "mse_by_round.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "val_mse", "gmm_eval", "finite", "diverged"])
        for i, (psi, mse) in enumerate(zip(psi_values, mse_values)):
            writer.writerow([i, mse, psi, "True", str(diverged)])


class EndToEndScoringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.campaign_dir = self.tmp / "psi_adjudication_20260819_v2"
        self.finals_dir = self.tmp / "deterministic_finals_20260813"
        self.new_run_root = self.tmp / "results_new"
        self.campaign_dir.mkdir()
        self.finals_dir.mkdir()
        self.new_run_root.mkdir()

        self._orig = (scorer.CAMPAIGN_V2_DIR, scorer.FINALS_DIR, scorer.NEW_RUN_RESULTS_ROOT)
        scorer.CAMPAIGN_V2_DIR = self.campaign_dir
        scorer.FINALS_DIR = self.finals_dir
        scorer.NEW_RUN_RESULTS_ROOT = self.new_run_root

    def tearDown(self):
        scorer.CAMPAIGN_V2_DIR, scorer.FINALS_DIR, scorer.NEW_RUN_RESULTS_ROOT = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_finals_manifest(self, rows):
        fieldnames = ["dataset", "method", "learning_rate", "critic_multiplier", "seed",
                      "alpha", "run_id", "final_result_dir"]
        with (self.finals_dir / "finals_manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_three_candidate_cell_clear_winner_reused_and_new_runs_mixed(self):
        """psi_rank1 (new run) clearly wins over mse_winner (reused from a
        finals-style path) and psi_rank2 (new run, deliberately diverged on
        one seed so it's excluded) -- exercises both resolve_new_run_dir and
        resolve_reused_run_dir against real files on disk end-to-end."""
        ds, method = "fake_ds", "fedgda_d"

        # mse_winner: reused_from_finals candidate, lr=0.03/cm=5, all 3 seeds
        # good but middling Psi -- lives under a *separate* finals-style root.
        finals_rows = []
        for seed in (0, 1, 2):
            run_dir = self.finals_dir / ds / method / f"seed_{seed}" / f"det_final_{ds}_{method}_seed{seed}_alpha0p5_lr0p03_cm5"
            write_run(run_dir, comm_round=60, psi_values=[3.0] * 60, mse_values=[0.3] * 60, diverged=False)
            finals_rows.append({
                "dataset": ds, "method": method, "learning_rate": "0.03", "critic_multiplier": "5",
                "seed": str(seed), "alpha": "0.5",
                "run_id": run_dir.name, "final_result_dir": str(run_dir),
            })
        self._write_finals_manifest(finals_rows)

        # psi_rank1: new run, lr=0.1/cm=10, clearly highest Psi (10.0) across all 3 seeds.
        for seed in (0, 1, 2):
            run_dir = scorer.resolve_new_run_dir(ds, method, seed, 0.1, 10.0)
            write_run(run_dir, comm_round=60, psi_values=[10.0] * 60, mse_values=[0.1] * 60, diverged=False)

        # psi_rank2: new run, lr=0.1/cm=5 -- seed 2 diverges, so this candidate
        # must be excluded even though its other two seeds look good (Psi=8.0).
        for seed in (0, 1):
            run_dir = scorer.resolve_new_run_dir(ds, method, seed, 0.1, 5.0)
            write_run(run_dir, comm_round=60, psi_values=[8.0] * 60, mse_values=[0.15] * 60, diverged=False)
        run_dir = scorer.resolve_new_run_dir(ds, method, 2, 0.1, 5.0)
        write_run(run_dir, comm_round=60, psi_values=[8.0] * 60, mse_values=[0.15] * 60, diverged=True)

        summary = {
            "campaign": "test", "new_runs": 6,
            "plan": [{
                "dataset": ds, "method": method,
                "candidates": [
                    {"lr": 0.03, "cm": 5.0, "labels": ["mse_winner"], "reused_from_finals": True},
                    {"lr": 0.1, "cm": 5.0, "labels": ["psi_rank2"], "reused_from_finals": False},
                    {"lr": 0.1, "cm": 10.0, "labels": ["psi_rank1"], "reused_from_finals": False},
                ],
            }],
        }
        (self.campaign_dir / "adjudication_x_summary.json").write_text(json.dumps(summary))

        results = scorer.score_manifest("x")
        self.assertEqual(list(results.keys()), [f"{ds}/{method}"])
        cell = results[f"{ds}/{method}"]

        self.assertEqual(cell["outcome"], "promoted")
        self.assertEqual(cell["winner"]["candidate_id"], f"{ds}/{method}/lr0.1_cm10")
        self.assertIn(f"{ds}/{method}/lr0.1_cm5", cell["excluded_candidates"])
        self.assertEqual(len(cell["eligible_candidates"]), 2)  # mse_winner + psi_rank1, psi_rank2 excluded

        # Confirm the reused candidate's seeds were actually read from the
        # finals-style path (not silently treated as missing).
        reused = next(c for c in cell["all_candidates"] if c["candidate_id"] == f"{ds}/{method}/lr0.03_cm5")
        self.assertTrue(reused["eligible"])
        self.assertAlmostEqual(reused["median_psi"], 3.0)
        self.assertTrue(str(self.finals_dir) in reused["seeds"][0]["run_dir"])

    def test_missing_run_directory_is_ineligible_not_a_crash(self):
        """A candidate whose run never happened (no directory at all) must
        come back as artifacts_complete=False / ineligible, not raise."""
        ds, method = "fake_ds2", "fedogda_d"
        self._write_finals_manifest([])  # empty finals -- nothing to reuse
        for seed in (0, 1, 2):
            run_dir = scorer.resolve_new_run_dir(ds, method, seed, 0.01, 5.0)
            write_run(run_dir, comm_round=60, psi_values=[1.0] * 60, mse_values=[0.5] * 60, diverged=False)
        # lr=0.02/cm=5's run directory is never created at all.

        summary = {
            "campaign": "test", "new_runs": 3,
            "plan": [{
                "dataset": ds, "method": method,
                "candidates": [
                    {"lr": 0.01, "cm": 5.0, "labels": ["psi_rank1"], "reused_from_finals": False},
                    {"lr": 0.02, "cm": 5.0, "labels": ["psi_rank2"], "reused_from_finals": False},
                ],
            }],
        }
        (self.campaign_dir / "adjudication_signal_summary.json").write_text(json.dumps(summary))

        results = scorer.score_manifest("signal")
        cell = results[f"{ds}/{method}"]
        self.assertEqual(cell["outcome"], "promoted")
        self.assertEqual(cell["winner"]["candidate_id"], f"{ds}/{method}/lr0.01_cm5")
        self.assertIn(f"{ds}/{method}/lr0.02_cm5", cell["excluded_candidates"])
        missing_candidate = next(c for c in cell["all_candidates"] if c["candidate_id"] == f"{ds}/{method}/lr0.02_cm5")
        self.assertFalse(missing_candidate["eligible"])
        self.assertFalse(missing_candidate["seeds"][0]["artifacts_complete"])


if __name__ == "__main__":
    unittest.main()
