"""Tests for scripts/prepare_highdim_deterministic_stability_alpha0p1_20260826.py.

No real V4 winners exist yet (V4 has not been launched -- closeout plan
Phase 5 is out of scope for this pass), so these tests build a synthetic
v4_winners.json fixture matching the preparer's documented input contract.
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

import prepare_highdim_deterministic_stability_alpha0p1_20260826 as prepare_stability  # noqa: E402

DATASETS = ["femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz"]
METHODS = ("fedgda_d", "fedogda_d")


def _fake_winners() -> dict:
    cells = {}
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            cell_name = f"{dataset}|{method}"
            cells[cell_name] = {
                "dataset": dataset, "method": method,
                "winner": {
                    "lr": 0.001 * (i + 1), "cm": 5.0 + j,
                    "run_ids": {
                        "0": f"det_adjudicate_v4_{dataset}_{method}_seed0",
                        "1": f"det_adjudicate_v4_{dataset}_{method}_seed1",
                        "2": f"det_adjudicate_v4_{dataset}_{method}_seed2",
                    },
                },
            }
    return {"status": "complete", "alpha": 0.5, "seeds": [0, 1, 2], "cells": cells}


class PrepareStabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_winners(self, winners: dict) -> Path:
        path = self.tmp / "v4_winners.json"
        path.write_text(json.dumps(winners))
        return path

    def test_generates_exactly_12_rows_one_per_cell(self):
        winners_path = self._write_winners(_fake_winners())
        output_dir = self.tmp / "stability"
        result = prepare_stability.prepare(winners_path, output_dir)
        self.assertEqual(result["runs"], 12)
        with (output_dir / "stability_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["alpha"] == "0.1" for row in rows))
        self.assertTrue(all(row["seed"] == "0" for row in rows))
        self.assertTrue(all(row["comm_round"] == "500" for row in rows))
        self.assertTrue(all(
            row["server_buffer_policy"] == "direct_client_aggregate" for row in rows
        ))
        self.assertEqual(len({row["run_id"] for row in rows}), 12)
        self.assertEqual(len({row["final_result_dir"] for row in rows}), 12)
        self.assertTrue((output_dir / "generated_artifact_hashes.json").exists())

    def test_incomplete_v4_winners_blocked(self):
        winners = _fake_winners()
        winners["status"] = "in_progress"
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "absent or incomplete"):
            prepare_stability.prepare(winners_path, self.tmp / "stability")

    def test_wrong_alpha_blocked(self):
        winners = _fake_winners()
        winners["alpha"] = 0.1
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "alpha=0.5"):
            prepare_stability.prepare(winners_path, self.tmp / "stability")

    def test_missing_cell_blocked(self):
        winners = _fake_winners()
        del winners["cells"][next(iter(winners["cells"]))]
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "exactly 12 cells"):
            prepare_stability.prepare(winners_path, self.tmp / "stability")

    def test_missing_seed_run_id_blocked(self):
        winners = _fake_winners()
        first_cell = next(iter(winners["cells"].values()))
        del first_cell["winner"]["run_ids"]["1"]
        winners_path = self._write_winners(winners)
        with self.assertRaisesRegex(ValueError, "seeds 0, 1, 2"):
            prepare_stability.prepare(winners_path, self.tmp / "stability")

    def test_real_v4_winners_do_not_exist_yet(self):
        # Ground truth for the current, real repo state: V4 has not run.
        self.assertFalse((
            REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
            / "psi_adjudication_post_bn_v4/v4_winners.json"
        ).exists())


if __name__ == "__main__":
    unittest.main()
