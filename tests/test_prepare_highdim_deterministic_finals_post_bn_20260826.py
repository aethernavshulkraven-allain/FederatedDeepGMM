"""Tests for scripts/prepare_highdim_deterministic_finals_post_bn_20260826.py.

Synthetic V4-winners + stability-results fixtures, matching each script's
documented input contract -- V4 has not run yet (Phase 5 is out of scope),
so this exercises the exact-reuse accounting frozen in
PROTOCOL_DECISION_ADDENDUM_20260826.md SS6 without depending on real data.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_highdim_deterministic_finals_post_bn_20260826 as prepare_finals  # noqa: E402

DATASETS = ["femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz"]
METHODS = ("fedgda_d", "fedogda_d")


def _fake_winners() -> dict:
    cells = {}
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            cells[f"{dataset}|{method}"] = {
                "dataset": dataset, "method": method,
                "winner": {
                    "lr": 0.001 * (i + 1), "cm": 5.0 + j,
                    "run_ids": {"0": "v4_seed0", "1": "v4_seed1", "2": "v4_seed2"},
                },
            }
    return {"status": "complete", "alpha": 0.5, "seeds": [0, 1, 2], "cells": cells}


def _fake_stability(tmp: Path, *, all_pass: bool):
    manifest_path = tmp / "stability_manifest.csv"
    fieldnames = ["run_id", "final_result_dir"]
    rows = []
    cells = {}
    for dataset in DATASETS:
        for method in METHODS:
            cell_name = f"{dataset}|{method}"
            run_id = f"stability_{dataset}_{method}"
            rows.append({"run_id": run_id, "final_result_dir": f"results/stability/{run_id}"})
            cells[cell_name] = {
                "dataset": dataset, "method": method, "run_id": run_id,
                "outcome": "pass" if all_pass else (
                    "retune_required" if cell_name == f"{DATASETS[0]}|{METHODS[0]}" else "pass"
                ),
            }
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    results_path = tmp / "stability_results.json"
    results_path.write_text(json.dumps({"status": "complete", "cells": cells}))
    return results_path, manifest_path


class PrepareFinalsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.winners_path = self.tmp / "v4_winners.json"
        self.winners_path.write_text(json.dumps(_fake_winners()))

    def test_exact_reuse_accounting_matches_the_frozen_addendum(self):
        results_path, manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        result = prepare_finals.prepare(self.winners_path, results_path, manifest_path, output_dir)
        self.assertEqual(result["total"], 180)
        self.assertEqual(result["reused"], 48)  # 36 V4 + 12 stability
        self.assertEqual(result["new"], 132)  # 48 + 24 + 60

        with (output_dir / "finals_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 180)
        self.assertEqual(len({row["run_id"] for row in rows}), 180)
        self.assertEqual(len({row["final_result_dir"] for row in rows}), 180)

        # Per-cell alpha/seed accounting, independent of the global totals check.
        by_cell = {}
        for row in rows:
            by_cell.setdefault((row["dataset"], row["method"]), []).append(row)
        self.assertEqual(len(by_cell), 12)
        for cell_rows in by_cell.values():
            self.assertEqual(len(cell_rows), 15)  # 3 alphas x 5 seeds
            alpha_seed_pairs = Counter((r["alpha"], r["seed"]) for r in cell_rows)
            self.assertEqual(len(alpha_seed_pairs), 15)  # no duplicate (alpha, seed)
            reused = [r for r in cell_rows if r["reused"] == "True"]
            self.assertEqual(len(reused), 4)  # 3 alpha0.5 seeds 0-2 + 1 alpha0.1 seed 0

    def test_retune_required_cell_blocks_the_whole_matrix(self):
        results_path, manifest_path = _fake_stability(self.tmp, all_pass=False)
        with self.assertRaisesRegex(ValueError, "retune"):
            prepare_finals.prepare(self.winners_path, results_path, manifest_path, self.tmp / "finals")

    def test_reused_alpha0p5_rows_point_at_the_real_v4_run_ids(self):
        results_path, manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        prepare_finals.prepare(self.winners_path, results_path, manifest_path, output_dir)
        with (output_dir / "finals_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        alpha0p5_seed0 = [
            r for r in rows
            if r["alpha"] == "0.5" and r["seed"] == "0"
            and r["dataset"] == DATASETS[0] and r["method"] == METHODS[0]
        ]
        self.assertEqual(len(alpha0p5_seed0), 1)
        self.assertEqual(alpha0p5_seed0[0]["source_run_id"], "v4_seed0")
        self.assertEqual(alpha0p5_seed0[0]["source_stage"], "psi_adjudication_post_bn_v4")

    def test_reused_alpha0p1_seed0_points_at_the_real_stability_run(self):
        results_path, manifest_path = _fake_stability(self.tmp, all_pass=True)
        output_dir = self.tmp / "finals"
        prepare_finals.prepare(self.winners_path, results_path, manifest_path, output_dir)
        with (output_dir / "finals_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        target = [
            r for r in rows
            if r["alpha"] == "0.1" and r["seed"] == "0"
            and r["dataset"] == DATASETS[0] and r["method"] == METHODS[0]
        ]
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0]["reused"], "True")
        self.assertEqual(target[0]["source_stage"], "deterministic_stability_alpha0p1_20260826")


if __name__ == "__main__":
    unittest.main()
