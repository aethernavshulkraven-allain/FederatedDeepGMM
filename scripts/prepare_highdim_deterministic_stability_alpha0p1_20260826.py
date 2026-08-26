#!/usr/bin/env python3
"""Prepare the alpha=0.1 stability manifest from the 12 frozen V4 winners.

Input contract (v4_winners.json), deliberately decoupled from whichever
specific V4 scoring script eventually produces it (closeout plan Phase 5 is
not this campaign's scope yet, and score_highdim_adjudication_20260819.py is
another session's in-progress work this closeout pass does not touch):

    {
      "status": "complete",
      "alpha": 0.5,
      "seeds": [0, 1, 2],
      "cells": {
        "<dataset>|<method>": {
          "dataset": "...", "method": "...",
          "winner": {"lr": <float>, "cm": <float>,
                     "run_ids": {"0": "...", "1": "...", "2": "..."}}
        },
        ... exactly 12 entries ...
      }
    }

Whatever V4 finalization step is eventually built must emit exactly this
shape (or a thin adapter must translate to it) before this preparer can run
for real. Until then it fails closed -- there are no real 12 frozen winners
yet, so no stability manifest should be generated for real.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import CORE_DATASET_FILES, CORE_SOURCES  # noqa: E402

PROTOCOL_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
DEFAULT_OUTPUT_DIR = PROTOCOL_ROOT / "deterministic_stability_alpha0p1_20260826"
RESULT_ROOT = "results/highdim_deterministic_stability_alpha0p1_20260826"
FIELDNAMES = [
    "run_id", "dataset", "method", "seed", "alpha", "learning_rate",
    "critic_multiplier", "client_optimizer", "comm_round", "final_result_dir",
    "server_buffer_policy", "source_v4_run_id",
]
METHOD_OPTIMIZERS = {"fedgda_d": "sgd", "fedogda_d": "ogda"}


def _token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def prepare(winners_path: Path, output_dir: Path) -> dict:
    winners = _load_json(winners_path)
    if winners.get("status") != "complete":
        raise ValueError("V4 winners are absent or incomplete")
    if float(winners.get("alpha", -1)) != 0.5:
        raise ValueError("V4 winners must be frozen at alpha=0.5")
    if list(winners.get("seeds", [])) != [0, 1, 2]:
        raise ValueError("V4 winners must cover exactly seeds 0, 1, 2")
    cells = winners.get("cells")
    if not isinstance(cells, dict) or len(cells) != 12:
        raise ValueError(f"V4 winners must contain exactly 12 cells, got {len(cells) if isinstance(cells, dict) else 'invalid'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell_name in sorted(cells):
        cell = cells[cell_name]
        dataset, method = cell["dataset"], cell["method"]
        if f"{dataset}|{method}" != cell_name:
            raise ValueError(f"cell key {cell_name!r} does not match its dataset/method fields")
        if method not in METHOD_OPTIMIZERS:
            raise ValueError(f"{cell_name}: unsupported method {method!r}")
        winner = cell["winner"]
        lr, cm = float(winner["lr"]), float(winner["cm"])
        run_ids_by_seed = winner["run_ids"]
        if set(run_ids_by_seed) != {"0", "1", "2"}:
            raise ValueError(f"{cell_name}: winner must carry run_ids for seeds 0, 1, 2")
        run_id = (
            f"det_stability_alpha0p1_{dataset}_{method}_seed0_alpha0p1_"
            f"lr{_token(lr)}_cm{_token(cm)}"
        )
        rows.append({
            "run_id": run_id,
            "dataset": dataset,
            "method": method,
            "seed": "0",
            "alpha": "0.1",
            "learning_rate": f"{lr:g}",
            "critic_multiplier": f"{cm:g}",
            "client_optimizer": METHOD_OPTIMIZERS[method],
            "comm_round": "500",
            "final_result_dir": f"{RESULT_ROOT}/{dataset}/{method}/seed_0/{run_id}",
            "server_buffer_policy": "direct_client_aggregate",
            "source_v4_run_id": run_ids_by_seed["0"],
        })

    manifest_path = output_dir / "stability_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "stability_summary.json"
    summary_path.write_text(json.dumps({
        "campaign": "highdim_deterministic_stability_alpha0p1_20260826",
        "fresh_initialization": True,
        "run_count": len(rows),
        "alpha": 0.1,
        "seed": 0,
        "server_buffer_policy": "direct_client_aggregate",
        "source_v4_winners": _relative_or_str(winners_path),
    }, indent=2, sort_keys=True) + "\n")

    hashed_paths = [
        manifest_path,
        summary_path,
        winners_path,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        REPO_ROOT / "scripts/validate_highdim_stability_alpha0p1_20260826.py",
        REPO_ROOT / "scripts/launch_highdim_deterministic_stability_alpha0p1_20260826.sh",
        Path(__file__),
    ]
    (output_dir / "generated_artifact_hashes.json").write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for path in sorted(hashed_paths)
    ], indent=2, sort_keys=True) + "\n")

    return {"manifest": str(manifest_path), "runs": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v4-winners", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = prepare(args.v4_winners.resolve(), args.output_dir.resolve())
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"STABILITY PREPARATION BLOCKED: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
