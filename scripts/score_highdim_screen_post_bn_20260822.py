#!/usr/bin/env python3
"""Score only a complete, corrected high-dimensional image screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import ManifestLaunchError, validate_artifacts  # noqa: E402
from score_highdim_screen_by_psi import boundary_flags, rank_cell  # noqa: E402

EXPECTED_CELLS = 12
SCREEN_COMM_ROUND = 150
LAST50_WINDOW = 50


def _last50_mean(run_dir: Path, run_id: str) -> tuple[float, float]:
    """Mean gmm_eval (Psi) and mean primary_val_mse over the last 50 of the
    screen's 150 communication rounds, per the frozen last-50 scorer rule
    (PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md). best_gmm_eval (the
    best-round value) is diagnostic only and must never be used here."""
    curve_path = run_dir / "mse_by_round.csv"
    with curve_path.open(newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    if len(curve_rows) != SCREEN_COMM_ROUND:
        raise ValueError(
            f"{run_id}: mse_by_round.csv has {len(curve_rows)} rows, "
            f"expected exactly {SCREEN_COMM_ROUND}"
        )
    for index, curve_row in enumerate(curve_rows):
        try:
            round_index = int(curve_row.get("round", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{run_id}: mse_by_round.csv[{index}].round is not an integer") from exc
        if round_index != index:
            raise ValueError(
                f"{run_id}: mse_by_round.csv[{index}].round is {round_index}, "
                f"expected {index} (missing, duplicated, or unordered rows)"
            )
    window = curve_rows[SCREEN_COMM_ROUND - LAST50_WINDOW:SCREEN_COMM_ROUND]
    psi_values = []
    mse_values = []
    for offset, curve_row in enumerate(window):
        round_index = SCREEN_COMM_ROUND - LAST50_WINDOW + offset
        for field, sink in (("gmm_eval", psi_values), ("primary_val_mse", mse_values)):
            raw_value = curve_row.get(field)
            if raw_value is None or str(raw_value).strip() == "":
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is blank")
            try:
                number = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is not numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is nonfinite")
            sink.append(number)
    return sum(psi_values) / len(psi_values), sum(mse_values) / len(mse_values)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def score_screen(manifest_path: Path) -> dict:
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 108:
        raise ValueError(f"corrected screen must contain 108 rows, got {len(rows)}")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("corrected screen manifest contains duplicate run_ids")

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    invalid = []
    terminal = []
    for row in rows:
        if row.get("server_buffer_policy") != "direct_client_aggregate":
            raise ValueError(f"{row['run_id']} does not freeze the corrected buffer policy")
        run_dir = Path(row["final_result_dir"])
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        try:
            validation = validate_artifacts(run_dir, row)
        except ManifestLaunchError as exc:
            invalid.append({"run_id": row["run_id"], "reason": str(exc)})
            continue
        if validation["terminal_ineligible"]:
            terminal.append({
                "run_id": row["run_id"],
                "reason": validation["terminal_reason"],
            })
            continue
        metrics = _load_json(run_dir / "metrics.json")
        try:
            psi_last50, mse_last50 = _last50_mean(run_dir, row["run_id"])
        except (OSError, ValueError) as exc:
            invalid.append({"run_id": row["run_id"], "reason": str(exc)})
            continue
        try:
            best_gmm_eval_diagnostic = float(metrics["best_gmm_eval"])
            if not math.isfinite(best_gmm_eval_diagnostic):
                raise ValueError("nonfinite best_gmm_eval")
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append({"run_id": row["run_id"], "reason": f"invalid diagnostic best_gmm_eval: {exc}"})
            continue
        cells[(row["dataset"], row["method"])].append({
            "run_id": row["run_id"],
            "lr": float(row["learning_rate"]),
            "cm": float(row["critic_multiplier"]),
            "gmm_eval": psi_last50,
            "val_mse": mse_last50,
            "best_gmm_eval_diagnostic": best_gmm_eval_diagnostic,
        })

    if invalid:
        raise ValueError(
            f"screen is incomplete or malformed ({len(invalid)} runs); first={invalid[0]}"
        )
    if len(cells) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} image cells, got {len(cells)}")

    planned_by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        planned_by_cell[(row["dataset"], row["method"])].append({
            "lr": float(row["learning_rate"]),
            "cm": float(row["critic_multiplier"]),
        })

    output_cells = {}
    boundary_review_cells = []
    terminal_ids = {entry["run_id"] for entry in terminal}
    for cell_key in sorted(planned_by_cell):
        eligible = cells.get(cell_key, [])
        if len(eligible) < 2:
            raise ValueError(f"{cell_key} has fewer than two eligible candidates")
        psi_ranked = rank_cell(eligible)
        # rank_cell's own sort key is (-round(gmm_eval, 9), val_mse) -- val_mse
        # already acts as the documented secondary tiebreak for a gmm_eval tie,
        # so that alone is not an unresolved tie. It's unresolved only when
        # BOTH keys tie exactly, at which point Python's stable sort would
        # otherwise silently fall back to manifest order.
        psi_top_key = (round(psi_ranked[0]["gmm_eval"], 9), psi_ranked[0]["val_mse"])
        psi_tie_set = [
            candidate for candidate in psi_ranked
            if (round(candidate["gmm_eval"], 9), candidate["val_mse"]) == psi_top_key
        ]
        if len(psi_tie_set) > 1:
            raise ValueError(
                f"{cell_key} has an unresolved exact Psi tie among "
                f"{[candidate['run_id'] for candidate in psi_tie_set]}"
            )
        mse_minimum = min(candidate["val_mse"] for candidate in eligible)
        mse_winners = [
            candidate for candidate in eligible if candidate["val_mse"] == mse_minimum
        ]
        if len(mse_winners) != 1:
            raise ValueError(f"{cell_key} has an unresolved exact validation-MSE tie")
        mse_winner = mse_winners[0]
        # BOUNDARY_RULE_AMENDMENT_20260818.md SS"Replacement rule" step 1 is
        # scoped to "the Psi rank-1 candidate" only -- score_highdim_screen_
        # by_psi.py's reference implementation likewise only ever checks
        # rank_cell's top candidate. mse_flags is retained as diagnostic
        # metadata (it's informative that the MSE winner also sits at a
        # boundary) but must not itself trigger the frozen review/expansion
        # rule -- that would review/expand a candidate the rule never named.
        psi_flags = boundary_flags(psi_ranked[0], planned_by_cell[cell_key])
        mse_flags = boundary_flags(mse_winner, planned_by_cell[cell_key])
        review_required = bool(psi_flags)
        cell_name = f"{cell_key[0]}|{cell_key[1]}"
        if review_required:
            boundary_review_cells.append(cell_name)
        output_cells[cell_name] = {
            "dataset": cell_key[0],
            "method": cell_key[1],
            "psi_rank_1": psi_ranked[0],
            "psi_rank_2": psi_ranked[1],
            "mse_winner": mse_winner,
            "eligible_candidates": len(eligible),
            "terminal_candidates": sum(
                row["run_id"] in terminal_ids
                for row in rows
                if (row["dataset"], row["method"]) == cell_key
            ),
            "boundary_review_required": review_required,
            "boundary_detail": {"psi_rank_1": psi_flags, "mse_winner": mse_flags},
        }
    return {
        "status": "complete",
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "server_buffer_policy": "direct_client_aggregate",
        "planned_runs": len(rows),
        "terminal_ineligible_runs": terminal,
        "boundary_review_required": bool(boundary_review_cells),
        "boundary_review_cells": boundary_review_cells,
        "cells": output_cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_screen(args.manifest.resolve())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"SCREEN SCORING BLOCKED: {exc}")
        return 2
    _atomic_json(args.out.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "cells": len(result["cells"]),
        "boundary_review_required": result["boundary_review_required"],
        "terminal_ineligible_runs": len(result["terminal_ineligible_runs"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
