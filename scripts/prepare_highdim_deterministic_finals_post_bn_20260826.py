#!/usr/bin/env python3
"""Prepare the 3-alpha x 5-seed final deterministic manifest (closeout plan
Phase 6 / SS9.2), under the exact-reuse accounting frozen in
PROTOCOL_DECISION_ADDENDUM_20260826.md SS6:

  V4 winners at alpha=0.5, seeds 0-2 (reused)........... 36
  alpha=0.1 stability runs, seed 0 (reused)............. 12
  New alpha=0.1 runs, seeds 1-4.......................... 48
  New alpha=0.5 runs, seeds 3-4.......................... 24
  New alpha=1.0 runs, seeds 0-4.......................... 60
  ------------------------------------------------------------
  Total final evidence (6 datasets x 2 methods x 3 alphas x 5 seeds) = 180
  New runs required after V4 + stability............... 132

Consumes the same v4_winners.json contract as
prepare_highdim_deterministic_stability_alpha0p1_20260826.py, plus that
stage's stability_results.json. Fails closed -- rather than silently
skipping or improvising -- if any cell's stability check required the
frozen alpha=0.1 retune escape hatch: retuning a cell is deliberate,
per-cell work (closeout plan SS9.3), not something this preparer performs
automatically.
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
DEFAULT_OUTPUT_DIR = PROTOCOL_ROOT / "deterministic_finals_post_bn_20260826"
RESULT_ROOT = "results/highdim_deterministic_finals_post_bn_20260826"
FIELDNAMES = [
    "run_id", "dataset", "method", "seed", "alpha", "learning_rate",
    "critic_multiplier", "client_optimizer", "comm_round", "final_result_dir",
    "server_buffer_policy", "reused", "source_stage", "source_run_id",
]
METHOD_OPTIMIZERS = {"fedgda_d": "sgd", "fedogda_d": "ogda"}
ALPHA_TOKENS = {0.1: "alpha0p1", 0.5: "alpha0p5", 1.0: "alpha1"}


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


def _new_row(dataset, method, seed, alpha, lr, cm):
    run_id = (
        f"det_finals_postbn_{dataset}_{method}_seed{seed}_{ALPHA_TOKENS[alpha]}_"
        f"lr{_token(lr)}_cm{_token(cm)}"
    )
    return {
        "run_id": run_id, "dataset": dataset, "method": method, "seed": str(seed),
        "alpha": f"{alpha:g}", "learning_rate": f"{lr:g}", "critic_multiplier": f"{cm:g}",
        "client_optimizer": METHOD_OPTIMIZERS[method], "comm_round": "500",
        "final_result_dir": f"{RESULT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}",
        "server_buffer_policy": "direct_client_aggregate",
        "reused": "False", "source_stage": "", "source_run_id": "",
    }


def _reused_row(dataset, method, seed, alpha, lr, cm, *, source_stage, source_run_id, source_final_result_dir):
    run_id = (
        f"det_finals_postbn_{dataset}_{method}_seed{seed}_{ALPHA_TOKENS[alpha]}_"
        f"lr{_token(lr)}_cm{_token(cm)}"
    )
    return {
        "run_id": run_id, "dataset": dataset, "method": method, "seed": str(seed),
        "alpha": f"{alpha:g}", "learning_rate": f"{lr:g}", "critic_multiplier": f"{cm:g}",
        "client_optimizer": METHOD_OPTIMIZERS[method], "comm_round": "500",
        "final_result_dir": source_final_result_dir,
        "server_buffer_policy": "direct_client_aggregate",
        "reused": "True", "source_stage": source_stage, "source_run_id": source_run_id,
    }


def prepare(winners_path: Path, stability_results_path: Path, stability_manifest_path: Path,
            output_dir: Path) -> dict:
    winners = _load_json(winners_path)
    if winners.get("status") != "complete":
        raise ValueError("V4 winners are absent or incomplete")
    if list(winners.get("seeds", [])) != [0, 1, 2]:
        raise ValueError("V4 winners must cover exactly seeds 0, 1, 2")
    v4_cells = winners.get("cells")
    if not isinstance(v4_cells, dict) or len(v4_cells) != 12:
        raise ValueError("V4 winners must contain exactly 12 cells")

    stability = _load_json(stability_results_path)
    if stability.get("status") != "complete":
        raise ValueError("stability results are absent or incomplete")
    stability_cells = stability.get("cells")
    if not isinstance(stability_cells, dict) or len(stability_cells) != 12:
        raise ValueError("stability results must contain exactly 12 cells")
    retune_required = [
        name for name, cell in stability_cells.items() if cell.get("outcome") != "pass"
    ]
    if retune_required:
        raise ValueError(
            "cannot generate the final matrix: these cells require the frozen "
            f"per-cell alpha=0.1 retune escape hatch first (closeout plan SS9.3), "
            f"which this preparer does not perform automatically: {sorted(retune_required)}"
        )

    with stability_manifest_path.open(newline="") as handle:
        stability_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    stability_run_id_by_cell = {}
    for cell_name, cell in stability_cells.items():
        run_id = cell.get("run_id")
        if run_id not in stability_rows:
            raise ValueError(f"{cell_name}: stability result run_id {run_id!r} not found in stability manifest")
        stability_run_id_by_cell[cell_name] = run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell_name in sorted(v4_cells):
        cell = v4_cells[cell_name]
        dataset, method = cell["dataset"], cell["method"]
        if method not in METHOD_OPTIMIZERS:
            raise ValueError(f"{cell_name}: unsupported method {method!r}")
        winner = cell["winner"]
        lr, cm = float(winner["lr"]), float(winner["cm"])
        run_ids_by_seed = winner["run_ids"]
        if set(run_ids_by_seed) != {"0", "1", "2"}:
            raise ValueError(f"{cell_name}: winner must carry run_ids for seeds 0, 1, 2")
        if cell_name not in stability_run_id_by_cell:
            raise ValueError(f"{cell_name}: present in V4 winners but not in stability results")

        # alpha=0.5, seeds 0-2: reused from V4 verbatim (36 rows).
        for seed_str, source_run_id in sorted(run_ids_by_seed.items()):
            seed = int(seed_str)
            source_final_result_dir = (
                f"results/highdim_psi_adjudication_post_bn_v4/{dataset}/{method}/"
                f"seed_{seed}/{source_run_id}"
            )
            rows.append(_reused_row(
                dataset, method, seed, 0.5, lr, cm,
                source_stage="psi_adjudication_post_bn_v4",
                source_run_id=source_run_id,
                source_final_result_dir=source_final_result_dir,
            ))
        # alpha=0.1, seed 0: reused from the stability stage verbatim (12 rows).
        stability_run_id = stability_run_id_by_cell[cell_name]
        rows.append(_reused_row(
            dataset, method, 0, 0.1, lr, cm,
            source_stage="deterministic_stability_alpha0p1_20260826",
            source_run_id=stability_run_id,
            source_final_result_dir=stability_rows[stability_run_id]["final_result_dir"],
        ))
        # New alpha=0.1, seeds 1-4 (48 rows).
        for seed in (1, 2, 3, 4):
            rows.append(_new_row(dataset, method, seed, 0.1, lr, cm))
        # New alpha=0.5, seeds 3-4 (24 rows).
        for seed in (3, 4):
            rows.append(_new_row(dataset, method, seed, 0.5, lr, cm))
        # New alpha=1.0, seeds 0-4 (60 rows).
        for seed in (0, 1, 2, 3, 4):
            rows.append(_new_row(dataset, method, seed, 1.0, lr, cm))

    manifest_path = output_dir / "finals_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    reused_count = sum(1 for row in rows if row["reused"] == "True")
    new_count = len(rows) - reused_count
    summary_path = output_dir / "finals_summary.json"
    summary_path.write_text(json.dumps({
        "campaign": "highdim_deterministic_finals_post_bn_20260826",
        "total_trajectories": len(rows),
        "reused_trajectories": reused_count,
        "new_trajectories": new_count,
        "server_buffer_policy": "direct_client_aggregate",
        "source_v4_winners": _relative_or_str(winners_path),
        "source_stability_results": _relative_or_str(stability_results_path),
    }, indent=2, sort_keys=True) + "\n")

    hashed_paths = [
        manifest_path, summary_path, winners_path, stability_results_path, stability_manifest_path,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        REPO_ROOT / "scripts/launch_highdim_deterministic_finals_post_bn_20260826.sh",
        REPO_ROOT / "scripts/aggregate_highdim_deterministic_finals_post_bn_20260826.py",
        Path(__file__),
    ]
    (output_dir / "generated_artifact_hashes.json").write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for path in sorted(hashed_paths)
    ], indent=2, sort_keys=True) + "\n")

    return {
        "manifest": str(manifest_path),
        "total": len(rows),
        "reused": reused_count,
        "new": new_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v4-winners", type=Path, required=True)
    parser.add_argument("--stability-results", type=Path, required=True)
    parser.add_argument("--stability-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = prepare(
            args.v4_winners.resolve(), args.stability_results.resolve(),
            args.stability_manifest.resolve(), args.output_dir.resolve(),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"FINALS PREPARATION BLOCKED: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
