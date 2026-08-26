#!/usr/bin/env python3
"""Fail unless every manifest row has exactly one resolved launcher result."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import ManifestLaunchError, validate_artifacts  # noqa: E402


RESOLVED_STATUSES = {
    "passed",
    "skipped_completed",
    "terminal_ineligible",
    "skipped_terminal_ineligible",
}


def check_stage(
    manifest_path: Path,
    results_path: Path,
    *,
    require_clean: bool = False,
    validate_stage_artifacts: bool = False,
) -> dict[str, int]:
    with manifest_path.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with results_path.open() as handle:
        result_rows = json.load(handle)
    if not isinstance(result_rows, list):
        raise ValueError("launcher results must be a JSON list")
    if not manifest_rows:
        raise ValueError("manifest contains no runs")

    manifest_ids = [str(row.get("run_id", "")) for row in manifest_rows]
    result_ids = [str(row.get("run_id", "")) for row in result_rows]
    if any(not run_id for run_id in manifest_ids):
        raise ValueError("manifest contains a blank run_id")
    if any(not run_id for run_id in result_ids):
        raise ValueError("launcher results contain a blank run_id")

    manifest_duplicates = sorted(
        run_id for run_id, count in Counter(manifest_ids).items() if count != 1
    )
    result_duplicates = sorted(
        run_id for run_id, count in Counter(result_ids).items() if count != 1
    )
    if manifest_duplicates:
        raise ValueError(f"duplicate manifest run_ids: {manifest_duplicates}")
    if result_duplicates:
        raise ValueError(f"duplicate launcher-result run_ids: {result_duplicates}")
    if set(manifest_ids) != set(result_ids):
        missing = sorted(set(manifest_ids) - set(result_ids))
        unexpected = sorted(set(result_ids) - set(manifest_ids))
        raise ValueError(
            f"manifest/results mismatch; missing={missing}, unexpected={unexpected}"
        )

    allowed_statuses = (
        {"passed", "skipped_completed"} if require_clean else RESOLVED_STATUSES
    )
    failed = {
        row["run_id"]: row.get("status")
        for row in result_rows
        if row.get("status") not in allowed_statuses
    }
    if failed:
        raise ValueError(f"unresolved launcher statuses: {failed}")

    if validate_stage_artifacts:
        results_by_id = {str(row["run_id"]): row for row in result_rows}
        for manifest_row in manifest_rows:
            run_id = str(manifest_row["run_id"])
            result_row = results_by_id[run_id]
            run_dir_text = str(manifest_row.get("final_result_dir", "")).strip()
            if not run_dir_text:
                raise ValueError(f"manifest {run_id} has no final_result_dir")
            run_dir = Path(run_dir_text)
            if not run_dir.is_absolute():
                run_dir = REPO_ROOT / run_dir
            result_run_dir_text = str(result_row.get("run_dir", "")).strip()
            if not result_run_dir_text:
                raise ValueError(f"launcher result {run_id} has no run_dir")
            result_run_dir = Path(result_run_dir_text)
            if not result_run_dir.is_absolute():
                result_run_dir = REPO_ROOT / result_run_dir
            if result_run_dir.resolve() != run_dir.resolve():
                raise ValueError(
                    f"launcher result {run_id} points to {result_run_dir}; "
                    f"manifest expects {run_dir}"
                )
            try:
                validation = validate_artifacts(run_dir, manifest_row)
            except ManifestLaunchError as exc:
                raise ValueError(f"artifact validation failed for {run_id}: {exc}") from exc
            terminal = bool(validation["terminal_ineligible"])
            status = str(result_row.get("status"))
            reports_terminal = status in {
                "terminal_ineligible",
                "skipped_terminal_ineligible",
            }
            if terminal != reports_terminal:
                raise ValueError(
                    f"launcher status/artifact classification mismatch for {run_id}: "
                    f"status={status!r}, terminal_ineligible={terminal}"
                )
    return {
        "manifest_rows": len(manifest_rows),
        "resolved_rows": len(result_rows),
        "terminal_ineligible": sum(
            row.get("status") in {"terminal_ineligible", "skipped_terminal_ineligible"}
            for row in result_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Reject terminal-ineligible results; intended for preflight diagnostics.",
    )
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="Revalidate each resolved run against its manifest before opening the gate.",
    )
    args = parser.parse_args()
    try:
        summary = check_stage(
            args.manifest,
            args.results,
            require_clean=bool(args.require_clean),
            validate_stage_artifacts=bool(args.validate_artifacts),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"STAGE BLOCKED: {exc}")
        return 1
    print(json.dumps({"stage_complete": True, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
