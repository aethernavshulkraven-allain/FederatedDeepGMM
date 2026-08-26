#!/usr/bin/env python3
"""Apply the SS6.2 screen-failure certification to the original screen rows.

The reproduction runs live in a dedicated namespace
(screen_failure_certification_20260826/) that never touches the original
screen directories. This script is the connective step Phase 4 needs: it
reads each reproduction's independently-gathered pretraining_failure.json,
confirms the failure genuinely reproduced (same phase, same terminal
reason), and writes a *new* pretraining_failure.json into the original
screen row's own directory -- the only file that directory gains; nothing
already there is touched or overwritten. That is what lets
score_highdim_screen_post_bn_20260822.py (which reads
validate_artifacts(row's own final_result_dir, row)) resolve these 4 rows
as terminal_pretraining_ineligible instead of "missing artifacts".

The applied artifact carries the original row's run_id (so
validate_pretraining_failure_artifact's run_id check passes for that row)
plus a certified_via_run_id field recording exactly which reproduction run
supplied the evidence, for full auditability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FEDGMM_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(FEDGMM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment_utils import config_checksum  # noqa: E402
from run_manifest import validate_pretraining_failure_artifact  # noqa: E402

SCREEN_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
    / "screen_manifest.csv"
)
CERT_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/screen_failure_certification_20260826"
    / "screen_failure_certification_manifest.csv"
)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    import csv

    with SCREEN_MANIFEST.open(newline="") as handle:
        screen_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    with CERT_MANIFEST.open(newline="") as handle:
        cert_rows = list(csv.DictReader(handle))

    applied = []
    for cert_row in cert_rows:
        cert_run_id = cert_row["run_id"]
        cert_run_dir = Path(cert_row["final_result_dir"])
        if not cert_run_dir.is_absolute():
            cert_run_dir = REPO_ROOT / cert_run_dir
        cert_payload = _load_json(cert_run_dir / "pretraining_failure.json")

        # The reproduction manifest's notes field records the original
        # screen run_id as "screen_failure_cert_20260826_<original_run_id>".
        original_run_id = cert_run_id.removeprefix("screen_failure_cert_20260826_")
        if original_run_id not in screen_rows:
            raise ValueError(f"cannot find original screen row for {cert_run_id}")
        original_row = screen_rows[original_run_id]
        original_run_dir = Path(original_row["final_result_dir"])
        if not original_run_dir.is_absolute():
            original_run_dir = REPO_ROOT / original_run_dir

        original_pretraining_failure = original_run_dir / "pretraining_failure.json"
        if original_pretraining_failure.exists():
            raise ValueError(
                f"{original_run_id}: pretraining_failure.json already exists in the "
                "original screen directory -- refusing to overwrite"
            )
        if cert_payload["failure_phase"] != "model_selection":
            raise ValueError(f"{cert_run_id}: reproduction is not a model_selection failure")

        original_config = _load_json(original_run_dir / "effective_config.json")
        applied_payload = dict(cert_payload)
        applied_payload["run_id"] = original_run_id
        applied_payload["effective_config_checksum"] = config_checksum(original_config)
        applied_payload["certified_via_run_id"] = cert_run_id
        applied_payload["certified_via_run_dir"] = str(
            cert_run_dir.relative_to(REPO_ROOT)
        )

        with original_pretraining_failure.open("w") as handle:
            json.dump(applied_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        # Re-validate through the real validator before trusting this.
        validate_pretraining_failure_artifact(original_run_dir, original_row)
        applied.append(original_run_id)

    print(json.dumps({"applied_to_original_screen_rows": applied}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
