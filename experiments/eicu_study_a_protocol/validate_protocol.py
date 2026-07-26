#!/usr/bin/env python3
"""Validate the machine-readable eICU Study A protocol artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_ROLE_COUNTS = {
    "confirmatory": 30,
    "centralized_baseline": 45,
    "aggregation_ablation": 30,
}
EXPECTED_PAIRING = {
    "confirmatory_01": (101, 1101),
    "confirmatory_02": (102, 1102),
    "confirmatory_03": (103, 1103),
    "confirmatory_04": (104, 1104),
    "confirmatory_05": (105, 1105),
}
EXPECTED_G0 = {"linear", "interaction", "frozen_random_mlp"}
EXPECTED_PRIMARY_METRIC = "equal_client_test_mse_at_best_validation"


def read_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_json() -> dict:
    with (ROOT / "protocol_v1.json").open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    assert protocol["protocol_version"] == "eicu_study_a_v1"
    assert set(protocol["roles"]) == {
        "smoke",
        "tuning",
        "confirmatory",
        "centralized_baseline",
        "aggregation_ablation",
    }
    assert protocol["required_matrix"]["total_rows"] == 105
    assert protocol["required_matrix"]["role_counts"] == EXPECTED_ROLE_COUNTS
    assert protocol["metrics_policy"]["test_tuning_allowed"] is False
    assert protocol["objective"]["lambda"] == 0.25
    assert protocol["objective"]["primary_aggregation_mode"] == "uniform_clients"
    assert protocol["objective"]["objective_mode"] == "paper_aligned"
    assert len(protocol["required_effective_config_fields"]) == len(
        set(protocol["required_effective_config_fields"])
    )
    assert len(protocol["required_metrics_fields"]) == len(
        set(protocol["required_metrics_fields"])
    )
    return protocol


def validate_matrix(protocol: dict) -> None:
    fields, rows = read_csv("confirmatory_matrix.csv")
    required_fields = {
        "experiment_id",
        "protocol_version",
        "g0",
        "seed_pair_id",
        "scenario_seed",
        "optimizer_seed",
        "method",
        "role",
        "aggregation_mode",
        "objective_mode",
        "expected_primary_metric",
    }
    assert set(fields) == required_fields
    assert len(rows) == 105, len(rows)

    ids = [row["experiment_id"] for row in rows]
    assert len(ids) == len(set(ids)), "experiment IDs are not unique"

    role_counts = Counter(row["role"] for row in rows)
    assert role_counts == Counter(EXPECTED_ROLE_COUNTS), role_counts
    assert Counter(row["g0"] for row in rows) == Counter(
        {"linear": 35, "interaction": 35, "frozen_random_mlp": 35}
    )
    assert set(row["g0"] for row in rows) == EXPECTED_G0
    assert Counter(row["method"] for row in rows) == Counter(
        {
            "fedgda_s": 30,
            "fedogda_s": 30,
            "gda_d": 15,
            "sgda_s": 15,
            "oadam_s": 15,
        }
    )

    for row in rows:
        assert row["protocol_version"] == protocol["protocol_version"]
        assert row["expected_primary_metric"] == EXPECTED_PRIMARY_METRIC
        assert row["objective_mode"] == "paper_aligned"
        pair = EXPECTED_PAIRING[row["seed_pair_id"]]
        assert (int(row["scenario_seed"]), int(row["optimizer_seed"])) == pair

        if row["role"] == "confirmatory":
            assert row["aggregation_mode"] == "uniform_clients"
            assert row["method"] in {"fedgda_s", "fedogda_s"}
        elif row["role"] == "centralized_baseline":
            assert row["aggregation_mode"] == "not_applicable_centralized"
            assert row["method"] in {"gda_d", "sgda_s", "oadam_s"}
        elif row["role"] == "aggregation_ablation":
            assert row["aggregation_mode"] == "sample_size"
            assert row["method"] in {"fedgda_s", "fedogda_s"}
        else:
            raise AssertionError(f"unexpected role: {row['role']}")

        if row["aggregation_mode"] == "sample_size":
            assert row["role"] == "aggregation_ablation"

    keys = Counter(
        (
            row["role"],
            row["g0"],
            row["seed_pair_id"],
            row["method"],
            row["aggregation_mode"],
        )
        for row in rows
    )
    assert all(count == 1 for count in keys.values())


def validate_registry() -> None:
    fields, rows = read_csv("proposed_registry_rows.csv")
    assert len(fields) == 57, len(fields)
    assert len(rows) == 21, len(rows)
    assert len({row["target_id"] for row in rows}) == 21
    assert all(row["track"] == "extension" for row in rows)
    assert all(row["paper_target_available"] == "false" for row in rows)
    assert all(row["test_tuning_allowed"] == "false" for row in rows)
    assert all(row["target_num_runs"] == "5" for row in rows)
    assert all(row["target_metric"] == EXPECTED_PRIMARY_METRIC for row in rows)
    assert sum("role=confirmatory" in row["notes"] for row in rows) == 6
    assert sum("role=centralized_baseline" in row["notes"] for row in rows) == 9
    assert sum("role=aggregation_ablation" in row["notes"] for row in rows) == 6


def main() -> int:
    protocol = validate_json()
    validate_matrix(protocol)
    validate_registry()
    print("protocol_v1.json: valid")
    print("confirmatory_matrix.csv: 105 rows, 105 unique IDs")
    print(
        "role counts: confirmatory=30, centralized_baseline=45, "
        "aggregation_ablation=30"
    )
    print("primary confirmatory invariant: uniform_clients + paper_aligned")
    print("sample_size invariant: aggregation_ablation only")
    print("proposed_registry_rows.csv: 21 extension rows, 21 unique target IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
