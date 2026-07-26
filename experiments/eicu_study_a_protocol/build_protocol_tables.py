#!/usr/bin/env python3
"""Build deterministic descriptive tables for eICU Study A protocol v1.

This script creates protocol artifacts only. It does not create launch
manifests, read eICU data, inspect results, or run training.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

G0_VARIANTS = ("linear", "interaction", "frozen_random_mlp")
SEED_PAIRS = (
    ("confirmatory_01", 101, 1101),
    ("confirmatory_02", 102, 1102),
    ("confirmatory_03", 103, 1103),
    ("confirmatory_04", 104, 1104),
    ("confirmatory_05", 105, 1105),
)
FEDERATED_METHODS = ("fedgda_s", "fedogda_s")
CENTRALIZED_METHODS = ("gda_d", "sgda_s", "oadam_s")
PRIMARY_METRIC = "equal_client_test_mse_at_best_validation"

MATRIX_FIELDS = (
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
)

REGISTRY_FIELDS = (
    "target_id",
    "track",
    "source_version",
    "source_location",
    "dataset_family",
    "dataset",
    "input_mode",
    "scenario",
    "true_function",
    "algorithm_id",
    "algorithm_family",
    "training_scope",
    "stochasticity",
    "paper_method_label",
    "paper_mapping_status",
    "paper_target_available",
    "target_status",
    "target_mean_test_mse",
    "target_std_test_mse",
    "paper_reported_value",
    "target_metric",
    "target_num_runs",
    "seed_values",
    "curve_required",
    "aggregate_test_mse_required",
    "train_size",
    "validation_size",
    "test_size",
    "model_architecture",
    "partition_method",
    "partition_alpha",
    "paper_batch_size",
    "paper_batch_size_status",
    "y_standardization",
    "learning_rate_tuning",
    "checkpoint_selection_method",
    "communication_rounds",
    "total_clients",
    "clients_per_round",
    "local_epochs",
    "server_learning_rate",
    "weight_decay",
    "gradient_clipping",
    "paper_protocol_match_required",
    "comparison_reference",
    "tuning_rule",
    "fairness_rule",
    "test_tuning_allowed",
    "our_result_status",
    "our_completed_seeds",
    "our_result_dir_pattern",
    "our_mean_test_mse",
    "our_std_test_mse",
    "absolute_mean_gap",
    "relative_mean_gap_pct",
    "protocol_match_status",
    "notes",
)


def matrix_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    role_specs = (
        ("confirmatory", FEDERATED_METHODS, "uniform_clients"),
        ("centralized_baseline", CENTRALIZED_METHODS, "not_applicable_centralized"),
        ("aggregation_ablation", FEDERATED_METHODS, "sample_size"),
    )
    for role, methods, aggregation_mode in role_specs:
        for g0 in G0_VARIANTS:
            for seed_pair_id, scenario_seed, optimizer_seed in SEED_PAIRS:
                for method in methods:
                    experiment_id = (
                        f"study_a_v1__{role}__{g0}__{method}__{seed_pair_id}"
                    )
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "protocol_version": "eicu_study_a_v1",
                            "g0": g0,
                            "seed_pair_id": seed_pair_id,
                            "scenario_seed": scenario_seed,
                            "optimizer_seed": optimizer_seed,
                            "method": method,
                            "role": role,
                            "aggregation_mode": aggregation_mode,
                            "objective_mode": "paper_aligned",
                            "expected_primary_metric": PRIMARY_METRIC,
                        }
                    )
    return rows


def _true_function(g0: str) -> str:
    return {
        "linear": "known linear g0(D,W)",
        "interaction": "known interaction g0(D,W)",
        "frozen_random_mlp": "known frozen-random-MLP g0(D,W)",
    }[g0]


def _method_metadata(method: str) -> dict[str, str]:
    return {
        "fedgda_s": {
            "family": "fedgda",
            "scope": "federated",
            "stochasticity": "stochastic",
            "paper_label": "FDeepGMM-SGDA",
            "mapping": "method_mapping_only_scenario_not_in_paper",
        },
        "fedogda_s": {
            "family": "fedogda",
            "scope": "federated",
            "stochasticity": "stochastic",
            "paper_label": "",
            "mapping": "method_not_in_paper",
        },
        "gda_d": {
            "family": "gda",
            "scope": "centralized",
            "stochasticity": "deterministic",
            "paper_label": "DeepGMM-GDA",
            "mapping": "method_mapping_only_scenario_not_in_paper",
        },
        "sgda_s": {
            "family": "sgda",
            "scope": "centralized",
            "stochasticity": "stochastic",
            "paper_label": "DeepGMM-SGDA",
            "mapping": "method_mapping_only_scenario_not_in_paper",
        },
        "oadam_s": {
            "family": "oadam",
            "scope": "centralized",
            "stochasticity": "stochastic",
            "paper_label": "DeepGMM-OAdam",
            "mapping": "method_mapping_only_scenario_not_in_paper",
        },
    }[method]


def registry_rows() -> list[dict[str, object]]:
    """Return aggregate target rows in the current main-registry schema."""

    rows: list[dict[str, object]] = []
    role_specs = (
        ("confirmatory", FEDERATED_METHODS, "uniform_clients"),
        ("centralized_baseline", CENTRALIZED_METHODS, "not_applicable_centralized"),
        ("aggregation_ablation", FEDERATED_METHODS, "sample_size"),
    )
    seed_text = "scenario:101|102|103|104|105;optimizer:1101|1102|1103|1104|1105"
    for role, methods, aggregation_mode in role_specs:
        for g0 in G0_VARIANTS:
            for method in methods:
                meta = _method_metadata(method)
                target_id = (
                    f"extension_eicu_study_a_{g0}_{method}_{aggregation_mode}"
                )
                if role == "aggregation_ablation":
                    comparison = (
                        f"extension_eicu_study_a_{g0}_{method}_uniform_clients"
                    )
                elif method == "fedogda_s":
                    comparison = (
                        f"extension_eicu_study_a_{g0}_fedgda_s_uniform_clients"
                    )
                else:
                    comparison = ""

                row = {field: "" for field in REGISTRY_FIELDS}
                row.update(
                    {
                        "target_id": target_id,
                        "track": "extension",
                        "source_version": "eicu_study_a_v1",
                        "source_location": (
                            "experiments/eicu_study_a_protocol/protocol_v1.md"
                        ),
                        "dataset_family": "eicu_semisynthetic",
                        "dataset": "eicu",
                        "input_mode": "structured_tabular",
                        "scenario": f"eicu_study_a_{g0}",
                        "true_function": _true_function(g0),
                        "algorithm_id": method,
                        "algorithm_family": meta["family"],
                        "training_scope": meta["scope"],
                        "stochasticity": meta["stochasticity"],
                        "paper_method_label": meta["paper_label"],
                        "paper_mapping_status": meta["mapping"],
                        "paper_target_available": "false",
                        "target_status": "new_experiment_no_paper_target",
                        "target_metric": PRIMARY_METRIC,
                        "target_num_runs": 5,
                        "seed_values": seed_text,
                        "curve_required": "true",
                        "aggregate_test_mse_required": "true",
                        "train_size": "full_eicu_audit_pending",
                        "validation_size": "full_eicu_audit_pending",
                        "test_size": "full_eicu_audit_pending",
                        "model_architecture": "not_frozen_pending_preflight",
                        "partition_method": "natural_hospital",
                        "partition_alpha": "",
                        "paper_batch_size": "",
                        "paper_batch_size_status": "not_applicable_extension",
                        "y_standardization": "not_frozen_pending_preflight",
                        "learning_rate_tuning": "validation_only_staged_grid",
                        "checkpoint_selection_method": (
                            "lowest_equal_client_validation_structural_mse"
                        ),
                        "communication_rounds": "not_frozen_pending_preflight",
                        "total_clients": "full_eicu_audit_pending",
                        "clients_per_round": (
                            "all_eligible_proposed_pending_preflight"
                            if meta["scope"] == "federated"
                            else "not_applicable_centralized"
                        ),
                        "local_epochs": (
                            "not_frozen_pending_preflight"
                            if meta["scope"] == "federated"
                            else "not_applicable_centralized"
                        ),
                        "server_learning_rate": (
                            "validation_tuned"
                            if meta["scope"] == "federated"
                            else "not_applicable_centralized"
                        ),
                        "weight_decay": "validation_tuned",
                        "gradient_clipping": "validation_tuned",
                        "paper_protocol_match_required": "false",
                        "comparison_reference": comparison,
                        "tuning_rule": "validation_only",
                        "fairness_rule": (
                            "matched scenario artifacts, seed pairs, architecture "
                            "family, training budget, and comparable tuning budget"
                        ),
                        "test_tuning_allowed": "false",
                        "our_result_status": "not_started",
                        "protocol_match_status": "not_applicable_extension",
                        "notes": (
                            f"Study A v1 proposal; role={role}; "
                            f"aggregation_mode={aggregation_mode}; "
                            "objective_mode=paper_aligned; full eICU required; "
                            "demo smoke-only; primary metric is equal-client; "
                            "no published numerical target."
                        ),
                    }
                )
                rows.append(row)
    return rows


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    matrix = matrix_rows()
    registry = registry_rows()
    write_csv(ROOT / "confirmatory_matrix.csv", MATRIX_FIELDS, matrix)
    write_csv(ROOT / "proposed_registry_rows.csv", REGISTRY_FIELDS, registry)
    print(f"wrote confirmatory_matrix.csv rows={len(matrix)}")
    print(f"wrote proposed_registry_rows.csv rows={len(registry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
