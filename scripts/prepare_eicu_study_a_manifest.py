"""Manifest generator for the Study A federated campaign.

Produces CSV manifests in the same schema ``run_manifest.py`` already
consumes (extended with ``scenario_name``, ``input_dim_g/f``,
``objective_mode``, ``aggregation_weighting`` -- all wired into
``run_manifest.py``'s ``build_config``/``write_config``), so launching,
resuming, and validating reuses that existing, tested machinery rather than
duplicating it.

Three stages, matching the frozen Study A protocol:

* ``tuning``       -- seed 0 only, 3 g0 x 2 methods x 6 LR/server-LR
                      candidates = 36 rows, ~100-150 rounds each.
* ``confirmatory`` -- 3 g0 x 5 seeds x 2 methods = 30 rows, frozen
                      hyperparameters (from the tuning selection), 500 rounds.
* ``ablation``     -- linear g0 only, 5 seeds x 2 methods x sample_size
                      weighting = 10 *additional* rows (the uniform_clients
                      arm is already covered by the confirmatory stage, so it
                      is not duplicated here).

Every row uses: client_optimizer sgd/ogda (batch_size>0 -> the *_s variants,
matching how every other manifest in this repo derives fedgda_s/fedogda_s),
partition_method natural, objective_mode paper_aligned. Every row explicitly
sets skip_model_selection=true (Study A always uses exactly one g/f/learning
setup; the alternative model-selection path does not thread theta~ through and
is out of scope for this protocol -- see PaperAlignedMomentObjective's usage
notes).

Usage:
    python scripts/prepare_eicu_study_a_manifest.py --stage tuning \\
        --scenario-dir fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth \\
        --out experiments/eicu_study_a/tuning_manifest.csv
"""

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

G0_VARIANTS = ("linear", "interaction", "mlp")
METHODS = ("fedgda", "fedogda")  # -> fedgda_s / fedogda_s (client_optimizer sgd/ogda)
TUNING_SEED = 0
CONFIRMATORY_SEEDS = (0, 1, 2, 3, 4)

# LR multiplier applied to the verified smoke config's learning_rate; server LR
# candidates. 3 x 2 = 6 candidates per (g0, method), matching the frozen protocol.
TUNING_LR_MULTIPLIERS = (0.5, 1.0, 2.0)
TUNING_SERVER_LRS = (1.0, 1.5)

# From the verified smoke config (experiments/eicu_v1_demo/smoke_config.yaml).
BASE_LEARNING_RATE = 0.001
BASE_WEIGHT_DECAY = 0.01
BASE_CRITIC_MULTIPLIER = 10.0
BASE_GRADIENT_CLIP_NORM = 1.0

TUNING_COMM_ROUND = 150
CONFIRMATORY_COMM_ROUND = 500

CSV_FIELDNAMES = [
    "run_id", "protocol_version", "run_group", "training_scope", "method", "method_label",
    "dataset", "seed", "alpha", "output_root", "final_result_dir", "implementation_status",
    "run_status", "preflight_required", "preflight_status", "model", "federated_optimizer",
    "client_optimizer", "client_num_in_total", "client_num_per_round", "comm_round", "epochs",
    "batch_size", "partition_method", "partition_alpha", "data_cache_dir", "learning_rate",
    "learning_rate_status", "weight_decay", "critic_multiplier", "server_learning_rate",
    "gradient_clip_norm", "simple_model_selection_epochs", "f_history_model_selection_epochs",
    "model_selection_batch_size", "using_gpu", "gpu_id", "notes",
    "scenario_name", "objective_mode", "aggregation_weighting", "input_dim_g", "input_dim_f",
    "g0", "skip_model_selection",
]


def load_scenario_metadata(scenario_dir, g0, seed):
    meta_path = os.path.join(scenario_dir, f"{g0}_seed{seed}_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"missing {meta_path} -- run scripts/prepare_eicu_semisynth.py --g0 {g0} --seed {seed} first"
        )
    with open(meta_path) as handle:
        return json.load(handle)


def selection_key(g0, method):
    """Must match select_eicu_study_a_tuning.py's grouping key exactly: it
    groups by the manifest row's full ``method`` column (e.g. ``fedgda_s``),
    not the bare method name used to loop over METHODS here.
    """
    return f"{g0}:{method}_s"


def base_row(run_id, run_group, g0, seed, method, metadata, comm_round, output_root):
    variant_suffix = "s"  # every row is stochastic-minibatch (batch_size > 0)
    client_optimizer = "ogda" if method == "fedogda" else "sgd"
    n_clients = int(metadata["n_clients"])
    return {
        "run_id": run_id,
        "protocol_version": "study_a_v1",
        "run_group": run_group,
        "training_scope": "federated",
        "method": f"{method}_{variant_suffix}",
        "method_label": f"{method.upper()}-S",
        "dataset": "eicu_semisynth",
        "seed": seed,
        "alpha": "",  # not applicable -- eicu uses natural partitioning, not Dirichlet
        "output_root": output_root,
        "final_result_dir": "",
        "implementation_status": "implemented",
        "run_status": "not_started",
        "preflight_required": "false",
        "preflight_status": "",
        "model": "lr",
        "federated_optimizer": "FedAvg",
        "client_optimizer": client_optimizer,
        "client_num_in_total": n_clients,
        "client_num_per_round": n_clients,  # full participation, per protocol
        "comm_round": comm_round,
        "epochs": 1,
        "batch_size": max(min(n_clients, 32), 1),
        "partition_method": "natural",
        "partition_alpha": 0.0,
        "data_cache_dir": "data",
        "critic_multiplier": BASE_CRITIC_MULTIPLIER,
        "gradient_clip_norm": BASE_GRADIENT_CLIP_NORM,
        "simple_model_selection_epochs": 100,
        "f_history_model_selection_epochs": 60,
        "model_selection_batch_size": 200,
        "using_gpu": "false",  # these runs complete in well under a second on CPU
        "gpu_id": 0,
        "notes": "",
        "scenario_name": f"{g0}_seed{seed}",
        "objective_mode": "paper_aligned",
        "aggregation_weighting": "uniform_clients",
        "input_dim_g": int(metadata["n_features_x"]),
        "input_dim_f": int(metadata["n_features_z"]),
        "g0": g0,
        "skip_model_selection": "true",
    }


def generate_tuning(scenario_dir, output_root):
    rows = []
    for g0 in G0_VARIANTS:
        metadata = load_scenario_metadata(scenario_dir, g0, TUNING_SEED)
        for method in METHODS:
            for lr_mult in TUNING_LR_MULTIPLIERS:
                for server_lr in TUNING_SERVER_LRS:
                    run_id = (
                        f"tuning_{g0}_{method}_lr{lr_mult}_slr{server_lr}".replace(".", "p")
                    )
                    row = base_row(
                        run_id, "tuning", g0, TUNING_SEED, method, metadata,
                        TUNING_COMM_ROUND, output_root,
                    )
                    row["learning_rate"] = BASE_LEARNING_RATE * lr_mult
                    row["learning_rate_status"] = "tuning_candidate"
                    row["weight_decay"] = BASE_WEIGHT_DECAY
                    row["server_learning_rate"] = server_lr
                    row["notes"] = f"lr_multiplier={lr_mult};server_lr_candidate={server_lr}"
                    rows.append(row)
    return rows


def generate_confirmatory(scenario_dir, output_root, selected_hyperparameters):
    """``selected_hyperparameters``: {(g0, method): {"learning_rate":, "server_learning_rate":}}
    from scripts/select_eicu_study_a_tuning.py's frozen selection.
    """
    rows = []
    for g0 in G0_VARIANTS:
        for method in METHODS:
            key = selection_key(g0, method)
            if key not in selected_hyperparameters:
                raise KeyError(f"no frozen hyperparameters for {key}; run tuning selection first")
            selected = selected_hyperparameters[key]
            for seed in CONFIRMATORY_SEEDS:
                metadata = load_scenario_metadata(scenario_dir, g0, seed)
                run_id = f"confirmatory_{g0}_{method}_seed{seed}"
                row = base_row(
                    run_id, "confirmatory", g0, seed, method, metadata,
                    CONFIRMATORY_COMM_ROUND, output_root,
                )
                row["learning_rate"] = selected["learning_rate"]
                row["learning_rate_status"] = "frozen_from_tuning"
                row["weight_decay"] = BASE_WEIGHT_DECAY
                row["server_learning_rate"] = selected["server_learning_rate"]
                rows.append(row)
    return rows


def generate_ablation(scenario_dir, output_root, selected_hyperparameters):
    """Linear g0 only. The uniform_clients arm already exists as the linear
    rows of the confirmatory stage, so only the sample_size arm is generated
    here -- 5 seeds x 2 methods = 10 new rows, not 20.
    """
    rows = []
    g0 = "linear"
    for method in METHODS:
        key = selection_key(g0, method)
        if key not in selected_hyperparameters:
            raise KeyError(f"no frozen hyperparameters for {key}; run tuning selection first")
        selected = selected_hyperparameters[key]
        for seed in CONFIRMATORY_SEEDS:
            metadata = load_scenario_metadata(scenario_dir, g0, seed)
            run_id = f"ablation_{g0}_{method}_sample_size_seed{seed}"
            row = base_row(
                run_id, "ablation", g0, seed, method, metadata,
                CONFIRMATORY_COMM_ROUND, output_root,
            )
            row["learning_rate"] = selected["learning_rate"]
            row["learning_rate_status"] = "frozen_from_tuning"
            row["weight_decay"] = BASE_WEIGHT_DECAY
            row["server_learning_rate"] = selected["server_learning_rate"]
            row["aggregation_weighting"] = "sample_size"
            row["notes"] = "aggregation ablation: sample_size arm"
            rows.append(row)
    return rows


def write_manifest(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})

    json_path = os.path.splitext(out_csv)[0] + ".json"
    with open(json_path, "w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_csv, json_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("tuning", "confirmatory", "ablation"))
    parser.add_argument(
        "--scenario-dir",
        default=os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example", "data", "eicu_semisynth"),
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(REPO_ROOT, "results", "eicu_study_a"),
        help="passed through as each row's output_root; still pass --output-root explicitly to run_manifest.py",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--selected-hyperparameters",
        default=None,
        help="JSON from select_eicu_study_a_tuning.py; required for confirmatory/ablation",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    selected = {}
    if args.selected_hyperparameters:
        with open(args.selected_hyperparameters) as handle:
            selected = json.load(handle)

    if args.stage == "tuning":
        rows = generate_tuning(args.scenario_dir, args.output_root)
    elif args.stage == "confirmatory":
        if not selected:
            raise ValueError("--selected-hyperparameters is required for --stage confirmatory")
        rows = generate_confirmatory(args.scenario_dir, args.output_root, selected)
    else:
        if not selected:
            raise ValueError("--selected-hyperparameters is required for --stage ablation")
        rows = generate_ablation(args.scenario_dir, args.output_root, selected)

    csv_path, json_path = write_manifest(rows, args.out)
    print(f"stage={args.stage} rows={len(rows)}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
