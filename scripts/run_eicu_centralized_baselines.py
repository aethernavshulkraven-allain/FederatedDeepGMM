"""Resumable runner for the Study A centralized baselines (GDA, OAdam).

3 g0 variants x 5 seeds x 2 methods = 30 runs. Centralized runs are a
different code path from the federated manifest (``run_centralized_lowdim.py``
pools all clients, no partitioning), so they are not launched through
``run_manifest.py`` -- this is a thin, dedicated, resumable loop instead.

Resume semantics match the rest of the repo: a run is skipped if its
``metrics.json`` already exists (unless ``--overwrite``), and one run's
failure does not stop the rest (``--stop-on-failure`` to opt into that).

Usage:
    python scripts/run_eicu_centralized_baselines.py \\
        --scenario-dir fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth \\
        --output-root results/eicu_study_a/centralized
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENARIO_DIR = os.path.join(
    REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example", "data", "eicu_semisynth"
)

G0_VARIANTS = ("linear", "interaction", "mlp")
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("gda", "oadam")

# Matches the federated protocol's frozen defaults; centralized runs use the
# same objective/learning-rate scale as the federated smoke config so the
# comparison is apples-to-apples.
DEFAULT_G_LR = 0.001
DEFAULT_F_LR = 0.01
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_ITERATIONS = 500


def run_dir_for(output_root, g0, seed, method):
    return os.path.join(output_root, f"{g0}_seed{seed}", method, f"seed_{seed}")


def already_complete(run_dir):
    return os.path.exists(os.path.join(run_dir, "metrics.json"))


def build_command(python_bin, g0, seed, method, scenario_dir, output_root, args):
    run_dir = run_dir_for(output_root, g0, seed, method)
    batch_size = 0 if method == "gda" else 256
    cmd = [
        python_bin,
        os.path.join(REPO_ROOT, "scripts", "run_centralized_lowdim.py"),
        "--dataset", "eicu_semisynth",
        "--scenario-name", f"{g0}_seed{seed}",
        "--method", method,
        "--seed", str(seed),
        "--output-dir", run_dir,
        "--iterations", str(args.iterations),
        "--batch-size", str(batch_size),
        "--g-lr", str(args.g_lr),
        "--f-lr", str(args.f_lr),
        "--weight-decay", str(args.weight_decay),
        "--data-dir", os.path.dirname(scenario_dir),
        "--run-id", f"{g0}_{method}_seed{seed}",
        "--no-cuda",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd, run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario-dir", default=DEFAULT_SCENARIO_DIR)
    parser.add_argument(
        "--output-root", default=os.path.join(REPO_ROOT, "results", "eicu_study_a", "centralized")
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--g-lr", type=float, default=DEFAULT_G_LR)
    parser.add_argument("--f-lr", type=float, default=DEFAULT_F_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    jobs = [(g0, seed, method) for g0 in G0_VARIANTS for seed in SEEDS for method in METHODS]
    results = {"passed": [], "failed": [], "skipped_completed": []}

    for g0, seed, method in jobs:
        cmd, run_dir = build_command(args.python, g0, seed, method, args.scenario_dir, args.output_root, args)
        run_id = f"{g0}_{method}_seed{seed}"

        if not args.overwrite and already_complete(run_dir):
            print(f"SKIP  {run_id} (already complete)")
            results["skipped_completed"].append(run_id)
            continue

        print(f"START {run_id}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        completed = subprocess.run(cmd, cwd=REPO_ROOT)
        if completed.returncode == 0:
            print(f"PASS  {run_id}")
            results["passed"].append(run_id)
        else:
            print(f"FAIL  {run_id} returncode={completed.returncode}")
            results["failed"].append(run_id)
            if args.stop_on_failure:
                break

    summary = {
        "total_jobs": len(jobs),
        "passed": len(results["passed"]),
        "failed": len(results["failed"]),
        "skipped_completed": len(results["skipped_completed"]),
        "failed_run_ids": results["failed"],
    }
    print(json.dumps(summary, indent=2))
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
