#!/usr/bin/env bash
# Payload for the corrected Psi-adjudication/confirmation stage
# (psi_adjudication_20260819_v2/). Meant to be run INSIDE a gpurun job
# (invoked by run_highdim_psi_adjudication_when_idle_20260819.sh, which
# requests exactly as many GPUs as are currently idle). Takes the
# process-local device-index list as $1 -- e.g. "0" for a single-GPU job,
# "0,1" for two. This is NOT the physical GPU id: gpurun/main.py preset
# CUDA_VISIBLE_DEVICES to whichever physical GPU(s) the broker granted, so
# inside the job the visible devices are always renumbered starting at 0
# regardless of which physical GPU was actually assigned.
#
# Progress safety: run_manifest.py writes --results-json after every job
# resolves (not just once at the end), so a hard kill (e.g. broker
# preemption, which gives a 60s warning per the broker's PREEMPTIBLE
# class) mid-manifest still leaves an accurate results file for everything
# that finished before the kill. Per-run training artifacts (metrics.json,
# checkpoints) are always written directly by the training process on
# completion, independent of this script or its parent process staying
# alive. --resume-skip-completed --keep-going means re-running this exact
# script after any interruption picks up only the incomplete/not-yet-
# started rows.
#
# Order (per the approved review packet): signal confirmation before _x
# adjudication.
set -uo pipefail
repo_root="/home/arnav22103/FederatedDeepGMM"
python_bin="/home/arnav22103/miniconda3/envs/fedgmm/bin/python"
campaign="experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"
result_root="results/highdim_psi_adjudication_20260819_v2"

gpu_ids="${1:?usage: launch_highdim_psi_adjudication_20260819_v2.sh <device-index-list, e.g. 0 or 0,1>}"
max_parallel=$(( $(echo "$gpu_ids" | tr ',' '\n' | wc -l) ))

cd "$repo_root"
export WANDB_MODE=disabled

echo "Using device index list: ${gpu_ids} (max_parallel=${max_parallel})"

echo "=== Stage 1/2: signal confirmation (8 cells, 42 rows) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_signal_manifest.csv" \
  --config-dir "$campaign/generated_configs_signal" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$max_parallel" \
  --resume-skip-completed --keep-going \
  --results-json "$campaign/adjudication_signal_launcher_results.json"

echo "=== Stage 2/2: _x adjudication (4 cells, 21 rows) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_x_manifest.csv" \
  --config-dir "$campaign/generated_configs_x" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$max_parallel" \
  --resume-skip-completed --keep-going \
  --results-json "$campaign/adjudication_x_launcher_results.json"

echo "=== DONE ==="
