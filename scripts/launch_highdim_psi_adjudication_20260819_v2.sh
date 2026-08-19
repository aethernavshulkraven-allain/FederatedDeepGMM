#!/usr/bin/env bash
# Threshold-gated launcher for the corrected Psi-adjudication/confirmation
# stage (psi_adjudication_20260819_v2/). Per the 2026-08-19 review packet
# go/no-go: run only after the packet has been approved, only if the
# broker reports at least cost*1.15 GPU-h remaining, signal confirmation
# before _x adjudication.
set -euo pipefail
repo_root="/home/arnav22103/FederatedDeepGMM"
python_bin="/home/arnav22103/miniconda3/envs/fedgmm/bin/python"
campaign="experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"
result_root="results/highdim_psi_adjudication_20260819_v2"
required_h="45.87"  # 39.89 GPU-h recalculated cost x 1.15 safety margin

cd "$repo_root"
export WANDB_MODE=disabled

remaining=$(gpurun --status | grep -oP 'remaining:\s*\K-?[0-9.]+')
echo "Broker reports ${remaining} GPU-h remaining; need >= ${required_h} GPU-h."
awk -v have="$remaining" -v need="$required_h" 'BEGIN { exit !(have+0 >= need+0) }' || {
  echo "REFUSING TO LAUNCH: remaining quota (${remaining} GPU-h) is below the required ${required_h} GPU-h threshold."
  exit 1
}

echo "=== Stage 1/2: signal confirmation (8 cells, 42 rows) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_signal_manifest.csv" \
  --config-dir "$campaign/generated_configs_signal" \
  --output-root "$result_root" \
  --gpu-ids 0,1 --max-parallel 2 \
  --resume-skip-completed --keep-going \
  --results-json "$campaign/adjudication_signal_launcher_results.json"

echo "=== Stage 2/2: _x adjudication (4 cells, 21 rows) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_x_manifest.csv" \
  --config-dir "$campaign/generated_configs_x" \
  --output-root "$result_root" \
  --gpu-ids 0,1 --max-parallel 2 \
  --resume-skip-completed --keep-going \
  --results-json "$campaign/adjudication_x_launcher_results.json"

echo "=== DONE ==="
