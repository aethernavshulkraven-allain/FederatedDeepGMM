#!/usr/bin/env bash
set -euo pipefail
repo_root="/home/arnav22103/FederatedDeepGMM"
python_bin="/home/arnav22103/miniconda3/envs/fedgmm/bin/python"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
result_root="results/highdim_deterministic_screen_20260813"
cd "$repo_root"
export WANDB_MODE=disabled
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/screen_expand2_corrected_v1_manifest.csv" \
  --config-dir "$campaign/generated_configs_expand2corr" \
  --output-root "$result_root" \
  --gpu-ids 0,1 --max-parallel 2 \
  --resume-skip-completed --keep-going \
  --results-json "$campaign/screen_expand2_corrected_v1_launcher_results.json"
echo "=== DONE ==="
