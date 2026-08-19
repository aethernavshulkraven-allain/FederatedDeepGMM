#!/usr/bin/env bash
# Checks how many GPUs are idle right now and, if any, requests exactly
# that many from the broker and launches the corrected Psi-adjudication
# stage on them -- regardless of weekly quota balance. Over-budget
# requests still go through: the broker downgrades them to PREEMPTIBLE
# (runs only on idle GPUs, evicted with a 60s warning if reclaimed) rather
# than refusing outright. Safe to re-run repeatedly: run_manifest.py
# resumes via --resume-skip-completed, skipping whatever already
# completed.
#
# Exits 0 with a message and does nothing if no GPU is currently idle --
# re-run later once one frees up.
set -uo pipefail
cd /home/arnav22103/FederatedDeepGMM

idle_gpus=$(gpurun --status | awk '$2=="idle"{c++} END{print c+0}')
if [ "$idle_gpus" -lt 1 ]; then
  echo "No idle GPUs right now. Nothing launched -- re-run this script once one frees up."
  exit 0
fi

device_ids=$(seq 0 $((idle_gpus - 1)) | paste -sd, -)
echo "${idle_gpus} GPU(s) idle -- requesting ${idle_gpus} from the broker, device index list inside the job will be ${device_ids}."

gpurun -g "$idle_gpus" bash scripts/launch_highdim_psi_adjudication_20260819_v2.sh "$device_ids"
