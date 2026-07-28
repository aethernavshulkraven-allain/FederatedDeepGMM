# Fixed-abs high-dimensional runbook

The protocol is fixed to `g(x) = |x|`, `alpha = 0.5`, six image scenarios, four federated methods, and final seeds 0–4.

## Step 6: run the 96 validation-tuning candidates

```bash
gpurun -g 2 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_manifest.py \
  --manifest experiments/rerun_protocol_v1_real_images_abs_alpha0p5/tuning/manifest.csv \
  --config-dir experiments/rerun_protocol_v1_real_images_abs_alpha0p5/tuning/generated_configs \
  --output-root results/rerun_protocol_v1_real_images_abs_alpha0p5_tuning \
  --gpu-ids 0,1 --max-parallel 2 --resume-skip-completed \
  --results-json experiments/rerun_protocol_v1_real_images_abs_alpha0p5/tuning/run_results.json
```

## Step 7: select configs using validation only

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_real_image_abs_runs.py \
  --manifest experiments/rerun_protocol_v1_real_images_abs_alpha0p5/tuning/manifest.csv \
  --output experiments/rerun_protocol_v1_real_images_abs_alpha0p5/tuning/artifact_validation.json
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/analyze_real_image_abs_tuning.py
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/materialize_real_image_abs_final_manifest.py
```

The analyzer excludes diverged candidates and ranks by `best_validation_mse`, then last-50 validation-MSE standard deviation, then the final-versus-best validation gap. It does not read test MSE until all 24 choices have been fixed.

## Step 8: run the 120 final jobs

```bash
gpurun -g 2 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_manifest.py \
  --manifest experiments/rerun_protocol_v1_real_images_abs_alpha0p5/final_manifest.csv \
  --config-dir experiments/rerun_protocol_v1_real_images_abs_alpha0p5/generated_configs \
  --output-root results/rerun_protocol_v1_real_images_abs_alpha0p5 \
  --gpu-ids 0,1 --max-parallel 2 --resume-skip-completed \
  --results-json experiments/rerun_protocol_v1_real_images_abs_alpha0p5/run_results.json
```

## Step 9: validate and report

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_real_image_abs_runs.py
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/analyze_real_image_abs_final.py
```

Final reporting uses `test_mse_at_best_validation` only after validation-selected hyperparameters and checkpoints are fixed.
