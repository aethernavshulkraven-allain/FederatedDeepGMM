Training is now launched through CSV manifests using scripts/run_manifest.py. The launcher generates temporary YAML internally, but you do not edit or
  run YAML files yourself.

  ## 1. Enter the repository and activate the environment

  cd /mnt/disk1/geetika/FedDeepGMM
  conda activate dl

  Confirm CUDA:

  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
  nvidia-smi

  Choose only idle GPUs.

  ## 2. Understand the three execution modes

  ### Serial SP

  Clients train one after another:

  client_execution_mode=sp

  Recommended for deterministic full-batch FEMNIST/CIFAR experiments because it was faster in our benchmarks.

  ### Multiple processes on one GPU

  Several clients train concurrently on the same GPU:

  client_execution_mode=multiprocessingsinglegpu
  multiprocessingsinglegpu_num_workers=2
  multiprocessingsinglegpu_gpu_id=0

  Use this mainly for workloads where minibatch client training does not already saturate the GPU.

  ### Multiple processes across GPUs

  One client worker is assigned to each logical GPU:

  client_execution_mode=multi_gpu_processes
  multiprocessing_num_workers=2
  multiprocessing_gpu_ids="0,1"

  This reduces the latency of one experiment but occupies several GPUs.

  In every mode, the communication backend remains SP. Multiprocessing only parallelizes client-local updates.

  ## 3. Start with the reviewed smoke manifest

  The checked-in smoke manifest is:

  experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv

  It includes FedGDA, FedOGDA, FedEG, and FedZO-EG variants.

  Preview one row without training:

  CUDA_VISIBLE_DEVICES=2 python scripts/run_manifest.py \
    --manifest experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv \
    --config-dir /tmp/feddeepgmm_smoke_configs \
    --output-root results/multiprocessing_single_gpu_smoke \
    --gpu-ids 0 \
    --max-parallel 1 \
    --only method=fedgda_s \
    --dry-run

  Important GPU mapping:

  Physical GPU 2 → CUDA_VISIBLE_DEVICES=2 → logical GPU 0

  Therefore, the manifest and --gpu-ids use 0, not 2.

  ## 4. Launch the smoke experiment

  Remove --dry-run:

  CUDA_VISIBLE_DEVICES=2 python scripts/run_manifest.py \
    --manifest experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv \
    --config-dir /tmp/feddeepgmm_smoke_configs \
    --output-root results/multiprocessing_single_gpu_smoke \
    --gpu-ids 0 \
    --max-parallel 1 \
    --only method=fedgda_s \
    --resume-skip-completed \
    --results-json experiments/multiprocessing_single_gpu/smoke_launcher_results.json

  Possible method filters are:

  fedgda_d
  fedgda_s
  fedogda_d
  fedogda_s
  fed_eg_d
  fed_eg_s
  fed_zo_eg_d
  fed_zo_eg_s

  Omit --only method=... to execute every row in the manifest sequentially.

  ## 5. Create a real experiment manifest

  Copy an appropriate reviewed manifest:

  cp experiments/multiprocessing_single_gpu/reviewed_smoke_manifest.csv \
     experiments/my_campaign/manifest.csv

  Then review each row. The important columns include:

  run_id
  method
  dataset
  seed
  client_num_in_total
  client_num_per_round
  comm_round
  epochs
  batch_size
  learning_rate
  critic_multiplier
  server_learning_rate
  gradient_clip_norm
  objective_lambda_1
  aggregation_weighting
  client_execution_mode
  multiprocessingsinglegpu_num_workers
  multiprocessingsinglegpu_gpu_id
  auxiliary_regression

  For a four-worker single-GPU run:

  client_execution_mode,multiprocessingsinglegpu_num_workers,multiprocessingsinglegpu_gpu_id,auxiliary_regression
  multiprocessingsinglegpu,4,0,False

  Keep auxiliary_regression=False unless a legacy protocol explicitly requires it.

  For stochastic training, use a positive minibatch size, such as:

  batch_size=256

  For deterministic full-batch training:

  batch_size=0

  ## 6. Preview the real campaign

  Always dry-run first:

  CUDA_VISIBLE_DEVICES=2 python scripts/run_manifest.py \
    --manifest experiments/my_campaign/manifest.csv \
    --config-dir /tmp/my_campaign_configs \
    --output-root results/my_campaign \
    --gpu-ids 0 \
    --max-parallel 1 \
    --dry-run

  Check:

  - Correct number of selected rows
  - Correct dataset and method
  - Correct output directories
  - Correct logical GPU
  - No missing hyperparameters

  ## 7. Run one selected experiment first

  Filter using manifest columns:

  CUDA_VISIBLE_DEVICES=2 python scripts/run_manifest.py \
    --manifest experiments/my_campaign/manifest.csv \
    --config-dir /tmp/my_campaign_configs \
    --output-root results/my_campaign \
    --gpu-ids 0 \
    --max-parallel 1 \
    --only dataset=femnist_x \
    --only method=fedgda_s \
    --only seed=0 \
    --limit 1 \
    --results-json experiments/my_campaign/launcher_results.json

  Once this completes successfully, remove --limit 1 to run the remaining selected rows.

  ## 8. Run one experiment across multiple GPUs

  Suppose physical GPUs 1 and 3 are idle. The shell exposes them as logical GPUs 0 and 1:

  CUDA_VISIBLE_DEVICES=1,3

  The manifest row should contain:

  client_execution_mode,multiprocessing_num_workers,multiprocessing_gpu_ids
  multi_gpu_processes,2,"0,1"

  Launch it with:

  CUDA_VISIBLE_DEVICES=1,3 python scripts/run_manifest.py \
    --manifest experiments/my_campaign/multi_gpu_manifest.csv \
    --config-dir /tmp/my_campaign_multigpu_configs \
    --output-root results/my_campaign_multigpu \
    --gpu-ids 0 \
    --max-parallel 1 \
    --limit 1 \
    --results-json experiments/my_campaign/multigpu_results.json

  Here:

  - --gpu-ids 0 assigns the coordinator to logical GPU 0.
  - multiprocessing_gpu_ids="0,1" assigns client workers across both visible GPUs.
  - --max-parallel 1 prevents the launcher from starting multiple complete experiments simultaneously.

  ## 9. Run separate experiments on separate GPUs

  For a large deterministic campaign, this is usually the best approach.

  Terminal 1:

  CUDA_VISIBLE_DEVICES=0 python scripts/run_manifest.py \
    --manifest experiments/my_campaign/manifest_gpu0.csv \
    --config-dir /tmp/campaign_gpu0_configs \
    --output-root results/my_campaign \
    --gpu-ids 0 \
    --max-parallel 1 \
    --results-json experiments/my_campaign/gpu0_results.json

  Terminal 2:

  CUDA_VISIBLE_DEVICES=1 python scripts/run_manifest.py \
    --manifest experiments/my_campaign/manifest_gpu1.csv \
    --config-dir /tmp/campaign_gpu1_configs \
    --output-root results/my_campaign \
    --gpu-ids 0 \
    --max-parallel 1 \
    --results-json experiments/my_campaign/gpu1_results.json

  Each process sees only one physical GPU, remapped to logical GPU 0.

  Do not place the same manifest rows in both files; that would run duplicate experiments and cause output conflicts.

  ## 10. Understand the two kinds of parallelism

  These settings control different things:

  --max-parallel

  Number of complete experiments the launcher runs concurrently.

  multiprocessingsinglegpu_num_workers

  Number of client updates running concurrently inside one experiment on one GPU.

  multiprocessing_num_workers

  Maximum client workers inside one multi-GPU experiment.

  For one single-GPU multiprocessing experiment, normally use:

  --max-parallel 1
  multiprocessingsinglegpu_num_workers 2 or 4

  ## 11. Monitor training

  Watch GPU use:

  watch -n 2 nvidia-smi

  Inspect the launcher result:

  cat experiments/my_campaign/launcher_results.json

  Inspect a completed run:

  find results/my_campaign -name metrics.json

  Each successful run should contain:

  effective_config.json
  metrics.json
  mse_by_round.csv
  predictions.npz
  checkpoints/best_validation.pt
  checkpoints/best_moment_violation.pt
  checkpoints/final.pt

  A successful launcher row reports:

  {
    "status": "passed"
  }

  ## 12. Resume a campaign safely

  Use:

  --resume-skip-completed

  Example:

  CUDA_VISIBLE_DEVICES=2 python scripts/run_manifest.py \
    --manifest experiments/my_campaign/manifest.csv \
    --config-dir /tmp/my_campaign_configs \
    --output-root results/my_campaign \
    --gpu-ids 0 \
    --max-parallel 1 \
    --resume-skip-completed \
    --results-json experiments/my_campaign/launcher_results.json

  Completed runs are validated and skipped. Incomplete output directories cause a failure unless you explicitly pass --overwrite-incomplete. Review those
  directories before using that option.

  ## 13. Verify deterministic equivalence

  For deterministic methods, matching Test MSE is insufficient. Compare:

  - best_validation.pt
  - best_moment_violation.pt
  - final.pt
  - Every tensor in g_state_dict
  - Every tensor in f_state_dict

  The required condition is:

  torch.equal(sp_tensor, mp_tensor)

  Our measured SP, MP2, and MP4 benchmarks passed this exact check for FEMNIST-X, FEMNIST-XZ, and CIFAR10-X.

  ## Practical recommendation

  Use:

  - SP plus separate experiments on separate GPUs for deterministic full-batch campaigns.
  - Same-GPU MP2/MP4 only after benchmarking a stochastic/minibatch workload.
  - Multi-GPU client processing when minimizing one experiment’s latency is more important than total GPU efficiency.

  The complete repository tutorial is Multiprocess_instructions.md.