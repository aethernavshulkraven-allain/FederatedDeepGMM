# FedEG / FedEG-Double: running tuning + finals on your own GPU

This covers the two new algorithms added in `a5513a57` ("Add double
extragradient federated method"):

- **`fed_eg`** -- extra-gradient at the server only. Clients train with
  plain SGD; the coordinator does the two-phase look-ahead/correction dance.
- **`fed_eg_double`** -- extra-gradient at *both* levels. Same server-side
  two-phase wrapper, but each client also runs true local extra-gradient
  (`optimizers/extragradient.py`) inside its own training step.

The plan is the same shape as the FedGDA-D/FedOGDA-D deterministic campaign
already running on the shared IIIT Delhi server: a cheap tuning **screen**
(150 rounds) to pick one learning-rate/critic-multiplier config per
scenario, then a full **finals** run (500 rounds x 3 alphas x 3 seeds) at
the frozen config. Everything below runs standalone on one GPU -- there's
no dependency on the shared server's `gpurun` broker.

## 0. One bug already found and fixed here, worth knowing about

`fed_eg_double`'s local optimizer crashed on the very first smoke test
during setup, inside PyTorch's model-selection warm-up phase
(`RuntimeError: ... modified by an inplace operation`). Root cause: the
f-objective's computation graph depends on g's forward output (see
`game_objectives/simple_moment_objective.py`), and the model-selection loop
calls `g_optimizer.step()` before `f_obj.backward()`. Every *other*
optimizer in this repo (`CustomSGD`, `SGDA`, `OGDA`) writes through
`.data`, which silently skips PyTorch's autograd safety check for exactly
this situation -- `ExtraGradient` was written the "correct" way
(tracked in-place ops), so it's the first one PyTorch actually stopped to
complain about.

Fixed in `optimizers/extragradient.py` (already in the branch you're
pulling) by writing through `.data` too, matching the rest of the repo's
custom optimizers. This is a narrow, self-contained fix -- it does not
touch the shared model-selection code that the live FedGDA-D/FedOGDA-D
campaign depends on. Worth flagging separately: this means f's gradient
during model selection (and the main training loop, for every method, not
just the new ones) is technically computed against g's *already-updated*
weights rather than the weights used in the forward pass. That's a
pre-existing property of the whole codebase, not something new -- just
raising it here since it only became visible while getting `fed_eg_double`
running.

## 1. Set up the repo and environment

```bash
git clone https://github.com/aethernavshulkraven-allain/FederatedDeepGMM.git
cd FederatedDeepGMM
git checkout experimentsrerun
git pull
```

Environment (conda shown, venv works the same way):

```bash
conda create -n fedgmm python=3.10 -y
conda activate fedgmm
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

Confirm the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

## 2. Calibrate real per-round cost on your A6000

The GPU-hour figures below (`~17.7 GPU-h` for the screen, `~81 GPU-h` for
finals) are measured on the shared server's H100s and **will not transfer
directly** -- different card, different FP64 throughput (this whole
protocol trains in float64). Get your own numbers first with a short probe
before committing to the full run. Generate the screen manifest (step 3
below covers this for real; running it now just gives you rows to pick a
calibration probe from), then run one row at a handful of rounds and time it:

```bash
python scripts/prepare_highdim_fedeg_screen.py \
    --campaign-dir experiments/highdim_coauthor_protocol_v1/fedeg_screen

time python scripts/run_manifest.py \
    --manifest experiments/highdim_coauthor_protocol_v1/fedeg_screen/screen_manifest.csv \
    --config-dir /tmp/fedeg_calib_configs \
    --output-root results/_calibration \
    --gpu-ids 0 --max-parallel 1 \
    --only dataset=femnist_z --only method=fed_eg_double_d \
    --override-comm-round 6 \
    --limit 1
```

Repeat with `--only dataset=cifar10_x` for a CIFAR-10 data point too --
CIFAR costs noticeably more per round than FEMNIST on the shared server
(roughly 2-4x), and your card's ratio may differ. Note the wall-clock time
and back out seconds/round (subtract a rough setup-time estimate first --
the first round includes one-time model-selection warm-up). Use these
numbers to size how much of the screen/finals you can run per session.
`results/_calibration/` is just scratch output, safe to delete afterward.

## 3. Tuning screen (72 runs, 150 rounds each)

Generates 6 scenarios x 2 new methods x 3 learning rates x 2 critic
multipliers, seed 0, alpha 0.5, `auxiliary_regression: false`. Learning-rate
grid is reused from FedOGDA-D (`{0.001, 0.003, 0.01}`) for both new
methods, per the agreed plan.

```bash
cd /path/to/FederatedDeepGMM
python scripts/prepare_highdim_fedeg_screen.py \
    --campaign-dir experiments/highdim_coauthor_protocol_v1/fedeg_screen
```

This writes `experiments/highdim_coauthor_protocol_v1/fedeg_screen/screen_manifest.csv`
(72 rows) and a `setup_summary.json`.

**Preview before spending GPU time:**

```bash
python scripts/run_manifest.py \
    --manifest experiments/highdim_coauthor_protocol_v1/fedeg_screen/screen_manifest.csv \
    --config-dir /tmp/fedeg_screen_configs \
    --output-root results/fedeg_screen \
    --gpu-ids 0 \
    --max-parallel 1 \
    --dry-run
```

Check: 72 rows shown, `launchable: 72`, correct dataset/method/GPU per row.

**Launch for real** (single GPU -- use `sp` execution, not
`multiprocessingsinglegpu`; same-GPU multiprocessing measured **slower**
than serial for this exact deterministic full-batch workload on the shared
server's H100s, see `Multiprocess_instructions.md` -- no reason to expect
your A6000 behaves differently for the same reason: full-batch training
already saturates one GPU, so extra worker processes just add overhead):

```bash
python scripts/run_manifest.py \
    --manifest experiments/highdim_coauthor_protocol_v1/fedeg_screen/screen_manifest.csv \
    --config-dir /tmp/fedeg_screen_configs \
    --output-root results/fedeg_screen \
    --gpu-ids 0 \
    --max-parallel 1 \
    --resume-skip-completed \
    --keep-going \
    --results-json experiments/highdim_coauthor_protocol_v1/fedeg_screen/launcher_results.json
```

`--resume-skip-completed` makes it safe to stop (Ctrl-C) and restart later
without re-running finished rows -- useful for splitting this across
multiple sessions once you know your real per-round cost from step 2.

## 4. Score the screen and pick winners

```bash
python scripts/score_highdim_screen_winners.py \
    --manifest experiments/highdim_coauthor_protocol_v1/fedeg_screen/screen_manifest.csv \
    --results-root results/fedeg_screen \
    --out experiments/highdim_coauthor_protocol_v1/fedeg_screen/winners.json
```

Prints one winner (lowest validation MSE) per (scenario, method) cell, and
flags any winner sitting at the edge of the tested lr/critic-multiplier
grid. A boundary flag isn't automatically wrong -- look at whether the
value trended *toward* that edge with a real improvement (worth adding one
more grid rung and re-running the scorer with both manifests via
`--manifest`), or was basically flat (safe to leave as-is). If you're not
sure which it is for a given cell, send us the printed table and we'll take
a look -- this exact judgment call is what we did by hand for the
FedGDA-D/FedOGDA-D screen and it's easy to eyeball wrong the first time.

## 5. Final evaluation (500 rounds x 3 alphas x 3 seeds per cell)

Once `winners.json` looks right:

```bash
python scripts/prepare_highdim_finals_from_winners.py \
    --winners experiments/highdim_coauthor_protocol_v1/fedeg_screen/winners.json \
    --campaign-dir experiments/highdim_coauthor_protocol_v1/fedeg_finals
```

This is the expensive part -- 12 cells x 3 alphas x 3 seeds = 108 runs at
500 rounds. Dry-run it first exactly as in step 3, then launch the same way
(swap the manifest/config-dir/output-root paths to the finals ones). Use
your step-2 calibration numbers to estimate real GPU-hours before starting,
and use `--resume-skip-completed` to spread it across multiple sessions if
needed -- there's no quota system on your end forcing you to finish in one
sitting.

## 6. Sending results back

`results/` is gitignored (run artifacts are large binary files, not meant
for git). Two things to send back separately:

1. **Small, trackable files** -- commit and push these (they're just CSV/JSON):
   `experiments/highdim_coauthor_protocol_v1/fedeg_screen/` and
   `experiments/highdim_coauthor_protocol_v1/fedeg_finals/` (manifests,
   `winners.json`, `setup_summary.json`, `launcher_results.json`). This is
   what lets us see exactly what grid you ran without needing the raw
   results.
2. **Run artifacts** (`results/fedeg_screen/`, `results/fedeg_finals/`,
   metrics/checkpoints/predictions per run) -- these need a different
   channel since they're not in git. Simplest: `tar` the two directories
   and `scp`/`rsync` them to the shared server, or drop them somewhere we
   already share files. Ping us for the exact destination path when you're
   ready to send.
