# A6000 shard runbook — finals, alpha = 1.0

Handoff for running one self-contained block of the high-dim deterministic
finals campaign on a second machine (an A6000) while the primary H100 server
runs the rest of the campaign in parallel. Written so anyone can pick this up
without prior context on the project.

- **Repo**: `aethernavshulkraven-allain/FederatedDeepGMM`
- **Branch**: `experimentsrerun`
- **Commit this was written against**: `c79dba49`
- **Your block**: finals, `alpha = 1.0` — 60 runs, 1 GPU, 500 rounds/run
- **Estimated wall-clock**: 3–5 days continuous (see [Timing](#timing) — this is scaled from H100 numbers, not measured on an A6000)

---

## Start setting up now — but you cannot launch yet

**You can do today:** clone the repository, build the Python environment, and
receive the dataset files. All of that is independent of the server.

**You are waiting on one thing:** the file your launch command reads,
`finals_launch_manifest.csv`, does not exist yet. It lists your exact 60 runs
and their settings, and it can only be generated after two stages finish on
the server — roughly 8 hours of GPU work. It will arrive as a small
follow-up push to this same branch, so a `git pull` is all you need to pick
it up.

Work through [Before you start](#before-you-start) below in the meantime.

---

## What your block is

The campaign's final result is a grid of 180 training runs: 6 datasets × 2
methods × 3 heterogeneity settings (`alpha`) × 5 random seeds. Your 60 runs
are **the entire alpha = 1.0 slice** — all 5 seeds, all 12 dataset/method
combinations.

That slice was chosen because it is the only one that stands completely on
its own. The other two alpha settings each have to be combined with runs the
server already produced weeks ago, so splitting *those* across two machines
would mix hardware inside a single comparison. The alpha = 1.0 slice reuses
nothing, so it can be produced anywhere without disturbing the rest of the
grid.

### Your 60 runs, by dataset (10 runs each: 2 methods × 5 seeds)

| Dataset | Runs | Min/run on H100 (measured) | Your block on H100-equivalent |
|---|---:|---:|---:|
| femnist_x | 10 | ~21 | 3.5 h |
| femnist_z | 10 | 21.0 | 3.5 h |
| femnist_xz | 10 | 40.7 | 6.8 h |
| cifar10_x | 10 | 37.8 | 6.3 h |
| cifar10_z | 10 | 37.9 | 6.3 h |
| cifar10_xz | 10 | 74.9 | 12.5 h |
| **Total** | **60** | | **38.9 h** |

Those minutes are measured medians from 65 completed runs on the server's
H100s, not estimates — except `femnist_x`, where no run has finished yet and
21 min is inferred from its sibling cells.

<a name="timing"></a>
An A6000 is roughly 2–3× slower than an H100 on this workload, so budget
somewhere around 80–120 wall-clock hours. **Please report the wall-clock
time of your first completed run** — it is the only real data point on A6000
throughput, and it lets us rebalance the split if the gap is very different
from expected.

---

## Why you are waiting

Your 60 runs need to know which learning rate and critic multiplier to use,
and those are not chosen by hand — they are the winners of an earlier
tournament stage still running on the server. Two things have to finish
before your manifest can be written:

| Stage | Runs on | What it produces | Blocks you? |
|---|---|---|---|
| V4 adjudication | Server | The winning settings for each of the 12 dataset/method combinations | **Yes** — your runs use these exact settings |
| Stability | Server | A pass/fail check on each winner under the hardest data split | **Yes**, indirectly — the script that writes your manifest refuses to run without it |
| Finals prep | Server, no GPU | `finals_launch_manifest.csv` — your 60 rows, plus 72 for the server | This *is* what you are waiting for |
| Finals — your block | **Your A6000** | 60 completed runs at alpha = 1.0 | — |

Once you start, the two machines are completely independent for several
days — the server works through its 72 runs while you work through your 60,
and neither waits on the other. The only point they meet again is when your
results come back for merging.

If stability fails for any cell, a small retune fallback kicks in — but that
only touches the alpha = 0.1 slice, which lives on the server. Your 60 rows
use the original V4 winner's settings regardless of what happens with
stability, so your block never changes and never needs to be redone.

---

## Before you start

- **The code.** Clone the fork and check out the branch:

  ```bash
  git clone https://github.com/aethernavshulkraven-allain/FederatedDeepGMM.git
  cd FederatedDeepGMM
  git checkout experimentsrerun   # currently at commit c79dba49
  ```

- **The datasets, sent to you separately.** Six `main.npz` files, ~2.4 GB
  total, which do **not** come with the clone — they are too large for
  GitHub, which rejects any file over 100 MB. They must land at exactly
  `fedgmm/sp_decentralized_mnist_lr_example/data/<name>/main.npz`, one
  directory per dataset: `femnist_x`, `femnist_z`, `femnist_xz`, `cifar10_x`,
  `cifar10_z`, `cifar10_xz`. Step 2 below checks all six against known
  hashes, so a wrong path or a truncated transfer gets caught immediately
  rather than hours into training.

- **Python environment.** A conda env matching the server's: Python 3.10.20,
  PyTorch 2.2.2+cu118, CUDA 11.8, cuDNN 8700, numpy 1.26.4. Version parity
  matters here — a different PyTorch build is a second difference on top of
  the different GPU.

- **Disk.** ~2.4 GB for the six dataset files, plus roughly 2.5 GB for the
  results your 60 runs will produce (each finished run directory averages
  ~41 MB, mostly checkpoints).

- **Uninterrupted GPU.** One GPU, held for several days. The job is
  resumable (see [Interruptions are safe](#if-it-gets-interrupted)), but
  fewer interruptions means a cleaner handback.

- **Nothing else on that GPU.** Run one job at a time
  (`--max-parallel 1`). Sharing the card with another workload will distort
  the timings we are asking you to report.

---

## Running it

### 1. Set the environment

From the repository root, with your conda env active:

```bash
export WANDB_MODE=disabled
cd /path/to/FederatedDeepGMM
```

### 2. Verify the code and data arrived intact

This checks every execution-critical source file, the six dataset files, and
the frozen protocol documents against recorded hashes. If it fails, stop and
tell us — something was corrupted or changed in transfer, and nothing you
run afterwards would be usable.

```bash
python scripts/verify_protocol_hashes.py \
  --hashes experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/generated_artifact_hashes.json
```

### 3. Dry-run first and confirm the count is exactly 60

This selects your rows and generates configs without training anything. It
is the single most important check in this document — see the warning
directly below it.

```bash
python scripts/run_manifest.py \
  --manifest experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/finals_launch_manifest.csv \
  --config-dir experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/generated_configs_a6000 \
  --output-root results/highdim_deterministic_finals_post_bn_20260826 \
  --only alpha=1 \
  --gpu-ids 0 --max-parallel 1 \
  --dry-run \
  --results-json experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/finals_launcher_results_a6000.json
```

The last line of output must read:

```json
{"dry_run": true, "launchable": 60, "shown": 60, "skipped_unlaunchable": 0}
```

> **The one mistake that fails silently.** It must be `--only alpha=1`. **Not
> `alpha=1.0`.** The filter compares text, not numbers, and the manifest
> stores this value as the single character `1`. Writing `alpha=1.0` matches
> zero rows, prints no error, and exits successfully — you would watch it
> finish in a second and believe the work was already done. We confirmed
> this against the real launcher: `alpha=1` selects 60 rows, `alpha=1.0`
> selects 0.
>
> This is exactly what step 3 exists to catch. If the dry run says anything
> other than `"launchable": 60`, do not proceed — send us the output.

### 4. Launch for real

Same command, with `--dry-run` removed and three resume flags added. Run it
under `tmux`, `screen`, or `nohup` — it will run for days.

```bash
python scripts/run_manifest.py \
  --manifest experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/finals_launch_manifest.csv \
  --config-dir experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/generated_configs_a6000 \
  --output-root results/highdim_deterministic_finals_post_bn_20260826 \
  --only alpha=1 \
  --gpu-ids 0 --max-parallel 1 \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/finals_launcher_results_a6000.json
```

---

## While it runs

### Checking progress

Each finished run drops a `metrics.json` into its own directory. Count them:

```bash
find results/highdim_deterministic_finals_post_bn_20260826 -name metrics.json | wc -l
```

Run directories are laid out as
`results/highdim_deterministic_finals_post_bn_20260826/<dataset>/<method>/seed_<n>/<run_id>`,
and every `run_id` in your block contains the token `alpha1`.

<a name="if-it-gets-interrupted"></a>
### Interruptions are safe

If the job dies, the machine reboots, or you need the GPU back for a while —
just re-run the exact same step 4 command when you are ready.
`--resume-skip-completed` validates and skips every run that already
finished, and `--overwrite-incomplete` cleanly restarts any run that was cut
off mid-training. Do not delete partial run directories by hand; the
launcher handles them correctly and deleting them only loses the check that
they were incomplete for an innocent reason.

### If a run fails

`--keep-going` means one failed run will not stop the other 59. Let it
continue to the end rather than intervening, then send us the results file —
it records the status of every run, and failures are diagnosable from it. A
run that fails for a real reason is a finding, not a mistake to hide.

---

## Sending results back

When the launcher finishes, three things need to come back. Please do not
prune or reorganise the directory tree — the paths are how runs get matched
to the manifest on our side.

- **The results tree** — `results/highdim_deterministic_finals_post_bn_20260826/`
  in full, roughly 2.5 GB.
- **The results file** — `finals_launcher_results_a6000.json`, in the
  campaign directory.
- **The attempt log** — `finals_launcher_results_a6000_attempts.jsonl`,
  written alongside it.

On our side we re-run the launcher over the full manifest in resume mode.
That re-validates every run you produced against the configuration the
manifest says it should have, so a transfer problem or a mismatched config
gets caught rather than absorbed. Nothing is taken on trust, which is also
why there is no need for you to verify anything beyond step 3.

---

## Please don't

- **Edit anything under `scripts/` or `fedgmm/`.** Both are covered by the
  hash check in step 2. A change there, however small or however obviously
  an improvement, invalidates the whole block. If something looks broken,
  tell us instead of fixing it.
- **Run any alpha other than 1.** The `0.1` and `0.5` rows in that same
  manifest belong to the server and must run there. Running them here would
  produce duplicate work that we would have to throw away.
- **Change the flags in step 4**, especially `--gpu-ids`, `--max-parallel`,
  or the two output paths. The separate `_a6000` config directory and
  results file are what keep your shard from colliding with the server's
  when the two are merged.

---

*Block: finals alpha=1.0, 60 of 180 trajectories · Campaign:
`experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826/`
· Server side runs V4 X, stability, and finals alpha=0.1 / 0.5 in parallel ·
Timings measured 2026-08-31 from 65 completed H100 runs and will drift.*
