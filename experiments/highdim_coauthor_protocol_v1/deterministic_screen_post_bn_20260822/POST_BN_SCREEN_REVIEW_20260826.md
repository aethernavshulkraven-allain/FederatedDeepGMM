# Post-BatchNorm deterministic screen review — 2026-08-26

## Executive summary

The simplest way to understand the current state is:

- The original BatchNorm server-update bug appears fixed.
- The 104 runs that actually trained are numerically and structurally healthy.
- Four configurations never started federated training, probably because their
  configured learning rates were beyond the stable model-selection boundary.
- The main remaining problem is not the trained models. It is how the code
  intends to rank them using the validation Ψ statistic.
- V4 signal/X must therefore remain blocked until the Ψ question is settled.

> **Current decision:** Preserve the 104 completed trajectories. Do not launch
> V4. At minimum, fix the scorer and failure evidence, then rerun the exact
> 120-round diagnostic and four failed configurations. A complete 108-run screen
> rerun is conditional on the scientific definition of Ψ that the campaign
> intends to claim.

## 1. What experiment was running?

This was the corrected high-dimensional deterministic screening experiment.
The screen covered:

- FEMNIST and CIFAR-10;
- X, Z, and XZ variants;
- FedGDA-D and FedOGDA-D;
- different structural-model learning rates and critic learning-rate
  multipliers;
- seed 0;
- 150 communication rounds;
- all 10 clients participating in every round; and
- full-batch deterministic client training.

There were 108 configurations in total.

The screen was not intended to produce the final scientific table. Its purpose
was to identify the best one or two hyperparameter configurations in each
dataset/method cell. Those candidates would subsequently be tested for 500
rounds across multiple seeds.

## 2. Why was a corrected screen necessary?

The earlier high-dimensional runs had a server-side BatchNorm bug.

A model contains two relevant kinds of state:

- **Trainable parameters**, such as neural-network weights and biases.
- **Buffers**, such as BatchNorm `running_mean`, `running_var`, and
  `num_batches_tracked`.

FedGDA and FedOGDA apply server-side arithmetic such as interpolation and
optimistic extrapolation to trainable parameters. Conceptually, an optimistic
update can look like:

```text
new_parameter = current_parameter + update + optimistic_correction
```

That operation makes sense for trainable weights. It does not make sense for a
BatchNorm variance or integer counter.

The old implementation applied that arithmetic to the complete model
`state_dict`, including BatchNorm buffers. FedOGDA extrapolation could therefore
make `running_var` negative. Once BatchNorm receives a negative variance, its
normalization can produce NaNs. This explains the earlier isolated critic
failure around round 93.

The corrected policy now does the following:

- trainable parameters receive the intended FedGDA/FedOGDA server arithmetic;
- floating-point buffers are directly aggregated from client values;
- integer counters use a deterministic maximum; and
- negative or nonfinite BatchNorm variances are detected and logged.

Because candidates in the earlier campaign were selected from trajectories
generated under the buggy policy, those candidates were correctly retired and
the corrected screen was launched from scratch.

## 3. What happened in the corrected screen?

The launcher attempted all 108 configurations:

```text
108 configurations attempted
├── 104 completed all 150 rounds
└──   4 failed before round 0
```

This distinction matters. It would be inaccurate to say that 104 converged and
four diverged during federated training. The four failed configurations never
entered federated training.

The campaign accounting is preserved in:

- [`screen_launcher_results.json`](screen_launcher_results.json)
- [`screen_launcher_results_attempts.jsonl`](screen_launcher_results_attempts.jsonl)
- [`screen_manifest.csv`](screen_manifest.csv)

The attempt ledger contains exactly 108 job starts and 108 resolutions, with no
silent retries, deletion, or artifact replacement.

## 4. Are the 104 completed runs good?

Yes, from the BatchNorm, numerical, and artifact perspectives.

Every completed run has:

- exactly 150 rows in its round-level CSV;
- rounds numbered exactly `0..149`;
- finite model states and required metrics at every round;
- `finite=True` and `diverged=False` at every round;
- the corrected `direct_client_aggregate` buffer policy;
- valid best-validation, best-Ψ, and final checkpoints;
- finite predictions;
- test MSE values reproducible from the prediction artifacts;
- matching effective configurations and configuration checksums;
- ten participating clients in every round;
- aggregation weights summing to one; and
- no use of test MSE for model or checkpoint selection.

The audit loaded all 312 checkpoints—three per successful run—and examined
6,984 tensors. Every tensor was finite, every BatchNorm `running_var` was
nonnegative, and every integer BatchNorm counter was nonnegative.

The smallest observed BatchNorm running variances were approximately:

```text
structural model g: 1.80e-7
critic f:           2.09e-12
```

This is strong evidence that the original BatchNorm bug is absent from these
104 runs.

> **BatchNorm rerun verdict:** The 104 completed trajectories do not need to be
> rerun because of the original BatchNorm bug.

One wording caution remains: these runs are **complete and finite**, but not all
are necessarily **well converged**. Some remain finite while becoming much
worse late in training. That is a scientific stability result, not an artifact
failure.

## 5. What happened to the four failed configurations?

The four failures were:

| Dataset | Method | Learning rate | Critic multiplier |
|---|---|---:|---:|
| FEMNIST-Z | FedGDA-D | 0.1 | 5 |
| CIFAR10-X | FedGDA-D | 0.333333 | 10 |
| FEMNIST-X | FedGDA-D | 0.333333 | 10 |
| FEMNIST-X | FedGDA-D | 0.333333 | 20 |

All four have three suspicious properties:

- they use FedGDA rather than FedOGDA;
- they lie at the highest learning-rate boundary of their respective grids;
  and
- training never started because the preliminary model-selection phase could
  not select a valid candidate.

Before federated training begins, the program performs a 60-epoch
model-selection warmup. It evaluates the configured model and optimizer using a
validation Ψ score. In these configurations, no valid post-burn-in score appears
to have been selected, leaving the internal best score at `-inf`. The code then
failed closed instead of attempting training with an invalid model.

### Important correction to the initial explanation

Each screen row contains only one learning rate. The active coordinator creates
`g_learning_rates = [self.args.learning_rate]` in
[`fedavg_api.py`](../../../fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py).
The model-selection phase did not try a list of alternative learning rates
inside one run.

A more accurate description is:

> For that one configured learning rate, the model-selection trajectory did not
> produce a selectable finite post-burn-in Ψ score.

### Why are they not yet certified boundary failures?

That interpretation is plausible, but the current screen did not preserve
enough evidence.

The four result directories contain only `effective_config.json`. The launcher
recorded `failed_process` and return code 1, but did not preserve:

- per-job stdout and stderr;
- the traceback;
- per-epoch model-selection scores;
- the first nonfinite model-selection epoch;
- whether the structural model, critic, residuals, or Ψ calculation became
  nonfinite; or
- a structured terminal-failure artifact.

Three identical configurations have older logs showing the expected
`best_score=-inf` failure. That is useful supporting evidence, but the corrected
campaign should be independently auditable from its own artifacts.

### Required remediation for the four failures

Rerun only these four configurations after improving failure logging. This is a
deterministic reproduction check, not a retry-until-success policy.

The rerun must use exactly the same:

- dataset;
- algorithm;
- seed;
- learning rate;
- critic multiplier;
- model-selection epochs;
- batch size; and
- server settings.

The outcome rule should be:

- if the same failure reproduces, classify the configuration as
  terminal-ineligible;
- if it unexpectedly succeeds, investigate nondeterminism or an environment or
  launcher discrepancy;
- do not substitute another learning rate; and
- do not add an unplanned replacement candidate.

Before that rerun, add a dated, general protocol rule for “model selection
failed before federated round 0.” It must apply to every configuration rather
than being tailored only to these four observed results.

## 6. Confirmed screen-scorer bug

This is the clearest remaining blocker.

Ψ is the validation statistic used to rank configurations. Each successful run
produces one Ψ value per communication round. Several trajectory summaries are
possible:

- the best Ψ observed anywhere in 150 rounds;
- final-round Ψ; or
- mean Ψ over the last 50 rounds.

The campaign already froze the following rule before the corrected screen:

> Use the last-50-round mean Ψ for selection. Best-round Ψ is diagnostic only.

See
[`PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md`](../deterministic_screen_20260813/PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md).

For a 150-round screen, the correct per-run score is therefore:

```text
mean Ψ over rounds 100 through 149
```

The current scorer instead reads `metrics["best_gmm_eval"]` in
[`score_highdim_screen_post_bn_20260822.py`](../../../scripts/score_highdim_screen_post_bn_20260822.py).
That is the maximum Ψ found anywhere in the run.

### Why this matters

The difference is material rather than cosmetic. Comparing the current
best-round ranking with the frozen last-50 ranking showed:

- top-two membership changes in 8 of 12 dataset/method cells; and
- the rank-one configuration changes in 4 of 12 cells.

Using the current scorer would therefore send several incorrect configurations
into the expensive multi-seed stage.

### Does fixing the scorer require training reruns?

No. Every successful run already stores all 150 round-level Ψ values. The
correct mean over rounds `100..149` can be calculated directly from the
existing CSVs.

Required work:

1. Fix the scorer to use the last 50 round-level Ψ values.
2. Use a protocol-consistent tail validation-MSE statistic for any defined
   fallback or tie handling.
3. Add tests proving the correct round window and ordering.
4. Rescore the preserved trajectories.

No V4 candidate packet should be generated using the current scorer.

## 7. The deeper Ψ-definition question

The scorer issue asks how a stored Ψ curve should be summarized. A second,
deeper question asks whether each stored per-round Ψ value was calculated using
the intended mathematical definition.

### 7.1 Critic-history pooling bug

The active model-selection code contains:

```python
f_of_z_dev_list.extend(f_of_z_dev_list)
```

See
[`model_selection_class.py`](../../../fedgmm/sp_decentralized_mnist_lr_example/model_selection_class.py).

This duplicates a list into itself. The apparent intended behavior is to append
the current candidate's critic history into a separate pool containing critic
histories from all model-selection candidates.

That distinction matters when model selection considers multiple architectures
or optimizer setups. The DeepGMM procedure is intended to pool critic iterates
across all such choices.

For this screen there was only one structural model, one critic model, and one
learning setup. The faulty statement therefore duplicated identical critic
entries. Because Ψ takes a minimum over the entries, duplicating identical
values does not change the numerical result.

This is a real code bug, but it did not numerically invalidate the current
screen. It must be fixed and tested before any future multi-candidate
model-selection experiment.

### 7.2 Which residual becomes the frozen tilde residual?

The Ψ helper searches for the maximum model-selection score. After that search,
however, it returns the residual from the last evaluated iteration, not
necessarily the iteration that attained the maximum. See
[`approximate_psi_objective.py`](../../../fedgmm/sp_decentralized_mnist_lr_example/game_objectives/approximate_psi_objective.py).

FedAvg then freezes that returned residual and uses it when calculating Ψ
throughout the 150 federated rounds.

This behavior is inherited from the
[official CausalML implementation](https://github.com/CausalML/DeepGMM/blob/master/game_objectives/approximate_psi_objective.py).
It was not introduced by the BatchNorm changes.

The scientific concern is that the campaign documentation describes the
paper-defined validation surrogate. The
[DeepGMM paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/12/Deep_Generalized_Method_of_Moments.pdf)
describes pooling critic iterates across hyperparameter choices and evaluating
each candidate structural iterate with its corresponding tilde iterate. The
legacy implementation's frozen returned residual may therefore represent a
different method.

### 7.3 Why this determines whether the complete screen must be rerun

The files preserve:

- per-round scalar Ψ;
- per-round scalar validation MSE;
- a best-validation checkpoint;
- a best-Ψ checkpoint; and
- a final checkpoint.

They do not preserve:

- the structural-model state at every round;
- per-sample validation residuals at every round;
- the complete model-selection critic history; or
- the tilde residual required to evaluate alternative Ψ definitions.

Consequently, a corrected paper-aligned Ψ trajectory cannot be reconstructed
for rounds `100..149` from the existing artifacts.

The decision is:

```text
Which method does the campaign claim?
│
├── Legacy repository/CausalML implementation
│   ├── Preserve the 104 completed trajectories
│   ├── Fix only the last-50 scorer
│   └── Rerun the diagnostic and four failures
│
└── Paper-defined Ψ surrogate
    ├── Fix and test the Ψ/model-selection implementation
    ├── Freeze the new implementation and hashes
    └── Rerun the complete 108-configuration screen
```

If the paper or thesis says that the study evaluated the existing legacy
FederatedDeepGMM/CausalML implementation, retaining the 104 runs is defensible,
provided that exact legacy definition is documented.

If the claim is that the study implemented the paper-defined DeepGMM validation
surrogate, the Ψ implementation should be corrected and the complete screen
rerun.

The current campaign notes lean toward the second claim because they describe
the implemented surrogate as matching the paper. The choice must therefore be
made explicitly with the coauthors before V4. It should be based on the method
the study intends to estimate, not on which definition produces more favorable
candidates.

## 8. Near-zero BatchNorm variance and possible critic collapse

Some critic BatchNorm variances are extremely small, with the minimum near
`2.09e-12`. This value is still finite, positive, and eligible under the frozen
numerical rule.

It may indicate that the critic has become nearly constant or has collapsed to
a weak representation, which could make Ψ less informative. However, the
protocol explicitly makes critic collapse diagnostic-only because no numerical
collapse threshold was frozen before these results were observed.

The correct treatment is therefore:

- record and report it;
- examine whether it persists across signal-stage seeds;
- do not retrospectively reject configurations solely because the variance is
  small; and
- do not invent a post-result threshold and apply it to this screen.

## 9. Why finite does not necessarily mean stable

The coordinator currently defines divergence narrowly as a nonfinite state or
metric. See
[`fedavg_api.py`](../../../fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py).

A run can therefore:

1. attain a good validation MSE early;
2. become progressively worse;
3. finish round 149 with a much worse but still finite MSE; and
4. retain `diverged=False`.

For this reason, “104 passed” should mean that 104 runs completed with finite
numerical state. It should not automatically be restated as 104 configurations
that converged to good stable solutions.

The last-50 Ψ and validation-MSE summaries are valuable precisely because they
measure late-run behavior rather than selecting one lucky checkpoint.

## 10. Other issues that do not require GPU training reruns

### 10.1 Oversized prediction artifacts

The prediction writer saves the complete test image tensor into every
`predictions.npz`. Across the 104 runs, these artifacts occupy approximately
10.08 GiB.

The relevant implementation is
[`save_predictions_npz()`](../../../fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py).

Image runs should instead save compact information such as:

- sample IDs or a compact evaluation coordinate;
- ground truth;
- best prediction;
- final prediction; and
- run metadata.

The predictions themselves are valid. Compact derived artifacts can be created
without rerunning training. Existing large files should only be removed after
numerical equality checks and explicit approval.

### 10.2 Incorrect selection-metric metadata

The generated configs label the primary selector as
`equal_client_validation_mse`. FEMNIST/CIFAR validation data does not carry the
client IDs required to calculate equal-client validation MSE, so the actual
code correctly selects checkpoints using pooled validation MSE.

Thus:

- the numerical selection performed by the run is correct; and
- the metadata label is misleading.

Correct the default in
[`run_manifest.py`](../../../scripts/run_manifest.py) for future runs and add a
correction note for the existing screen. No training rerun is necessary.

### 10.3 Incomplete hash closure

The current shared source list in
[`highdim_protocol_hash_closure_20260822.py`](../../../scripts/highdim_protocol_hash_closure_20260822.py)
does not include every execution-critical dependency or the dataset artifacts.
Representative omissions include:

- client execution code;
- Ψ and model-selection modules;
- optimizer implementations;
- active data loaders; and
- dataset NPZ checksums.

The omitted tracked files are currently clean, and observed source/data mtimes
predate the screen. There is no affirmative evidence that the experiment ran
with different inputs. This is a formal provenance weakness rather than
evidence that the numerical results are wrong.

Before another launch, expand the closure and include all execution-critical
source and dataset hashes. If the publication standard requires perfect
prelaunch cryptographic closure, only a fresh full run can completely remove
the historical caveat; otherwise, retain the screen with a transparent
retrospective-provenance note.

### 10.4 Diagnostic hash timing

The 120-round BatchNorm diagnostic scientifically passed:

- 120/120 rounds were present;
- critic outputs remained finite;
- BatchNorm variance remained positive;
- the corrected buffer policy was recorded; and
- the former round-93 nonfinite failure did not recur.

Its launch-hash record was created after the diagnostic execution. The current
certification therefore validates the artifact/code relationship
retrospectively rather than proving that all inputs were cryptographically
frozen before execution.

Because the diagnostic is inexpensive, the strict remediation is to rerun it
under a fresh namespace after the expanded hashes are frozen.

## 11. Verification performed

The audit verified the complete 104-run artifact set, all checkpoints and
prediction arrays, metric/curve consistency, aggregation traces, campaign
accounting, and protocol hashes currently recorded.

At audit time:

- the complete suite passed 548 tests; and
- the current focused safety/scoring suite passed 39 tests plus 4 subtests.

Passing tests establishes that the implemented plumbing behaves as tested. It
does not override a mismatch between the implemented scientific statistic and
the frozen protocol definition, which is why the Ψ decision remains a blocker.

## 12. Recommended next steps

Execute the following in order:

1. Stop concurrent editing of the campaign files and assign one integration
   owner.
2. Preserve the current 108-attempt screen and ledger unchanged.
3. Decide whether the scientific method is the legacy CausalML Ψ or the
   paper-defined Ψ surrogate.
4. Fix the screen scorer to use the last-50-round statistic.
5. Add structured model-selection failure diagnostics and per-job log capture.
6. Define a general terminal-pretraining-failure artifact and protocol rule.
7. Fix and test critic-history pooling before future multi-candidate model
   selection.
8. Correct the selection-metric metadata.
9. Replace future image prediction artifacts with the compact result contract.
10. Expand the source and dataset hash closure.
11. Run the full verification suite and freeze all hashes.
12. Rerun the exact 120-round BatchNorm diagnostic in a fresh namespace.
13. Rerun the four exact failed configurations without changing their
    hyperparameters.
14. If legacy Ψ is retained, rescore the preserved 104 trajectories together
    with the certified terminal outcomes.
15. If paper-defined Ψ is adopted, rerun the complete 108-configuration screen
    under the corrected implementation.
16. Perform the frozen boundary review.
17. Generate and hash the V4 candidate packet.
18. Launch V4 signal first.
19. Launch X only if every signal cell has a valid frozen promotion.

## Final decision statement

The corrected training campaign is mostly healthy. The expensive 104 completed
trajectories should not be discarded casually. They are valid evidence for the
legacy implementation and their MSE/checkpoint results remain sound.

Candidate selection and V4 launch must nevertheless remain blocked until:

- the last-50 scorer is fixed;
- the four pretraining failures are properly reproduced and certified; and
- the campaign explicitly chooses and documents its intended Ψ definition.

The minimum rerun is the diagnostic plus the four failed configurations. A
complete screen rerun is required only if the campaign adopts a corrected
paper-defined Ψ calculation or insists on eliminating the incomplete prelaunch
hash-closure caveat entirely.