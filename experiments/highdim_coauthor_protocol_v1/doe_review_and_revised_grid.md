# Independent Review of the Federated DeepGMM Hyperparameter DOE

Prepared 2026-08-07. Audit of the proposed grid against primary literature, plus
a revised design of experiment.

**Provenance labels used throughout:**

* `[PAPER]` — directly stated in the published paper, with location.
* `[CODE]`  — confirmed in the official repository, with file path.
* `[INFER]` — my inference from a reported fact; reasoning given.
* `[REC]`   — my recommendation, not a literature claim.
* `[INTERNAL]` — evidence from this repository's own completed runs.

---

# Part I — Executive verdict

**Verdict: usable but substantially underjustified in its literature
attribution. Several stated anchors do not survive checking, and one is
contradicted outright by the source paper.**

The grid is not unreasonable as engineering. The problem is that its
justifications point at DeepGMM, and DeepGMM's actual configuration says
something materially different.

The single most important finding:

> DeepGMM does **not** use one critic multiplier. It uses `λ_f = 5.0` for the
> low-dimensional, MNIST_z and MNIST_xz scenarios, and **`λ_f = 1000.0` for
> MNIST_x** — a 200x difference, chosen per scenario. `[PAPER]` Table 3,
> Appendix B.2; `[CODE]` confirmed.

Since this project's scenarios *are* DeepGMM's MNIST_x/z/xz scenarios (same
`abs` response, same 20,000 points), a proposal that fixes `c ∈ {1,3,5,10}`
uniformly across x, z and xz is directly at odds with the paper it cites.

### Highest-risk assumptions, ranked

1. **Transferring OAdam-tuned ratios to SGD/OGDA.** Every DeepGMM number is for
   OAdam with `betas=(0.5,0.9)`. Adam-family updates are gradient-magnitude
   normalised; SGD updates are not. Neither the absolute learning rates nor the
   ratio transfer cleanly. The proposal's `c=5` "inspired by DeepGMM" inherits a
   number whose meaning depends on the optimiser it was tuned for.
2. **One critic multiplier across x / z / xz.** Contradicted by Table 3 above.
   The `*_x` scenarios (CNN `g`, small MLP `f`) are exactly where DeepGMM needed
   a very large ratio.
3. **`local_steps = 3` presented as a default.** No surveyed paper uses 3.
   Federated minimax theory gives an explicit `(τ-1)²` penalty on the
   local-update error term.
4. **`server_learning_rate = 1.5`.** No external support found. The federated
   minimax paper aggregates by plain averaging (server LR = 1.0); the adaptive
   federated optimisation literature grid-searches server LR and finds the
   optimum is task-dependent.
5. **Reusing candidates selected at a different client count.** The incumbent
   `lr=0.03, cm=3` was selected on one cell at `client_num_in_total=1000`,
   full participation, ~20 samples/client. The design now targets
   `client_num_in_total=10`, also full participation, but ~2,000
   samples/client — a ~100x change in local batch size and therefore in
   gradient noise, which the old candidate was never tested against.
6. **Selecting on validation MSE.** DeepGMM selects on a validation *variational
   surrogate* `Ψ̂_n`, which is `g0`-free. Validation MSE requires knowing the
   true `g0` and therefore does not exist outside synthetic benchmarks.

---

# Part II — Literature evidence table

`NR` = not reported.

| Paper | Problem class | Optimizer | Model LR (η_g) | Critic LR (η_f) | Ratio | Local steps | Participation | Rounds | Relevance |
|---|---|---|---|---|---|---|---|---|---|
| Bennett, Kallus & Schnabel 2019, **DeepGMM**, NeurIPS — low-dim | NC-NC smooth game | OAdam β=(0.5,0.9) | 5e-4, 2e-4, 1e-3 | λ_f·η_g | **5.0** | 1:1 alternating | centralized | 100 epochs | Same estimator |
| ″ — **MNIST_z** | ″ | ″ | 2e-5, 5e-5, 1e-4 | λ_f·η_g | **5.0** | 1:1 | centralized | 60/100 ep | **Same scenario** |
| ″ — **MNIST_x** | ″ | ″ | 1e-6, 2e-6, 5e-6 | λ_f·η_g | **1000.0** | 1:1 | centralized | 60/100 ep | **Same scenario** |
| ″ — **MNIST_xz** | ″ | ″ | 1e-6, 2e-6, 5e-6 | λ_f·η_g | **5.0** | 1:1 | centralized | 60/100 ep | **Same scenario** |
| Sharma, Panda, Joshi & Varshney 2022, ICML (Local SGDA) | NC-C, NC-PL, NC-1PC | SGDA / momentum | NR (theory) | NR | NR | τ, error ∝ **(τ-1)²** | **full** sync | T/τ | Local-step penalty |
| Sharma et al. 2023, TMLR (Fed-Norm-SGDA+) | NC-C, NC-1PC | Normalized SGDA | NR | NR | separate γ_x^s, γ_y^s | **E = 5 (and 7)**; τ∈{1,5,10} | **partial**, P of n | **200** | Closest fed-minimax |
| Reddi et al. 2021, ICLR (Adaptive Federated Optimization) | NC minimization | FedAvg/FedAvgM/FedAdam/FedYogi | grid | — | — | NR | partial | task-dep. | Server-LR practice |
| Lewis & Syrgkanis 2018, **AGMM** | one-step GMM, ‖·‖_∞ | no-regret (Hedge) + jitter | NR | **not a critic LR** | — | — | centralized | per-epoch jitter | See caveat below |

**AGMM caveat.** AGMM performs *no-regret learning over moment conditions* with
an `‖·‖_∞` norm and a jitter step after each epoch — this is `[PAPER]` from
DeepGMM's own baseline description (§5, baseline 5). Its "learning rate" is a
**Hedge/no-regret rate over a simplex of moments**, *not* a critic network
learning rate. It must **not** be cited as evidence for a critic multiplier. I
did not open AGMM's own repository, so I make no claim about its internal
constants.

---

# Part III — DeepGMM / AGMM analysis

### What the multiplier actually is

`[PAPER]` Appendix B.2: *"in every scenario we explore a range of learning rates
for g, and compute the f learning rate as `lr_f = λ_f lr_g`, where `λ_f` is
chosen separately for each scenario."*

`[CODE]` `methods/mnist_x_model_selection_method.py`:

```python
g_learning_rates = [0.000005, 0.000002, 0.000001]
f_optimizer_factory = OptimizerFactory(OAdam, lr=1000*g_lr, betas=(0.5, 0.9))
```

`[CODE]` `methods/mnist_xz_model_selection_method.py`: `g_lr ∈ [5e-6, 2e-6, 1e-6]`,
f multiplier `5.0`, same betas.

This settles the disambiguation the brief asked for. `λ_f` is **(1) a
multiplication of the critic learning rate**. It is *not* a loss coefficient,
*not* extra critic steps, *not* a critic regulariser. The regularisation term
`-¼C_θ̃(f,f)` is a separate, fixed part of the objective (Eq. 9), not a tunable
critic weight.

### Update ratio, clipping, weight decay

`[CODE]` `learning/learning_dev_f.py`: strictly **1 g step then 1 f step** per
batch — zero g grads, backward with `retain_graph=True`, step g; zero f grads,
backward, step f. **No gradient clipping and no weight decay anywhere.**

`[PAPER]` §3.3 discussion of Assumption 5: bounding weights *"(equivalently,
using weight decay)"* is available but *"We do not find doing this is necessary
in practice."*

**Consequence for this project `[INFER]`:** DeepGMM's ratios were tuned in a
regime with *no gradient clipping*. This project clips both players to norm 1.0
before the step. Once both gradients are clipped to the same norm, the ratio of
*applied* step sizes is `c` exactly, whereas unclipped it was `c` times the
natural gradient-magnitude ratio. Clipping therefore changes what `c` means.
This is a real reason DeepGMM's `λ_f` values cannot be read across numerically.

### Why MNIST_x needs a huge ratio `[INFER]`

Architecture pairing drives it:

| Scenario | `g` | `f` | λ_f |
|---|---|---|---:|
| MNIST_z | FCNN(200,200) | CNN | 5.0 |
| MNIST_xz | CNN | CNN | 5.0 |
| **MNIST_x** | **CNN** | **FCNN(20)** | **1000.0** |

`λ_f` is large exactly when `g` is a CNN and `f` is a *small* MLP. The g
learning rate is driven down to `1e-6` by the CNN; a 20-unit MLP critic at
`1e-6` would not move at all. `λ_f=1000` restores the critic to `1e-3…5e-3`, a
normal Adam rate. So the ratio is compensating for the *g* rate being set by the
harder network, not expressing "the critic should be 1000x faster".

**This is the key transferable insight, and it is structural rather than
numerical:** the correct `c` depends on the `g`/`f` architecture pairing, and
the `*_x` scenarios sit in a different regime from `*_z` and `*_xz`.

### Model selection

`[PAPER]` §4 and Appendix B.1: hyperparameters are chosen by a **validation
variational surrogate** `Ψ̂_n(θ) = Ψ_n(θ; F̂, θ)`, where `F̂` pools the critic
iterates encountered across *all* hyperparameter choices. Early stopping
*"periodically evaluate our iterate θ using Ψ̂_n, and return the best evaluated
iterate."* `[CODE]` best iterate restored post-hoc via
`max(range(len(eval_history)), key=...)`.

Note this repository already implements that surrogate
(`approx_psi_eval`, `dev_f_collection`), but the current protocol selects on
`best_validation_mse`. That is a deviation from the cited method — see Part V.

---

# Part IV — Federated minimax local-step analysis

### What the theory actually says

`[PAPER]` Sharma et al., ICML 2022, Theorem 3 decomposes the error as

```
E‖∇Φ‖² ≤ [ error with full synchronization ]
       + O( (τ-1)² [ (η_x²+η_y²)σ² + (η_x²ς_x² + η_y²ς_y²) ] )   <- local updates
```

Remark 4: the local-update term's dependence *"is quadratic"*, so *"for small
enough α and carefully chosen τ, having multiple local updates does not affect
the asymptotic convergence rate."*

**Read precisely:** more local steps are *tolerable*, not *beneficial*, and the
penalty grows as `(τ-1)²` and multiplies the heterogeneity constants `ς_x, ς_y`.
Relative to `τ=1`, the penalty is **0 (τ=1), 4x (τ=3), 16x (τ=5)**. At
`alpha=0.1` — the most heterogeneous cell — `ς` is largest, so this is exactly
where large `τ` hurts most. This directly addresses Concern 3: yes, local steps
interact with heterogeneity, multiplicatively.

**Limitation:** Algorithm 2 aggregates over *all* `i ∈ [n]` (full
synchronization) and the server does a plain average. So this theorem does not
itself cover 1% participation.

### Partial participation — not applicable to this design

Sharma et al., TMLR 2023, Theorem 2's partial-participation error term
(`+ O(((n-P)/(n-1))·(E_w/(P·T))^{1/4})`) only applies when `P < n` — some
clients are sampled out each round. **This design uses
`client_num_per_round = client_num_in_total = 10`** (full participation, no
sampling — see `deterministic_10client_proposal.md`), so `P = n` and this
term is exactly zero. It does not justify anything about this arm's round
count.

This term *is* the reason the completed stochastic arm (`fedgda_s`/
`fedogda_s`, `client_num_per_round=10` of `client_num_in_total=1000`) needed a
long horizon — that arm genuinely has `P/n = 1%`. It is not evidence for this
deterministic arm's round count; see `deterministic_10client_proposal.md` §4
for why 500 rounds, not 1500, is the right figure here.

### What was actually run

`[PAPER]` TMLR 2023 §6: `n = 15` clients, Dirichlet `Dir_n(α)`, `α = 0.1`,
CIFAR-10 with VGG11, robust-NN and fair-classification objectives. Figure 3 uses
**`E = 5` (and `E = 7`)** over **200 communications**. The earlier survey result
reports `τ ∈ {1, 5, 10}` tested.

**Direct answer to the brief's question:** *no surveyed paper uses exactly 3
local steps.* The empirical values are 1, 5, 7, 10. `τ=3` cannot be described as
literature-supported. It can be defended on a different ground — see Part V.

**Local-step normalisation.** Fed-Norm-SGDA normalises client updates by the
number of local steps because clients may perform *different* `τ_i`. In this
project every client performs exactly 3, so the normalisation is a uniform
constant and folds into the server learning rate `[INFER]`. Not an issue here,
but it would become one if straggler/variable-work clients were ever introduced.

---

# Part V — Critique of the current grid

| Item | Current | Verdict | Reason |
|---|---|---|---|
| Model LR `{0.003, 0.01, 0.03}` | 3 values | **Keep, re-label** | Reasonable for clipped SGD, but has **no** DeepGMM support (DeepGMM: 1e-6…1e-3 under OAdam). Justify internally, not by citation. |
| Multiplier `{1, 5, 10}` "literature" | uniform | **Modify** | `c=5` is real `[PAPER]` but only for low-dim/z/xz. `c=10` is not a DeepGMM value. `c=1` appears nowhere. Must become scenario-dependent. |
| `c=3` incumbent | retained | **Keep as incumbent only** | `[INTERNAL]` full-participation evidence. Not literature. Label honestly. |
| No large `c` for `*_x` | absent | **Add** | Contradicts `λ_f=1000` for MNIST_x `[PAPER]`. Highest-value single fix. |
| Critic LR cap `0.1` | hard cap | **Modify** | Arbitrary. With `clip=1.0`, per-step move is `≤ η_f · 1.0 · server_lr`; state the cap as an effective-step bound or replace with divergence rejection. |
| `local_steps = 3` | primary | **Keep, re-justify** | Not literature-backed. Defensible only as continuity with the completed stochastic arm (which used `epochs=3`). Changing it would confound cross-arm comparison. |
| `{1,3,5}` ablation | gate | **Keep, extend** | Good. Prefer `{1,3,5}` reported *with* the `(τ-1)²` prediction as a stated hypothesis. |
| `server_lr = 1.5` | fixed | **Unresolved → measure** | No external support. Fed-minimax uses plain averaging (=1.0); FedOpt grid-searches it. Keep 1.5 for continuity **and** measure `{1.0, 1.5}` once in the gate. |
| `clip_norm = 1.0` | fixed | **Keep, document** | DeepGMM used none. Fine to keep, but it changes the meaning of `c` and interacts with OGDA (below). |
| 150 / 500 / 1500 | schedule | **Revise → 150 / 500 / 500** | Under full participation (`client_num_in_total=10`, no sampling), the `T^{-1/4}` partial-participation argument for 1500 does not apply. Use 500, matching the pre-existing `protocol_summary.json` deterministic-arm target; revisit only if convergence is incomplete at 500. |
| Seed-0 tuning | single seed | **Modify** | Even under full participation, network initialization differs by seed; a single seed 0 can still land on a lucky/unlucky init. (The client-sampling variance concern does not apply — there is no client sampling in this design.) |
| Shared GDA/OGDA grid | shared | **Modify** | See Concern 2 below — separate grids. |

### Concern 1 — multiplier vs independent grid

`[REC]` **Keep the multiplier parameterisation.** DeepGMM itself uses
`lr_f = λ_f·lr_g` with `λ_f` fixed per scenario and only `lr_g` searched
`[PAPER]`. It is not more restrictive in principle — `(η_g, c)` and `(η_g, η_f)`
are related by a bijection — the restriction comes only from *which* pairs you
enumerate. The multiplier form is preferable here because the DeepGMM evidence
is expressed in that coordinate system, making the design traceable. Enumerate
`c` per scenario group rather than globally.

### Concern 2 — OGDA effective step

The OGDA update is `θ ← θ - η(2g_t - g_{t-1})`. Gradients are clipped to norm
1.0 *individually*, before the optimistic combination, so the combined direction
can reach norm **3.0** when consecutive gradients oppose — 3x the GDA bound
`[INFER]`, consistent with the existing handoff analysis.

`[INTERNAL]` The learning gate corroborates this: at `lr=0.03`, FedOGDA-D had
**33 non-finite metric rows at cm=10** and 3 at cm=3, while FedGDA-D at
`0.03/cm3` was clean across all 150 rounds.

**Decided 2026-08-07:** keep clip-then-combine (no code change) and compensate
with OGDA's downshifted LR grid (Part VI: `η_g ∈ {0.001,0.003,0.01}` vs
GDA's `{0.003,0.01,0.03}`). The combined-direction-clip alternative was not
adopted. This convention applies to every result in this campaign — state it
wherever a GDA/OGDA stability claim is reported, since "OGDA is less stable"
is only a fair claim under a stated clipping convention.

### Concern 4 — scenario-specific critic difficulty

`[PAPER]` Answered decisively: DeepGMM uses **5.0 / 5.0 / 1000.0** for
z / xz / x. Scenario-specific ratios are the source paper's own practice, not a
deviation from it. Group by architecture pairing, not by dataset — FEMNIST and
CIFAR-10 share the pairing structure, so `[REC]` group by `x` / `z` / `xz` and
**not** separately by dataset, at least initially.

### Concern 5 — selection criterion

`[PAPER]` DeepGMM selects by validation `Ψ̂_n` and returns the best evaluated
iterate. `[REC]`:

* **Primary selector: validation Ψ surrogate** (`approx_psi_eval`, already
  implemented) — matches the cited method and is `g0`-free, so it is the only
  criterion that would survive on real data.
* Report validation MSE as a secondary, synthetic-only diagnostic.
* Report **both** best-iterate and last-K-round average. `[INTERNAL]` the
  stochastic finals already show 83/90 and 81/90 runs ending above 2x their best
  checkpoint — reporting only best-iterate hides this.

**On optimistic bias:** selecting the best of ~T validation evaluations biases
the *validation* number upward. It does **not** bias the reported test number,
because test is read only after both the configuration and the checkpoint are
fixed — that protocol is already correct. State the validation figure as
selection-biased and quote test-at-best-validation as the headline.

### Concern 6 — tuning seeds

`[REC]` Under full participation there is no client-sampling variance to
control for (no sampling occurs — every client trains every round). Seed-to-
seed variance still comes from network initialization. Add a **3-seed
confirmation of the top-2 candidates** before committing to finals; promote by
median, not best.

---

# Part VI — Revised recommended grid

This grid targets the deterministic arm as decided in
`deterministic_10client_proposal.md`: `client_num_in_total =
client_num_per_round = 10`, full participation, no client sampling.

Scenario groups by architecture pairing (`[PAPER]`-motivated):

```text
Group Z  (*_z) : g = MLP,  f = CNN     -> critic already the larger net
Group XZ (*_xz): g = CNN,  f = CNN     -> balanced
Group X  (*_x) : g = CNN,  f = small MLP -> critic starved; DeepGMM used 1000
```

### Learning-rate grids

```text
FedGDA   η_g ∈ {0.003, 0.01, 0.03}
FedOGDA  η_g ∈ {0.001, 0.003, 0.01}      # shifted 3x down, see Concern 2

critic multiplier c, by group:
  Group Z   c ∈ {1, 5}
  Group XZ  c ∈ {1, 5}
  Group X   c ∈ {5, 50}                  # tests the DeepGMM MNIST_x regime
```

6 candidates per (scenario, alpha, method) = 3 learning rates x 2 critic
multipliers. `c=3` is retained implicitly as the incumbent for the
`femnist_z, alpha=0.5` cell only, for continuity.

**`[INTERNAL]` corroboration — the repo's own default already encodes this
grouping.** `experiment_utils.py:237` `default_critic_multiplier()` returns
**5.0** for `{mnist_z, femnist_z, mnist_xz, femnist_xz, cifar10_z,
cifar10_xz}` and **20.0** otherwise — i.e. 20.0 for exactly the Group X
scenarios (`femnist_x`, `cifar10_x`). So three independent sources agree that
Group X needs a larger multiplier than Z/XZ: DeepGMM's paper (1000 vs 5), the
`g`/`f` architecture-pairing argument, and this repository's own default
(20 vs 5). The proposed Group X range `{5, 50}` brackets that internal
default of 20, which is a better-supported position than the single uniform
`c=3` the pre-review grid used everywhere.

**No critic-LR cap exists in code.** Part V lists a "critic LR cap 0.1" as a
current item; `experiment_utils.py:342` computes
`f_learning_rate = critic_multiplier * learning_rate` with no clamp, so the
cap was a grid-design convention rather than an enforced limit. Group X's
`c=50` arm will therefore take effect as written (e.g. `η_g=0.01, c=50`
gives `η_f=0.5`); divergence is caught by Screen, per the Part V verdict on
that row.

### Fixed

```text
client_num_in_total   = client_num_per_round = 10   # full participation, no client sampling
local_steps           = 3        # continuity with completed stochastic arm
server_learning_rate  = 1.5      # continuity; measured against 1.0 in the gate
gradient_clip_norm    = 1.0
objective_mode        = legacy, lambda_1 = 0.1
aggregation           = sample_size
auxiliary_regression  = false
```

### Staged elimination

Rounds updated for full participation (`client_num_in_total=10` — no
coupon-collector argument for a long horizon; see
`deterministic_10client_proposal.md` §4). This table is the extended
(per-alpha-tuning) shape; the **adopted** plan is the minimal plan in Part
VII, priced below in "Cost — measured."

| Stage | Rounds | Runs | Rule |
|---|---:|---:|---|
| 0. Gate | 500 | 24 | `τ ∈ {1,3,5}` x `server_lr ∈ {1.0,1.5}` x 2 xz-scenarios x 2 methods, alpha 0.1, seed 0 |
| 1. Screen | 150 | 216 | Reject non-finite state **or** metrics; reject worse than constant predictor |
| 2. Rank | 500 | 72 | Top-2 per cell by validation Ψ |
| 3. Confirm | 500 | 216 | Top-2 x seeds {0,1,2}; promote by **median** validation Ψ |
| 4. Finals | 500 | 72 | Winner per cell x seeds {3,4}; seeds 0–2 reused from stage 3 |
| **Total** | | **600** | Reported table = 36 cells x 5 seeds = 180 runs @500 |

Selection: validation Ψ surrogate primary, validation MSE secondary, test read
only after both config and checkpoint are frozen. No client sampling exists in
this design, so there is nothing to pair across candidates.

### Cost — measured 2026-08-07

Real benchmark run (`deterministic_10client_runtime_profile_20260807/runtime_profile_findings.md`)
replaces the earlier placeholder. Measured per-round cost at the adopted
config: `femnist_z` 2.48 s/round, `femnist_x` 2.45 s/round, `cifar10_xz`
8.99 s/round (setup 54–165s, roughly unchanged from N=1000). This is a real
but more modest speedup than guessed pre-benchmark — **3.3–7.7x faster per
round than the 1000-client measurement**, not the 9.5–22x range assumed
earlier, because setup time doesn't shrink with client count and the
per-round win narrows for the more GPU-bound CIFAR scenarios.

**Adopted minimal plan (300 runs, Part VII): 188.8 GPU-h = 3.93 quota weeks**
(after the Rank/Confirm de-duplication recorded in Part VII). The extended
plan above was not adopted — see Part VII. Note the same duplication exists
in the extended table's stage 2/3 rows and would need the same fix if it were
ever adopted.

**Concurrency is not a further lever here.** Measured GPU utilization at the
adopted config is **82% (FEMNIST) and 95% (CIFAR)** — the GPU is genuinely
busy, unlike the N=1000 regime where it idled at 42–66% waiting on
kernel launches. Larger local batches (~2,000 samples vs ~20) converted that
idle time into real compute. Memory is no longer the binding constraint
(21.5 GB FEMNIST / 54.2 GB CIFAR peak, of 95.8 GB), but packing two runs onto
one GPU would now mostly split the same throughput rather than add any, so it
would not reduce billed GPU-h. The N=1000 profile's optimisation item 3
("more concurrent runs per GPU") does **not** carry over to this design.

---

# Part VII — Minimal and extended plans

### Run-count formulas

```text
Let  C = candidates per cell, S = scenarios (6), A = alphas (3),
     M = methods (2), K = confirm seeds, F = final seeds

Minimal  : N = C·S·M            (screen, alpha-shared)
           + 2·S·M              (rank)
           + S·M·F              (finals)
Extended : N = C·S·A·M + 2·S·A·M + 2·S·A·M·K + S·A·M·F
```

### Minimal defensible plan — 300 runs, **188.8 GPU-h (3.93 quota weeks) — measured 2026-08-07** — **FINAL, adopted 2026-08-07**

Tune at `alpha = 0.5` only, share the selection across alphas, verify at
`alpha = 0.1`. The reported table is still the full 3 alphas x 5 seeds — this
keeps the heterogeneity-sensitivity result in the finals while cutting tuning
cost roughly 3x. GPU-h from the real benchmark in
`deterministic_10client_runtime_profile_20260807/runtime_profile_findings.md`
(per-scenario cost, not a flat rate — CIFAR cells cost ~3.6x more per run
than FEMNIST cells).

| Stage | Rounds | Runs | GPU-h |
|---|---:|---:|---:|
| 1. Screen (alpha 0.5 only) | 150 | 72 | 17.7 |
| 2. Rank (top-2/cell, seed 0) | 500 | 24 | 18.0 |
| 3. Confirm (top-2 x seeds **{1,2}**) | 500 | 48 | 36.0 |
| 3b. Stability check @ alpha 0.1 | 500 | 12 | 9.0 |
| 4. Finals (3 alphas x 5 seeds) | 500 | 144 | 107.9 |
| **Total** | | **300** | **188.8** |

**Corrected 2026-08-07 (−24 runs, −18.0 GPU-h).** Confirm previously listed
seeds `{0,1,2}`, but Rank already runs exactly those top-2 candidates at 500
rounds on seed 0 — identical config, horizon and seed, so 24 of Confirm's 72
runs were re-executions of Rank. Confirm now runs seeds `{1,2}` only and
takes seed 0 from Rank; the median-of-3 promotion rule is unchanged. Pure
bookkeeping — no scientific content is lost.

**Escape hatch (unchanged):** any cell whose winner diverges or fails the
constant-predictor test at the alpha=0.1 stability check must be re-tuned at
that alpha specifically — for that cell only, fall back to the extended
plan's per-alpha tuning below (adds cost only for the affected cell, not the
whole campaign).

### Extended plan — 600 runs — not adopted, fallback per-cell only

The Part VI staged design in full — per-`(alpha, scenario, method)` tuning,
matching the completed stochastic protocol's 36-cell structure. Superseded as
the default by the minimal plan above; still the correct fallback for any
individual cell that fails the alpha=0.1 stability check. Not separately
priced — its run-volume is ~1.85x the minimal plan's, so whole-campaign cost
would be roughly 1.85x the 206.6 GPU-h figure if ever adopted in full.

### Must every cell be tuned independently?

`[REC]` **No — group by architecture pairing, tune per (scenario, method), and
treat alpha as a stability check rather than a tuning axis.** Justification: `c`
compensates for the `g`/`f` capacity imbalance `[INFER from PAPER]`, which is a
property of the scenario, not of the partition. Heterogeneity affects the stable
*step size* through `ς` in the `(τ-1)²` term `[PAPER]`, so alpha can shift the
usable `η_g` — hence the stability check rather than blind sharing. The
completed stochastic arm tuned all 36 cells; matching that is the conservative
option and is affordable in the extended plan.

---

# Part VIII — Implementation checklist

1. **`critic_multiplier` semantics.** Confirmed `[INTERNAL]`
   `experiment_utils.py:342` sets `f_learning_rate = critic_multiplier *
   learning_rate` — it scales the **learning rate**, matching DeepGMM's `λ_f`.
   No loss-coefficient or extra-step interpretation. ✔
2. **Optimizer parameter groups.** Verify `g` and `f` have genuinely separate
   optimizers with separate LRs, and that `weight_decay` reaches neither (the
   manifest field is known not to affect the GMM optimizers).
3. **Gradient-clipping order — decided.** Clip-then-combine, per-player
   (unchanged code); OGDA's downshifted LR grid (Part VI) is the chosen
   compensation, not a combined-direction clip. Document this convention
   wherever GDA/OGDA results are compared (Concern 2).
4. **Local-step semantics.** Confirm `epochs=3` means 3 optimizer steps per
   client per round (1 full batch each), not 3 passes over multiple batches.
5. **Full-batch vs minibatch — no longer equivalent.** That equivalence held
   only at `client_num_in_total=1000` (~20 samples/client). At the adopted
   `client_num_in_total=10` (~2,000 samples/client), `batch_size=0` (full
   local batch) and `batch_size=256` are genuinely different settings — see
   `deterministic_10client_proposal.md` §2. Use `batch_size=0`; do not carry
   forward the old "identical" claim.
6. **No client sampling to control for.** `client_num_in_total =
   client_num_per_round = 10` — every client participates every round, so
   there is no client sequence to pair across candidates. Seed still controls
   network initialization; confirm on ≥1 additional seed per Part V "Seed-0
   tuning".
7. **Validation/test separation.** Test must be read only after config *and*
   checkpoint are frozen. Already enforced; keep `test_mse_used_for_selection=false`.
8. **Checkpoint selection.** Record best-Ψ, best-val-MSE, final, and last-K
   average for every run.
9. **GDA vs OGDA fairness.** Separate LR grids, identical everything else,
   documented clipping convention.
10. **Manifest columns.** The `alpha0p5` source manifest predates
    `auxiliary_regression` and friends; rows written against its header drop
    them silently and default to aux **on**. Append `EXTRA_FIELDS` and verify
    generated YAML before launch.

---

# Part IX — Terminology

**Decided 2026-08-07: the 10-of-1000 sampled option this section originally
analyzed was rejected — see `deterministic_10client_proposal.md`.** The
adopted design uses `client_num_in_total = client_num_per_round = 10`: full
participation, no client sampling.

Under full participation the global update genuinely is a deterministic
function of the current iterate — there is no sampled client set to
introduce randomness. **`FedGDA-D` / `FedOGDA-D` is the correct label; no
rename is needed.**

<details>
<summary>Superseded: analysis of the rejected 10-of-1000 option (kept for the record)</summary>

Do not call a 10-of-1000-sampled arm "deterministic". The global update would
be stochastic because the sampled client set changes every round; full local
batches remove only *within-client* minibatch noise. It would need the label
**partial-participation full-local-batch FedGDA / FedOGDA**, abbreviated
**FedGDA-PP / FedOGDA-PP**. In the paper: *"each sampled client performs
full-batch local updates; the global update remains stochastic through 1%
client sampling."*

Under that rejected option, a "full-batch vs minibatch" contrast would also
not have been meaningful at 10-of-1000 client sizes — both settings process
exactly one local batch for essentially every client. Under the adopted
`client_num_in_total = 10` design that problem does not arise — see
`deterministic_10client_proposal.md` §2 — full-batch and minibatch are now
genuinely different settings.

</details>

---

# What I did not verify

Stated so nothing here is over-claimed:

* AGMM's own repository and configuration constants — characterised only via
  DeepGMM's `[PAPER]` description of it as a baseline.
* Deng & Mahdavi 2021 (Local SGDA / Local SGDA+) — referenced through Sharma et
  al.'s descriptions, not opened directly.
* DeepGMM Appendix B.2.2 ("MNIST_z Scenario") is **truncated mid-sentence** in
  the published arXiv v2 PDF; the per-scenario prose for MNIST_z is unavailable.
  Table 3 and the code are the usable sources.
* No federated paper was found that runs DeepGMM-style minimax specifically;
  the federated evidence is from robust-training and fair-classification
  minimax objectives, which share the saddle-point structure but not the
  moment-condition objective.

## Sources

* Bennett, Kallus & Schnabel, *Deep Generalized Method of Moments for
  Instrumental Variable Analysis*, NeurIPS 2019 — https://arxiv.org/pdf/1905.12495
  (Table 3, Appendix B.2; §4; §3.3)
* Official code — https://github.com/CausalML/DeepGMM
  (`methods/mnist_x_model_selection_method.py`,
  `methods/mnist_xz_model_selection_method.py`, `learning/learning_dev_f.py`)
* Sharma, Panda, Joshi & Varshney, *Federated Minimax Optimization: Improved
  Convergence Analyses and Algorithms*, ICML 2022 —
  https://proceedings.mlr.press/v162/sharma22c/sharma22c.pdf (Thm 3, Rmk 4–5)
* Sharma et al., *Federated Minimax Optimization with Client Heterogeneity*,
  TMLR 2023 — https://arxiv.org/abs/2302.04249 (Thm 2; §6, Fig. 3)
* Reddi et al., *Adaptive Federated Optimization*, ICLR 2021 —
  https://arxiv.org/pdf/2003.00295 (Appendix D.2, Table 8)
* Lewis & Syrgkanis, *Adversarial Generalized Method of Moments*, 2018 —
  https://arxiv.org/abs/1803.07164
