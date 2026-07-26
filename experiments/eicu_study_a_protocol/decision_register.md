# Study A v1 decision register

This register contains decisions that remain unresolved despite the frozen
scientific invariants in `protocol_v1.md`. None may be inferred from demo data
or from undocumented assumptions about the full eICU cohort.

| ID | Status | Decision required before | Recommended choice | Rationale / required evidence |
|---|---|---|---|---|
| D01 | unresolved | full cohort build | Freeze the exact full eICU release/version, credentialed source path, file inventory, and checksums. | Reproducibility requires immutable source identity; the demo release is not a substitute. |
| D02 | unresolved | full cohort build | Adopt the existing adult, first-stay, near-admission-sepsis cohort proposal only after a blinded full-release flow audit. | The protocol must not invent full-release counts, coverage, or representativeness. Publish each inclusion/exclusion count. |
| D03 | unresolved | scenario generation | Freeze hospital eligibility gates for minimum total/train/validation/test rows, wards, and real structural instrument variation before outcome simulation. | Each client needs evaluable splits and meaningful within-client \(Z\); thresholds must not be outcome-selected. |
| D04 | unresolved | scenario generation | Use training-only cross-fitted, shrunk ward preference as \(Z\), conditional on passing leakage and structural-variation certification. | This preserves hospital clients and ward structure while avoiding own-row treatment leakage. Exact folds, priors, and fallback behavior need freezing. |
| D05 | unresolved | scenario generation | Do not use grouped hospitals as primary clients; if too few hospitals pass, stop and redesign under a new protocol version. | “Hospital is the client” is a frozen Study A invariant. |
| D06 | unresolved | scenario generation | Freeze the within-hospital split fractions at 60/20/20 if the full audit shows adequate rows in every split; otherwise amend fractions before tuning. | The demo cannot determine full-release split feasibility. Admission-level separation and train-only preprocessing remain mandatory. |
| D07 | unresolved | scenario generation | Calibrate instrument strength, confounding, outcome-confounder coefficient, noise, and hospital-offset scale using treatment balance, first-stage diagnostics, and nontriviality checks—not recovery Test MSE. | The benchmark needs endogeneity, overlap, finite outcomes, and nondegenerate within-client relevance without tuning toward a favored method. |
| D08 | unresolved | scenario generation | Freeze a dedicated random-MLP architecture, scaling rule, `g0_seed`, serialized weights, and checksum; keep it identical across scenario seeds. | “Frozen random MLP” must describe one reproducible target rather than a new target per optimizer comparison. |
| D09 | unresolved | tuning launch | Extend orchestration to carry separate `scenario_seed`, `optimizer_seed`, and `seed_pair_id`. | Current read-only implementation context uses one `seed` field; that would violate the pairing policy. |
| D10 | unresolved | tuning launch | Record the exact frozen-\(\tilde\theta\) refresh cadence for federated and centralized objectives and verify it in effective configuration. | The paper-aligned invariant fixes a frozen reference and \(\lambda=1/4\), but cadence must be operationally unambiguous. |
| D11 | unresolved | tuning launch | Use full client participation if feasible; otherwise freeze a paired participation schedule before tuning. | Uniform aggregation is over participating clients. Partial participation can add avoidable comparison noise. |
| D12 | proposed pending runtime preflight | tuning launch | Use the three tuning pairs in the protocol, a cheap first-pair screen, then all-three-pair evaluation of every shortlisted candidate. | This separates tuning and confirmation while controlling full-eICU cost. Shortlist size must be set from runtime evidence. |
| D13 | proposed pending runtime preflight | tuning launch | Start from the listed learning-rate, critic-multiplier, weight-decay, server-rate, clipping, and batch-size factor levels; use a documented fractional/racing design with equal method budgets. | A full Cartesian grid may be wasteful. Runtime may change the design, but Test MSE may not. |
| D14 | unresolved | tuning launch | Freeze training rounds/epochs, validation frequency, early-termination rules, wall-clock cap, and retry policy per method family. | Equal budgets and auditable stopping are needed; demo runtime does not predict full-eICU runtime. |
| D15 | unresolved | tuning selection | Decide whether centralized baselines receive the same absolute compute budget or a method-appropriate equal search-count budget; document the choice. | “Fair” can mean equal wall-clock, candidate count, or gradient evaluations. The chosen rule must precede results. |
| D16 | unresolved | confirmatory launch | Update aggregation-ablation orchestration from linear-only to all three \(g_0\) variants, yielding 30 sample-size rows. | The required protocol matrix is 30 ablation rows; current read-only implementation context schedules only 10. |
| D17 | unresolved | confirmatory launch | Verify centralized checkpoints can be evaluated by hospital and produce equal-client validation/test metrics. | Pooling training data must not silently turn the primary metric into sample-weighted MSE. |
| D18 | unresolved | confirmatory launch | Freeze model architectures for \(g\) and \(f\), initialization, precision, batch handling for small clients, and gradient accumulation. | These are effective-config requirements and can materially affect stability. |
| D19 | unresolved | confirmatory launch | Freeze scenario and configuration artifact locations plus checksum-verification behavior; refuse mismatches rather than regenerating in place. | Paired comparisons require byte-identical scenario artifacts. |
| D20 | unresolved | analysis | Decide the exact optional exploratory uncertainty display (paired bootstrap interval, seed range, or none) before test access. | Five pairs do not support a strong significance claim; the core report remains descriptive. |
| D21 | unresolved | registry integration | Decide whether the main registry gains explicit `protocol_version`, `role`, `aggregation_mode`, `scenario_seed_values`, and `optimizer_seed_values` columns. | The current registry schema cannot represent these Study A distinctions except in `notes`; `proposed_registry_rows.csv` preserves exact compatibility meanwhile. |
| D22 | unresolved | manuscript claim | Approve wording that Study A is a semi-synthetic methods extension with no published numerical target and no clinical-effect claim. | Prevents conflating real hospital structure with a real-outcome causal analysis or paper reproduction. |

## Coordinator sign-off checklist

Before tuning, the coordinator should close D01–D15 and D18–D19 in a versioned
amendment or freeze record. Before confirmatory launch, also close D16–D17.
Before opening confirmatory test outputs, close D20. D21–D22 must close before
registry integration or manuscript claims.

No unresolved item authorizes use of demo results as evidence about the full
eICU cohort.
