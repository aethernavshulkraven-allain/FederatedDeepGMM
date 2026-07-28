# Fixed-abs high-dimensional execution status

The data and code path are ready: all six datasets are certified, all 12 deterministic representative checks pass, all four stochastic end-to-end smokes pass, and the 96-row tuning queue dry-runs with every row launchable.

GPU access is provided through the institute `gpurun` scheduler. A full 150-round stochastic pilot passed on an H100, and all four method-specific tuning queues have been submitted through the scheduler.

No tuning candidate will be promoted until all 96 tuning artifacts are complete and validated. This preserves the validation-only selection rule. Final evaluation will use seeds 0–4, giving 120 final runs. Commands and scientific gates are recorded in [RUNBOOK.md](RUNBOOK.md).
