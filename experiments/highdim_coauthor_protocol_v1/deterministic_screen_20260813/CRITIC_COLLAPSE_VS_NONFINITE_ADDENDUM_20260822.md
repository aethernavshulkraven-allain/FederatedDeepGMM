# Critic collapse vs. nonfinite critic output — 2026-08-22

`PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md` §0 says "critic collapse is
diagnostic-only" and that critic-collapse reporting does not independently
change promotion. Read on its own, that sentence is easy to over-generalize
into "nothing about the critic ever gates promotion." That is false, and
was already false in the code this amendment describes — this addendum
names the distinction precisely rather than changing either rule.

## The two things "critic" trouble can mean here

**Critic collapse** (what §0 actually means): the `_x` diagnostic
(`PSI_X_SCENARIO_DIAGNOSTIC_20260818.md`) observed the critic `f` producing
near-constant output across inputs — a finite, numerically valid, but
degenerate value. No threshold for "how constant is too constant" is
frozen, so this pattern cannot and does not gate anything by itself. §0's
rule stands exactly as written for this case.

**Nonfinite critic output** (a different, more severe failure): each
training round, `fedavg_api.py`'s `train()` loop checks
`critic_outputs_finite = metric_values_are_finite((f_of_z_train, f_of_z_dev))`
— whether the critic's raw output tensors contain NaN/Inf at all. If they
do, the round is marked `diverged`, recorded in `nonfinite_diagnostics`
(with `critic_outputs_finite: False` in that round's diagnostic dict), and
`nonfinite_diagnostics` being non-empty forces `terminal_ineligible=True` in
`run_manifest.py`'s `validate_artifacts` — sticky for the whole run, exactly
like any other nonfinite evidence. This is intentional: no metric computed
downstream of a critic outputting NaN/Inf can be trusted, regardless of
whether every other quantity that round stayed finite.

## What this addendum changes

Nothing about behavior. Both rules already existed in the code exactly as
described above; this is a documentation fix, not a policy fix. The
stricter rule (nonfinite critic output → terminal-ineligible) is correct
and is being kept. What changes is that "critic collapse is diagnostic-only"
in §0 should be read narrowly, as it was intended: it covers finite-but-
degenerate critic output, not a critic that has gone numerically nonfinite.

`tests/test_run_manifest_resume_safety.py`'s
`test_nonfinite_critic_output_alone_is_terminal_ineligible` locks in the
nonfinite-critic-output side of this distinction directly against
`validate_artifacts`, using the exact diagnostic-dict shape
`fedavg_api.py` produces.
