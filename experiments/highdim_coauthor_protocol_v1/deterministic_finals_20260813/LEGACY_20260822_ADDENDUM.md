# Legacy addendum — 2026-08-22

`RELABEL_20260818.md` (this directory) says the 108 runs here "remain valid
data under their correct label" and that, once Rank/Confirm adjudication
picks a winner per cell, "this directory's existing runs for unchanged cells
are reused, not repeated." That reuse plan is now superseded.

## Why

All 108 runs in this directory were produced under the same buggy
federated-server update described in
`../psi_adjudication_20260822_v3/CORRECTION_ADDENDUM_20260822.md`: server
learning-rate interpolation / OGDA optimistic extrapolation was applied to
the entire model `state_dict`, including BatchNorm `running_var` and
`num_batches_tracked`, not just trainable parameters. Every FEMNIST/CIFAR10
critic in this campaign has BatchNorm, so this directory's runs are
contaminated by the same mechanism as the v2 adjudication signal manifest,
not just the runs that visibly diverged.

**Actual GPU cost of the 108 runs remains 66.66 GPU-hours as measured**
(unchanged fact, `RELABEL_20260818.md` section "Corrected figures") — that
number is not being disputed, only the runs' eligibility for reuse.

## What changes

The "existing runs for unchanged cells are reused, not repeated" plan in
`RELABEL_20260818.md` does not apply. The post-fix confirmation campaign
(`../psi_adjudication_20260822_v3/`) reruns every planned candidate and
every seed from initialization for all 12 cells — `reused_from_finals` is
`False` for every candidate in both its signal and `_x` manifests, verified
directly against the generated summaries.

This directory's 108 runs are kept exactly as `RELABEL_20260818.md`
describes (not discarded, overwritten, or deleted) — they remain available
as the historical "three-seed exploratory finals using seed-0/150-round-MSE
selected configurations" record. They are additionally now ineligible as
adjudication evidence or as a reuse source for any future finals rerun.
