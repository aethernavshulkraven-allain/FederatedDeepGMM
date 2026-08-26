# High-Dimensional Psi Adjudication V3 Correction Addendum

## Status

The v2 launcher was stopped on 2026-08-22 before it entered the X stage. Its
completed and partial artifacts remain in their original v2 namespaces. The
active partial run had reached round 7 of 500; it was not deleted or promoted.

V2 is not resumable as scientific evidence after this correction. The server
state transition changed, so v2 candidate states and the older deterministic
finals are not mixed with post-fix states.

The v3 signal/X packet is also superseded before launch. Although its 99 runs
would have started from fresh initialization, its candidate shortlist was
selected from pre-fix screening trajectories. Its combined launcher now fails
closed. The strict replacement is a fresh 108-run corrected image screen,
followed by a newly frozen v4 adjudication packet.

## Numerical Fix

The old server update applied interpolation or optimistic extrapolation to the
entire model `state_dict`. For a critic containing BatchNorm, that included
`running_var` and `num_batches_tracked`. FedOGDA could therefore extrapolate a
valid running variance below zero, causing a nonfinite evaluation even though
the next training round could repair the buffer.

V3 applies server optimizer arithmetic only to keys returned by
`named_parameters()`. Floating BatchNorm buffers use direct weighted client
aggregation, and nonfloating counters use a deterministic maximum policy;
counters are never averaged or extrapolated. The coordinator checks every
BatchNorm running variance after each server transition and records the
minimum observed values in the round curve and final metrics.

The replacement policy is frozen as:

```text
server_buffer_policy = direct_client_aggregate
```

## Artifact And Scoring Fixes

- A complete terminal-divergent run is immutable and resolves as
  `terminal_ineligible`; resume never deletes or reruns it.
- A genuinely partial directory is moved to `_interrupted_attempts` before a
  fresh attempt. Complete malformed artifacts block the queue and remain
  untouched.
- Launcher summaries are written atomically. An append-only JSONL ledger keeps
  invocation, start, archive, and resolution events across resumes.
- Validation requires exact config identity, exactly `comm_round` curve rows,
  and ordered round indices from zero.
- Adjudication requires exactly 500 ordered rounds for all three seeds of every
  planned candidate. Missing, corrupt, short, long, or mismatched work makes
  the cell incomplete and produces no winner.
- Sticky stability uses the full 500-round curve, not only the last-50 window.
  The last 50 rounds are used only after full-run eligibility is established.
- An exact fallback median-MSE tie is reported unresolved; manifest order is
  not a tie breaker.
- The launcher has a hard stage barrier. X work cannot start unless every
  signal manifest row has exactly one resolved launcher status and every
  signal cell has a frozen promotion. `retune_required`, an exact fallback
  MSE tie, or incomplete evidence blocks the transition.

## Preflight Result And Replacement Launch Order

The exact flagged FEMNIST-Z FedOGDA seed/config completed all 120 rounds under
the corrected policy. It had finite critic outputs, no divergence, and minimum
critic BatchNorm running variance `9.897330425614937e-07`. The result and its
six required artifacts are recorded in
`bn_buffer_diagnostic_certification.json`.

No v3 signal or X run is eligible to launch. The replacement order is:

1. run and score the full corrected image screen;
2. resolve any frozen boundary-review cells;
3. generate the all-fresh v4 packet;
4. run signal only and require a valid promotion in every signal cell; and
5. launch X separately.
