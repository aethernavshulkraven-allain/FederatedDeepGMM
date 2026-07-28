# A2-lite Selection Audit

**Selection was validation-only: `true`.**

## Evidence

- A1 ranking source: `experiments/sine_fedogda_tuning/stage_A1_mini_top_candidates.md` explicitly says candidate ranking used validation metrics only.
- A2-lite lock source: `experiments/sine_fedogda_tuning/stage_A2_lite_decision.md` records that Test MSE was not used to choose the candidate.
- Analyzer: `scripts/analyze_sine_stage_a1_mini.py` ranks by divergence/finite status, last-50 validation mean, best validation MSE, validation CV, and validation range.
- All three effective configs and metrics files set `selection_metric_source = validation` and `test_mse_used_for_selection = false`.
- For every seed, `best_validation_round` exactly equals the argmin of the stored per-round validation MSE.
- Test MSE was logged for transparency and evaluated only after the recipe was locked.

## Conclusion

The locked recipe and best checkpoints were selected without Test MSE.
