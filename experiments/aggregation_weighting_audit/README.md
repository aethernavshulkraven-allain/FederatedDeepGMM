# Aggregation-weighting provenance audit

This is a **labeling pass only** -- nothing was rerun, retrained, or modified. It exists to make explicit which existing results used the pre-fix, unconditional sample-size-weighted aggregation before any scaled training resumes.

Total runs discovered: **953**

## Counts by label

| label | count | meaning |
|---|---|---|
| `legacy_sample_weighted` | 953 | predates this fix; used the old, unconditional sample-size weighting |

## By result family

| family | legacy_sample_weighted | no_effective_config | sample_size_explicit | uniform_clients | unrecognized_value |
|---|---|---|---|---|---|
| `_failed` | 4 | 0 | 0 | 0 | 0 |
| `_golden` | 1 | 0 | 0 | 0 | 0 |
| `_profiling` | 8 | 0 | 0 | 0 | 0 |
| `_smoke` | 4 | 0 | 0 | 0 | 0 |
| `abs` | 13 | 0 | 0 | 0 | 0 |
| `centralized_lowdim_v1` | 36 | 0 | 0 | 0 | 0 |
| `centralized_lowdim_v1_smoke` | 3 | 0 | 0 | 0 | 0 |
| `centralized_lowdim_v1_smoke_tiny` | 3 | 0 | 0 | 0 | 0 |
| `centralized_lowdim_v1_tuning` | 56 | 0 | 0 | 0 | 0 |
| `curve_fitting_tuning` | 193 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1` | 144 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_real_images_abs_alpha0p1_tuning` | 24 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_real_images_abs_alpha0p5` | 35 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_real_images_abs_alpha0p5_tuning` | 55 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_real_images_abs_alpha1` | 11 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_real_images_abs_alpha1_tuning` | 24 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_real_images_abs_remaining_safe_speedup_v1_20260719_123539` | 84 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_real_images_abs_remaining_safe_speedup_v2_20260719_233531` | 77 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_smoke` | 1 | 0 | 0 | 0 | 0 |
| `rerun_protocol_v1_tuning` | 144 | 0 | 0 | 0 | 0 |
| `sine_fedogda_tuning` | 15 | 0 | 0 | 0 | 0 |
| `stability_probe_v1_20260722` | 16 | 0 | 0 | 0 | 0 |
| `step` | 2 | 0 | 0 | 0 | 0 |

## Consequence for scaled training

Every run labeled `legacy_sample_weighted` was produced under sample-size-weighted aggregation, silently, before this option existed. **None of the old experiment matrix has been automatically rerun.** Whether to rerun any of it under `uniform_clients` is a separate, deliberate decision -- this audit only establishes the current provenance so that decision can be made with full information.

