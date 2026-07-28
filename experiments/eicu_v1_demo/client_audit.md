# eICU federated IV — client feasibility audit

- cohort rows: **201**
- hospitals: **89**
- wards: **97**
- treated (early vasopressor): **32**
- in-hospital deaths: **33**

## Decision

**construction = `insufficient_data`**

- ward-eligible hospitals: 0 (need >= 5); eligible hospital groups: 0 (need >= 5)
- no construction has enough clients with within-client instrument variation; this release cannot support the federated IV analysis

Thresholds (pre-registered, frozen before any effect estimate):

| threshold | value |
|---|---|
| `min_deaths` | 0 |
| `min_patients` | 200 |
| `min_per_ward` | 50 |
| `min_treated` | 20 |
| `min_untreated` | 20 |
| `min_wards` | 2 |

## Per-hospital distribution

| statistic | value |
|---|---|
| hospitals | 89 |
| patients per hospital (median) | 2 |
| patients per hospital (max) | 6 |
| hospitals with >1 ward | 8 |
| hospitals meeting client eligibility | 0 |

## Instrument variation probe

`structural sd` is the between-unit (ward, or hospital in the grouped fallback) spread of the instrument within a client. `raw sd` additionally contains cross-fitting fold noise, which is not identifying variation and is excluded from the decision.

| construction | clients | with structural variation | mean structural sd | mean raw sd |
|---|---|---|---|---|
| ward | 89 | 3 | 0.0092 | 0.0635 |
| hospital | 21 | 14 | 0.0156 | 0.0258 |

## Consequence

This release cannot back a reported estimate. It is usable only as a pipeline harness: the ETL, instrument construction, splitting and training code can be exercised end to end, but every number produced from it is a smoke-test artefact, not a result.

