# eICU Study A status

Generated: 2026-07-27T00:52:22Z (derived from artifacts, not hand-written)

## Campaign phase and run counts

| study | phase | tuning (done/planned) | final (done/planned) | cohort rows | clients |
|---|---|---|---|---|---|
| Study A v1 (archived pipeline validation) | **final_complete** | n/a (skipped by design) | 105/105 | 9 (after 2 collapses; 201 after clinical gates only) | 3 |
| Study A v2 (eicu_study_a_v2_offhours) | **tuning_complete** | 36/36 | 0/105 (not materialized) | 2031 | 179 |

## Study A v2 cohort numbers (read from cohort_metadata.json)

- admissions: **2031**
- hospital clients: **179**
- split (train/dev/test): **1420/306/305** ("dev" is canonical; some write-ups say "Validation" for the identical split)
- off-hours rate: **0.5500**

## Study A v2 client-size heterogeneity

- rows per client: min **7**, median **11**, max **26**; **14** clients have fewer than 10 rows
- candidate hospitals **186** -> eligible **179** (7 dropped)
  - `fewer_than_min_client_rows`: 5
  - `missing_train_dev_or_test`: 1
  - `no_training_off_hours_variation`: 3
  - `no_within_hospital_off_hours_variation`: 3

## Cohort funnels (clinical gates vs instrument-variation gates)

- v1 clinical gates: 2520 -> **201** rows (89 hospitals)
- v1 instrument-variation gates: 201 rows / 89 hospitals -> **9 rows / 3 hospitals** ([184, 243, 407])
- v2: 2520 -> **2031 rows / 179 clients**. v2 applies no clinical requirement (no sepsis / mortality / vasopressor / infusion-interface gate) by protocol design, so there is no v1-style clinical-gate collapse; the only collapse is linkage (one hospital per patient) plus the instrument / min-rows / split-eligibility gate.

## Notes

- `setup_validation_summary.json`'s `full_test_suite` counts are a frozen snapshot from campaign setup time, not the live suite size; do not compare them to a fresh test run.
- This report is a live read of the tree. If a tuning or final campaign is running concurrently, re-running this script will show a different, more complete count.
