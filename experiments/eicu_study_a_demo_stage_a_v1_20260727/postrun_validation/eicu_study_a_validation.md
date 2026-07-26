# eICU Study A campaign validation

- Phase: `postrun`
- Contract version: `2.0.0`
- Manifest rows: 105
- Fixed rows: 105 / 105
- Result directories found: 105
- Results validated: 105
- Valid completed results: 105
- Blocking errors: 0
- Warnings: 0
- **reportable: `true`**

## Role coverage

| Role | Observed | Expected | Complete | g0 | Seeds | Methods |
|---|---:|---:|:---:|---|---|---|
| confirmatory | 30 | 30 | yes | interaction, linear, mlp | 101, 102, 103, 104, 105 | fedgda_s, fedogda_s |
| centralized_baseline | 45 | 45 | yes | interaction, linear, mlp | 101, 102, 103, 104, 105 | gda_d, oadam_s, sgda_s |
| aggregation_ablation | 30 | 30 | yes | interaction, linear, mlp | 101, 102, 103, 104, 105 | fedgda_s, fedogda_s |
| tuning | 0 | variable | yes | — | — | — |
| smoke | 0 | variable | yes | — | — | — |

## Pairing audit

- groups checked: 30
- violations: 0

## Provenance audit

- configs checked: 0
- scenarios checked: 105
- artifact checksums recomputed: 105
- violations: 0

## Selection-policy audit

- rows checked: 105
- violations: 0

## Blocking errors

None.

## Warnings

None.
