# eICU Study A campaign validation

- Phase: `prelaunch`
- Contract version: `2.1.0-study-a-v2-offhours`
- Manifest rows: 105
- Fixed rows: 105 / 105
- Result directories found: 0
- Results validated: 0
- Valid completed results: 0
- Blocking errors: 0
- Warnings: 0
- **launchable: `true`**

## Role coverage

| Role | Observed | Expected | Complete | g0 | Seeds | Methods |
|---|---:|---:|:---:|---|---|---|
| aggregation_ablation | 30 | 30 | yes | interaction, linear, mlp | 101, 102, 103, 104, 105 | fedgda_s, fedogda_s |
| centralized_baseline | 45 | 45 | yes | interaction, linear, mlp | 101, 102, 103, 104, 105 | gda_d, oadam_s, sgda_s |
| confirmatory | 30 | 30 | yes | interaction, linear, mlp | 101, 102, 103, 104, 105 | fedgda_s, fedogda_s |
| smoke | 0 | variable | yes | — | — | — |
| tuning | 0 | variable | yes | — | — | — |

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
