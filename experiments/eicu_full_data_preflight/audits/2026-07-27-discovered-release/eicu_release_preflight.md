# eICU release preflight

This is a data-readiness check for cohort construction. It does **not** certify scientific validity, causal identification, or paper alignment.

## Release

- Classification: `demo`
- Classification basis: path contains an explicit eICU demo marker
- Resolved root: `/home/arnav22103/FederatedDeepGMM/physionet.org/files/eicu-crd-demo/2.0.1`
- Detected version: `2.0.1`
- Launchable for demo smoke: `true`
- Launchable for full cohort build: `false`
- Require-full requested: `true`
- Require-full satisfied: `false`
- Streamed patient-table rows: `2520`

## Table status

### Required tables

| Table | File | Header | On-disk size | Uncompressed size |
|---|---|---|---:|---:|
| diagnosis | diagnosis.csv.gz | ok | 352.1 KiB | not safely available |
| infusiondrug | infusiondrug.csv.gz | ok | 394.2 KiB | not safely available |
| lab | lab.csv.gz | ok | 5.5 MiB | not safely available |
| patient | patient.csv.gz | ok | 133.6 KiB | not safely available |
| vitalPeriodic | vitalPeriodic.csv.gz | ok | 18.5 MiB | not safely available |

### Optional tables

| Table | File | Header | On-disk size | Uncompressed size |
|---|---|---|---:|---:|
| admissionDx | admissionDx.csv.gz | ok | 93.6 KiB | not safely available |
| hospital | hospital.csv.gz | ok | 759.0 B | not safely available |
| pastHistory | pastHistory.csv.gz | ok | 139.1 KiB | not safely available |
| vitalAperiodic | vitalAperiodic.csv.gz | ok | 2.5 MiB | not safely available |

### Sensitivity tables

| Table | File | Header | On-disk size | Uncompressed size |
|---|---|---|---:|---:|
| apacheApsVar | apacheApsVar.csv.gz | ok | 66.6 KiB | not safely available |

## Storage

- Known required/optional table bytes on disk: 27.6 MiB
- Available on output filesystem: 10.3 TiB
- Output filesystem probe: `/home/arnav22103/FederatedDeepGMM/experiments/eicu_full_data_preflight`

## Blocking reasons

- Full eICU was required, but this root is not a validated likely-full release ready for cohort construction.

## Warnings

- Exact uncompressed byte sizes are not reported for gzip files because obtaining them safely can require a full decompression pass.
- This is a demo release: it is suitable only for pipeline smoke testing.
- This preflight checks data readiness for cohort construction only; it does not establish scientific validity or paper alignment.
