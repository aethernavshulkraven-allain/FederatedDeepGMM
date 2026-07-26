# Full eICU data preflight

This directory defines and documents a read-only preflight for an eICU-CRD
release before cohort construction. It answers a narrow operational question:
does the supplied root have the tables and headers needed to build the current
cohort, and is it recognizably the demo or likely the full release?

It does **not** inspect patient values, certify the cohort, establish instrument
validity, or claim that an experiment matches the paper. A passing full preflight
means only **data ready for cohort build**.

## Data contract

[`required_tables.json`](required_tables.json) was derived from the table reads
and direct column accesses in `scripts/prepare_eicu_cohort.py`.

- Required: `patient`, `diagnosis`, `infusiondrug`, `lab`, `vitalPeriodic`.
- Optional enrichment: `hospital`, `admissionDx`, `vitalAperiodic`,
  `pastHistory`.
- Sensitivity-only: `apacheApsVar`, used by the explicit APACHE sensitivity
  arm.

Missing a required table or required column is blocking. Missing optional or
sensitivity data is reported as a warning and does not prevent the primary
cohort build.

## Usage

Use the project environment:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python \
  scripts/preflight_eicu_release.py \
  --eicu-root /path/to/eicu-crd/2.0 \
  --out experiments/eicu_full_data_preflight/run \
  --require-full
```

The root must be the directory that directly contains files such as
`patient.csv.gz`, `diagnosis.csv.gz`, and `vitalPeriodic.csv.gz`. Filenames are
matched case-insensitively and may be compressed or uncompressed CSVs.

Useful opt-in checks:

```bash
# Stream records without loading the patient table into memory.
... --count-patient-rows

# Hash every resolved table file. This can take substantial time on full eICU.
... --checksum

# Check and print JSON to stdout without creating the output directory.
... --dry-run
```

Without `--dry-run`, the tool writes:

- `eicu_release_preflight.json` for machines;
- `eicu_release_preflight.md` for review.

Both outputs contain file metadata and header status, never patient rows or
patient-level values. Exact uncompressed sizes for gzip files are left
unreported because safely determining them can require a full decompression
pass. On-disk compressed sizes are always reported when file metadata is
readable.

## Classification and failure behavior

An obvious path containing `eicu-crd-demo` is classified as `demo`. A standard
non-demo `eicu-crd/<version>` path, or a streamed patient-table count of at
least 100,000 records, is classified as `likely_full`. Other roots are
`unknown`; a complete unknown root can support a smoke run but is not enabled
for a claimed full cohort build.

`--require-full` exits nonzero for `demo`, `unknown`, missing required tables,
or missing required columns. Demo table completeness does not override the
demo classification.

## What must be mounted for Study A

Provide the credentialed full eICU-CRD release locally; this repository does not
download it. Mount or expose the release directory read-only if desired, with
the original table files directly beneath one root (conventionally
`eicu-crd/2.0`). The process needs read/traverse permission on that root and
write space only on the output filesystem. No credentials should be placed in
the repository or passed to this preflight.

After a passing `--require-full` preflight, retain the JSON/Markdown provenance
with the run artifacts and run cohort construction as a separate step. A later
cohort audit and scientific validation are still required.
