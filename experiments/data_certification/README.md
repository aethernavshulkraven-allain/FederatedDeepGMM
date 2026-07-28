# Low-dimensional synthetic-data certification

This certification compares array contents rather than relying only on `.npz` file hashes. 
An NPZ container can differ because of ZIP metadata or layout while every stored array is identical.

## Method

Candidates were generated twice with the repository's own `generate_zoo_data.create_dataset` helper, 
resetting its documented NumPy seed `527` for every dataset. Candidates are stored only under 
`results/_data_certification/`; no file under `data/zoo/` is written or modified.

Array checksums hash the key name, shape, dtype, and contiguous bytes. The code uses 
`np.random.normal(0, 0.1, ...)`; NumPy's second positional parameter is the standard deviation, 
so the code implements standard deviation 0.1 (variance 0.01).

## Standardization

The generator's `Standardizer` captures mean/std from the first generated (train) Y split and applies 
that transform to both Y and stored true-g for train, dev, and test before writing the NPZ file.

## Decision

Overall decision: `blocked`.

### Absolute

- Legacy comparison: `content_match_file_container_differs`
- Generator deterministic: `true`
- Legacy reusable for paper-aligned replication: `false`
- Paper-function max error: `0.0`
- Selected data path: `none (blocked)`
- Notes: generator_dgp_differs_from_paper_expected;legacy_file_is_exactly_reproducible_from_current_generator

### Step

- Legacy comparison: `content_match_file_container_differs`
- Generator deterministic: `true`
- Legacy reusable for paper-aligned replication: `false`
- Paper-function max error: `0.6339470630742816`
- Selected data path: `none (blocked)`
- Notes: generator_dgp_differs_from_paper_expected;stored_true_g_does_not_match_paper_true_function;legacy_file_is_exactly_reproducible_from_current_generator

### Linear

- Legacy comparison: `content_match_file_container_differs`
- Generator deterministic: `true`
- Legacy reusable for paper-aligned replication: `false`
- Paper-function max error: `0.0`
- Selected data path: `none (blocked)`
- Notes: generator_dgp_differs_from_paper_expected;legacy_file_is_exactly_reproducible_from_current_generator

## Blocker

The current author generator does not implement the requested paper DGP: it is configured with 
`two_gps=False`, so X omits Z2, and it uses a `2 * confounder` term in Y. Its Step function is 
also `1` below zero and `2.5` at/above zero rather than `1{x >= 0}`. Therefore the legacy files may 
be exactly reproducible author-code artifacts, but they are not certified for paper-aligned reuse.
No `data/paper_v1/` replacement was created.
