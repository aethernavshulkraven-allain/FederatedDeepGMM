# Centralized C5b Confirmation Report

## Scope

Confirmed the C5a validation-selected centralized GDA/SGDA recipe on seeds 1 and 2.

Recipe:

```text
g_lr = 0.005
f_lr = 0.03
iterations = 500
batch_size = 0 for gda, 256 for sgda
```

Manifest:

```text
experiments/centralized_baselines/centralized_c5b_confirm_manifest.csv
```

Output root:

```text
results/centralized_lowdim_v1_tuning/c5b_gda_sgda_lr_confirm/
```

## C5b Completion

- Expected C5b runs: 16
- Completed/skipped-valid runs: 16
- Failed/incomplete runs: 0
- Validation pass: 16
- Validation fail: 0
- Wall-clock runtime: 3084.52 sec (51.41 min)

Status counts:

```text
{'completed_valid': 16}
```

Validation counts:

```text
{'pass': 16}
```

All C5b validation checks passed, including centralized config flags, finite histories, predictions/checkpoints, and validation-only best checkpoint selection.

## Final Tuned GDA/SGDA Summary

Seed 0 comes from the C5a selected run. Seeds 1 and 2 come from C5b confirmation. Test MSE is a post-selection readout only.

| dataset | method | seeds | mean test@best | std test@best | mean best val | mean final test | test improvement vs C3 % | best-at-final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | gda | 0 1 2 | 0.056169 | 0.052764 | 0.0545191 | 0.056169 | 81.6 | 3/3 |
| abs | sgda | 0 1 2 | 0.0562582 | 0.0551927 | 0.0546616 | 0.0562582 | 81.49 | 3/3 |
| step | gda | 0 1 2 | 0.0321919 | 0.00542391 | 0.0318329 | 0.0348947 | 44.74 | 1/3 |
| step | sgda | 0 1 2 | 0.0319682 | 0.00573632 | 0.0315825 | 0.0368767 | 45.07 | 1/3 |
| linear | gda | 0 1 2 | 0.00954376 | 0.00761666 | 0.00934649 | 0.0513145 | 88.69 | 0/3 |
| linear | sgda | 0 1 2 | 0.0079973 | 0.00593145 | 0.00780445 | 0.0579672 | 90.67 | 0/3 |
| sin | gda | 0 1 2 | 0.0871174 | 0.00698819 | 0.0855477 | 0.0894263 | 13.86 | 0/3 |
| sin | sgda | 0 1 2 | 0.08768 | 0.00720353 | 0.086022 | 0.0926086 | 13.36 | 0/3 |

Machine-readable outputs:

```text
experiments/centralized_baselines/centralized_c5b_confirm_results.csv
experiments/centralized_baselines/centralized_c5_final_gda_sgda_tuned_results.csv
experiments/centralized_baselines/centralized_c5_final_gda_sgda_tuned_summary.csv
```

## Improvement Over C3

- C5 tuned GDA/SGDA improved validation-selected Test MSE over C3 for all dataset/method pairs: `true`.
- C5 tuned GDA/SGDA improved mean validation MSE over C3 for all dataset/method pairs: `true`.
- Best validation occurred at the final iteration in 8/24 tuned runs.

## Paper-Readiness Readout

The tuned C5 GDA/SGDA results are much stronger than C3 and are validated centralized baselines. They are suitable as tuned centralized GDA/SGDA candidates for reporting, subject to one remaining check: compare against paper target values and decide whether any dataset/method with best-at-final behavior needs optional longer-run confirmation.

OAdam is not merged into the C5 tuned GDA/SGDA table. It remains the C3 OAdam result unless separately tuned.
