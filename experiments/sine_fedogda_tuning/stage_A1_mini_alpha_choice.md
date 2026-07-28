# Stage A1-mini Primary Alpha Choice

Alpha was selected using validation metrics only.

Test MSE was not used for alpha selection.

Selection rule: prefer lower mean `last50_validation_mse_mean`; tie-break by lower mean `best_validation_mse`, then lower mean `last50_validation_mse_std`.

| alpha | runs | mean_last50_validation_mse_mean | mean_best_validation_mse | mean_last50_validation_mse_std |
| --- | --- | --- | --- | --- |
| 0.1 | 3 | 0.08528940036 | 0.08366757414 | 8.966225769e-05 |
| 0.5 | 3 | 0.08527845776 | 0.08372549595 | 0.0001124301935 |
| 1 | 3 | 0.08522633048 | 0.08361703112 | 0.0001236532618 |

Selected `primary_alpha = 1.0` because it has the lowest validation-only ranking under the rule above.
