# Study A metric policy

## Primary metric

The primary metric is **equal-client structural Test MSE at the
equal-client-validation-selected checkpoint**. For test clients
\(\mathcal C_T\), client \(i\)'s test set \(T_i\), and fixed selected checkpoint
\(\hat g^\star\),

\[
m_i^{test}=
\frac{1}{|T_i|}\sum_{j\in T_i}
\left[\hat g^\star(D_{ij},W_{ij})-g_0(D_{ij},W_{ij})\right]^2,
\]

\[
\operatorname{MSE}_{EC}^{test}
=\frac{1}{|\mathcal C_T|}\sum_{i\in\mathcal C_T}m_i^{test}.
\]

This value is stored as
`equal_client_test_mse_at_best_validation` and as the required compatibility
field `test_mse_at_best_validation`. For Study A, the compatibility field must
have equal-client semantics; a sample-weighted value may not be written under
that name.

The checkpoint is chosen using validation data only:

\[
r^\star=\arg\min_r
\frac{1}{|\mathcal C_V|}\sum_{i\in\mathcal C_V}
\frac{1}{|V_i|}\sum_{j\in V_i}
\left[\hat g_r(D_{ij},W_{ij})-g_0(D_{ij},W_{ij})\right]^2.
\]

Exact ties use the earlier round. Test arrays and test summaries are forbidden
inputs to configuration or checkpoint selection.

## Equal-client versus sample-weighted aggregation

For any per-client scalar metric \(q_i\), with \(n_i\) evaluated rows,

\[
Q_{EC}=\frac{1}{N}\sum_{i=1}^{N}q_i,
\qquad
Q_{SW}=\frac{\sum_{i=1}^{N}n_iq_i}{\sum_{i=1}^{N}n_i}.
\]

Equal-client aggregation gives every hospital the same weight. Sample-weighted
aggregation gives every patient row the same weight. Study A's primary
estimand, validation selector, federated aggregation, and primary report use
equal-client weighting. Sample-weighted metrics are secondary even for the
sample-size aggregation ablation.

Only clients eligible under the frozen cohort policy and represented in the
relevant split enter that split's aggregate. The denominator and client IDs
must be recorded; silently dropping a client is invalid.

## Structural MSE

Structural MSE compares \(\hat g(D,W)\) with known \(g_0(D,W)\), not with noisy
outcome \(Y\). Report:

- per-client structural MSE;
- equal-client structural MSE;
- sample-weighted structural MSE;
- mean, population standard deviation, median, minimum, and maximum across
  confirmatory seed pairs; and
- best-validation-checkpoint and final-checkpoint values.

## ATE error and individual-effect MAE

For row \(j\) in client \(i\), define true and estimated individual structural
effects

\[
\delta_{ij}^0=g_0(1,W_{ij})-g_0(0,W_{ij}),\qquad
\hat\delta_{ij}=\hat g(1,W_{ij})-\hat g(0,W_{ij}).
\]

Client-specific ATEs are

\[
\Delta_i^0=\frac{1}{n_i}\sum_j\delta_{ij}^0,\qquad
\hat\Delta_i=\frac{1}{n_i}\sum_j\hat\delta_{ij}.
\]

The equal-client absolute ATE error is

\[
\left|
\frac{1}{N}\sum_i\hat\Delta_i-
\frac{1}{N}\sum_i\Delta_i^0
\right|,
\]

and the sample-weighted absolute ATE error is

\[
\left|
\frac{\sum_i n_i\hat\Delta_i}{\sum_i n_i}-
\frac{\sum_i n_i\Delta_i^0}{\sum_i n_i}
\right|.
\]

Also report the distribution of per-client absolute ATE errors
\(|\hat\Delta_i-\Delta_i^0|\).

Absolute ATE error is not individual-effect MAE. The latter is

\[
\operatorname{ITE\_MAE}_{EC}
=\frac{1}{N}\sum_i\frac{1}{n_i}\sum_j
|\hat\delta_{ij}-\delta_{ij}^0|,
\]

with a sample-weighted analogue. ATE error can be small because positive and
negative individual errors cancel; individual-effect MAE cannot. The two must
have distinct field names and may not be substituted for one another.

## Held-out moment violation

At a fixed checkpoint, for split \(S_i\), define the empirical client moment

\[
\hat\mu_i=
\frac{1}{|S_i|}\sum_{j\in S_i}
f_{\hat\tau}(Z_{ij},W_{ij})
\{Y_{ij}-g_{\hat\theta}(D_{ij},W_{ij})\}.
\]

The client moment-violation metric is
\(\|\hat\mu_i\|_2^2\). Report its equal-client and sample-weighted aggregates
and per-client distribution. Validation/test rows are evaluation-only: neither
\(g\) nor \(f\), instrument fits, preprocessing, or hyperparameters may be
updated from them.

Moment violation is a secondary diagnostic and a tuning tie-breaker. It does
not replace equal-client validation structural MSE as the primary checkpoint
selector in Study A.

## Stability metrics

For both validation and test where meaningful, retain:

- `final_validation_mse`;
- `final_test_mse`;
- `final_vs_best_validation_gap`, defined as final minus best equal-client
  validation structural MSE;
- `final_vs_best_test_gap`, defined as final-checkpoint minus
  best-validation-checkpoint equal-client test structural MSE;
- standard deviation and range of equal-client validation structural MSE over
  the last `min(50, number_of_recorded_rounds)` validation points;
- best and final held-out moment violation;
- per-round metric curves; and
- wall-clock runtime, rounds completed, peak memory when available, and
  hardware description.

A negative final-versus-best gap is possible only when checkpoint evaluation
frequency or numerical ties make “best” inconsistent; it must be investigated,
not clipped to zero.

## Divergence and run failures

`diverged: true` is reserved for a NaN or infinite model parameter or required
metric. A large but finite MSE, a bad ATE, oscillation, or losing to another
method is not divergence.

OOM, missing input, checksum mismatch, invalid configuration, timeout,
scheduler failure, or interruption is a run failure with an explicit
`failure_reason`. Report failures and divergence separately. Finite outliers
remain in summaries and per-seed tables.

## Selection firewall

The following are prohibited:

- selecting a hyperparameter using Test MSE, test ATE, test moment violation,
  or any test-client statistic;
- selecting a checkpoint using a test metric;
- changing a seed, scenario, grid, budget, or stopping rule after viewing
  confirmatory test results;
- reporting final-iterate Test MSE under the name
  `test_mse_at_best_validation`; or
- replacing equal-client primary MSE with sample-weighted MSE.

Test metrics are computed and reported only after the hyperparameter
configuration and best-validation checkpoint are fixed.
