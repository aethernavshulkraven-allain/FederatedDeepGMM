# Final Curve-Fitting Plot Tuning Contract

Geetika confirmed that the four low-dimensional curve-fitting plots should include:

```text
Actual Causal Effect
DeepGMM-OAdam
DeepGMM-GDA
DeepGMM-SGDA
FedDeepGMM-GDA
FedDeepGMM-SGDA
FedDeepGMM-OGDA-D
FedDeepGMM-OGDA-S
```

## Method Sources

| plot label | source family | current status |
| --- | --- | --- |
| Actual Causal Effect | saved true `g`/response in prediction artifacts | available |
| DeepGMM-OAdam | true centralized C3 | completed and validated for all four functions |
| DeepGMM-GDA | true centralized C5 tuned GDA | completed and validated for all four functions |
| DeepGMM-SGDA | true centralized C5 tuned SGDA | completed and validated for all four functions |
| FedDeepGMM-GDA | federated rerun protocol baseline | completed for all four functions |
| FedDeepGMM-SGDA | federated rerun protocol baseline | completed for all four functions |
| FedDeepGMM-OGDA-D | federated optimistic deterministic/local-full-batch variant | completed baseline; Sine tuned; Step reproduction in progress |
| FedDeepGMM-OGDA-S | federated optimistic stochastic variant | completed baseline; Step reproduction in progress; Sine still needs tuning |

## Current Risk By Function

| function | main risk | action |
| --- | --- | --- |
| Absolute | baseline curves already usable; centralized methods available | no urgent tuning |
| Linear | baseline curves already usable; centralized methods available | no urgent tuning |
| Step | current FedOGDA curves are visually/metric-wise weaker than old tuned commit | reproduce Geetika old Step recipe family first |
| Sine | FedOGDA-D is tuned and wins; FedOGDA-S currently loses all stochastic pairs | tune Sine FedOGDA-S after Step-1 |

## Active Step Action

Stage Step-1 old-recipe reproduction was launched from:

```text
experiments/curve_fitting_tuning/step_geetika_repro_v1/manifest.csv
```

Output root:

```text
results/curve_fitting_tuning/step_geetika_repro_v1/
```

Launcher log:

```text
experiments/curve_fitting_tuning/step_geetika_repro_v1/launcher_20260715_212223_1gpu.log
```

This stage contains four seed-0 rows:

| row | purpose |
| --- | --- |
| FedOGDA-S old minibatch Step recipe | reproduce Geetika's strongest stochastic OGDA neighborhood |
| FedGDA-S old minibatch Step recipe | matched stochastic baseline |
| FedOGDA-D partial-client full-batch Step recipe | reproduce old "OGDA-FB" curve; note only 10 clients per round |
| FedGDA-D partial-client full-batch Step recipe | matched partial-client full-batch baseline |

Selection remains validation-only. Test MSE and curve metrics are post-selection readouts.

## Next Sine Action

Sine deterministic FedOGDA-D is already validation-locked and positive:

```text
FedOGDA-D mean test@best validation = 0.080011535
FedGDA-D mean test@best validation = 0.086106863
FedOGDA-D wins 3/3 seeds
```

The next Sine tuning target is `FedDeepGMM-OGDA-S`.

Recommended Sine-SOGDA seed-0 screen:

```text
dataset = sin
alpha = 1.0
client_num_in_total = 1000
client_num_per_round = 10
batch_size = 256
T = 500
R = [3, 7]
g_lr = [0.0005, 0.001, 0.002]
weight_decay = [0.02, 0.05, 0.1]
critic_multiplier = [10, 15]
server_lr = [1.0, 1.5]
```

For ASAP use, reduce this to:

```text
R = [3, 7]
g_lr = [0.0005, 0.001, 0.002]
weight_decay = [0.02, 0.1]
critic_multiplier = [15]
server_lr = [1.5]
```

That is 12 rows. Rank by validation only.

## Reporting Rule

The final plot should show the validation-selected prediction for each tuned method. If a method is not tuned, use the validated baseline artifact and label it as baseline-derived in the plot build notes.
