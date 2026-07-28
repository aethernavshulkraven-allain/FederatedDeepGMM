# Step 2: Weakest Curve-Fit Function Diagnosis

## Decision

The weakest current curve-fitting target among the old coauthor plot family functions is **Step**.

This diagnosis uses the Step-1 metric tables plus visual shape. It does not select hyperparameters and does not use Test MSE for tuning.

## Numeric Evidence

Among Absolute, Step, and Linear, current validated FedOGDA-D has the highest mean curve MSE and MAE on Step:

| function | dataset | alpha | pairs | fedogda_curve_mse_wins | mean_fedgda_curve_mse | mean_fedogda_curve_mse | curve_mse_relative_improvement_pct | mean_fedgda_curve_mae | mean_fedogda_curve_mae | mean_fedgda_curve_max_abs | mean_fedogda_curve_max_abs | fedogda_curve_mse_rank_worst_1 | fedogda_curve_mae_rank_worst_1 | fedogda_max_abs_rank_worst_1 | relative_improvement_rank_worst_1 | visual_shape_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step | step | 0.1 | 3 | 3 | 0.0297879823 | 0.02918822 | 2.01343719 | 0.142805218 | 0.141370411 | 0.629348087 | 0.612536285 | 1 | 1 | 1 | 1 | Weakest visual shape: current FedOGDA-D is still almost ramp-like instead of flat two-level step; high MSE/MAE among Abs/Step/Linear. |
| Absolute | abs | 1 | 3 | 2 | 0.01772472 | 0.0164571807 | 7.15125116 | 0.102531716 | 0.0987821462 | 0.500816676 | 0.481766078 | 2 | 2 | 2 | 2 | Moderate issue: captures V-shape but bottom and tails are visibly biased; old committed plot suggests larger possible improvement. |
| Linear | linear | 0.1 | 3 | 2 | 0.00428732767 | 0.0029284127 | 31.6960837 | 0.0442661011 | 0.0371540325 | 0.27765568 | 0.197817038 | 3 | 3 | 3 | 3 | Looks acceptable: low MSE/MAE and shape is close to true linear response. |

## Visual Evidence

- Step: current FedOGDA-D still looks ramp-like and does not reproduce the two flat levels/discontinuity well.
- Absolute: current FedOGDA-D captures the V-shape, but the bottom/tails are biased; tune second if time allows.
- Linear: current FedOGDA-D is already close to the true linear shape; leave as-is unless later metrics show a problem.

## Why Not Sine Here?

Sine has larger absolute curve MSE, but it is already a separate validation-locked tuned A2-lite result. The old committed `All_3*` curve plot family does not include Sine, so the immediate side-by-side improvement target is Step/Absolute/Linear.

## Step 3 Prepared Grid

A launch-ready Step FedOGDA-D mini manifest has been prepared:

- `experiments/curve_fitting_tuning/step_fedogda_d_mini_v1/manifest.csv`
- rows: 12
- output root: `results/curve_fitting_tuning/step_fedogda_d_mini_v1`
- safe to launch without overwrite: `True`

Grid:

```text
dataset: step
method: fedogda_d
seed: 0
alpha: 0.1
T: 200
R: [3, 5, 10]
critic_multiplier: [15, 20]
g_lr: [0.002, 0.005]
critic-side f_lr: derived as critic_multiplier * g_lr, not an independent launcher knob
derived f_lr values: [0.03, 0.04, 0.075, 0.10]
server_lr: 1.5
weight_decay: 0.1
```

Selection rule for later analysis must remain validation-only.
