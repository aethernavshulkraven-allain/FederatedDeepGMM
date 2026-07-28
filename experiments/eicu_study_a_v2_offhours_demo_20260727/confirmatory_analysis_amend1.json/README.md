# Study A v2 confirmatory results

Primary endpoint: equal-client Test structural MSE at the validation-selected checkpoint. Paired FedOGDA-minus-FedGDA differences by (scenario seed, optimizer seed); 5 pairs per g0, 15 pooled. Decision rule is pre-declared in the module docstring and reproduced below -- no p-value is computed; n=5 does not support one.

> A scope (one g0, or the 15-pair pooled set) is 'favored' for a method only if that method is better in >= 4/5 paired seeds for a single g0 (or >= 12/15 pooled) AND the mean paired difference agrees in sign. No p-value is computed. Pre-declared in this module's docstring; not settable via CLI flag.

## g0 = `interaction`

| method | seeds ok | diverged | primary Test MSE @ best-val (mean +/- std) |
|---|---|---|---|
| fedgda_s | 5/5 | 0 | 0.4218 +/- 0.074 |
| fedogda_s | 5/5 | 0 | 0.6069 +/- 0.11 |

### Paired FedOGDA - FedGDA, g0=`interaction`

- matched seeds: [1101, 1102, 1103, 1104, 1105]
- per-seed differences: [0.2558, 0.2185, 0.262, 0.1237, 0.0653]
- mean difference: 0.1851
- FedOGDA better in 0/5 seeds (needs >= 4 for a verdict)
- **verdict: fedgda_favored**

## g0 = `linear`

| method | seeds ok | diverged | primary Test MSE @ best-val (mean +/- std) |
|---|---|---|---|
| fedgda_s | 5/5 | 0 | 0.2695 +/- 0.057 |
| fedogda_s | 5/5 | 0 | 0.3607 +/- 0.044 |

### Paired FedOGDA - FedGDA, g0=`linear`

- matched seeds: [1101, 1102, 1103, 1104, 1105]
- per-seed differences: [0.0483, 0.146, 0.1084, 0.0752, 0.0778]
- mean difference: 0.09114
- FedOGDA better in 0/5 seeds (needs >= 4 for a verdict)
- **verdict: fedgda_favored**

## g0 = `mlp`

| method | seeds ok | diverged | primary Test MSE @ best-val (mean +/- std) |
|---|---|---|---|
| fedgda_s | 5/5 | 0 | 0.2257 +/- 0.055 |
| fedogda_s | 5/5 | 0 | 0.2865 +/- 0.066 |

### Paired FedOGDA - FedGDA, g0=`mlp`

- matched seeds: [1101, 1102, 1103, 1104, 1105]
- per-seed differences: [0.1043, 0.0402, 0.0516, 0.0578, 0.05]
- mean difference: 0.06081
- FedOGDA better in 0/5 seeds (needs >= 4 for a verdict)
- **verdict: fedgda_favored**

## Pooled (15 pairs across all g0)

- FedOGDA better in 0/15 pooled pairs
- mean difference: 0.1123
- **pooled verdict: fedgda_favored**

