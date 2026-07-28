# eICU instrument diagnostics

> **Pipeline artefact.** The Stage-1 audit reports `insufficient_data` for this release: no client meets the pre-registered eligibility thresholds. The numbers below verify that the code runs end to end. They are not estimates.

## Construction: `ward`

- rows: 201, clients: 89
- covariates in the design: 32
- clients with structural instrument variation: 3
- treatment rate: 0.159
- mortality rate: 0.164

### Relevance (first stage)

| quantity | value |
|---|---|
| instrument coefficient | 0.1810 |
| robust SE | 0.1112 |
| partial F | 3.29 |
| partial R^2 | 0.0193 |
| weak-instrument warning (F < 10) | yes |

### Overlap: treatment rate by instrument quintile

| bin | Z range | n | treatment rate |
|---|---|---|---|
| 0 | [0.000, 0.159] | 150 | 0.147 |
| 1 | [0.159, 0.250] | 19 | 0.105 |
| 2 | [0.250, 1.000] | 32 | 0.250 |

### Covariate balance (top / bottom instrument quintile)

| covariate | mean low Z | mean high Z | SMD |
|---|---|---|---|
| `lab_bicarbonate_missing` | 0.955 | 0.833 | -0.401 |
| `lab_bun_missing` | 0.946 | 0.833 | -0.364 |
| `lab_creatinine_missing` | 0.946 | 0.833 | -0.364 |
| `lab_sodium_missing` | 0.929 | 0.833 | -0.295 |
| `lab_bilirubin` | 0.507 | 0.494 | -0.286 |
| `lab_creatinine` | 1.600 | 1.438 | -0.221 |
| `lab_wbc_missing` | 0.938 | 0.875 | -0.214 |
| `lab_platelets_missing` | 0.938 | 0.875 | -0.214 |
| `lab_wbc` | 12.392 | 13.354 | +0.183 |
| `lab_bun` | 26.259 | 24.208 | -0.178 |
| `lab_ph_missing` | 0.938 | 0.896 | -0.150 |
| `lab_lactate` | 2.124 | 2.026 | -0.145 |
| `admissionweight` | 79.655 | 76.554 | -0.128 |
| `vital_sao2_missing` | 0.071 | 0.104 | +0.115 |
| `vital_map` | 76.348 | 74.469 | -0.099 |

Large |SMD| indicates the instrument may encode patient composition or unit specialisation rather than practice style — the principal exclusion-restriction threat here.

### Effect estimates

| estimator | effect on in-hospital mortality | robust SE |
|---|---|---|
| naive OLS (confounded) | +0.1229 | 0.0877 |
| 2SLS | +0.1130 | 0.5274 |

The gap between the two illustrates confounding by indication. It is not evidence that the IV estimate is correct.

## Construction: `hospital`

- rows: 201, clients: 21
- covariates in the design: 32
- clients with structural instrument variation: 14
- treatment rate: 0.159
- mortality rate: 0.164

### Relevance (first stage)

| quantity | value |
|---|---|
| instrument coefficient | 0.8628 |
| robust SE | 0.7997 |
| partial F | 1.37 |
| partial R^2 | 0.0082 |
| weak-instrument warning (F < 10) | yes |

### Overlap: treatment rate by instrument quintile

| bin | Z range | n | treatment rate |
|---|---|---|---|
| 0 | [0.106, 0.122] | 43 | 0.116 |
| 1 | [0.122, 0.145] | 70 | 0.143 |
| 2 | [0.145, 0.159] | 42 | 0.167 |
| 3 | [0.159, 0.185] | 11 | 0.182 |
| 4 | [0.185, 0.299] | 35 | 0.229 |

### Covariate balance (top / bottom instrument quintile)

| covariate | mean low Z | mean high Z | SMD |
|---|---|---|---|
| `lab_creatinine` | 1.525 | 1.402 | -0.477 |
| `vital_map_missing` | 0.186 | 0.067 | -0.361 |
| `vital_sbp_missing` | 0.186 | 0.067 | -0.361 |
| `lab_bun` | 25.419 | 23.267 | -0.334 |
| `vital_map` | 79.965 | 74.911 | -0.281 |
| `lab_bun_missing` | 0.930 | 0.844 | -0.271 |
| `lab_bicarbonate_missing` | 0.930 | 0.844 | -0.271 |
| `lab_creatinine_missing` | 0.930 | 0.844 | -0.271 |
| `vital_temperature_missing` | 0.884 | 0.956 | +0.264 |
| `lab_lactate` | 2.081 | 2.001 | -0.229 |
| `lab_bilirubin_missing` | 0.930 | 0.978 | +0.226 |
| `lab_lactate_missing` | 0.837 | 0.911 | +0.222 |
| `lab_wbc` | 13.872 | 12.568 | -0.210 |
| `lab_ph` | 7.296 | 7.291 | -0.200 |
| `vital_sbp` | 114.651 | 109.578 | -0.189 |

Large |SMD| indicates the instrument may encode patient composition or unit specialisation rather than practice style — the principal exclusion-restriction threat here.

### Effect estimates

| estimator | effect on in-hospital mortality | robust SE |
|---|---|---|
| naive OLS (confounded) | +0.1229 | 0.0877 |
| 2SLS | +0.5750 | 0.9394 |

The gap between the two illustrates confounding by indication. It is not evidence that the IV estimate is correct.

