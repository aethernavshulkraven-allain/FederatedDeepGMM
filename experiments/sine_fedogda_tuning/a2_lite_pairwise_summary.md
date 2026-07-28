# A2-lite FedOGDA-D vs FedGDA-D

Primary comparison: `test_mse_at_best_validation` after the FedOGDA recipe was locked by validation only.

- FedOGDA-D mean Test MSE: `0.0800115346`
- FedGDA-D mean Test MSE: `0.0861068629`
- Absolute improvement: `0.0060953283`
- Relative improvement: `7.0788%`
- FedOGDA-D seed wins: `3/3`
- FedGDA-D seed wins: `0/3`

FedOGDA-D has per-round Test MSE for secondary last-50 reporting. The legacy FedGDA-D baseline does not, so no paired last-50 Test MSE claim is made.

**SUPPORTED: FedOGDA-D achieves lower validation-selected Test MSE than paired FedGDA-D on Sine.**
