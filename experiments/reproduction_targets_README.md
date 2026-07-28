# Reproduction target registry

`reproduction_targets.csv` is the single registry for the requested 10-scenario by 8-algorithm experiment matrix. It separates two tracks.

- `paper_reproduction` contains the 45 numerical targets reported in arXiv:2505.21012v1, Section 5 / Figure 1 / Table 1. These rows require the captured paper protocol before they can be described as reproductions.
- `extension` contains the 35 requested comparisons that have no published numerical target. Extension rows must not be evaluated as numeric matches or misses against nonexistent paper results.

The published mappings are `oadam_s` → DeepGMM-OAdam, `gda_d` → DeepGMM-GDA, `sgda_s` → DeepGMM-SGDA, `fedgda_d` → FDeepGMM-GDA, and `fedgda_s` → FDeepGMM-SGDA. The paper has no Sine scenario, no FedOGDA method, and no separately reported deterministic OAdam result. Therefore every Sine, FedOGDA, and `oadam_d` row is an extension.

Existing repository-config runs are stored separately in the `our_*` fields. They may be scientifically useful, but they are not paper-protocol matched when their effective configuration differs—for example, the current ABS and Step pilots use `partition_alpha = 0.5`, while the paper target uses `0.3`.

All tuning and checkpoint selection must use validation data only. Test MSE is reported only after the configuration is selected; it must never select a configuration or checkpoint. The registry records no requirement that FedOGDA win—its comparison is a research hypothesis to evaluate with paired, fair, validation-tuned experiments.

Validate the registry with:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python \
  scripts/validate_reproduction_targets.py \
  experiments/reproduction_targets.csv
```
