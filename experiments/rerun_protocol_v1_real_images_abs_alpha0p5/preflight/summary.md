# High-dimensional preflight summary

- Fixed response function: `abs`
- Certified data scenarios: `6/6`
- Exact image split isolation: `pass`
- Deterministic full-gradient checks: `12/12`
- Stochastic end-to-end smokes: `4/4`
- Overall preflight: `pass`

The deterministic checks cover both image families, both deterministic optimizers, and all three seeds. 
The stochastic smokes cover both image families and both stochastic optimizers through artifact writing.
