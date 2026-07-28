# High-dimensional fixed-abs data certification

Overall decision: `certified`.

## Certified invariants

- Response function metadata is `abs` for every scenario.
- FEMNIST source is TFF Federated EMNIST with `only_digits=True`.
- CIFAR-10 source is `torchvision.datasets.CIFAR10`.
- Generator seed is `527`; split sizes are 20,000/10,000/10,000.
- All NPZ hashes, shapes, numeric finiteness checks, and standardized abs-response checks passed.
- All six scenarios share the exact same toy Y arrays for paired comparison.

## Dataset results

| Dataset | Core invariants | Abs error | Split isolation | Train/test image overlap |
| --- | --- | ---: | --- | ---: |
| femnist_x | true | 2.22e-15 | true | 0 |
| femnist_z | true | 1.11e-15 | true | 0 |
| femnist_xz | true | 2.22e-15 | true | 0 |
| cifar10_x | true | 2.22e-15 | true | 0 |
| cifar10_z | true | 1.11e-15 | true | 0 |
| cifar10_xz | true | 2.22e-15 | true | 0 |

## Decision

All required protocol and split-isolation checks passed. The files are certified for the 
fixed-abs high-dimensional experiment.
