# Sweep: train_n_requested

All figures measured; only complete available seeds included.

| Dataset | Model | Budget | n | Mean test score | Mean estimated MACs/sample | Median batch ms |
|---|---|---:|---:|---:|---:|---:|
| bilinear | learned_product | 128 | 5 | 0.1170 | 268.0 | 0.4987 |
| bilinear | learned_product | 512 | 5 | 0.0007 | 268.0 | 0.4614 |
| bilinear | learned_product | 2048 | 5 | 0.0003 | 268.0 | 0.4592 |
| bilinear | mlp | 128 | 5 | 0.0632 | 312.0 | 0.0313 |
| bilinear | mlp | 512 | 5 | 0.0112 | 312.0 | 0.0317 |
| bilinear | mlp | 2048 | 5 | 0.0095 | 312.0 | 0.0308 |
| bilinear | routed_product | 128 | 5 | 0.1038 | 342.5 | 0.7956 |
| bilinear | routed_product | 512 | 5 | 0.0070 | 348.3 | 0.7225 |
| bilinear | routed_product | 2048 | 5 | 0.0029 | 340.2 | 0.7643 |
| quartic | learned_product | 128 | 5 | 1.2982 | 268.0 | 0.5174 |
| quartic | learned_product | 512 | 5 | 1.1133 | 268.0 | 0.4491 |
| quartic | learned_product | 2048 | 5 | 0.1382 | 268.0 | 0.4711 |
| quartic | mlp | 128 | 5 | 1.2741 | 312.0 | 0.0319 |
| quartic | mlp | 512 | 5 | 1.0732 | 312.0 | 0.0311 |
| quartic | mlp | 2048 | 5 | 0.2928 | 312.0 | 0.0319 |
| quartic | routed_product | 128 | 5 | 1.2937 | 364.0 | 0.5970 |
| quartic | routed_product | 512 | 5 | 1.3249 | 340.3 | 0.7541 |
| quartic | routed_product | 2048 | 5 | 0.2301 | 330.9 | 0.7528 |
| sextic | learned_product | 128 | 5 | 1.6755 | 268.0 | 0.4635 |
| sextic | learned_product | 512 | 5 | 1.4043 | 268.0 | 0.4708 |
| sextic | learned_product | 2048 | 5 | 1.8703 | 268.0 | 0.4534 |
| sextic | mlp | 128 | 5 | 1.6397 | 312.0 | 0.0311 |
| sextic | mlp | 512 | 5 | 1.3490 | 312.0 | 0.0312 |
| sextic | mlp | 2048 | 5 | 1.8284 | 312.0 | 0.0314 |
| sextic | routed_product | 128 | 5 | 1.6887 | 364.0 | 0.6445 |
| sextic | routed_product | 512 | 5 | 1.3906 | 364.0 | 0.5639 |
| sextic | routed_product | 2048 | 5 | 1.8432 | 359.5 | 0.6834 |
| switched | learned_product | 128 | 5 | 0.9726 | 268.0 | 0.4754 |
| switched | learned_product | 512 | 5 | 0.3317 | 268.0 | 0.4661 |
| switched | learned_product | 2048 | 5 | 0.1050 | 268.0 | 0.4740 |
| switched | mlp | 128 | 5 | 0.9817 | 312.0 | 0.0314 |
| switched | mlp | 512 | 5 | 0.1778 | 312.0 | 0.0315 |
| switched | mlp | 2048 | 5 | 0.0760 | 312.0 | 0.0314 |
| switched | routed_product | 128 | 5 | 1.1082 | 359.3 | 0.6892 |
| switched | routed_product | 512 | 5 | 0.2884 | 325.8 | 0.7655 |
| switched | routed_product | 2048 | 5 | 0.1181 | 315.9 | 0.7506 |

**Not an established superlinear score/compute law.** For regression, lower is better. For digits, higher is better. MAC counts ignore memory/control; inference elapsed times are CPU measurements with fixed batch sizes. Selecting a favorable interval post hoc cannot establish a scaling exponent.
