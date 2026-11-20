# Sweep: nodes_budget

All figures measured; only complete available seeds included.

| Dataset | Model | Budget | n | Mean test score | Mean estimated MACs/sample | Median batch ms |
|---|---|---:|---:|---:|---:|---:|
| bilinear | learned_product | 1 | 5 | 0.0000 | 37.0 | 0.0903 |
| bilinear | learned_product | 2 | 5 | 0.0000 | 64.0 | 0.1388 |
| bilinear | learned_product | 4 | 5 | 0.0001 | 124.0 | 0.2383 |
| bilinear | learned_product | 8 | 5 | 0.0004 | 268.0 | 0.4671 |
| bilinear | learned_product | 16 | 5 | 0.0007 | 652.0 | 0.9528 |
| bilinear | routed_product | 1 | 5 | 0.0033 | 48.8 | 0.1353 |
| bilinear | routed_product | 2 | 5 | 0.0077 | 85.4 | 0.2293 |
| bilinear | routed_product | 4 | 5 | 0.0154 | 163.2 | 0.3887 |
| bilinear | routed_product | 8 | 5 | 0.0034 | 341.7 | 0.7590 |
| bilinear | routed_product | 16 | 5 | 0.0034 | 827.0 | 1.4342 |
| digits | learned_product | 1 | 5 | 0.9506 | 778.0 | 0.1209 |
| digits | learned_product | 2 | 5 | 0.9450 | 918.0 | 0.1719 |
| digits | learned_product | 4 | 5 | 0.9422 | 1204.0 | 0.2735 |
| digits | learned_product | 8 | 5 | 0.9517 | 1800.0 | 0.5447 |
| digits | learned_product | 16 | 5 | 0.9650 | 3088.0 | 0.9848 |
| digits | routed_product | 1 | 5 | 0.9506 | 802.0 | 0.1725 |
| digits | routed_product | 2 | 5 | 0.9517 | 983.0 | 0.2588 |
| digits | routed_product | 4 | 5 | 0.9439 | 1339.4 | 0.4395 |
| digits | routed_product | 8 | 5 | 0.9572 | 2136.2 | 0.8224 |
| digits | routed_product | 16 | 5 | 0.9594 | 3857.4 | 1.5612 |
| quartic | learned_product | 1 | 5 | 0.8359 | 37.0 | 0.0895 |
| quartic | learned_product | 2 | 5 | 0.7607 | 64.0 | 0.1496 |
| quartic | learned_product | 4 | 5 | 0.7769 | 124.0 | 0.2351 |
| quartic | learned_product | 8 | 5 | 0.7206 | 268.0 | 0.4678 |
| quartic | learned_product | 16 | 5 | 0.8702 | 652.0 | 0.9449 |
| quartic | routed_product | 1 | 5 | 0.8409 | 42.2 | 0.1325 |
| quartic | routed_product | 2 | 5 | 0.7092 | 64.1 | 0.2240 |
| quartic | routed_product | 4 | 5 | 1.0179 | 137.7 | 0.3943 |
| quartic | routed_product | 8 | 5 | 0.8012 | 316.4 | 0.7537 |
| quartic | routed_product | 16 | 5 | 20716743885.2794 | 796.4 | 1.4639 |
| sextic | learned_product | 1 | 5 | 0.8713 | 37.0 | 0.0894 |
| sextic | learned_product | 2 | 5 | 0.9217 | 64.0 | 0.1377 |
| sextic | learned_product | 4 | 5 | 0.9334 | 124.0 | 0.2359 |
| sextic | learned_product | 8 | 5 | 0.8783 | 268.0 | 0.4519 |
| sextic | learned_product | 16 | 5 | 0.8945 | 652.0 | 1.0116 |
| sextic | routed_product | 1 | 5 | 0.9314 | 43.3 | 0.1299 |
| sextic | routed_product | 2 | 5 | 0.9298 | 82.2 | 0.1681 |
| sextic | routed_product | 4 | 5 | 0.9584 | 163.5 | 0.3490 |
| sextic | routed_product | 8 | 5 | 0.9173 | 363.6 | 0.5550 |
| sextic | routed_product | 16 | 5 | 0.9767 | 841.3 | 1.2005 |
| switched | learned_product | 1 | 5 | 0.7081 | 37.0 | 0.0890 |
| switched | learned_product | 2 | 5 | 0.3552 | 64.0 | 0.1379 |
| switched | learned_product | 4 | 5 | 0.1718 | 124.0 | 0.2341 |
| switched | learned_product | 8 | 5 | 0.2226 | 268.0 | 0.4759 |
| switched | learned_product | 16 | 5 | 0.3767 | 652.0 | 0.9431 |
| switched | routed_product | 1 | 5 | 0.5790 | 39.3 | 0.1346 |
| switched | routed_product | 2 | 5 | 0.2726 | 70.9 | 0.2284 |
| switched | routed_product | 4 | 5 | 0.1814 | 144.1 | 0.3937 |
| switched | routed_product | 8 | 5 | 0.1730 | 314.6 | 0.7870 |
| switched | routed_product | 16 | 5 | 0.1937 | 804.5 | 1.4872 |

**Not an established superlinear score/compute law.** For regression, lower is better. For digits, higher is better. MAC counts ignore memory/control; inference elapsed times are CPU measurements with fixed batch sizes. Selecting a favorable interval post hoc cannot establish a scaling exponent.
