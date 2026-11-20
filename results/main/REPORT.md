# Empirical results (automatically generated)

**Source:** `results/main/runs.jsonl` (each row = an actually executed training run).

**Interpretation:** Advantage is positive when candidate is better. Classification: accuracy-point difference; regression: reduction in normalized MSE. Scores are held-out, checkpoints selected exclusively by validation loss. All synthetic and real provenance is tagged per run. Confidence intervals are exploratory uncorrected bootstrap over seed-level paired runs; real splits overlap, and multiple dataset comparisons are not corrected.

## Learned multiplication vs MLP
| Dataset (source) | Mean advantage | Exploratory 95% bootstrap CI | Wins / seeds | MACs ref → candidate |
|---|---:|---:|---:|---:|
| additive (synthetic) | -0.0002 | [-0.0003, -0.0001] | 1/8 | 312 → 268 |
| bilinear (synthetic) | +0.0083 | [+0.0065, +0.0103] | 8/8 | 312 → 268 |
| breast_cancer (real_observational) | -0.0033 | [-0.0099, +0.0044] | 2/8 | 768 → 612 |
| diabetes (real_observational) | -0.0487 | [-0.1184, +0.0048] | 3/8 | 264 → 234 |
| digits (real_observational) | -0.0104 | [-0.0181, -0.0024] | 3/8 | 1776 → 1800 |
| noise_only (synthetic) | -0.0181 | [-0.0257, -0.0098] | 1/8 | 312 → 268 |
| parity6 (synthetic) | -0.4849 | [-0.8328, -0.1745] | 0/8 | 312 → 268 |
| quartic (synthetic) | -0.4460 | [-0.6630, -0.2626] | 0/8 | 312 → 268 |
| sextic (synthetic) | -0.0280 | [-0.0735, +0.0124] | 3/8 | 312 → 268 |
| switched (synthetic) | -0.0964 | [-0.1256, -0.0704] | 0/8 | 312 → 268 |
| wine (real_observational) | -0.0347 | [-0.0590, -0.0139] | 0/8 | 384 → 327 |

## Input-routed multiplication vs always-executed learned multiplication
| Dataset (source) | Mean advantage | Exploratory 95% bootstrap CI | Wins / seeds | MACs ref → candidate |
|---|---:|---:|---:|---:|
| additive (synthetic) | -0.0006 | [-0.0008, -0.0004] | 0/8 | 268 → 336 |
| bilinear (synthetic) | -0.0041 | [-0.0053, -0.0028] | 0/8 | 268 → 341 |
| breast_cancer (real_observational) | -0.0022 | [-0.0088, +0.0044] | 3/8 | 612 → 770 |
| diabetes (real_observational) | +0.0364 | [-0.0158, +0.1111] | 5/8 | 234 → 311 |
| digits (real_observational) | +0.0028 | [-0.0045, +0.0097] | 5/8 | 1800 → 2138 |
| noise_only (synthetic) | -0.0090 | [-0.0374, +0.0136] | 3/8 | 268 → 364 |
| parity6 (synthetic) | +0.0785 | [-0.1914, +0.4176] | 3/8 | 268 → 313 |
| quartic (synthetic) | -0.0958 | [-0.2693, +0.0722] | 2/8 | 268 → 314 |
| sextic (synthetic) | -0.0135 | [-0.0698, +0.0346] | 3/8 | 268 → 361 |
| switched (synthetic) | +0.0220 | [-0.0105, +0.0586] | 4/8 | 268 → 318 |
| wine (real_observational) | +0.0278 | [+0.0104, +0.0451] | 5/8 | 327 → 418 |

## What this does NOT prove

- Degree doubling is algebraic, not a measured learning-efficiency gain.
- Strong synthetic performance is not evidence of advantage on natural or deployment data.
- A significant positive pairwise score does not mean superlinear capability relative to *actual* inference cost.
- Node counts/MAC estimates omit memory, routing overhead, energy and device effects; CPU batch latency is measured separately in the raw runs.
- The learned product network uses a bounded, ordered graph grammar; it is not unconstrained architecture discovery.
