# Results: completed experiments, not predictions

## Study chronology

- One-seed pilot: 80 epochs, six datasets, five architectures. Exploratory debugging, retained separately. Not pooled with the larger study.
- Main study: 8 random seeds × 11 datasets (7 synthetic; 4 real observational) × 5 architectures = **440 trained models**, all validation-selected with 180 full-batch gradient updates.
- Oracle capacity: 8 explicit rotated degree-4/6 polynomial assignments to the *same* learned-node architecture show the target functions are representable. This is **not** a learning experiment.
- Node-budget sweep and sample-size sweep: results in adjacent directories, with raw score/compute/provenance rows and failure cases.
- Post-hoc gradient diagnostics: exploring the reason learned reuse may not bootstrap; findings here are exploratory rather than predeclared.

## Main result (8 seeds)

| Task | Provenance | MLP | Learned product | Interpretation |
|---|---|---:|---:|---|
| Rotated bilinear | synthetic | 0.0086 MSE | **0.0003 MSE** | Strong positive; 8/8 seeds |
| Rotated quartic | synthetic | **0.3692 MSE** | 0.8152 MSE | Learnable by an oracle but poorly found by SGD; 0/8 seeds |
| Rotated sextic | synthetic | **1.0477 MSE** | 1.0756 MSE | Neither generally solves; no superlinearity |
| Switched pair product | synthetic | **0.1179 MSE** | 0.2143 MSE | Learned products lag generic MLP |
| 6-way parity | synthetic | **0.0130 MSE** | 0.4979 MSE | 8/8 losses, large optimization gap |
| Digits | real observational | **96.18%** | 95.14% | No advantage |
| Wine | real observational | **98.61%** | 95.14% | No advantage |
| Breast cancer | real observational | 97.26% | 96.93% | Near tie, no claim |
| Diabetes | real observational | **0.5394 MSE** | 0.5881 MSE | No advantage |

*Lower is better on regression; higher is better on classification. Values are across-seed arithmetic means, not confidence bounds. See `results/main/REPORT.md` for paired bootstrap intervals and all 11 datasets, including additive and pure noise controls.*

## Cost evidence

On bilinear test batches (512 examples), learned product model: 268 estimated MACs/sample and ~0.47 ms CPU inference. MLP: 312 estimated MACs/sample and ~0.03 ms inference. Thus the learned product uses slightly fewer scalar MACs but roughly 15× more measured time in this Python eager-mode implementation. That is an implementation-specific execution cost, not a universal property of multiplicative models. The routed version generally paid more cost in routing than it saved in products at its initial small penalty.

## Main conclusion

**Algebraically, degree multiplication is confirmed. Empirically, the tested learned-operand architecture excels at discovering bilinear interactions but fails to reliably build high-order ones or provide real-data, cost-aware scaling advantages.** The result does not refute other representations of multiplicative computation, but it refutes the shortcut claim that multiplying linear pieces is enough to obtain trainable superlinear capability.

Exact per-run measurements are retained, including negative examples, split seeds, operation counts, latency, parameters and dataset source. Do not promote synthetic successes to real-world proof or replace the architecture with a fixed feature pair table.

## Additional sweep findings

- 250 **node-budget** training runs (5 datasets × 2 architectures × 5 budgets × 5 seeds) did **not** show monotonic benefit from extra multiplicative nodes. Bilinear often solves with a single node, while larger budgets can worsen higher-order test behavior.
- 180 **sample-size** runs (4 synthetic datasets × 3 architectures × 3 sizes × 5 seeds) demonstrated that data volume can sharply affect high-order performance; on quartic, learned-product MSE dropped from 1.2982 at 128 training examples to 0.1382 at 2048. The improvement is not a proof of general asymptotic scaling.
- 20 additional **post-hoc gradient-trace** training runs (4 synthetic datasets × 5 seeds, 360 updates) collected 1280 node-level snapshots. Raw gradients and activation magnitudes are in `results/diagnostics/gradient_trace.csv`. They show changing product activation scales and gradients, but do not, by themselves, establish a causal gradient-starvation mechanism.
- Reproduced a serious 16-node routed outlier (quartic, seed 2), including its checkpoint, exact held-out example, and every intermediate multiplication. See `docs/INCIDENT.md`.

**In total:** 870 standard benchmark training runs + 20 post-hoc gradient-trace training runs + 1 repeated incident replication. The automated test suite has 58 passed checks. The algebra and the larger learned-architecture hypothesis are recorded distinctly.
