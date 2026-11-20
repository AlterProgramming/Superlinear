# Chapter 4 — The empirical record

**Snapshot: November 19, 2026.** This is a reading guide to the existing experiment artifacts, not a newly executed result. The authoritative numbers remain in [docs/RESULTS.md](../docs/RESULTS.md), [results/main/REPORT.md](../results/main/REPORT.md) and the run-level files.

## What happened

- **440** main-study runs: 8 seeds × 11 datasets × 5 architectures. Main data cover seven synthetic and four real observational datasets. The bilinear advantage of learned multiplication was consistent across eight seeds, while quartic and six-way parity were not solved reliably compared with an MLP.
- **250** node-budget sweeps found no monotonic performance gain from additional nodes. Thus *operation count* is not a substitute for *useful learned computation*.
- **180** sample-size runs showed a substantial quartic improvement from increasing data, indicating that optimization, statistics and effective representation cannot yet be neatly separated.
- **20** post-hoc gradient-diagnostic runs observed variation in magnitudes but did not establish a causal gradient-starvation theory.
- **One** additional reproduced sixteen-node numerical instability incident shows that stability matters. Its existence does not diagnose all other failures.

**870 standard benchmark runs + 20 post-hoc gradient runs + 1 incident replication** are described in the results record. An oracle capacity demonstration confirms existence of weight assignments for higher-order targets; it is not a learned success.

## The counterexample that should stay visible

In the bilinear task, learned products used **268 estimated MACs/example** and **~0.47 ms CPU** inference, while the MLP used **312 estimated MACs/example** and **~0.03 ms CPU** inference on the reported 512-example batch. The model with fewer estimated MACs was about **15× slower in the reported eager Python implementation**. We have not demonstrated a hardware-independent cost advantage.

## What we can claim today

**Yes:** multiplication of linear computations is algebraically capable of high-order interactions; the bounded learned-operand model can sometimes discover low-order interactions very well.

**No:** the study has not shown reliable high-order discovery, superior real-observational performance, or cost-effective superlinear scaling. It also has not proven such performance impossible in alternative representations.

## Open decision

The next experiment should distinguish whether high-order failure is caused predominantly by (i) gradient/initialization path, (ii) missing state/graph representation, or (iii) statistical/generalization and forward-stability issues. **Do not preselect one explanation as the winner.** The human assignment [P001](problems/P001.md) helps articulate predictions before we run a new intervention.
