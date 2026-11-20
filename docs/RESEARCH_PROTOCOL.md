# Research protocol (initial executable iteration)

## Claim separation

- **A — Algebra:** Degree 2^L from L pure homogeneous product layers. The mathematical identity can be verified independently of labels, training, optimization or datasets. A geometric/exponential degree *is not* an exponential useful performance gain.
- **B — Discoverability:** Trainable operands and reuse learn hidden, unspecified targets from data. Tests compare to a simple linear predictor, generic MLP, and frozen-random product topology. Fixed-random explicitly tests an *easier proxy* and cannot establish learned computation.
- **C — Generalization:** Seen synthetic (bilinear/quartic/sextic/switched/parity) and non-synthetic (digits, wine, Wisconsin, diabetes) must remain separated. Test wider-input-support OOD for selected synthetic tasks. For real datasets, hold-out splits are IID within a curated set, **not true domain transfer**.
- **D — Economic improvement:** More favorable score at given MACs, latency, train time, and parameter counts. A positive accuracy difference alone is insufficient. Node budgets 1,2,4,8,16 test scaling; no monotonic improvement is presumed. Conditional routes must show *real* skipped operations, not soft sparsity on paper.

## Decision criteria, before looking at the 8-seed main study

1. **Representable** if the oracle assignments reproduce rotated degree-4 and degree-6 polynomials numerically. This does not entail SGD discovery.
2. **Trainable in practice** if repeated seeded runs outperform linear controls, measured on held-out data. Isolated single-seed success will be labeled exploratory.
3. **Wider advantage** only if learned multiplication beats an MLP on a majority of non-synthetic datasets across at least 8 seeds, with positive paired bootstrap intervals and comparable realized inference cost; otherwise it remains unconfirmed.
4. **Cost-aware routing beneficial** only if routed model uses at least 25% fewer total estimated MACs and doesn't lose more than 1 standard error of held-out score versus unrouted model; require actual batch latency reduction as secondary check.
5. **Superlinear model capability per compute** is **not established** by the current pilot unless a preregistered comparison of score/compute curves shows supra-linear gains over a clearly identified range, plus matched control models. Saturating bounded accuracy often makes an arbitrary power-law fit meaningless.

## Failure modes worth preserving

- Multiplicative nodes can represent higher-degree targets but SGD may stall due to weak compositional gradients, symmetry, rare high-magnitude targets, routing variance, or poor conditioning.
- Product nets have many more operations than a linear baseline; accuracy comparisons require a compute control.
- Synthetic target families are selected to have multiplication-based generating structure; real curated measurements and labels provide a counterweight.
- 8 seeds on a few observational datasets do not give broad external validity; tests are exploratory and can be overly sensitive to split or hyperparameters.
- Fixed budgets of eight ordered operations restrict the hypothesis class. No claim of architecture-level novelty or universal discovery follows.
