# Next falsifiable architectural experiment (not yet run)

The initial result is not a vindication or a general impossibility result. It shows three distinct problems: (i) training discovers bilinear interactions readily, (ii) credit assignment for reuse struggles on hidden higher-order targets, (iii) deep repeated products can catastrophically explode even on held-out IID samples.

The next study should operate on the **same raw datasets and preregistered seeds** and test a genuine alternative to unconstrained scalar product reuse, without substituting a hand-coded degree-4 formula or known variable pair table. Candidate intervention: constrain the *state representation*, not the target topology, using paired sign/log-magnitude and an explicitly budgeted arithmetic path. A second ablation could use additive residual bridges with a learnable multiplicative allocation.

Preserve the original target: feature-pair vocabulary-free learned operands, adaptive interactions, explicit cost-aware execution, and useful novel compositions. Require representation proofs for higher-order exact products, training efficacy on held-out randomized projection tasks, natural dataset comparisons, OOD stress tests, and latency/FLOP accounting. A fixed topology can be a *diagnostic control* but never the main result. Negative results must remain visible.

Success would mean reliable improvement across seeds and orders without outlier catastrophes or cost regressions, not a single favorable benchmark.
