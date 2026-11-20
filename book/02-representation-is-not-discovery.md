# Chapter 2 — Representation is not discovery

## 1. The observed gap

The existing experiment has an oracle construction demonstrating that the **same bounded learned-product architecture** can represent explicit rotated quartic and sextic products. That is a *capacity witness*, not a learning result. When the parameters are initialized and trained using the recorded optimizer, the architecture does not reliably recover those high-order functions.

Eight-seed means from [the recorded study](../docs/RESULTS.md):

| Task | Type | MLP normalized MSE | Learned-product normalized MSE |
|---|---|---:|---:|
| Hidden rotated bilinear | Synthetic | 0.0086 | **0.0003** |
| Hidden rotated quartic | Synthetic | **0.3692** | 0.8152 |
| Six-way parity | Synthetic | **0.0130** | 0.4979 |

The gap is not a proof that gradient descent cannot learn these functions; it establishes failure of **this model, training recipe and data regime**. Training data size and numerical stability are known confounders. The proper problem is to identify a *mechanism* rather than simply repeat the failure.

## 2. A candidate mechanism: weak high-order learning signal

Analyze a simple model, which is **not identical to the repository model**:

\[
 f_\theta(x)=\prod_{j=1}^k(\theta_j^\top x),\quad
 L(\theta;x,y)=\tfrac12(f_\theta(x)-y)^2.
\]

For each vector `θ_j`, the chain rule gives

\[
 \nabla_{\theta_j}L=(f_\theta(x)-y)\,x\,\prod_{i\ne j}(\theta_i^\top x).
\]

If all operands are initialized as `θ_i=εv_i`, with bounded `v_i` and `x`, and `y` remains nonzero, then the leading-order gradient norm is proportional to `ε^(k−1)` as `ε→0` (except in degenerate directions). Four factors can produce a much smaller gradient than two factors. **For `k>1`, all-zero operands form a stationary point under this toy loss.**

This is a plausible optimization obstacle: a high-order target can be representable without providing useful early gradients to train the factors jointly.

**Do not overclaim:** real networks can use biases, skip paths, staged learning, different parameterizations, initializations, optimizers and many examples. Gradient magnitude is not the same as gradient *information* or the odds of convergence. The toy derivation suggests an intervention; it does not establish the cause of our quartic results.

## 3. An alternative explanation: unstable forward computation

The existing 16-node routed model produced a catastrophic held-out quartic outlier, even though the raw inputs were in the training distribution. Intermediate products amplified until the prediction was millions of units from its target. See [the exact reproduced incident](../docs/INCIDENT.md).

Gradient clipping cannot stop this: clipping changes optimizer updates, not the forward intermediate values. A method can suffer **both** weak useful gradients and unbounded forward states.

But note a crucial distinction: the initial eight-node quartic failure and the documented sixteen-node routed failure are not the same case. One incident does not prove the mechanism behind every failure.

## 4. Three rival explanations, not one settled story

- **Optimization:** suitable factors exist but the gradients/initialization rarely discover them.
- **Representation and path dependence:** useful compositions require intermediate states or a different graph grammar; the chosen bounded ordered nodes make discovery unnecessarily difficult.
- **Statistics and evaluation:** available samples, target variance, scaling, splits, or rare activations confound generalization even if train fit improves.

The next experiment should force these explanations to make **different predictions**. Merely comparing our network against a weaker control will not answer the question.

## 5. What we should try to falsify

> Hypothesis H1 (still unconfirmed): in the tested high-order product regime, the main learning barrier comes from the *path by which useful intermediate operands become trainable*, rather than a shortage of expressivity.

A useful test should keep the target/data/operation budget fixed, measure train **and** held-out loss, and compare alternative operand initializations or staged training. It should check whether a change merely hand-supplies an answer. Fixing the factors to the oracle would diagnose capacity, not discovery.

### Reflection

If a different initialization solves quartic but increasing the node count does not, what does this say about the optimization and capacity explanations? What *doesn't* it prove?

Go to [P001](problems/P001.md) to work through the derivative and an experiment design before we choose the next intervention.
