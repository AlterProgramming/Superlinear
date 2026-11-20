# Chapter 1 — Linear parts, nonlinear wholes

## 1. The easy theorem, stated precisely

Let `x` be a vector and define a sequence of vector-valued maps

\[
 h_0(x)=x,\qquad h_{\ell+1}(x)=(A_\ell h_\ell(x))\odot(B_\ell h_\ell(x)),
\]

where `⊙` multiplies matching coordinates; all matrices are fixed during evaluation. Assume **no biases, skips, normalizations, or degree-mixing inputs** in this statement.

**Proposition.** Each coordinate of `h_l(x)` is a homogeneous polynomial of degree `2^l` *or the zero polynomial*.

**Proof.** At layer zero, every coordinate of `x` is homogeneous of degree one. If every coordinate at layer `l` has degree `d` (or is zero), its linear combinations are homogeneous of degree `d` (or zero); multiplying two such combinations yields degree `2d` (or zero). Induction gives `d=2^l`. □

Consequently, for any real scalar `t`,

\[
 h_\ell(tx)=t^{2^\ell}h_\ell(x).
\]

**Precise limit:** this says nothing about how many independent coefficients the model can realize, how much data it needs, whether the function is useful, or how quickly it can be evaluated. The number `2^l` measures *polynomial degree*, not performance per operation.

Compare the implemented numerical check in [`src/multisuper/algebra.py`](../src/multisuper/algebra.py) and its [`tests/test_algebra.py`](../tests/test_algebra.py). Its numerical check is illustrative; the proof above is the mathematical result.

## 2. Our actual network is a different grammar

The network under study stores raw input `x` and previous scalar computations `z_1,…,z_{i-1}`. Its next node has the form

\[
 s_i=[x,z_1,\ldots,z_{i-1}],\qquad
 z_i=(a_i^\top s_i+\alpha_i)(b_i^\top s_i+\beta_i).
\]

Then a trainable linear readout combines raw inputs and node outputs. This supports high-degree terms and mixtures of degrees, including constants and skips **via the available input channels**; it does not imply every node is homogeneous or even that its degree doubles. Some nodes may ignore earlier nodes or have zero weights.

An upper bound on possible degree is `2^i` for node `i`, if the previous maximum doubles at each node, but the exact degree depends on learned coefficients and cancellations. Because the maximum node count and order are declared in advance, the architecture learns **operand selection within a bounded graph grammar**, not arbitrary self-growing topology.

Read the source: [`LearnedProductNet`](../src/multisuper/models.py). The forward pass, rather than a metaphor about 'thinking', specifies what can execute.

## 3. The question the theorem does NOT answer

Consider a quadratic target `y=(uᵀx)(vᵀx)`. A product node can represent it using two learned operands. A quartic target `y=∏_{j=1}^4 (u_jᵀx)` can be computed by building and multiplying intermediate products. In both cases the target is *representable*; only the latter requires a useful sequence of interactions.

What makes the higher-order computation discoverable from outcome-only feedback? Does a network learn useful intermediate parts, or does the presence of those parts simply expand a difficult search space?

That is the subject of [Chapter 2](02-representation-is-not-discovery.md).

### Reflection

What does the statement `degree = 2^depth` **actually measure**? Give an example where adding a product operation increases theoretical maximum degree but makes the practical model worse.
