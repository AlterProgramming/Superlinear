# Superlinear: Learning Computation from Multiplication

*A living research book and shared problem notebook.*

**Central question:** Can a model **discover**, **reuse**, and **select** higher-order computations composed from learned linear parts, with improvements that justify their actual execution cost?

We are not trying to prove that products are nonlinear. That is easy. We are trying to understand what makes multiplicative computation *learnable* and *economically useful*, without assuming a fixed vocabulary of interactions.

## Read in this order

1. [Chapter 1 — Linear parts, nonlinear wholes](01-linear-parts-nonlinear-wholes.md): the mathematical result we actually know.
2. [Chapter 2 — Representation is not discovery](02-representation-is-not-discovery.md): the failure that motivates our work.
3. [Chapter 3 — What evidence would count?](03-what-evidence-counts.md): tests that could change our minds.
4. [Chapter 4 — The empirical record](04-empirical-record.md): what has been observed, with caveats and source data.
5. [Current assignment: P001](problems/P001.md): one problem set; proof, interpretation, one optional experiment.

You can start with **P001** after Chapter 2. Each chapter is self-contained enough to reread later. There is no fixed schedule or expectation that the reader keeps up with automation.

## The three distinct claims

| Claim | Present status | What would make it stronger? |
|---|---|---|
| A. **Algebraic expressivity**: homogeneous compositions of products can have degree growing as `2^depth`. | Proven for the defined operation. | Generalize the theorem carefully to other graph grammars. |
| B. **Learned discovery**: ordinary learning can identify and reuse useful high-order computations when their operands are not given. | **Not established**: bilinear successes and high-order failures coexist. | Ablations separating optimization, representation, scale and data. |
| C. **Cost-effective adaptive computation**: learned interactions give disproportionately better out-of-sample performance for measured resources. | **Not established** in the current implementation. | Matched cost–quality curves, real data, uncertainty, and honest runtime measurements. |

An improvement in A does not entail B; an improvement in B does not entail C.

## Working agreement

- The human participant is a **co-investigator**, not an audience. Pause at **decision checkpoints** before changing the research objective or promoting a hypothesis as an explanation.
- Work may advance without the participant being online: reproduce runs, inspect failures, or prepare competing hypotheses. But separate **observations**, **explanations**, and **decisions**. No assumption that silence equals agreement.
- Each experiment must link to its *specific* hypothesis, code, data source, seed, budget, outcome, and failure cases. Synthetic vs real observational data must remain labeled.
- A fixed feature table, oracle-weight assignment, or a restricted grammar may be a useful control, but it is not evidence that unrestricted topology or feature discovery works. Mark proxies conspicuously.
- Negative results remain in the record. A broad claim must never rest on the best seed or a convenient example.
- If discussion repeats itself, name the unresolved assumption and change the proposed test or stop that line. More experiments are not automatically more understanding.

See [AGENTS.md](../AGENTS.md) for executable-research directives and [decision log](DECISIONS.md) for the currently open choice.

## How to study alongside the experiments

1. Read a chapter and attempt its problem on paper or in a small notebook.
2. Compare your reasoning with **the experiment that would distinguish alternatives**, not just with a reported score.
3. Reply in chat, or comment on the corresponding GitHub issue, with a proof, counterexample, objection, or partial thought. Incomplete work is useful if it identifies the point of uncertainty.
4. We decide together whether the next experiment should target the representation, the optimizer, the cost measure, or a hidden confounder.

The repository's existing implementation, datasets and measurements remain in [src](../src/multisuper), [results](../results), and [docs/RESULTS.md](../docs/RESULTS.md). The book interprets those materials; it does not replace them.

## Optional reference shelf

These are **background references**, not substitute evidence for our architecture:

- Goodfellow, Bengio & Courville, *Deep Learning*: backpropagation, optimization, and representation learning. Free official text: https://www.deeplearningbook.org/
- Nocedal & Wright, *Numerical Optimization*: gradient methods, conditioning, and numerical diagnostics.
- Boyd & Vandenberghe, *Convex Optimization*: mathematical language for objectives and constraints. **Our product architectures are not generally convex**; do not infer global guarantees.

**Revision rule:** Chapters can change when we learn something new; the [empirical record](04-empirical-record.md) and [decision log](DECISIONS.md) should preserve when and why the interpretation changed.
