# Chapter 3 — What evidence would count?

We distinguish three kinds of evidence and refuse to exchange one for another.

## A. A theorem about a defined operation

Define the operation, assumptions and quantifiers. A proof of polynomial-degree growth is strong evidence about **representability**; it is *not* a theorem about generalization, trainability, adaptive computation, or resource efficiency.

## B. A learning experiment

A claim of discovery needs honest training of previously unspecified operands and evaluation on held-out examples. The original goal includes not hardcoding a feature vocabulary or a fixed table of permitted interactions. Our bounded sequential node grammar is a *partial test* of that goal: operand weights are learned, but the number/order of node slots are supplied.

A useful comparison must explicitly identify exactly which architectural freedom was or was not tested. Oracle parameters and fixed-random topologies are **controls**, not evidence of learned topology.

We seek a causal distinction between at least two rival hypotheses and must keep *all* seeds and failure cases. A rescued benchmark might reveal an optimization fix, or might merely add an inductive bias tailored to a synthetic target.

## C. Quality achieved for actual computation

First define the measurable performance–cost curve `Q(C)` on a specified task distribution and hardware implementation. `Q` must be oriented so **higher is better**; for example use positive accuracy, or negative log loss rather than declaring MSE itself a quality measure.

The phrase *superlinear improvement* is meaningless without a specified reference relation and interval. For a concrete proposed test, compare controlled cost levels `C1 < C2`, and ask whether

\[
 \frac{Q(C_2)-Q(C_1)}{C_2-C_1}
\]

exceeds a **preregistered** baseline's incremental quality gain over the same cost interval, with uncertainty and an application-relevant cost measure. This is a definition of *disproportionate marginal benefit relative to a control*, not a universal definition of superlinearity. If one instead means the mathematical condition `Q(λC) > λQ(C)`, then `Q` must have a meaningful zero and ratio scale; bounded accuracy normally does **not**.

Report at least: estimated operations, wall-clock latency, parameter counts, actual skipped work, training cost, and any implementation caveats. More nodes and more parameters are not a matched computation budget.

## Protocol for a discriminating experiment

1. **Question and rivals.** State the original ambitious research objective and two competing explanations with predictions that diverge.
2. **Status label.** Mark the experiment as *mathematical*, *synthetic*, *real observational*, or *deployment*; mark any easier proxy.
3. **Controls.** Use suitable classical/MLP, ablation, and oracle controls without mistaking controls for the main architecture.
4. **Cost and uncertainty.** Predeclare split, seeds, training budget, holdout use, time/computation accounting and primary metric. Retain individual negative runs and numerical failures.
5. **Interpretation checkpoint.** Report observed evidence separately from a proposed causal explanation; flag post-hoc searches. Let the human co-investigator challenge the proposed pivot before it becomes the new default.

## Data distinction

The benchmark here includes intentionally constructed mathematical targets (**synthetic**) and curated real datasets (**real observational**). A synthetic multiplication task is the right place to isolate a mechanism, but a positive result on it is not proof of a natural-data advantage. Likewise, running a real dataset with predefined features does not demonstrate feature-vocabulary-free architecture discovery in a rich environment.

See [the study's provenance](../README.md), [raw JSONL](../results/main/runs.jsonl), and [report](../results/main/REPORT.md).
