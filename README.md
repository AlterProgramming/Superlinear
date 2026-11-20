# Multiplicative Superlinearity Lab

**Question:** Does multiplying learned linear computations yield something more than algebraic high-order expressivity—specifically useful learned computation, adaptivity, and better performance at a justifiable cost?

**Status:** An executable experimental research program, not a claim that the architectural thesis is proven.

## Three non-equivalent hypotheses

1. **Algebraic:** multiplying linear forms yields terms of order 2; stacking homogeneous product layers of depth L produces degree 2^L under nondegenerate weights. This is a mathematical identity, verified in numerical tests.
2. **Optimization / discovery:** gradient descent learns useful *previously unspecified* multiplicative operands and their reuse more reliably than competing architectures. Must be measured, can fail.
3. **Economic superlinearity:** gains in held-out performance grow more than proportionally to extra *actual* computation, over a specified range. Not implied by (1) or (2); must be demonstrated with controlled cost scaling, comparable baselines, and uncertainty intervals.

## Architectures and what each controls

- `linear`: affine predictor (cannot express interactions on raw inputs).
- `mlp`: 24-wide GELU predictor, generic nonlinear control. Parameter counts reported; not perfectly matched.
- `fixed_random`: frozen random, sequential product graph + learned readout. **Deliberately easier fixed-topology proxy**. Never treat this as a test of learned topology.
- `learned_product`: new node = product of two trainable linear forms over the raw input and previously created nodes. Learns operands and reuse through gradient descent, with a budget of 8 sequential nodes. These are genuine *learned* operands, not predefined named products. The maximum graph size and execution order remain a prior.
- `routed_product`: same, but an input-conditioned hard decision selects whether each node executes. A straight-through gate surrogate is used in training; at inference, skipped nodes really skip their dot products, even per sample. Differentiable budget penalty included.

**Important caveat:** this is NOT an unconstrained self-growing network and it does NOT discover an unlimited vocabulary-free computational language. Both models receive raw feature vectors; the representations are learned but the slot grammar is predeclared. Do not substitute benchmark success for that broader claim.

## Datasets and provenance

| Dataset | Provenance | Task | Notes |
|---|---|---|---|
| bilinear | **synthetic** | regression | two hidden rotated linear projections multiplied |
| quartic | **synthetic** | regression | four hidden rotated projections multiplied |
| sextic | **synthetic** | regression | six hidden rotated projections multiplied |
| switched | **synthetic** | regression | input-dependent choice between two products |
| additive | **synthetic** | regression | negative control, additive target |
| parity6 | **synthetic** | regression | six-way Rademacher parity with nuisance features |
| noise_only | **synthetic** | regression | independent random labels (negative control) |
| breast_cancer | **real observational** | classification | sklearn bundled UCI Wisconsin dataset |
| wine | **real observational** | classification | sklearn bundled UCI wine chemical measurements |
| digits | **real observational** | classification | sklearn bundled handwritten digit images, 8x8 pixels |
| diabetes | **real observational** | regression | sklearn bundled diabetes progression dataset |

Real datasets are **curated observational records**, not dynamically collected raw environments. The real datasets have labeled, pre-existing input features. Synthetic tests deliberately favor access to multiplicative targets; additive is the counterexample. Train/validation/test are disjoint; synthetic OOD uses wider support without tuning on it. Never pool synthetic and real results to claim real-world improvement.

## Run

```bash
python -m pip install -e '.[test]'
pytest -q
python -m multisuper.experiment --epochs 140 --seeds 0 1 2 --out results/pilot
```

`results/pilot/runs.jsonl`: every observed run, score, exact provenance, number of parameters, and wall-clock training time. `summary.csv`: across-seed means and SDs. `config.json`: environment. `algebra.json`: numerical scaling checks.

**Metrics:** classification test accuracy (higher is better); regression normalized MSE using *training* target std (lower is better), so MSE near 1 is uninformative. These are validation-selected checkpoints. Synthetic OOD scores use the training-set normalization. Training seconds are noisy and exclude data loading and evaluation; `mean_products_executed_per_example` counts bilinear product nodes, not all FLOPs or total hardware energy. `estimated_macs_per_example` estimates multiply-accumulate operations including projections, readout and routing but excludes elementwise operations, memory traffic and control overhead; measured inference batch latency is provided separately. Neither alone establishes equal hardware cost.

## To justify the stronger thesis later

Preregister a compute-scaling curve over node budget and matched control FLOPs/latency; vary training set sizes and out-of-distribution shifts; compare with MLPs with the same wall-clock *inference* budget; avoid constructed target families and input-space selection bias; and verify adaptivity actually saves compute rather than merely adding a gate. Report variability and failures. End-to-end hardware cost and parameter storage matter.

### Additional scaling sweeps

```bash
python -m multisuper.sweep --kind nodes --epochs 180 --seeds 0 1 2 3 4
python -m multisuper.sweep_report results/node_sweep/all_runs.jsonl
python -m multisuper.sweep --kind samples --epochs 180 --seeds 0 1 2 3 4
python -m multisuper.sweep_report results/sample_sweep/all_runs.jsonl
python -m multisuper.analyze results/main
```

**Important:** A node-budget sweep does not achieve perfectly equal FLOPs across architectures; its purpose is to *reveal* where improvements appear or fail to appear as operation budgets change, and to prepare a defensible cost-matched study. Unsuccessful runs are retained. Running results are not treated as observations until a completed row appears in JSONL.
