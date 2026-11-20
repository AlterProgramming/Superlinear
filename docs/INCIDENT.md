# Confirmed numerical failure: 16-node routed model, quartic task, seed 2

**This was actually observed and independently reproduced**, not a hypothetical risk.

- Architecture: 16-node sequential learned-operand multiplicative computation with input-dependent hard execution routes.
- Data: **synthetic** degree-4 target built from four hidden random orthogonal projections of 12 independent uniform inputs.
- 180 Adam gradient updates, selected epoch 179 from validation loss (no test-based selection).
- Train normalized MSE **0.12964**, validation **1.23604**, test **103,583,719,424**.
- Worst individual test prediction **−7,282,497.5**, target **6.3446** (test index 114). No input scale extrapolation: this happened on the held-out IID test set.
- Trace of the anomalous sample reveals multiplication amplification: node 8 ≈ 26.1, node 11 ≈ −127.2, node 12 ≈ 486.2, node 14 ≈ −70,248, node 15 ≈ −41.6 million. The last readout produces a −7.28 million prediction.
- **Ten node-16 runs** have non-finite synthetic wide-support OOD scores (NaN or infinity), recorded as `ood_normalized_mse: null` plus `ood_nonfinite: "nan"` or `"inf"`; do not silently filter those failures. All records are retained in the sweep JSONL.

## Inference

The architecture's repeated multiplication of unconstrained intermediate results can suffer catastrophic amplification under rare input configurations. Gradient clipping controls optimization gradient norm, **not the forward-state magnitude**. Validation-set selection did not catch the rare test point. This is one architectural obstacle to high-order generalization.

A potential next remedy is a theoretically controlled state representation (e.g., sign/log-magnitude paired with bounded residual channels, normalization whose effect on representation is explicit, or explicit degree/scale tracking). These are proposals **not yet implemented or shown successful**. A stabilization mechanism must be tested on the same hidden-order tasks and real datasets, with representational capacity and cost accounting preserved.

## Reproduce

```bash
python -m multisuper.inspect_outlier
```

The model weights for the reproduced incident are saved in `results/incident_quartic16_seed2/model.pt`; per-dataset errors and the node trace are in `incident.json`. Do not deserialize untrusted weights with unsafe pickle loading.
