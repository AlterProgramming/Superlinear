# Audit, provenance, and verification

The recorded results were produced by executing the checked-in Python models, with explicit seeded data and validation-selected checkpoints. **Every dataset and every model run is labeled as synthetic or real_observational**. The term real means a curated observed sklearn source, not a deployment stream.

Command records:

```bash
# Initial single-seed smoke pilot: 30 runs, 80 steps
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m multisuper.experiment --datasets bilinear quartic switched additive breast_cancer digits --models linear mlp fixed_random learned_product routed_product --seeds 0 --epochs 80 --out results/pilot_one_seed

# Main study: 11 datasets × 5 models × 8 seeds = 440 runs
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m multisuper.experiment --epochs 180 --seeds 0 1 2 3 4 5 6 7 --out results/main
python -m multisuper.analyze results/main

# Two scaling studies: 250 + 180 = 430 runs
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m multisuper.sweep --kind nodes --epochs 180 --seeds 0 1 2 3 4
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m multisuper.sweep --kind samples --epochs 180 --seeds 0 1 2 3 4
python -m multisuper.sweep_report results/node_sweep/all_runs.jsonl
python -m multisuper.sweep_report results/sample_sweep/all_runs.jsonl

# Post-hoc diagnostic: 4 tasks × 5 seeds = 20 runs, 360 updates each
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m multisuper.diagnose

# Reproduce the anomalous model with its saved weights and per-node tracing
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m multisuper.inspect_outlier

# Unit / property / representability tests
PYTHONPATH=src pytest -q
```

`results/manifest.json` holds content SHA-256 checksums of the main result files and version metadata. Results were generated on the current CPU environment; replaying with different Torch versions/hardware can produce different timings and minor learning differences. CPU inference timing uses single-thread eager-mode execution and is **not an optimized-kernel ceiling**. The results are deliberately not aggregated across synthetic and real sources.

The 10 missing OOD scores in the 16-node sweep reflect actually observed NaN or infinity, preserved as explicit `*_nonfinite` flags in strictly valid JSON instead of being discarded. An unusually large but finite IID test failure remains in the raw score distribution, not silently excluded.
