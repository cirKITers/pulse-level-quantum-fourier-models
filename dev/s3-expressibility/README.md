# s3 — Expressibility

How does the expressibility of a circuit change when the
pulse parameters of its ansatz are perturbed?

**1530 runs** of the `expressibility` flow: 17 ansaetze × 9 pulse scaler
variances (0 to 0.008) × 10 sample seeds (1000–1009). Everything else stays at
the flow defaults: 3 qubits, 1 layer, ternary `RY`
encoding, 500 samples drawn jointly over the unitary parameters and the ansatz
pulse scalers, 50 histogram bins.

## How to re-run it

```sh
dev/serve.sh                                           # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms                   # upload the flows
uv run python dev/s3-expressibility/run.py --dry-run   # the grid, submitting nothing
uv run python dev/s3-expressibility/run.py --jobs 30   # submit it, resuming what finished
uv run python dev/s3-expressibility/figures.py         # the figures and study-3.csv, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine. `figures.py` reads the finished
runs back from it and writes this folder's `figures/` (gitignored).
