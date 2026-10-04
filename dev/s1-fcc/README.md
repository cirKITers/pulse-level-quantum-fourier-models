# s1 — Fourier coefficient concentration

## Question

How concentrated are a model's Fourier coefficients, and
how do the concentration (FCC) and the coefficient variances change when the
pulse parameters of the ansatz are perturbed?

## Method

**1530 runs** of the `fcc` flow: 17 ansaetze × 9 pulse scaler variances
(0 to 0.008) × 10 sample seeds (1000–1009). Everything else stays at the flow
defaults: 3 qubits, 1 layer, ternary `RY` encoding,
500 samples drawn jointly over the unitary parameters and the ansatz pulse
scalers.

## Reproduce

```sh
dev/serve.sh                                     # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms             # upload the flows
uv run python dev/s1-fcc/run.py --dry-run        # the grid, submitting nothing
uv run python dev/s1-fcc/run.py --jobs 15        # submit it, resuming what finished
uv run python dev/s1-fcc/figures.py              # the figures and study-1.csv, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine. `figures.py` reads the finished
runs back from it and writes this folder's `figures/` (gitignored).
