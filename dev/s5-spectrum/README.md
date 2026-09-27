# s5 — Spectrum

Where do a model's Fourier coefficients sit when only the pulse parameters of
its encoding gates are perturbed, while the ansatz runs as a unitary? The
frequency axis is oversampled (`mts=4`, bin spacing 1/4), which is what makes
a frequency shift distinguishable from a change in amplitude.

**1530 runs** of the `spectrum` flow: 17 ansaetze × 9 pulse scaler variances
(0 to 0.008) × 10 sample seeds (1000–1009), at `n_layers=2`, which the driver
pins because the flow default is the paper's single layer
(`docs/DECISIONS.md` D5). Samples are drawn jointly over the unitary
parameters and the encoding pulse scalers.

## How to re-run it

```sh
dev/serve.sh                                     # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms             # upload the flows
uv run python dev/s5-spectrum/run.py --dry-run   # the grid, submitting nothing
uv run python dev/s5-spectrum/run.py --jobs 5    # submit it, resuming what finished
uv run python dev/s5-spectrum/figures.py         # the figures, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine. `figures.py` reads the finished
runs back from it and writes this folder's `figures/` (gitignored).
