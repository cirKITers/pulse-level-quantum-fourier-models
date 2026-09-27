# s2 — Fidelity

The paper's study 2. How far does a pulse-level circuit drift from its
unitary ideal, in fidelity and trace distance, when its pulse parameters are
perturbed?

**1530 runs** of the `fidelity` flow: 17 ansaetze × 9 pulse scaler variances
(0 to 0.008) × 10 sample seeds (1000–1009). Everything else stays at the flow
defaults, which are the paper's: 3 qubits, 1 layer, ternary `RY` encoding,
500 samples.

## How to re-run it

```sh
dev/serve.sh                                     # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms             # upload the flows
uv run python dev/s2-fidelity/run.py --dry-run   # the grid, submitting nothing
uv run python dev/s2-fidelity/run.py --jobs 30   # submit it, resuming what finished
uv run python dev/s2-fidelity/figures.py         # the figures and study-2.csv, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine. `figures.py` reads the finished
runs back from it and writes this folder's `figures/` (gitignored).
