# s6 — Loss landscape

Why are the encoding pulse scalers hard to train? The flow sweeps the
amplitude scaler of one encoding gate at a time against an off-grid target
(`offgrid_mode=generator`) and records the loss along it, one curve per
encoding generator, next to its analytic counterpart. The loss along a scaler
is expected to oscillate with a period set by the generator it acts on, so the
questions are how the basin around the aligned scaler and the number of local
minima on the way to it scale with the generator.

**Two blocks** of the `landscape` flow, each cell with `model_seed = data_seed`
over 10 seeds (1000–1009), all at `n_layers=2` (pinned, `docs/DECISIONS.md`
D5), `mts=4` and `offgrid_resolution=4`, so the target sits between the comb
lines rather than near one:

| block | axis | cells |
| --- | --- | --- |
| scaling | `Circuit_3`, `encoding_strategy` ∈ {hamming, binary, ternary} × `n_qubits` ∈ {2, 3, 4} | 90 |
| ansatz | 16 ansaetze (all of s1's but `Hardware_Efficient`), ternary, 3 qubits | 160 |

The two blocks share their `Circuit_3`/ternary/3-qubit corner, which the
driver submits once.

## How to re-run it

```sh
dev/serve.sh                                     # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms             # upload the flows
uv run python dev/s6-landscape/run.py --dry-run  # the grid, submitting nothing
uv run python dev/s6-landscape/run.py --jobs 5   # submit it, resuming what finished
uv run python dev/s6-landscape/figures.py        # the figures, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine. `figures.py` reads the finished
runs back from it and writes this folder's `figures/` (gitignored).
