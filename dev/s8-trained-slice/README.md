# s8 - Trained slice

Does training the ansatz at every encoding scaler bring the fixed slice of s6
toward the concentrated one, and do the trainable-gate pulse scalers bring it
closer than the unitary parameters alone? The fixed slice narrows the basin to
about 0.6 of its Dirichlet prediction; the concentrated slice, which fits free
coefficients, does not. The `trained` curve of the `landscape` flow trains a
copy of the initial parameters at every scaler for `steps` Adam steps, from
the same start as the fixed slice.

36 runs of the `landscape` flow: 6 ansaetze (C10, C2, C3, C15,
Strongly_Entangling, C14; C9 is left out, its $\gamma = 3$ slice is flat)
$\times$ 3 seeds (1000 to 1002, `model_seed = data_seed`) $\times$ 2 gate modes
(`enc_pulse` trains $\theta$, `all_pulse` $\theta$ and $\kappa$). Everything
else is the thesis' fixed-slice sweep: one layer, ternary encoding on three
qubits, `mts=4`, `offgrid_resolution=4`, 20 points per oscillation. Each
gate's grid keeps the stretch from $\eta = 1$ to its target plus three
oscillation periods on either side (`eta_window=3`), which holds everything
basin width and minima density read, and training runs 100 steps at learning
rate 1e-2, eight scalers per chunk (about 1.6 GB per run). Each gate records
the mean loss over its grid after every tenth of the steps, so how far the
training settled is on the run.

The `all_pulse` runs dominate: a step costs about 30 times one with $\theta$
alone, and on 20 cores the 18 of them take roughly half a day at once.

## How to re-run it

```sh
RUNS=18 dev/serve.sh                                 # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms                 # upload the flows
uv run python dev/s8-trained-slice/run.py --dry-run  # the grid, submitting nothing
uv run python dev/s8-trained-slice/run.py --jobs 18  # submit it, resuming what finished
uv run python dev/s8-trained-slice/figures.py        # study-8.csv, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine.
