# s4 — Training

Does training the pulse-level parameters of the ansatz, or
the scalers of its basis-gate decomposition, fit a Fourier series better than
the unitary parameters alone?

**510 runs** of the `train` flow: 17 ansaetze × 3 arms × 10 data seeds
(1000–1009). The target is drawn on the model's own frequency comb over one
period, so every arm can represent it; the encoding stays unitary throughout.

| arm | `gate_mode` | `decompose_circuit` |
| --- | --- | --- |
| unitary | `unitary` | `false` |
| decomposed | `unitary` | `true` |
| ansatz pulse | `ansatz_pulse` | `false` |

Everything else stays at the flow defaults (3 qubits,
1 layer, ternary `RY` encoding, 2000 steps at 1e-3, Jacobian ranks every 50
steps), except `frame=lab`, which the archived runs recorded although the
rotating-wave approximation makes it inert.

The archived runs are in the MLflow experiment `study-4-8`
(`mlruns_paper/513065903306889723`, parameters and metrics only). Each of its
510 parameter sets is exactly one cell of `run.py`, under this mapping:

| MLflow parameter | `train` input |
| --- | --- |
| `model.<name>`, `data.<name>`, `train.<name>` | `<name>` |
| `model.seed` | `model_seed` |
| `data.seed` | `data_seed` |
| `data.batch_size` | `batch_size` |
| `train.rank_eval.enabled` / `.tol_rel` / `.report_interval` | `rank_eval` / `rank_tol_rel` / `rank_report_interval` |
| `train.train_pulse` `False` / `True` | `gate_mode` `unitary` / `ansatz_pulse` |

`model.n_gate_params`, `model.n_pulse_params`, `model.n_decomposed_param_slots`
and `model.n_scaler_params` are results (`model_spec`), and
`train.train_unitary` was always `True`, which the flow does unconditionally.
Inputs absent from the archived runs use their inert on-grid defaults:
`mts=1`, `mfs=1`, `offgrid_mode=none`, `offgrid_prob=0`, `offgrid_resolution=2`,
`enc_pulse_init=ones`, `train_enc_params=false`.

Those 510 runs are in the engine, imported by `dev/import-mlflow/run.py`, so the
driver finds every cell done; a fresh run of the study needs them deleted first.

## How to re-run it

```sh
dev/serve.sh                                     # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms             # upload the flows
uv run python dev/s4-training/run.py --dry-run   # the grid, submitting nothing
uv run python dev/s4-training/run.py --jobs 5    # submit it, resuming what finished
uv run python dev/s4-training/figures.py         # the figures and study-4.csv, into figures/
```

The driver writes one record per finished cell to this folder's `results/driver.json`
(gitignored); the runs themselves live in the engine. `figures.py` reads the finished
runs back from it and writes this folder's `figures/` (gitignored).
