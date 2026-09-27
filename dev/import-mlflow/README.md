# import-mlflow — the paper's MLflow runs, in the engine

The paper's figures were drawn from MLflow runs made under kedro, before the
studies were flows. `run.py` imports exactly those runs, so the figures can be
redrawn from the engine like any native run. Which runs they are is fixed by the
CSVs the R plots were built from: the `run_id` column of `study-N.csv` at the
repo root (gitignored, like the MLflow stores).

| CSV | flow | runs | MLflow store |
| --- | --- | --- | --- |
| `study-1.csv` | `fcc` | 1530 | not located yet |
| `study-2.csv` | `fidelity` | 1530 | not located yet |
| `study-3.csv` | `expressibility` | 1530 | not located yet |
| `study-4.csv` | `train` | 510 | `mlruns_paper/513065903306889723` (`study-4-8`), imported |

Each run becomes a run of its flow with `cause` "import", its MLflow `run_id`
as `external_id`, the label `paper`, and the MLflow start and end as its times:

- **inputs**: `<group>.<name>` is `<name>`, except `model.seed`, `data.seed`,
  `<study>.seed` (`model_seed`, `data_seed`, `sample_seed`), the
  `train.rank_eval.*` trio (`rank_eval`, `rank_tol_rel`, `rank_report_interval`)
  and `train.train_pulse` (`gate_mode` `unitary` / `ansatz_pulse`). An input the
  run did not record takes the flow default, and is listed in
  `result.legacy.defaulted`.
- **outputs**: `model_spec` (envelope, RWA, frame and the `model.n_*` counts),
  plus what the flow's node reports: `fcc` and `coefficients`, `fidelity` and
  `trace_distance`, `expressibility`, or `training`.
- **curves** (`train`): `train_mse`, `pulse_scaler_*` and `rank_*` under the
  names and step counting of the training node, with `rank_step` carrying the
  training step each rank was measured at.
- **`result.legacy`**: the MLflow identity and tags, and every parameter and
  metric with no counterpart today (`train.train_unitary`, the `train_fmse`
  curve).

Not imported: artifacts (`model.txt`, `time_domain.html`), and what the flows
report that MLflow never recorded (`dataset_info`, the solver and summary in
`model_spec`, `trained_model`). `docs/DECISIONS.md` D9–D11 give the reasons.

## How to run it

```sh
dev/serve.sh &                                   # the engine, on ./.fluksio
uv run fluksio sync pulse_level_qfms             # the flows it imports into
uv run python dev/import-mlflow/run.py --dry-run # ids found, inputs defaulted, kept as legacy
uv run python dev/import-mlflow/run.py --mlruns /path/to/mlruns 1 2 3
```

`--mlruns` is an MLflow file store (default `mlruns_paper/`); runs are found by id
under `<mlruns>/<experiment_id>/<run_id>/` whatever the experiment. The engine
keeps the first import of an `external_id` and answers every later one with
`created: False`, so a changed mapping only lands after the flow's imported runs
are deleted. Check the dry run's `inputs from the flow defaults` first: an input
the run did not record is assumed to be at today's default.
