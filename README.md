# Pulse Level Quantum Fourier Models

Studies of what a quantum Fourier model gains, and loses, when its gates run as
calibrated pulses rather than as ideal unitaries. s1–s4 examine ansatz pulses;
s5–s8 examine pulse-level encodings.

Tech stack:
- [qml-essentials](https://github.com/cirKITers/qml-essentials): quantum Fourier models
- [jaqsi](https://github.com/cirKITers/jaqsi): simulator in JAX
- JAX: array computation and automatic differentiation
- Fluksio: data pipeline and experiment tracking


## Layout

```
pulse_level_qfms/     the model and what is measured on it -- nothing study-specific
├── model.py          the circuit, optionally decomposed into basis gates, and how it
│                     travels between nodes
├── data.py           the target Fourier series, on or off the model's frequency comb
├── fcc.py            Fourier coefficient concentration, pulse level as a sampling axis
├── fidelity.py       fidelity and trace distance of the distorted circuit
├── expressibility.py expressibility, pulse level as a sampling axis
├── spectrum.py       the oversampled spectrum under encoding pulse distortion
├── training.py       the fit, with Jacobian ranks along the way
├── landscape.py      the loss along each encoding pulse scaler
├── utils.py          losses
├── pipeline.py       the seven Fluksio flows
├── sweep.py          the bounded, resumable grid submitter every study shares
├── table.py          a flow's finished runs, read from the engine as one table
└── viz.py            the figures, plotted from that table
dev/                  the research: one folder per study, plus the engine script
├── serve.sh          the engine, on ./.fluksio
├── export.sh         every flow's runs and curves, as CSV
├── import-mlflow/    archived MLflow runs (study-N.csv), into the engine
└── sN-<name>/        README.md (question, grid, how to re-run), run.py (the driver) and
                      figures.py; each keeps its own results/ figures/ logs/
notebooks/            exploratory scripts
tests/                run with `uv run pytest`
```

A flow is named by what it computes; a study is a driver over one flow:

| study | flow | runs | question |
| --- | --- | --- | --- |
| `s1-fcc` | `fcc` | 1530 | Fourier coefficient concentration under ansatz pulse distortion |
| `s2-fidelity` | `fidelity` | 1530 | fidelity and trace distance under the same distortion |
| `s3-expressibility` | `expressibility` | 1530 | expressibility under the same distortion |
| `s4-training` | `train` | 510 | training with unitary, decomposed and pulse-level ansatz parameters |
| `s5-spectrum` | `spectrum` | 1530 | where the coefficients sit under encoding pulse distortion |
| `s6-landscape` | `landscape` | 240 | the loss along each encoding pulse scaler |
| `s7-ablation` | `train_enc` | 364 | training on an off-grid target with the encoding at pulse level |
| `s8-trained-slice` | `landscape` | 102 | the loss along each encoding pulse scaler with the ansatz trained along the sweep |

Every flow reports `model_spec`: the parameter counts and the pulse configuration
the run was built under.

Everything a run reads or writes is gitignored: `.fluksio/` (the engine's store,
at the repo root), the root `logs/` and `results/`, and each study's `results/`,
`figures/` and `logs/`. So are the archived MLflow files (`mlruns*/`, the
tarballs, `study-*.csv`). Archived s4 runs are imported into the engine
(`dev/import-mlflow`); the s1–s3 runs have not been located.

## Getting started

1. Install dependencies: `uv sync`
2. Start the engine: `dev/serve.sh` (`RUNS=n` sets how many runs it takes at once;
   JAX sizes its thread pool per worker, so on a shared machine start low)
3. Upload the flows: `uv run fluksio sync pulse_level_qfms`
4. Run one: `uv run fluksio run fcc --sync pulse_level_qfms --circuit_type Circuit_15 --wait`
5. Run a study: `uv run python dev/s1-fcc/run.py --dry-run` shows its grid; without
   `--dry-run` it submits it, `--jobs` at a time, and on a second call only what has
   no finished run yet

Pass `--sync pulse_level_qfms` to every `fluksio run`: the default sync root is the
whole directory, which would import the drivers and notebooks too.

Every input and its default is in `pulse_level_qfms/pipeline.py`. A study using
other values pins them in its driver. Override an input on the command line
(`--circuit_type Circuit_15`), in a driver's grid, or in
`Client.submit(flow, params)`. Final numbers are read with
`fluksio export runs`, streamed curves with `fluksio export metrics`; `dev/export.sh`
writes both for every flow.

## Visualization

`uv run python dev/sN-<name>/figures.py` draws a study's figures from the engine's
finished runs into its `figures/`; s1–s4 also write `study-N.csv` exports.
The PDFs go through kaleido, which needs Chrome (`uv run plotly_get_chrome`).
