# Pulse Level Quantum Fourier Models

## Getting started

1. Install dependencies: `uv sync`
2. Run `uv run kedro run` to launch the default pipeline
3. Initialize a local mlflow config by running `uv run kedro mlflow init`
3. In a separate terminal, run `uv run mlflow ui` and navigate to `http://127.0.0.1:5000/#/experiments`

Note that the project currently requires python <3.13

## Slurm

For convenience, there are scripts for running experiments located in `./scripts`.
In `slurm-job.sh` you can specify the required resources and which study to run.
The other `study-x.sh` files then specify *what* is done in a single job.
To submit a job, you can use `sbatch`, e.g. `sbatch ./scripts/slurm-job.sh`