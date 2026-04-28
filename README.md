# Pulse Level Quantum Fourier Models

## Getting started :rocket:

1. Install dependencies: `uv sync`
2. Run `uv run kedro run --pipeline study-x` to run a study (see below for available studies)
3. Initialize a local mlflow config by running `uv run kedro mlflow init`
4. In a separate terminal, run `uv run mlflow ui` and navigate to `http://127.0.0.1:5000/#/experiments`

## Studies :book:

We sort the different experiments in our work into the studies below.
Each study can be run using `uv run kedro run --pipeline study-x`.

- **Study-1**: This study primarily focusses on the FCC and coefficient variances and how they change when the pulse parameters are perturbed.
- **Study-2**: In this study, we evaluate how the fidelity and trace distance changes when the pulse parameters are perturbed.
- **Study-3**: Similarly to study-1 but we look at the expressibility instead.
- **Study-4**:This study evaluates the different circuits by training on a Fourier series dataset. Here, the pulse parameters are either included in the optimization or not.

You can associate an MlFlow experiment to a study by setting the name of the study in `./conf/local/mlflow.yml` under `tracking.experiment.name`.

## Slurm :cloud:

For convenience, there are scripts for running experiments located in `./scripts`.
In `slurm-job.sh` you can specify the required resources and which study to run.
The other `study-x.sh` files then specify *what* is done in a single job.
To submit a job, you can use `sbatch`, e.g. `sbatch ./scripts/slurm-job.sh`

You can view all existing experiments using `uv run mlflow experiments search`.

## Tweaking :wrench:

You can change the parameters in `conf/base/parameters.yml`.
The parameters listed there are sorted into categories (`model`, `fcc`, `expressibility`, `training` and `data`).
Note that command line arguments take precedence over parameters in this file, thus the Slurm scripts will overwrite some parameters.

## Visualization :chart_with_upwards_trend:

After running the experiments, results are stored in `./mlruns`.
To generate plotly figures and a `.csv` export (which we use for the R plots in the paper), you can head over to `./notebooks/generate_plots.py`, enter the name of the studies you want to include and run `uv run python ./notebooks/generate_plots.py`.
Note that this requires that you have set the name of the study accordingly in `./conf/local/mlflow.yml` under `tracking.experiment.name`.
The results are stored in `./results` in a subfolder corresponding to the mlflow experiment id (see command line output).