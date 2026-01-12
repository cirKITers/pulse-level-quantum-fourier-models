# Spectral Bias X FCC

## Getting started

1. Install dependencies: `uv sync`
2. Run `uv run kedro run` to launch the default pipeline
3. In a separate terminal, run `uv run mlflow ui` and navigate to `http://127.0.0.1:5000/#/experiments`

Note that the project currently requires python <3.13


## Study 1

1. given the parameters
   - *common model parameters*
   - `encoding`
   - `model.seed`
   - `data.seed`
2. calculate the FCC
3. train the model
   - record frequency-wise loss
4. aggregate results
5. generate plot