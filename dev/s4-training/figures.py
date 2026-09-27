"""s4 -- figures, and the paper's study-4.csv, from the finished `train` runs.

    dev/serve.sh &
    uv run python dev/s4-training/figures.py
"""

from pathlib import Path

from pulse_level_qfms import viz
from pulse_level_qfms.table import table

#: This study's own figures, gitignored.
FIGURES = Path(__file__).resolve().parent / "figures"

#: Which finished runs: all of them, or narrowed as `table` allows, e.g.
#: `{"group": "<sweep id>"}` or `{"since": "2026-09-01T00:00"}`. The paper's
#: 510, imported from MLflow, are `{"group": "mlflow:513065903306889723"}`.
SELECT = {}

if __name__ == "__main__":
    df = table("train", **SELECT)
    FIGURES.mkdir(exist_ok=True)
    # the table the paper's R plots were built from
    df.to_csv(FIGURES / "study-4.csv", index=False)
    scaler_mean, scaler_std = viz.pulse_mean_and_variance_over_step(df, show_error=True)
    viz.save(
        {
            "mse": viz.pulse_param_mse_comparison(df, show_error=True),
            "pulse_scaler_mean": scaler_mean,
            "pulse_scaler_std": scaler_std,
            "loss": viz.loss_over_step(df, show_error=True),
        },
        FIGURES,
    )
