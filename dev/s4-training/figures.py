"""s4 -- figures and study-4.csv from finished `train` runs.

    dev/serve.sh &
    uv run python dev/s4-training/figures.py
"""

from pathlib import Path

from pulse_level_qfms import viz
from pulse_level_qfms.table import table

#: This study's own figures, gitignored.
FIGURES = Path(__file__).resolve().parent / "figures"

#: Which finished runs: all of them, or narrowed as `table` allows, e.g.
#: `{"group": "<sweep id>"}` or `{"since": "2026-09-01T00:00"}`. The 510
#: archived runs have `{"group": "mlflow:513065903306889723"}`.
SELECT = {}

if __name__ == "__main__":
    df = table("train", **SELECT)
    FIGURES.mkdir(exist_ok=True)
    # export the run table for further analysis
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
