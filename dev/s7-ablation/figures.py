"""s7 -- figures from the finished `train_enc` runs of the ansatz block.

The ablation block is left out: its arms change more than the gate mode (the
target, the initialisation, the learning rate), so they are not bars of the
same comparison.

    dev/serve.sh &
    uv run python dev/s7-ablation/figures.py
"""

from pathlib import Path

from pulse_level_qfms import viz
from pulse_level_qfms.sweep import client
from pulse_level_qfms.table import table

#: This study's own figures, gitignored.
FIGURES = Path(__file__).resolve().parent / "figures"

#: Which finished runs: all of them, or narrowed as `table` allows, e.g.
#: `{"group": "<sweep id>"}` or `{"since": "2026-09-01T00:00"}`.
SELECT = {}


def ansatz_block() -> list:
    """The runs of the ansatz block: the ablation arms train with ranks off."""
    runs = client().export_runs(
        flow="train_enc", status="ok", params="rank_eval", **SELECT
    )
    return [run["id"] for run in runs if run["param.rank_eval"]]


if __name__ == "__main__":
    ids = ansatz_block()
    if not ids:
        raise SystemExit("no finished runs of the ansatz block of 'train_enc'")
    df = table("train_enc", ids=ids)
    scaler_mean, scaler_std = viz.pulse_mean_and_variance_over_step(df, show_error=True)
    viz.save(
        {
            "mse": viz.pulse_param_mse_comparison(df, show_error=True),
            "enc_pulse_scaler_mean": scaler_mean,
            "enc_pulse_scaler_std": scaler_std,
            "loss": viz.loss_over_step(df, show_error=True),
        },
        FIGURES,
    )
