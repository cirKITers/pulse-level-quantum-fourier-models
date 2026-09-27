"""s6 -- figures from the finished `landscape` runs.

    dev/serve.sh &
    uv run python dev/s6-landscape/figures.py
"""

from pathlib import Path

from pulse_level_qfms import viz
from pulse_level_qfms.table import table

#: This study's own figures, gitignored.
FIGURES = Path(__file__).resolve().parent / "figures"

#: Which finished runs: all of them, or narrowed as `table` allows, e.g.
#: `{"group": "<sweep id>"}` or `{"since": "2026-09-01T00:00"}`.
SELECT = {}

if __name__ == "__main__":
    df = table("landscape", **SELECT)
    viz.save(
        {
            "loss_profile": viz.landscape_over_eta(df, "profile"),
            "loss_fixed": viz.landscape_over_eta(df, "fixed"),
            "scaling": viz.landscape_scaling(df),
            "scaling_frequencies": viz.landscape_scaling_frequencies(df),
            "over_circuits": viz.landscape_over_circuits(df),
        },
        FIGURES,
    )
