"""s2 -- figures, and the paper's study-2.csv, from the finished `fidelity` runs.

    dev/serve.sh &
    uv run python dev/s2-fidelity/figures.py
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
    df = table("fidelity", **SELECT)
    FIGURES.mkdir(exist_ok=True)
    # the table the paper's R plots were built from
    df.to_csv(FIGURES / "study-2.csv", index=False)
    viz.save(
        {
            "infidelity": viz.fidelity_over_distortion(df, show_error=True),
            "trace_distance": viz.trace_distance_over_distortion(df, show_error=True),
        },
        FIGURES,
    )
