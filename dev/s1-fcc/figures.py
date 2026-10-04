"""s1 -- figures and study-1.csv from finished `fcc` runs.

    dev/serve.sh &
    uv run python dev/s1-fcc/figures.py
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
    df = table("fcc", **SELECT)
    FIGURES.mkdir(exist_ok=True)
    # export the run table for further analysis
    df.to_csv(FIGURES / "study-1.csv", index=False)
    viz.save(
        {
            "fcc": viz.fcc_over_distortion(df, show_error=False),
            "coeff_var_ratio": viz.coeff_var_delta_over_distortion(
                df, show_error=False
            ),
            "n_frequencies": viz.frequency_histogram_by_distortion(
                df, threshold=1e-4, show_error=False
            ),
        },
        FIGURES,
    )
