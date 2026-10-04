"""s5 -- figures from the finished `spectrum` runs.

    dev/serve.sh &
    uv run python dev/s5-spectrum/figures.py
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
    df = table("spectrum", **SELECT)
    spectra = viz.spectrum_over_distortion(df, show_error=True)
    viz.save(
        {
            **{
                f"spectrum_{viz.circuit_name_to_str(ansatz)}": fig
                for ansatz, fig in spectra.items()
            },
            "offgrid_mass": viz.offgrid_mass_over_distortion(df, show_error=True),
        },
        FIGURES,
    )
