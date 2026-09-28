"""s8 -- the finished `landscape` runs of this study as study-8.csv.

    dev/serve.sh &
    uv run python dev/s8-trained-slice/figures.py

Basin width and minima density are read off the slices where the thesis
figures are drawn, from this table.
"""

from pathlib import Path

from pulse_level_qfms.table import table

#: This study's own output, gitignored.
FIGURES = Path(__file__).resolve().parent / "figures"

if __name__ == "__main__":
    df = table("landscape")
    df = df[df["landscape.steps"] > 0]  # s6's untrained sweeps share the flow
    FIGURES.mkdir(exist_ok=True)
    df.to_csv(FIGURES / "study-8.csv", index=False)
    print(f"{len(df)} runs -> {FIGURES / 'study-8.csv'}")
