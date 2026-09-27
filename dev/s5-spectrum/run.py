"""s5 -- Frequency spectrum over ansatz, distortion and seed.

    dev/serve.sh &
    uv run python dev/s5-spectrum/run.py --jobs 5
"""

from pathlib import Path

from pulse_level_qfms.sweep import CIRCUITS, SEEDS, VARIANCES, main

#: This study's own record of the sweep.
OUT = Path(__file__).resolve().parent / "results" / "driver.json"

#: The depth this study runs at, which is not the paper's (docs/DECISIONS.md D5).
PINNED = {"n_layers": 2}


def cells():
    """Every ansatz, at every pulse variance, repeated over the seeds."""
    return [
        {
            **PINNED,
            "circuit_type": circuit,
            "pulse_params_variance": variance,
            "sample_seed": seed,
        }
        for circuit in CIRCUITS
        for variance in VARIANCES
        for seed in SEEDS
    ]


if __name__ == "__main__":
    main("spectrum", cells, seed_key="sample_seed", out=OUT)
