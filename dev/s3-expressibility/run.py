"""s3 -- Expressibility over ansatz, distortion and seed.

    dev/serve.sh &
    uv run python dev/s3-expressibility/run.py --jobs 30
"""

from pathlib import Path

from pulse_level_qfms.sweep import CIRCUITS, SEEDS, VARIANCES, main

#: This study's own record of the sweep.
OUT = Path(__file__).resolve().parent / "results" / "driver.json"


def cells():
    """Every ansatz, at every pulse variance, repeated over the seeds."""
    return [
        {
            "circuit_type": circuit,
            "pulse_params_variance": variance,
            "sample_seed": seed,
        }
        for circuit in CIRCUITS
        for variance in VARIANCES
        for seed in SEEDS
    ]


if __name__ == "__main__":
    main("expressibility", cells, seed_key="sample_seed", out=OUT)
