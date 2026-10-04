"""s6 -- how the loss landscape scales with the encoding.

Two blocks over the same flow. The first varies what sets the size of the
generators -- the encoding strategy and the qubit count -- because the
oscillation period along a scaler is inversely proportional to the generator
it drives. The second varies the ansatz at a fixed encoding, which changes
what the variational parameters can do with the comb rather than the comb.

    dev/serve.sh &
    uv run python dev/s6-landscape/run.py --jobs 5
"""

from pathlib import Path

from pulse_level_qfms.sweep import SEEDS, main

#: This study's own record of the sweep.
OUT = Path(__file__).resolve().parent / "results" / "driver.json"

#: A finer off-grid resolution than the flow default, so the target sits
#: between the comb lines rather than near one. This study uses two layers.
COMMON = {"offgrid_resolution": 4, "mts": 4, "n_layers": 2}

STRATEGIES = ["hamming", "binary", "ternary"]
QUBITS = [2, 3, 4]

CIRCUITS = [
    "Circuit_2",
    "Circuit_4",
    "Circuit_8",
    "Circuit_14",
    "Circuit_15",
    "Circuit_17",
    "Circuit_19",
    "Circuit_20",
    "Strongly_Entangling",
    "Circuit_3",
    "Circuit_9",
    "Circuit_10",
    "Circuit_16",
    "Circuit_18",
    "Circuit_7",
    "Circuit_13",
]


def cells():
    """The scaling block, then the ansatz block, deduplicated by the driver."""
    scaling = [
        {
            **COMMON,
            "encoding_strategy": strategy,
            "n_qubits": n_qubits,
            "model_seed": seed,
            "data_seed": seed,
        }
        for strategy in STRATEGIES
        for n_qubits in QUBITS
        for seed in SEEDS
    ]
    ansaetze = [
        {**COMMON, "circuit_type": circuit, "model_seed": seed, "data_seed": seed}
        for circuit in CIRCUITS
        for seed in SEEDS
    ]
    return scaling + ansaetze


if __name__ == "__main__":
    main("landscape", cells, seed_key="data_seed", out=OUT)
