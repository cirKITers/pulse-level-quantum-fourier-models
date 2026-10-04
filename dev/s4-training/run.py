"""s4 -- train with unitary, decomposed, and pulse-level parameters.

Every ansatz is trained on a Fourier series drawn on its own frequency comb,
in three arms: as a unitary circuit, decomposed into basis gates with a
trainable scaler on every structural angle, and with the ansatz at pulse
level. The encoding stays unitary in all three.

    dev/serve.sh &
    uv run python dev/s4-training/run.py --jobs 5
"""

from pathlib import Path

from pulse_level_qfms.sweep import CIRCUITS, SEEDS, main

#: This study's own record of the sweep.
OUT = Path(__file__).resolve().parent / "results" / "driver.json"

#: (gate_mode, decompose_circuit) of each arm. A decomposed circuit trains
#: as a unitary.
ARMS = [("unitary", False), ("unitary", True), ("ansatz_pulse", False)]

#: The archived runs recorded this value instead of the flow default.
#: The frame is ignored under the rotating-wave
#: approximation, but it is part of the parameter set all the same.
PINNED = {"frame": "lab"}


def cells():
    """Every ansatz, in every arm, repeated over the data seeds."""
    return [
        {
            **PINNED,
            "circuit_type": circuit,
            "gate_mode": gate_mode,
            "decompose_circuit": decompose,
            "data_seed": seed,
        }
        for circuit in CIRCUITS
        for gate_mode, decompose in ARMS
        for seed in SEEDS
    ]


if __name__ == "__main__":
    main("train", cells, seed_key="data_seed", out=OUT)
