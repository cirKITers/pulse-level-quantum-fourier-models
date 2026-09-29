"""s8 -- the fixed slice of s6, with the ansatz trained along the sweep.

The fixed slice holds the variational parameters at their initial values, so
its ripples mix the Dirichlet geometry of the window with coefficients that do
not fit the target. Training the ansatz along the sweep moves the slice toward
the concentrated one as far as the ansatz can realize free coefficients, and
the trainable-gate pulse scalers enlarge what it can realize. Every cell is
therefore trained twice from the same parameters: theta alone (enc_pulse) and
theta with the pulse scalers (all_pulse).

    dev/serve.sh &
    uv run python dev/s8-trained-slice/run.py --jobs 18
"""

from pathlib import Path

from pulse_level_qfms.sweep import main

#: This study's own record of the sweep.
OUT = Path(__file__).resolve().parent / "results" / "driver.json"

#: s6's ansatz block as the thesis ran it (one layer, ternary on three qubits,
#: all flow defaults), within three oscillation periods of the stretch from
#: eta = 1 to the target. eta_max reaches past 2 so that a gamma = 1 target at
#: 1.75 keeps its right flank. The ansatz is fitted at the target from five
#: starts, then continued out and back with 20 Adam steps per scaler, for up
#: to three rounds (the fit defaults of the flow).
COMMON = {"offgrid_resolution": 4, "steps": 20, "eta_window": 3.0, "eta_max": 2.5}

#: The sixteen ansaetze of the thesis' fixed slices, costliest first by pulse
#: parameter count, so the long all_pulse runs start early. C9's gamma = 3
#: slice is flat and drops out of the statistics, as it does in the thesis.
CIRCUITS = [
    "Circuit_14",
    "Circuit_19",
    "Strongly_Entangling",
    "Circuit_13",
    "Circuit_8",
    "Circuit_18",
    "Circuit_20",
    "Circuit_4",
    "Circuit_17",
    "Circuit_3",
    "Circuit_16",
    "Hardware_Efficient",
    "Circuit_15",
    "Circuit_2",
    "Circuit_9",
    "Circuit_10",
]

#: The seeds of the thesis' fixed slices.
SEEDS = [1000, 1001, 1002]

#: Whether 20 steps per scaler is enough: three of the ansaetze once more at
#: three times the steps.
SENSITIVITY = [
    {**COMMON, "steps": 60, "circuit_type": circuit, "model_seed": 1000}
    | {"data_seed": 1000}
    for circuit in ("Circuit_15", "Circuit_3", "Circuit_10")
]


def cells():
    """theta alone first, it takes minutes; then theta with the pulse scalers,
    which takes hours, the sensitivity runs being the longest."""
    grid = [
        {**COMMON, "circuit_type": circuit, "model_seed": seed, "data_seed": seed}
        for circuit in CIRCUITS
        for seed in SEEDS
    ]
    return [
        {**cell, "gate_mode": mode}
        for mode, block in (
            ("enc_pulse", grid + SENSITIVITY),
            ("all_pulse", SENSITIVITY + grid),
        )
        for cell in block
    ]


if __name__ == "__main__":
    main("landscape", cells, seed_key="data_seed", out=OUT)
