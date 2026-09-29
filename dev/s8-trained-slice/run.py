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
#: eta = 1 to the target. The ansatz is fitted at the target (fit_steps, 500 by
#: default), then continued outward with 20 Adam steps per scaler.
COMMON = {"offgrid_resolution": 4, "steps": 20, "eta_window": 3.0}

#: A pilot over the range of pulse parameter counts. C9 is left out: its
#: gamma = 3 slice is flat.
CIRCUITS = [
    "Circuit_10",
    "Circuit_2",
    "Circuit_3",
    "Circuit_15",
    "Strongly_Entangling",
    "Circuit_14",
]

#: The seeds of the thesis' fixed slices.
SEEDS = [1000, 1001, 1002]

#: theta alone first: it takes minutes, the pulse scalers take hours.
GATE_MODES = ["enc_pulse", "all_pulse"]


def cells():
    """Every ansatz and seed, once per gate mode."""
    return [
        {
            **COMMON,
            "gate_mode": mode,
            "circuit_type": circuit,
            "model_seed": seed,
            "data_seed": seed,
        }
        for mode in GATE_MODES
        for circuit in CIRCUITS
        for seed in SEEDS
    ]


if __name__ == "__main__":
    main("landscape", cells, seed_key="data_seed", out=OUT)
