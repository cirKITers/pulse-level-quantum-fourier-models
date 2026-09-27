"""s7 -- training on an off-grid target, with the encoding at pulse level.

Two blocks over the `train_enc` flow.

The ablation block is one ansatz, one target, and the training changed one
thing at a time. The floor arms train on an on-grid target, where the
unitary parameters already reach every component, so whatever the pulse arm
gains there is not reachability. The rest separate initialisation, the
trainable-frequency knob and the learning rate from the pulse scalers
themselves.

The ansatz block asks whether the encoding pulse scalers reach frequencies
the unitary parameters cannot, so every ansatz is trained twice from the
same data seed: once with the encoding at gate level and once at pulse level.

    dev/serve.sh &
    uv run python dev/s7-ablation/run.py --jobs 1
"""

from pathlib import Path

from pulse_level_qfms.sweep import CIRCUITS, SEEDS, main

#: This study's own record of the sweep.
OUT = Path(__file__).resolve().parent / "results" / "driver.json"

#: The depth both blocks run at, which is not the paper's (docs/DECISIONS.md D5).
PINNED = {"n_layers": 2}

#: Shared by every ablation arm, so a difference between two rows is the arm.
COMMON = {
    **PINNED,
    "circuit_type": "Circuit_15",
    "encoding_strategy": "ternary",
    "steps": 2000,
    "rank_eval": False,
}

ABLATION_SEEDS = [1000, 1001, 1002]

#: What each arm changes, by the name the study reports it under.
ARMS = {
    # on-grid target: the unitary parameters can reach it, so this is the
    # floor both regimes are measured against
    "B0-floor-unitary": {"gate_mode": "unitary", "offgrid_prob": 0.0},
    "B0-floor-encpulse": {"gate_mode": "enc_pulse", "offgrid_prob": 0.0},
    "unitary": {"gate_mode": "unitary"},
    "encpulse-ones": {"gate_mode": "enc_pulse"},
    # start the scalers where the target is, which separates reachability
    # from the optimisation that has to find it
    "B2-encpulse-target": {"gate_mode": "enc_pulse", "enc_pulse_init": "target"},
    # the unitary trainable-frequency knob, as the alternative route to a
    # detuned comb
    "B3-enc-params": {"gate_mode": "unitary", "train_enc_params": True},
    "B4-lr1e-2": {"gate_mode": "enc_pulse", "pulse_learning_rate": 1e-2},
    "B6-hamming-encpulse": {"gate_mode": "enc_pulse", "encoding_strategy": "hamming"},
}

GATE_MODES = ["unitary", "enc_pulse"]


def cells():
    """The ablation block, then the ansatz block.

    An ablation cell is what its parameters say it is: one per arm and seed.
    """
    ablation = [
        {**COMMON, **arm, "data_seed": seed}
        for arm in ARMS.values()
        for seed in ABLATION_SEEDS
    ]
    ansaetze = [
        {
            **PINNED,
            "circuit_type": circuit,
            "gate_mode": gate_mode,
            "decompose_circuit": False,
            "data_seed": seed,
        }
        for circuit in CIRCUITS
        for gate_mode in GATE_MODES
        for seed in SEEDS
    ]
    return ablation + ansaetze


if __name__ == "__main__":
    main("train_enc", cells, seed_key="data_seed", out=OUT)
