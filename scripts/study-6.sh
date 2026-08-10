#!/bin/bash

# Scaling sweep for the encoding-scaler loss landscape: one run per encoding
# strategy and qubit count at a single layer. The per-gate hardness follows
# the largest generator, which grows with the spectrum size for binary and
# ternary and stays at one for hamming, so this sweep traces hardness over
# the number of frequencies with hamming as the flat control.
#
# offgrid_resolution is raised to 4: with the default 2, the displaced
# generators collide often enough that a slice reaches the target support at
# several scalers instead of one, which hides the basin the sweep is meant to
# show. mts must stay at or above the resolution so the domain covers a full
# period of the off-grid components.

set -e

MAX_JOBS=3

for strategy in hamming binary ternary; do
    for n_qubits in 2 3 4; do
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done

        echo "--- $strategy, $n_qubits qubits ---"
        uv run kedro run --pipeline study-6 \
            --params="data.offgrid_resolution=4,data.mts=4,model.encoding_strategy=$strategy,model.n_qubits=$n_qubits" &

        sleep 20
    done
done

wait
echo "All runs completed"
