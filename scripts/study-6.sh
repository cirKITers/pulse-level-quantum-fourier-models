#!/bin/bash

# Loss landscape of the encoding scalers, in two blocks.
#
# Block A traces hardness over the size of the spectrum: the per-gate geometry
# follows the largest generator, which grows with the spectrum size for binary
# and ternary and stays at one for hamming, so sweeping strategies and qubit
# counts spans the frequency axis with hamming as the flat control.
#
# Block B varies the trainable unitary at a fixed encoding. The Dirichlet
# geometry is a property of the encoding generators alone, so the oscillation
# period of the loss along a scaler should be the same for every ansatz. This
# block is what tests that.
#
# offgrid_resolution is raised to 4: with the default 2, the displaced
# generators collide often enough that a slice reaches the target support at
# several scalers instead of one, which hides the basin the sweep is meant to
# show. mts must stay at or above the resolution so the domain covers a full
# period of the off-grid components.

set -e

MAX_JOBS=6
SEEDS="1000 1001 1002"
CIRCUITS="Circuit_2 Circuit_3 Circuit_4 Circuit_8 Circuit_9 Circuit_10 Circuit_13 Circuit_14 Circuit_15 Circuit_16 Circuit_17 Circuit_18 Circuit_19 Circuit_20 Strongly_Entangling Hardware_Efficient"
COMMON="data.offgrid_resolution=4,data.mts=4"

run() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done
    uv run kedro run --pipeline study-6 --params="$COMMON,$1" &
    sleep 20
}

# Block A: spectrum size, one ansatz
for strategy in hamming binary ternary; do
    for n_qubits in 2 3 4; do
        for seed in $SEEDS; do
            echo "--- A: $strategy, $n_qubits qubits, seed $seed ---"
            run "model.encoding_strategy=$strategy,model.n_qubits=$n_qubits,model.seed=$seed,data.seed=$seed"
        done
    done
done

# Block B: trainable unitary, fixed encoding
for circuit in $CIRCUITS; do
    for seed in $SEEDS; do
        echo "--- B: $circuit, seed $seed ---"
        run "model.circuit_type=$circuit,model.seed=$seed,data.seed=$seed"
    done
done

wait
echo "All runs completed"
