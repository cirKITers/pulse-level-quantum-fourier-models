#!/bin/bash
set -e

# run experiments with all different circuits
MAX_JOBS=30

# Circuits without full spectrum
# for circuit in Circuit_3 Circuit_9 Circuit_10 Circuit_16 Circuit_18 Circuit_7 Circuit_13 Hardware_Efficient
# Circuits with full spectrum
# for circuit in Circuit_2 Circuit_4 Circuit_8 Circuit_14 Circuit_15 Circuit_17 Circuit_19 Circuit_20 Strongly_Entangling
# All circuits
for circuit in Circuit_2 Circuit_4 Circuit_8 Circuit_14 Circuit_15 Circuit_17 Circuit_19 Circuit_20 Strongly_Entangling Circuit_3 Circuit_9 Circuit_10 Circuit_16 Circuit_18 Circuit_7 Circuit_13 Hardware_Efficient
do
    for variance in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008
    do
        for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
        do
            # Wait if we already have MAX_JOBS running
            while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
                sleep 10
            done

            echo "--- $circuit Ansatz, Variance $variance, Seed $seed ---"
            uv run kedro run --pipeline "study-3" --params="expressibility.seed=$seed,expressibility.pulse_params_variance=$variance,model.circuit_type=$circuit" &

            sleep 20
        done
    done
done

# Wait for all remaining jobs to complete
wait
echo "All seeds completed"