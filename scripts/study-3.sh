#!/bin/bash

set -e

# run experiments with all different circuits
MAX_JOBS=20

# Hardware_Efficient Circuit_15 Circuit_17 Circuit_19
for circuit in Hardware_Efficient Circuit_15 Circuit_16 Circuit_17 Circuit_18 Circuit_19
do
    # 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.010 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.020
    for variance in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.010
    do
        # 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
        for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
        do
            # Wait if we already have MAX_JOBS running
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 10
            done

            echo "--- $circuit Ansatz, Variance $variance, Seed $seed ---"
            uv run kedro run --pipeline "study-2" --params="expressibility.seed=$seed,expressibility.pulse_params_variance=$variance,model.circuit_type=$circuit" &

            sleep 30
        done
    done
done

# Wait for all remaining jobs to complete
wait
echo "All seeds completed"
