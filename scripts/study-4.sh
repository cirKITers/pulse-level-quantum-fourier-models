#!/bin/bash

set -e

# run experiments with all different circuits
MAX_JOBS=20

# Hardware_Efficient Circuit_15 Circuit_17 Circuit_19
for circuit in Circuit_3 Circuit_16 Circuit_18 Circuit_7 
do
    # unitary-only baseline vs. joint unitary+pulse training
    for train_axis in "unitary" "unitary,pulse"
    do
        # 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
        for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
        do
            # Wait if we already have MAX_JOBS running
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 10
            done

            echo "--- $circuit Ansatz, Train Axis [$train_axis], Seed $seed ---"
            uv run kedro run --pipeline "study-4" --params="data.seed=$seed,model.circuit_type=$circuit,train.train_axis=[$train_axis]" &

            sleep 30
        done
    done
done

# Wait for all remaining jobs to complete
wait
echo "All seeds completed"
