#!/bin/bash

set -e

# run experiments with all different circuits
MAX_JOBS=20

# Circuits without full spectrum
# for circuit in Circuit_3 Circuit_9 Circuit_10 Circuit_16 Circuit_18 Circuit_7 Circuit_13 Hardware_Efficient
# Circuits with full spectrum
# for circuit in Circuit_2 Circuit_4 Circuit_8 Circuit_14 Circuit_15 Circuit_17 Circuit_19 Circuit_20 Strongly_Entangling
# All circuits
for circuit in Circuit_2 Circuit_4 Circuit_8 Circuit_14 Circuit_15 Circuit_17 Circuit_19 Circuit_20 Strongly_Entangling Circuit_3 Circuit_9 Circuit_10 Circuit_16 Circuit_18 Circuit_7 Circuit_13 Hardware_Efficient
do
    for train_pulse in False True
    do
        for decompose_circuit in False True
        do
            # skip those combinations
            if [[ "$train_pulse" == "True" && "$decompose_circuit" == "True" ]]; then
                continue
            fi

            for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
            do
                # Wait if we already have MAX_JOBS running
                while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                    sleep 10
                done

                echo "--- $circuit Ansatz, Train Pulse=$train_pulse, Decompose=$decompose_circuit, Seed $seed ---"
                uv run kedro run --pipeline "study-4" --params="data.seed=$seed,model.circuit_type=$circuit,model.decompose_circuit=$decompose_circuit,train.train_pulse=$train_pulse" &

                sleep 20
            done
        done
    done
done

# Wait for all remaining jobs to complete
wait
echo "All seeds completed"