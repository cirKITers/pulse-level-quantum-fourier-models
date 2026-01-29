#!/bin/bash

set -e

# run experiments with all different circuits
MAX_JOBS=3

for variance in 0.015 0.016 0.017 0.018 0.019
do
    for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
    do
        # Wait if we already have MAX_JOBS running
        while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
            sleep 5
        done

        echo "--- Variance $variance, Seed $seed ---"
        uv run kedro run --params="fcc.seed=$seed,fcc.pulse_params_variance=$variance" &

        sleep 30
    done
done

# Wait for all remaining jobs to complete
wait
echo "All seeds completed"
