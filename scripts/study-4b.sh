#!/bin/bash

# Diagnostic sweep for the enc_pulse training question: does the off-grid
# advantage fail to show because of reachability or because of optimization?
# Each arm logs the per-gate eta trajectory (eta.f*.l*.q*), the distance to the
# reachable target (eta.target_dist) and a post-training spectrum
# (spectrum.trained.f*, mfs=2 so the scaled frequencies do not alias).
#
# Runs on one small circuit; widen CIRCUITS / SEEDS once an arm looks promising.

set -e

MAX_JOBS=1
CIRCUITS="Circuit_15"
SEEDS="1000 1001 1002"
STRATEGY="ternary" # pair with hamming via the B6 arm below
STEPS=2000

# common params; rank eval off (it is a separate, slower diagnostic)
COMMON="model.encoding_strategy=$STRATEGY,train.steps=$STEPS,train.rank_eval.enabled=False,train.trained_spectrum.enabled=True"

run() {
    # run <label> <extra-params>
    local label="$1"; shift
    local extra="$1"; shift
    for circuit in $CIRCUITS; do
        for seed in $SEEDS; do
            while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done
            echo "--- $label | $circuit seed $seed ---"
            uv run kedro run --pipeline study-4 \
                --params="data.seed=$seed,model.circuit_type=$circuit,$COMMON,$extra" &
            sleep 5
        done
    done
}

# B0: on-grid floor. Amplitude-fitting limit with no frequency mismatch; the
# effect size is off-grid MSE minus this floor, not raw MSE.
run "B0-floor-unitary"  "train.gate_mode=unitary,data.offgrid_prob=0.0"
run "B0-floor-encpulse" "train.gate_mode=enc_pulse,data.offgrid_prob=0.0"

# Baselines on the off-grid target
run "unitary"            "train.gate_mode=unitary"
run "encpulse-ones"      "train.gate_mode=enc_pulse"

# B2: oracle-init the amplitude scalers at the reachable target. Reaching the
# B0 floor here while encpulse-ones stalls means the problem is optimization,
# not reachability.
run "B2-encpulse-target" "train.gate_mode=enc_pulse,train.enc_pulse_init=target"

# B3: same frequency knob without the pulse ODE. If this also stalls, the issue
# is the generic trainable-frequency landscape, not anything pulse-specific.
run "B3-enc-params"      "train.gate_mode=unitary,train.train_enc_params=True"

# B4: faster pulse learning rate (target displacement is 0.5 in eta).
run "B4-lr1e-2"          "train.gate_mode=enc_pulse,train.pulse_learning_rate=1e-2"

# B5: train only the amplitude scaler, freezing the redundant sigma/duration.
run "B5-amplitude-only"  "train.gate_mode=enc_pulse,train.pulse_amplitude_only=True"

# B6: hamming reference for the default arm (distinct generator scales in
# ternary improve per-gate identifiability).
STRATEGY_SAVE="$STRATEGY"
COMMON="model.encoding_strategy=hamming,train.steps=$STEPS,train.rank_eval.enabled=False,train.trained_spectrum.enabled=True"
run "B6-hamming-encpulse" "train.gate_mode=enc_pulse"

wait
echo "All arms completed"
