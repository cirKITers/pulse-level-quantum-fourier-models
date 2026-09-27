"""A model has to survive the trip to another process.

Each node runs in a worker of its own, so the model a study builds reaches
the node that measures it as a pickle. The pulse envelope, the rotating-wave
approximation and the frame are class-level state of :mod:`jaqsi` that no
pickle carries, which is why the spec travels beside the artifact and is
applied before anything is read back.
"""

import multiprocessing as mp
import pickle

import numpy as np
import pytest
from qml_essentials.ansaetze import Encoding
from qml_essentials.model import Model

from pulse_level_qfms.model import (
    _apply_scaler_mask,
    apply_pulse_settings,
    decomposed_circuit_class,
    pulse_settings,
)

SPEC = pulse_settings("gaussian", True, "drive")
INPUTS = np.linspace(0.0, 1.0, 5)


def _build(decompose: bool) -> Model:
    apply_pulse_settings(SPEC)
    circuit_type = "Circuit_19"
    mask = None
    if decompose:
        circuit_type = decomposed_circuit_class("Circuit_19", 2)
        mask = circuit_type().scaler_mask(2)

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type=circuit_type,
        data_reupload=True,
        encoding=Encoding(strategy="ternary", gates=["RY"]),
        output_qubit=-1,
        initialization="random",
        initialization_domain=[0.0, 6.283],
        random_seed=1000,
    )
    if mask is not None:
        _apply_scaler_mask(model, mask)
    return model


def _predict(model: Model, gate_mode: str) -> np.ndarray:
    return np.asarray(
        model(
            params=model.params,
            inputs=INPUTS,
            execution_type="expval",
            force_mean=True,
            gate_mode=gate_mode,
        )
    )


def _reopen(payload: bytes, gate_mode: str, sink) -> None:
    """Unpickle in a process that has never seen the model, as a worker does."""
    apply_pulse_settings(SPEC)
    model = pickle.loads(payload)
    sink.send(
        (str(model.params.dtype), int(model.params.size), _predict(model, gate_mode))
    )
    sink.close()


@pytest.mark.parametrize("decompose", [False, True])
@pytest.mark.parametrize("gate_mode", ["unitary", "enc_pulse"])
def test_model_survives_a_fresh_process(decompose, gate_mode):
    model = _build(decompose)
    expected = _predict(model, gate_mode)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    receive, send = mp.Pipe(duplex=False)
    process = mp.get_context("spawn").Process(
        target=_reopen, args=(payload, gate_mode, send)
    )
    process.start()
    dtype, size, prediction = receive.recv()
    process.join(timeout=300)

    assert process.exitcode == 0
    # x64 is enabled on import, so a worker that only imports the package
    # reads the same numbers back rather than silently halving them
    assert dtype == "float64"
    assert size == model.params.size
    assert np.allclose(prediction, expected)


def test_the_encoding_gate_is_the_same_gate_after_a_round_trip():
    """`Gates.RY` is what the encoding holds, and it has to keep its name."""
    model = _build(decompose=False)
    reopened = pickle.loads(pickle.dumps(model))
    assert [g.__name__ for g in reopened._enc._gates] == [
        g.__name__ for g in model._enc._gates
    ]
