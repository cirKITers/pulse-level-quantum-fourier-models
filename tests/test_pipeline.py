"""The flow declarations, checked without an engine.

`fluksio sync` validates these too, but it needs a running engine to reach,
and a wiring mistake is worth catching in the test suite rather than at the
start of a sweep.
"""

import pytest
from fluksio.flow.messages import MessageSpec
from fluksio.sdk import FLOWS, build

import pulse_level_qfms.pipeline  # noqa: F401  -- importing it registers the flows

NAMES = {
    "fcc",
    "fidelity",
    "expressibility",
    "train",
    "train_enc",
    "spectrum",
    "landscape",
}


def test_every_flow_is_declared():
    assert set(FLOWS) == NAMES


@pytest.mark.parametrize("name", sorted(NAMES))
def test_flow_document_builds(name):
    """What `fluksio sync` would upload, without uploading it."""
    document = build(FLOWS[name])
    assert document["name"] == name
    assert document["nodes"], f"{name} has no nodes"


@pytest.mark.parametrize("name", sorted(NAMES))
def test_every_flow_reports_its_model(name):
    """The parameter counts the plots read come off `model_spec`."""
    assert "model_spec" in FLOWS[name].outputs


@pytest.mark.parametrize("name", sorted(NAMES))
def test_inputs_are_typed_and_defaulted(name):
    """A driver overrides some inputs; the rest have to have a value."""
    for port in FLOWS[name].inputs:
        assert port.explicit, f"{name}.{port.name} leaves its type to be inferred"
        assert port.initial is not None, f"{name}.{port.name} has no default"


def _port(flow: str, name: str) -> MessageSpec:
    for node in FLOWS[flow].nodes:
        for port in node.provides:
            if port.name == name:
                return MessageSpec(**port.spec())
    raise AssertionError(f"{flow} has no port {name}")


def test_result_payloads_are_port_legal():
    """What the nodes return has to survive the ports it returns on.

    The engine's own spec class rather than a copy of its rules: a `float`
    port refuses NaN and None, and a record carrying either is refused whole,
    which is what `jsonable` is for.
    """
    _port("fcc", "coefficients").check(
        {"mean": {"-1.0": 0.5, "0.0": 1.0}, "var": {"-1.0": 0.01, "0.0": 0.0}}
    )
    _port("landscape", "landscape").check(
        {
            "mts": 4,
            "n_gates": 2,
            "n_frequencies": 9,
            "check_pulse": 1e-9,
            "gates": [{"key": "l0.q0", "layer": 0, "qubit": 0, "generator": 1.0}],
        }
    )
    _port("landscape", "profile").check(
        {"lines": [{"label": "l0.q0", "points": [[0.0, 1.0], [0.5, 0.25]]}]}
    )
    _port("train_enc", "training").check(
        {"gate_mode": "enc_pulse", "steps": 10, "train_mse": 0.5, "rank": {}}
    )


def test_a_float_port_refuses_a_gap():
    """Why `fit` leaves a non-finite value out rather than publishing it."""
    stream = _port("train", "train_mse")
    stream.check(0.5)
    for rejected in (float("nan"), None):
        with pytest.raises(TypeError):
            stream.check(rejected)
