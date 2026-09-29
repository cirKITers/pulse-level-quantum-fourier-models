"""The trained slice starts where the fixed slice is, and training lowers it."""

import numpy as np
from qml_essentials.ansaetze import Encoding
from qml_essentials.coefficients import Datasets
from qml_essentials.model import Model

from pulse_level_qfms.landscape import _pulse_sweep, _trained_sweeps
from pulse_level_qfms.model import apply_pulse_settings, pulse_settings


def test_trained_slice_starts_at_the_fixed_slice():
    apply_pulse_settings(pulse_settings("gaussian", True, "drive"))
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        data_reupload=True,
        encoding=Encoding(strategy="ternary", gates=["RY"]),
        output_qubit=-1,
        initialization="random",
        initialization_domain=[0.0, 6.283],
        random_seed=1000,
    )
    x = Datasets.construct_domain_samples(model, mts=4, mfs=1)
    y = np.sin(2.3 * np.asarray(x)).ravel()
    slot = int(model._enc_pulse_offsets[0])
    frozen = np.array([[1.0, 1.05]])
    gates = [(0, 1, 3.0)]
    # the target scaler sits inside, so the continuation runs both ways
    grid = np.array([0.95, 1.0, 1.05, 1.1])
    sweep = (model, x, y, slot, frozen, gates, {"l0.q1": grid}, "enc_pulse")

    fixed = _pulse_sweep(model, x, y, 0, 1, slot, grid, frozen, 2)
    start, fit = _trained_sweeps(*sweep, 0, 0, 1e-2)
    assert np.allclose(start["l0.q1"], fixed, rtol=1e-10, atol=1e-12)
    assert np.isclose(fit[0], fixed[2])

    trained, fit = _trained_sweeps(*sweep, 10, 3, 1e-2)
    assert fit[-1] < fit[0]
    assert (trained["l0.q1"] < fixed).all()
