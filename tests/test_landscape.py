"""The trained slice starts where the fixed slice is, and training lowers it."""

import numpy as np
from qml_essentials.ansaetze import Encoding
from qml_essentials.coefficients import Datasets
from qml_essentials.model import Model

from pulse_level_qfms.landscape import _pulse_sweep, _trained_sweep
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
    frozen = np.ones((1, 2))
    # three scalers in chunks of two, so the padding is exercised
    grid = np.array([0.9, 1.0, 1.1])
    sweep = (model, x, y, 0, 1, slot, grid, frozen, 2)

    fixed = _pulse_sweep(*sweep)
    start, settled = _trained_sweep(*sweep, "enc_pulse", 0, 1e-2)
    assert np.allclose(start, fixed, rtol=1e-10, atol=1e-12)
    assert len(settled) == 1

    trained, settled = _trained_sweep(*sweep, "enc_pulse", 5, 1e-2)
    assert settled[-1] < settled[0]
    assert np.isclose(settled[0], fixed.mean())
    assert trained.mean() < fixed.mean()
