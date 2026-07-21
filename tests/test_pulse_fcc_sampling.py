"""Unit tests for the sampling axes of :class:`PulseFCC`.

``PulseFCC._calculate_coefficients`` randomises a subset of the variational
parameters $\\theta$, the ansatz pulse scalers $\\lambda$ and the encoding
pulse scalers $\\eta$, so that the FCC can separate the effect of sampling
pulse parameters for ansatz gates from that for encoding gates.

The tests check that every ``gate_mode`` produces a spectrum with the
requested number of samples, that entries the mode cannot honour are
ignored, and that the selected quantities are drawn jointly rather than as
an outer product. They run on a tiny model to stay fast.
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from qml_essentials.ansaetze import Encoding
from qml_essentials.coefficients import Coefficients
from qml_essentials.model import Model

from pulse_level_qfms.pipelines.processing.nodes import PulseFCC, _MODE_BY_GROUPS

N_SAMPLES = 3
SEED = 1000
VARIANCE = 0.05


def _tiny_model():
    return Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_3",
        data_reupload=True,
        encoding=Encoding(strategy="hamming", gates=["RY"]),
        random_seed=1,
    )


@pytest.mark.parametrize(
    "sample_axis, expected_mode",
    [
        (["unitary"], "unitary"),
        (["unitary", "pulse"], "ansatz_pulse"),
        (["pulse"], "ansatz_pulse"),
        (["unitary", "enc_pulse"], "enc_pulse"),
        (["enc_pulse"], "enc_pulse"),
        (["unitary", "pulse", "enc_pulse"], "all_pulse"),
        (["pulse", "enc_pulse"], "all_pulse"),
    ],
)
def test_all_four_regimes_are_reachable_from_sample_axis(sample_axis, expected_mode):
    """sample_axis alone selects each of the four pulse regimes.

    This is what lets gate_mode stay derived: the ansatz and encoding regimes
    are addressed separately, so the four scenarios remain distinguishable
    without a second parameter.
    """
    sampled = frozenset(g for g in ("pulse", "enc_pulse") if g in sample_axis)
    assert _MODE_BY_GROUPS[sampled] == expected_mode


def test_every_regime_has_a_sample_axis():
    """No mode is unreachable, in particular enc_pulse."""
    assert set(_MODE_BY_GROUPS.values()) == {
        "unitary",
        "ansatz_pulse",
        "enc_pulse",
        "all_pulse",
    }


def _coefficients(model, gate_mode, sample_axis, variance=VARIANCE, **kwargs):
    return PulseFCC._calculate_coefficients(
        model,
        N_SAMPLES,
        SEED,
        False,
        sample_axis=sample_axis,
        gate_mode=gate_mode,
        pulse_params_variance=variance,
        numerical_cap=-1,
        **kwargs,
    )


@pytest.mark.parametrize(
    "gate_mode, sample_axis",
    [
        ("unitary", ["unitary"]),
        ("ansatz_pulse", ["unitary", "pulse"]),
        ("ansatz_pulse", ["pulse"]),
        ("enc_pulse", ["unitary", "enc_pulse"]),
        ("enc_pulse", ["enc_pulse"]),
        ("all_pulse", ["unitary", "pulse", "enc_pulse"]),
        ("all_pulse", ["pulse", "enc_pulse"]),
        ("all_pulse", ["enc_pulse"]),
    ],
)
def test_sample_axis_spans_requested_samples(gate_mode, sample_axis):
    """Each mode yields one spectrum per sample and the samples differ."""
    _, coeffs, _ = _coefficients(_tiny_model(), gate_mode, sample_axis)

    assert coeffs.shape[-1] == N_SAMPLES
    assert jnp.all(jnp.isfinite(jnp.abs(coeffs)))
    # the sampled quantity has to move the coefficients, otherwise the axis
    # never reached the circuit
    assert float(jnp.abs(coeffs).std(axis=-1).max()) > 1e-9


def test_enc_pulse_entry_ignored_without_enc_pulse_mode():
    """A mode that runs encoding gates as unitaries ignores ``enc_pulse``."""
    _, ignored, _ = _coefficients(_tiny_model(), "ansatz_pulse", ["unitary", "enc_pulse"])
    _, unitary_only, _ = _coefficients(_tiny_model(), "unitary", ["unitary"])

    assert jnp.allclose(jnp.abs(ignored), jnp.abs(unitary_only))


@pytest.mark.parametrize(
    "gate_mode, sample_axis",
    [
        ("ansatz_pulse", ["unitary", "pulse"]),
        ("ansatz_pulse", ["pulse"]),
        ("enc_pulse", ["unitary", "enc_pulse"]),
        ("all_pulse", ["unitary", "pulse", "enc_pulse"]),
        ("all_pulse", ["pulse", "enc_pulse"]),
    ],
)
def test_samples_are_drawn_jointly(gate_mode, sample_axis):
    """Sample $j$ must be one joint draw of every selected quantity.

    Each column of the batched spectrum is compared against a separate run
    that uses only that sample's own draws. This is what breaks when the
    pulse scalers are laid out along the flattened batch in the wrong order,
    because the Fourier transform then runs over inputs carrying different
    pulse draws and no single coherent run reproduces the column.
    """
    model = _tiny_model()
    _, coeffs, _ = _coefficients(model, gate_mode, sample_axis)

    # batch element k of a scaler tiled onto the flattened batch holds
    # sample k % N_SAMPLES, so sample j sits at index j
    theta, lam, eta = model.params, model.pulse_params, model.enc_pulse_params

    for j in range(N_SAMPLES):
        reference, _ = Coefficients.get_spectrum(
            _tiny_model(),
            shift=True,
            trim=True,
            numerical_cap=-1,
            gate_mode=gate_mode,
            params=theta[j] if "unitary" in sample_axis else theta[0],
            pulse_params=lam[j][None] if "pulse" in sample_axis else None,
            enc_pulse_params=eta[j][None] if "enc_pulse" in sample_axis else None,
        )
        assert jnp.allclose(reference.squeeze(), coeffs[:, j], atol=1e-8)


def test_zero_variance_keeps_default_pulse_params():
    """Zero variance leaves the pulse scalers untouched at one."""
    model = _tiny_model()
    _, coeffs, _ = _coefficients(
        model, "all_pulse", ["unitary", "pulse", "enc_pulse"], variance=0.0
    )

    assert jnp.allclose(model.pulse_params, 1.0)
    assert jnp.allclose(model.enc_pulse_params, 1.0)
    # the unitary axis still varies
    assert coeffs.shape[-1] == N_SAMPLES
    assert float(jnp.abs(coeffs).std(axis=-1).max()) > 1e-9


@pytest.mark.parametrize(
    "sample_axis, expected_mode",
    [(["pulse"], "ansatz_pulse"), (["enc_pulse"], "enc_pulse")],
)
def test_derived_gate_mode_matches_explicit(sample_axis, expected_mode):
    """Leaving gate_mode unset reproduces passing the derived mode by hand."""
    _, derived, _ = _coefficients(_tiny_model(), None, sample_axis)
    _, explicit, _ = _coefficients(_tiny_model(), expected_mode, sample_axis)

    assert jnp.allclose(derived, explicit)


def test_unknown_gate_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown gate_mode"):
        _coefficients(_tiny_model(), "bogus", ["pulse"])


def _offgrid_mass(coeffs, freqs):
    """Share of the coefficient magnitude on non-integer frequencies."""
    magnitudes = jnp.abs(coeffs).mean(axis=1)
    off_grid = freqs != jnp.round(freqs)

    return float(magnitudes[off_grid].sum() / magnitudes.sum())


def test_enc_pulse_moves_mass_off_the_integer_grid():
    """Distorting the encoding pulses populates the non-integer frequencies.

    Sampled with $mts = 4$, the spectrum carries bins at a spacing of
    $1/4$. An undistorted model is supported on the integer frequencies
    alone, so mass appearing in between means the encoding gates no longer
    encode at integer frequencies. This is the effect that study-5 measures.
    """
    _, undistorted, freqs = _coefficients(
        _tiny_model(), None, ["unitary", "enc_pulse"], variance=0.0, mts=4
    )
    _, distorted, _ = _coefficients(
        _tiny_model(), None, ["unitary", "enc_pulse"], variance=VARIANCE, mts=4
    )

    # the oversampled grid has to actually contain non-integer bins,
    # otherwise the assertions below are vacuous
    assert bool((freqs != jnp.round(freqs)).any())

    assert _offgrid_mass(undistorted, freqs) < 1e-6
    assert _offgrid_mass(distorted, freqs) > 1e-3
