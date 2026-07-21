"""Unit tests for the Jacobian-rank diagnostics introduced for
the manuscript TODO at line 757 of main.tex.

The diagnostics compute

* ``rank J_θ`` — Jacobian of the (real-stacked) Fourier coefficients
  w.r.t. the unitary parameters θ alone.
* ``rank J_ext`` — Jacobian w.r.t. the extended parameters, i.e. θ plus
  the pulse-scaling groups the gate mode runs at pulse level: λ for the
  ansatz, η for the encoding, or both.
* ``Δr = rank J_ext − rank J_θ`` — the count of *additional* search
  directions in coefficient space unlocked by those scalers.

These tests check the helper itself (shape, finite values,
consistency with the manuscript's bounds) on a tiny model so they
stay well below 1 minute end-to-end.
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from qml_essentials.ansaetze import Encoding
from qml_essentials.model import Model

from pulse_level_qfms.pipelines.processing import nodes
from pulse_level_qfms.pipelines.processing.nodes import _jacobian_rank


def _tiny_model():
    return Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        data_reupload=True,
        encoding=Encoding(strategy="ternary", gates=["RY"]),
        output_qubit=-1,
        initialization="random",
        initialization_domain=[0.0, 6.283],
    )


@pytest.fixture(scope="module")
def model_and_params():
    m = _tiny_model()
    theta = jnp.asarray(m.params)
    lam = jnp.ones_like(jnp.asarray(m.pulse_params))
    eta = jnp.ones_like(jnp.asarray(m.enc_pulse_params))
    return m, theta, lam, eta


def test_jacobian_rank_J_theta_shape_and_bounds(model_and_params):
    m, theta, lam, eta = model_and_params
    rank, sv_min, shape = _jacobian_rank(
        m, theta, lam, eta, gate_mode="ansatz_pulse", argnums=(0,), tol_rel=1e-8
    )
    # The Jacobian rows are [Re(c_ω), Im(c_ω)] stacked, columns are flattened θ.
    assert len(shape) == 2
    rows, cols = shape
    assert rows == 2 * 9, f"expected 2|Ω|=18 rows, got {rows}"
    assert cols == int(jnp.size(theta))
    assert 0 <= rank <= min(rows, cols)
    assert sv_min >= 0.0


def test_jacobian_rank_J_ext_dominates_J_theta(model_and_params):
    """rank J_ext >= rank J_θ always — adding extra columns can only
    keep the rank the same or increase it."""
    m, theta, lam, eta = model_and_params
    r_theta, _, _ = _jacobian_rank(
        m, theta, lam, eta, gate_mode="ansatz_pulse", argnums=(0,), tol_rel=1e-8
    )
    r_ext, _, shape_ext = _jacobian_rank(
        m, theta, lam, eta, gate_mode="ansatz_pulse", argnums=(0, 1), tol_rel=1e-8
    )
    rows, cols_ext = shape_ext
    assert cols_ext == int(jnp.size(theta)) + int(jnp.size(lam))
    delta_r = r_ext - r_theta
    assert delta_r >= 0, f"Δr must be non-negative, got {delta_r}"
    # rank is bounded above by 2|Ω|
    assert r_ext <= rows


def test_jacrev_does_not_leave_dead_tracer(model_and_params):
    """Regression test for the qml-essentials tracer-leak fix:
    two consecutive jacrev calls on the same model must succeed."""
    m, theta, lam, eta = model_and_params
    # First call.
    _jacobian_rank(
        m, theta, lam, eta, gate_mode="ansatz_pulse", argnums=(0,), tol_rel=1e-8
    )
    # Second call must not raise UnexpectedTracerError.
    _jacobian_rank(
        m, theta, lam, eta, gate_mode="ansatz_pulse", argnums=(0, 1), tol_rel=1e-8
    )


def test_jacobian_rank_J_ext_for_enc_pulse(model_and_params):
    """In "enc_pulse" the ansatz stays unitary, so the extra search
    directions come from the encoding scalers eta rather than lambda."""
    m, theta, lam, eta = model_and_params
    r_theta, _, _ = _jacobian_rank(
        m, theta, lam, eta, gate_mode="enc_pulse", argnums=(0,), tol_rel=1e-8
    )
    r_ext, _, shape_ext = _jacobian_rank(
        m, theta, lam, eta, gate_mode="enc_pulse", argnums=(0, 2), tol_rel=1e-8
    )
    rows, cols_ext = shape_ext
    assert cols_ext == int(jnp.size(theta)) + int(jnp.size(eta))
    delta_r = r_ext - r_theta
    assert delta_r >= 0, f"Δr must be non-negative, got {delta_r}"
    assert r_ext <= rows


def test_jacobian_rank_J_ext_for_all_pulse(model_and_params):
    """"all_pulse" extends J_θ by both scaler groups at once."""
    m, theta, lam, eta = model_and_params
    r_theta, _, _ = _jacobian_rank(
        m, theta, lam, eta, gate_mode="all_pulse", argnums=(0,), tol_rel=1e-8
    )
    r_ext, _, shape_ext = _jacobian_rank(
        m, theta, lam, eta, gate_mode="all_pulse", argnums=(0, 1, 2), tol_rel=1e-8
    )
    rows, cols_ext = shape_ext
    assert cols_ext == (
        int(jnp.size(theta)) + int(jnp.size(lam)) + int(jnp.size(eta))
    )
    assert r_ext - r_theta >= 0
    assert r_ext <= rows


def test_log_jacobian_ranks_argnums_follow_gate_mode(monkeypatch):
    """_log_jacobian_ranks must extend J_θ by exactly the scaler groups
    the gate mode runs at pulse level, and skip J_ext for "unitary"."""
    seen = []

    def _fake_rank(model, theta, lam, eta, gate_mode, argnums, tol_rel):
        seen.append(argnums)
        return 1, 1.0, (2, 2)

    monkeypatch.setattr(nodes, "_jacobian_rank", _fake_rank)
    monkeypatch.setattr(nodes.mlflow, "log_metric", lambda *a, **k: None)

    m = _tiny_model()
    expected = {
        "unitary": [(0,)],
        "ansatz_pulse": [(0,), (0, 1)],
        "enc_pulse": [(0,), (0, 2)],
        "all_pulse": [(0,), (0, 1, 2)],
    }
    for mode, want in expected.items():
        seen.clear()
        nodes._log_jacobian_ranks(
            m,
            theta=jnp.asarray(m.params),
            lam=jnp.ones_like(jnp.asarray(m.pulse_params)),
            eta=jnp.ones_like(jnp.asarray(m.enc_pulse_params)),
            gate_mode=mode,
            tol_rel=1e-8,
            step=0,
        )
        assert seen == want, f"{mode}: expected argnums {want}, got {seen}"
