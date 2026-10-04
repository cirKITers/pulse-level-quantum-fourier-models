"""How far a distorted pulse-level circuit drifts from its unitary ideal."""

import logging
from typing import Dict

import jax
import jax.numpy as jnp
from fluksio import Port, node
from jaqsi.math import fidelity, trace_distance

from pulse_level_qfms.model import load_model

log = logging.getLogger(__name__)


@node(
    requires=[
        Port("model", "artifact"),
        Port("model_spec", "json"),
        Port("sample_seed", "int"),
        Port("n_samples", "int"),
        Port("scale", "bool"),
        Port("pulse_params_variance", "float"),
    ],
    provides=[Port("fidelity", "float"), Port("trace_distance", "float")],
    # Two density-matrix simulations over every sample, silent throughout.
    timeout=21600,
    cache=False,
)
def evaluate_fidelity(
    *,
    model: Dict,
    model_spec: Dict,
    sample_seed: int,
    n_samples: int,
    scale: bool,
    pulse_params_variance: float,
) -> Dict:
    """Overlap of the unitary and the distorted pulse-level state."""
    circuit = load_model(model, model_spec)
    log.info(f"Seed for fidelity check: {sample_seed}")

    if scale:
        total_samples = int(jnp.power(2, circuit.n_qubits) * n_samples)
    else:
        total_samples = n_samples

    log.info(f"Using {total_samples} samples for fidelity check")

    random_key = jax.random.PRNGKey(sample_seed)
    random_key = circuit.initialize_params(random_key=random_key, repeat=total_samples)

    # calculate density matrices for unitary and pulse circuits
    unitary_states = circuit(execution_type="density")

    scaler = 1.0 + pulse_params_variance * jax.random.normal(
        random_key,
        shape=(total_samples, *circuit.pulse_params.shape[1:]),
    )
    # the pulse scalers pair 1:1 with the already-batched unitary params, so
    # their axis does not own a batch of its own. Four entries because the
    # mask covers [inputs, params, pulse_params, enc_pulse_params].
    circuit.repeat_batch_axis = [True, True, False, True]

    pulse_states = circuit(
        pulse_params=scaler,
        gate_mode="ansatz_pulse",
        execution_type="density",
    )

    return {
        "fidelity": float(jnp.mean(fidelity(unitary_states, pulse_states))),
        "trace_distance": float(
            jnp.mean(trace_distance(unitary_states, pulse_states))
        ),
    }
