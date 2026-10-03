"""Measure circuit expressibility while sampling unitary or pulse parameters."""

import logging
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from fluksio import Port, node
from qml_essentials.expressibility import Expressibility
from qml_essentials.model import Model
from scipy.linalg import sqrtm

from pulse_level_qfms.model import load_model

log = logging.getLogger(__name__)


class PulseExpressibility(Expressibility):
    """Sample unitary parameters, pulse scalers, or both for expressibility.

    ``sample_axis`` selects the sampled groups. Pulse scalers receive Gaussian
    distortion controlled by ``pulse_params_variance``.
    """

    @staticmethod
    def _sample_state_fidelities(
        model: Model,
        n_samples: int,
        random_key: jax.random.PRNGKey,
        sample_axis: List[str],
        pulse_params_variance: float,
        scale: bool = False,
    ) -> jnp.ndarray:
        """Compute state fidelities for pairs of sampled parameter sets.

        Args:
            model (Model): The quantum model.
            n_samples (int): Number of *pairs* of parameter sets.
            random_key (jax.random.PRNGKey): JAX random key for parameter
                initialization and pulse scaler generation.
            sample_axis (List[str]): Any of ``"unitary"`` and ``"pulse"``.
            pulse_params_variance (float): Std-dev of the multiplicative
                Gaussian noise applied to pulse parameters.
            scale (bool): Whether to scale the number of samples.

        Returns:
            jnp.ndarray: One fidelity per sampled pair.
        """
        if scale:
            total_samples = int(jnp.power(2, model.n_qubits) * n_samples)
        else:
            total_samples = n_samples

        if "unitary" in sample_axis:
            random_key = model.initialize_params(
                random_key=random_key, repeat=total_samples * 2
            )
            log.info("Expressibility: sampling unitary parameters")
        else:
            random_key = model.initialize_params(random_key=random_key)
            log.info("Expressibility: re-initializing unitary parameters")

        scaler = None
        gate_mode = "unitary"

        if "pulse" in sample_axis:
            gate_mode = "ansatz_pulse"
            if pulse_params_variance == 0.0:
                log.info("Expressibility: using default pulse parameters")
            else:
                if "unitary" in sample_axis:
                    # Both axes active: create a scaler that pairs 1:1 with
                    # the already-batched unitary params.
                    scaler = 1.0 + pulse_params_variance * jax.random.normal(
                        random_key,
                        shape=(
                            total_samples * 2,
                            *model.pulse_params.shape[1:],
                        ),
                    )
                    model.repeat_batch_axis = [True, True, False, True]
                    log.info("Expressibility: sampling (unitary+pulse) parameters")
                else:
                    # Pulse only: unitary params are *not* batched (B_P=1).
                    scaler = 1.0 + pulse_params_variance * jax.random.normal(
                        random_key,
                        shape=(
                            total_samples * 2,
                            *model.pulse_params.shape[1:],
                        ),
                    )
                    log.info("Expressibility: sampling pulse parameters only")

        log.info(
            f"Expressibility: using {total_samples} sample pairs "
            f"(gate_mode={gate_mode})"
        )

        sv: jnp.ndarray = model(
            params=model.params,
            execution_type="density",
            gate_mode=gate_mode,
            pulse_params=scaler,
        )

        sqrt_sv1: jnp.ndarray = jnp.array([sqrtm(m) for m in sv[:total_samples]])
        inner_fidelity = sqrt_sv1 @ sv[total_samples:] @ sqrt_sv1

        fid: jnp.ndarray = (
            jnp.trace(
                jnp.array([sqrtm(m) for m in inner_fidelity]),
                axis1=1,
                axis2=2,
            )
            ** 2
        )

        return jnp.abs(fid)

    @staticmethod
    def state_fidelities(
        n_samples: int,
        n_bins: int,
        model: Model,
        random_key: jax.random.PRNGKey,
        sample_axis: List[str],
        pulse_params_variance: float,
        scale: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Bin sampled state fidelities into a histogram."""
        if scale:
            n_samples = int(jnp.power(2, model.n_qubits) * n_samples)
            n_bins = model.n_qubits * n_bins

        fidelities = PulseExpressibility._sample_state_fidelities(
            model=model,
            n_samples=n_samples,
            random_key=random_key,
            sample_axis=sample_axis,
            pulse_params_variance=pulse_params_variance,
            scale=False,  # already applied above
        )

        y: jnp.ndarray = jnp.linspace(0, 1, n_bins + 1)
        z, _ = jnp.histogram(fidelities, bins=y)
        z = z / n_samples

        return y, z


@node(
    requires=[
        Port("model", "artifact"),
        Port("model_spec", "json"),
        Port("sample_seed", "int"),
        Port("n_samples", "int"),
        Port("n_bins", "int"),
        Port("scale", "bool"),
        Port("sample_axis", "list", item="str"),
        Port("pulse_params_variance", "float"),
    ],
    provides=[Port("expressibility", "float")],
    # A density-matrix simulation per sample pair, silent throughout.
    timeout=21600,
    cache=False,
)
def evaluate_expressibility(
    *,
    model: Dict,
    model_spec: Dict,
    sample_seed: int,
    n_samples: int,
    n_bins: int,
    scale: bool,
    sample_axis: List[str],
    pulse_params_variance: float,
) -> Dict:
    """KL divergence between the sampled fidelities and the Haar integral."""
    circuit = load_model(model, model_spec)
    log.info(f"Seed for expressibility: {sample_seed}")
    log.info(
        f"Sample axis: {sample_axis}, pulse_params_variance: {pulse_params_variance}"
    )

    _, dist_circuit = PulseExpressibility.state_fidelities(
        n_samples=n_samples,
        n_bins=n_bins,
        scale=scale,
        model=circuit,
        random_key=jax.random.PRNGKey(sample_seed),
        sample_axis=sample_axis,
        pulse_params_variance=pulse_params_variance,
    )

    _, dist_haar = Expressibility.haar_integral(
        n_qubits=circuit.n_qubits,
        n_bins=n_bins,
        cache=True,
        scale=scale,
    )

    kl_dist = Expressibility.kullback_leibler_divergence(dist_circuit, dist_haar)

    return {"expressibility": float(jnp.mean(kl_dist))}
