from qml_essentials.qoc import QOC
from qml_essentials.ansaetze import Gates, Ansaetze
from qml_essentials.model import Model
import pennylane as qml
import jax
import jax.numpy as jnp
from rich.progress import track

n_samples = 2
n_qubits = 2
n_layers = 1
circuit = "Circuit_18"


model_a = Model(n_qubits=n_qubits, n_layers=n_layers, circuit_type=circuit)
model_b = Model(n_qubits=n_qubits, n_layers=n_layers, circuit_type=circuit)

seed = 1000
key = jax.random.PRNGKey(seed)


def sample_params(key, shape):
    params: jnp.ndarray = jax.random.uniform(
        key=key,
        shape=model_a._params_shape,
        minval=0,
        maxval=2 * jnp.pi,
    )
    return params


abs_diff = 0
phase_diff = 0
for _ in track(range(n_samples)):
    key, subkey = jax.random.split(key)
    params = sample_params(subkey, model_a._params_shape)

    state_a = model_a(params=params, gate_mode="pulse", execution_type="state")
    state_b = model_b(params=params, gate_mode="unitary", execution_type="state")

    dot_prod = jnp.vdot(state_a, state_b)
    fidelity = 1 - jnp.abs(dot_prod) ** 2  # one if no diff
    abs_diff += fidelity
    phase_diff += jnp.abs(jnp.angle(dot_prod)) / jnp.pi  # zero if no diff

    print(f"Fidelity: {(1-fidelity)*100:.8f}%")

abs_diff /= n_samples
phase_diff /= n_samples
print(f"Fidelity: {(1-abs_diff)*100:.8f}%")
print(f"Phase Mismatch: {(phase_diff)*100:.8f}%")
