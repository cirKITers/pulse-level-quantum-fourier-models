from qml_essentials.model import Model
import jax
import jax.numpy as jnp

ppv = 0.01
n_params_samples = 3
n_input_samples = 2

model = Model(n_qubits=3, n_layers=1, circuit_type="Circuit_15", encoding="RY")

random_key = model.initialize_params(repeat=n_params_samples)

# randomize across all standard params
scaler = 1.0 + ppv * jax.random.normal(
    random_key,
    shape=(
        *model.pulse_params.shape,
        n_params_samples,
    ),
)

# but keep the distortion the constant for all inputs
# [..., 1, B_P] -> [..., B_I, B_R]
scaler = scaler.repeat(n_input_samples, axis=-2)

# do some merging to end up with
# [..., B]
scaler = scaler.reshape(
    *model.pulse_params.shape[:-1],
    n_input_samples * n_params_samples,
)

# repeat along the input and params axis, but ignore the pulse params
model.repeat_batch_axis = [True, True, False]

# some dumy inputs
inputs = jnp.zeros((n_input_samples,))

res = model(inputs=inputs, pulse_params=scaler, gate_mode="pulse")

# we want to end up with [B_I, B_P, n_qubits]
# because all inputs are equal, the output should be the same for all inputs
assert jnp.allclose(res[0], res[1], rtol=1e-5), "Output is not the same for all inputs"
