import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os

from qml_essentials.ansaetze import Ansaetze
from qml_essentials.model import Model
from qml_essentials.coefficients import FCC, Coefficients

jax.config.update("jax_enable_x64", True)


numerical_cap = 1e-10
n_samples = 500
seed = 1000
scale = True
pulse_params_variance = 0.001

ansatzes = Ansaetze.get_available(parameterized_only=True)

kwargs = {
    "n_samples": n_samples,
    "seed": seed,
    "scale": scale,
    # "nan_to_one": True,
    "numerical_cap": numerical_cap,
}

fccs = {}
no_coeffs = {}
for circuit_type in ansatzes:
    model = Model(
        n_qubits=4,
        n_layers=1,
        circuit_type=circuit_type,
        output_qubit=-1,
        encoding=["RY"],
    )

    random_key = model.initialize_params(random_key=model.random_key, repeat=n_samples)
    # sample differently for params
    scaler = 1.0 + pulse_params_variance * jax.random.normal(
        random_key,
        shape=(
            n_samples,
            *model.pulse_params.shape[1:],  # starting from batch dimension
        ),
    )
    degree = jnp.prod(jnp.array(model.degree))
    # but repeat over the input dimension
    # Note, that the following steps are identical to what happens in
    # _assimilate_batch
    # [B_I, 1, B_R, ...]
    scaler = scaler.repeat(degree, axis=0)
    # [..., B]
    scaler = scaler.reshape(
        degree * n_samples,
        *model.pulse_params.shape[1:],
    )
    # disable repeat for pulse parameters (to not further extend batch axis)
    model.repeat_batch_axis = [True, True, False]

    coeffs, freqs = Coefficients.get_spectrum(
        model,
        shift=True,
        trim=True,
        gate_mode="pulse",
        pulse_params=scaler,
        mts=2,
        **kwargs,
    )

    no_coeffs[circuit_type] = np.array(coeffs).flatten()

# --- Histogram ---
fig, ax = plt.subplots(figsize=(10, 6))
for circuit_type, coeff_vals in no_coeffs.items():
    ax.hist(
        np.abs(coeff_vals),
        bins=50,
        alpha=0.5,
        label=circuit_type,
        density=True,
    )

ax.set_xlabel("Coefficient magnitude |c|")
ax.set_ylabel("Density")
ax.set_title("Histogram of Fourier coefficients (pulse-level)")
ax.legend(fontsize="small", ncol=2)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "check_leakage_histogram.png")
plt.savefig(output_path, dpi=150)
plt.close()
print(f"Histogram saved to {output_path}")
