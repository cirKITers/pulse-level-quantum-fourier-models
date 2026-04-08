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
n_samples = 100
seed = 1000
scale = True
pulse_params_variance = 0.001
mts = 2

ansatzes = Ansaetze.get_available(parameterized_only=True)

kwargs = {
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
    scaler = scaler.repeat(degree * mts, axis=0)
    # [..., B]
    scaler = scaler.reshape(
        mts * degree * n_samples,
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
        mts=mts,
        **kwargs,
    )

    no_coeffs[circuit_type] = np.array(coeffs).flatten()

    # --- Spectrum per circuit ---
    coeff_magnitudes = np.abs(np.array(coeffs))  # shape: (n_freqs, n_samples)
    mean_magnitudes = coeff_magnitudes.mean(axis=1)
    std_magnitudes = coeff_magnitudes.std(axis=1)
    freq_array = np.array(freqs)

    # positive half only
    pos_mask = freq_array >= 0
    freq_array = freq_array[pos_mask]
    mean_magnitudes = mean_magnitudes[pos_mask]
    std_magnitudes = std_magnitudes[pos_mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(freq_array, mean_magnitudes, width=0.4, yerr=std_magnitudes, capsize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Mean |c| (log scale)")
    ax.set_title(f"Spectrum - {circuit_type.__name__}")
    ax.set_xticks(freq_array)
    plt.tight_layout()
    output_path = os.path.join(
        os.path.dirname(__file__),
        f"spectrum_{circuit_type.__name__}.png",
    )
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Spectrum saved to {output_path}")
