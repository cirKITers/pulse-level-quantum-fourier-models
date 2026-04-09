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
n_samples = 200
seed = 1000
scale = True
pulse_params_variance = 0.01
mts = 4

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

    # --- Pulse mode ---
    coeffs_pulse, freqs_pulse = Coefficients.get_spectrum(
        model,
        shift=True,
        trim=True,
        gate_mode="pulse",
        pulse_params=scaler,
        mts=mts,
        **kwargs,
    )

    no_coeffs[circuit_type] = np.array(coeffs_pulse).flatten()

    # --- Unitary mode ---
    coeffs_unitary, freqs_unitary = Coefficients.get_spectrum(
        model,
        shift=True,
        trim=True,
        gate_mode="unitary",
        pulse_params=None,
        mts=mts,
        **kwargs,
    )

    # --- Spectrum per circuit ---
    freq_array = np.array(freqs_pulse)

    pulse_magnitudes = np.abs(np.array(coeffs_pulse))
    pulse_mean = pulse_magnitudes.mean(axis=1)
    pulse_std = pulse_magnitudes.std(axis=1)

    unitary_magnitudes = np.abs(np.array(coeffs_unitary))
    unitary_mean = unitary_magnitudes.mean(axis=1)
    unitary_std = unitary_magnitudes.std(axis=1)

    # positive half only
    pos_mask = freq_array >= 0
    freq_array = freq_array[pos_mask]
    pulse_mean = pulse_mean[pos_mask]
    pulse_std = pulse_std[pos_mask]
    unitary_mean = unitary_mean[pos_mask]
    unitary_std = unitary_std[pos_mask]

    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        freq_array - bar_width / 2,
        unitary_mean,
        width=bar_width,
        yerr=unitary_std,
        capsize=4,
        label="Unitary",
        color="#4C72B0",
    )
    ax.bar(
        freq_array + bar_width / 2,
        pulse_mean,
        width=bar_width,
        yerr=pulse_std,
        capsize=4,
        label="Pulse",
        color="#DD8452",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Mean |c| (log scale)")
    ax.set_title(f"Spectrum - {circuit_type.__name__}")
    ax.set_xticks(freq_array)
    ax.set_xticklabels([str(int(f)) if f == int(f) else "" for f in freq_array])
    ax.legend()
    plt.tight_layout()
    output_path = os.path.join(
        os.path.dirname(__file__),
        f"spectrum_{circuit_type.__name__}.png",
    )
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Spectrum saved to {output_path}")
