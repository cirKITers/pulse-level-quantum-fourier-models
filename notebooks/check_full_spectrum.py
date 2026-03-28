import jax
import jax.numpy as jnp

from qml_essentials.ansaetze import Ansaetze
from qml_essentials.model import Model
from qml_essentials.coefficients import FCC, Coefficients

jax.config.update("jax_enable_x64", True)


numerical_cap = 1e-10
n_samples = 500
seed = 1000
scale = True

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

    fp, freqs = FCC.get_fourier_fingerprint(model=model, **kwargs)

    no_coeffs[circuit_type] = fp.shape[0]


with open("no_coeffs.csv", "w") as f:
    for circuit_type, n_coeffs in sorted(no_coeffs.items(), key=lambda item: item[1]):
        f.write(f"{circuit_type},{n_coeffs}\n")
