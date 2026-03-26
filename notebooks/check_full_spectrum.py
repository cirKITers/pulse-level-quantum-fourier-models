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

    if fp.shape[0] < (model.frequencies[0].size // 2) - 1:  # unique correlations -1
        print(f"Skipping {circuit_type} as it does not have a full spectrum")
        no_coeffs[circuit_type] = fp.shape[0]
        continue

    # fcc = FCC.get_fcc(model=model, **kwargs)
    # print(f"FCC for {circuit_type}: {fcc}")
    # fccs[circuit_type] = fcc

# # print fccs for circuits sorted by values
# with open("fccs.csv", "w") as f:
#     for circuit_type, fcc in sorted(fccs.items(), key=lambda item: item[1]):
#         print(f"{circuit_type}: {fcc}")
#         f.write(f"{circuit_type},{fcc}\n")

with open("no_coeffs.csv", "w") as f:
    for circuit_type, n_coeffs in sorted(no_coeffs.items(), key=lambda item: item[1]):
        f.write(f"{circuit_type},{n_coeffs}\n")
