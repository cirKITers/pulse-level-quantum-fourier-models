import pennylane as qml

from qml_essentials.model import Model
from qml_essentials.ansaetze import Gates

from typing import List, Dict, Union, Callable
import numpy as np
import mlflow

import torch
from torch.utils.data import TensorDataset, DataLoader

import logging

log = logging.getLogger(__name__)

from pulse_level_qfms.utils import Sampling


def generate_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    data_reupload: bool,
    encoding_gates: Union[str, Callable, List[str], List[Callable]],
    initialization: str,
    initialization_domain: List[float],
    output_qubit: int,
    mp_threshold: int,
    seed: int,
) -> Dict[str, Model]:
    log.info(
        f"Creating model with {n_qubits} qubits, {n_layers} layers, and {circuit_type} circuit."
    )

    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=circuit_type,
        data_reupload=data_reupload,
        encoding=encoding_gates,
        output_qubit=output_qubit,
        initialization=initialization,
        initialization_domain=initialization_domain,
        mp_threshold=mp_threshold,
        random_seed=seed,
    )

    log.debug(f"Created quantum model with {model.params.size} trainable parameters.")
    mlflow.log_text(str(model), "model.txt")

    return {"model": model}


def generate_fourier_series(
    model: Model,
    domain: List[float],
    omegas: List[List[float]],
    coefficients_min: float,
    coefficients_max: float,
    zero_centered: bool,
    seed: int,
) -> np.ndarray:
    """
    Generates the Fourier series representation of a function.

    Parameters
    ----------
    domain_samples : np.ndarray
        Grid of domain samples.
    omega : List[List[float]]
        List of frequencies for each dimension.

    Returns
    -------
    np.ndarray
        Fourier series representation of the function.
    """
    mfs = 1
    mts = 1

    n_freqs: int = 2 * omegas + 1

    start, stop, step = domain[0], domain[1], 2 * np.pi / n_freqs
    # Stretch according to the number of frequencies
    inputs: np.ndarray = np.arange(start, stop, step)

    # permute with input dimensionality
    domain_samples = np.array(np.meshgrid(*[inputs] * model.n_input_feat)).T.reshape(
        -1, model.n_input_feat
    )

    rng = np.random.default_rng(seed)

    frequencies = np.stack(
        np.meshgrid(
            *[
                np.linspace(-omegas, omegas, 2 * omegas + 1)
                for _ in range(model.n_input_feat)
            ]
        )
    ).T.reshape(-1, model.n_input_feat)

    n_freqs: int = int(2 * mfs * omegas + 1)

    coefficients = Sampling.uniform_circle(
        rng,
        coefficients_min,
        coefficients_max,
        int(np.ceil(frequencies.shape[0] / 2)),
    )

    coefficients = coefficients.flatten()

    if not zero_centered:
        coefficients[0] = 0.0
    else:
        # ensure the first coefficient is real
        coefficients[0] = coefficients[0].real

    # ensure symmetry
    coefficients = np.concat(
        [np.flip(coefficients[1:]).conjugate(), coefficients],
    )

    def y(x: np.ndarray) -> float:
        return (
            np.real_if_close(np.sum(coefficients * np.exp(1j * frequencies.dot(x))))
            / coefficients.size
        )

    values = np.stack([y(x) for x in domain_samples])

    coefficients_hat = np.fft.fftshift(
        np.fft.fftn(
            values.reshape([n_freqs] * model.n_input_feat),
            axes=list(range(model.n_input_feat)),
        )
    )
    assert np.allclose(
        coefficients, coefficients_hat.flatten(), atol=1e-6
    ), "Frequencies don't match"

    return {
        "domain_samples": domain_samples,
        "fourier_samples": values.flatten(),
        "coefficients": coefficients,
    }


def build_fourier_series_dataloader(
    batch_size: int, domain_samples, fourier_samples, coefficients: np.ndarray
):
    if batch_size < 1:
        batch_size = domain_samples.shape[0]
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(domain_samples),
            torch.from_numpy(fourier_samples).squeeze(),
            torch.from_numpy(coefficients).squeeze(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": train_loader,
    }
