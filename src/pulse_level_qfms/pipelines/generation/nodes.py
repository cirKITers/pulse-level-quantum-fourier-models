import pennylane as qml

from qml_essentials.model import Model
from qml_essentials.ansaetze import Gates
from qml_essentials.coefficients import Datasets

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
    use_multithreading: bool,
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
        use_multithreading=use_multithreading,
        random_seed=seed,
    )

    log.debug(f"Created quantum model with {model.params.size} trainable parameters.")
    mlflow.log_text(str(model), "model.txt")

    return {"model": model}


def generate_fourier_series(
    model: Model,
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
    rng = np.random.default_rng(seed)

    domain_samples, fourier_samples, coefficients = Datasets.generate_fourier_series(
        rng=rng,
        model=model,
        coefficients_min=coefficients_min,
        coefficients_max=coefficients_max,
        zero_centered=zero_centered,
    )

    return {
        "domain_samples": domain_samples,
        "fourier_samples": fourier_samples.flatten(),
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
