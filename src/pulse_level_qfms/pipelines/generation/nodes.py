from qml_essentials.model import Model
from qml_essentials.coefficients import Datasets
from qml_essentials.ansaetze import Encoding

from typing import List, Dict, Union, Callable
import jax
import jax.numpy as jnp
import numpy as np
import mlflow

import torch
from torch.utils.data import TensorDataset, DataLoader

import logging

log = logging.getLogger(__name__)


jax.config.update("jax_enable_x64", True)


def generate_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    data_reupload: bool,
    encoding_gates: Union[str, Callable, List[str], List[Callable]],
    encoding_strategy: str,
    initialization: str,
    initialization_domain: List[float],
    output_qubit: int,
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
        encoding=Encoding(strategy=encoding_strategy, gates=encoding_gates),
        output_qubit=output_qubit,
        initialization=initialization,
        initialization_domain=initialization_domain,
        random_seed=seed,
    )

    log.debug(f"Created quantum model with {model.params.size} trainable parameters.")
    mlflow.log_text(str(model), "model.txt")

    mlflow.log_param("model.n_pulse_params", model.pulse_params.size)
    mlflow.log_param("model.n_gate_params", model.params.size)

    return {"model": model}


def generate_fourier_series(
    model: Model,
    coefficients_min: float,
    coefficients_max: float,
    zero_centered: bool,
    seed: int,
) -> jnp.ndarray:
    """
    Generates the Fourier series representation of a function.

    Parameters
    ----------
    domain_samples : jnp.ndarray
        Grid of domain samples.
    omega : List[List[float]]
        List of frequencies for each dimension.

    Returns
    -------
    jnp.ndarray
        Fourier series representation of the function.
    """
    domain_samples, fourier_samples, coefficients = Datasets.generate_fourier_series(
        random_key=jax.random.PRNGKey(seed),
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
    batch_size: int, domain_samples, fourier_samples, coefficients: jnp.ndarray
):
    if batch_size < 1:
        batch_size = domain_samples.shape[0]
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.array(domain_samples)),
            torch.from_numpy(np.array(fourier_samples).squeeze()),
            torch.from_numpy(np.array(coefficients).squeeze()),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": train_loader,
    }
