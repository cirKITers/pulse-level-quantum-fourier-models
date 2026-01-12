from typing import List, Dict
from rich.progress import track

import mlflow
from torch.utils.data import DataLoader

import pennylane as qml
import pennylane.numpy as np

from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients, FCC

from pulse_level_qfms.utils import (
    Losses,
    create_time_domain_viz,
)

import logging

log = logging.getLogger(__name__)


def calculate_fcc(
    model: Model,
    seed: int,
    n_samples: int,
    noise_params: Dict,
):
    fcc = FCC.get_fcc(model, n_samples, seed, noise_params=noise_params)

    mlflow.log_metric("fcc", fcc)

    return {
        "fcc": fcc,
    }


def train_model(
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    noise_params: Dict,
    loss_functions: List,
    loss_scalers: List,
    steps: int,
    learning_rate: float,
) -> None:
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    try:
        loss_functions = [getattr(Losses, loss) for loss in loss_functions]
    except AttributeError:
        log.error(f"Loss function is not valid. {loss_functions} must be in {Losses}")
        raise

    def cost(params, targets, **kwargs):
        predictions = model(params=params, **kwargs)

        # map loss_scaler and loss_function to a single loss using arbitrary numbers
        return np.sum(
            ls * lf(predictions, targets)
            for ls, lf in zip(loss_scalers, loss_functions)
        )

    def log_metrics(model, data, step, prefix=""):
        domain_samples = data.dataset.tensors[0].numpy()
        fourier_series = data.dataset.tensors[1].numpy()
        target_coeffs = data.dataset.tensors[2].numpy()

        prediction = model(
            params=model.params,
            inputs=domain_samples,
            noise_params=noise_params,
            execution_type="expval",
            force_mean=True,
        )
        predicted_coeffs = Coefficients.get_spectrum(
            model,
            shift=True,
            params=model.params,
            noise_params=noise_params,
            execution_type="expval",
            force_mean=True,
        )[0]

        mlflow.log_metric(
            f"{prefix}_mse", Losses.mse(prediction, fourier_series).item(), step=step
        )
        mlflow.log_metric(
            f"{prefix}_fmse",
            Losses.fmse(predicted_coeffs, target_coeffs).item(),
            step=step,
        )

    for step in track(range(steps), description="Training..", total=steps):

        for domain_samples, fourier_samples, coefficients in train_loader:
            domain_samples = domain_samples.numpy()
            fourier_samples = fourier_samples.numpy()

            model.params = opt.step(
                cost,
                model.params,
                inputs=domain_samples,
                targets=fourier_samples,
                noise_params=noise_params,
                execution_type="expval",
                force_mean=True,
            )

        log_metrics(model, data=train_loader, step=step, prefix="train")
        # log_metrics(model, data=valid_loader, step=step, prefix="valid")

    return {
        "model": model,
    }
