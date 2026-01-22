from typing import List, Dict, Tuple, Optional
from rich.progress import track

import mlflow
from torch.utils.data import DataLoader

import pennylane as qml
import pennylane.numpy as np

from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients, FCC
from qml_essentials.ansaetze import PulseInformation as pinfo

from pulse_level_qfms.utils import (
    Losses,
    create_time_domain_viz,
)


import logging

log = logging.getLogger(__name__)


class PulseFCC(FCC):
    @staticmethod
    def get_fourier_fingerprint(
        model: Model,
        n_samples: int,
        seed: int,
        method: Optional[str] = "pearson",
        scale: Optional[bool] = False,
        weight: Optional[bool] = False,
        trim_redundant: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shortcut method to get just the fourier fingerprint.
        This includes
        1. Calculating the coefficients (using `n_samples` and `seed`)
        2. Correlating the result from 1) using `method`
        3. Weighting the correlation matrix (if `weight` is True)
        4. Remove redundancies (if `trim_redundant` is True)

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            seed (int): Seed to initialize random parameters
            method (Optional[str], optional): Correlation method. Defaults to "pearson".
            scale (Optional[bool], optional): Whether to scale the number of samples.
                Defaults to False.
            weight (Optional[bool], optional): Whether to weight the correlation matrix.
                Defaults to False.
            trim_redundant (Optional[bool], optional): Whether to remove redundant
                correlations. Defaults to True.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The fourier fingerprint
            and the frequency indices
        """
        _, coeffs, freqs = PulseFCC._calculate_coefficients(
            model, n_samples, seed, scale, **kwargs
        )
        fourier_fingerprint = FCC._correlate(coeffs.transpose(), method=method)

        # perform weighting if requested
        fourier_fingerprint = (
            FCC._weighting(fourier_fingerprint) if weight else fourier_fingerprint
        )

        if trim_redundant:
            mask = FCC._calculate_mask(freqs)

            # apply the mask on the fingerprint
            fourier_fingerprint = mask * fourier_fingerprint

            row_mask = np.any(np.isfinite(fourier_fingerprint), axis=1)
            col_mask = np.any(np.isfinite(fourier_fingerprint), axis=0)

            fourier_fingerprint = fourier_fingerprint[row_mask][:, col_mask]

        return fourier_fingerprint, freqs

    @staticmethod
    def _calculate_coefficients(
        model: Model,
        n_samples: int,
        seed: int,
        scale: bool = False,
        pulse_params_variance: float = 0.1,
        fundamental_gates_only: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Fourier coefficients of a given model
        using `n_samples` and `seed`.
        Optionally, `noise_params` can be passed to perform noisy simulation.

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            seed (int): Seed to initialize random parameters
            scale (bool, optional): Whether to scale the number of samples.
                Defaults to False.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Parameters and Coefficients of size NxK
        """
        if n_samples > 0:
            if scale:
                total_samples = int(
                    np.power(2, model.n_qubits) * n_samples * model.n_input_feat
                )
                log.info(f"Using {total_samples} samples.")
            else:
                total_samples = n_samples
            rng = np.random.default_rng(seed)

            coeffs = []

            for _ in track(
                range(total_samples),
                description="Calculating Fourier Coefficients",
                total=total_samples,
            ):
                model.initialize_params(rng=rng)

                if fundamental_gates_only:
                    scaler = np.ones(model.pulse_params.shape)
                else:
                    scaler = rng.normal(
                        scale=pulse_params_variance, size=model.pulse_params.shape
                    )

                coeffs.append(
                    Coefficients.get_spectrum(
                        model,
                        shift=True,
                        trim=True,
                        gate_mode="pulse",
                        # pulse_params=scaler * model.pulse_params,
                        **kwargs,
                    )[0]
                )
        else:
            total_samples = 1
            coeffs, freqs = Coefficients.get_spectrum(
                model, shift=True, trim=True, **kwargs
            )

        return model.params, coeffs, freqs

    def _collect_pulse_params(self, model: Model):
        return model.params


def calculate_fcc(
    model: Model,
    seed: int,
    n_samples: int,
    fundamental_gates_only: bool,
    pulse_params_variance: float,
):
    fourier_fingerprint, _ = PulseFCC.get_fourier_fingerprint(
        model,
        n_samples,
        seed,
        method="pearson",
        scale=False,
        weight=False,
        trim_redundant=True,
        fundamental_gates_only=fundamental_gates_only,
        pulse_params_variance=pulse_params_variance,
    )

    fcc = PulseFCC.calculate_fcc(fourier_fingerprint)

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
