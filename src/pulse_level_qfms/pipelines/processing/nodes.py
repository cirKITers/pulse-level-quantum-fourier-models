from typing import List, Dict, Tuple, Optional
from rich.progress import track
import io

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
    def _calculate_coefficients(
        model: Model,
        n_samples: int,
        seed: int,
        scale: bool = False,
        sample_axis: str = "pulse",
        pulse_params_variance: float = 0.1,
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
            sample_axis (str, optional): Allows specifying "unitary", "pulse" or
                both. If both are specified, only unitary params actually receive
                the total number of samples and pulse parameter will get "distorted".
                If "pulse" is specified, a pulse simulation will be performed, else
                a unitary simulation will be performed.
            pulse_params_variance (float, optional): Variance of pulse parameters.
                If this is set to 0.0, the pulse parameters will not be distorted.
                I.e. a pulse simulation with default pulse parameters will run.


            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Parameters and Coefficients of size NxK
        """
        if n_samples > 0:
            if scale:
                total_samples = int(
                    np.power(2, model.n_qubits) * n_samples * model.n_input_feat
                )
            else:
                total_samples = n_samples

            rng = np.random.default_rng(seed)

            if "unitary" in sample_axis:
                model.initialize_params(rng=rng, repeat=total_samples)
                log.info(f"Sampling unitary parameters")
            else:
                model.initialize_params(rng=rng)
                log.info(f"Re-initializing unitary parameters")

            scaler = None

            if "pulse" in sample_axis:
                if "unitary" in sample_axis:
                    if pulse_params_variance == 0.0:
                        log.info(f"Using default pulse parameters")
                    else:
                        scaler = rng.normal(
                            loc=1.0,
                            scale=pulse_params_variance,
                            size=model.pulse_params.shape,
                        )
                        log.info(f"Distorting pulse parameters")
                else:
                    scaler = rng.normal(
                        loc=1.0,
                        scale=pulse_params_variance,
                        size=(*model.pulse_params.shape[:-1], total_samples),
                    )
                    log.info(f"Sampling pulse parameters")

            log.info(f"Using {total_samples} samples for FCC calculation")

        else:
            total_samples = 1

        # always a pulse simulation
        coeffs, freqs = Coefficients.get_spectrum(
            model,
            shift=True,
            trim=True,
            gate_mode="pulse" if "pulse" in sample_axis else "unitary",
            pulse_params=scaler,
            **kwargs,
        )

        variances = np.var(np.abs(coeffs), axis=1)
        means = np.mean(np.abs(coeffs), axis=1)

        for freq in freqs:
            mlflow.log_metric(f"coeff.mean.f{freq}", variances[int(freq)])
            mlflow.log_metric(f"coeff.var.f{freq}", means[int(freq)])

        return model.params, coeffs, freqs

    def _collect_pulse_params(self, model: Model):
        return model.params


# overwrite static method
FCC._calculate_coefficients = PulseFCC._calculate_coefficients


def calculate_fcc(
    model: Model,
    seed: int,
    n_samples: int,
    sample_axis: str,
    pulse_params_variance: float,
):
    log.info(f"Seed for FCC: {seed}")

    fourier_fingerprint, _ = PulseFCC.get_fourier_fingerprint(
        model,
        n_samples,
        seed,
        method="pearson",
        scale=True,
        weight=False,
        trim_redundant=True,
        sample_axis=sample_axis,
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


def evaluate_fidelity(
    model: Model,
    seed: int,
    n_samples: int,
    pulse_params_variance: float,
):
    log.info(f"Seed for fidelity check: {seed}")

    model.initialize_params(rng=np.random.default_rng(seed), repeat=n_samples)

    unitary_states = model(execution_type="density")
    pulse_states = model(
        pulse_params=np.random.normal(
            loc=1.0, scale=pulse_params_variance, size=model.pulse_params.shape
        ),
        execution_type="density",
    )

    fidelity = qml.math.fidelity(unitary_states, pulse_states)

    mlflow.log_metric("fidelity", fidelity)

    return {
        "fidelity": fidelity,
    }
