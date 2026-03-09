from typing import List, Dict, Tuple, Optional
from rich.progress import track
import jax
import optax

import mlflow
from torch.utils.data import DataLoader

import numpy as np

from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients, FCC
from qml_essentials.expressibility import Expressibility
from qml_essentials.math import fidelity, trace_distance, phase_difference

from pulse_level_qfms.utils import (
    Losses,
)

jax.config.update("jax_enable_x64", True)

import logging

log = logging.getLogger(__name__)


class PulseFCC(FCC):
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

            random_key = jax.random.PRNGKey(seed)
            # initialize model with new parameters and use batching if
            # "unitary" is specified in sampling axis
            if "unitary" in sample_axis:
                random_key = model.initialize_params(
                    random_key=random_key, repeat=total_samples
                )
                log.info(f"Sampling unitary parameters")
            else:
                random_key = model.initialize_params(random_key=random_key)
                log.info(f"Re-initializing unitary parameters")

            scaler = None

            # specifying "pulse" in sampling axis...
            if "pulse" in sample_axis:
                # either only distort pulse parameters...
                if "unitary" in sample_axis:
                    if pulse_params_variance == 0.0:
                        log.info(f"Using default pulse parameters")
                    else:
                        # sample differently for params
                        scaler = 1.0 + pulse_params_variance * jax.random.normal(
                            random_key,
                            shape=(
                                total_samples,
                                *model.pulse_params.shape,
                            ),
                        )
                        # but repeat over the input dimension
                        # Note, that the following steps are identical to what happens in
                        # _assimilate_batch
                        # [..., 1, B_P] -> [..., B_I, B_R]
                        scaler = scaler.repeat(np.prod(model.degree), axis=0)
                        # [..., B]
                        scaler = scaler.reshape(
                            np.prod(model.degree) * total_samples,
                            *model.pulse_params.shape,
                        )
                        # disable repeat for pulse parameters (to not further extend batch axis)
                        model.repeat_batch_axis = [True, True, False]
                        log.info(f"Sampling (pulse+std) parameters")
                # or actually samples them if we didn't do that before
                else:
                    scaler = 1.0 + pulse_params_variance * jax.random.normal(
                        random_key,
                        shape=(
                            *model.pulse_params.shape[:-1],
                            total_samples,
                        ),
                    )
                    log.info(f"Sampling pulse parameters")
            else:
                if pulse_params_variance == 0.0:
                    log.info(f"Using default pulse parameters")
                else:
                    scaler = 1.0 + pulse_params_variance * jax.random.normal(
                        random_key,
                        shape=model.pulse_params.shape,
                    )
                    log.info(f"Distorting pulse parameters")

            log.info(f"Using {total_samples} samples for FCC calculation")

        else:
            total_samples = 1

        # always a pulse simulation for coefficient calculation (consistency)
        coeffs, freqs = Coefficients.get_spectrum(
            model,
            shift=True,
            trim=True,
            gate_mode="pulse" if "pulse" in sample_axis else "unitary",
            pulse_params=scaler if "pulse" in sample_axis else None,
            **kwargs,
        )

        # calculate variances and means over all samples (preserve freq. axis)
        variances = np.abs(coeffs).var(axis=1)
        means = np.abs(coeffs).mean(axis=1)

        # log values for each frequency component
        for freq, var, mean in zip(freqs, variances, means, strict=True):
            mlflow.log_metric(f"coeff.mean.f{freq}", mean)
            mlflow.log_metric(f"coeff.var.f{freq}", var)

        return model.params, coeffs, freqs


def calculate_fcc(
    model: Model,
    seed: int,
    n_samples: int,
    scale: bool,
    method: str,
    weighting: bool,
    sample_axis: str,
    pulse_params_variance: float,
):
    log.info(f"Seed for FCC: {seed}")

    # log before we do any batching
    mlflow.log_metric("n_pulse_params", model.pulse_params.size)
    mlflow.log_metric("n_gate_params", model.params.size)

    # call our modified class to calculate the fourier fingerprint
    fourier_fingerprint, _ = PulseFCC.get_fourier_fingerprint(
        model,
        n_samples,
        seed,
        method=method,
        scale=scale,
        weight=weighting,
        trim_redundant=True,
        sample_axis=sample_axis,
        pulse_params_variance=pulse_params_variance,
    )

    # and finally the fcc
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
    opt = optax.adam(learning_rate)

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
    scale: bool,
    pulse_params_variance: float,
):
    log.info(f"Seed for fidelity check: {seed}")

    if scale:
        total_samples = int(np.power(2, model.n_qubits) * n_samples)
    else:
        total_samples = n_samples

    log.info(f"Using {total_samples} samples for fidelity check")

    random_key = jax.random.PRNGKey(seed)
    random_key = model.initialize_params(random_key=random_key, repeat=total_samples)

    # calculate density matrices for unitary and pulse circuits
    unitary_states = model(execution_type="density")

    scaler = 1.0 + pulse_params_variance * jax.random.normal(
        random_key,
        shape=(
            *model.pulse_params.shape[:-1],
            total_samples,
        ),
    )
    # disable repeat for pulse parameters
    model.repeat_batch_axis = [True, True, False]

    pulse_states = model(
        pulse_params=scaler,
        gate_mode="pulse",
        execution_type="density",
    )

    # calculate overlap
    fi = fidelity(unitary_states, pulse_states)
    # ph = phase_difference(unitary_states, pulse_states)
    td = trace_distance(unitary_states, pulse_states)

    # average over all samples
    mlflow.log_metric("fidelity", np.mean(fi))
    # mlflow.log_metric("phase", np.mean(ph))
    mlflow.log_metric("trace-distance", np.mean(td))

    return {
        "fidelity": fidelity,
    }


def evaluate_expressibility(
    model: Model,
    seed: int,
    n_samples: int,
    n_bins: int,
    scale: bool,
    pulse_params_variance: float,
):
    log.info(f"Seed for expressibility: {seed}")

    if scale:
        total_samples = int(np.power(2, model.n_qubits) * n_samples)
    else:
        total_samples = n_samples

    log.info(f"Using {total_samples} samples for expressibility")

    random_key = jax.random.PRNGKey(seed)

    scaler = 1.0 + pulse_params_variance * jax.random.normal(
        random_key,
        shape=(
            *model.pulse_params.shape[:-1],
            total_samples,
        ),
    )
    model.repeat_batch_axis = [True, True, False]

    input_domain, bins, dist_circuit = Expressibility.state_fidelities(
        seed=seed,
        n_samples=total_samples,
        n_bins=n_bins,
        scale=False,
        model=model,
        pulse_params=scaler,
        gate_mode="pulse",
    )

    input_domain, dist_haar = Expressibility.haar_integral(
        n_qubits=model.n_qubits,
        n_bins=n_bins,
        cache=True,
    )

    kl_dist = Expressibility.kullback_leibler_divergence(dist_circuit, dist_haar)
    expressibility = np.mean(kl_dist)

    # average over all samples
    mlflow.log_metric("expressibility", expressibility)

    return {
        "expressibility": expressibility,
    }
