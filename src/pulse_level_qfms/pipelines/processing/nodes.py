from typing import List, Dict, Tuple, Optional
from rich.progress import track
import jax
import optax

import mlflow
from torch.utils.data import DataLoader

import jax.numpy as jnp

from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients, FCC
from qml_essentials.expressibility import Expressibility
from qml_essentials.math import fidelity, trace_distance, phase_difference

from scipy.linalg import sqrtm

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
            Tuple[jnp.ndarray, jnp.ndarray]: The fourier fingerprint
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

            row_mask = jnp.any(jnp.isfinite(fourier_fingerprint), axis=1)
            col_mask = jnp.any(jnp.isfinite(fourier_fingerprint), axis=0)

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
            Tuple[jnp.ndarray, jnp.ndarray]: Parameters and Coefficients of size NxK
        """
        if n_samples > 0:
            if scale:
                total_samples = int(
                    jnp.power(2, model.n_qubits) * n_samples * model.n_input_feat
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
                                *model.pulse_params.shape[
                                    1:
                                ],  # starting from batch dimension
                            ),
                        )
                        degree = jnp.prod(jnp.array(model.degree))
                        # but repeat over the input dimension
                        # Note, that the following steps are identical to what happens in
                        # _assimilate_batch
                        # [B_I, 1, B_R, ...]
                        scaler = scaler.repeat(degree, axis=0)
                        # [..., B]
                        scaler = scaler.reshape(
                            degree * total_samples,
                            *model.pulse_params.shape[1:],
                        )
                        # disable repeat for pulse parameters (to not further extend batch axis)
                        model.repeat_batch_axis = [True, True, False]
                        log.info(f"Sampling (pulse+std) parameters")
                # or actually samples them if we didn't do that before
                else:
                    scaler = 1.0 + pulse_params_variance * jax.random.normal(
                        random_key,
                        shape=(
                            total_samples,
                            *model.pulse_params.shape[1:],
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
        variances = jnp.abs(coeffs).var(axis=1)
        means = jnp.abs(coeffs).mean(axis=1)

        # log values for each frequency component
        for freq, var, mean in zip(freqs, variances, means, strict=True):
            mlflow.log_metric(f"coeff.mean.f{freq}", mean)
            mlflow.log_metric(f"coeff.var.f{freq}", var)

        return model.params, coeffs, freqs


class PulseExpressibility(Expressibility):
    """Override the expressibility sampling to support a ``sample_axis``
    parameter, mirroring the approach used in :class:`PulseFCC`.

    When ``sample_axis`` contains ``"unitary"`` the unitary parameters are
    randomised across ``n_samples`` sets (original library behaviour).
    When it contains ``"pulse"`` the pulse parameters are distorted with
    a Gaussian scaler controlled by ``pulse_params_variance``.
    Both may be active at the same time.
    """

    @staticmethod
    def _sample_state_fidelities(
        model: Model,
        n_samples: int,
        random_key: jax.random.PRNGKey,
        sample_axis: List[str],
        pulse_params_variance: float,
        scale: bool = False,
    ) -> jnp.ndarray:
        """
        Compute the state fidelities for pairs of random parameter sets,
        with control over which axes (unitary / pulse) are sampled.

        Args:
            model (Model): The quantum model.
            n_samples (int): Number of *pairs* of parameter sets.
            random_key (jax.random.PRNGKey): JAX random key for parameter
                initialization and pulse scaler generation.
            sample_axis (List[str]): Subset of ``["unitary", "pulse"]``.
            pulse_params_variance (float): Std-dev of the multiplicative
                Gaussian noise applied to pulse parameters.
            scale (bool): Whether to scale the number of samples.

        Returns:
            jnp.ndarray: Array of shape ``(n_samples,)`` with fidelities.
        """
        if scale:
            total_samples = int(jnp.power(2, model.n_qubits) * n_samples)
        else:
            total_samples = n_samples

        if "unitary" in sample_axis:
            random_key = model.initialize_params(
                random_key=random_key, repeat=total_samples * 2
            )
            log.info("Expressibility: sampling unitary parameters")
        else:
            random_key = model.initialize_params(random_key=random_key)
            log.info("Expressibility: re-initializing unitary parameters")

        scaler = None
        gate_mode = "unitary"

        if "pulse" in sample_axis:
            gate_mode = "pulse"
            if pulse_params_variance == 0.0:
                log.info("Expressibility: using default pulse parameters")
            else:
                if "unitary" in sample_axis:
                    # Both axes active: create a scaler that pairs 1:1 with
                    # the already-batched unitary params.
                    scaler = 1.0 + pulse_params_variance * jax.random.normal(
                        random_key,
                        shape=(
                            total_samples * 2,
                            *model.pulse_params.shape[1:],
                        ),
                    )
                    model.repeat_batch_axis = [True, True, False]
                    log.info("Expressibility: sampling (unitary+pulse) parameters")
                else:
                    # Pulse only: unitary params are *not* batched (B_P=1).
                    scaler = 1.0 + pulse_params_variance * jax.random.normal(
                        random_key,
                        shape=(
                            total_samples * 2,
                            *model.pulse_params.shape[1:],
                        ),
                    )
                    log.info("Expressibility: sampling pulse parameters only")

        log.info(
            f"Expressibility: using {total_samples} sample pairs "
            f"(gate_mode={gate_mode})"
        )

        sv: jnp.ndarray = model(
            params=model.params,
            execution_type="density",
            gate_mode=gate_mode,
            pulse_params=scaler,
        )

        sqrt_sv1: jnp.ndarray = jnp.array([sqrtm(m) for m in sv[:total_samples]])
        inner_fidelity = sqrt_sv1 @ sv[total_samples:] @ sqrt_sv1

        fid: jnp.ndarray = (
            jnp.trace(
                jnp.array([sqrtm(m) for m in inner_fidelity]),
                axis1=1,
                axis2=2,
            )
            ** 2
        )

        return jnp.abs(fid)

    @staticmethod
    def state_fidelities(
        n_samples: int,
        n_bins: int,
        model: Model,
        random_key: jax.random.PRNGKey,
        sample_axis: List[str],
        pulse_params_variance: float,
        scale: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample the state fidelities and histogram them.

        Wraps :meth:`_sample_state_fidelities` with histogram binning,
        identical to the base class but routed through our override.
        """
        if scale:
            n_samples = int(jnp.power(2, model.n_qubits) * n_samples)
            n_bins = model.n_qubits * n_bins

        fidelities = PulseExpressibility._sample_state_fidelities(
            model=model,
            n_samples=n_samples,
            random_key=random_key,
            sample_axis=sample_axis,
            pulse_params_variance=pulse_params_variance,
            scale=False,  # already applied above
        )

        y: jnp.ndarray = jnp.linspace(0, 1, n_bins + 1)
        z, _ = jnp.histogram(fidelities, bins=y)
        z = z / n_samples

        return y, z


def calculate_fcc(
    model: Model,
    seed: int,
    n_samples: int,
    scale: bool,
    method: str,
    weighting: bool,
    sample_axis: str,
    pulse_params_variance: float,
    numerical_cap: float,
):
    log.info(f"Seed for FCC: {seed}")

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
        numerical_cap=numerical_cap,
    )

    # and finally the fcc
    fcc = PulseFCC.calculate_fcc(fourier_fingerprint)

    mlflow.log_metric("fcc", fcc)

    return {
        "fcc": fcc,
    }


def log_metrics(
    model,
    data,
    step,
    prefix="",
    gate_mode="unitary",
    noise_params=None,
    pulse_params=None,
):
    domain_samples = data.dataset.tensors[0].numpy()
    fourier_series = data.dataset.tensors[1].numpy()
    target_coeffs = data.dataset.tensors[2].numpy()

    prediction = model(
        params=model.params,
        inputs=domain_samples,
        execution_type="expval",
        force_mean=True,
        gate_mode=gate_mode,
        pulse_params=pulse_params,
        noise_params=noise_params,
    )
    predicted_coeffs = Coefficients.get_spectrum(
        model,
        shift=True,
        params=model.params,
        execution_type="expval",
        force_mean=True,
        gate_mode=gate_mode,
        pulse_params=pulse_params,
        noise_params=noise_params,
    )[
        0
    ]  # get only coeffs, not freqs

    mlflow.log_metric(
        f"{prefix}_mse", Losses.mse(prediction, fourier_series).item(), step=step
    )
    mlflow.log_metric(
        f"{prefix}_fmse",
        Losses.fmse(predicted_coeffs, target_coeffs).item(),
        step=step,
    )


def _jacobian_rank(
    model: Model,
    theta: jnp.ndarray,
    lam: jnp.ndarray,
    gate_mode: str,
    argnums: Tuple[int, ...],
    tol_rel: float,
) -> Tuple[int, float, Tuple[int, ...]]:
    """Compute the numerical rank of the Jacobian of the Fourier coefficients
    of *model* w.r.t. the parameter groups indicated by *argnums*.

    The Fourier coefficients are stacked into a real vector
    ``[Re(c_ω), Im(c_ω)]`` so the resulting Jacobian is a real
    ``(2|Ω|, |params|)`` matrix from which a meaningful rank can be
    obtained via SVD.  ``tol_rel`` is multiplied with the largest
    singular value to obtain the cutoff used for the numerical rank
    estimate (matches ``numpy.linalg.matrix_rank``'s default policy).

    Args:
        model: Already-instantiated quantum Fourier model.
        theta: Unitary parameter vector ``θ`` (shape as in ``model.params``).
        lam: Pulse-scaling parameter vector ``λ`` (shape as in
            ``model.pulse_params``).
        gate_mode: ``"pulse"`` or ``"unitary"`` — must match the regime
            in which the manuscript's J_θ / J_ext are defined.
        argnums: Subset of ``(0, 1)`` indicating which arguments to
            differentiate w.r.t. — ``(0,)`` gives ``J_θ``, ``(0, 1)``
            gives ``J_ext``.
        tol_rel: Relative tolerance for the numerical rank.

    Returns:
        Tuple ``(rank, sv_min_above_tol, jacobian_shape)`` — ``rank`` is
        the integer numerical rank, ``sv_min_above_tol`` is the smallest
        singular value above the cutoff (or ``0.0`` when the matrix is
        zero) and ``jacobian_shape`` records the flattened Jacobian
        shape for diagnostics.
    """
    def _coeff_vec(theta_, lam_):
        coeffs, _ = Coefficients.get_spectrum(
            model,
            params=theta_,
            pulse_params=lam_,
            gate_mode=gate_mode,
            shift=False,
            trim=False,
            numerical_cap=-1,
            force_mean=True,
            execution_type="expval",
        )
        # Stack real and imaginary parts so SVD gives a real-valued rank.
        return jnp.concatenate([coeffs.real.ravel(), coeffs.imag.ravel()])

    jac = jax.jacrev(_coeff_vec, argnums=argnums)(theta, lam)
    if isinstance(jac, tuple):
        # Flatten each block on its parameter axes and concatenate columns.
        blocks = [j.reshape(j.shape[0], -1) for j in jac]
        J = jnp.concatenate(blocks, axis=1)
    else:
        J = jac.reshape(jac.shape[0], -1)

    J_np = jnp.asarray(J)
    s = jnp.linalg.svd(J_np, compute_uv=False)
    s_max = float(jnp.max(s)) if s.size > 0 else 0.0
    cutoff = tol_rel * s_max
    rank = int(jnp.sum(s > cutoff))
    sv_min_above = float(jnp.min(jnp.where(s > cutoff, s, jnp.inf))) if rank > 0 else 0.0
    return rank, sv_min_above, tuple(int(d) for d in J.shape)


def _log_jacobian_ranks(
    model: Model,
    theta: jnp.ndarray,
    lam: jnp.ndarray,
    gate_mode: str,
    tol_rel: float,
    step: int,
) -> None:
    """Compute ``rank J_θ``, ``rank J_ext`` and ``Δr`` and log to MLflow.

    A non-zero ``Δr = rank J_ext - rank J_θ`` certifies that
    the pulse-scaling parameters provide new search directions in
    Fourier-coefficient space beyond what the unitary parameters alone
    can reach.

    Args:
        model: The model whose autodiff is exercised.
        theta: Current unitary parameters.
        lam: Current pulse-scaling parameters.
        gate_mode: ``"pulse"`` or ``"unitary"`` — the regime in which
            ranks are evaluated.
        tol_rel: Relative SVD cutoff used for the numerical rank.
        when: Tag for the metric name (``"init"``/``"trained"``).
        step: MLflow step coordinate.
    """
    log.info(f"Computing Jacobian ranks (gate_mode={gate_mode}) ...")
    r_theta, sv_theta, shape_theta = _jacobian_rank(
        model, theta, lam, gate_mode, argnums=(0,), tol_rel=tol_rel
    )
    r_ext, sv_ext, shape_ext = _jacobian_rank(
        model, theta, lam, gate_mode, argnums=(0, 1), tol_rel=tol_rel
    )
    delta_r = r_ext - r_theta
    log.info(
        f"  J_θ shape={shape_theta} rank={r_theta} | "
        f"J_ext shape={shape_ext} rank={r_ext} | Δr={delta_r}"
    )
    mlflow.log_metric(f"rank.r_theta", r_theta, step=step)
    mlflow.log_metric(f"rank.r_ext", r_ext, step=step)
    mlflow.log_metric(f"rank.sv_theta", sv_theta, step=step)
    mlflow.log_metric(f"rank.sv_ext", sv_ext, step=step)


def train_model(
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    noise_params: Dict,
    loss_functions: List,
    loss_scalers: List,
    steps: int,
    learning_rate: float,
    train_unitary: bool,
    train_pulse: bool,
    pulse_learning_rate: Optional[float] = None,
    rank_eval_enabled: bool = False,
    rank_eval_tol_rel: float = 1e-8,
    rank_report_interval: int = 100,
) -> None:
    gate_mode = "pulse" if train_pulse else "unitary"

    # create params dict
    params = {"unitary": model.params}
    if train_pulse:
        # pulse_params scaler: starts at ones (i.e. no deviation from default)
        params["pulse"] = jnp.ones_like(model.pulse_params)

    # set a per-group optimizer
    pulse_lr = (
        pulse_learning_rate if pulse_learning_rate is not None else learning_rate * 0.1
    )
    log.info(
        f"Learning rates - unitary: {learning_rate}, "
        f"pulse: {pulse_lr if train_pulse else 'N/A (not training pulse params)'}"
    )

    if train_pulse:
        # Separate optimizer chains: aggressive clipping + smaller lr for pulse
        pulse_opt = optax.adam(pulse_lr)
        unitary_opt = optax.adam(learning_rate)

        # Combine into a single optimizer keyed by the param labels
        label_fn = lambda params: {k: k for k in params}  # noqa: E731
        opt = optax.multi_transform(
            {"unitary": unitary_opt, "pulse": pulse_opt},
            label_fn,
        )
    else:
        opt = optax.adam(learning_rate)

    opt_state = opt.init(params)

    if rank_eval_enabled and train_pulse:
        # Initial / "generic" Jacobian ranks at the untrained parameters.
        # Always evaluate in pulse mode: J_θ and J_ext are both defined w.r.t.
        # (θ, λ) of the pulse model, with λ=ones recovering the unitary
        # baseline coefficients
        _log_jacobian_ranks(
            model,
            theta=params["unitary"],
            lam=params.get("pulse", jnp.ones_like(model.pulse_params)),
            gate_mode="pulse",
            tol_rel=rank_eval_tol_rel,
            step=0,
        )

    try:
        loss_functions = [getattr(Losses, loss) for loss in loss_functions]
    except AttributeError:
        log.error(f"Loss function is not valid. {loss_functions} must be in {Losses}")
        raise

    log.info(f"Using gate mode: {gate_mode} for training")
    if train_pulse:
        log.info(f"Pulse params are trainable (pulse_lr={pulse_lr})")

    def cost(params_dict, targets, **kwargs):
        predictions = model(
            params=params_dict["unitary"],
            pulse_params=params_dict.get("pulse", None) if train_pulse else None,
            **kwargs,
        )

        total_loss = jnp.array(0.0)
        for ls, lf in zip(loss_scalers, loss_functions):
            total_loss = total_loss + ls * lf(predictions, targets)
        return total_loss

    for step in track(range(steps), description="Training..", total=steps):

        for domain_samples, fourier_samples, coefficients in train_loader:
            domain_samples = jnp.array(domain_samples.numpy())
            fourier_samples = jnp.array(fourier_samples.numpy())

            grads = jax.grad(cost)(
                params,
                inputs=domain_samples,
                targets=fourier_samples,
                execution_type="expval",
                force_mean=True,
                gate_mode=gate_mode,
                noise_params=noise_params,
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        model.params = params["unitary"]
        if train_pulse:
            model.pulse_params = params["pulse"]
            mlflow.log_metric(
                "pulse_scaler_mean", float(jnp.mean(params["pulse"])), step=step
            )
            mlflow.log_metric(
                "pulse_scaler_std", float(jnp.std(params["pulse"])), step=step
            )

        log_metrics(
            model,
            data=train_loader,
            step=step,
            prefix="train",
            gate_mode=gate_mode,
            noise_params=noise_params,
            pulse_params=params.get("pulse", None) if train_pulse else None,
        )
        if rank_eval_enabled and train_pulse and step + 1 % rank_report_interval == 0:
            # Initial / "generic" Jacobian ranks at the untrained parameters.
            # Always evaluate in pulse mode: J_θ and J_ext are both defined w.r.t.
            # (θ, λ) of the pulse model, with λ=ones recovering the unitary
            # baseline coefficients
            _log_jacobian_ranks(
                model,
                theta=params["unitary"],
                lam=params.get("pulse", jnp.ones_like(model.pulse_params)),
                gate_mode="pulse",
                tol_rel=rank_eval_tol_rel,
                when="training",
                step=step + 1, # because we report init
            )

        # log_metrics(model, data=valid_loader, step=step, prefix="valid",
        #             gate_mode=gate_mode,
        #             pulse_params=params.get("pulse", None) if train_pulse else None)

    # if rank_eval_enabled:
    #     # Trained Jacobian ranks at the converged parameters.
    #     _log_jacobian_ranks(
    #         model,
    #         theta=params["unitary"],
    #         lam=params.get("pulse", jnp.ones_like(model.pulse_params)),
    #         gate_mode="pulse",
    #         tol_rel=rank_eval_tol_rel,
    #         when="trained",
    #         step=max(steps - 1, 0),
    #     )

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
        total_samples = int(jnp.power(2, model.n_qubits) * n_samples)
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
            total_samples,
            *model.pulse_params.shape[1:],
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
    mlflow.log_metric("fidelity", jnp.mean(fi))
    # mlflow.log_metric("phase", jnp.mean(ph))
    mlflow.log_metric("trace-distance", jnp.mean(td))

    return {
        "fidelity": fidelity,
    }


def evaluate_expressibility(
    model: Model,
    seed: int,
    n_samples: int,
    n_bins: int,
    scale: bool,
    sample_axis: List[str],
    pulse_params_variance: float,
):
    log.info(f"Seed for expressibility: {seed}")
    log.info(
        f"Sample axis: {sample_axis}, pulse_params_variance: {pulse_params_variance}"
    )

    random_key = jax.random.PRNGKey(seed)

    _, dist_circuit = PulseExpressibility.state_fidelities(
        n_samples=n_samples,
        n_bins=n_bins,
        scale=scale,
        model=model,
        random_key=random_key,
        sample_axis=sample_axis,
        pulse_params_variance=pulse_params_variance,
    )

    _, dist_haar = Expressibility.haar_integral(
        n_qubits=model.n_qubits,
        n_bins=n_bins,
        cache=True,
        scale=scale,
    )

    kl_dist = Expressibility.kullback_leibler_divergence(dist_circuit, dist_haar)
    expressibility = jnp.mean(kl_dist)

    mlflow.log_metric("expressibility", expressibility)

    return {
        "expressibility": expressibility,
    }
