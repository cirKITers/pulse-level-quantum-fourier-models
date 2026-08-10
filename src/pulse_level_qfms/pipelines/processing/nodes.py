from typing import List, Dict, Tuple, Optional
from rich.progress import track
import time
import math
import jax
import optax

import mlflow
from mlflow.entities import Metric
from torch.utils.data import DataLoader

import numpy as np
import jax.numpy as jnp

from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients, Datasets, FCC
from qml_essentials.expressibility import Expressibility
from qml_essentials.math import fidelity, trace_distance, phase_difference

from scipy.linalg import sqrtm

from pulse_level_qfms.utils import (
    Losses,
)

jax.config.update("jax_enable_x64", True)

import logging

log = logging.getLogger(__name__)

_PULSE_GROUPS = {
    "unitary": (),
    "ansatz_pulse": ("pulse",),
    "enc_pulse": ("enc_pulse",),
    "all_pulse": ("pulse", "enc_pulse"),
}

# position of each pulse scaler group in the differentiated coefficient
# function, i.e. the ``argnums`` handle used to extend ``J_\theta``
_GROUP_ARGNUM = {"pulse": 1, "enc_pulse": 2}

# batch axis each pulse scaler group owns in ``model.repeat_batch_axis``,
# which is ordered [inputs, params, pulse_params, enc_pulse_params]
_GROUP_BATCH_AXIS = {"pulse": 2, "enc_pulse": 3}

# inverse of _PULSE_GROUPS, i.e. the mode that runs exactly the given pulse
# groups. Used to derive the gate mode from the sampled quantities, since
# running a group at pulse level without perturbing it reproduces the unitary
# result (the pulses are calibrated to the ideal gates).
_MODE_BY_GROUPS = {frozenset(groups): mode for mode, groups in _PULSE_GROUPS.items()}

# model attribute each trainable group writes back to. "enc" is the unitary
# trainable-frequency knob enc_params, the others are pulse scalers.
_GROUP_ATTR = {
    "pulse": "pulse_params",
    "enc_pulse": "enc_pulse_params",
    "enc": "enc_params",
}


class PulseFCC(FCC):
    @classmethod
    def _calculate_coefficients(
        cls,
        model: Model,
        n_samples: int,
        seed: int,
        scale: bool = False,
        sample_axis: List[str] = ("unitary", "pulse"),
        gate_mode: Optional[str] = None,
        pulse_params_variance: float = 0.1,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculates the Fourier coefficients of a given model
        using `n_samples` and `seed`.

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            seed (int): Seed to initialize random parameters
            scale (bool, optional): Whether to scale the number of samples.
                Defaults to False.
            sample_axis (List[str], optional): Which quantities are randomised
                across the samples, a subset of "unitary" (the variational
                parameters $\\theta$), "pulse" (the ansatz pulse scalers
                $\\lambda$) and "enc_pulse" (the encoding pulse scalers
                $\\eta$). Entries are matched exactly, so "pulse" does not
                select "enc_pulse". All selected quantities are drawn jointly,
                i.e. sample $j$ is one draw of every selected quantity rather
                than an outer product over them. The four pulse regimes are
                selected by which pulse groups appear here: none gives
                "unitary", "pulse" gives "ansatz_pulse", "enc_pulse" gives
                "enc_pulse" and both give "all_pulse".
            gate_mode (Optional[str], optional): Gate execution backend used
                for the coefficient calculation, one of "unitary",
                "ansatz_pulse", "enc_pulse" or "all_pulse". Defaults to None,
                in which case it is derived from `sample_axis` as the mode
                that runs exactly the sampled pulse groups. Pass it explicitly
                only to run a group at pulse level without perturbing it,
                which differs from the unitary result only once the pulses are
                detuned or noise is enabled. A pulse entry in `sample_axis`
                that the mode does not run at pulse level is ignored with a
                warning.
            pulse_params_variance (float, optional): Variance of the pulse
                scalers. If this is set to 0.0, the pulse parameters are not
                distorted, i.e. the simulation runs with default pulse
                parameters.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Parameters,
            coefficients of size NxK and the corresponding frequencies.
        """
        if gate_mode is None:
            # the sampled pulse groups fully determine the regime, so there is
            # no combination left that the mode could fail to honour
            gate_mode = _MODE_BY_GROUPS[
                frozenset(g for g in _GROUP_BATCH_AXIS if g in sample_axis)
            ]
            log.info(f"Derived gate_mode={gate_mode} from sample_axis={sample_axis}")
        elif gate_mode not in _PULSE_GROUPS:
            raise ValueError(
                f"Unknown gate_mode: {gate_mode}. Use one of {list(_PULSE_GROUPS)}."
            )

        # only the groups this mode runs at pulse level can be sampled, the
        # model rejects a scaler group it does not run and such a group would
        # be inert on the coefficients anyway
        groups = _PULSE_GROUPS[gate_mode]
        sampled_groups = [group for group in groups if group in sample_axis]
        ignored = [
            group
            for group in _GROUP_BATCH_AXIS
            if group in sample_axis and group not in groups
        ]
        if ignored:
            log.warning(
                f"sample_axis entries {ignored} have no effect under "
                f"gate_mode={gate_mode}, which runs {list(groups)} at pulse level"
            )

        scalers = {group: None for group in groups}

        if n_samples > 0:
            if scale:
                total_samples = int(
                    jnp.power(2, model.n_qubits) * n_samples * model.n_input_feat
                )
            else:
                total_samples = n_samples

            if pulse_params_variance == 0.0 and sampled_groups:
                log.info("Zero pulse variance, using default pulse parameters")
                sampled_groups = []

            random_key = jax.random.PRNGKey(seed)
            # initialize model with new parameters and use batching if
            # "unitary" is specified in sampling axis
            if "unitary" in sample_axis:
                random_key = model.initialize_params(
                    random_key=random_key, repeat=total_samples
                )
                # the parameter axis B_P already carries the samples
                sample_axis_owned = True
                log.info("Sampling unitary parameters")
            else:
                random_key = model.initialize_params(random_key=random_key)
                sample_axis_owned = False
                log.info("Re-initializing unitary parameters")

            # number of input samples the Fourier transform evaluates, i.e. B_I
            mfs, mts = kwargs.get("mfs", 1), kwargs.get("mts", 1)
            n_inputs = int(jnp.prod(jnp.array([mts * mfs * d for d in model.degree])))

            repeat_batch_axis = [True, True, True, True]

            for group in sampled_groups:
                random_key, sub_key = jax.random.split(random_key)
                payload_shape = getattr(model, f"{group}_params").shape[1:]
                scaler = 1.0 + pulse_params_variance * jax.random.normal(
                    sub_key,
                    shape=(total_samples, *payload_shape),
                )

                if sample_axis_owned:
                    # Another quantity already spans the samples, so this group
                    # cannot own a batch axis of its own without turning the
                    # draws into an outer product. Instead lay it out along the
                    # flattened batch, whose element k holds sample
                    # k % total_samples, and mask its axis so _assimilate_batch
                    # leaves it as is. This is the same tiling _assimilate_batch
                    # applies to the axis that does not own the batch.
                    scaler = jnp.tile(scaler, (n_inputs, *([1] * len(payload_shape))))
                    repeat_batch_axis[_GROUP_BATCH_AXIS[group]] = False
                else:
                    # nothing sampled yet, so this group owns the sample axis
                    # and _assimilate_batch expands it against the inputs
                    sample_axis_owned = True

                scalers[group] = scaler
                log.info(f"Sampling {group} parameters")

            model.repeat_batch_axis = repeat_batch_axis

            if not sample_axis_owned:
                log.warning(
                    "Neither unitary nor pulse parameters are sampled, the "
                    "coefficients will be identical across all samples"
                )

            log.info(f"Using {total_samples} samples for FCC calculation")

        else:
            total_samples = 1

        coeffs, freqs = Coefficients.get_spectrum(
            model,
            shift=True,
            trim=True,
            gate_mode=gate_mode,
            pulse_params=scalers.get("pulse"),
            enc_pulse_params=scalers.get("enc_pulse"),
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
            gate_mode = "ansatz_pulse"
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
    sample_axis: List[str],
    pulse_params_variance: float,
    numerical_cap: float,
):
    log.info(f"Seed for FCC: {seed}")
    log.info(f"Sample axis: {sample_axis}")

    # gate_mode is derived from sample_axis, see _calculate_coefficients
    fourier_fingerprint, freqs, coeffs = PulseFCC.get_fourier_fingerprint(
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


def calculate_spectrum(
    model: Model,
    seed: int,
    n_samples: int,
    scale: bool,
    sample_axis: List[str],
    pulse_params_variance: float,
    mfs: int,
    mts: int,
):
    """
    Logs the frequency spectrum of the model under pulse distortion.

    Oversamples the frequency axis by `mts`, which gives a bin spacing of
    $1/mts$ instead of the integer grid used for the FCC. Only on such a
    grid can a frequency shift be told apart from a pure change in
    amplitude, since an integer grid has no bins for the shifted
    components to occupy.

    The per-frequency metrics are logged by
    :meth:`PulseFCC._calculate_coefficients`.

    Args:
        model (Model): The QFM model
        seed (int): Seed to initialize random parameters
        n_samples (int): Number of samples to average the coefficients over
        scale (bool): Whether to scale the number of samples
        sample_axis (List[str]): Which quantities are randomised across the
            samples, see :meth:`PulseFCC._calculate_coefficients`
        pulse_params_variance (float): Variance of the pulse scalers
        mfs (int): Multiplicator for the highest frequency
        mts (int): Multiplicator for the number of time samples, i.e. the
            frequency resolution
    """
    log.info(f"Seed for spectrum: {seed}")
    log.info(f"Sample axis: {sample_axis}, mfs={mfs}, mts={mts}")

    # gate_mode is derived from sample_axis, see _calculate_coefficients.
    # numerical_cap is disabled so that every run reports the same frequency
    # grid: with a cap, the bins that vanish at zero variance would be
    # dropped and the runs could no longer be aggregated per frequency.
    PulseFCC._calculate_coefficients(
        model,
        n_samples,
        seed,
        scale,
        sample_axis=sample_axis,
        pulse_params_variance=pulse_params_variance,
        numerical_cap=-1,
        mfs=mfs,
        mts=mts,
    )

    return {}


def _encoding_gates(model: Model, feature: int = 0) -> List[Tuple[int, int, float]]:
    """
    The encoding gate instances of one input feature and their generators.

    Every position the data re-upload mask marks is one encoding gate, and each
    of them carries its own scaler. Which frequency a gate contributes follows
    from the encoding strategy, where qubit $q$ is driven at
    $\\text{base}^q$ with base 1, 2 or 3 for hamming, binary and ternary. Two
    gates can therefore share a generator, either across layers or, under
    hamming, across qubits as well.

    Args:
        model (Model): The QFM model.
        feature (int, optional): Index of the input feature. Defaults to 0.

    Returns:
        List[Tuple[int, int, float]]: Layer, qubit and generator frequency of
        each encoding gate, in data re-upload order.

    Raises:
        ValueError: If the encoding strategy has no per-gate generator to
            scale.
    """
    base = {"hamming": 1, "binary": 2, "ternary": 3}.get(model._enc._strategy)
    if base is None:
        raise ValueError(
            f"The landscape sweep does not support the "
            f"{model._enc._strategy!r} encoding strategy, which has no "
            "per-gate generator to scale."
        )

    mask = np.asarray(model.data_reupload[..., feature], dtype=bool)
    #TODO: the following should be replacable by some qml-essentials internal tool
    return [(int(l), int(q), float(base**q)) for l, q in zip(*np.nonzero(mask))]


def _comb_support(generators: np.ndarray, etas: np.ndarray) -> np.ndarray:
    """
    The frequency comb the encoding reaches at the given scalers.

    Each encoding gate contributes $-1$, $0$ or $+1$ times its scaled generator
    $\\gamma_j \\eta_j$, so the comb is their Minkowski sum. This mirrors
    `Datasets._displace_generators`, which is what builds the off-grid target,
    down to the rounding it deduplicates on. Deduplicating as the sum is built
    keeps the intermediate sets at the size of the comb rather than
    $3^{n_\\text{gates}}$.

    Args:
        generators (np.ndarray): Generator frequency of each encoding gate.
        etas (np.ndarray): Scaler of each encoding gate.

    Returns:
        np.ndarray: The sorted comb.
    """
    reach = {0.0}
    for generator, eta in zip(generators, etas):
        step = generator * eta
        reach = {round(a + s * step, 9) for a in reach for s in (-1.0, 0.0, 1.0)}

    return np.array(sorted(reach))


def _sweep_grid(
    eta_min: float, eta_max: float, points_per_period: int, mts: int, generator: float
) -> np.ndarray:
    """
    Sample the scaler axis fine enough to resolve the loss oscillations.

    Two frequencies separated by $\\Delta$ correlate over the sample window as
    a Dirichlet kernel whose sidelobes sit $1/mts$ apart. A scaler on a
    generator of size $\\gamma$ detunes its components at rate $\\gamma$, so the
    loss oscillates with period $1/(mts \\, \\gamma)$ along $\\eta$.

    Args:
        eta_min, eta_max (float): Bounds of the scaler axis.
        points_per_period (int): Samples per loss oscillation.
        mts (int): Domain oversampling of the dataset, which sets the
            Dirichlet sidelobe spacing.
        generator (float): Generator frequency the scaler acts on.

    Returns:
        np.ndarray: The scaler grid.
    """
    step = 1.0 / (points_per_period * mts * generator)
    return np.arange(eta_min, eta_max + 0.5 * step, step)


def _profile_mse(x: np.ndarray, y: np.ndarray, support: np.ndarray) -> float:
    """
    Loss of the best real Fourier fit on `support`, i.e. the loss with the
    coefficients concentrated out.

    This is the separable (variable projection) view of the problem: the
    coefficients enter linearly and are solved for in closed form, leaving a
    cost that depends on the frequencies alone. In classical spectral
    estimation this concentrated cost is what carries the sidelobe minima, and
    it is the quantity an optimizer would see if its linear parameters always
    caught up with the current frequencies.

    Being a relaxation, it is a lower bound on what the model can reach: it
    lets every component carry a free coefficient, which the variational
    parameters do not. It also steps up at the isolated scalers where two
    components coincide exactly, because the fit loses a direction there.

    Args:
        x (np.ndarray): Domain samples of shape (n_points,).
        y (np.ndarray): Target values of shape (n_points,).
        support (np.ndarray): Comb frequencies, signs allowed.

    Returns:
        float: Mean squared residual of the least-squares fit.
    """
    phases = np.unique(np.abs(support[support != 0.0]))[None, :] * x[:, None]
    design = np.concatenate(
        [np.ones((len(x), 1)), np.cos(phases), np.sin(phases)], axis=1
    )

    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.mean(residual**2))


def _pulse_sweep(
    model: Model,
    x: jnp.ndarray,
    y: np.ndarray,
    layer: int,
    qubit: int,
    slot: int,
    grid: np.ndarray,
    frozen: np.ndarray,
    chunk: int,
) -> np.ndarray:
    """
    Evaluate the loss over the scaler grid with the encoding gates at pulse
    level.

    The grid is pushed through the encoding pulse batch axis rather than a
    Python loop, so one call covers `chunk` scalers at once. Blocks are padded
    to a constant size to keep a single compiled shape.

    Args:
        model (Model): The QFM model.
        x (jnp.ndarray): Domain samples of shape (n_points, n_input_feat).
        y (np.ndarray): Target values of shape (n_points,).
        layer (int): Layer of the encoding gate whose scaler is swept.
        qubit (int): Qubit of that gate.
        slot (int): Pulse parameter slot holding the amplitude.
        grid (np.ndarray): Scaler values to evaluate.
        frozen (np.ndarray): Amplitude scalers of the remaining gates, shape
            (n_layers, n_qubits).
        chunk (int): Number of scalers per model call.

    Returns:
        np.ndarray: Mean squared error per grid point.
    """
    losses = np.empty(len(grid))
    # the model keeps whatever it is called with, so the batched scalers would
    # otherwise stay behind and break the next unbatched call
    saved = (model.enc_pulse_params, list(model.repeat_batch_axis))
    try:
        for start in track(
            range(0, len(grid), chunk),
            description=f"Sweeping layer {layer} qubit {qubit}..",
        ):
            block = grid[start : start + chunk]
            padded = np.full(chunk, block[-1])
            padded[: len(block)] = block

            enc_pulse_params = np.ones((chunk, *model._enc_pulse_shape))
            enc_pulse_params[..., slot] = frozen
            enc_pulse_params[:, layer, qubit, slot] = padded

            prediction = np.asarray(
                model(
                    params=model.params,
                    inputs=x,
                    execution_type="expval",
                    force_mean=True,
                    gate_mode="enc_pulse",
                    enc_pulse_params=jnp.array(enc_pulse_params),
                )
            ).reshape(len(y), chunk)

            losses[start : start + len(block)] = ((prediction - y[:, None]) ** 2).mean(
                axis=0
            )[: len(block)]
    finally:
        model.enc_pulse_params, model.repeat_batch_axis = saved

    return losses


def _log_curves(curves: Dict[str, np.ndarray]) -> None:
    """
    Log the sweep curves as per-step metric histories, batched.

    Args:
        curves (Dict[str, np.ndarray]): Metric name to values, indexed by the
            position on the scaler grid.
    """
    timestamp = int(time.time() * 1000)
    metrics = [
        Metric(key=key, value=float(value), timestamp=timestamp, step=step)
        for key, values in curves.items()
        for step, value in enumerate(values)
    ]

    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    for start in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[start : start + 1000])


def _profile_grid(
    model: Model,
    supports: List[np.ndarray],
    target_frequencies: jnp.ndarray,
    coefficients: jnp.ndarray,
    mts: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Dense domain grid for the concentrated fit, with the target re-evaluated
    on it.

    The concentrated fit assigns a free coefficient to every comb component,
    so it is only meaningful while the domain carries more samples than fit
    parameters. The training grid guarantees that for the nominal comb, but a
    detuned multi-layer comb inflates past it: the number of distinct
    components approaches $3^{n_\\text{gates}}$, and the swept lines exceed
    the Nyquist frequency of the grid. Both are cured by raising the sample
    density within the same window, i.e. `mfs`. The window itself, and with it
    the Dirichlet resolution that shapes the landscape, is set by `mts` and is
    deliberately left untouched.

    Args:
        model (Model): The QFM model.
        supports (List[np.ndarray]): The comb at every point of the sweep.
        target_frequencies (jnp.ndarray): Target frequencies of shape
            (n_frequencies, n_input_feat).
        coefficients (jnp.ndarray): Target coefficients of shape
            (n_frequencies,).
        mts (int): Domain oversampling of the dataset, i.e. the window length
            in periods.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: Domain samples, target values and
        the chosen sample density `mfs`.
    """
    degree = int(np.prod(model.degree))
    max_size = max(len(np.unique(np.abs(s[s != 0.0]))) for s in supports)
    max_frequency = max(float(np.max(np.abs(s))) for s in supports)

    mfs = max(
        1,
        math.ceil((2 * max_size + 1) / (mts * degree)),
        math.ceil(2 * max_frequency / degree),
    )

    x = Datasets.construct_domain_samples(model, mts=mts, mfs=mfs)
    y = np.asarray(Datasets.calculate_values(x, target_frequencies, coefficients))

    return np.asarray(x).ravel(), y, mfs


def _dirichlet_weight(delta: np.ndarray, mts: int, n_samples: int) -> np.ndarray:
    """
    Correlation of two unit sinusoids separated by `delta` over the sample
    window.

    Closed form of $|\\frac{1}{T} \\sum_t e^{i \\delta x_t}|$ for the uniform
    grid of $T$ samples covering $mts$ periods, i.e. the Dirichlet kernel with
    nulls at multiples of $1/mts$. A vanishing denominator means the
    separation is a multiple of the sampling rate, where the two sinusoids
    alias onto each other and the correlation returns to one.

    Args:
        delta (np.ndarray): Frequency separations.
        mts (int): Window length in periods.
        n_samples (int): Number of samples $T$ in the window.

    Returns:
        np.ndarray: Correlation magnitudes in $[0, 1]$.
    """
    numerator = np.sin(np.pi * mts * delta)
    denominator = n_samples * np.sin(np.pi * mts * delta / n_samples)
    aliased = np.abs(denominator) < 1e-12

    return np.where(
        aliased, 1.0, np.abs(numerator) / np.where(aliased, 1.0, np.abs(denominator))
    )


def _analytic_mse(
    supports: List[np.ndarray],
    omegas: np.ndarray,
    powers: np.ndarray,
    mts: int,
    n_samples: int,
) -> np.ndarray:
    """
    Closed-form approximation of the concentrated loss over the sweep.

    Treats every target component independently: a component of power $p$ at
    distance $\\delta$ from the nearest comb line retains the energy
    $p \\, (1 - |D(\\delta)|^2)$ in the residual, with $D$ the Dirichlet
    kernel of the sample window. Summing over components gives the classical
    multi-tone estimation cost, which matches the concentrated loss while the
    components stay separated by more than a kernel lobe and shares its
    geometry everywhere: oscillation period $1/(mts \\, \\gamma)$ along the
    scaler of a gate driving the generator $\\gamma$, main lobe of width
    $2/(mts \\, \\gamma)$ around the aligned scaler.

    Args:
        supports (List[np.ndarray]): The comb at every point of the sweep.
        omegas (np.ndarray): Target frequencies, duplicates merged.
        powers (np.ndarray): Power of each target component.
        mts (int): Window length in periods.
        n_samples (int): Number of samples in the window.

    Returns:
        np.ndarray: Approximate loss per sweep point.
    """
    values = np.empty(len(supports))
    for i, support in enumerate(supports):
        delta = np.min(np.abs(omegas[:, None] - support[None, :]), axis=1)
        weight = _dirichlet_weight(delta, mts, n_samples)
        values[i] = np.sum(powers * (1.0 - weight**2))

    return values


def _target_components(
    target_frequencies: jnp.ndarray, coefficients: jnp.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Target frequencies and their powers, with duplicated components merged.

    The generator off-grid mode can displace two model frequencies onto the
    same target component, whose coefficients then add coherently. Powers
    follow the normalization of `Datasets.calculate_values`.

    Args:
        target_frequencies (jnp.ndarray): Target frequencies of shape
            (n_frequencies, n_input_feat).
        coefficients (jnp.ndarray): Target coefficients of shape
            (n_frequencies,).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Distinct frequencies and their powers.
    """
    raw = np.round(np.asarray(target_frequencies).ravel(), 9)
    c = np.asarray(coefficients).ravel()

    omegas, inverse = np.unique(raw, return_inverse=True)
    merged = np.zeros(len(omegas), dtype=complex)
    np.add.at(merged, inverse, c)

    return omegas, np.abs(merged) ** 2 / c.size**2


def sweep_loss_landscape(
    model: Model,
    train_loader: DataLoader,
    target_etas: Optional[jnp.ndarray],
    coefficients: jnp.ndarray,
    target_frequencies: jnp.ndarray,
    mts: int,
    eta_min: float,
    eta_max: float,
    points_per_period: int,
    chunk: int,
) -> Dict:
    """
    Sweeps the scaler of each encoding gate and logs the resulting loss
    landscape, one curve per gate.

    Training the encoding scalers is a frequency estimation problem: the scaler
    $\\eta_j$ multiplies the generator $\\gamma_j$ of its gate, so the loss
    along $\\eta_j$ is the misfit between a detuned comb and the target. That
    misfit oscillates with period $1/(mts \\cdot \\gamma_j)$, which puts a local
    minimum on every sidelobe and shrinks the basin around the aligned scaler
    in proportion to the generator. Sweeping one gate at a time therefore
    resolves the landscape per spectral component, which is what the per-gate
    curves report. Since the largest generator grows with the size of the
    frequency spectrum for the binary and ternary strategies, and stays at one
    for hamming, the per-gate hardness ties directly to the number of
    frequencies the encoding provides.

    Three curves are logged per gate:

    - `profile`, the loss with the coefficients concentrated out, see
      :func:`_profile_mse`. This is the landscape of the frequencies alone and
      reaches zero wherever the comb covers the target support. It is
      evaluated on a sample grid dense enough for the detuned comb, see
      :func:`_profile_grid`. Gates that share a generator across layers make
      the detuned comb locally denser than the window resolution, and a
      cluster of sub-resolution lines spans every nearby sinusoid on the
      finite window. The concentrated fit then tracks the target along most
      of such a gate's slice, so for multi-layer models the alignment
      structure is carried by `analytic` and `fixed` instead.
    - `analytic`, the closed-form Dirichlet approximation of `profile`, see
      :func:`_analytic_mse`.
    - `fixed`, the loss at the current variational parameters on the training
      grid, i.e. the slice the optimizer sees before its coefficients adapt.

    The remaining gates are held at their target scalers, so each slice contains
    the aligned configuration and the path from the initial scaler $\\eta = 1$
    to it.

    Args:
        model (Model): The QFM model, which must use a hamming, binary or
            ternary encoding.
        train_loader (DataLoader): Loader holding the domain samples and the
            target series.
        target_etas (Optional[jnp.ndarray]): The scalers that align the comb
            with the target, produced by `offgrid_mode="generator"`.
        coefficients (jnp.ndarray): Coefficients of the target series.
        target_frequencies (jnp.ndarray): Frequencies of the target series.
        mts (int): Domain oversampling of the dataset, which sets the
            oscillation period along the scaler axis.
        eta_min, eta_max (float): Bounds of the scaler axis.
        points_per_period (int): Samples per loss oscillation.
        chunk (int): Number of scalers per model call.
    """
    if target_etas is None:
        raise ValueError(
            "The landscape sweep needs target_etas, which are only produced by "
            "offgrid_mode='generator'."
        )
    if model.n_input_feat != 1:
        raise ValueError(
            f"The landscape sweep supports a single input feature, got "
            f"{model.n_input_feat}."
        )

    domain = train_loader.dataset.tensors[0].numpy()
    y = train_loader.dataset.tensors[1].numpy()

    gates = _encoding_gates(model)
    generators = np.array([generator for _, _, generator in gates])
    # amplitude is the leading pulse parameter of the encoding gate and the
    # only one that scales the rotation angle, i.e. the generator
    slot = int(model._enc_pulse_offsets[0])
    frozen = np.asarray(target_etas)[0]
    aligned = np.array([frozen[layer, qubit] for layer, qubit, _ in gates])
    omegas, powers = _target_components(target_frequencies, coefficients)

    log.info(f"Encoding gates (layer, qubit, generator): {gates}")
    log.info(f"Target etas: {aligned.tolist()}")
    mlflow.log_param("landscape.mts", mts)
    mlflow.log_param("landscape.n_gates", len(gates))
    mlflow.log_param("landscape.n_frequencies", len(model.frequencies[0]))
    for layer, qubit, generator in gates:
        mlflow.log_param(f"landscape.generator.l{layer}.q{qubit}", generator)
        mlflow.log_param(
            f"landscape.target_eta.l{layer}.q{qubit}", float(frozen[layer, qubit])
        )

    _verify_landscape(model, domain, generators, frozen, aligned, slot)

    for j, (layer, qubit, generator) in enumerate(gates):
        grid = _sweep_grid(eta_min, eta_max, points_per_period, mts, generator)

        etas = np.tile(aligned, (len(grid), 1))
        etas[:, j] = grid
        supports = [_comb_support(generators, eta) for eta in etas]

        x_dense, y_dense, mfs = _profile_grid(
            model, supports, target_frequencies, coefficients, mts
        )
        log.info(
            f"Gate {j} (layer {layer}, qubit {qubit}): {len(grid)} scalers on "
            f"generator {generator}, profile mfs={mfs} ({len(x_dense)} samples)"
        )
        mlflow.log_param(f"landscape.profile_mfs.l{layer}.q{qubit}", mfs)

        profile = np.array(
            [_profile_mse(x_dense, y_dense, support) for support in supports]
        )
        analytic = _analytic_mse(supports, omegas, powers, mts, len(x_dense))
        fixed = _pulse_sweep(model, domain, y, layer, qubit, slot, grid, frozen, chunk)

        _log_curves(
            {
                f"landscape.eta.l{layer}.q{qubit}": grid,
                f"landscape.profile.l{layer}.q{qubit}": profile,
                f"landscape.analytic.l{layer}.q{qubit}": analytic,
                f"landscape.fixed.l{layer}.q{qubit}": fixed,
            }
        )

    return {}


def _verify_landscape(
    model: Model,
    domain: jnp.ndarray,
    generators: np.ndarray,
    frozen: np.ndarray,
    aligned: np.ndarray,
    slot: int,
) -> None:
    """
    Check the two assumptions the sweep rests on.

    The comb is enumerated from the encoding generators rather than read off
    the model, so at unit scalers it has to reproduce the model's own comb. And
    the scalers are swept at pulse level, which only stands in for a frequency
    scaling while the calibrated pulses reproduce their gates.

    Args:
        model (Model): The QFM model.
        domain (jnp.ndarray): Domain samples of shape (n_points, n_input_feat).
        generators (np.ndarray): Generator frequency of each encoding gate.
        frozen (np.ndarray): Target scalers of shape (n_layers, n_qubits).
        aligned (np.ndarray): The same scalers, per encoding gate.
        slot (int): Pulse parameter slot holding the amplitude.

    Raises:
        ValueError: If either check fails.
    """
    comb = _comb_support(generators, np.ones_like(generators))
    expected = np.asarray(model.frequencies[0], dtype=float)
    if not np.array_equal(comb, expected):
        raise ValueError(
            f"The comb enumerated from the encoding generators has "
            f"{len(comb)} components, the model reports {len(expected)}. The "
            "per-gate generators do not describe this encoding."
        )

    unitary = model(
        params=model.params,
        inputs=domain,
        execution_type="expval",
        force_mean=True,
        gate_mode="unitary",
        enc_params=jnp.array(frozen[..., None]),
    )
    model.enc_params = jnp.ones((*frozen.shape, 1))

    saved = (model.enc_pulse_params, list(model.repeat_batch_axis))
    try:
        enc_pulse_params = np.ones((1, *model._enc_pulse_shape))
        enc_pulse_params[..., slot] = frozen
        pulse = model(
            params=model.params,
            inputs=domain,
            execution_type="expval",
            force_mean=True,
            gate_mode="enc_pulse",
            enc_pulse_params=jnp.array(enc_pulse_params),
        )
    finally:
        model.enc_pulse_params, model.repeat_batch_axis = saved

    deviation = float(np.max(np.abs(np.asarray(pulse) - np.asarray(unitary))))
    log.info(f"Landscape checks: comb={len(comb)} components, pulse={deviation:.3e}")
    mlflow.log_metric("landscape.check.pulse", deviation)

    if deviation > 1e-3:
        raise ValueError(
            f"Pulse backend deviates from the unitary model by {deviation:.3e} "
            "at the target scalers, the pulses no longer reproduce their gates."
        )

def log_metrics(
    model,
    data,
    step,
    prefix="",
    gate_mode="unitary",
    noise_params=None,
    pulse_params=None,
    enc_pulse_params=None,
    enc_params=None,
):
    domain_samples = data.dataset.tensors[0].numpy()
    fourier_series = data.dataset.tensors[1].numpy()

    call_kwargs = dict(
        params=model.params,
        inputs=domain_samples,
        execution_type="expval",
        force_mean=True,
        gate_mode=gate_mode,
        pulse_params=pulse_params,
        enc_pulse_params=enc_pulse_params,
        noise_params=noise_params,
    )
    if enc_params is not None:
        call_kwargs["enc_params"] = enc_params
    prediction = model(**call_kwargs)

    # only the time-domain error is reported: once the target carries off-grid
    # frequencies, its coefficients and the model's live on different supports
    # and a coefficient-space comparison is not defined
    mlflow.log_metric(
        f"{prefix}_mse", Losses.mse(prediction, fourier_series).item(), step=step
    )


def _jacobian_rank(
    model: Model,
    theta: jnp.ndarray,
    lam: jnp.ndarray,
    eta: jnp.ndarray,
    gate_mode: str,
    argnums: Tuple[int, ...],
    tol_rel: float,
) -> Tuple[int, float, Tuple[int, ...]]:
    """Compute the numerical rank of the Jacobian of the Fourier coefficients
    of *model* w.r.t. the parameter groups indicated by *argnums*.

    The Fourier coefficients are stacked into a real vector
    ``[Re(c_\\omega), Im(c_\\omega)]`` so the resulting Jacobian is a real
    ``(2|\\Omega|, |params|)`` matrix from which a meaningful rank can be
    obtained via SVD.  ``tol_rel`` is multiplied with the largest
    singular value to obtain the cutoff used for the numerical rank
    estimate (matches ``numpy.linalg.matrix_rank``'s default policy).

    Args:
        model: Already-instantiated quantum Fourier model.
        theta: Unitary parameter vector ``\\theta`` (shape as in ``model.params``).
        lam: Ansatz pulse-scaling parameter vector ``\\lambda`` (shape as in
            ``model.pulse_params``).
        eta: Encoding pulse-scaling parameter vector ``\\eta`` (shape as in
            ``model.enc_pulse_params``).
        gate_mode: ``"unitary"``, ``"ansatz_pulse"``, ``"enc_pulse"`` or
            ``"all_pulse"``
        argnums: Subset of ``(0, 1, 2)`` indicating which arguments to
            differentiate w.r.t. — ``(0,)`` gives ``J_\\theta``, adding the
            argnums of the groups the mode runs at pulse level gives
            ``J_ext``.
        tol_rel: Relative tolerance for the numerical rank.

    Returns:
        Tuple ``(rank, sv_min_above_tol, jacobian_shape)`` — ``rank`` is
        the integer numerical rank, ``sv_min_above_tol`` is the smallest
        singular value above the cutoff (or ``0.0`` when the matrix is
        zero) and ``jacobian_shape`` records the flattened Jacobian
        shape for diagnostics.
    """
    groups = _PULSE_GROUPS[gate_mode]

    def _coeff_vec(theta_, lam_, eta_):
        coeff_kwargs = dict(
            params=theta_,
            gate_mode=gate_mode,
            shift=False,
            trim=False,
            numerical_cap=-1,
            force_mean=True,
            execution_type="expval",
        )
        # The model rejects a scaler group that its gate_mode does not run at
        # pulse level, and such a group is inert on the coefficients anyway,
        # so only the groups the mode owns are passed through.
        if "pulse" in groups:
            coeff_kwargs["pulse_params"] = lam_
        if "enc_pulse" in groups:
            coeff_kwargs["enc_pulse_params"] = eta_
        coeffs, _ = Coefficients.get_spectrum(model, **coeff_kwargs)
        # Stack real and imaginary parts so SVD gives a real-valued rank.
        return jnp.concatenate([coeffs.real.ravel(), coeffs.imag.ravel()])

    # The model stores whatever params it is called with on itself, so under
    # jacrev it ends up holding tracers that are dead once the transform
    # returns. Restore the concrete values, otherwise a later call that falls
    # back to a model attribute raises UnexpectedTracerError.
    saved = {
        name: getattr(model, name)
        for name in ("params", "pulse_params", "enc_pulse_params")
    }
    try:
        jac = jax.jacrev(_coeff_vec, argnums=argnums)(theta, lam, eta)
    finally:
        for name, value in saved.items():
            setattr(model, name, value)

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
    eta: jnp.ndarray,
    gate_mode: str,
    tol_rel: float,
    step: int,
) -> None:
    """Compute ``rank J_\\theta``, ``rank J_ext`` and ``\\Delta r`` and log to MLflow.

    A non-zero ``\\Delta r = rank J_ext - rank J_\\theta`` certifies that
    the pulse-scaling parameters provide new search directions in
    Fourier-coefficient space beyond what the unitary parameters alone
    can reach.

    Args:
        model: The model whose autodiff is exercised.
        theta: Current unitary parameters.
        lam: Current ansatz pulse-scaling parameters ``\\lambda``.
        eta: Current encoding pulse-scaling parameters ``\\eta``.
        gate_mode: ``"unitary"``, ``"ansatz_pulse"``, ``"enc_pulse"`` or
            ``"all_pulse"`` — the regime in which ranks are evaluated.
            ``J_ext`` extends ``J_\\theta`` by exactly the scaler groups the
            mode runs at pulse level, so ``\\Delta r`` is reported for every
            mode except ``"unitary"``, which has no such group.
        tol_rel: Relative SVD cutoff used for the numerical rank.
        when: Tag for the metric name (``"init"``/``"trained"``).
        step: MLflow step coordinate.
    """
    log.info(f"Computing Jacobian ranks (gate_mode={gate_mode}) ...")
    # _jacobian_rank restores the model's parameter attributes itself
    r_theta, sv_theta, shape_theta = _jacobian_rank(
        model, theta, lam, eta, gate_mode, argnums=(0,), tol_rel=tol_rel
    )
    mlflow.log_metric(f"rank.r_theta", r_theta, step=step)
    mlflow.log_metric(f"rank.sv_theta", sv_theta, step=step)

    # extend J_\theta by the scaler groups this mode actually runs at
    # pulse level, i.e. \\lambda, \\eta or both
    extra_argnums = tuple(_GROUP_ARGNUM[group] for group in _PULSE_GROUPS[gate_mode])
    if not extra_argnums:
        # "unitary" has no pulse scalers, so J_ext would equal J_\theta.
        log.info(f"  J_\\theta shape={shape_theta} rank={r_theta}")
        return

    r_ext, sv_ext, shape_ext = _jacobian_rank(
        model, theta, lam, eta, gate_mode, argnums=(0,) + extra_argnums,
        tol_rel=tol_rel,
    )
    delta_r = r_ext - r_theta
    log.info(
        f"  J_\\theta shape={shape_theta} rank={r_theta} | "
        f"J_ext shape={shape_ext} rank={r_ext} | \\Delta r={delta_r}"
    )
    mlflow.log_metric(f"rank.r_ext", r_ext, step=step)
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
    gate_mode: str = "unitary",
    pulse_learning_rate: Optional[float] = None,
    rank_eval_enabled: bool = False,
    rank_eval_tol_rel: float = 1e-8,
    rank_report_interval: int = 100,
    target_etas: Optional[jnp.ndarray] = None,
    enc_pulse_init: str = "ones",
    train_enc_params: bool = False,
) -> None:
    if gate_mode not in _PULSE_GROUPS:
        raise ValueError(
            f"Unknown gate_mode: {gate_mode}. Use one of {list(_PULSE_GROUPS)}."
        )

    # trainable pulse scaler groups, each starting at ones
    pulse_groups = {
        group: jnp.ones_like(getattr(model, f"{group}_params"))
        for group in _PULSE_GROUPS[gate_mode]
    }

    # oracle-initialize the encoding amplitude scalers at the target etas
    if enc_pulse_init == "target" and "enc_pulse" in pulse_groups:
        if target_etas is None:
            raise ValueError(
                "enc_pulse_init='target' requires target_etas, which are only "
                "produced by offgrid_mode='generator'."
            )
        eta0 = pulse_groups["enc_pulse"]
        te = jnp.asarray(target_etas)  # (n_input_feat, n_layers, n_qubits)
        for idx, off in enumerate(model._enc_pulse_offsets):
            eta0 = eta0.at[0, :, :, off].set(te[idx])
        pulse_groups["enc_pulse"] = eta0

    extra_groups = dict(pulse_groups)
    if train_enc_params:
        extra_groups["enc"] = jnp.ones_like(model.enc_params)

    params = {"unitary": model.params, **extra_groups}

    # set a per-group optimizer
    pulse_lr = (
        pulse_learning_rate if pulse_learning_rate is not None else learning_rate * 0.1
    )
    log.info(
        f"Learning rates - unitary: {learning_rate}, "
        f"pulse: {pulse_lr if extra_groups else 'N/A (not training pulse params)'}"
    )

    if extra_groups:
        # separate Adam per parameter group so the pulse / enc scalers can use
        # their own learning rate (pulse_learning_rate; defaults to
        # learning_rate*0.1 when unset, though the study config sets it equal)
        transforms = {"unitary": optax.adam(learning_rate)}
        transforms.update({k: optax.adam(pulse_lr) for k in extra_groups})

        # Combine into a single optimizer keyed by the param labels
        label_fn = lambda params: {k: k for k in params}  # noqa: E731
        opt = optax.multi_transform(transforms, label_fn)
    else:
        opt = optax.adam(learning_rate)

    opt_state = opt.init(params)

    try:
        loss_functions = [getattr(Losses, loss) for loss in loss_functions]
    except AttributeError:
        log.error(f"Loss function is not valid. {loss_functions} must be in {Losses}")
        raise

    log.info(f"Using gate mode: {gate_mode} for training")
    for group in pulse_groups:
        log.info(f"{group}_params are trainable (pulse_lr={pulse_lr})")

    def cost(params_dict, targets, **kwargs):
        # a group is absent (-> None) exactly when it is not trained; passing
        # None keeps the model on its own params and is valid in every mode
        call_kwargs = dict(
            params=params_dict["unitary"],
            pulse_params=params_dict.get("pulse"),
            enc_pulse_params=params_dict.get("enc_pulse"),
        )
        # enc_params only when trained, to avoid the "enc_params is None" warning
        if "enc" in params_dict:
            call_kwargs["enc_params"] = params_dict["enc"]
        predictions = model(**call_kwargs, **kwargs)

        total_loss = jnp.array(0.0)
        for ls, lf in zip(loss_scalers, loss_functions):
            total_loss = total_loss + ls * lf(predictions, targets)
        return total_loss

    for step in track(range(steps), description="Training..", total=steps):
        if rank_eval_enabled and step % rank_report_interval == 0:
            _log_jacobian_ranks(
                model,
                theta=params["unitary"],
                lam=params.get("pulse", jnp.ones_like(model.pulse_params)),
                eta=params.get("enc_pulse", jnp.ones_like(model.enc_pulse_params)),
                gate_mode=gate_mode,
                tol_rel=rank_eval_tol_rel,
                step=step,
            )

        for domain_samples, fourier_samples in train_loader:
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
        for group in extra_groups:
            setattr(model, _GROUP_ATTR[group], params[group])
            mlflow.log_metric(
                f"{group}_scaler_mean", float(jnp.mean(params[group])), step=step
            )
            mlflow.log_metric(
                f"{group}_scaler_std", float(jnp.std(params[group])), step=step
            )

        log_metrics(
            model,
            data=train_loader,
            step=step,
            prefix="train",
            gate_mode=gate_mode,
            noise_params=noise_params,
            pulse_params=params.get("pulse"),
            enc_pulse_params=params.get("enc_pulse"),
            enc_params=params.get("enc"),
        )
        
    # final reporting
    if rank_eval_enabled:
        _log_jacobian_ranks(
            model,
            theta=params["unitary"],
            lam=params.get("pulse", jnp.ones_like(model.pulse_params)),
            eta=params.get("enc_pulse", jnp.ones_like(model.enc_pulse_params)),
            gate_mode=gate_mode,
            tol_rel=rank_eval_tol_rel,
            step=steps, # the last step
        )

    return {
        # "model": model,
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
        gate_mode="ansatz_pulse",
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
        "fidelity": fi,
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
