"""The loss landscape along an encoding scaler, one curve per gate.

Training the encoding scalers is a frequency estimation problem: the scaler
of a gate multiplies its generator, so the loss along it is the misfit
between a detuned comb and the target. That misfit oscillates with a period
set by the generator and the window length, which puts a local minimum on
every sidelobe and shrinks the basin around the aligned scaler in proportion
to the generator. Sweeping one gate at a time resolves that per spectral
component, which is what the per-gate curves report.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from fluksio import Port, node
from qml_essentials.coefficients import Datasets
from qml_essentials.model import Model
from rich.progress import track

from pulse_level_qfms.data import load_dataset
from pulse_level_qfms.model import load_model

log = logging.getLogger(__name__)


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


def _trained_sweeps(
    model: Model,
    x: jnp.ndarray,
    y: np.ndarray,
    slot: int,
    frozen: np.ndarray,
    gates: List[Tuple[int, int, float]],
    grids: Dict[str, np.ndarray],
    gate_mode: str,
    fit_steps: int,
    fit_starts: int,
    steps: int,
    rounds: int,
    learning_rate: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    """
    Evaluate the loss over the scaler grid of every gate with the ansatz
    trained along the sweep.

    The ansatz is first fitted at the target scalers for `fit_steps` Adam
    steps, from the parameters the fixed slice holds and from `fit_starts`
    random draws of $\\theta$, keeping the best fit: $\\theta$ under
    ``gate_mode="enc_pulse"``, $\\theta$ and the trainable-gate pulse scalers
    $\\kappa$ (from ones) under ``"all_pulse"``. Each slice then continues from
    that fit outward from the target scaler to both ends of its grid and back,
    every point starting from its neighbour's parameters and taking `steps`
    steps with a fresh optimizer.

    A local fit is path dependent: moving the scalers can lead the parameters
    out of a poor optimum, and a slice that keeps improving along the way
    records the optimizer's progress rather than the landscape. So the best
    chain returning to the target is fitted there once more, and while that
    lowers the target loss by more than a percent the sweep is repeated from
    it, for at most `rounds` rounds. Every point keeps the lowest loss any pass
    reached. Without any steps this is the fixed slice.

    Args:
        model (Model): The QFM model.
        x (jnp.ndarray): Domain samples of shape (n_points, n_input_feat).
        y (np.ndarray): Target values of shape (n_points,).
        slot (int): Pulse parameter slot holding the amplitude.
        frozen (np.ndarray): Target scalers of shape (n_layers, n_qubits).
        gates (List[Tuple[int, int, float]]): Layer, qubit and generator of
            each encoding gate.
        grids (Dict[str, np.ndarray]): Scaler grid of each gate, by gate key.
        gate_mode (str): ``"enc_pulse"`` or ``"all_pulse"``.
        fit_steps (int): Adam steps of a fit at the target scalers.
        fit_starts (int): Random draws of $\\theta$ fitted besides the initial one.
        steps (int): Adam steps per point of the continuation.
        rounds (int): Most sweeps from an improved target fit.
        learning_rate (float): Adam learning rate, for $\\theta$ and $\\kappa$.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]: Mean squared
        error per grid point of every gate, and what the fits reached: the best
        start's loss after every tenth of its steps (`fit_loss`), every start's
        final loss (`start_loss`) and the target loss before the first and after
        every round (`target_loss`).
    """
    if gate_mode not in ("enc_pulse", "all_pulse"):
        raise ValueError(
            f"The trained slice runs the encoding at pulse level, so gate_mode "
            f"has to be 'enc_pulse' or 'all_pulse', got {gate_mode!r}."
        )
    y = jnp.asarray(y)
    theta = jnp.asarray(model.params)
    low, high = model._initialization_domain
    starts = [theta] + [
        jax.random.uniform(key, theta.shape, minval=low, maxval=high)
        for key in jax.random.split(model.random_key, fit_starts)
    ]
    extra = {}
    if gate_mode == "all_pulse":
        extra["kappa"] = jnp.ones_like(model.pulse_params)

    def loss(params, enc_pulse_params):
        prediction = model(
            params=params["theta"],
            pulse_params=params.get("kappa"),
            inputs=x,
            execution_type="expval",
            force_mean=True,
            gate_mode=gate_mode,
            enc_pulse_params=enc_pulse_params,
        )
        return jnp.mean((prediction.ravel() - y) ** 2)

    optimizer = optax.adam(learning_rate)

    @jax.jit
    def step(params, state, enc_pulse_params):
        grads = jax.grad(loss)(params, enc_pulse_params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    evaluate = jax.jit(loss)

    def train(params, enc_pulse_params, n, marks=()):
        state, curve = optimizer.init(params), []
        for k in range(n + 1):
            if k in marks:
                curve.append(float(evaluate(params, enc_pulse_params)))
            if k < n:
                params, state = step(params, state, enc_pulse_params)
        return params, float(evaluate(params, enc_pulse_params)), curve

    def scalers(layer: int, qubit: int, eta: float) -> jnp.ndarray:
        enc_pulse_params = np.ones((1, *model._enc_pulse_shape))
        enc_pulse_params[..., slot] = frozen
        enc_pulse_params[0, layer, qubit, slot] = eta
        return jnp.array(enc_pulse_params)

    # the model keeps whatever it is called with, tracers included
    saved = {
        name: getattr(model, name)
        for name in ("params", "pulse_params", "enc_pulse_params")
    }
    try:
        target = scalers(*gates[0][:2], frozen[gates[0][0], gates[0][1]])
        marks = np.unique(np.linspace(0, fit_steps, 11).astype(int))
        fits = [
            train({"theta": start, **extra}, target, fit_steps, marks)
            for start in track(starts, description="Fitting at the target..")
        ]
        best, best_loss, fit_loss = min(fits, key=lambda fit: fit[1])
        record = {
            "fit_loss": fit_loss,
            "start_loss": [fit[1] for fit in fits],
            "target_loss": [best_loss],
        }

        curves = {key: np.full(len(grid), np.inf) for key, grid in grids.items()}
        for _ in range(rounds):
            returned = []
            for layer, qubit, _ in gates:
                key = f"l{layer}.q{qubit}"
                grid = grids[key]
                center = int(np.abs(grid - frozen[layer, qubit]).argmin())
                for out in (range(center, len(grid)), range(center, -1, -1)):
                    point, value = best, best_loss
                    for i in track(
                        [*out, *reversed(out)],
                        description=f"Continuing along layer {layer} qubit {qubit}..",
                    ):
                        enc_pulse_params = scalers(layer, qubit, grid[i])
                        point, value, _ = train(point, enc_pulse_params, steps)
                        curves[key][i] = min(curves[key][i], value)
                    returned.append((point, value))
            back = min(returned, key=lambda chain: chain[1])[0]
            point, value, _ = train(back, target, fit_steps)
            improved = value < 0.99 * best_loss
            if improved:
                best, best_loss = point, value
            record["target_loss"].append(best_loss)
            if not improved:
                break
    finally:
        for name, value in saved.items():
            setattr(model, name, value)

    return curves, record


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


def _verify_landscape(
    model: Model,
    domain: jnp.ndarray,
    generators: np.ndarray,
    frozen: np.ndarray,
    aligned: np.ndarray,
    slot: int,
) -> float:
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

    Returns:
        float: How far the pulse-level circuit deviates from the unitary one
        at the target scalers.

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

    if deviation > 1e-3:
        raise ValueError(
            f"Pulse backend deviates from the unitary model by {deviation:.3e} "
            "at the target scalers, the pulses no longer reproduce their gates."
        )

    return deviation


def sweep_landscape(
    model: Model,
    domain: np.ndarray,
    y: np.ndarray,
    target_etas: Optional[np.ndarray],
    coefficients: np.ndarray,
    target_frequencies: np.ndarray,
    mts: int,
    eta_min: float,
    eta_max: float,
    points_per_period: int,
    chunk: int,
    steps: int = 0,
    fit_steps: int = 500,
    fit_starts: int = 4,
    rounds: int = 3,
    learning_rate: float = 1e-2,
    gate_mode: str = "enc_pulse",
    eta_window: float = 0.0,
) -> Tuple[Dict, Dict[str, Dict[str, np.ndarray]]]:
    """Sweep the scaler of every encoding gate and report the three curves,
    four once the ansatz is trained along the sweep.

    Curves per gate:

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
    - `trained`, with ``steps > 0``: the same loss with the ansatz trained
      along the sweep, see :func:`_trained_sweeps`. It lies between `fixed`
      and `profile` as far as the ansatz can realize the free coefficients.

    The remaining gates are held at their target scalers, so each slice
    contains the aligned configuration and the path from the initial scaler
    ``eta = 1`` to it.

    Args:
        model (Model): The QFM model, which must use a hamming, binary or
            ternary encoding.
        domain (np.ndarray): Domain samples of the training grid.
        y (np.ndarray): Target values on that grid.
        target_etas (Optional[np.ndarray]): The scalers that align the comb
            with the target, produced by ``offgrid_mode="generator"``.
        coefficients (np.ndarray): Coefficients of the target series.
        target_frequencies (np.ndarray): Frequencies of the target series.
        mts (int): Domain oversampling of the dataset, which sets the
            oscillation period along the scaler axis.
        eta_min, eta_max (float): Bounds of the scaler axis.
        points_per_period (int): Samples per loss oscillation.
        chunk (int): Number of scalers per model call.
        steps (int, optional): Adam steps per scaler for the `trained` curve,
            0 leaves it out. Defaults to 0.
        fit_steps (int, optional): Adam steps of its fits at the target
            scalers. Defaults to 500.
        fit_starts (int, optional): Random starts of that fit besides the
            initial parameters. Defaults to 4.
        rounds (int, optional): Most sweeps from an improved target fit.
            Defaults to 3.
        learning_rate (float, optional): Its learning rate. Defaults to 1e-2.
        gate_mode (str, optional): What it trains, ``"enc_pulse"`` for
            $\\theta$ or ``"all_pulse"`` for $\\theta$ and $\\kappa$. Defaults to
            ``"enc_pulse"``.
        eta_window (float, optional): Oscillation periods kept on either side
            of the stretch between ``eta = 1`` and the target scaler, 0 keeps
            the whole axis. Defaults to 0.

    Returns:
        Tuple[Dict, Dict]: What the sweep is of -- gates, generators, target
        scalers and the backend check -- and the curves themselves, keyed by
        curve name and then by gate.
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

    info = {
        "mts": mts,
        "n_gates": len(gates),
        "n_frequencies": len(model.frequencies[0]),
        "eta_min": eta_min,
        "eta_max": eta_max,
        "points_per_period": points_per_period,
        "chunk": chunk,
        "steps": steps,
        "fit_steps": fit_steps,
        "fit_starts": fit_starts,
        "rounds": rounds,
        "learning_rate": learning_rate,
        "gate_mode": gate_mode,
        "eta_window": eta_window,
        "check_pulse": _verify_landscape(
            model, domain, generators, frozen, aligned, slot
        ),
        "gates": [],
    }
    # keyed by gate rather than dotted into the record: a key holding a dot is
    # not addressable as an export path
    curves: Dict[str, Dict[str, np.ndarray]] = {
        name: {} for name in ("eta", "profile", "analytic", "fixed")
    }

    for j, (layer, qubit, generator) in enumerate(gates):
        key = f"l{layer}.q{qubit}"
        grid = _sweep_grid(eta_min, eta_max, points_per_period, mts, generator)
        if eta_window:
            # all that basin width and minima density read: the stretch from
            # the calibrated to the aligned scaler and the lobes around both,
            # with a tolerance that keeps a bound landing on the grid
            margin = eta_window / (mts * generator) + 1e-9
            lo, hi = sorted((1.0, aligned[j]))
            grid = grid[(grid >= lo - margin) & (grid <= hi + margin)]

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

        info["gates"].append(
            {
                "key": key,
                "layer": layer,
                "qubit": qubit,
                "generator": generator,
                "target_eta": float(frozen[layer, qubit]),
                "profile_mfs": mfs,
            }
        )
        curves["eta"][key] = grid
        curves["profile"][key] = np.array(
            [_profile_mse(x_dense, y_dense, support) for support in supports]
        )
        curves["analytic"][key] = _analytic_mse(
            supports, omegas, powers, mts, len(x_dense)
        )
        curves["fixed"][key] = _pulse_sweep(
            model, domain, y, layer, qubit, slot, grid, frozen, chunk
        )

    if steps:
        curves["trained"], record = _trained_sweeps(
            model,
            domain,
            y,
            slot,
            frozen,
            gates,
            curves["eta"],
            gate_mode,
            fit_steps,
            fit_starts,
            steps,
            rounds,
            learning_rate,
        )
        info.update(record)

    return info, curves


def as_series(values: Dict[str, np.ndarray], grids: Dict[str, np.ndarray]) -> Dict:
    """One line per gate, plotted against the scaler rather than the index.

    A series carries its own x, which is what keeps the curves of two gates
    comparable: their grids have different spacings, because the oscillation
    period follows the generator.
    """
    return {
        "lines": [
            {
                "label": key,
                "points": [
                    [float(x), float(y)]
                    for x, y in zip(grids[key], curve, strict=True)
                ],
            }
            for key, curve in values.items()
        ]
    }


@node(
    requires=[
        Port("model", "artifact"),
        Port("model_spec", "json"),
        Port("dataset", "artifact"),
        Port("dataset_info", "json"),
        Port("mts", "int"),
        Port("eta_min", "float"),
        Port("eta_max", "float"),
        Port("points_per_period", "int"),
        Port("chunk", "int"),
        Port("steps", "int"),
        Port("fit_steps", "int"),
        Port("fit_starts", "int"),
        Port("rounds", "int"),
        Port("learning_rate", "float"),
        Port("gate_mode", "str"),
        Port("eta_window", "float"),
    ],
    provides=[
        Port("landscape", "json"),
        Port("profile", "series"),
        Port("analytic", "series"),
        Port("fixed", "series"),
        Port("trained", "series"),
    ],
    # One model call per chunk of scalers, per gate, and the progress bar it
    # prints is not something the watchdog can see. Training along the sweep
    # with pulse-level trainable gates takes hours per run.
    timeout=172800,
    cache=False,
)
def sweep_loss_landscape(
    *,
    model: Dict,
    model_spec: Dict,
    dataset: Dict,
    dataset_info: Dict,
    mts: int,
    eta_min: float,
    eta_max: float,
    points_per_period: int,
    chunk: int,
    steps: int,
    fit_steps: int,
    fit_starts: int,
    rounds: int,
    learning_rate: float,
    gate_mode: str,
    eta_window: float,
) -> Dict:
    """The per-gate loss landscape of one model against its target series."""
    circuit = load_model(model, model_spec)
    arrays = load_dataset(dataset)

    info, curves = sweep_landscape(
        circuit,
        domain=arrays["domain_samples"],
        y=arrays["fourier_samples"],
        target_etas=arrays.get("target_etas"),
        coefficients=arrays["coefficients"],
        target_frequencies=arrays["target_frequencies"],
        mts=mts,
        eta_min=eta_min,
        eta_max=eta_max,
        points_per_period=points_per_period,
        chunk=chunk,
        steps=steps,
        fit_steps=fit_steps,
        fit_starts=fit_starts,
        rounds=rounds,
        learning_rate=learning_rate,
        gate_mode=gate_mode,
        eta_window=eta_window,
    )

    grids = curves["eta"]
    return {
        "landscape": info,
        "profile": as_series(curves["profile"], grids),
        "analytic": as_series(curves["analytic"], grids),
        "fixed": as_series(curves["fixed"], grids),
        "trained": as_series(curves.get("trained", {}), grids),
    }
