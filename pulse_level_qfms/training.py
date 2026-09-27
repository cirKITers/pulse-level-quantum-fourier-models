"""Fitting a model to its target series, and what the fit reports on the way.

The optimisation is a generator: every step publishes its loss and its scaler
statistics the moment they exist, so a run's curve is visible while it is
still being drawn, and the node that wraps it does nothing but move artifacts
in and out.
"""

import logging
from typing import Dict, Generator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from fluksio import Port, node
from qml_essentials.coefficients import Coefficients
from qml_essentials.model import Model
from rich.progress import track

from pulse_level_qfms.data import batches, load_dataset
from pulse_level_qfms.model import (
    GROUP_ARGNUM,
    GROUP_ATTR,
    PULSE_GROUPS,
    load_model,
    save_model,
)
from pulse_level_qfms.utils import Losses

log = logging.getLogger(__name__)

#: The streamed statistics of each trainable scaler group.
SCALER_PORTS = tuple(
    f"{group}_scaler_{stat}"
    for group in ("pulse", "enc_pulse", "enc")
    for stat in ("mean", "std")
)


def jsonable(value):
    """A record a `json` port will accept: no NaN, no infinity.

    A subset with nothing in it produces one, and a port refuses it rather
    than storing a number that is not one. ``None`` says the same thing and
    travels.
    """
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (int, float, np.floating, np.integer)):
        number = float(value)
        return None if not np.isfinite(number) else number
    return value


def train_mse(
    model: Model,
    x: jnp.ndarray,
    y: jnp.ndarray,
    gate_mode: str = "unitary",
    noise_params: Optional[Dict] = None,
    pulse_params: Optional[jnp.ndarray] = None,
    enc_pulse_params: Optional[jnp.ndarray] = None,
    enc_params: Optional[jnp.ndarray] = None,
) -> float:
    """The time-domain error of the model on the training grid.

    Only the time domain is reported: once the target carries off-grid
    frequencies, its coefficients and the model's live on different supports
    and a coefficient-space comparison is not defined.
    """
    call_kwargs = dict(
        params=model.params,
        inputs=x,
        execution_type="expval",
        force_mean=True,
        gate_mode=gate_mode,
        pulse_params=pulse_params,
        enc_pulse_params=enc_pulse_params,
        noise_params=noise_params,
    )
    if enc_params is not None:
        call_kwargs["enc_params"] = enc_params

    return float(Losses.mse(model(**call_kwargs), y))


def jacobian_rank(
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
    groups = PULSE_GROUPS[gate_mode]

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
    above = jnp.min(jnp.where(s > cutoff, s, jnp.inf))
    sv_min_above = float(above) if rank > 0 else 0.0
    return rank, sv_min_above, tuple(int(d) for d in J.shape)


def jacobian_ranks(
    model: Model,
    theta: jnp.ndarray,
    lam: jnp.ndarray,
    eta: jnp.ndarray,
    gate_mode: str,
    tol_rel: float,
) -> Dict[str, float]:
    """``rank J_theta`` and, where the mode has pulse scalers, ``rank J_ext``.

    A non-zero ``delta r = rank J_ext - rank J_theta`` certifies that the
    pulse-scaling parameters provide new search directions in
    Fourier-coefficient space beyond what the unitary parameters alone can
    reach.

    Args:
        model: The model whose autodiff is exercised.
        theta: Current unitary parameters.
        lam: Current ansatz pulse-scaling parameters.
        eta: Current encoding pulse-scaling parameters.
        gate_mode: The regime in which ranks are evaluated. ``J_ext`` extends
            ``J_theta`` by exactly the scaler groups the mode runs at pulse
            level, so it is reported for every mode except ``"unitary"``,
            which has no such group.
        tol_rel: Relative SVD cutoff used for the numerical rank.

    Returns:
        Dict[str, float]: ``r_theta`` and ``sv_theta``, plus ``r_ext`` and
        ``sv_ext`` once the mode runs a scaler group at pulse level.
    """
    log.info(f"Computing Jacobian ranks (gate_mode={gate_mode}) ...")
    # jacobian_rank restores the model's parameter attributes itself
    r_theta, sv_theta, shape_theta = jacobian_rank(
        model, theta, lam, eta, gate_mode, argnums=(0,), tol_rel=tol_rel
    )
    ranks = {"r_theta": r_theta, "sv_theta": sv_theta}

    # extend J_theta by the scaler groups this mode actually runs at pulse
    # level, i.e. lambda, eta or both
    extra_argnums = tuple(GROUP_ARGNUM[group] for group in PULSE_GROUPS[gate_mode])
    if not extra_argnums:
        # "unitary" has no pulse scalers, so J_ext would equal J_theta.
        log.info(f"  J_theta shape={shape_theta} rank={r_theta}")
        return ranks

    r_ext, sv_ext, shape_ext = jacobian_rank(
        model,
        theta,
        lam,
        eta,
        gate_mode,
        argnums=(0,) + extra_argnums,
        tol_rel=tol_rel,
    )
    log.info(
        f"  J_theta shape={shape_theta} rank={r_theta} | "
        f"J_ext shape={shape_ext} rank={r_ext} | delta r={r_ext - r_theta}"
    )
    return {**ranks, "r_ext": r_ext, "sv_ext": sv_ext}


def fit(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    *,
    noise_params: Dict,
    loss_functions: List[str],
    loss_scalers: List[float],
    steps: int,
    learning_rate: float,
    gate_mode: str = "unitary",
    pulse_learning_rate: Optional[float] = None,
    rank_eval: bool = False,
    rank_tol_rel: float = 1e-8,
    rank_report_interval: int = 100,
    target_etas: Optional[np.ndarray] = None,
    enc_pulse_init: str = "ones",
    train_enc_params: bool = False,
    batch_size: int = -1,
) -> Generator[Dict[str, float], None, Dict]:
    """Fit the model, yielding what each step measured.

    Yields one record per step, and one more whenever the Jacobian ranks are
    evaluated. A value that is not finite is left out of its record rather
    than published as one: a port refuses it, and a gap in a curve says more
    than a number that is not one. Returns the final state of everything it
    streamed.
    """
    if gate_mode not in PULSE_GROUPS:
        raise ValueError(
            f"Unknown gate_mode: {gate_mode}. Use one of {list(PULSE_GROUPS)}."
        )

    # trainable pulse scaler groups, each starting at ones
    pulse_groups = {
        group: jnp.ones_like(getattr(model, f"{group}_params"))
        for group in PULSE_GROUPS[gate_mode]
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
        opt = optax.multi_transform(transforms, lambda p: {k: k for k in p})
    else:
        opt = optax.adam(learning_rate)

    opt_state = opt.init(params)

    try:
        losses = [getattr(Losses, loss) for loss in loss_functions]
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
        for scaler, loss in zip(loss_scalers, losses, strict=True):
            total_loss = total_loss + scaler * loss(predictions, targets)
        return total_loss

    def ranks_at(step: int) -> Dict[str, float]:
        return {
            "rank_step": step,
            **{
                f"rank_{name}": value
                for name, value in jacobian_ranks(
                    model,
                    theta=params["unitary"],
                    lam=params.get("pulse", jnp.ones_like(model.pulse_params)),
                    eta=params.get(
                        "enc_pulse", jnp.ones_like(model.enc_pulse_params)
                    ),
                    gate_mode=gate_mode,
                    tol_rel=rank_tol_rel,
                ).items()
            },
        }

    skipped = 0
    scalers: Dict[str, Dict[str, float]] = {}
    ranks: Dict[str, float] = {}
    mse = float("nan")

    for step in track(range(steps), description="Training..", total=steps):
        if rank_eval and step % rank_report_interval == 0:
            ranks = ranks_at(step)
            yield ranks

        for domain_samples, fourier_samples in batches(x, y, batch_size):
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
        record: Dict[str, float] = {}
        for group in extra_groups:
            setattr(model, GROUP_ATTR[group], params[group])
            scalers[group] = {
                "mean": float(jnp.mean(params[group])),
                "std": float(jnp.std(params[group])),
            }
            record[f"{group}_scaler_mean"] = scalers[group]["mean"]
            record[f"{group}_scaler_std"] = scalers[group]["std"]

        mse = train_mse(
            model,
            jnp.asarray(x),
            jnp.asarray(y),
            gate_mode=gate_mode,
            noise_params=noise_params,
            pulse_params=params.get("pulse"),
            enc_pulse_params=params.get("enc_pulse"),
            enc_params=params.get("enc"),
        )
        record["train_mse"] = mse

        published = {k: v for k, v in record.items() if np.isfinite(v)}
        skipped += len(record) - len(published)
        if published:
            yield published

    if rank_eval:
        ranks = ranks_at(steps)  # the last step
        yield ranks

    return {
        "gate_mode": gate_mode,
        "steps": steps,
        "train_mse": mse,
        "scalers": scalers,
        "rank": {k[len("rank_") :]: v for k, v in ranks.items() if k != "rank_step"},
        "skipped_nonfinite": skipped,
    }


@node(
    requires=[
        Port("model", "artifact"),
        Port("model_spec", "json"),
        Port("dataset", "artifact"),
        Port("dataset_info", "json"),
        Port("noise_params", "json"),
        Port("loss_functions", "list", item="str"),
        Port("loss_scalers", "list", item="float"),
        Port("steps", "int"),
        Port("learning_rate", "float"),
        Port("gate_mode", "str"),
        Port("pulse_learning_rate", "float"),
        Port("rank_eval", "bool"),
        Port("rank_tol_rel", "float"),
        Port("rank_report_interval", "int"),
        Port("enc_pulse_init", "str"),
        Port("train_enc_params", "bool"),
        Port("batch_size", "int"),
    ],
    provides=[
        Port("train_mse", "float", stream=True),
        *(Port(name, "float", stream=True) for name in SCALER_PORTS),
        # the training step the ranks were measured at: they are evaluated
        # every `rank_report_interval` steps, so a rank series counts its own
        # emissions rather than the loop's
        Port("rank_step", "int", stream=True),
        Port("rank_r_theta", "int", stream=True),
        Port("rank_sv_theta", "float", stream=True),
        Port("rank_r_ext", "int", stream=True),
        Port("rank_sv_ext", "float", stream=True),
        Port("trained_model", "artifact"),
        Port("training", "json"),
    ],
    # Yields every step, which resets the watchdog -- but the first step also
    # pays for jit compilation and, when ranks are on, for the Jacobian at
    # step zero.
    timeout=3600,
    cache=False,
)
def train_model(
    *,
    model: Dict,
    model_spec: Dict,
    dataset: Dict,
    dataset_info: Dict,
    noise_params: Dict,
    loss_functions: List[str],
    loss_scalers: List[float],
    steps: int,
    learning_rate: float,
    gate_mode: str = "enc_pulse",
    pulse_learning_rate: float = 1e-3,
    rank_eval: bool = False,
    rank_tol_rel: float = 1e-8,
    rank_report_interval: int = 100,
    enc_pulse_init: str = "ones",
    train_enc_params: bool = False,
    batch_size: int = -1,
) -> Generator[Dict[str, float], None, Dict]:
    """Fit one model to its target series and store what it became."""
    circuit = load_model(model, model_spec)
    arrays = load_dataset(dataset)

    final = yield from fit(
        circuit,
        arrays["domain_samples"],
        arrays["fourier_samples"],
        noise_params=noise_params,
        loss_functions=loss_functions,
        loss_scalers=loss_scalers,
        steps=steps,
        learning_rate=learning_rate,
        gate_mode=gate_mode,
        pulse_learning_rate=pulse_learning_rate,
        rank_eval=rank_eval,
        rank_tol_rel=rank_tol_rel,
        rank_report_interval=rank_report_interval,
        target_etas=arrays.get("target_etas"),
        enc_pulse_init=enc_pulse_init,
        train_enc_params=train_enc_params,
        batch_size=batch_size,
    )

    return {
        "trained_model": save_model(circuit),
        "training": jsonable(final),
    }
