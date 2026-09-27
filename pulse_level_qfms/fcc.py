"""Fourier coefficient concentration, and the spectrum it is measured on.

`PulseFCC` extends the library's `FCC` with a ``sample_axis``: which of the
unitary parameters, the ansatz pulse scalers and the encoding pulse scalers
are randomised across the samples, drawn jointly rather than as an outer
product. Which of the four pulse regimes the model runs in follows from that
choice.
"""

import logging
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from fluksio import Port, node
from qml_essentials.coefficients import FCC, Coefficients
from qml_essentials.model import Model

from pulse_level_qfms.model import (
    GROUP_BATCH_AXIS,
    MODE_BY_GROUPS,
    PULSE_GROUPS,
    load_model,
)

log = logging.getLogger(__name__)


def frequency_key(frequency: float) -> str:
    """What a frequency is called inside the coefficient record.

    A record's keys are strings, and this one is read back through
    `fluksio export runs --metrics coefficients.mean.<key>`, which addresses
    a nested field by a dotted path. A key holding a dot would be read as two
    levels of nesting and come back empty, so the decimal point is written as
    an underscore: -1.0 is `-1_0`, and the quarter-integer grid the spectrum
    study samples on is `0_25`.
    """
    return f"{float(frequency)}".replace(".", "_")


def coefficient_stats(freqs, means, variances) -> Dict:
    """The per-frequency spectrum, keyed by frequency."""
    return {
        "mean": {
            frequency_key(f): float(m) for f, m in zip(freqs, means, strict=True)
        },
        "var": {
            frequency_key(f): float(v) for f, v in zip(freqs, variances, strict=True)
        },
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
        stats: Optional[Dict] = None,
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
            stats (Optional[Dict], optional): Filled in with the per-frequency
                mean and variance of the coefficient magnitudes, which is the
                spectrum the studies report. `get_fourier_fingerprint` trims
                the coefficients it returns, so a caller that wants the whole
                spectrum has to be handed it from in here.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Parameters,
            coefficients of size NxK and the corresponding frequencies.
        """
        if gate_mode is None:
            # the sampled pulse groups fully determine the regime, so there is
            # no combination left that the mode could fail to honour
            gate_mode = MODE_BY_GROUPS[
                frozenset(g for g in GROUP_BATCH_AXIS if g in sample_axis)
            ]
            log.info(f"Derived gate_mode={gate_mode} from sample_axis={sample_axis}")
        elif gate_mode not in PULSE_GROUPS:
            raise ValueError(
                f"Unknown gate_mode: {gate_mode}. Use one of {list(PULSE_GROUPS)}."
            )

        # only the groups this mode runs at pulse level can be sampled, the
        # model rejects a scaler group it does not run and such a group would
        # be inert on the coefficients anyway
        groups = PULSE_GROUPS[gate_mode]
        sampled_groups = [group for group in groups if group in sample_axis]
        ignored = [
            group
            for group in GROUP_BATCH_AXIS
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
                    repeat_batch_axis[GROUP_BATCH_AXIS[group]] = False
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

        if stats is not None:
            stats.update(coefficient_stats(freqs, means, variances))

        return model.params, coeffs, freqs


@node(
    requires=[
        Port("model", "artifact"),
        Port("model_spec", "json"),
        Port("sample_seed", "int"),
        Port("n_samples", "int"),
        Port("scale", "bool"),
        Port("method", "str"),
        Port("weighting", "bool"),
        Port("sample_axis", "list", item="str"),
        Port("pulse_params_variance", "float"),
        Port("numerical_cap", "float"),
    ],
    provides=[Port("fcc", "float"), Port("coefficients", "json")],
    # One coefficient pass over every sample, silent throughout.
    timeout=21600,
    cache=False,
)
def calculate_fcc(
    *,
    model: Dict,
    model_spec: Dict,
    sample_seed: int,
    n_samples: int,
    scale: bool,
    method: str,
    weighting: bool,
    sample_axis: List[str],
    pulse_params_variance: float,
    numerical_cap: float,
) -> Dict:
    """How concentrated the model's Fourier coefficients are under distortion."""
    circuit = load_model(model, model_spec)
    log.info(f"Seed for FCC: {sample_seed}")
    log.info(f"Sample axis: {sample_axis}")

    stats: Dict = {}
    # gate_mode is derived from sample_axis, see _calculate_coefficients
    fourier_fingerprint, _, _ = PulseFCC.get_fourier_fingerprint(
        circuit,
        n_samples,
        sample_seed,
        method=method,
        scale=scale,
        weight=weighting,
        trim_redundant=True,
        sample_axis=sample_axis,
        pulse_params_variance=pulse_params_variance,
        numerical_cap=numerical_cap,
        stats=stats,
    )

    return {
        "fcc": float(PulseFCC.calculate_fcc(fourier_fingerprint)),
        "coefficients": stats,
    }
