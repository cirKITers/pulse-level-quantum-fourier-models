"""Measure Fourier coefficient concentration and the sampled spectrum.

``PulseFCC`` jointly samples the parameter groups selected by ``sample_axis``.
The selected pulse groups determine the gate mode unless one is specified.
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
    """Encode a frequency as a record key with no dots.

    Fluksio treats dots as nested-field separators, so ``-1.0`` becomes
    ``-1_0``.
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
        """Sample the model's Fourier coefficients and frequency spectrum.

        Args:
            model (Model): Quantum Fourier model.
            n_samples (int): Number of parameter samples.
            seed (int): Random seed.
            scale (bool): Whether to scale the sample count by model size.
            sample_axis (List[str]): Groups to sample jointly: ``"unitary"``
                (variational parameters), ``"pulse"`` (ansatz scalers), and
                ``"enc_pulse"`` (encoding scalers).
            gate_mode (Optional[str]): Execution mode. If omitted, derive it
                from the selected pulse groups. Explicit modes may run an
                unsampled pulse group; sampled groups excluded by the mode
                are ignored with a warning.
            pulse_params_variance (float): Variance of pulse scalers; zero
                leaves them at their defaults.
            stats (Optional[Dict]): If provided, receive the full per-frequency
                mean and variance of coefficient magnitudes.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Parameters,
            sampled coefficients, and frequencies.
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
