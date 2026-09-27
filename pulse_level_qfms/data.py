"""The target series a study trains on, and how it travels.

One node, and the two helpers that move its arrays. The series is built from
the `Datasets` primitives rather than from `Datasets.generate_fourier_series`,
so the target frequencies are a result in their own right and the domain can
be oversampled independently of the number of components.
"""

import io
import logging
from typing import Dict, Iterator, Tuple

import fluksio
import jax
import jax.numpy as jnp
import numpy as np
from fluksio import Port, node
from qml_essentials.coefficients import Datasets

from pulse_level_qfms.model import load_model

log = logging.getLogger(__name__)


def save_dataset(arrays: Dict[str, np.ndarray]) -> Dict:
    """Store the target series as a run artifact and return its reference."""
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    return fluksio.save_artifact(buffer.getvalue(), "dataset.npz")


def load_dataset(ref: Dict) -> Dict[str, np.ndarray]:
    """Read a target series back into arrays."""
    with np.load(fluksio.load_artifact(ref)) as data:
        return {key: data[key] for key in data.files}


def batches(
    x: np.ndarray, y: np.ndarray, batch_size: int
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Walk the training set once, in order.

    A batch size below one is the whole set, which is what every study runs.
    Nothing is shuffled: the domain samples are a grid, and the target is a
    function of where on it they sit.
    """
    if batch_size < 1:
        batch_size = x.shape[0]
    for start in range(0, x.shape[0], batch_size):
        yield (
            jnp.asarray(x[start : start + batch_size]),
            jnp.asarray(y[start : start + batch_size]),
        )


@node(
    requires=[
        Port("model", "artifact"),
        Port("model_spec", "json"),
        Port("coefficients_min", "float"),
        Port("coefficients_max", "float"),
        Port("zero_centered", "bool"),
        Port("data_seed", "int"),
        Port("mts", "int"),
        Port("mfs", "int"),
        Port("offgrid_mode", "str"),
        Port("offgrid_prob", "float"),
        Port("offgrid_resolution", "int"),
    ],
    provides=[Port("dataset", "artifact"), Port("dataset_info", "json")],
    # Silent while it draws the series, and the generator etas walk the whole
    # encoding.
    timeout=600,
    cache=False,
)
def generate_fourier_series(
    *,
    model: Dict,
    model_spec: Dict,
    coefficients_min: float,
    coefficients_max: float,
    zero_centered: bool,
    data_seed: int,
    mts: int,
    mfs: int,
    offgrid_mode: str,
    offgrid_prob: float,
    offgrid_resolution: int,
) -> Dict:
    """Draw the target Fourier series on the model's own frequency comb.

    Which frequencies the target carries is the study's independent variable
    under ``offgrid_mode="generator"``, so they and the scalers that reach
    them are recorded rather than left to be inferred from the arrays.
    """
    circuit = load_model(model, model_spec)
    # An on-grid target draws its coefficients from the seed's own key, as
    # `Datasets.generate_fourier_series` does and so the paper's study 4 did.
    # Only an off-grid target needs a second key, to displace frequencies with.
    key = jax.random.PRNGKey(data_seed)
    if offgrid_mode == "none":
        random_key, frequency_key = key, None
    else:
        random_key, frequency_key = jax.random.split(key)

    domain_samples = Datasets.construct_domain_samples(circuit, mts=mts, mfs=mfs)
    frequencies = Datasets.construct_frequencies(
        circuit,
        frequency_key,
        offgrid_mode=offgrid_mode,
        offgrid_prob=offgrid_prob,
        offgrid_resolution=offgrid_resolution,
    )
    coefficients = Datasets.construct_coefficients(
        random_key,
        circuit,
        coefficients_min=coefficients_min,
        coefficients_max=coefficients_max,
        zero_centered=zero_centered,
    )
    fourier_samples = Datasets.calculate_values(
        domain_samples, frequencies, coefficients
    )

    log.info(f"Target frequencies: {frequencies.flatten().tolist()}")
    info = {
        "target_frequencies": frequencies.flatten().tolist(),
        "n_offgrid": int(jnp.sum(frequencies != jnp.round(frequencies))),
        "n_points": int(domain_samples.shape[0]),
        "mts": mts,
        "mfs": mfs,
        "target_etas": None,
    }

    arrays = {
        "domain_samples": np.asarray(domain_samples),
        "fourier_samples": np.asarray(fourier_samples).squeeze(),
        "coefficients": np.asarray(coefficients),
        "target_frequencies": np.asarray(frequencies),
    }

    if offgrid_mode == "generator":
        target_etas = jnp.stack(
            Datasets.generator_etas(
                circuit, frequency_key, offgrid_prob, offgrid_resolution
            )
        )
        arrays["target_etas"] = np.asarray(target_etas)
        info["target_etas"] = np.asarray(target_etas).tolist()

    return {"dataset": save_dataset(arrays), "dataset_info": info}
