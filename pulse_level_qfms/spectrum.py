"""The frequency spectrum of a model under pulse distortion.

The FCC study measures on the integer grid, where a shifted component has no
bin to move into. Oversampling the frequency axis by ``mts`` gives a bin
spacing of ``1/mts``, which is the resolution at which a frequency shift can
be told apart from a change in amplitude.
"""

import logging
from typing import Dict, List

from fluksio import Port, node

from pulse_level_qfms.fcc import PulseFCC
from pulse_level_qfms.model import load_model

log = logging.getLogger(__name__)


@node(
    requires=[
        Port("model", "artifact"),
        Port("model_spec", "json"),
        Port("sample_seed", "int"),
        Port("n_samples", "int"),
        Port("scale", "bool"),
        Port("sample_axis", "list", item="str"),
        Port("pulse_params_variance", "float"),
        Port("mfs", "int"),
        Port("mts", "int"),
    ],
    provides=[Port("coefficients", "json")],
    timeout=21600,
    cache=False,
)
def calculate_spectrum(
    *,
    model: Dict,
    model_spec: Dict,
    sample_seed: int,
    n_samples: int,
    scale: bool,
    sample_axis: List[str],
    pulse_params_variance: float,
    mfs: int,
    mts: int,
) -> Dict:
    """Report the oversampled spectrum, one mean and variance per frequency."""
    circuit = load_model(model, model_spec)
    log.info(f"Seed for spectrum: {sample_seed}")
    log.info(f"Sample axis: {sample_axis}, mfs={mfs}, mts={mts}")

    stats: Dict = {}
    # gate_mode is derived from sample_axis, see _calculate_coefficients.
    # numerical_cap is disabled so that every run reports the same frequency
    # grid: with a cap, the bins that vanish at zero variance would be
    # dropped and the runs could no longer be aggregated per frequency.
    PulseFCC._calculate_coefficients(
        circuit,
        n_samples,
        sample_seed,
        scale,
        sample_axis=sample_axis,
        pulse_params_variance=pulse_params_variance,
        numerical_cap=-1,
        mfs=mfs,
        mts=mts,
        stats=stats,
    )

    return {"coefficients": stats}
