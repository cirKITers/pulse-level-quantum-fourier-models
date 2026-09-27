"""The flows, each named by what it computes.

A study is a driver over one flow (``dev/<study>/run.py``): it varies its own
axis over a grid of runs of that flow. The nodes the flows share -- the model,
the target series -- are the same functions in every flow, wired by name.

    dev/serve.sh                                      # the engine, once
    uv run fluksio sync pulse_level_qfms              # upload the flows
    uv run fluksio run fcc --circuit_type Circuit_15 --wait
    uv run python dev/s1-fcc/run.py --jobs 15         # the whole grid

Input defaults are the paper's, as ``conf/base/parameters.yml`` on ``main``
states them. An input the paper did not have keeps the value its study
introduced it with, and a study that ran at other values pins those in its
driver.

Declarations only: the nodes live in :mod:`pulse_level_qfms.model`,
:mod:`pulse_level_qfms.data`, :mod:`pulse_level_qfms.fcc`,
:mod:`pulse_level_qfms.spectrum`, :mod:`pulse_level_qfms.fidelity`,
:mod:`pulse_level_qfms.expressibility`, :mod:`pulse_level_qfms.training` and
:mod:`pulse_level_qfms.landscape`.

Every flow reports ``model_spec``, which carries the parameter counts the
plots read and the pulse configuration the run was built under. Streams are
not outputs: a training curve is read with ``fluksio export metrics``, the
final numbers with ``fluksio export runs``.
"""

from typing import List, Sequence

from fluksio import Flow, Port

from pulse_level_qfms.data import generate_fourier_series
from pulse_level_qfms.expressibility import evaluate_expressibility
from pulse_level_qfms.fcc import calculate_fcc
from pulse_level_qfms.fidelity import evaluate_fidelity
from pulse_level_qfms.landscape import sweep_loss_landscape
from pulse_level_qfms.model import generate_model
from pulse_level_qfms.spectrum import calculate_spectrum
from pulse_level_qfms.training import train_model

#: The zero noise configuration every study runs at. Spelled out rather than
#: left empty so a run records which channels were available to it.
NO_NOISE = {
    "BitFlip": 0.0,
    "PhaseFlip": 0.0,
    "AmplitudeDamping": 0.0,
    "PhaseDamping": 0.0,
    "Depolarizing": 0.0,
}


def model_inputs() -> List[Port]:
    """What every study's circuit is built from."""
    return [
        Port("n_qubits", "int", initial=3),
        Port("n_layers", "int", initial=1),
        Port("circuit_type", "str", initial="Circuit_3"),
        Port("data_reupload", "bool", initial=True),
        Port("encoding_gates", "list", item="str", initial=["RY"]),
        # hamming, binary, ternary
        Port("encoding_strategy", "str", initial="ternary"),
        Port("output_qubit", "int", initial=-1),
        Port("initialization", "str", initial="random"),
        Port(
            "initialization_domain",
            "list",
            item="float",
            initial=[0.0, 6.283185307179586],
        ),
        Port("decompose_circuit", "bool", initial=False),
        Port("envelope", "str", initial="gaussian"),
        Port("rwa", "bool", initial=True),
        # ignored when rwa is set
        Port("frame", "str", initial="drive"),
        Port("model_seed", "int", initial=1000),
    ]


def sampling_inputs(sample_axis: Sequence[str] = ()) -> List[Port]:
    """What a study that measures a distorted model draws its samples from.

    ``sample_axis`` is a subset of "unitary", "pulse" and "enc_pulse",
    matched exactly. Which pulse entries appear also selects the regime the
    model runs in: neither gives unitary, "pulse" gives ansatz_pulse,
    "enc_pulse" gives enc_pulse and both give all_pulse.
    """
    ports = [
        Port("sample_seed", "int", initial=1000),
        Port("n_samples", "int", initial=500),
        Port("scale", "bool", initial=True),
        Port("pulse_params_variance", "float", initial=0.008),
    ]
    if sample_axis:
        ports.append(
            Port("sample_axis", "list", item="str", initial=list(sample_axis))
        )
    return ports


def data_inputs(on_grid: bool = False) -> List[Port]:
    """What the target Fourier series is drawn from.

    ``offgrid_mode`` says where the target frequencies come from: "none"
    keeps the model's own comb, "index" displaces each component
    independently, and "generator" displaces the per-gate generators, which
    is the only mode an encoding pulse configuration can reach.

    ``on_grid`` gives the paper's target instead: the model's own comb over
    one period, [0, 2π]. The off-grid inputs are still declared there, at
    values that leave them inert, because a flow cannot leave a node's port
    at the function default.
    """
    return [
        Port("data_seed", "int", initial=1000),
        Port("coefficients_min", "float", initial=0.0),
        Port("coefficients_max", "float", initial=1.0),
        Port("zero_centered", "bool", initial=True),
        # domain oversampling: mts sets the window in periods, mfs the density
        Port("mts", "int", initial=1 if on_grid else 4),
        Port("mfs", "int", initial=1),
        Port("offgrid_mode", "str", initial="none" if on_grid else "generator"),
        Port("offgrid_prob", "float", initial=0.0 if on_grid else 1.0),
        # offsets are multiples of 1/resolution; a power of two keeps the
        # frequencies exact
        Port("offgrid_resolution", "int", initial=2),
    ]


def training_inputs(gate_mode: str) -> List[Port]:
    """How a model is fitted to its target series, by default in ``gate_mode``."""
    return [
        Port("batch_size", "int", initial=-1),
        Port("noise_params", "json", initial=dict(NO_NOISE)),
        Port("steps", "int", initial=2000),
        Port("learning_rate", "float", initial=1e-3),
        Port("pulse_learning_rate", "float", initial=1e-3),
        Port("loss_functions", "list", item="str", initial=["mse"]),
        Port("loss_scalers", "list", item="float", initial=[1.0]),
        # must be "unitary" when decompose_circuit is set
        Port("gate_mode", "str", initial=gate_mode),
        # "ones" starts the encoding scalers at identity; "target"
        # oracle-initializes them at the etas that make the off-grid comb
        # reachable, which separates reachability from optimization
        Port("enc_pulse_init", "str", initial="ones"),
        Port("train_enc_params", "bool", initial=False),
        Port("rank_eval", "bool", initial=True),
        Port("rank_tol_rel", "float", initial=1e-8),
        Port("rank_report_interval", "int", initial=50),
    ]


fcc = Flow(
    "fcc",
    title="Fourier coefficient concentration under pulse distortion",
    nodes=[generate_model, calculate_fcc],
    inputs=[
        *model_inputs(),
        *sampling_inputs(["unitary", "pulse"]),
        Port("method", "str", initial="pearson"),
        Port("weighting", "bool", initial=False),
        Port("numerical_cap", "float", initial=1e-10),
    ],
    outputs=["model_spec", "fcc", "coefficients"],
)

fidelity = Flow(
    "fidelity",
    title="Fidelity of the distorted pulse-level circuit",
    nodes=[generate_model, evaluate_fidelity],
    inputs=[*model_inputs(), *sampling_inputs()],
    outputs=["model_spec", "fidelity", "trace_distance"],
)

expressibility = Flow(
    "expressibility",
    title="Expressibility under pulse distortion",
    nodes=[generate_model, evaluate_expressibility],
    inputs=[
        *model_inputs(),
        *sampling_inputs(["unitary", "pulse"]),
        Port("n_bins", "int", initial=50),
    ],
    outputs=["model_spec", "expressibility"],
)

train = Flow(
    "train",
    title="Train a model on its own Fourier series",
    nodes=[generate_model, generate_fourier_series, train_model],
    # "ansatz_pulse" is the paper's train_pulse=True
    inputs=[
        *model_inputs(),
        *data_inputs(on_grid=True),
        *training_inputs("ansatz_pulse"),
    ],
    outputs=["model_spec", "dataset_info", "training", "trained_model"],
)

train_enc = Flow(
    "train_enc",
    title="Train a model on an off-grid Fourier series",
    nodes=[generate_model, generate_fourier_series, train_model],
    inputs=[*model_inputs(), *data_inputs(), *training_inputs("enc_pulse")],
    outputs=["model_spec", "dataset_info", "training", "trained_model"],
)

spectrum = Flow(
    "spectrum",
    title="Frequency spectrum under encoding pulse distortion",
    nodes=[generate_model, calculate_spectrum],
    inputs=[
        *model_inputs(),
        *sampling_inputs(["unitary", "enc_pulse"]),
        # frequency oversampling: mts sets the bin spacing to 1/mts, which is
        # what makes a frequency shift resolvable at all
        Port("mfs", "int", initial=1),
        Port("mts", "int", initial=4),
    ],
    outputs=["model_spec", "coefficients"],
)

landscape = Flow(
    "landscape",
    title="Loss landscape along each encoding scaler",
    nodes=[generate_model, generate_fourier_series, sweep_loss_landscape],
    inputs=[
        *model_inputs(),
        *data_inputs(),
        # eta=1 is the calibrated configuration; the target scalers lie
        # within [1-1/resolution, 1+1/resolution]
        Port("eta_min", "float", initial=0.0),
        Port("eta_max", "float", initial=2.0),
        Port("points_per_period", "int", initial=20),
        Port("chunk", "int", initial=64),
    ],
    outputs=["model_spec", "dataset_info", "landscape", "profile", "analytic", "fixed"],
)
