"""A flow's finished runs, read from the engine as one table.

One row per run: the inputs it was given, its final numbers and, for a
training flow, its streamed curves as paired ``.steps`` / ``.values`` lists.
The columns are the ones the paper's R plots were built from, i.e. what
``generate_df`` on ``main`` read out of MLflow into ``study-N.csv``, so for
s1–s4 this table is that CSV: ``ansatz`` for ``circuit_type``, ``model.seed``
for ``model_seed``, ``<flow>.seed`` for ``sample_seed``, and so on.
:mod:`pulse_level_qfms.viz` draws from it.
"""

from collections import defaultdict
from itertools import batched
from typing import Any, Dict, List

import pandas as pd
from fluksio.sdk.client import Client

from pulse_level_qfms.sweep import client

#: The flows whose node streams a training curve.
TRAINING = ("train", "train_enc")

#: The streamed series, by the port that streams them, under their column name
#: and in the paper's column order. A rank is measured every few steps, so its
#: x is the training step `rank_step` names rather than the count of emissions.
CURVES = {
    "train_mse": "train_mse",
    "pulse_scaler_mean": "pulse_scaler_mean",
    "pulse_scaler_std": "pulse_scaler_std",
    "enc_pulse_scaler_mean": "enc_pulse_scaler_mean",
    "enc_pulse_scaler_std": "enc_pulse_scaler_std",
    "rank_r_theta": "rank.r_theta",
    "rank_r_ext": "rank.r_ext",
    "rank_sv_theta": "rank.sv_theta",
    "rank_sv_ext": "rank.sv_ext",
}

#: Runs per curve request. A trained run streams some 6000 points, and the
#: export answers with one list (`docs/NOTEPAD.md`).
PAGE = 25


def table(flow: str, **filters: Any) -> pd.DataFrame:
    """Every finished run of `flow`, newest first, as the figures read it.

    Args:
        flow: Which flow's runs.
        **filters: Narrow the selection as the engine's export does:
            ``group=`` one sweep, ``since=`` / ``until=`` a creation time,
            ``ids=`` named runs.
    """
    engine = client()
    runs = engine.export_runs(flow=flow, status="ok", **filters)
    if not runs:
        raise SystemExit(f"no finished runs of '{flow}' on {engine.url}")
    print(f"{flow}: {len(runs)} finished runs")

    ids = [run["id"] for run in runs]
    rows = [_row(run) for run in runs]
    if flow in TRAINING:
        _curves(engine, flow, ids, rows)
    if flow == "landscape":
        _landscape(engine, ids, rows)
    return pd.DataFrame(rows)


def _row(run: Dict[str, Any]) -> Dict[str, Any]:
    """One exported run, under the paper's column names.

    The order of the keys is the column order of ``study-N.csv``, which is the
    order ``generate_df`` on ``main`` wrote them in; that is why fidelity's
    columns come after ``decompose_circuit`` and the other flows' before it.
    """
    flow = run["flow"]

    def param(name: str) -> Any:
        return run.get(f"param.{name}")

    def metric(path: str) -> Any:
        return run.get(f"metric.{path}")

    row = {
        # where the run was made: MLflow's id for an imported run
        "run_id": run["external_id"] or run["id"],
        "ansatz": param("circuit_type"),
        "model.seed": param("model_seed"),
    }
    if param("data_seed") is not None:
        row["data.seed"] = param("data_seed")
    row["model.n_pulse_params"] = metric("model_spec.n_pulse_params")
    row["model.n_gate_params"] = metric("model_spec.n_gate_params")

    # the spectrum's positive half, one column per frequency and statistic.
    # The record writes a frequency's decimal point as an underscore
    # (`pulse_level_qfms.fcc.frequency_key`)
    prefix = "metric.coefficients.var."
    keys = [
        name[len(prefix) :]
        for name, value in run.items()
        if name.startswith(prefix) and value is not None
    ]
    for key in sorted(keys, key=lambda key: float(key.replace("_", "."))):
        frequency = float(key.replace("_", "."))
        if frequency >= 0:
            row[f"coeff.var.f{frequency}"] = metric(f"coefficients.var.{key}")
            row[f"coeff.mean.f{frequency}"] = metric(f"coefficients.mean.{key}")

    if flow in ("fcc", "expressibility"):
        row[flow] = metric(flow)
    if flow in ("fcc", "expressibility", "spectrum"):
        row[f"{flow}.seed"] = param("sample_seed")
        row["pulse_params_variance"] = param("pulse_params_variance")

    if flow in TRAINING:
        row["train_mse"] = metric("training.train_mse")
        # the paper's `train.train_pulse`: some pulse scalers were trained
        row["train_pulse"] = param("gate_mode") != "unitary"
        if row["train_pulse"]:
            row["pulse_scaler_mean"] = metric("training.scalers.pulse.mean")
            row["pulse_scaler_std"] = metric("training.scalers.pulse.std")

    row["decompose_circuit"] = param("decompose_circuit")

    if flow == "fidelity":
        row["fidelity"] = metric("fidelity")
        row["fidelity.seed"] = param("sample_seed")
        row["pulse_params_variance"] = param("pulse_params_variance")
        row["trace-distance"] = metric("trace_distance")

    if flow == "landscape":
        row["model.encoding_strategy"] = param("encoding_strategy")
        row["model.n_qubits"] = param("n_qubits")

    return row


def _curves(
    engine: Client, flow: str, ids: List[str], rows: List[Dict[str, Any]]
) -> None:
    """Fold each run's streamed series into its row, as paired lists."""
    for page in batched(range(len(ids)), PAGE):
        # (run, port) -> {step: value}
        series: Dict[tuple, Dict[int, float]] = defaultdict(dict)
        for point in engine.export_metrics(flow=flow, ids=[ids[i] for i in page]):
            port = point["name"].removeprefix(f"{flow}.")
            series[point["run"], port][int(point["step"])] = float(point["value"])

        for i in page:
            at_step = series.get((ids[i], "rank_step"), {})
            for port, column in CURVES.items():
                points = series.get((ids[i], port))
                if not points:
                    continue
                steps = sorted(points)
                if port.startswith("rank_"):
                    rows[i][f"{column}.steps"] = [int(at_step[s]) for s in steps]
                else:
                    rows[i][f"{column}.steps"] = steps
                rows[i][f"{column}.values"] = [points[s] for s in steps]


def _landscape(engine: Client, ids: List[str], rows: List[Dict[str, Any]]) -> None:
    """Fold each run's per-gate record and loss curves into its row.

    Every curve of a gate shares that gate's scaler grid, and the grids of two
    gates differ, because the oscillation period follows the generator; so the
    grid travels as its own column, ``landscape.eta.<gate>``.
    """
    for run_id, row in zip(ids, rows, strict=True):
        result = engine.run(run_id)["result"]
        info = result["landscape"]
        for name in ("mts", "n_gates", "n_frequencies"):
            row[f"landscape.{name}"] = info[name]
        for gate in info["gates"]:
            row[f"landscape.generator.{gate['key']}"] = float(gate["generator"])
            row[f"landscape.target_eta.{gate['key']}"] = float(gate["target_eta"])
        for curve in ("profile", "analytic", "fixed"):
            for line in result[curve]["lines"]:
                key = line["label"]
                row[f"landscape.eta.{key}.values"] = [x for x, _ in line["points"]]
                row[f"landscape.{curve}.{key}.values"] = [y for _, y in line["points"]]
