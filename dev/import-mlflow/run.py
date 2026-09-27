"""The paper's MLflow runs, imported into the engine as runs of today's flows.

The paper's figures were drawn from MLflow runs made under kedro. Which runs
those are is fixed by the CSVs its R plots were built from: the `run_id`
column of `study-1.csv` ... `study-4.csv` at the repo root. Each run is looked
up by that id in an MLflow file store and becomes a run of the flow that
computes the same thing today: its parameters mapped onto the flow's inputs,
its final metrics onto the flow's outputs, its curves onto the node's streams.

What has no counterpart today travels under `result.legacy`, beside the run's
MLflow identity, so a row's provenance is on the row. Artifacts are not
imported.

    uv run python dev/import-mlflow/run.py --dry-run
    uv run python dev/import-mlflow/run.py 4
    uv run python dev/import-mlflow/run.py --mlruns /path/to/mlruns 1 2 3

The engine answers an id it has already imported with `created: False` and
keeps what it has, so a run is imported once: try a mapping on a throwaway
store first.
"""

import argparse
import ast
import csv
import math
from collections import Counter
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from pulse_level_qfms import pipeline
from pulse_level_qfms.fcc import frequency_key
from pulse_level_qfms.sweep import client
from pulse_level_qfms.training import jsonable

REPO = Path(__file__).resolve().parents[2]

#: The flow each study CSV's runs become.
STUDIES = {
    "1": pipeline.fcc,
    "2": pipeline.fidelity,
    "3": pipeline.expressibility,
    "4": pipeline.train,
}

#: The input a study driver hands to the engine as the run's seed.
SEED = {
    "fcc": "sample_seed",
    "fidelity": "sample_seed",
    "expressibility": "sample_seed",
    "train": "data_seed",
}

#: MLflow parameters whose input is not their name without the group prefix.
RENAMED = {
    "model.seed": "model_seed",
    "data.seed": "data_seed",
    "fcc.seed": "sample_seed",
    "fidelity.seed": "sample_seed",
    "expressibility.seed": "sample_seed",
    "train.rank_eval.enabled": "rank_eval",
    "train.rank_eval.tol_rel": "rank_tol_rel",
    "train.rank_eval.report_interval": "rank_report_interval",
    "train.train_pulse": "gate_mode",
}

#: `train.train_pulse`, as the gate mode it selected.
GATE_MODE = {"False": "unitary", "True": "ansatz_pulse"}

#: The model counts `generate_model` reports in `model_spec`. MLflow has them
#: as `model.<name>`, a parameter or a metric.
COUNTS = (
    "n_pulse_params",
    "n_gate_params",
    "n_decomposed_param_slots",
    "n_scaler_params",
)

#: MLflow run status, as the engine calls it.
STATUS = {3: "ok", 4: "error", 5: "cancelled"}

#: Runs per request. A trained run carries some 6000 curve points.
PAGE = 50


def paper_ids(study: str) -> List[str]:
    """The run ids a study's figures were built from."""
    with open(REPO / f"study-{study}.csv", newline="") as handle:
        return [row["run_id"] for row in csv.DictReader(handle)]


def run_dirs(root: Path) -> Dict[str, Path]:
    """Every run of an MLflow file store, `<root>/<experiment_id>/<run_id>/`."""
    return {meta.parent.name: meta.parent for meta in root.glob("*/*/meta.yaml")}


def parse(port: Any, raw: str) -> Any:
    """A logged parameter back as the input's type. MLflow stored its `str()`,
    so `[0, 6.28]` has to be told that it is a list of floats."""
    if port.dtype == "str":
        return raw
    value = ast.literal_eval(raw)
    if port.dtype == "float":
        return float(value)
    if port.item == "float":
        return [float(item) for item in value]
    return value


def read_metrics(path: Path) -> Dict[str, List[Tuple[int, int, float]]]:
    """Each metric's history as (step, timestamp in ms, value), in step order."""
    metrics = {}
    for file in (path / "metrics").iterdir():
        points = []
        for line in file.read_text().splitlines():
            ts, value, step = line.split()
            points.append((int(step), int(ts), float(value)))
        metrics[file.name] = sorted(points)
    return metrics


def convert(flow: Any, path: Path) -> Dict[str, Any]:
    """One MLflow run as a `POST /runs/import` entry for `flow`."""
    meta = yaml.safe_load((path / "meta.yaml").read_text())
    experiment = yaml.safe_load((path.parent / "meta.yaml").read_text())["name"]
    params = {f.name: f.read_text() for f in (path / "params").iterdir()}
    tags = {f.name: f.read_text() for f in (path / "tags").iterdir()}
    metrics = read_metrics(path)
    last = {name: points[-1][2] for name, points in metrics.items()}

    ports = {port.name: port for port in flow.inputs}
    streamed = {
        port.name for node in flow.nodes for port in node.provides if port.stream
    }

    inputs: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    legacy_params: Dict[str, str] = {}
    for name, raw in params.items():
        target = RENAMED.get(name, name.split(".", 1)[-1])
        if name == "train.train_pulse":
            inputs[target] = GATE_MODE[raw]
        elif target in ports:
            inputs[target] = parse(ports[target], raw)
        elif target in COUNTS:
            counts[target] = int(raw)
        else:
            legacy_params[name] = raw
    completed = {**{name: port.initial for name, port in ports.items()}, **inputs}

    result: Dict[str, Any] = {}
    coefficients: Dict[str, Dict[str, float]] = {}
    legacy_metrics: Dict[str, Dict[str, list]] = {}
    rows = []
    for name, points in metrics.items():
        # `fcc`, `trace-distance`, `rank.r_theta`, as the port they are today
        port = name.replace(".", "_").replace("-", "_")
        if name.split(".", 1)[-1] in COUNTS:
            counts[name.split(".", 1)[-1]] = int(last[name])
        elif name.startswith("coeff.") and "coefficients" in flow.outputs:
            # coeff.mean.f-1.0 -> coefficients["mean"]["-1_0"]
            kind, frequency = name[len("coeff.") :].split(".f", 1)
            spectrum = coefficients.setdefault(kind, {})
            spectrum[frequency_key(float(frequency))] = last[name]
        elif port in streamed:
            # a stream's step counts its own emissions, as the engine's does
            rows += [
                {"name": f"{flow.name}.{port}", "step": i, "ts": ts / 1000, "value": v}
                for i, (_, ts, v) in enumerate(points)
            ]
        elif port in flow.outputs:
            result[port] = last[name]
        else:
            legacy_metrics[name] = {
                "steps": [step for step, _, _ in points],
                "values": [value for _, _, value in points],
            }

    # the ranks were measured every few training steps; which one travels
    # beside them, as it does from the training node
    if "rank_step" in streamed and "rank.r_theta" in metrics:
        rows += [
            {"name": f"{flow.name}.rank_step", "step": i, "ts": ts / 1000, "value": s}
            for i, (s, ts, _) in enumerate(metrics["rank.r_theta"])
        ]

    result["model_spec"] = {
        **{name: completed[name] for name in ("envelope", "rwa", "frame")},
        # None is what the node reports for a circuit that is not decomposed
        **{name: counts.get(name) for name in COUNTS},
    }
    if coefficients:
        result["coefficients"] = coefficients
    if "training" in flow.outputs:
        per_step = ("train_mse", "pulse_scaler_mean", "pulse_scaler_std")
        pulse = {stat: last.get(f"pulse_scaler_{stat}") for stat in ("mean", "std")}
        result["training"] = jsonable(
            {
                "gate_mode": completed["gate_mode"],
                "steps": completed["steps"],
                "train_mse": last["train_mse"],
                "scalers": {"pulse": pulse} if "pulse_scaler_mean" in last else {},
                "rank": {
                    name[len("rank.") :]: value
                    for name, value in last.items()
                    if name.startswith("rank.")
                },
                "skipped_nonfinite": sum(
                    not math.isfinite(value)
                    for name in per_step
                    for _, _, value in metrics.get(name, [])
                ),
            }
        )
    result["legacy"] = {
        "experiment": experiment,
        "experiment_id": meta["experiment_id"],
        "run_id": meta["run_id"],
        "run_name": meta["run_name"],
        "user_id": meta["user_id"],
        "tags": tags,
        "params": legacy_params,
        "metrics": legacy_metrics,
        # the inputs MLflow has no record of, i.e. that took the flow's default
        "defaulted": sorted(set(ports) - set(inputs)),
    }

    def stamp(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, UTC).isoformat()

    return {
        "flow": flow.name,
        "external_id": meta["run_id"],
        "params": inputs,
        "seed": completed[SEED[flow.name]],
        "status": STATUS.get(meta["status"], "error"),
        "created_at": stamp(meta["start_time"]),
        "started_at": stamp(meta["start_time"]),
        "finished_at": stamp(meta["end_time"]) if meta.get("end_time") else None,
        "result": result,
        "metrics": rows,
        "labels": ["paper", f"mlflow:{experiment}"],
        "group_id": f"mlflow:{meta['experiment_id']}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "studies", nargs="*", default=list(STUDIES), help="which study CSVs (1-4)"
    )
    parser.add_argument(
        "--mlruns", type=Path, default=REPO / "mlruns_paper", help="MLflow file store"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="convert and report, import nothing"
    )
    args = parser.parse_args()

    runs = run_dirs(args.mlruns)
    engine = None if args.dry_run else client()
    for study in args.studies:
        flow = STUDIES[study]
        ids = paper_ids(study)
        found = [runs[run_id] for run_id in ids if run_id in runs]
        print(f"study-{study}.csv -> {flow.name}: {len(found)} of {len(ids)} ids found")

        defaulted: Counter = Counter()
        kept: Counter = Counter()
        created = 0
        for page in batched((convert(flow, path) for path in found), PAGE):
            for entry in page:
                legacy = entry["result"]["legacy"]
                defaulted.update(legacy["defaulted"])
                kept.update([*legacy["params"], *legacy["metrics"]])
            if engine is not None:
                created += sum(row["created"] for row in engine.import_runs(list(page)))

        def listed(counter: Counter) -> str:
            return ", ".join(f"{name} ({n})" for name, n in sorted(counter.items()))

        print(f"  inputs from the flow defaults: {listed(defaulted) or '-'}")
        print(f"  kept under result.legacy: {listed(kept) or '-'}")
        if engine is not None:
            print(f"  {created} imported, {len(found) - created} already there")


if __name__ == "__main__":
    main()
