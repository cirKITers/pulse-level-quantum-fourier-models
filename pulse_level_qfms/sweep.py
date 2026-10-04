"""Submit bounded study grids and resume incomplete sweeps.

The driver records completed cells locally and checks the engine for finished
runs before submitting missing parameter sets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from fluksio.sdk.client import Client

#: The ansaetze every distortion study sweeps, in the order the studies ran.
CIRCUITS = [
    "Circuit_2",
    "Circuit_4",
    "Circuit_8",
    "Circuit_14",
    "Circuit_15",
    "Circuit_17",
    "Circuit_19",
    "Circuit_20",
    "Strongly_Entangling",
    "Circuit_3",
    "Circuit_9",
    "Circuit_10",
    "Circuit_16",
    "Circuit_18",
    "Circuit_7",
    "Circuit_13",
    "Hardware_Efficient",
]

#: Pulse scaler variances, from calibrated to visibly detuned.
VARIANCES = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]

#: Seeds each cell is repeated over.
SEEDS = list(range(1000, 1010))

#: How long to wait between two passes over the runs in flight.
POLL_S = 10.0


def client() -> Client:
    """Create a Fluksio client with environment credentials and a 300 s timeout."""
    url = os.environ.get("FLUKSIO_URL", "")
    token = os.environ.get("FLUKSIO_TOKEN", "")
    return Client(url=url, token=token, timeout=300.0)


def flow_defaults(engine: Client, flow: str) -> Dict[str, Any]:
    """Every input of a flow at the value it declares."""
    stored = engine.get_flow(flow)
    if stored is None:
        raise SystemExit(f"no flow '{flow}' on that engine — run `fluksio sync` first")
    definition = stored.get("definition") or {}
    return {
        entry["spec"]["name"]: entry.get("initial")
        for entry in definition.get("inputs") or []
    }


def key(cell: Dict[str, Any]) -> str:
    """What makes two runs the same run: their whole parameter set."""
    return json.dumps(cell, sort_keys=True, default=str)


def engine_done(engine: Client, flow: str, defaults: Dict[str, Any]) -> set:
    """The cells this engine already has a finished run of."""
    done = set()
    # the inputs by name: left to itself the export splits a record input
    # (`noise_params`) into one column per field, which no cell matches
    for row in engine.export_runs(flow=flow, status="ok", params=",".join(defaults)):
        params = {
            name[len("param.") :]: value
            for name, value in row.items()
            if name.startswith("param.") and value not in ("", None)
        }
        done.add(key({**defaults, **params}))
    return done


def run_grid(
    flow: str,
    cells: Sequence[Dict[str, Any]],
    *,
    seed_key: str,
    jobs: int,
    out: Path,
    limit: int = 0,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Submit one cell per parameter set, `jobs` of them at a time.

    Args:
        flow: Which flow to run.
        cells: The parameter sets, each one run.
        seed_key: Which input carries the repetition seed. It is also handed
            to the engine as the run's seed, so a run says what it varied
            without anyone reading its parameters.
        jobs: How many runs to keep in flight.
        out: Where the record of the sweep is written, after every cell.
        limit: Submit at most this many cells, for a trial pass.
        dry_run: Print the grid and submit nothing.

    Returns:
        One record per cell that has finished, this call or before.
    """
    if dry_run:
        # the grid as written. What a live call submits is this minus the
        # cells the engine already has, and minus any two that are the same
        # cell once the flow's defaults are filled in -- neither of which can
        # be known without asking the engine.
        print(f"{flow}: {len(cells)} cells as written")
        for cell in cells[:5]:
            print("  ", json.dumps(cell, sort_keys=True))
        if len(cells) > 5:
            print(f"   ... and {len(cells) - 5} more")
        return []

    engine = client()
    defaults = flow_defaults(engine, flow)

    out.parent.mkdir(parents=True, exist_ok=True)
    done: List[Dict[str, Any]] = json.loads(out.read_text()) if out.exists() else []
    seen = {
        key({**defaults, **row["cell"]})
        for row in done
        if row.get("status") == "ok"
    }
    seen |= engine_done(engine, flow, defaults)

    # `seen` grows as the queue is built, so a grid that names the same cell
    # twice -- two blocks of one study meeting at their shared corner -- runs
    # it once rather than racing two identical runs into the engine
    queue = []
    for cell in cells:
        signature = key({**defaults, **cell})
        if signature not in seen:
            seen.add(signature)
            queue.append(cell)
    remaining = len(queue)
    if limit:
        queue = queue[:limit]
    total = len(done) + len(queue)
    held = remaining - len(queue)
    print(
        f"{flow}: {len(queue)} to submit "
        f"({len(cells) - remaining} already done"
        + (f", {held} held back by --limit" if held else "")
        + f"), {jobs} in flight",
        flush=True,
    )

    inflight: List[tuple] = []
    while queue or inflight:
        while queue and len(inflight) < jobs:
            cell = queue.pop(0)
            handle = engine.submit(flow, cell, seed=cell.get(seed_key))
            inflight.append((cell, handle))
        if not inflight:
            break
        time.sleep(POLL_S)
        for cell, handle in list(inflight):
            if not handle.refresh().done:
                continue
            inflight.remove((cell, handle))
            done.append({"cell": cell, "run": handle.id, "status": handle.status})
            out.write_text(json.dumps(done, indent=1))
            print(f"  [{len(done)}/{total}] {handle.status} {key(cell)}", flush=True)

    failed = [row for row in done if row.get("status") != "ok"]
    if failed:
        print(f"{flow}: {len(failed)} of {len(done)} did not finish", flush=True)
    return done


def main(
    flow: str,
    cells: Callable[[], List[Dict[str, Any]]],
    seed_key: str,
    out: Path,
) -> None:
    """The command line every study driver shares.

    ``out`` is the study's own record of the sweep, i.e.
    ``dev/<study>/results/driver.json``.
    """
    parser = argparse.ArgumentParser(description=f"Sweep {flow}.")
    parser.add_argument("--jobs", type=int, default=5, help="runs in flight")
    parser.add_argument("--limit", type=int, default=0, help="submit at most this many")
    parser.add_argument("--dry-run", action="store_true", help="print the grid only")
    parser.add_argument(
        "--out", type=Path, default=out, help="where the record of the sweep goes"
    )
    args = parser.parse_args()

    run_grid(
        flow,
        cells(),
        seed_key=seed_key,
        jobs=args.jobs,
        out=args.out,
        limit=args.limit,
        dry_run=args.dry_run,
    )
