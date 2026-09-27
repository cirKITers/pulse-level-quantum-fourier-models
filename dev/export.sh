#!/usr/bin/env bash
# The engine's own view of the studies, as CSV.
#
#   runs.csv    one row per run: its inputs as `param.*` columns, its final
#               numbers as `metric.*` columns, and `code_digest` -- the
#               engine's hash of the code it ran
#   curves.csv  tidy (run, name, step, ts, value): the streamed series
#
# This is the raw table. The figures come from each study's `figures.py`,
# which reads the same runs through the same export and reshapes them
# (`pulse_level_qfms/table.py`).
#
# Metric names are flow-qualified: `train.train_mse`, not `train_mse`.
# `fluksio export metrics --flow train --list` names what is available.
#
#   SINCE=2026-09-01T00:00 dev/export.sh [flow ...]
set -u
cd "$(dirname "$0")/.."
FLOWS=${*:-"fcc fidelity expressibility train train_enc spectrum landscape"}
SINCE=${SINCE:-}
export FLUKSIO_URL=${FLUKSIO_URL:-http://127.0.0.1:8767}
export FLUKSIO_TOKEN=${FLUKSIO_TOKEN:-$(python3 -c "import json;print(json.load(open('.fluksio/client.json'))['token'])")}

for flow in $FLOWS; do
  out="results/exports/$flow"
  mkdir -p "$out"
  uv run fluksio export runs --flow "$flow" --status ok ${SINCE:+--since "$SINCE"} \
    --format csv -o "$out/runs.csv"
  if [ "$flow" = "train" ] || [ "$flow" = "train_enc" ]; then
    uv run fluksio export metrics --flow "$flow" --status ok ${SINCE:+--since "$SINCE"} \
      --format csv -o "$out/curves.csv"
  fi
  wc -l "$out"/*.csv
done
