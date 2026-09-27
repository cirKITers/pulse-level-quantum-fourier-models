#!/usr/bin/env bash
# The engine. Run from the repo root, so its store is the default ./.fluksio
# and there is no data-dir to pass. Concurrency comes from the flags.
#
# JAX sizes its thread pool per worker, so raising RUNS on a shared machine
# oversubscribes the cores rather than using more of them. Start low.
set -u
cd "$(dirname "$0")/.."
exec env JAX_PLATFORMS=cpu uv run fluksio serve --port "${PORT:-8765}" \
  --max-runs "${RUNS:-5}" --max-cascades "${RUNS:-5}" --max-workers "${RUNS:-5}"
