#!/bin/bash

# Loss landscape of the encoding scalers, one curve per encoding generator.
#
# offgrid_resolution is raised to 4: with the default 2, the displaced
# generators of a base-3 comb collide often enough that a slice reaches the
# target support at several scalers instead of one, which hides the basin the
# sweep is meant to show. mts must stay at or above the resolution so the
# domain covers a full period of the off-grid components.

set -e

uv run kedro run --pipeline study-6 --params="data.offgrid_resolution=4,data.mts=4"
