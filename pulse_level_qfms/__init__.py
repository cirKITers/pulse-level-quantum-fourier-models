"""Pulse-level quantum Fourier models: the studies, as Fluksio flows.

The node functions live beside the code they use — one module per computation —
and :mod:`pulse_level_qfms.pipeline` says which of them make up a flow.
Every module here is importable and callable without an engine, which is
what keeps the science testable.

Importing the package enables 64-bit JAX. A node runs in a worker process
of its own, so this has to happen wherever a node module is imported, not
once in a run script.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1"
