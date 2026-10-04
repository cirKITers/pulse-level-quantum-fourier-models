"""Pulse-level quantum Fourier models and their Fluksio flows.

Nodes live in computation-specific modules and are wired in :mod:`pipeline`.
Importing this package enables 64-bit JAX in each worker process.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1"
