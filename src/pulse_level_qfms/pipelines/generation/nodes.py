from qml_essentials.model import Model
from qml_essentials.coefficients import Datasets
from qml_essentials.ansaetze import Ansaetze, Circuit, DeclarativeCircuit, Block, Encoding
from qml_essentials.gates import Gates, PulseInformation

from typing import List, Dict, Tuple, Union, Callable, Optional
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numpy as np
import mlflow

import torch
from torch.utils.data import TensorDataset, DataLoader

import logging

log = logging.getLogger(__name__)


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Gate decomposition utilities
# ---------------------------------------------------------------------------


@dataclass
class FlatStep:
    """A single leaf-level gate produced by recursively flattening a
    :class:`PulseParams` decomposition tree.

    Attributes
    ----------
    gate_name : str
        Gates name, e.g. ``"RZ"``, ``"CZ"``.
    wire_fn : str
        Wire selector inherited from the decomposition hierarchy:
        ``"all"``, ``"target"``, or ``"control"``.
    has_param : bool
        Whether this gate takes a rotation angle.
    is_fixed : bool
        ``True`` when the rotation angle is a structural constant
        (independent of the original gate parameter *w*).
        These slots should be initialised to 1.0 (scaler).
    fixed_value : float
        The numerical value of the fixed angle.  Only meaningful when
        ``is_fixed is True``.  For free parameters ``fixed_value`` is
        set to ``1.0`` so that ``scaler * fixed_value`` equals the
        trainable angle directly.
    """

    gate_name: str
    wire_fn: str
    has_param: bool
    is_fixed: bool = False
    fixed_value: float = 1.0


def _flatten_pulse_params(pp, parent_wire_fn: str = "all"):
    """Recursively flatten a :class:`PulseParams` tree into leaf-level steps.

    Returns a list of ``(FlatStep, angle_chain)`` pairs where *angle_chain*
    is a composed callable mapping the **original** gate parameter ``w`` all
    the way down to the angle seen by the leaf gate.
    """
    if pp.is_leaf:
        has_param = pp.name in ("RX", "RY", "RZ")
        return [
            (
                FlatStep(
                    gate_name=pp.name,
                    wire_fn=parent_wire_fn,
                    has_param=has_param,
                ),
                lambda w: w,
            )
        ]

    results = []
    for step in pp.decomposition:
        # Resolve wire context: a child's "all" inherits parent's specificity
        eff_wire = step.wire_fn
        if parent_wire_fn != "all" and step.wire_fn == "all":
            eff_wire = parent_wire_fn

        sub = _flatten_pulse_params(step.gate, eff_wire)

        for flat_step, child_fn in sub:
            if step.angle_fn is not None:
                outer_fn = step.angle_fn

                def composed(w, _outer=outer_fn, _inner=child_fn):
                    return _inner(_outer(w))

                results.append((flat_step, composed))
            else:
                results.append((flat_step, child_fn))
    return results


def _classify_flat_steps(pp) -> List[FlatStep]:
    """Flatten a :class:`PulseParams` tree and classify each parameterised
    step as *fixed* (structural scaler) or *free* (depends on the original
    gate angle ``w``).
    """
    raw = _flatten_pulse_params(pp)
    classified: List[FlatStep] = []
    for flat_step, angle_chain in raw:
        if not flat_step.has_param:
            flat_step.is_fixed = False
            flat_step.fixed_value = 0.0
            classified.append(flat_step)
            continue
        # Probe with two different w values to check dependence.
        # Some gates (e.g. Rot) expect w to be a list/array, so we
        # try both scalar and array probes.
        try:
            v1 = float(angle_chain(1.0))
            v2 = float(angle_chain(2.0))
        except (TypeError, IndexError):
            # angle_fn expects an indexable w (e.g. Rot: lambda w: w[0])
            v1 = float(angle_chain([1.0, 2.0, 3.0]))
            v2 = float(angle_chain([4.0, 5.0, 6.0]))
        if abs(v1 - v2) < 1e-12:
            flat_step.is_fixed = True
            flat_step.fixed_value = v1
        else:
            flat_step.is_fixed = False
            flat_step.fixed_value = 1.0
        classified.append(flat_step)
    return classified


@dataclass
class DecomposedBlock:
    """Stores the flat decomposition for one :class:`Block` of the original
    circuit, together with the topology / iteration metadata needed to
    replay the gates in :meth:`DecomposedCircuit.build`.
    """

    original_block: Block
    flat_steps: List[FlatStep] = field(default_factory=list)

    @property
    def n_param_steps(self) -> int:
        """Number of parameterised leaf steps (each consumes one ``w`` entry
        per wire-set)."""
        return sum(1 for s in self.flat_steps if s.has_param)


def _resolve_wires(wire_fn: str, wires) -> Union[int, list]:
    """Map ``"all"`` / ``"target"`` / ``"control"`` to actual qubit indices.

    ``wires`` is either an ``int`` (single-qubit gate) or a tuple/list
    ``(control, target)`` for two-qubit gates.
    """
    if isinstance(wires, int):
        return wires
    wires_list = list(wires)
    if wire_fn == "all":
        return wires_list if len(wires_list) > 1 else wires_list[0]
    if wire_fn == "target":
        return wires_list[-1]
    if wire_fn == "control":
        return wires_list[0]
    raise ValueError(f"Unknown wire_fn: {wire_fn!r}")


# ---------------------------------------------------------------------------
# DecomposedCircuit
# ---------------------------------------------------------------------------


class DecomposedCircuit(Circuit):
    """A :class:`Circuit` whose gates are replaced by their basis-gate
    decomposition derived from :class:`PulseInformation`.

    For every decomposed gate that carries a *numeric* predefined parameter
    (e.g. ``pi/2`` from a Hadamard decomposition) a trainable **scaler**
    parameter is introduced so that the effective angle becomes
    ``scaler * fixed_value``.  For decomposition steps whose parameter
    depends on the original gate's angle (``w``) the scaler directly
    acts as the full trainable angle (``fixed_value = 1.0``).

    With all scalers equal to ``1.0`` the decomposed circuit is functionally
    equivalent to the original one.
    """

    def __init__(
        self,
        decomposed_blocks: List[DecomposedBlock],
    ) -> None:
        super().__init__()
        self._decomposed_blocks = decomposed_blocks

    # -- Circuit interface ---------------------------------------------------

    def n_params_per_layer(self, n_qubits: int) -> int:
        total = 0
        for db in self._decomposed_blocks:
            block = db.original_block
            if db.flat_steps:
                # Decomposed: one w entry per parameterised step per wire-set
                if block.is_entangling:
                    if not block.enough_qubits(n_qubits):
                        continue
                    n_wires = len(
                        block.topology(n_qubits=n_qubits, **block.kwargs)
                    )
                else:
                    n_wires = n_qubits
                total += db.n_param_steps * n_wires
            else:
                # Undecomposed: delegate to original block
                total += block.n_params(n_qubits)
        return total

    def n_pulse_params_per_layer(self, n_qubits: int) -> int:
        # Decomposed circuit uses only unitary basis gates;
        # pulse params are not meaningful.
        return 0

    def get_control_indices(self, n_qubits: int) -> Optional[List[int]]:
        # After decomposition there are no controlled-rotation gates in
        # the traditional sense.
        return None

    def build(self, w, n_qubits: int, **kwargs) -> None:
        w_idx = 0
        for db in self._decomposed_blocks:
            block = db.original_block

            if db.flat_steps:
                # --- decomposed block ---
                iterator = (
                    block.topology(n_qubits=n_qubits, **block.kwargs)
                    if block.is_entangling
                    else range(n_qubits)
                )
                for wires in iterator:
                    if block.is_entangling and not block.enough_qubits(n_qubits):
                        continue
                    for step in db.flat_steps:
                        step_wires = _resolve_wires(step.wire_fn, wires)
                        gate_fn = getattr(Gates, step.gate_name)
                        if step.has_param:
                            gate_fn(
                                w[w_idx] * step.fixed_value,
                                wires=step_wires,
                                **kwargs,
                            )
                            w_idx += 1
                        else:
                            gate_fn(wires=step_wires, **kwargs)
            else:
                # --- undecomposed block (no known decomposition) ---
                w_idx = block.apply(n_qubits, w, w_idx, **kwargs)

            Gates.Barrier(wires=list(range(n_qubits)), **kwargs)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _build_decomposed_blocks(
    structure: tuple,
) -> Tuple[List[DecomposedBlock], List[bool]]:
    """Analyse an original circuit's ``structure()`` and build decomposed blocks.

    Returns
    -------
    decomposed_blocks : list[DecomposedBlock]
        One entry per original block.
    scaler_mask_per_step : list[bool]
        Flat list -- ``True`` for every parameterised step that is a
        structural scaler (one entry per step, *not* yet expanded by
        the number of wire-sets).
    """
    decomposed_blocks: List[DecomposedBlock] = []
    scaler_mask_per_step: List[bool] = []

    for block in structure:
        gate_name = block.gate.__name__
        pulse_pp = PulseInformation.gate_by_name(gate_name)

        if pulse_pp is not None and not pulse_pp.is_leaf:
            flat = _classify_flat_steps(pulse_pp)
            db = DecomposedBlock(original_block=block, flat_steps=flat)
            decomposed_blocks.append(db)
            for s in flat:
                if s.has_param:
                    scaler_mask_per_step.append(s.is_fixed)
        else:
            # No decomposition available -- already a basis gate
            db = DecomposedBlock(original_block=block, flat_steps=[])
            decomposed_blocks.append(db)
            # Original block params are free (not scalers)
            n_per_wire = (
                3
                if block.gate.__name__ == "Rot"
                else (1 if block.is_rotational else 0)
            )
            scaler_mask_per_step.extend([False] * n_per_wire)

    return decomposed_blocks, scaler_mask_per_step


def _expand_scaler_mask(
    decomposed_blocks: List[DecomposedBlock],
    scaler_mask_per_step: List[bool],
    n_qubits: int,
) -> jnp.ndarray:
    """Expand the per-step scaler mask to the full flat parameter vector.

    Each parameterised step is repeated once per wire-set (qubit or
    entangling pair), matching the layout of ``w`` consumed by
    :meth:`DecomposedCircuit.build`.
    """
    full_mask: List[bool] = []
    step_idx = 0
    for db in decomposed_blocks:
        block = db.original_block
        if db.flat_steps:
            if block.is_entangling:
                if not block.enough_qubits(n_qubits):
                    # Skip but still advance step_idx
                    step_idx += db.n_param_steps
                    continue
                n_wires = len(
                    block.topology(n_qubits=n_qubits, **block.kwargs)
                )
            else:
                n_wires = n_qubits
            for s in db.flat_steps:
                if s.has_param:
                    full_mask.extend(
                        [scaler_mask_per_step[step_idx]] * n_wires
                    )
                    step_idx += 1
        else:
            n_per_wire = (
                3
                if block.gate.__name__ == "Rot"
                else (1 if block.is_rotational else 0)
            )
            if block.is_entangling:
                if not block.enough_qubits(n_qubits):
                    step_idx += n_per_wire
                    continue
                n_wires = len(
                    block.topology(n_qubits=n_qubits, **block.kwargs)
                )
            else:
                n_wires = n_qubits
            for _ in range(n_per_wire):
                full_mask.extend(
                    [scaler_mask_per_step[step_idx]] * n_wires
                )
                step_idx += 1

    return jnp.array(full_mask, dtype=bool)


def _make_decomposed_circuit_class(
    structure: tuple,
    n_qubits: int,
) -> Tuple[type, jnp.ndarray]:
    """Create a :class:`DecomposedCircuit` sub-class and its scaler mask.

    Parameters
    ----------
    structure : tuple[Block, ...]
        The ``structure()`` of the original :class:`DeclarativeCircuit`.
    n_qubits : int
        Number of qubits (needed to expand the mask over wire-sets).

    Returns
    -------
    circuit_cls : type
        A :class:`DecomposedCircuit` sub-class that :class:`Model` can
        instantiate with ``circuit_cls()``.
    scaler_mask : jnp.ndarray
        Boolean array of shape ``(n_params_per_layer,)`` -- ``True`` for
        every parameter slot that is a structural scaler.
    """
    decomposed_blocks, scaler_mask_per_step = _build_decomposed_blocks(
        structure
    )
    scaler_mask = _expand_scaler_mask(
        decomposed_blocks, scaler_mask_per_step, n_qubits
    )

    class _Decomposed(DecomposedCircuit):
        def __init__(self):
            super().__init__(decomposed_blocks=decomposed_blocks)

    _Decomposed.__name__ = "DecomposedCircuit"
    _Decomposed.__qualname__ = "DecomposedCircuit"
    return _Decomposed, scaler_mask


def _apply_scaler_mask(model: Model, scaler_mask: jnp.ndarray) -> None:
    """Force structural-scaler parameter slots to ``1.0`` in-place.

    ``scaler_mask`` has shape ``(n_params_per_layer,)`` and is broadcast
    across the batch and layer dimensions of ``model.params``.
    """
    # model.params shape: (batch, n_layers, n_params_per_layer)
    mask_bc = scaler_mask[jnp.newaxis, jnp.newaxis, :]  # (1, 1, P)
    model.params = jnp.where(mask_bc, 1.0, model.params)




def generate_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    data_reupload: bool,
    encoding_gates: Union[str, Callable, List[str], List[Callable]],
    encoding_strategy: str,
    initialization: str,
    initialization_domain: List[float],
    output_qubit: int,
    seed: int,
    train_pulse: bool,
) -> Dict[str, Model]:
    log.info(
        f"Creating model with {n_qubits} qubits, {n_layers} layers, "
        f"and {circuit_type} circuit."
    )

    effective_circuit_type: Union[str, type] = circuit_type
    scaler_mask: Optional[jnp.ndarray] = None

    if not train_pulse:
        log.info(
            f"train_pulse=False: decomposing '{circuit_type}' into basis gates "
            f"with trainable scalers."
        )
        # Obtain the original structure from the named ansatz
        original_class = getattr(Ansaetze, circuit_type)
        structure = original_class.structure()

        effective_circuit_type, scaler_mask = _make_decomposed_circuit_class(
            structure=structure,
            n_qubits=n_qubits,
        )

        log.info(
            f"Decomposed circuit: {int(scaler_mask.sum())} scaler params, "
            f"{len(scaler_mask) - int(scaler_mask.sum())} free params "
            f"(total {len(scaler_mask)} per layer)"
        )

    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=effective_circuit_type,
        data_reupload=data_reupload,
        encoding=Encoding(strategy=encoding_strategy, gates=encoding_gates),
        output_qubit=output_qubit,
        initialization=initialization,
        initialization_domain=initialization_domain,
        random_seed=seed,
    )

    # After Model randomly initialises *all* params, force structural
    # scaler slots to exactly 1.0 so that the decomposed circuit starts
    # out functionally equivalent to the original.
    if scaler_mask is not None:
        n_scalers = int(scaler_mask.sum())
        log.info(
            f"Applying scaler mask: setting {n_scalers}/{model.params.shape[-1]} "
            f"parameter slots to 1.0"
        )
        _apply_scaler_mask(model, scaler_mask)

    log.debug(f"Created quantum model with {model.params.size} trainable parameters.")
    mlflow.log_text(str(model), "model.txt")

    mlflow.log_param("model.n_pulse_params", model.pulse_params.size)
    mlflow.log_param("model.n_gate_params", model.params.size)
    mlflow.log_param("model.train_pulse", train_pulse)
    if scaler_mask is not None:
        mlflow.log_param("model.decomposed", True)
        mlflow.log_param(
            "model.n_decomposed_param_slots", int(len(scaler_mask))
        )
        mlflow.log_param("model.n_scaler_params", int(scaler_mask.sum()))

    return {"model": model}


def generate_fourier_series(
    model: Model,
    coefficients_min: float,
    coefficients_max: float,
    zero_centered: bool,
    seed: int,
) -> jnp.ndarray:
    """
    Generates the Fourier series representation of a function.

    Parameters
    ----------
    domain_samples : jnp.ndarray
        Grid of domain samples.
    omega : List[List[float]]
        List of frequencies for each dimension.

    Returns
    -------
    jnp.ndarray
        Fourier series representation of the function.
    """
    domain_samples, fourier_samples, coefficients = Datasets.generate_fourier_series(
        random_key=jax.random.PRNGKey(seed),
        model=model,
        coefficients_min=coefficients_min,
        coefficients_max=coefficients_max,
        zero_centered=zero_centered,
    )

    return {
        "domain_samples": domain_samples,
        "fourier_samples": fourier_samples.flatten(),
        "coefficients": coefficients,
    }


def build_fourier_series_dataloader(
    batch_size: int, domain_samples, fourier_samples, coefficients: jnp.ndarray
):
    if batch_size < 1:
        batch_size = domain_samples.shape[0]
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.array(domain_samples)),
            torch.from_numpy(np.array(fourier_samples).squeeze()),
            torch.from_numpy(np.array(coefficients).squeeze()),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": train_loader,
    }
