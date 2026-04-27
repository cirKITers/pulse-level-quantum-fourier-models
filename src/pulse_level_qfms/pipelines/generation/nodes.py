from qml_essentials.model import Model
from qml_essentials.coefficients import Datasets
from qml_essentials.ansaetze import Ansaetze, Circuit, Block, Encoding
from qml_essentials.gates import Gates, PulseInformation, PulseEnvelope
from qml_essentials.yaqsi import Yaqsi

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


@dataclass
class LeafStep:
    """A single leaf-level gate produced by flattening a
    :class:`PulseParams` decomposition tree.

    Two kinds of leaf steps exist:

    * **Fixed** (``is_fixed=True``): the decomposition angle is a structural
      constant (e.g. ``π/2`` from a Hadamard).  The effective angle at
      build time is ``scaler * fixed_value`` where ``scaler`` is a
      trainable parameter initialised to ``1.0``.  At init the gate
      reproduces the decomposition exactly.

    * **Free** (``is_fixed=False``): the decomposition angle depends on
      the *original* gate's rotation parameter ``w`` via a composed
      ``angle_chain``.  At build time the angle is ``angle_chain(w_orig)``
      where ``w_orig`` is the **single** trainable parameter (or triple,
      for :class:`Rot`) associated with the parent block -- it is
      *shared* across all free steps of the same decomposition.

    * **No-param** (``has_param=False``): leaf gate carries no rotation
      angle (e.g. ``CZ``/``CX`` when they appear as basis gates).

    Attributes
    ----------
    gate_name : str
        Basis gate name (e.g. ``"RZ"``, ``"CPhase"``).
    wire_fn : str
        Wire selector ``"all"`` / ``"target"`` / ``"control"``.
    has_param : bool
        Whether this gate takes a rotation angle argument.
    is_fixed : bool
        True ⇒ angle = scaler * fixed_value.  False ⇒ angle =
        angle_chain(w_orig).
    fixed_value : float
        Structural constant for fixed steps.  Unused otherwise.
    angle_chain : Optional[Callable]
        Composition of all parent ``angle_fn``s for free steps, mapping
        the original gate's ``w`` to this leaf's rotation angle.
    """

    gate_name: str
    wire_fn: str
    has_param: bool
    is_fixed: bool = False
    fixed_value: float = 1.0
    angle_chain: Optional[Callable] = None


def _flatten_decomposition(
    pp, parent_wire_fn: str = "all", angle_chain: Optional[Callable] = None
) -> List[LeafStep]:
    """Walk a :class:`PulseParams` tree top-down and emit leaf-level steps.

    ``angle_chain`` is the composition of all ``angle_fn`` encountered on
    the path from the root; probing it at two distinct ``w`` values tells
    us whether the effective angle is a structural constant (*fixed*) or
    still depends on the parent's ``w`` (*free*).  For free steps the
    chain itself is stored so that :meth:`DecomposedCircuit.build` can
    reproduce the original ``w`` → leaf-angle mapping with a single
    shared ``w`` per parent block.
    """
    # Leaf: emit a LeafStep.
    if pp.is_leaf:
        # CZ → CPhase(π) so the phase becomes a trainable scaler
        # (identical to CZ when scaler == 1.0).
        # if pp.name == "CZ":
        #     return [LeafStep("CPhase", parent_wire_fn, True, True, float(jnp.pi))]
        if pp.name not in ("RX", "RY", "RZ"):
            return [LeafStep(pp.name, parent_wire_fn, False)]

        # Parameterised leaf: classify by probing the composed angle chain.
        if angle_chain is None:
            # RX/RY/RZ appearing *as* the root block -- pure passthrough.
            return [
                LeafStep(pp.name, parent_wire_fn, True, False, 1.0, lambda w: w)
            ]
        try:
            v1, v2 = float(angle_chain(1.0)), float(angle_chain(2.0))
        except (TypeError, IndexError):
            # e.g. Rot-style angle_fn expecting an indexable w
            v1 = float(angle_chain([1.0, 2.0, 3.0]))
            v2 = float(angle_chain([4.0, 5.0, 6.0]))
        if abs(v1 - v2) < 1e-12:
            return [LeafStep(pp.name, parent_wire_fn, True, True, v1)]
        return [
            LeafStep(pp.name, parent_wire_fn, True, False, 1.0, angle_chain)
        ]

    # Composite: descend into each DecompositionStep.
    steps: List[LeafStep] = []
    for step in pp.decomposition:
        # A child's "all" inherits the parent's wire specificity.
        eff_wire = (
            step.wire_fn
            if (step.wire_fn != "all" or parent_wire_fn == "all")
            else parent_wire_fn
        )
        # Compose angle: new_chain(w) = step.angle_fn(angle_chain(w)).
        if step.angle_fn is None:
            next_chain = angle_chain
        elif angle_chain is None:
            next_chain = step.angle_fn
        else:
            next_chain = (
                lambda w, _f=step.angle_fn, _c=angle_chain: _f(_c(w))
            )
        steps.extend(_flatten_decomposition(step.gate, eff_wire, next_chain))
    return steps


@dataclass
class DecomposedBlock:
    """One original :class:`Block` paired with its flat leaf-level
    decomposition.  Empty ``leaf_steps`` means the block is already a
    basis gate and should be applied via :meth:`Block.apply` unchanged.
    """

    original_block: Block
    leaf_steps: List[LeafStep] = field(default_factory=list)


def _resolve_wires(wire_fn: str, wires) -> Union[int, list]:
    """Map ``"all"`` / ``"target"`` / ``"control"`` to qubit indices.

    Mirrors :meth:`qml_essentials.pulses.PulseGates._resolve_wires`.
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


class DecomposedCircuit(Circuit):
    """A :class:`Circuit` whose gates are replaced by their basis-gate
    decomposition derived from :class:`PulseInformation`.

    Layout per wire-set of an originally parameterised block
    (``RX``/``CRX``/``Rot``/…):

    * ``n_w_orig`` **free** slots carry the original gate's rotation
      parameter(s) -- exactly as many as the un-decomposed gate would
      have consumed (1 for rotation gates, 3 for :class:`Rot`).  They
      are randomly initialised by :class:`Model` and are shared across
      *all* free leaf steps through the stored ``angle_chain``.
    * One **scaler** slot per *fixed* leaf step, initialised to ``1.0``,
      so that the effective angle is ``scaler * fixed_value``.

    At initialisation (all scalers == 1.0) the decomposed block is
    functionally identical to the original one: a CRX(w) decomposes into
    one random ``w`` plus two scalers for ``±π/2`` -- i.e. still a CRX.

    For non-parameterised originals (H, CX, CY, CZ) only the scaler
    slots exist.
    """

    def __init__(self, decomposed_blocks: List[DecomposedBlock]) -> None:
        super().__init__()
        self._decomposed_blocks = decomposed_blocks

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _n_w_orig(block: Block) -> int:
        """Number of original trainable params per wire-set of a block."""
        if not block.is_rotational:
            return 0
        return 3 if block.gate.__name__ == "Rot" else 1

    @staticmethod
    def _n_scalers_per_wireset(db: "DecomposedBlock") -> int:
        return sum(1 for s in db.leaf_steps if s.is_fixed)

    def _iter_wire_sets(self, block: Block, n_qubits: int):
        """Yield the wire-sets a block acts on (or nothing if skipped)."""
        if block.is_entangling:
            if not block.enough_qubits(n_qubits):
                return
            yield from block.topology(n_qubits=n_qubits, **block.kwargs)
        else:
            yield from range(n_qubits)

    def _slots_per_wireset(self, db: "DecomposedBlock") -> int:
        if not db.leaf_steps:
            return 0  # handled by Block.n_params directly
        return self._n_w_orig(db.original_block) + self._n_scalers_per_wireset(db)

    # --- Circuit API -----------------------------------------------------
    def n_params_per_layer(self, n_qubits: int) -> int:
        total = 0
        for db in self._decomposed_blocks:
            if not db.leaf_steps:
                total += db.original_block.n_params(n_qubits)
                continue
            slots = self._slots_per_wireset(db)
            total += slots * sum(
                1 for _ in self._iter_wire_sets(db.original_block, n_qubits)
            )
        return total

    def n_pulse_params_per_layer(self, n_qubits: int) -> int:
        return 0

    def get_control_indices(self, n_qubits: int) -> Optional[List[int]]:
        return None

    def scaler_mask(self, n_qubits: int) -> jnp.ndarray:
        """Boolean mask over ``n_params_per_layer``: ``True`` for every
        slot that is a structural scaler (to be initialised to ``1.0``)."""
        mask: List[bool] = []
        for db in self._decomposed_blocks:
            if not db.leaf_steps:
                mask.extend([False] * db.original_block.n_params(n_qubits))
                continue
            n_w = self._n_w_orig(db.original_block)
            n_s = self._n_scalers_per_wireset(db)
            per_ws = [False] * n_w + [True] * n_s
            for _ in self._iter_wire_sets(db.original_block, n_qubits):
                mask.extend(per_ws)
        return jnp.array(mask, dtype=bool)

    def build(self, w, n_qubits: int, **kwargs) -> None:
        w_idx = 0
        for db in self._decomposed_blocks:
            block = db.original_block

            if not db.leaf_steps:
                # Already a basis gate -- delegate to the original Block.
                w_idx = block.apply(n_qubits, w, w_idx, **kwargs)
                Gates.Barrier(wires=list(range(n_qubits)), **kwargs)
                continue

            n_w = self._n_w_orig(block)
            for wires in self._iter_wire_sets(block, n_qubits):
                # Original gate's trainable parameter(s), shared between
                # all free leaf steps of this wire-set.
                if n_w == 0:
                    w_orig = None
                elif n_w == 1:
                    w_orig = w[w_idx]
                    w_idx += 1
                else:
                    w_orig = w[w_idx : w_idx + n_w]
                    w_idx += n_w
                # Scalers and free angles are emitted in the order
                # leaf steps appear in db.leaf_steps.
                for step in db.leaf_steps:
                    gate_fn = getattr(Gates, step.gate_name)
                    step_wires = _resolve_wires(step.wire_fn, wires)

                    if not step.has_param:
                        gate_fn(wires=step_wires, **kwargs)
                    elif step.is_fixed:
                        gate_fn(
                            w[w_idx] * step.fixed_value,
                            wires=step_wires,
                            **kwargs,
                        )
                        w_idx += 1
                    else:
                        # Free step: derive angle from shared w_orig.
                        angle = (
                            step.angle_chain(w_orig)
                            if step.angle_chain is not None
                            else w_orig
                        )
                        gate_fn(angle, wires=step_wires, **kwargs)

            Gates.Barrier(wires=list(range(n_qubits)), **kwargs)


def _build_decomposed_blocks(structure: tuple) -> List[DecomposedBlock]:
    """Analyse an original circuit's ``structure()`` and return the
    corresponding :class:`DecomposedBlock` list.
    """
    blocks: List[DecomposedBlock] = []
    for block in structure:
        pulse_pp = PulseInformation.gate_by_name(block.gate.__name__)
        leaf_steps = (
            _flatten_decomposition(pulse_pp)
            if (pulse_pp is not None and not pulse_pp.is_leaf)
            else []
        )
        blocks.append(DecomposedBlock(original_block=block, leaf_steps=leaf_steps))
    return blocks


def _make_decomposed_circuit_class(
    structure: tuple, n_qubits: int
) -> Tuple[type, jnp.ndarray]:
    """Create a :class:`DecomposedCircuit` sub-class and its scaler mask.

    The mask is computed by the circuit itself to guarantee consistency
    with :meth:`DecomposedCircuit.build` / :meth:`n_params_per_layer`.
    """
    decomposed_blocks = _build_decomposed_blocks(structure)

    class _Decomposed(DecomposedCircuit):
        def __init__(self):
            super().__init__(decomposed_blocks=decomposed_blocks)

    _Decomposed.__name__ = "DecomposedCircuit"
    _Decomposed.__qualname__ = "DecomposedCircuit"

    scaler_mask = _Decomposed().scaler_mask(n_qubits)
    return _Decomposed, scaler_mask


def _apply_scaler_mask(model: Model, scaler_mask: jnp.ndarray) -> None:
    """Force structural-scaler parameter slots to ``1.0`` in-place.

    ``scaler_mask`` has shape ``(n_params_per_layer,)`` and is broadcast
    across the batch and layer dimensions of ``model.params``.
    """
    # model.params shape: (batch, n_layers, n_params_per_layer)
    model.params = jnp.where(
        scaler_mask[jnp.newaxis, jnp.newaxis, :], 1.0, model.params
    )




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
    decompose_circuit: bool,
    envelope: str,
    rwa: bool,
    frame: str,
) -> Dict[str, Model]:
    available_envelopes = PulseEnvelope.available()
    if envelope not in available_envelopes:
        raise ValueError(
            f"Unknown pulse envelope '{envelope}'. "
            f"Available: {available_envelopes}"
        )
    PulseInformation.set_envelope(envelope)
    PulseInformation.set_rwa(rwa)
    PulseInformation.set_frame(frame)
    log.info(f"Using pulse envelope: {envelope} with RWA={rwa} and frame={frame}")

    if not rwa:
        log.info(f"Using magnus4 solver as RWA is not enabled.")
        Yaqsi.set_solver_defaults(max_steps=1024, throw=False, solver="magnus4")

    log.info(
        f"Creating model with {n_qubits} qubits, {n_layers} layers, "
        f"and {circuit_type} circuit."
    )

    effective_circuit_type: Union[str, type] = circuit_type
    scaler_mask: Optional[jnp.ndarray] = None

    if decompose_circuit:
        log.info(
            f"Decomposing '{circuit_type}' into basis gates "
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
    
    if scaler_mask is not None:
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
