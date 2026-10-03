"""Build, decompose, and transfer quantum Fourier models.

``DecomposedCircuit`` assigns trainable scalers to structural gate angles.
Models travel between workers as pickle artifacts with a ``model_spec``;
loading applies the pulse settings that the pickle cannot carry.
"""

import logging
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import fluksio
import jax.numpy as jnp
from fluksio import Port, node
from jaqsi.evolution import Evolution
from jaqsi.gates import Gates, PulseInformation
from jaqsi.pulses import PulseEnvelope
from qml_essentials.ansaetze import Ansaetze, Block, Circuit, Encoding
from qml_essentials.model import Model

log = logging.getLogger(__name__)

#: The pulse scaler groups each gate mode runs at pulse level.
PULSE_GROUPS = {
    "unitary": (),
    "ansatz_pulse": ("pulse",),
    "enc_pulse": ("enc_pulse",),
    "all_pulse": ("pulse", "enc_pulse"),
}

# position of each pulse scaler group in the differentiated coefficient
# function, i.e. the ``argnums`` handle used to extend ``J_\theta``
GROUP_ARGNUM = {"pulse": 1, "enc_pulse": 2}

# batch axis each pulse scaler group owns in ``model.repeat_batch_axis``,
# which is ordered [inputs, params, pulse_params, enc_pulse_params]
GROUP_BATCH_AXIS = {"pulse": 2, "enc_pulse": 3}

# inverse of PULSE_GROUPS, i.e. the mode that runs exactly the given pulse
# groups. Used to derive the gate mode from the sampled quantities, since
# running a group at pulse level without perturbing it reproduces the unitary
# result (the pulses are calibrated to the ideal gates).
MODE_BY_GROUPS = {frozenset(groups): mode for mode, groups in PULSE_GROUPS.items()}

# model attribute each trainable group writes back to. "enc" is the unitary
# trainable-frequency knob enc_params, the others are pulse scalers.
GROUP_ATTR = {
    "pulse": "pulse_params",
    "enc_pulse": "enc_pulse_params",
    "enc": "enc_params",
}

#: What the solver is set to when the rotating-wave approximation is off.
SOLVER_NO_RWA = {"max_steps": 1024, "throw": False, "solver": "magnus4"}

#: The solver the library ships with, for the keys the no-RWA path replaces.
#: There is no read-only accessor: setting the defaults returns the previous
#: values, so this asks for them and immediately puts them back rather than
#: restating constants that belong to jaqsi.
SOLVER_DEFAULTS = Evolution.set_solver_defaults(**SOLVER_NO_RWA)
Evolution.set_solver_defaults(**SOLVER_DEFAULTS)


@dataclass
class LeafStep:
    """One basis gate in a flattened pulse decomposition.

    Fixed steps scale a structural angle with a parameter initialized to one.
    Free steps share the parent gate's rotation parameter through
    ``angle_chain``. Steps without a rotation angle set ``has_param=False``.

    Attributes:
        gate_name: Basis gate name.
        wire_fn: ``"all"``, ``"target"``, or ``"control"``.
        has_param: Whether the gate takes a rotation angle.
        is_fixed: Whether the angle uses ``fixed_value`` and a scaler.
        fixed_value: Structural angle of a fixed step.
        angle_chain: Map from the parent rotation to a free step's angle.
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
    """Flatten a ``PulseParams`` tree into basis-gate steps.

    Probe the composed ``angle_chain`` at two values to distinguish fixed
    angles from those driven by the parent's shared rotation parameter.
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

    Mirrors :meth:`jaqsi.pulses.PulseGates._resolve_wires`.
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
    """Circuit with basis-gate decompositions from ``PulseInformation``.

    Each original rotation keeps its shared parameter slots. Each fixed
    decomposition angle gains a trainable scaler initialized to one, so the
    initial circuit implements the original gates. Gates without original
    parameters have only scaler slots.
    """

    def __init__(self, circuit_type: str, n_qubits: int) -> None:
        super().__init__()
        self._circuit_type = circuit_type
        self._n_qubits = n_qubits
        self._decomposed_blocks = _build_decomposed_blocks(
            getattr(Ansaetze, circuit_type).structure()
        )

    def __reduce__(self):
        """Rebuild from the ansatz name rather than from the blocks.

        The leaf steps hold composed ``angle_fn`` closures, which no pickle
        can carry. Recomputing them is cheap and gives the same circuit, as
        long as the pulse settings are applied first -- which is what
        :func:`load_model` is for.
        """
        return (rebuild_decomposed_circuit, (self._circuit_type, self._n_qubits))

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


def rebuild_decomposed_circuit(circuit_type: str, n_qubits: int) -> DecomposedCircuit:
    """What a pickled :class:`DecomposedCircuit` comes back as."""
    return decomposed_circuit_class(circuit_type, n_qubits)()


def decomposed_circuit_class(circuit_type: str, n_qubits: int) -> type:
    """A zero-argument :class:`DecomposedCircuit` for one ansatz.

    :class:`Model` instantiates whatever class it is handed, so the ansatz
    name and the qubit count have to be bound before it gets there.
    """

    def __init__(self):
        DecomposedCircuit.__init__(self, circuit_type, n_qubits)

    return type(
        "DecomposedCircuit",
        (DecomposedCircuit,),
        {"__init__": __init__},
    )


def _apply_scaler_mask(model: Model, scaler_mask: jnp.ndarray) -> None:
    """Force structural-scaler parameter slots to ``1.0`` in-place.

    ``scaler_mask`` has shape ``(n_params_per_layer,)`` and is broadcast
    across the batch and layer dimensions of ``model.params``.
    """
    # model.params shape: (batch, n_layers, n_params_per_layer)
    model.params = jnp.where(
        scaler_mask[jnp.newaxis, jnp.newaxis, :], 1.0, model.params
    )


def pulse_settings(envelope: str, rwa: bool, frame: str) -> Dict:
    """The pulse configuration a model was built under.

    Class-level state of :mod:`jaqsi`, so it is recorded per run and applied
    again wherever the model is opened.
    """
    return {
        "envelope": envelope,
        "rwa": rwa,
        "frame": frame,
        "solver": dict(SOLVER_NO_RWA if not rwa else SOLVER_DEFAULTS),
    }


def apply_pulse_settings(spec: Dict) -> None:
    """Put the process into the pulse configuration a spec describes.

    A worker outlives the run that started it, so this always states the
    whole configuration rather than only what differs: a process that last
    ran without the rotating-wave approximation must not leave its solver
    behind for the next model.
    """
    PulseInformation.set_envelope(spec["envelope"])
    PulseInformation.set_rwa(spec["rwa"])
    PulseInformation.set_frame(spec["frame"])
    Evolution.set_solver_defaults(**spec["solver"])


def save_model(model: Model) -> Dict:
    """Store a model as a run artifact and return the reference to it."""
    return fluksio.save_artifact(
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL), "model.pkl"
    )


def load_model(ref: Dict, spec: Dict) -> Model:
    """Read a model back, in the pulse configuration it was built under."""
    apply_pulse_settings(spec)
    with open(fluksio.load_artifact(ref), "rb") as handle:
        return pickle.load(handle)


@node(
    requires=[
        Port("n_qubits", "int"),
        Port("n_layers", "int"),
        Port("circuit_type", "str"),
        Port("data_reupload", "bool"),
        Port("encoding_gates", "list", item="str"),
        Port("encoding_strategy", "str"),
        Port("initialization", "str"),
        Port("initialization_domain", "list", item="float"),
        Port("output_qubit", "int"),
        Port("model_seed", "int"),
        Port("decompose_circuit", "bool"),
        Port("envelope", "str"),
        Port("rwa", "bool"),
        Port("frame", "str"),
    ],
    provides=[Port("model", "artifact"), Port("model_spec", "json")],
    # Building a decomposed circuit walks every gate's decomposition tree and
    # says nothing while it does.
    timeout=600,
    # The stage digest stops at installed packages, and qml-essentials and
    # jaqsi are installed from a working tree that is edited between runs, so
    # a cache hit here could replay a model the current library would not
    # build.
    cache=False,
)
def generate_model(
    *,
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    data_reupload: bool,
    encoding_gates: Union[str, Callable, List[str], List[Callable]],
    encoding_strategy: str,
    initialization: str,
    initialization_domain: List[float],
    output_qubit: int,
    model_seed: int,
    decompose_circuit: bool,
    envelope: str,
    rwa: bool,
    frame: str,
) -> Dict:
    """Build the study's model and store it for the nodes downstream."""
    available_envelopes = PulseEnvelope.available()
    if envelope not in available_envelopes:
        raise ValueError(
            f"Unknown pulse envelope '{envelope}'. Available: {available_envelopes}"
        )

    spec = pulse_settings(envelope, rwa, frame)
    apply_pulse_settings(spec)
    log.info(f"Using pulse envelope: {envelope} with RWA={rwa} and frame={frame}")
    if not rwa:
        log.info("Using magnus4 solver as RWA is not enabled.")

    log.info(
        f"Creating model with {n_qubits} qubits, {n_layers} layers, "
        f"and {circuit_type} circuit."
    )

    effective_circuit_type: Union[str, type] = circuit_type
    scaler_mask: Optional[jnp.ndarray] = None

    if decompose_circuit:
        log.info(
            f"Decomposing '{circuit_type}' into basis gates with trainable scalers."
        )
        effective_circuit_type = decomposed_circuit_class(circuit_type, n_qubits)
        scaler_mask = effective_circuit_type().scaler_mask(n_qubits)
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
        random_seed=model_seed,
    )

    # After Model randomly initialises *all* params, force structural
    # scaler slots to exactly 1.0 so that the decomposed circuit starts
    # out functionally equivalent to the original.
    if scaler_mask is not None:
        log.info(
            f"Applying scaler mask: setting {int(scaler_mask.sum())}/"
            f"{model.params.shape[-1]} parameter slots to 1.0"
        )
        _apply_scaler_mask(model, scaler_mask)

    log.debug(f"Created quantum model with {model.params.size} trainable parameters.")

    spec.update(
        n_pulse_params=int(model.pulse_params.size),
        n_gate_params=int(model.params.size),
        n_decomposed_param_slots=None if scaler_mask is None else int(len(scaler_mask)),
        n_scaler_params=None if scaler_mask is None else int(scaler_mask.sum()),
        summary=str(model),
    )

    return {"model": save_model(model), "model_spec": spec}
