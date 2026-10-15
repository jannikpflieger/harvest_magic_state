"""Litinski's "Game of Surface Codes" data blocks as an analytic matched-IR baseline.

Daniel Litinski, "A Game of Surface Codes: Large-Scale Quantum Computing with
Lattice Surgery", Quantum 3, 128 (2019), arXiv:1808.02892.  The paper is a
resource model, not a released compiler; this module evaluates its closed-form
data-block geometries on the exact matched Pauli-product stream (the same
``±π/8`` products PureMagic receives via ``pauli_dag_to_trans``).

Model (section numbers refer to arXiv:1808.02892v3):

* Pauli-based computation: the stream is a sequence of π/8 rotations executed
  **one after another** (Conclusion, "one T gate is performed after the
  other"); Clifford corrections are commuted forward and cost no time.
* Data blocks (§2.1–2.3):

  ============  =====================  ==================================
  block         tiles for n qubits     steps per π/8 rotation
  ============  =====================  ==================================
  compact       1.5n + 3               ≤ 9 = Y-fix + 3 (upper row) + 3 (lower row) + 1
  intermediate  2n + 4                 ≤ 5 = Y-fix + 2 (rotate/move) + 1
  fast          2n + √(8n) + 1         1
  ============  =====================  ==================================

  The per-rotation cost applies the paper's case rules instead of always
  charging the worst case: Y-fix is 0 without Y, 1 for an odd and 2 for an
  even (non-zero) number of Y's (explicit π/4 rotations, §2.1/Fig. 10);
  compact blocks expose only Z edges, so each row containing an X/Y qubit pays
  one 3-step patch rotation (Fig. 11a), rows rotate in parallel within a row;
  the intermediate block pays 2 steps if any X/Y qubit needs access (§2.2).
  Qubit ``i`` sits in the upper compact row for ``i < ceil(n/2)`` (the paper
  does not fix a placement).  The maximum of each rule equals the published
  worst case.
* Magic states follow the matched-IR assumption shared with HARVEST and Silva
  EAF: they are free and always ready; one tile is charged for the |m⟩ port.
  Distillation blocks (§3) are therefore not added.  The fast-block tile count
  is ``ceil(2n + √(8n) + 1)``; the paper's own n=100 example uses 231 tiles
  instead of the formula's 229.3 because it rounds the square arrangement.
"""

from __future__ import annotations

import math
import random
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .cultivation import cultivation_mean, sample_cycles
from .metrics import space_time_volume

VARIANTS = ("litinski_compact", "litinski_intermediate", "litinski_fast")
MAGIC_PORT_TILES = 1
# Published worst-case steps per rotation (§2.1–2.3).
WORST_CASE_STEPS = {"litinski_compact": 9, "litinski_intermediate": 5, "litinski_fast": 1}


def cultivation_ports(variant: str, mean_cycles: float) -> int:
    """Ports needed to keep the block busy on average (§4: size supply to the block)."""
    return max(1, math.ceil(mean_cycles / WORST_CASE_STEPS[variant]))


def schedule_with_cultivation(
    costs: List[int], ports: int, mean_cycles: float, seed: Optional[int]
) -> Tuple[int, int]:
    """Sequential rotations fed by ``ports`` cultivating ports; returns (steps, wait).

    A rotation starts when the previous one ended and some port holds a state;
    the port restarts cultivation at that moment (common cultivation model).
    """
    rng = random.Random(0 if seed is None else int(seed))
    ready = [0] * ports
    now = waited = 0
    for cost in costs:
        port = min(range(ports), key=lambda i: ready[i])
        start = max(now, ready[port])
        waited += start - now
        ready[port] = start + sample_cycles(rng, mean_cycles)
        now = start + cost
    return now, waited
_TRANS_LINE = re.compile(r"^([+-])([XYZ_]+)<([A-Za-z]+)>$")


def block_tiles(variant: str, num_qubits: int) -> int:
    """Data-block tile count from the paper's closed forms (§2.1–2.3)."""
    n = int(num_qubits)
    if n < 1:
        raise ValueError("num_qubits must be positive")
    if variant == "litinski_compact":
        return math.ceil(1.5 * n) + 3
    if variant == "litinski_intermediate":
        return 2 * n + 4
    if variant == "litinski_fast":
        return math.ceil(2 * n + math.sqrt(8 * n) + 1)
    raise ValueError(f"unknown Litinski variant {variant!r}")


def _y_fix_steps(paulis: str) -> int:
    ys = paulis.count("Y")
    if ys == 0:
        return 0
    return 1 if ys % 2 else 2


def rotation_steps(variant: str, paulis: str) -> int:
    """Time steps for one π/8 rotation on the Pauli word ``paulis`` (``_`` = I)."""
    if not paulis or set(paulis) - set("XYZ_") or set(paulis) <= {"_"}:
        raise ValueError(f"invalid Pauli word {paulis!r}")
    if variant == "litinski_fast":
        return 1
    needs_x = [i for i, p in enumerate(paulis) if p in "XY"]
    if variant == "litinski_intermediate":
        return _y_fix_steps(paulis) + (2 if needs_x else 0) + 1
    if variant == "litinski_compact":
        upper = math.ceil(len(paulis) / 2)
        rotations = 3 * any(i < upper for i in needs_x) + 3 * any(i >= upper for i in needs_x)
        return _y_fix_steps(paulis) + rotations + 1
    raise ValueError(f"unknown Litinski variant {variant!r}")


def parse_pauli_stream(trans_text: str) -> List[str]:
    """Pauli words of a ``.trans`` stream; only ``<T>`` rotations are accepted."""
    words = []
    for lineno, raw in enumerate(trans_text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = _TRANS_LINE.match(line)
        if match is None or match.group(3) != "T":
            raise ValueError(f"line {lineno} is not a π/8 rotation: {raw!r}")
        words.append(match.group(2))
    if not words:
        raise ValueError("empty Pauli-product stream")
    return words


def evaluate(variant: str, words: List[str], num_qubits: int) -> Tuple[int, int, Dict[str, int]]:
    """Return ``(tiles, steps, step histogram)`` for one data block."""
    histogram: Dict[str, int] = {}
    steps = 0
    for word in words:
        if len(word) != num_qubits:
            raise ValueError("Pauli word width does not match the register")
        cost = rotation_steps(variant, word)
        histogram[str(cost)] = histogram.get(str(cost), 0) + 1
        steps += cost
    return block_tiles(variant, num_qubits) + MAGIC_PORT_TILES, steps, histogram


class LitinskiAdapter(BaselineAdapter):
    """Analytic Litinski data blocks evaluated on the matched Pauli-product stream."""

    name = "litinski"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        input_repr = "Pauli-product rotations (±pi/8), analytic data-block model"
        # Both regimes see the same input: the paper's own front end is the
        # Litinski transformation of the source (Pauli-based computation), which
        # is exactly HARVEST's matched Pauli-product stream.  matched_architecture
        # (a HARVEST layout) does not apply to this model.
        if config.comparison_mode not in ("matched_ir", "native_end_to_end"):
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="Litinski data blocks are only qualified for matched_ir and native_end_to_end",
                seed=None,
            )
        if config.variant not in VARIANTS:
            raise ValueError(f"unknown Litinski variant {config.variant!r}")
        dag = kwargs.get("dag")
        if dag is None:
            raise ValueError("LitinskiAdapter.run requires dag=")
        from .puremagic import UnsupportedPureMagicIR, pauli_dag_to_trans

        started = perf_counter()
        try:
            words = parse_pauli_stream(pauli_dag_to_trans(dag, config.num_qubits))
        except (UnsupportedPureMagicIR, ValueError) as exc:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"Litinski matched-IR requires a ±pi/8 Pauli-product stream: {exc}",
                notes=["Unsupported operations were not discarded or rewritten."],
                seed=None,
            )
        tiles, steps, histogram = evaluate(config.variant, words, int(config.num_qubits))
        mean = cultivation_mean(config.options)
        ports, magic_wait = MAGIC_PORT_TILES, 0
        if mean is not None:
            ports = cultivation_ports(config.variant, mean)
            costs = [rotation_steps(config.variant, w) for w in words]
            steps, magic_wait = schedule_with_cultivation(costs, ports, mean, config.seed)
            tiles = block_tiles(config.variant, int(config.num_qubits)) + ports
        runtime = perf_counter() - started
        return BaselineResult(
            baseline=config.baseline,
            variant=config.variant,
            circuit=config.circuit,
            num_qubits=config.num_qubits,
            input_representation=input_repr,
            comparison_mode=config.comparison_mode,
            schedule_length=steps,
            logical_patches=tiles,
            space_time_volume=space_time_volume(tiles, steps),
            routing_wirelength=None,
            compiler_runtime_s=runtime,
            completed=True,
            trial=config.trial,
            seed=config.seed if mean is not None else None,
            upstream_version="Litinski, Quantum 3, 128 (2019), arXiv:1808.02892v3 §2",
            command="analytic evaluation (harvest.baselines.litinski)",
            notes=[
                "Analytic resource model, not a compiler: rotations execute sequentially.",
                "Per-rotation steps follow the paper's case rules; their maximum is the published worst case.",
                (
                    "Magic states are free (matched-IR assumption); one |m> port tile is charged."
                    if mean is None
                    else f"Common cultivation model (mean {mean} cycles) with {ports} cultivation port(s) "
                    "sized to the block's worst-case rate (§4), counted as tiles."
                ),
                "Compact-block row assignment (first ceil(n/2) qubits in the upper row) is not fixed by the paper.",
            ],
            initial_logical_patches=tiles,
            metadata={
                "block_tiles": block_tiles(config.variant, int(config.num_qubits)),
                "magic_port_tiles": ports,
                "magic_wait_steps": magic_wait,
                "num_rotations": len(words),
                "steps_per_rotation_histogram": histogram,
                "space_time_definition": "(data-block tiles + magic port) * sequential steps",
            },
        )
