"""Subprocess adapter for the official BQSKit/PureMagic implementation."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from math import isclose, pi
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from harvest.compilation.utils import get_pauli_type_for_qubit

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .metrics import space_time_volume

_SCHEDULE_RE = re.compile(
    r"Scheduled\s+\d+\s+in\s+(\d+)\s+logical cycles(?:,\s*volume\s+(\d+))?",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(
    r"PureMagic\s+-\s+Git branch:\s*([^|]+)\|\s*Commit:\s*([^|\s]+)",
    re.IGNORECASE,
)


class UnsupportedPureMagicIR(ValueError):
    """Raised when exact conversion to PureMagic's `.trans` IR is impossible."""


def _operator_coefficient(operator: Any) -> complex:
    coeffs = getattr(operator, "coeffs", None)
    if coeffs is None:
        return 1.0 + 0.0j
    values = list(coeffs)
    if len(values) != 1:
        raise UnsupportedPureMagicIR(
            f"expected one Pauli term, found {len(values)} terms"
        )
    return complex(values[0])


def pauli_dag_to_trans(dag, num_qubits: Optional[int] = None) -> str:
    """Convert an exact ±π/8 HARVEST PPR DAG into PureMagic `.trans`.

    Unsupported operations and non-T rotation angles raise instead of being
    dropped.  The sign is retained even though routing cost is sign-invariant.
    """
    if num_qubits is None:
        num_qubits = len(dag.qubits)
    lines: List[str] = []

    for index, node in enumerate(dag.topological_op_nodes()):
        if node.op.name not in {"PauliEvolution", "PauliProductMeasurement"}:
            raise UnsupportedPureMagicIR(
                f"operation #{index} ({node.op.name}) has no exact `.trans` mapping"
            )

        qindices = [dag.find_bit(qubit).index for qubit in node.qargs]
        paulis = ["_"] * num_qubits
        for qindex in qindices:
            pauli = get_pauli_type_for_qubit(node, qindex, qindices)
            if pauli not in {"X", "Y", "Z"}:
                raise UnsupportedPureMagicIR(
                    f"operation #{index} contains unsupported Pauli {pauli!r}"
                )
            paulis[qindex] = pauli

        coefficient = _operator_coefficient(getattr(node.op, "operator", None))
        if not isclose(coefficient.imag, 0.0, abs_tol=1e-12) or not isclose(
            abs(coefficient.real), 1.0, abs_tol=1e-12
        ):
            raise UnsupportedPureMagicIR(
                f"operation #{index} has non-unit-real coefficient {coefficient}"
            )

        if node.op.name == "PauliProductMeasurement":
            sign = "+" if coefficient.real > 0 else "-"
            tag = "M"
        else:
            if not node.op.params:
                raise UnsupportedPureMagicIR(
                    f"operation #{index} has no rotation angle"
                )
            try:
                angle = float(node.op.params[0])
            except (TypeError, ValueError) as exc:
                raise UnsupportedPureMagicIR(
                    f"operation #{index} has a symbolic rotation angle"
                ) from exc
            effective_angle = angle * coefficient.real
            if not isclose(abs(effective_angle), pi / 8, rel_tol=1e-9, abs_tol=1e-10):
                raise UnsupportedPureMagicIR(
                    f"operation #{index} angle {effective_angle} is not ±pi/8"
                )
            sign = "+" if effective_angle > 0 else "-"
            tag = "T"

        lines.append(f"{sign}{''.join(paulis)}<{tag}>")

    if not lines:
        raise UnsupportedPureMagicIR(
            "the transformed circuit contains no schedulable products"
        )
    return "\n".join(lines) + "\n"


def write_trans_file(dag, path: Path, num_qubits: Optional[int] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pauli_dag_to_trans(dag, num_qubits=num_qubits))
    return path


def parse_puremagic_schedule(path: Path) -> Dict[str, Optional[int]]:
    """Parse invariant metrics from a native `.schedule` header."""
    metrics: Dict[str, Optional[int]] = {
        "schedule_length": None,
        "active_logical_cycles": None,
    }
    for line in path.read_text().splitlines():
        if match := re.match(r"#\s*Total logical cycles:\s*(\d+)", line):
            metrics["schedule_length"] = int(match.group(1))
        elif match := re.match(r"#\s*Total active logical cycles:\s*(\d+)", line):
            metrics["active_logical_cycles"] = int(match.group(1))
    return metrics


def parse_puremagic_output(
    stdout: str, schedule_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Parse PureMagic stdout, with the schedule file as an independent check."""
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
    parsed: Dict[str, Any] = {
        "schedule_length": None,
        "volume": None,
        "data_qubits": None,
        "bus_qubits": None,
        "magic_qubits": None,
        "total_qubits": None,
        "upstream_version": None,
    }
    if match := _SCHEDULE_RE.search(clean):
        parsed["schedule_length"] = int(match.group(1))
        parsed["volume"] = int(match.group(2)) if match.group(2) else None
    if match := _VERSION_RE.search(clean):
        parsed["upstream_version"] = (
            f"branch={match.group(1).strip()}, commit={match.group(2).strip()}"
        )

    in_qubits = False
    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if line == "Number of qubits:":
            in_qubits = True
            continue
        if not in_qubits:
            continue
        for label, key in (
            ("data", "data_qubits"),
            ("bus", "bus_qubits"),
            ("magic", "magic_qubits"),
            ("total", "total_qubits"),
        ):
            if match := re.match(rf"{label}:\s+(\d+)", line):
                parsed[key] = int(match.group(1))
                if label == "total":
                    in_qubits = False
                break

    if schedule_path is not None and schedule_path.exists():
        schedule = parse_puremagic_schedule(schedule_path)
        file_length = schedule["schedule_length"]
        if parsed["schedule_length"] is None:
            parsed["schedule_length"] = file_length
        elif file_length is not None and file_length != parsed["schedule_length"]:
            raise ValueError(
                "PureMagic stdout/schedule mismatch: "
                f"{parsed['schedule_length']} != {file_length}"
            )
        parsed["active_logical_cycles"] = schedule["active_logical_cycles"]
    return parsed


def _resolve_executable(path: Optional[Path], default: str) -> Optional[str]:
    candidate = str(path) if path is not None else default
    if Path(candidate).is_file():
        return str(Path(candidate).resolve())
    return shutil.which(candidate)


def _run_command(
    command: List[str], cwd: Path, timeout_s: float
) -> Tuple[subprocess.CompletedProcess, float]:
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return completed, perf_counter() - started


def _find_newest(directory: Path, patterns: Iterable[str]) -> Optional[Path]:
    matches = [path for pattern in patterns for path in directory.glob(pattern)]
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None


def _validate_clifford_t_qasm(path: Path) -> None:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit.from_qasm_file(str(path))
    allowed = {
        "id",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "cx",
        "cz",
        "cy",
        "swap",
        "iswap",
        "ecr",
        "dcx",
        "t",
        "tdg",
        "barrier",
        "measure",
        "reset",
    }
    unsupported = sorted(set(circuit.count_ops()) - allowed)
    if unsupported:
        raise UnsupportedPureMagicIR(
            "native PureMagic transpilation requires Clifford+T input unless a "
            f"compiler is configured; unsupported gates: {unsupported}"
        )


class PureMagicAdapter(BaselineAdapter):
    name = "puremagic"

    def _prepare_input(
        self,
        config: BaselineRunConfig,
        work_dir: Path,
        commands: List[str],
        stdout_parts: List[str],
        stderr_parts: List[str],
    ) -> Tuple[Path, float]:
        runtime = 0.0
        source = config.input_path.resolve()
        input_format = str(
            config.options.get("input_format", source.suffix.lstrip("."))
        ).lower()

        if input_format == "trans" or source.suffix.startswith(".trans"):
            target = work_dir / source.name
            if source != target.resolve():
                shutil.copy2(source, target)
            return target, runtime

        if input_format not in {"qasm", "qasm2"}:
            raise UnsupportedPureMagicIR(
                f"unsupported PureMagic input format: {input_format}"
            )

        local_qasm = work_dir / source.name
        if source != local_qasm.resolve():
            shutil.copy2(source, local_qasm)
        compile_bin = config.options.get("compile_bin")
        if compile_bin:
            resolved = _resolve_executable(Path(compile_bin), str(compile_bin))
            if resolved is None:
                raise FileNotFoundError(
                    f"PureMagic Clifford+T compiler not found: {compile_bin}"
                )
            command = [resolved, local_qasm.name]
            commands.append(shlex.join(command))
            proc, elapsed = _run_command(command, work_dir, config.timeout_s)
            runtime += elapsed
            stdout_parts.append(proc.stdout)
            stderr_parts.append(proc.stderr)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"PureMagic Clifford+T compilation failed ({proc.returncode}): {proc.stderr.strip()}"
                )
            compiled = _find_newest(work_dir, ["*.cliffordt.qasm"])
            if compiled is None:
                raise RuntimeError(
                    "PureMagic compiler produced no .cliffordt.qasm file"
                )
        else:
            _validate_clifford_t_qasm(local_qasm)
            # PureMagic's `transpile` binary requires the `.cliffordt.qasm`
            # suffix as a marker that a Clifford+T compiler already ran; it
            # is a filename convention, not a content check
            # (transpile.rs: `ends_with(".cliffordt.qasm")`).  We validate
            # the gate set ourselves above, so it is safe to rename here.
            if not local_qasm.name.endswith(".cliffordt.qasm"):
                renamed = work_dir / f"{local_qasm.stem}.cliffordt.qasm"
                local_qasm.replace(renamed)
                local_qasm = renamed
            compiled = local_qasm

        transpile_bin = config.options.get("transpile_bin")
        resolved_transpile = _resolve_executable(
            Path(transpile_bin) if transpile_bin else None,
            str(transpile_bin or "transpile"),
        )
        if resolved_transpile is None:
            raise FileNotFoundError(
                "PureMagic transpile executable not found; configure --puremagic-transpile-bin"
            )
        command = [resolved_transpile, "-i", compiled.name]
        old_trans_files = {
            path
            for pattern in ("*.trans", "*.transauto", "*.trans[0-9]*")
            for path in work_dir.glob(pattern)
            if path.is_file()
        }
        for old in old_trans_files:
            old.unlink()
        commands.append(shlex.join(command))
        proc, elapsed = _run_command(command, work_dir, config.timeout_s)
        runtime += elapsed
        stdout_parts.append(proc.stdout)
        stderr_parts.append(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"PureMagic transpilation failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        trans = _find_newest(work_dir, ["*.trans", "*.transauto", "*.trans[0-9]*"])
        if trans is None:
            raise RuntimeError("PureMagic transpiler produced no .trans file")
        return trans, runtime

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        del kwargs
        work_dir = config.work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        work_dir = work_dir.resolve()
        commands: List[str] = []
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        runtime = 0.0
        input_repr = "PureMagic .trans Pauli-product stream"

        executable = _resolve_executable(config.executable, "puremagic")
        if executable is None:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="PureMagic executable not found",
                notes=[
                    "Install/build the official BQSKit/PureMagic QCE snapshot and pass --puremagic-bin."
                ],
            )

        overall_started = perf_counter()
        try:
            trans_path, prep_runtime = self._prepare_input(
                config, work_dir, commands, stdout_parts, stderr_parts
            )
            runtime += prep_runtime

            command = [
                executable,
                "--circuit",
                trans_path.name,
                "--magic-state-lambda",
                str(config.options.get("magic_state_lambda", 0.0387396)),
                "--rseed",
                str(config.seed if config.seed is not None else 29),
                "--ancilla-rows",
                str(config.options.get("ancilla_rows", 1)),
            ]
            if config.variant == "puremagic":
                command.append("--use-magic-routing")
            elif config.variant != "puremagic_bus":
                raise ValueError(f"unknown PureMagic variant: {config.variant}")
            if config.options.get("no_t_failures", False):
                command.append("--no-t-failures")
            topology = config.options.get("topology")
            if topology:
                topology_source = Path(topology).resolve()
                if not topology_source.is_file():
                    raise FileNotFoundError(
                        f"PureMagic topology file not found: {topology_source}"
                    )
                topology_copy = work_dir / topology_source.name
                if topology_source != topology_copy:
                    shutil.copy2(topology_source, topology_copy)
                command.extend(["--topo", topology_copy.name])
            else:
                topology_copy = None

            for old in work_dir.glob("*.schedule"):
                if old.is_file():
                    old.unlink()
            # Record the exact invocation before execution so timeouts retain
            # the command that was attempted.
            commands.append(shlex.join(command))
            proc, elapsed = _run_command(command, work_dir, config.timeout_s)
            runtime += elapsed
            stdout_parts.append(proc.stdout)
            stderr_parts.append(proc.stderr)
            stdout = "\n".join(part for part in stdout_parts if part)
            stderr = "\n".join(part for part in stderr_parts if part)
            (work_dir / "stdout.txt").write_text(stdout)
            (work_dir / "stderr.txt").write_text(stderr)
            (work_dir / "command.txt").write_text("\n".join(commands) + "\n")
            runtime = perf_counter() - overall_started
            if proc.returncode != 0:
                return BaselineResult.failed(
                    config,
                    input_representation=input_repr,
                    error=f"PureMagic exited with code {proc.returncode}",
                    command=" && ".join(commands),
                    runtime_s=runtime,
                    notes=[stderr.strip()] if stderr.strip() else [],
                )

            schedule_path = _find_newest(work_dir, ["*.schedule"])
            parsed = parse_puremagic_output(stdout, schedule_path)
            length = parsed["schedule_length"]
            patches = parsed["total_qubits"]
            if length is None or patches is None:
                raise ValueError(
                    "PureMagic completed but schedule length or complete topology size was absent"
                )
            computed_volume = space_time_volume(patches, length)
            if parsed.get("volume") is not None and parsed["volume"] != computed_volume:
                raise ValueError(
                    f"PureMagic volume mismatch: native={parsed['volume']} normalized={computed_volume}"
                )
            runtime = perf_counter() - overall_started
            reported_upstream = parsed.get("upstream_version")
            upstream = reported_upstream or (
                f"unreported by executable; requested={config.upstream_version}"
            )
            notes = [
                "Logical patch count includes data, bus, and cultivation/magic patches.",
                "Native output does not expose a comparable aggregate routing wirelength.",
                "Peak memory is null because the native CLI output does not report it.",
            ]
            if reported_upstream is None:
                notes.append(
                    "The executable emitted no parseable version banner; the requested snapshot "
                    "is not treated as a verified version."
                )
            elif "QCE" not in reported_upstream:
                notes.append(
                    "The executable did not identify itself as the requested QCE paper snapshot; "
                    "the exact banner/commit is recorded."
                )
            return BaselineResult(
                baseline=config.baseline,
                variant=config.variant,
                circuit=config.circuit,
                num_qubits=config.num_qubits,
                input_representation=input_repr,
                comparison_mode=config.comparison_mode,
                schedule_length=length,
                logical_patches=patches,
                space_time_volume=computed_volume,
                routing_wirelength=None,
                compiler_runtime_s=runtime,
                completed=True,
                trial=config.trial,
                seed=config.seed if config.seed is not None else 29,
                upstream_version=upstream,
                command=" && ".join(commands),
                notes=notes,
                initial_logical_patches=patches,
                metadata={
                    "native_metrics": parsed,
                    "magic_state_lambda": config.options.get(
                        "magic_state_lambda", 0.0387396
                    ),
                    "ancilla_rows": config.options.get("ancilla_rows", 1),
                    "t_injection_failures": not config.options.get(
                        "no_t_failures", False
                    ),
                    "random_seed": config.seed if config.seed is not None else 29,
                    "topology": str(topology_copy)
                    if topology_copy
                    else "PureMagic native default",
                    "space_time_definition": "reported_total_qubits * logical_cycles",
                    "work_dir": str(work_dir),
                    "schedule_file": str(schedule_path) if schedule_path else None,
                },
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                stdout_parts.append(str(exc.stdout))
            if exc.stderr:
                stderr_parts.append(str(exc.stderr))
            stdout = "\n".join(part for part in stdout_parts if part)
            stderr = "\n".join(part for part in stderr_parts if part)
            (work_dir / "stdout.txt").write_text(stdout)
            (work_dir / "stderr.txt").write_text(stderr)
            (work_dir / "command.txt").write_text("\n".join(commands) + "\n")
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"PureMagic timed out after {config.timeout_s}s",
                command=" && ".join(commands) or None,
                timed_out=True,
                runtime_s=perf_counter() - overall_started,
            )
        except (OSError, RuntimeError, UnsupportedPureMagicIR, ValueError) as exc:
            stdout = "\n".join(part for part in stdout_parts if part)
            stderr = "\n".join(part for part in stderr_parts if part)
            (work_dir / "stdout.txt").write_text(stdout)
            (work_dir / "stderr.txt").write_text(stderr)
            (work_dir / "command.txt").write_text("\n".join(commands) + "\n")
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=str(exc),
                command=" && ".join(commands) or None,
                runtime_s=perf_counter() - overall_started,
                notes=["Unsupported operations were not silently discarded."],
            )
