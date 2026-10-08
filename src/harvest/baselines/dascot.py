"""Adapter for DASCOT as distributed by the official qqq-wisc/wisq CLI."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .metrics import space_time_volume


class UnsupportedDASCOTInput(ValueError):
    pass


def harvest_layout_to_wisq_architecture(layout_engine) -> Dict[str, Any]:
    """Export a compatible one-tile HARVEST layout to wisq architecture JSON.

    Every unlisted grid location is a routing resource in wisq.  Therefore the
    conversion is exact for HARVEST layouts made of one-tile data and magic
    patches with no blocked cells; multi-tile/blocked layouts are rejected.
    """
    blocked = [cell for cell, value in layout_engine.occ.items() if value == "BLOCKED"]
    if blocked:
        raise ValueError(
            "wisq custom architectures cannot encode HARVEST blocked cells"
        )

    alg_qubits: List[int] = []
    magic_states: List[int] = []
    for patch in layout_engine.patches.values():
        if len(patch.cells) != 1:
            raise ValueError(
                f"patch {patch.name!r} occupies {len(patch.cells)} tiles; "
                "matched architecture conversion requires one-tile patches"
            )
        x, y = next(iter(patch.cells))
        index = y * layout_engine.W + x
        if patch.kind.startswith("data"):
            alg_qubits.append(index)
        elif patch.kind == "magic":
            magic_states.append(index)
        else:
            raise ValueError(f"unsupported HARVEST patch kind {patch.kind!r}")

    return {
        "height": layout_engine.H,
        "width": layout_engine.W,
        "alg_qubits": sorted(alg_qubits),
        "magic_states": sorted(magic_states),
    }


def write_wisq_architecture(layout_engine, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(harvest_layout_to_wisq_architecture(layout_engine), indent=2) + "\n"
    )
    return path


def parse_wisq_output(source: Any) -> Dict[str, Any]:
    """Parse and validate wisq's ``map/steps/arch/gates`` JSON contract."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = (
            json.loads(path.read_text()) if path.exists() else json.loads(str(source))
        )
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError("wisq output must be a path, JSON string, or dict")

    missing = [key for key in ("map", "steps", "arch", "gates") if key not in data]
    if missing:
        raise ValueError(f"wisq output is incomplete; missing keys: {missing}")
    if data["steps"] == "timeout":
        raise TimeoutError("wisq reported an internal mapping/routing timeout")
    if not isinstance(data["steps"], list):
        raise ValueError("wisq 'steps' must be a list")

    arch = data["arch"]
    width = int(arch["width"])
    height = int(arch["height"])
    if width <= 0 or height <= 0:
        raise ValueError("wisq architecture dimensions must be positive")

    wirelength = 0
    routed_gates = 0
    for step in data["steps"]:
        if not isinstance(step, list):
            raise ValueError("each wisq schedule step must be a list")
        for operation in step:
            if not isinstance(operation, dict) or "path" not in operation:
                raise ValueError("wisq scheduled operation is missing its path")
            path = operation["path"]
            if not isinstance(path, list):
                raise ValueError("wisq route path must be a list")
            wirelength += max(0, len(path) - 1)
            routed_gates += 1

    return {
        "schedule_length": len(data["steps"]),
        "logical_patches": width * height,
        "routing_wirelength": wirelength,
        "routed_gates": routed_gates,
        "architecture": arch,
        "mapping": data["map"],
        "num_gates": len(data["gates"]),
        "raw": data,
    }


def validate_wisq_qasm(path: Path) -> None:
    """Reject gates that `wisq --mode scmr` would silently ignore incorrectly."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit.from_qasm_file(str(path))
    # DASCOT routes CX and T/Tdg.  The listed single-qubit Cliffords are local
    # operations in its model.  Arbitrary rotations require a prior synthesis
    # pass and are not accepted in scheduler-only SCMR mode.
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
        "t",
        "tdg",
        "barrier",
        "measure",
        "reset",
    }
    unsupported = sorted(set(circuit.count_ops()) - allowed)
    if unsupported:
        raise UnsupportedDASCOTInput(
            "wisq SCMR input must already use its supported Clifford+T/CX model; "
            f"unsupported gates: {unsupported}"
        )


def _resolve_executable(path: Optional[Path]) -> Optional[str]:
    candidate = str(path) if path is not None else "wisq"
    if Path(candidate).is_file():
        return str(Path(candidate).resolve())
    return shutil.which(candidate)


class DASCOTAdapter(BaselineAdapter):
    name = "dascot"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        layout_engine = kwargs.get("layout_engine")
        input_repr = "OpenQASM 2 Clifford+T/CX (wisq SCMR)"
        if config.comparison_mode == "matched_ir":
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="DASCOT and HARVEST do not share a scheduler-level Pauli-product IR",
                notes=[
                    "Use native_end_to_end or matched_architecture; do not label this scheduler-only."
                ],
                seed=None,
            )

        executable = _resolve_executable(config.executable)
        if executable is None:
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error="wisq executable not found",
                notes=[
                    "Install the official qqq-wisc/wisq package and pass --wisq-bin."
                ],
                seed=None,
            )

        work_dir = config.work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        work_dir = work_dir.resolve()
        output_path = work_dir / "out.json"
        if output_path.exists():
            output_path.unlink()
        command: List[str] = []
        started = perf_counter()
        try:
            validate_wisq_qasm(config.input_path)
            if config.comparison_mode == "matched_architecture":
                if layout_engine is None:
                    raise ValueError(
                        "matched_architecture requires a HARVEST layout_engine"
                    )
                architecture_path = write_wisq_architecture(
                    layout_engine, work_dir / "architecture.json"
                )
                architecture_arg = str(architecture_path)
                variant = "matched_architecture"
            else:
                architectures = {
                    "dascot_square_sparse": "square_sparse_layout",
                    "dascot_compact": "compact_layout",
                }
                if config.variant not in architectures:
                    raise ValueError(f"unknown DASCOT variant: {config.variant}")
                architecture_arg = architectures[config.variant]
                variant = config.variant

            command = [
                executable,
                str(config.input_path.resolve()),
                "--mode",
                "scmr",
                "--architecture",
                architecture_arg,
                "--output_path",
                str(output_path),
                "--mr_timeout",
                str(max(1, int(config.timeout_s))),
            ]
            proc = subprocess.run(
                command,
                cwd=work_dir,
                text=True,
                capture_output=True,
                timeout=config.timeout_s + 5.0,
                check=False,
            )
            runtime = perf_counter() - started
            (work_dir / "stdout.txt").write_text(proc.stdout)
            (work_dir / "stderr.txt").write_text(proc.stderr)
            (work_dir / "command.txt").write_text(shlex.join(command) + "\n")
            if proc.returncode != 0:
                return BaselineResult.failed(
                    config,
                    input_representation=input_repr,
                    error=f"wisq exited with code {proc.returncode}",
                    command=shlex.join(command),
                    runtime_s=runtime,
                    notes=[proc.stderr.strip()] if proc.stderr.strip() else [],
                    seed=None,
                )
            parsed = parse_wisq_output(output_path)
            length = parsed["schedule_length"]
            patches = parsed["logical_patches"]
            notes = [
                "Mapping/routing-only SCMR mode; GUOQ circuit optimization is disabled.",
                "DASCOT has no CLI seed option, so the requested seed cannot be enforced.",
                "Schedule cycles operate on CX/T dependencies, not HARVEST's PPR stream.",
                "Logical patch count is the full architecture width × height, including magic sites.",
                "Peak memory is null because wisq's JSON/CLI output does not report it.",
                "The upstream version is declared by --wisq-version; wisq result JSON does not self-identify a commit.",
            ]
            pre_synthesis_note = config.options.get("pre_synthesis_note")
            if pre_synthesis_note:
                notes.append(pre_synthesis_note)
            return BaselineResult(
                baseline=config.baseline,
                variant=variant,
                circuit=config.circuit,
                num_qubits=config.num_qubits,
                input_representation=input_repr,
                comparison_mode=config.comparison_mode,
                schedule_length=length,
                logical_patches=patches,
                space_time_volume=space_time_volume(patches, length),
                routing_wirelength=parsed["routing_wirelength"],
                compiler_runtime_s=runtime,
                completed=True,
                trial=config.trial,
                seed=None,
                upstream_version=config.upstream_version,
                command=shlex.join(command),
                notes=notes,
                initial_logical_patches=patches,
                metadata={
                    "native_metrics": {
                        key: value for key, value in parsed.items() if key != "raw"
                    },
                    "architecture_argument": architecture_arg,
                    "requested_seed": config.seed,
                    "seed_supported": False,
                    "declared_upstream_version": config.upstream_version,
                    "version_self_reported": False,
                    "space_time_definition": "architecture_width * architecture_height * schedule_length",
                    "work_dir": str(work_dir),
                    "output_file": str(output_path),
                    "pre_synthesis_command": config.options.get("pre_synthesis_command"),
                },
            )
        except subprocess.TimeoutExpired as exc:
            runtime = perf_counter() - started
            (work_dir / "stdout.txt").write_text(str(exc.stdout or ""))
            (work_dir / "stderr.txt").write_text(str(exc.stderr or ""))
            (work_dir / "command.txt").write_text(
                (shlex.join(command) + "\n") if command else ""
            )
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=f"wisq timed out after {config.timeout_s}s",
                command=shlex.join(command) if command else None,
                timed_out=True,
                runtime_s=runtime,
                seed=None,
            )
        except TimeoutError as exc:
            runtime = perf_counter() - started
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=str(exc),
                command=shlex.join(command) if command else None,
                timed_out=True,
                runtime_s=runtime,
                seed=None,
            )
        except (
            OSError,
            KeyError,
            TypeError,
            UnsupportedDASCOTInput,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            runtime = perf_counter() - started
            return BaselineResult.failed(
                config,
                input_representation=input_repr,
                error=str(exc),
                command=shlex.join(command) if command else None,
                runtime_s=runtime,
                notes=["Unsupported operations were not silently discarded."],
                seed=None,
            )
