#!/usr/bin/env python3
"""Prepare a benchmark family as Clifford+T inputs for the SOTA comparison.

Every compiler of the matched-IR regime receives the same Clifford+T circuit,
so circuits with arbitrary-angle rotations (QAOA, QFT, Ising, Heisenberg, kNN)
are synthesized *once*, by one shared front end, before any compiler sees
them:

1. barriers and final measurements are removed; a circuit with a mid-circuit
   measurement or reset is refused;
2. a circuit with an exact Clifford+T rewrite (Clifford, T, CCX/CSWAP,
   phase rotations at multiples of pi/4) is lowered exactly;
3. otherwise PureMagic's ``compile_cliffordt`` synthesizes it with a fixed
   per-rotation precision ``--epsilon`` (approximate, recorded);
4. the result goes through ``prepare_cliffordt_benchmarks.lower`` (gate-set
   and T-count validation) into ``<output-dir>/<family>/``.

``--max-t`` skips circuits whose Clifford+T form would exceed a T budget; the
skip and its T count are recorded in ``skipped.json`` instead of silently
dropping the circuit.  For <= ``--fidelity-max-qubits`` qubits the process
fidelity of the synthesized circuit against the source is recorded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
for path in (PROJECT_ROOT / "src", SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.quantum_info import Operator, process_fidelity  # noqa: E402

from prepare_cliffordt_benchmarks import (  # noqa: E402
    EXACT_SOURCE_GATES,
    PHASE_ROTATIONS,
    lower,
    rewrite_phase_rotations,
    sha256,
)

EXACT_EXTRA = {"cswap"}  # Fredkin: exact via Toffoli


def strip(source: Path) -> QuantumCircuit:
    circuit = QuantumCircuit.from_qasm_file(str(source))
    circuit.remove_final_measurements(inplace=True)
    names = {i.operation.name for i in circuit.data}
    if names & {"measure", "reset"}:
        raise ValueError(f"{source.name}: mid-circuit measurement/reset")
    bare = QuantumCircuit(*circuit.qregs, name=circuit.name)
    for instruction in circuit.data:
        if instruction.operation.name != "barrier":
            bare.append(instruction)
    return bare


def is_exact(circuit: QuantumCircuit) -> bool:
    try:
        rewritten = rewrite_phase_rotations(circuit)
    except ValueError:
        return False
    return set(rewritten.count_ops()) <= EXACT_SOURCE_GATES | EXACT_EXTRA


def single_register(circuit: QuantumCircuit) -> QuantumCircuit:
    """One ``q`` register (some tools only parse ``qreg q``); gate order unchanged."""
    out = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    for instruction in circuit.data:
        out.append(instruction.operation, [circuit.find_bit(q).index for q in instruction.qubits])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synth-bin", type=Path, required=True,
                        help="PureMagic compile_cliffordt executable.")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--max-t", type=int, default=40000)
    parser.add_argument("--fidelity-max-qubits", type=int, default=8)
    args = parser.parse_args(argv)

    out_dir = args.output_dir / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, skipped_path = out_dir / "manifest.json", out_dir / "skipped.json"
    manifest = {r["output"]: r for r in json.loads(manifest_path.read_text())} \
        if manifest_path.exists() else {}
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    synth_version = subprocess.run([str(args.synth_bin), "--version"], capture_output=True,
                                   text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        for source in sorted(p.resolve() for p in args.sources):
            name = f"{source.stem}.qasm"
            previous = manifest.get(name)
            if previous and previous.get("original_sha256") == sha256(source) \
                    and previous.get("epsilon") in (None, args.epsilon) \
                    and (out_dir / name).exists():
                print(f"{source.stem}: unchanged, skipped")
                continue
            if skipped.get(source.stem, {}).get("original_sha256") == sha256(source) \
                    and skipped[source.stem].get("max_t", 0) >= args.max_t:
                print(f"{source.stem}: previously over budget, skipped")
                continue
            try:
                circuit = single_register(strip(source))
            except ValueError as exc:
                skipped[source.stem] = {"original_sha256": sha256(source), "reason": str(exc)}
                print(f"{source.stem}: refused ({exc})")
                continue
            stripped = Path(tmp) / name
            stripped.write_text(qasm2.dumps(circuit))
            exact = is_exact(circuit)
            if exact:
                to_lower, epsilon = stripped, None
            else:
                to_lower, epsilon = Path(tmp) / f"synth_{name}", args.epsilon
                subprocess.run([str(args.synth_bin), "-e", str(args.epsilon), "-o", str(to_lower),
                                str(stripped)], check=True, capture_output=True)
                to_lower.write_text(qasm2.dumps(QuantumCircuit.from_qasm_file(str(to_lower))))
            check = QuantumCircuit.from_qasm_file(str(to_lower))
            ops = check.count_ops()
            t_est = ops.get("t", 0) + ops.get("tdg", 0) + 7 * (ops.get("ccx", 0) + ops.get("cswap", 0))
            if t_est > args.max_t:
                skipped[source.stem] = {"original_sha256": sha256(source), "t_count": t_est,
                                        "max_t": args.max_t, "epsilon": epsilon,
                                        "reason": "T count over budget"}
                print(f"{source.stem}: T={t_est} > --max-t {args.max_t}, skipped")
                continue
            final = Path(tmp) / "final" / name
            final.parent.mkdir(exist_ok=True)
            final.write_text(to_lower.read_text())
            record = lower(final, out_dir, unitary_max_qubits=0)
            fidelity = None
            if circuit.num_qubits <= args.fidelity_max_qubits:
                fidelity = process_fidelity(Operator(QuantumCircuit.from_qasm_file(
                    str(out_dir / name))), Operator(circuit))
            record.update({
                "source": str(source.relative_to(PROJECT_ROOT)),
                "original_sha256": sha256(source),
                "family": args.family,
                "front_end": "exact" if exact else f"compile_cliffordt -e {args.epsilon}",
                "synthesizer_version": None if exact else synth_version,
                "epsilon": epsilon,
                "process_fidelity": fidelity,
                "stripped_ops": dict(circuit.count_ops()),
            })
            manifest[name] = record
            skipped.pop(source.stem, None)
            print(f"{source.stem}: n={record['num_qubits']} T={record['t_count']} "
                  f"{record['front_end']}" + (f" F={fidelity:.6f}" if fidelity is not None else ""))
    manifest_path.write_text(json.dumps(sorted(manifest.values(), key=lambda r: r["output"]),
                                        indent=2) + "\n")
    skipped_path.write_text(json.dumps(skipped, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
