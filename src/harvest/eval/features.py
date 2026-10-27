"""Per-circuit features of the matched Pauli-product IR (EVAL_PLAN E2), cached by input SHA-256."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

from .manifest import file_sha256

FEATURES = ("num_qubits", "ops", "critical_path", "parallelism", "t_fraction",
            "mean_pauli_weight", "y_fraction", "source_gates", "source_t_gates")


def circuit_features(source_circuit, dag) -> Dict[str, Any]:
    from .instrument import critical_path
    from .validate import pauli_factors

    ops = list(dag.op_nodes())
    weights = []
    y_factors = 0
    for node in ops:
        factors = pauli_factors(dag, node)
        weights.append(len(factors))
        y_factors += sum(1 for p in factors.values() if p == "Y")
    counts = source_circuit.count_ops()
    gates = sum(v for k, v in counts.items() if k not in ("barrier", "measure"))
    t_gates = counts.get("t", 0) + counts.get("tdg", 0)
    cp = critical_path(dag)
    factors_total = sum(weights)
    return {
        "num_qubits": source_circuit.num_qubits,
        "ops": len(ops),
        "critical_path": cp,
        "parallelism": len(ops) / cp if cp else None,
        # Share of T/Tdg gates in the Clifford+T source; every operation of the
        # lowered stream is a pi/8 rotation, so this is measured on the source.
        "t_fraction": t_gates / gates if gates else None,
        "mean_pauli_weight": factors_total / len(ops) if ops else None,
        "y_fraction": y_factors / factors_total if factors_total else None,
        "source_gates": gates,
        "source_t_gates": t_gates,
    }


def load_features(
    paths: Iterable[Path], load: Callable[[Path], Any], cache_path: Path
) -> Dict[str, Dict[str, Any]]:
    """``{circuit stem: features}``; recomputes only inputs whose SHA-256 is new."""
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    out, dirty = {}, False
    for path in paths:
        path = Path(path)
        sha = file_sha256(str(path.resolve()))
        key = f"{path.stem}|{sha}"
        if key not in cache:
            circuit, dag = load(path)
            cache[key] = {"circuit": path.stem, "input_sha256": sha,
                          **circuit_features(circuit, dag)}
            dirty = True
        out[path.stem] = cache[key]
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return out
