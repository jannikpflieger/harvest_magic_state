"""Exact routes of a schedule, exported next to each raw result.

A raw result keeps a compact ``metadata.schedule`` (one entry per product:
time step, qubits with their Pauli factors, magic terminal, edge count).  The
full route of every product (data ports, routing nodes and edges of its tree)
goes to a gzipped sidecar ``<raw record stem>.routes.json.gz``: on large
circuits it is several times the size of the record itself.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .validate import pauli_factors

ROUTES_KEY = "routes"
SUFFIX = ".routes.json.gz"


def support(dag, node) -> Dict[int, str]:
    """``{qubit: 'X'|'Y'|'Z'}`` of a Pauli product (``'?'`` factors if it cannot be read)."""
    try:
        return pauli_factors(dag, node)
    except (ValueError, AttributeError):
        return {dag.find_bit(q).index: "?" for q in node.qargs}


def schedule_entry(dag, result: Dict[str, Any], **extra) -> Dict[str, Any]:
    """Compact per-product summary stored in ``metadata.schedule``."""
    factors = support(dag, result["node"]) if result.get("node") is not None else {}
    qubits = sorted(factors)
    return {
        "node": getattr(result.get("node"), "_node_id", None),
        "gate_name": result.get("gate_name"),
        "qubits": qubits,
        "paulis": "".join(factors[q] for q in qubits),
        "time_step": result.get("time_step"),
        "magic_terminal": result.get("magic_terminal"),
        "routing_edges": len(result.get("steiner_edges", ())),
        **extra,
    }


def route_entries(dag, results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Full route of every routed product, in commit order."""
    out = []
    for r in results:
        if not r.get("success", True):
            continue
        entry = schedule_entry(dag, r)
        entry.pop("routing_edges")
        entry.update({
            "qubit_terminals": [str(t) for t in r.get("qubit_terminals", ())],
            "steiner_nodes": sorted(str(n) for n in r.get("steiner_nodes", ())),
            "steiner_edges": sorted(sorted(map(str, e)) for e in r.get("steiner_edges", ())),
        })
        out.append(entry)
    return out


def sidecar_path(record_path: Path) -> Path:
    return record_path.with_name(record_path.name[: -len(".json")] + SUFFIX)


def write_routes(record_path: Path, routes: List[Dict[str, Any]]) -> Path:
    """Atomically write the gzipped route sidecar of ``record_path``."""
    path = sidecar_path(record_path)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as handle:
        json.dump(routes, handle, separators=(",", ":"))
    tmp.replace(path)
    return path


def read_routes(record_path: Path) -> List[Dict[str, Any]]:
    with gzip.open(sidecar_path(Path(record_path)), "rt") as handle:
        return json.load(handle)
