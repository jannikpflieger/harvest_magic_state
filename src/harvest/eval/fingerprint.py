"""Canonical schedule fingerprints for the bit-identity regression gate (EVAL_PLAN B.4)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence


def canonical_schedule(results: Sequence[Dict[str, Any]]) -> List[list]:
    """Everything that defines a schedule, in a JSON-stable form.

    One entry per routed operation, in the order the scheduler committed them:
    time step, DAG node id, magic terminal, data terminals, and the sorted
    routing nodes and edges of its tree.
    """
    out = []
    for r in results:
        if not r.get("success", True):
            continue
        out.append([
            r.get("time_step"),
            getattr(r.get("node"), "_node_id", None),
            r.get("magic_terminal"),
            list(r.get("qubit_terminals", [])),
            sorted(str(n) for n in r.get("steiner_nodes", ())),
            sorted(str(sorted(map(str, e))) for e in r.get("steiner_edges", ())),
        ])
    return out


def schedule_fingerprint(results: Sequence[Dict[str, Any]], total_steps: int) -> str:
    payload = {"steps": total_steps, "schedule": canonical_schedule(results)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
