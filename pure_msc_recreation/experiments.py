from __future__ import annotations
from typing import Dict, List, Set
import random
from models import PauliProduct, PauliTerm

def make_synthetic_products(n_qubits: int, n_products: int, avg_weight: int, seed: int = 0) -> Dict[int, PauliProduct]:
    rng = random.Random(seed)
    products: Dict[int, PauliProduct] = {}
    last_on_qubit = [-1] * n_qubits

    for pid in range(n_products):
        k = max(1, int(rng.expovariate(1.0 / avg_weight)))
        qs = rng.sample(range(n_qubits), min(k, n_qubits))
        terms = [PauliTerm(q, rng.choice(["X","Y","Z"])) for q in qs]

        deps: Set[int] = set()
        # simple dependency: must follow last op on any involved qubit
        for q in qs:
            if last_on_qubit[q] != -1:
                deps.add(last_on_qubit[q])
            last_on_qubit[q] = pid

        products[pid] = PauliProduct(pid=pid, terms=terms, deps=deps)

    return products
