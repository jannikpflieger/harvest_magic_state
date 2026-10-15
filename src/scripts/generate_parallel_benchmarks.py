#!/usr/bin/env python3
"""Generate log-depth (highly parallel) reversible benchmark circuits.

The Feynman suite is dominated by near-serial circuits (Pauli products per
critical-path step ~1.1-2.2); only ``qcla_*`` and ``csum_mux_9`` are wide.
These two textbook families fill that regime with tunable size:

* ``ks_adder_n<N>`` -- out-of-place Kogge-Stone prefix adder ``s = a + b``
  (N-bit inputs, N+1-bit sum).  Generate/propagate prefixes are computed in
  ceil(log2 N) levels of Toffolis into fresh ancillas, the sum is copied out,
  and all ancillas and ``b`` are uncomputed (Bennett), so the circuit is clean.
* ``mcx_tree_n<N>`` -- N-control Toffoli via a balanced AND tree of Toffolis
  into ancillas (log-depth compute, one CX to the target, uncompute).

Gates are ``x``, ``cx`` and ``ccx`` only; every circuit is checked by
classical simulation on random (or, when small, all) basis inputs before it
is written.  Lower the outputs to Clifford+T with
``prepare_cliffordt_benchmarks.py`` (exact 7-T Toffoli).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit, QuantumRegister, qasm2


def simulate(circuit: QuantumCircuit, bits: Dict[int, int]) -> Dict[int, int]:
    state = dict(bits)
    for instruction in circuit.data:
        q = [circuit.find_bit(b).index for b in instruction.qubits]
        name = instruction.operation.name
        if name == "x":
            state[q[0]] ^= 1
        elif name == "cx":
            state[q[1]] ^= state[q[0]]
        elif name == "ccx":
            state[q[2]] ^= state[q[0]] & state[q[1]]
        else:
            raise ValueError(f"unexpected gate {name}")
    return state


def ks_adder(n: int) -> QuantumCircuit:
    a, b, s = QuantumRegister(n, "a"), QuantumRegister(n, "b"), QuantumRegister(n + 1, "sum")
    circuit = QuantumCircuit(a, b, s, name=f"ks_adder_n{n}")
    ancillas: List[QuantumRegister] = []

    def fresh(size: int, name: str) -> QuantumRegister:
        reg = QuantumRegister(size, name)
        circuit.add_register(reg)
        ancillas.append(reg)
        return reg

    compute = QuantumCircuit(*circuit.qregs)  # filled below, then appended and inverted
    g0 = fresh(n, "g0")
    compute.add_register(g0)
    for i in range(n):
        compute.ccx(a[i], b[i], g0[i])  # generate
        compute.cx(a[i], b[i])          # propagate, in place: b_i := a_i xor b_i
    # G[i] / P[i] of the span ending at bit i; P of level 0 is b itself.
    G: Dict[int, object] = {i: g0[i] for i in range(n)}
    P: Dict[int, object] = {i: b[i] for i in range(n)}
    level, d = 1, 1
    while d < n:
        targets = list(range(d, n))
        g_new = fresh(len(targets), f"g{level}")
        compute.add_register(g_new)
        needs_p = 2 * d < n
        p_new = fresh(len(targets), f"p{level}") if needs_p else None
        if p_new is not None:
            compute.add_register(p_new)
        nextG, nextP = dict(G), dict(P)
        for k, i in enumerate(targets):
            # G_hi and P_hi are mutually exclusive, so OR == XOR.
            compute.cx(G[i], g_new[k])
            compute.ccx(P[i], G[i - d], g_new[k])
            nextG[i] = g_new[k]
            if p_new is not None:
                compute.ccx(P[i], P[i - d], p_new[k])
                nextP[i] = p_new[k]
        G, P = nextG, nextP
        level, d = level + 1, 2 * d
    circuit.compose(compute, inplace=True)
    # carry into bit i+1 is G[i]; s_i = p_i xor c_i
    for i in range(n):
        circuit.cx(b[i], s[i])
        if i > 0:
            circuit.cx(G[i - 1], s[i])
    circuit.cx(G[n - 1], s[n])
    circuit.compose(compute.inverse(), inplace=True)
    return circuit


def mcx_tree(n: int) -> QuantumCircuit:
    controls, target = QuantumRegister(n, "ctl"), QuantumRegister(1, "tgt")
    anc = QuantumRegister(n - 2, "anc")
    circuit = QuantumCircuit(controls, target, anc, name=f"mcx_tree_n{n}")
    compute = QuantumCircuit(controls, target, anc)
    layer, used = list(controls), 0
    while len(layer) > 2:
        nxt = []
        for k in range(0, len(layer) - 1, 2):
            compute.ccx(layer[k], layer[k + 1], anc[used])
            nxt.append(anc[used])
            used += 1
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    circuit.compose(compute, inplace=True)
    circuit.ccx(layer[0], layer[1], target[0])
    circuit.compose(compute.inverse(), inplace=True)
    return circuit


def check(circuit: QuantumCircuit, expected, inputs: Sequence[Tuple[str, int]], trials: int,
          rng: random.Random) -> None:
    """Simulate on basis inputs; ``expected`` maps register values to output values."""
    regs = {r.name: r for r in circuit.qregs}
    widths = [(name, regs[name].size) for name, _ in inputs]
    total = sum(w for _, w in widths)
    cases = range(2 ** total) if total <= 12 else (rng.getrandbits(total) for _ in range(trials))
    for case in cases:
        values, shift = {}, 0
        for name, width in widths:
            values[name] = (case >> shift) & ((1 << width) - 1)
            shift += width
        bits = {circuit.find_bit(q).index: 0 for q in circuit.qubits}
        for name, value in values.items():
            for i, q in enumerate(regs[name]):
                bits[circuit.find_bit(q).index] = (value >> i) & 1
        out = simulate(circuit, bits)
        want = expected(values)
        for reg in circuit.qregs:
            got = sum(out[circuit.find_bit(q).index] << i for i, q in enumerate(reg))
            if got != want.get(reg.name, values.get(reg.name, 0)):
                raise AssertionError(f"{circuit.name}: register {reg.name} = {got} for {values}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks-sizes", type=int, nargs="*", default=[4, 6, 8, 12, 16])
    parser.add_argument("--mcx-sizes", type=int, nargs="*", default=[8, 16, 24, 32])
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circuits = []
    for n in args.ks_sizes:
        c = ks_adder(n)
        check(c, lambda v: {"sum": v["a"] + v["b"]}, [("a", n), ("b", n)], args.trials, rng)
        circuits.append(c)
    for n in args.mcx_sizes:
        c = mcx_tree(n)
        full = (1 << n) - 1
        check(c, lambda v, full=full: {"tgt": v["tgt"] ^ int(v["ctl"] == full)},
              [("ctl", n), ("tgt", 1)], args.trials, rng)
        # random inputs almost never hit the all-ones control; check it explicitly
        bits = {c.find_bit(q).index: 1 for q in c.qregs[0]}
        bits.update({c.find_bit(q).index: 0 for r in c.qregs[1:] for q in r})
        out = simulate(c, bits)
        assert out[c.find_bit(c.qregs[1][0]).index] == 1, f"{c.name}: all-ones case"
        assert all(out[c.find_bit(q).index] == 0 for q in c.qregs[2]), f"{c.name}: dirty ancilla"
        circuits.append(c)
    for c in circuits:
        path = args.output_dir / f"{c.name}.qasm"
        path.write_text(qasm2.dumps(c))
        ops = dict(c.count_ops())
        print(f"{path}: {c.num_qubits} qubits, ops {ops}, depth {c.depth()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
