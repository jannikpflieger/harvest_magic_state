# Harvest Magic State

Lattice-surgery routing toolkit for fault-tolerant quantum computing.  
Converts quantum circuits into Pauli-block (PCB) form, maps them onto a
lattice-surgery layout, and routes magic states to non-Clifford gates using
Steiner-tree algorithms.

> **Note** — this is part of an ongoing master thesis (target: mid-June 2025).
> Reach out if you want to discuss design choices.

## Repository layout

```
src/harvest/
  compilation/      # QASM loading, PCB conversion, circuit analysis
  layout/           # Lattice-surgery patch geometry & routing graph
  routing/          # DAG scheduling + Steiner routing
  evaluation/       # Benchmark runner, metrics, plotting
  cli.py            # Unified CLI  (python -m harvest …)

scripts/            # Standalone helper scripts
tests/              # Integration / smoke tests
results/            # CSV / JSON experiment output (gitignored)
```

## Quick start

```bash
# Install in editable mode (from the repo root)
pip install -e .

# Run the pipeline on a random 20-qubit circuit
python -m harvest run --qubits 20 --depth 10 --rows 5 --cols 5

# Run a depth-sweep benchmark
python -m harvest bench --rows 5 --cols 5 --depth-start 10 --depth-end 100

# Analyze experiment results
python -m harvest analyze results/routing_experiments --plots

# Plot wirelength comparison
python -m harvest plot results/routing_experiments

# Analyze benchmark QASM circuits (gate counts, non-Clifford stats, …)
python -m harvest analyze-circuits --subdirs feynman bigint
```

## CLI reference

| Command   | Description |
|-----------|-------------|
| `run`     | Convert a circuit and route it through the lattice |
| `bench`   | Run a systematic depth-sweep experiment |
| `analyze` | Generate summary reports and performance plots |
| `plot`    | Create wirelength bar charts from results |
| `analyze-circuits` | Batch-analyze benchmark QASM files (gate counts, non-Clifford stats) |

Run `python -m harvest <command> --help` for full option lists.

## Module overview

### `harvest.compilation`
- **`qasm_loader`** — load OpenQASM 2.0 / 3.0 files
- **`pauli_block_conversion`** — Litinski transform, random circuit generation
- **`circuit_analysis`** — gate counts, DAG layer analysis, batch reporting
- **`visualizer`** — DAG and circuit visualisation helpers

### `harvest.layout`
- **`engine`** — `LayoutEngine` class: patch placement, routing-graph
  construction, Steiner tree / packing / PathFinder algorithms
- **`presets`** — ready-made layout builders (`nxm_ring_layout_single_qubits`, …)

### `harvest.routing`
- **`processor`** — `DAGProcessor` orchestration + `process_dag_with_steiner`
  convenience function
- **`scheduler`** — DAG traversal strategies (sequential, packing, pathfinder)
- **`magic_terminal_selection`** — nearest-terminal heuristic

### `harvest.evaluation`
- **`benchmark_runner`** — `ComprehensiveRoutingPipeline`, depth-sweep experiments
- **`plotting`** — wirelength bar charts

## Dependencies

- Python ≥ 3.10
- [Qiskit](https://qiskit.org/) (circuit representation, transpiler passes)
- NetworkX, Matplotlib, NumPy, Pandas, Seaborn
- Optional: [BQSKit](https://bqskit.lbl.gov/), [MQT Bench](https://github.com/cda-tum/mqt-bench)