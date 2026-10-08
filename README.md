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

## Reproducing the SOTA baseline comparison

The comparison driver keeps the three scientifically distinct regimes
(`matched_ir`, `matched_architecture`, and `native_end_to_end`) separate and
stores one JSON record per trial:

```bash
PYTHONPATH=src python src/scripts/compare_sota_baselines.py \
  --benchmarks benchmark_circuits/qasm/qaoa \
  --baselines harvest,silva,puremagic,dascot \
  --comparison-mode matched_ir \
  --trials 20 --seed 42 \
  --output-dir results/sota_comparison
```

HARVEST and the in-tree `silva_eaf` reproduction need no additional tool.
For the external baselines:

1. Build the official [BQSKit/PureMagic](https://github.com/BQSKit/PureMagic)
   paper snapshot (the upstream project identifies it as `QCE`) with
   `cargo build --release`. Pass its scheduler with `--puremagic-bin`; for a
   native QASM pipeline also pass `--puremagic-transpile-bin` and, when the
   source is not already Clifford+T, `--puremagic-compile-bin`. If the QCE ref
   is unavailable, check out an explicit commit and pass that identity through
   `--puremagic-version`; the adapter will not relabel an observed `main`
   banner as the paper snapshot. Custom topology files can be supplied with
   `--puremagic-topology` and are copied into the per-run artifact directory.
2. Install the official [qqq-wisc/wisq](https://github.com/qqq-wisc/wisq)
   package, preferably the recorded `v0.2.7` tag, and pass `--wisq-bin` if it
   is not on `PATH`. The adapter always invokes `--mode scmr`, so GUOQ circuit
   optimization is not mixed into the mapping/routing result. wisq's `setup.py`
   requires `gcc` and `java >= 21` to be importable at install time (it shells
   out to both just to check versions); install a JDK first if `pip install`
   fails with `ValueError: invalid literal for int() with base 10: ''`.

```bash
python -m pip install "git+https://github.com/qqq-wisc/wisq.git@v0.2.7"
```

For a quick in-tree smoke test:

```bash
PYTHONPATH=src python src/scripts/compare_sota_baselines.py \
  --benchmarks examples/sota_baselines \
  --baselines harvest,silva \
  --comparison-mode matched_ir --trials 1 \
  --output-dir results/sota_smoke

# Re-running does not execute completed trials again.
PYTHONPATH=src python src/scripts/compare_sota_baselines.py \
  --benchmarks examples/sota_baselines \
  --baselines harvest,silva \
  --comparison-mode matched_ir --trials 1 \
  --output-dir results/sota_smoke --resume

# Optional real-upstream smoke tests (normal tests use fixtures/mocks).
RUN_EXTERNAL_BASELINES=1 PUREMAGIC_BIN=/path/to/puremagic \
  WISQ_BIN=/path/to/wisq PYTHONPATH=src \
  pytest -m integration tests/test_external_baselines_integration.py
```

Outputs include raw trials, successful-trial counts, failure records, 95%
confidence intervals, paper-ready PDFs, pairwise shared-subset geometric means,
and complete environment/version metadata. See
[`docs/sota_baselines.md`](docs/sota_baselines.md) before interpreting any
cross-system ratio; DASCOT cycles and PPR scheduler cycles are not the same
compilation problem.

The checked-in tiny run and its plots are under
[`examples/sota_baselines/example_results`](examples/sota_baselines/example_results).
