# QASM Circuit Comprehensive Analysis Pipeline

This pipeline performs comprehensive analysis of QASM files including circuit conversion, PCB transformation, DAG analysis, and saves individual JSON files per circuit in a dedicated results folder.

## Key Features

- **Individual JSON files**: Each circuit gets its own detailed JSON file in `benchmark_circuits/circuit_analysis_results/`
- **Automatic gate conversion**: Converts unsupported `rx`/`ry` gates to equivalent `rz`+Clifford sequences before analysis
- **Smart circuit filtering**: By default, skips circuits with truly unsupported gates (ccx, measure, etc.) to focus on PCB-convertible circuits
- **Comprehensive analysis**: Collects detailed information about original circuits, PCB conversion, and DAG layer structure
- **Robust error handling**: Graceful handling of conversion failures with detailed error reporting
- **Flexible options**: Command-line control over filtering, output locations, and analysis scope

## Circuit Processing Pipeline

### 1. Gate Conversion Preprocessing
**Automatic rx/ry → rz conversion:**
- `rx(θ) → h, rz(θ), h` (using Clifford gates + rz)
- `ry(θ) → sx, rz(θ), sxdg` (using sx Clifford gates + rz)
- Conversion time tracked and reported
- Dramatically increases number of analyzable circuits

**Before conversion:** `{'rx': 1, 'ry': 1, 'rz': 1, 'cx': 5, 'h': 2}`
**After conversion:** `{'rz': 15, 'sx': 4, 'h': 1, 'cx': 5}`

### 2. Circuit Filtering

### Enhanced Supported Gates (with conversion)
**Native Clifford gates:** `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `sx`, `sxdg`, `cx`, `cz`, `cy`, `swap`, `iswap`, `ecr`, `dcx`

**RZ-rotation gates:** `t`, `tdg`, `rz` (converted to PauliEvolution gates)

**Convertible gates:** `rx`, `ry` (automatically converted to supported equivalents)

### Still Unsupported Gates 
**Complex gates:** `ccx` (Toffoli), `measure`, `u3`, `cH`, custom gates, barriers

**Behavior:**
- **Convertible circuits** (rx/ry): Automatically processed with conversion
- **Unsupported circuits** (ccx/measure): Skipped by default with detailed reporting  
- **Override**: Use `--include-unsupported` to process all circuits (may have PCB conversion failures)

## Individual JSON File Structure

Each circuit produces a JSON file named: `{circuit_id}.json`

```
benchmark_circuits/circuit_analysis_results/
├── benchmark_circuits_qasm_clifford_clifford_20_12345.json
├── benchmark_circuits_qasm_feynman_qft_4.json
└── ...
```

**JSON Content:**
```json
{
  "metadata": {
    "analysis_timestamp": "2026-02-14 18:02:25",
    "circuit_file": "benchmark_circuits/qasm/qft/qft_N002.qasm"
  },
  "circuit": {
    "circuit_name": "qft_N002",
    "circuit_id": "benchmark_circuits_qasm_qft_qft_N002",
    "num_qubits": 2,
    "depth": 28,
    "total_gates": 28,
    "non_clifford_gates": 15,
    "gate_counts": {"cx": 5, "rz": 15, "sx": 8},
    "rx_ry_conversion_needed": true,
    "rx_ry_conversion_time_seconds": 0.001258,
    "pcb_conversion_successful": true,
    "pcb_depth": 15,
    "pcb_gate_counts": {"PauliEvolution": 15},
    "dag_analysis_successful": true,
    "num_layers": 15,
    "max_pauli_per_layer": 1,
    "median_pauli_per_layer": 1,
    "total_pauli_evolutions": 15,
    "pauli_evolutions_per_layer": [1, 1, 1, ...]
  }
}
```

## Usage

### Basic usage (analyzes all supported circuits):
```bash
python circuit_info_pipeline.py
```

### Analyze specific subdirectories:
```bash
python circuit_info_pipeline.py --subdirs clifford feynman
```

### Limit number of circuits analyzed:
```bash
python circuit_info_pipeline.py --max-files 10
```

### Include circuits with unsupported gates:
```bash
python circuit_info_pipeline.py --include-unsupported
```

### Custom output summary JSON:
```bash
python circuit_info_pipeline.py --output-json my_summary.json
```

### Combined options:
```bash
python circuit_info_pipeline.py --subdirs clifford --max-files 5 --output-json clifford_test.json
```

## Output Files

### Individual Circuit JSON Files
**Location**: `benchmark_circuits/circuit_analysis_results/`
- One JSON file per analyzed circuit
- Comprehensive analysis data for each circuit
- Timestamped metadata

### Summary JSON File  
**Location**: Current directory (unless `--output-json` specified)
- Aggregated metadata and statistics
- List of skipped circuits with reasons
- Overall analysis summary
- References to individual JSON files directory

## Console Output

### Circuit Filtering Phase
```
================================================================================
Filtering circuits for supported gates...
================================================================================
✓ OK - benchmark_circuits/qasm/clifford/clifford_20_12345.qasm
SKIP - benchmark_circuits/qasm/feynman/W-state.qasm: unsupported gates ['measure', 'ccx']
```

### Analysis Phase
```
================================================================================  
Performing comprehensive analysis of 2 QASM files
================================================================================
[1/2] Analyzing: benchmark_circuits/qasm/clifford/clifford_20_12345.qasm
✓ SUCCESS - 20 qubits, depth 605, 0 non-Clifford gates
  PCB: ✓, DAG: ✓  
  Layers: 605, Max Pauli/layer: 0
```

## Example Workflows

### Analyze circuits with automatic rx/ry conversion:
```bash
# Many more circuits now supported with automatic conversion
python circuit_info_pipeline.py --max-files 50

# QFT circuits now work great (rx/ry → rz conversion)  
python circuit_info_pipeline.py --subdirs qft --max-files 10

# Focus on pure Clifford circuits (no conversion needed)
python circuit_info_pipeline.py --subdirs clifford
```

### Find circuits with significant Pauli evolutions:
```bash
# QFT and other rotation-heavy circuits
python circuit_info_pipeline.py --subdirs qft qaoa --max-files 20
```

### Debug conversion and PCB issues:
```bash  
# Include problematic circuits to see specific error messages
python circuit_info_pipeline.py --subdirs feynman --include-unsupported --max-files 5

# See conversion statistics for rx/ry heavy circuits
python circuit_info_pipeline.py --subdirs qft --max-files 5 --output-json qft_analysis.json
```

## File Organization

### Input Structure
```
circuit/benchmark_circuits/qasm/
├── bigint/           # Integer arithmetic circuits
├── clifford/         # Pure Clifford circuits (✓ PCB compatible)  
├── dtc/             # Discrete time crystal circuits
├── feynman/         # Classical logic circuits (mostly ✗ unsupported gates)
├── qaoa/            # Quantum approximate optimization
├── qasmbench-large/ # Large benchmark circuits
├── qasmbench-medium/# Medium benchmark circuits  
├── qasmbench-small/ # Small benchmark circuits
├── qft/             # Quantum Fourier transform circuits
├── qv/              # Quantum volume circuits
└── square-heisenberg/ # Heisenberg model circuits
```

### Output Structure 
```
circuit/benchmark_circuits/
├── qasm/                          # Original QASM files
└── circuit_analysis_results/      # Generated JSON results
    ├── benchmark_circuits_qasm_clifford_clifford_100_12345.json
    ├── benchmark_circuits_qasm_clifford_clifford_20_12345.json  
    └── ...
```

## Summary Statistics

The pipeline provides comprehensive statistics:
- **Circuit filtering**: Number supported vs. skipped with detailed reasons
- **Analysis success rates**: Per-stage success rates (basic, PCB, DAG)
- **Circuit characteristics**: Qubit counts, depths, gate counts across all circuits
- **PCB conversion metrics**: Success rates, depth/gate count changes  
- **DAG analysis insights**: Layer counts, Pauli evolution distributions
- **Performance metrics**: Processing times per circuit and stage

## Error Handling & Robustness

### Intelligent Filtering
- **Pre-analysis**: Checks gate compatibility before processing
- **Detailed reporting**: Lists specific unsupported gates for each skipped circuit
- **Flexible control**: Option to override filtering for debugging

### Graceful Degradation  
- **Partial success**: Basic analysis succeeds even if PCB/DAG stages fail
- **Error isolation**: One circuit failure doesn't stop pipeline  
- **Comprehensive logging**: All errors captured in individual JSON files

### Common Issues & Solutions
**Issue**: "All circuits skipped - no supported gates found"  
**Solution**: Use `--include-unsupported` or try `--subdirs clifford` 

**Issue**: PCB conversion fails with "unsupported gates"  
**Solution**: Expected behavior; check individual JSON for specific gates

**Issue**: Large circuits taking too long  
**Solution**: Use `--max-files` to limit scope during testing

## Performance Characteristics

- **Circuit filtering**: ~0.01s per circuit (quick gate checking)
- **rx/ry conversion**: ~0.001-0.01s per circuit (very fast basis transformation)  
- **Basic analysis**: ~0.001-0.5s per circuit (depends on size)  
- **PCB conversion**: ~0.001-2s per circuit (depends on complexity)
- **DAG analysis**: ~0.0001-0.01s per circuit (very fast)
- **JSON writing**: ~0.001s per individual file

**Impact of rx/ry conversion:**
- **Gate count increase**: ~3-5x more gates after conversion (rx→h,rz,h expansion)
- **Processing overhead**: Negligible (~1ms per conversion)
- **Compatibility gain**: Massive increase in analyzable circuits (e.g., all QFT circuits now work)

**Scalability**: Can process 100s of circuits in minutes; 1000s in reasonable time.

## Dependencies

- **Qiskit**: Core circuit operations and LitinskiTransformation  
- **Python Standard Library**: os, glob, json, statistics, pathlib, time, argparse
- **Type Hints**: typing module for better code documentation

## Version Compatibility

- **Python**: 3.8+ (uses Tuple/List type hints)
- **Qiskit**: Recent versions with LitinskiTransformation support
- **Platform**: Cross-platform (Linux, macOS, Windows)