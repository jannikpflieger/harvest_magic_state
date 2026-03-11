#!/usr/bin/env python3
"""
Analyse benchmark QASM circuits: gate counts, PCB conversion, DAG metrics.

This is the standalone script equivalent of the old circuit_info_pipeline.py.
It scans a directory of .qasm files, runs the full analysis pipeline, and
writes per-circuit JSON results plus a consolidated summary.

Usage:
    # Analyse all benchmark circuits (default directory)
    python scripts/analyze_circuits.py

    # Only specific sub-folders
    python scripts/analyze_circuits.py --subdirs feynman bigint

    # Custom benchmark directory, limit to 50 files
    python scripts/analyze_circuits.py --benchmark-dir /path/to/qasm --max-files 50

    # Include circuits with unsupported gates (normally skipped)
    python scripts/analyze_circuits.py --include-unsupported
"""

import argparse

from harvest.compilation.circuit_analysis import run_qasm_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive QASM circuit analysis pipeline"
    )
    parser.add_argument(
        "--max-files", type=int,
        help="Maximum number of files to analyze",
    )
    parser.add_argument(
        "--subdirs", nargs="+",
        help="Specific subdirectories to test (e.g., feynman bigint)",
    )
    parser.add_argument(
        "--benchmark-dir",
        help="Path to benchmark directory (default: circuit/benchmark_circuits/qasm)",
    )
    parser.add_argument(
        "--output-json",
        help="Path to save consolidated JSON results (default: auto-generated)",
    )
    parser.add_argument(
        "--include-unsupported", action="store_true",
        help="Include circuits with unsupported gates (default: skip them)",
    )

    args = parser.parse_args()

    run_qasm_pipeline(
        benchmark_dir=args.benchmark_dir,
        max_files=args.max_files,
        subdirs=args.subdirs,
        output_json=args.output_json,
        skip_unsupported=not args.include_unsupported,
    )


if __name__ == "__main__":
    main()
