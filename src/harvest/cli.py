#!/usr/bin/env python3
"""
Unified CLI entry point for the Harvest Magic State pipeline.

Usage:
    python -m harvest run   --qubits 20 --depth 10 --rows 5 --cols 5 --mode steiner_packing
    python -m harvest bench --rows 5 --cols 5 --depth-start 10 --depth-end 100 --depth-step 25 --runs 10
    python -m harvest analyze routing_experiment_results --plots --detailed
    python -m harvest plot   routing_experiment_results
"""

import argparse
import sys


def cmd_run(args):
    """Run the pipeline on a single circuit."""
    from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag, create_random_circuit
    from harvest.compilation.qasm_loader import qasm_to_circuit
    from harvest.routing import process_dag_with_steiner

    if args.qasm:
        circuit = qasm_to_circuit(args.qasm)
        print(f"Loaded circuit from {args.qasm}")
    else:
        circuit = create_random_circuit(num_qubits=args.qubits, depth=args.depth, seed=args.seed)
        print(f"Generated random circuit: {args.qubits} qubits, depth {args.depth}")

    pcb = convert_to_PCB(circuit)
    dag = create_dag(pcb)
    print(f"DAG: {dag.count_ops()} ops, depth {dag.depth()}, {dag.num_qubits()} qubits")

    processor, results = process_dag_with_steiner(
        dag, layout_rows=args.rows, layout_cols=args.cols,
        visualize_steps=args.visualize, mode=args.mode,
    )

    print(f"Routed {len(results)} DAG nodes successfully.")
    print(processor.get_summary(args.mode))


def cmd_bench(args):
    """Run a depth-sweep benchmark experiment."""
    from harvest.evaluation import ComprehensiveRoutingPipeline

    pipeline = ComprehensiveRoutingPipeline(args.output_dir)
    pipeline.run_systematic_depth_experiment(
        layout_rows=args.rows,
        layout_cols=args.cols,
        depth_start=args.depth_start,
        depth_end=args.depth_end,
        depth_step=args.depth_step,
        runs_per_depth=args.runs,
        experiment_name=args.name,
    )


def cmd_analyze(args):
    """Analyze routing experiment results."""
    # Re-use the standalone analyzer script
    from scripts.analyze_results import RoutingResultsAnalyzer

    analyzer = RoutingResultsAnalyzer(args.results_dir)
    loaded = analyzer.load_results()
    if loaded == 0:
        print("No results found.")
        return

    analyzer.generate_summary_report()
    analyzer.create_comparison_dataframe()

    if args.plots:
        analyzer.create_performance_plots()
    if args.detailed:
        analyzer.print_detailed_analysis()


def cmd_plot(args):
    """Generate wirelength plots from results."""
    from harvest.evaluation.plotting import extract_wirelength_data, create_wirelength_plot

    data = extract_wirelength_data(args.results_dir)
    if not data:
        print("No data found.")
        return
    create_wirelength_plot(data, output_filename=args.output)


def cmd_analyze_circuits(args):
    """Analyse benchmark QASM circuits (gate counts, PCB conversion, DAG metrics)."""
    from harvest.compilation.circuit_analysis import run_qasm_pipeline

    run_qasm_pipeline(
        benchmark_dir=args.benchmark_dir,
        max_files=args.max_files,
        subdirs=args.subdirs,
        output_json=args.output_json,
        skip_unsupported=not args.include_unsupported,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="harvest",
        description="Harvest Magic State — lattice-surgery routing toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Run pipeline on a single circuit")
    p_run.add_argument("--qasm", type=str, default=None, help="Path to QASM file (omit for random)")
    p_run.add_argument("--qubits", type=int, default=20, help="Number of qubits for random circuit")
    p_run.add_argument("--depth", type=int, default=10, help="Depth for random circuit")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--rows", type=int, default=5, help="Layout rows")
    p_run.add_argument("--cols", type=int, default=5, help="Layout columns")
    p_run.add_argument("--mode", choices=["steiner_tree", "steiner_packing", "steiner_pathfinder"],
                       default="steiner_packing", help="Routing mode")
    p_run.add_argument("--visualize", action="store_true", help="Visualize each step")

    # --- bench ---
    p_bench = sub.add_parser("bench", help="Run depth-sweep benchmark experiment")
    p_bench.add_argument("--rows", type=int, default=5)
    p_bench.add_argument("--cols", type=int, default=5)
    p_bench.add_argument("--depth-start", type=int, default=10)
    p_bench.add_argument("--depth-end", type=int, default=100)
    p_bench.add_argument("--depth-step", type=int, default=25)
    p_bench.add_argument("--runs", type=int, default=10, help="Runs per depth")
    p_bench.add_argument("--name", type=str, default="depth_sweep", help="Experiment name")
    p_bench.add_argument("--output-dir", type=str, default="results/routing_experiments")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Analyze experiment results")
    p_analyze.add_argument("results_dir", help="Directory with JSON results")
    p_analyze.add_argument("--plots", action="store_true")
    p_analyze.add_argument("--detailed", action="store_true")

    # --- plot ---
    p_plot = sub.add_parser("plot", help="Generate wirelength plots")
    p_plot.add_argument("results_dir", help="Directory with JSON results")
    p_plot.add_argument("--output", type=str, default="wirelength_comparison.png")

    # --- analyze-circuits ---
    p_ac = sub.add_parser("analyze-circuits", help="Analyse benchmark QASM circuits")
    p_ac.add_argument("--benchmark-dir", help="Path to benchmark QASM directory")
    p_ac.add_argument("--max-files", type=int, help="Max files to analyze")
    p_ac.add_argument("--subdirs", nargs="+", help="Subdirectories to scan (e.g. feynman bigint)")
    p_ac.add_argument("--output-json", help="Path for consolidated JSON results")
    p_ac.add_argument("--include-unsupported", action="store_true",
                      help="Include circuits with unsupported gates")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "bench": cmd_bench,
        "analyze": cmd_analyze,
        "plot": cmd_plot,
        "analyze-circuits": cmd_analyze_circuits,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
