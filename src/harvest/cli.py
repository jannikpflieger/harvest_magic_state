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
from harvest.compilation.visualizer import process_all_circuit_analyses, visualize_circuit_information
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag, create_random_circuit
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.compilation.circuit_analysis import run_qasm_pipeline
from harvest.routing import process_dag_with_steiner
from harvest.evaluation import ComprehensiveRoutingPipeline
from harvest.evaluation.plotting import extract_wirelength_data, create_wirelength_plot
from scripts.analyze_results import RoutingResultsAnalyzer
from pathlib import Path


def cmd_run(args):
    """Run the pipeline on a single circuit."""
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

    data = extract_wirelength_data(args.results_dir)
    if not data:
        print("No data found.")
        return
    create_wirelength_plot(data, output_filename=args.output)


def cmd_plot_pauli(args):
    """Generate Pauli evolution metric plots from circuit analysis results."""

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f"{file_path.stem}_analysis.png"
        print(f"Processing: {file_path}")
        visualize_circuit_information(
            str(file_path),
            window_size=args.window_size,
            save_file=str(save_path),
            show_plot=args.show,
        )
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return
        print(f"Processing all circuit analysis files in: {results_dir}")
        print(f"Saving plots to: {args.plots_dir}")
        process_all_circuit_analyses(
            results_dir=str(results_dir),
            plots_dir=args.plots_dir,
        )

def cmd_bench_qasm(args):
    """Run routing experiments on QASM benchmark circuits."""
    
    pipeline = ComprehensiveRoutingPipeline(args.output_dir)
    pipeline.run_qasm_experiment(
        qasm_dir=args.qasm_dir,
        layout_rows=args.rows,
        layout_cols=args.cols,
        experiment_name=args.name,
    )

def cmd_plot_qasm(args):
    """Generate wirelength plots from QASM experiment results (grouped by qubit count)."""
    from harvest.evaluation.plotting import extract_qasm_wirelength_data, create_qasm_wirelength_plot

    data = extract_qasm_wirelength_data(args.results_dir)
    if not data:
        print("No data found.")
        return
    create_qasm_wirelength_plot(
        data,
        output_filename=args.output,
        title_prefix=f"{args.title_prefix} " if args.title_prefix else '',
    )

def cmd_analyze_circuits(args):
    """Analyse benchmark QASM circuits (gate counts, PCB conversion, DAG metrics)."""

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

    # --- plot-pauli ---
    p_pp = sub.add_parser("plot-pauli", help="Generate Pauli evolution metric plots")
    p_pp.add_argument("--results-dir",
                      default="../benchmark_circuits/circuit_analysis_results/",
                      help="Directory containing circuit analysis JSON files")
    p_pp.add_argument("--plots-dir", default="../plots/",
                      help="Directory to save generated plots")
    p_pp.add_argument("--file", default=None,
                      help="Process a single JSON file instead of the whole directory")
    p_pp.add_argument("--window-size", type=int, default=None,
                      help="Moving window size (default: auto-calculated)")
    p_pp.add_argument("--show", action="store_true",
                      help="Display plots interactively")

    # --- bench-qasm ---
    p_bq = sub.add_parser("bench-qasm", help="Run routing experiments on QASM benchmark circuits")
    p_bq.add_argument("qasm_dir", help="Directory containing QASM files")
    p_bq.add_argument("--rows", type=int, default=None, help="Layout rows (auto-sized if omitted)")
    p_bq.add_argument("--cols", type=int, default=None, help="Layout columns (auto-sized if omitted)")
    p_bq.add_argument("--name", type=str, default="qasm_sweep", help="Experiment name")
    p_bq.add_argument("--output-dir", type=str, default="routing_experiment_results",
                      help="Directory for result JSON files")

    # --- plot-qasm ---
    p_pq = sub.add_parser("plot-qasm", help="Generate wirelength plots from QASM experiment results")
    p_pq.add_argument("results_dir", help="Directory with JSON result files")
    p_pq.add_argument("--output", type=str, default="qasm_wirelength_comparison.png")
    p_pq.add_argument("--title-prefix", type=str, default="",
                      help="Prefix for the plot title (e.g. 'QAOA')")

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
        "plot-pauli": cmd_plot_pauli,
        "bench-qasm": cmd_bench_qasm,
        "plot-qasm": cmd_plot_qasm,
        "analyze-circuits": cmd_analyze_circuits,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
