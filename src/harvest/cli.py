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
from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag, create_random_circuit
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.compilation.circuit_analysis import run_qasm_pipeline
from harvest.routing import process_dag_with_steiner
from harvest.evaluation import ComprehensiveRoutingPipeline
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

    # Build ILP config if the user selected the ILP mode.
    _ilp_config = None
    if getattr(args, 'mode', None) == 'ilp_steiner_packing':
        from harvest.routing.ilp_steiner_packing import ILPConfig
        _ilp_config = ILPConfig(
            time_limit_ms=getattr(args, 'ilp_time_limit_ms', 3000),
            max_ready_products=getattr(args, 'ilp_max_ready', 20),
            max_magic_candidates=getattr(args, 'ilp_max_magic_candidates', 8),
        )

    # Build magic-state source (factory, cultivation, or unlimited)
    magic_source = None
    source_mode = getattr(args, 'magic_source_type', 'unlimited')

    if source_mode == 'factory':
        prep_cycles = args.magic_prep_cycles
        if prep_cycles is None:
            prep_cycles = 15
        print(f"Magic-state factory: {prep_cycles}-cycle cooldown per terminal")
        # Let DAGProcessor build the factory from prep_cycles
        processor, results = process_dag_with_steiner(
            dag, layout_rows=args.rows, layout_cols=args.cols,
            visualize_steps=args.visualize, mode=args.mode,
            magic_prep_cycles=prep_cycles,
            ilp_config=_ilp_config,
        )
    elif source_mode == 'cultivation':
        from harvest.routing.magic_state_cultivator import (
            MagicStateCultivator, geometric_sampler,
        )
        mean_cycles = getattr(args, 'cultivation_mean_cycles', 15) or 15
        seed = getattr(args, 'cultivation_seed', 42) or 42
        max_cycles = getattr(args, 'cultivation_max_cycles', None)
        p = 1.0 / mean_cycles
        print(f"Magic-state cultivation: geometric(p={p:.4f}), "
              f"mean={mean_cycles} cycles, seed={seed}"
              + (f", max={max_cycles}" if max_cycles else ""))
        # Source will be constructed once we know the terminal list,
        # so we build processor first with unlimited, then attach.
        from harvest.routing.processor import DAGProcessor as _DP
        processor = _DP(
            layout_rows=args.rows, layout_cols=args.cols,
        )
        cultivator = MagicStateCultivator(
            processor.magic_terminals,
            geometric_sampler(p),
            seed=seed,
            max_cycles=max_cycles,
        )
        processor.magic_source = cultivator
        if _ilp_config is not None:
            processor.set_ilp_config(_ilp_config)
        results = processor.process_entire_dag(
            dag, visualize_each_step=args.visualize, mode=args.mode,
        )
        processor.processing_results = results
    else:
        # unlimited (legacy default) — also handles --magic-prep-cycles
        prep_cycles = args.magic_prep_cycles
        if prep_cycles is not None:
            print(f"Magic-state factory: {prep_cycles}-cycle cooldown per terminal")
        processor, results = process_dag_with_steiner(
            dag, layout_rows=args.rows, layout_cols=args.cols,
            visualize_steps=args.visualize, mode=args.mode,
            magic_prep_cycles=prep_cycles,
            ilp_config=_ilp_config,
        )

    print(f"Routed {len(results)} DAG nodes successfully.")
    print(processor.get_summary(args.mode))

    if processor.magic_source and not processor.magic_source.unlimited:
        stats = processor.magic_source.get_stats()
        source_label = stats.get('source_type', 'factory')
        print(f"Magic {source_label}: consumed={stats['total_consumed']}, "
              f"wait_cycles={stats['total_wait_cycles']}")


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



def cmd_bench_qasm(args):
    """Run routing experiments on QASM benchmark circuits."""
    
    pipeline = ComprehensiveRoutingPipeline(args.output_dir)
    pipeline.run_qasm_experiment(
        qasm_dir=args.qasm_dir,
        layout_rows=args.rows,
        layout_cols=args.cols,
        experiment_name=args.name,
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
    p_run.add_argument(
        "--mode",
        choices=[
            "steiner_tree",
            "steiner_packing",
            "steiner_pathfinder",
            "ilp_steiner_packing",
        ],
        default="steiner_packing",
        help="Routing mode (default: steiner_packing)",
    )
    p_run.add_argument(
        "--ilp-time-limit-ms",
        type=int,
        default=3000,
        help="ILP solver wall-clock budget in ms (ilp_steiner_packing only, default: 3000)",
    )
    p_run.add_argument(
        "--ilp-max-ready",
        type=int,
        default=20,
        help=(
            "Max ready products before ILP falls back to greedy "
            "(ilp_steiner_packing only, default: 20)"
        ),
    )
    p_run.add_argument(
        "--ilp-max-magic-candidates",
        type=int,
        default=8,
        help=(
            "Max magic-state candidates per product in ILP "
            "(ilp_steiner_packing only, default: 8)"
        ),
    )
    p_run.add_argument("--magic-prep-cycles", type=int, default=None,
                       help="Per-terminal cooldown cycles (e.g. 15 for 15-to-1). Omit for unlimited.")
    p_run.add_argument("--magic-source-type", choices=["unlimited", "factory", "cultivation"],
                       default="unlimited",
                       help="Magic-state preparation model (default: unlimited)")
    p_run.add_argument("--cultivation-mean-cycles", type=float, default=15,
                       help="Mean cultivation readiness in cycles (default: 15)")
    p_run.add_argument("--cultivation-seed", type=int, default=42,
                       help="RNG seed for cultivation (default: 42)")
    p_run.add_argument("--cultivation-max-cycles", type=int, default=None,
                       help="Hard cap on sampled cultivation time (optional)")

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
        "bench-qasm": cmd_bench_qasm,
        "analyze-circuits": cmd_analyze_circuits,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
