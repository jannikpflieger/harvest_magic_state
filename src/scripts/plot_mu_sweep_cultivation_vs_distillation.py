#!/usr/bin/env python3
"""
Sweep the cultivation mean (μ) and compare against a fixed distillation cycle time.

Runs the QAOA 100-qubit circuit on a 10×10 single-spacing ring layout
(magic states on all four sides) using Greedy Packing (steiner_packing).

Sweep:
    Cultivation  — geometric distribution, μ ∈ {1, 2, …, 20}
    Distillation — deterministic factory, fixed at 15 preparation cycles

Plot:
    x-axis  : μ (cultivation mean, 1 … 20)
    y-axis  : timesteps
    Series  : cultivation (line) + distillation at μ=15 (horizontal reference)

Outputs:
    PDF → plots/
    PNG → plots/
    JSON results → routing_experiment_results/
"""

import json
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harvest.compilation.pauli_block_conversion import convert_to_PCB, create_dag
from harvest.compilation.qasm_loader import qasm_to_circuit
from harvest.compilation.circuit_analysis import convert_rx_ry_to_rz, pre_prep_circuit
from harvest.layout.presets import nxm_ring_layout_single_qubits
from harvest.routing.processor import DAGProcessor
from harvest.routing.magic_state_factory import MagicStateFactory
from harvest.routing.magic_state_cultivator import MagicStateCultivator, geometric_sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("MuSweep")

# ── Constants ──────────────────────────────────────────────────────────────
MU_RANGE = range(1, 21)          # μ = 1 … 20
DISTILLATION_CYCLES = 15         # fixed factory cycle time
SCHEDULER_MODE = "steiner_packing"
CULTIVATION_SEED = 42
LAYOUT_ROWS = 10
LAYOUT_COLS = 10


# ── Single run ─────────────────────────────────────────────────────────────

def run_single(dag, layout_engine, source_label, magic_source):
    """Route *dag* on *layout_engine* and return a metrics dict."""
    try:
        processor = DAGProcessor(layout_engine=layout_engine, magic_source=magic_source)
        results = processor.process_entire_dag(
            dag, visualize_each_step=False, mode=SCHEDULER_MODE
        )

        meta = getattr(processor, "_scheduling_metadata", {})
        total_elapsed = meta.get("total_elapsed_steps", len(results))
        completed = meta.get("completed", True)
        nodes_completed = meta.get("num_nodes_completed", len(results))
        nodes_total = meta.get("num_nodes_total", len(results))
        total_wirelength = sum(len(r.get("steiner_edges", set())) for r in results)

        result = {
            "source_label": source_label,
            "num_timesteps": total_elapsed,
            "num_nodes_processed": nodes_completed,
            "num_nodes_total": nodes_total,
            "completed": completed,
            "total_wirelength": total_wirelength,
            "success": True,
        }

        source = processor.magic_source
        if source and not source.unlimited:
            stats = source.get_stats()
            result["magic_wait_cycles"] = stats["total_wait_cycles"]
        else:
            result["magic_wait_cycles"] = 0

        return result
    except Exception as exc:
        logger.error(f"    FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return {"source_label": source_label, "success": False, "error": str(exc)}


def main():
    # ---- Load QAOA 100-qubit circuit ----
    qasm_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "benchmark_circuits", "qasm",
        "qaoa", "big_100q", "qaoa_barabasi_albert_N100_3reps.qasm",
    )
    qasm_path = os.path.normpath(qasm_path)

    logger.info(f"Loading QAOA 100-qubit circuit from {qasm_path}")
    circuit = qasm_to_circuit(qasm_path)
    circuit = pre_prep_circuit(circuit)
    logger.info(
        f"Loaded: {circuit.num_qubits} qubits, depth {circuit.depth()}, "
        f"gates {dict(circuit.count_ops())}"
    )

    if "rx" in circuit.count_ops() or "ry" in circuit.count_ops():
        logger.info("Converting rx/ry gates to rz equivalents …")
        circuit = convert_rx_ry_to_rz(circuit)

    logger.info("Converting to PCB format …")
    pcb = convert_to_PCB(circuit, verbose=False)
    dag = create_dag(pcb)

    # ---- Build 10×10 ring layout (single spacing, magic all around) ----
    logger.info(
        f"Building {LAYOUT_ROWS}×{LAYOUT_COLS} ring layout "
        "(single spacing, magic on all four sides) …"
    )
    engine = nxm_ring_layout_single_qubits(LAYOUT_ROWS, LAYOUT_COLS)

    # Discover magic terminals
    temp_proc = DAGProcessor(layout_engine=engine)
    magic_terminals = list(temp_proc.magic_terminals)
    logger.info(f"Magic terminals: {len(magic_terminals)}")

    # ---- Run distillation (factory) at fixed μ=15 ----
    logger.info(f"\n{'='*60}")
    logger.info(f"Distillation — factory, prep_cycles={DISTILLATION_CYCLES}")
    logger.info(f"{'='*60}")
    factory_src = MagicStateFactory(magic_terminals, DISTILLATION_CYCLES)
    distillation_result = run_single(
        dag, engine,
        f"Distillation ({DISTILLATION_CYCLES} cyc)",
        factory_src,
    )
    distillation_result["mu"] = DISTILLATION_CYCLES
    if distillation_result["success"]:
        distillation_steps = distillation_result["num_timesteps"]
        logger.info(f"    → {distillation_steps} timesteps")
    else:
        distillation_steps = None
        logger.error("Distillation run FAILED")

    # ---- Sweep μ for cultivation ----
    mu_values = list(MU_RANGE)
    cultivation_steps = []
    all_cultivation_results = []

    for mu in mu_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Cultivation — geometric μ={mu}  (p={1.0/mu:.4f})")
        logger.info(f"{'='*60}")

        cultivation_src = MagicStateCultivator(
            magic_terminals,
            readiness_sampler=geometric_sampler(1.0 / mu),
            seed=CULTIVATION_SEED,
        )
        result = run_single(
            dag, engine,
            f"Cultivation (geo μ≈{mu})",
            cultivation_src,
        )
        result["mu"] = mu
        all_cultivation_results.append(result)

        if result["success"]:
            cultivation_steps.append(result["num_timesteps"])
            logger.info(f"    → {result['num_timesteps']} timesteps")
        else:
            cultivation_steps.append(None)
            logger.error(f"    Cultivation μ={mu} FAILED")

    # ---- Save results ----
    os.makedirs("routing_experiment_results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"mu_sweep_cultivation_vs_distillation_qaoa_100q_{ts}"

    json_path = f"routing_experiment_results/{tag}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "circuit": "qaoa_barabasi_albert_N100_3reps",
                    "num_qubits": circuit.num_qubits,
                    "layout": f"ring_{LAYOUT_ROWS}x{LAYOUT_COLS}_single_spacing",
                    "grid": f"{2*LAYOUT_ROWS+3}x{2*LAYOUT_COLS+3}",
                    "magic_topology": "all_around",
                    "num_magic_terminals": len(magic_terminals),
                    "scheduler": SCHEDULER_MODE,
                    "mu_range": [MU_RANGE.start, MU_RANGE.stop - 1],
                    "distillation_cycles": DISTILLATION_CYCLES,
                    "cultivation_seed": CULTIVATION_SEED,
                },
                "distillation_result": distillation_result,
                "cultivation_results": all_cultivation_results,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Saved results → {json_path}")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — Timesteps by μ")
    logger.info(f"  Distillation (factory, μ={DISTILLATION_CYCLES}): "
                f"{distillation_steps if distillation_steps is not None else 'FAILED'}")
    logger.info(f"{'μ':>5s}  {'Cultivation':>12s}")
    logger.info("-" * 22)
    for i, mu in enumerate(mu_values):
        val = cultivation_steps[i] if cultivation_steps[i] is not None else "FAIL"
        logger.info(f"  {mu:3d}  {str(val):>12s}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
