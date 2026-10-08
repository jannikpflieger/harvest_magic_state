"""Common-schema adapter for HARVEST itself."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .metrics import space_time_volume


def layout_patch_count(layout_engine) -> int:
    """Count logical tiles in the initial layout, excluding explicit blocks."""
    blocked = sum(1 for value in layout_engine.occ.values() if value == "BLOCKED")
    return layout_engine.W * layout_engine.H - blocked


def processor_patch_count(processor) -> int:
    """Count retained routing tiles plus retained data/magic patch tiles."""
    routing_tiles = sum(isinstance(node, tuple) for node in processor.graph)
    retained_patch_tiles = 0
    for name in processor.ports_by_patch:
        patch = processor.eng.patches.get(name)
        retained_patch_tiles += len(patch.cells) if patch is not None else 1
    return routing_tiles + retained_patch_tiles


class HarvestAdapter(BaselineAdapter):
    name = "harvest"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        dag = kwargs.get("dag")
        layout_engine = kwargs.get("layout_engine")
        preprocessing_runtime = float(kwargs.get("preprocessing_runtime_s", 0.0))
        if dag is None or layout_engine is None:
            raise ValueError("HarvestAdapter.run requires dag= and layout_engine=")

        from harvest.routing.processor import DAGProcessor

        initial_patches = layout_patch_count(layout_engine)
        started = perf_counter()
        processor = DAGProcessor(layout_engine=layout_engine)
        results = processor.process_entire_dag(dag, mode="harvest")
        metadata = dict(getattr(processor, "_scheduling_metadata", {}))
        completed = bool(metadata.get("completed", False))
        length = metadata.get("total_elapsed_steps") if completed else None
        pruning_stats = processor.prune_after_scheduling(results) if completed else None
        post_patches = processor_patch_count(processor) if completed else None
        scheduler_runtime = perf_counter() - started
        runtime = preprocessing_runtime + scheduler_runtime
        wirelength = sum(len(result.get("steiner_edges", ())) for result in results)
        schedule = [
            {
                "gate_name": result.get("gate_name"),
                "qubits": result.get("qubits", []),
                "time_step": result.get("time_step"),
                "magic_terminal": result.get("magic_terminal"),
                "routing_edges": len(result.get("steiner_edges", ())),
            }
            for result in results
            if result.get("success", True)
        ]
        return BaselineResult(
            baseline=config.baseline,
            variant=config.variant,
            circuit=config.circuit,
            num_qubits=config.num_qubits,
            input_representation="HARVEST Pauli-product DAG",
            comparison_mode=config.comparison_mode,
            schedule_length=length,
            logical_patches=post_patches,
            space_time_volume=space_time_volume(post_patches, length),
            routing_wirelength=wirelength if completed else None,
            compiler_runtime_s=runtime,
            completed=completed,
            trial=config.trial,
            seed=None,
            upstream_version=config.upstream_version,
            command="DAGProcessor.process_entire_dag(mode='harvest')",
            notes=[
                "Pruning is applied only after scheduling and does not change schedule length.",
                "Both initial and post-pruning patch counts/space-time values are reported.",
                "Peak memory is null because in-process memory instrumentation is not enabled.",
                "Runtime includes frontend/layout in native_end_to_end and only scheduling/pruning otherwise.",
            ],
            error=None if completed else "HARVEST did not route every DAG operation",
            initial_logical_patches=initial_patches,
            post_pruning_logical_patches=post_patches,
            space_time_volume_initial=space_time_volume(initial_patches, length),
            space_time_volume_post_pruning=space_time_volume(post_patches, length),
            metadata={
                "scheduler": metadata,
                "schedule": schedule,
                "pruning": pruning_stats,
                "runtime_breakdown_s": {
                    "frontend_and_layout": preprocessing_runtime,
                    "scheduling_and_pruning": scheduler_runtime,
                },
                "space_time_definition": {
                    "initial": "initial_logical_patches * schedule_length",
                    "post_pruning": "post_pruning_logical_patches * schedule_length",
                    "primary_field": "post_pruning",
                },
            },
        )
