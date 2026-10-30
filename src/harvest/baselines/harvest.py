"""Common-schema adapter for HARVEST itself."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Optional

from .base import BaselineAdapter, BaselineResult, BaselineRunConfig
from .metrics import space_time_volume


def layout_patch_count(layout_engine) -> int:
    """Count logical tiles in the initial layout, excluding explicit blocks."""
    blocked = sum(1 for value in layout_engine.occ.values() if value == "BLOCKED")
    return layout_engine.W * layout_engine.H - blocked


def processor_patch_count(processor, num_qubits: Optional[int] = None) -> int:
    """Count retained routing tiles plus retained data/magic patch tiles.

    Pruning drops every patch the schedule never routed to, including the data
    patch of a logical qubit that no Pauli product touches.  Such a qubit still
    holds state, so with ``num_qubits`` its data patch ``q_<i>`` is counted
    even when pruned.  Unbound layout slots (``q_<i>`` with ``i >= num_qubits``)
    stay pruned.
    """
    routing_tiles = sum(isinstance(node, tuple) for node in processor.graph)
    retained = set(processor.ports_by_patch)
    if num_qubits is not None:
        retained |= {f"q_{i}" for i in range(num_qubits)}
    retained_patch_tiles = 0
    for name in retained:
        patch = processor.eng.patches.get(name)
        if patch is None and name not in processor.ports_by_patch:
            raise ValueError(f"layout has no data patch {name} for a register qubit")
        retained_patch_tiles += len(patch.cells) if patch is not None else 1
    return routing_tiles + retained_patch_tiles


class ValidationError(RuntimeError):
    """A schedule violated a validator rule while strict validation was on."""


def validate_run(processor, dag, results, config: BaselineRunConfig, total_steps, *,
                 consumed=None):
    """Validator report for one stage (pre/post pruning) of an in-tree run."""
    from harvest.eval.validate import validate

    from .cultivation import make_magic_source

    return validate(
        processor, dag, results,
        make_source=lambda terminals: make_magic_source(config.options, terminals, config.seed),
        total_steps=total_steps, consumed=consumed,
    )


def validation_metadata(pre, post) -> dict:
    """Flat validator columns (EVAL_PLAN B.3) from the pre/post-pruning reports."""
    reports = [r for r in (pre, post) if r is not None]
    return {
        "validated_pre": None if pre is None else pre["ok"],
        "validated_post": None if post is None else post["ok"],
        "n_violations": sum(r["n_violations"] for r in reports) if reports else None,
        "first_violation_rule": next(
            (r["first_violation_rule"] for r in reports if r["first_violation_rule"]), None
        ),
        "validate_time_s": sum(r["validate_time_s"] for r in reports) if reports else None,
        "validation_pre": pre,
        "validation_post": post,
    }


class HarvestAdapter(BaselineAdapter):
    """HARVEST through the common schema.

    ``config.options`` may carry ``harvest_config`` (HarvestConfig fields that
    differ from the defaults), ``scheduler`` (``"steiner_tree"``: the
    sequential ablation baseline; ``"harvest_reuse"``: HARVEST-2 with
    ``reuse_cost``, see ``harvest.routing.scheduler_reuse``) and a magic model (``magic_model`` plus its
    mean, see ``harvest.baselines.cultivation``).  Keyword arguments besides
    ``dag``/``layout_engine``: ``preprocessing_runtime_s``, ``phase_times``
    (front-end phase timings), ``placement_cost``, ``validate`` (default
    True), ``strict_validate`` (raise ``ValidationError`` on a violation) and
    ``sink`` (a dict that receives the processor, results and magic source).
    """

    name = "harvest"

    def run(self, config: BaselineRunConfig, **kwargs: Any) -> BaselineResult:
        dag = kwargs.get("dag")
        layout_engine = kwargs.get("layout_engine")
        preprocessing_runtime = float(kwargs.get("preprocessing_runtime_s", 0.0))
        if dag is None or layout_engine is None:
            raise ValueError("HarvestAdapter.run requires dag= and layout_engine=")

        from harvest.eval.instrument import (
            initial_area, process_peak_rss_mb, retained_area, schedule_stats,
        )
        from harvest.eval.routes import route_entries, schedule_entry
        from harvest.routing.config import HarvestConfig
        from harvest.routing.processor import DAGProcessor

        from .cultivation import make_magic_source

        harvest_config = HarvestConfig.from_mapping(config.options.get("harvest_config"))
        mode = config.options.get("scheduler", "harvest")
        if mode not in ("harvest", "steiner_tree", "harvest_reuse"):
            raise ValueError(f"unknown HARVEST scheduler mode {mode!r}")
        validate = kwargs.get("validate", True)

        from harvest import faults

        initial_patches = layout_patch_count(layout_engine)
        faults.reset_activations()
        started = perf_counter()
        processor = DAGProcessor(layout_engine=layout_engine)
        source = make_magic_source(config.options, processor.magic_terminals, config.seed)
        if source is not None:
            # Every magic patch prepares one state at a time (see
            # harvest.baselines.cultivation for the supply models).
            processor.magic_source = source
        if mode == "harvest_reuse":
            # HARVEST-2: reuse cost on unused tiles (harvest.routing.scheduler_reuse).
            from harvest.routing.scheduler_reuse import ReuseCost, process_dag_with_harvest_reuse

            results = process_dag_with_harvest_reuse(
                processor, dag, config=harvest_config,
                cost=ReuseCost.from_mapping(config.options.get("reuse_cost")))
        else:
            results = processor.process_entire_dag(dag, mode=mode, config=harvest_config)
        schedule_time = perf_counter() - started
        metadata = dict(getattr(processor, "_scheduling_metadata", {}))
        completed = bool(metadata.get("completed", False))
        length = metadata.get("total_elapsed_steps") if completed else None
        consumed = source.get_stats()["total_consumed"] if source is not None else None

        validation_pre = validation_post = None
        validate_time = 0.0
        if validate and completed:
            validation_pre = validate_run(processor, dag, results, config, length,
                                          consumed=consumed)
            validate_time += validation_pre["validate_time_s"]

        prune_started = perf_counter()
        pruning_stats = processor.prune_after_scheduling(results) if completed else None
        post_patches = (
            processor_patch_count(processor, config.num_qubits) if completed else None
        )
        prune_time = perf_counter() - prune_started
        if validate and completed:
            # Pruning never reschedules: the same schedule must still be valid
            # on the pruned routing graph (and keep its length).
            validation_post = validate_run(processor, dag, results, config, length,
                                           consumed=consumed)
            validate_time += validation_post["validate_time_s"]
        scheduler_runtime = schedule_time + prune_time
        runtime = preprocessing_runtime + scheduler_runtime
        wirelength = sum(len(result.get("steiner_edges", ())) for result in results)
        schedule = [schedule_entry(dag, result) for result in results
                    if result.get("success", True)]
        routes = route_entries(dag, results)
        phase_times = dict(kwargs.get("phase_times") or {})
        phase_times.update({
            "t_schedule_s": schedule_time,
            "t_prune_s": prune_time,
            "t_validate_s": validate_time,
        })
        area = {"initial": initial_area(layout_engine)}
        if completed:
            area["pruned"] = retained_area(processor, config.num_qubits)
        instrumentation = schedule_stats(
            dag, results, length, len(processor.magic_terminals),
            step_log=getattr(processor, "_step_log", None),
        )
        instrumentation["magic_wait_cycles"] = (
            source.get_stats()["total_wait_cycles"] if source is not None else 0
        )
        instrumentation["magic_consumed"] = consumed
        instrumentation["placement_cost"] = kwargs.get("placement_cost")
        validation = validation_metadata(validation_pre, validation_post)
        sink = kwargs.get("sink")
        if sink is not None:
            # Callers that post-process the schedule (replay, re-scheduling on
            # the pruned footprint) get the pruned processor and the results.
            sink.update(processor=processor, results=results, source=source)
        if kwargs.get("strict_validate") and validation["n_violations"]:
            raise ValidationError(
                f"{config.circuit}/{config.variant}: {validation['n_violations']} violation(s); "
                + "; ".join((validation_pre or validation_post or {}).get("messages", [])[:3])
            )
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
            seed=config.seed if source is not None else None,
            upstream_version=config.upstream_version,
            command=f"DAGProcessor.process_entire_dag(mode={mode!r}, config=HarvestConfig(**"
                    f"{harvest_config.overrides()!r}))",
            notes=[
                "Pruning is applied only after scheduling and does not change schedule length.",
                "Both initial and post-pruning patch counts/space-time values are reported.",
                "Post-pruning patches keep every register qubit's data patch, even if no product touches it.",
                "Peak memory is null; metadata.process_peak_rss_mb is the process-wide peak so far.",
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
                "routes": routes,
                "pruning": pruning_stats,
                "harvest_config": harvest_config.to_dict(),
                "scheduler_mode": mode,
                "runtime_breakdown_s": {
                    "frontend_and_layout": preprocessing_runtime,
                    "scheduling_and_pruning": scheduler_runtime,
                },
                "phase_times_s": phase_times,
                "area": area,
                "instrumentation": instrumentation,
                "validation": validation,
                "fault": faults.current(),
                "fault_activations": faults.activations(),
                "process_peak_rss_mb": process_peak_rss_mb(),
                "space_time_definition": {
                    "initial": "initial_logical_patches * schedule_length",
                    "post_pruning": "post_pruning_logical_patches * schedule_length",
                    "primary_field": "post_pruning",
                },
            },
        )
