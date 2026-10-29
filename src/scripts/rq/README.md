# RQ1–RQ5 experiments

Scripts for the five research questions. Every trial appends one row to
`results/rq/<exp>/<exp>.csv` and fsyncs it as soon as it finishes, so an
interrupted run (laptop closed, unit stopped, OOM) loses only the trials in
flight. Rerunning a command skips every row already in the CSV and retries
rows with `status=error`. When the data is read, the newest row for a
`run_key` wins.

## Suite

`prepare_suite.py` → `results/rq/suite.csv`

* Selection: every circuit in `benchmark_circuits/qasm` with < 100 qubits,
  except the `qv`, `chemical` and `feynman` folders, plus every Ising circuit
  of any size (n10–n98, n400–n440) and every wstate circuit (n3–n380). QASMBench uses `<name>/<name>.qasm`.
* Every compiler gets the same Clifford+T input. Inputs from earlier
  evaluations are reused. Missing ones are lowered with
  `prepare_family_benchmarks.py` into `results/rq_inputs/<family>/`: exact
  lowering where possible, otherwise `compile_cliffordt -e 1e-3`, with a T
  budget of 40k.
* Status per circuit: `ok`, `clifford_only` (nothing to schedule),
  `over_budget`, or `failed` (for example, a mid-circuit measurement). The
  unusable circuits are listed with their reason.
* Tiers:
  * `core` (RQ1/RQ5): all usable circuits, except that the dense qaoa/qft
    size scans are sampled at N ∈ {4,6,8,10,12,14,16,20,24,28,32,40,48,56,64,72,80,90,99}
    and random circuits keep seeds 1–2.
  * `extra`: the remaining qaoa/qft sizes and random seeds. Run them with
    `--rq1-tiers core,extra`.
  * `small` (RQ2–RQ4): every Ising circuit below 100 qubits, plus up to three
    sizes per family with 100 ≤ ops ≤ 6000.

## Run

```bash
python src/scripts/rq/prepare_suite.py            # once
python src/scripts/rq/run_all.py --jobs 5         # all five, one pool, cheapest first
python src/scripts/rq/run_all.py --plot-only      # redraw every table/figure from the CSVs
python src/scripts/rq/rq2_ablation.py --jobs 4    # or a single experiment
```

The long run uses a memory-capped user unit:
`systemd-run --user --unit=harvest-rq -p MemoryMax=7G ... run_all.py --jobs 5`.
Watch it with `journalctl --user -u harvest-rq -f` or `results/rq/run_all/run.log`.

Common settings:
* Layout templates: `ring`, `blocks4` and `dense4`, all with one-tile patches.
  Full Harvest uses circuit-aware placement (cached per circuit/template in
  `results/rq/cache/placement`).
* Magic models: instant; cultivation (Exp, mean 3 cycles per state per port);
  distillation μ = 4 and 10 (fixed latency); and two models new for RQ4-B:
  `multiplexed` and `lognormal`.
* Per-run timeout: 3600 s, recorded as `status=timeout`. A crashed worker is
  retried once, then recorded as `status=crashed`.

| exp | script | what runs | output figures/tables |
|---|---|---|---|
| rq1 | `rq1_sota.py` | Harvest, EAF (Harvest layout, pruned pricing), PureMagic (bus / magic routing), Litinski (3 blocks), O3LS (3 sizes); instant (1 run) + cultivation (10 seeds); blocks4 template; `core` tier. For n ≥ 400, EAF and O3LS run 3 cultivation seeds (cost). Ising n ≥ 400 and every wstate size also run `cultivation_pm` (PureMagic's published rate λ = 0.0387, mean ≈ 25.8 cycles/state): Harvest and EAF on blocks4 with 1 seed (4 h timeout), PureMagic with 10 seeds. | Table 1 (`rq1_table.csv/.tex`), `fig1_family_volume`, `fig1_per_circuit_volume`, `fig2a_volume_decomposition`, `fig1_ising_large_instant`, `fig1_ising_large_cultivation_pm`, `rq1_losses.csv`, `rq1_per_circuit.csv` |
| rq2 | `rq2_ablation.py` | full, −negotiation, −retry, −admission, −matching, −priority, −placement, −yfactor (−pruning derived) × 3 templates × instant + 3 cultivation seeds; sensitivity I_max ∈ {0,1,2,4,8}, α ∈ {1,2,4}, β ∈ {1,1.5,3} (blocks4) | `fig3_ablation`, `fig3_sensitivity`, `fig3_negotiation_cdf`, `rq2_ablation_summary.csv`, `rq2_placement_by_template.csv` |
| rq3 | `rq3_provisioning.py` | magic terminals M ∈ {1,2,4,8,16,32,all} × 3 templates; per M: schedule large (pruned and unpruned volume) and re-schedule from scratch on the right-sized layout (= the pruned footprint) | `fig4_provisioning`, `fig4_removed_tiles`, `rq3_summary.csv` |
| rq4 | `rq4_magic.py` | ring, M ∈ {1..64}; aware schedules under 6 models; an oblivious (instant-magic) schedule replayed under dist4/dist10/cultivation: V4 violations of the unmodified schedule + stall-inserted makespan (re-validated) | `fig5_magic_terminals`, `rq4_oblivious_vs_aware.csv`, `rq4_pluggability.csv/.tex` |
| rq5 | `rq5_validity.py` | 6 injected scheduler faults; 4 schedule corruptions × 20 trials (instant + cultivation) | `rq5_census.csv` (all schedules of rq1–rq4), `rq5_faults.csv`, `fig_rq5_faults`, `fig_rq5_scaling` (+ `rq5_scaling_fit.csv`), `fig_rq5_progress_bound` |

## Columns

Each row has the run key, experiment, status and error; the circuit features
(family, qubits, ops, critical path, parallelism, T fraction, Pauli weight,
Y fraction); compiler / variant / template / model / seed; T, the final area
(pruned for Harvest and EAF), the volume, the initial and pruned area and
volume, and routing/magic/data tiles before and after pruning; compile time
and its phases (place / schedule / prune / validate); validator results
before and after pruning, by rule (V1–V4); scheduler instrumentation
(negotiation iterations, reroutes, drops, fills, magic retries, deferrals,
stalls, idle steps, ops per step, route length, terminal utilisation, magic
wait cycles); and peak RSS. RQ-specific columns are listed in each script's
`COLUMNS`. Keys outside the header go to `extra_json`, and `read_rows`
merges them back in.

## Caveats

* PureMagic, Litinski and O3LS report T and area from their own models. They
  emit no routes in Harvest's schedule format, so only Harvest and EAF
  schedules go through the validator.
* PureMagic "instant" is approximated with λ = 1e6 and `--no-t-failures`.
* Baselines with several variants are represented by their best variant per
  circuit (lowest mean volume).
* `−admission`: without admission control, a product routed to an empty
  terminal stalls (its route is discarded after occupying routing space that
  step). With instant magic the variant equals full Harvest.
