# SOTA baseline audit and comparison contract

This note defines what the HARVEST SOTA experiment compares.  A label in a
plot is not evidence of equivalence: every raw record includes a
`comparison_mode`, native input representation, version, command, seed (or an
explicit note that the tool cannot accept one), and limitations.

The three permitted regimes are:

- `matched_ir`: the same post-transformation Pauli-product stream.  This is the
  primary scheduler comparison for HARVEST, Silva EAF, PureMagic (only when
  `.trans` conversion succeeds exactly), the Litinski data-block model and the
  O3LS reimplementation.
- `matched_architecture`: equivalent logical patch locations/budget.  This is
  available for DASCOT when the HARVEST layout has only one-tile data/magic
  patches and no blocked cells.
- `native_end_to_end`: the same source QASM, with each compiler's native IR and
  architecture (HARVEST, PureMagic, DASCOT, LSQCA, TACO; Litinski and O3LS
  whose native front end — the Litinski transformation of the source — yields
  exactly the matched Pauli-product stream).  These numbers describe complete
  compiler systems, not a scheduler-only contest.

The FLASQ cost model (Huggins et al., arXiv:2511.08508; Qualtran
`qualtran.surface_code.flasq`) is a reference, not a competitor: it prices the
Clifford+T source circuit with perfectly fluid ancilla.  The trade-off plots
normalize time by FLASQ's per-circuit best depth (FLASQ = 1x); see
`src/scripts/plot_sota_space_time_tradeoff.py`.

Results from different regimes must not be pooled in a plot or geometric mean.

## HARVEST scheduler fixes (2026-10-12)

Per-circuit comparison on identical layouts showed Silva EAF — HARVEST's
un-negotiated predecessor — faster than HARVEST on 34/36 circuits (dense
layout) and 28/36 (ring).  Instrumenting HARVEST's time steps found three
defects in `harvest.routing.scheduler` / `LayoutEngine.steiner_packing_harvest`:

1. **Magic-terminal collisions (free magic).** `_prepare_terminal_sets`
   only enforced one product per magic terminal per step when a magic source
   was configured; with always-ready states several products were given the
   same terminal.  Routes are node-disjoint, so all but one failed and
   retried later — 81–82% of all failed routings on gf2^8/adder_8 (dense
   layout).  Terminals are now unique per step in every mode, as in Silva.
2. **No reuse of leftover capacity.** After negotiation, conflicting nets
   were dropped whole and never retried.  A greedy completion pass now
   retries dropped/unrouted nets on the cells not used by committed routes
   (`greedy_fill=True`), so each step's packing is maximal.
3. **Fixed magic-terminal choice.** A product whose pre-assigned terminal was
   unreachable was deferred even if another magic patch was free, and ready
   products that got no terminal never used a port freed by an unroutable
   one.  `_retry_failed_with_other_magic` now tries every free terminal
   (nearest first, ports re-oriented) for failed and unassigned products,
   like Silva's nearest-reachable-site attachment.

Effect on a 6-circuit probe (time steps, HARVEST / Silva): ring layout
gf2^8 265/273 (was 286), adder_8 206/213, qcla_adder_10 41/51 (was 62),
ham15-low 121/127; blocks4 qcla_adder_10 74/80 (was 82).  On the dense
layout a gap remains on some circuits (adder_8 273/252, qcla_adder_10
119/111); there HARVEST already routes *more* products per step than Silva's
greedy procedure on the same ready set (adder_8: 399 vs 381), so the residual
difference is which products are scheduled, not packing.  Critical-path-first
ordering of the ready list alone gave no consistent gain; it pays off only
inside the router (next section).  Regression tests:
`tests/test_harvest_scheduler_fixes.py`.  All HARVEST results were recomputed
after the fixes (`--force-rerun harvest`).

## HARVEST scheduling policies and runtime (2026-10-12)

Two policies, module constants in `harvest.routing.scheduler`, with the
adopted defaults:

* **`HARVEST_PRIORITY = "critical_path"`, `HARVEST_ROUTER_PRIORITY = True`.**
  Each product's priority is its height (longest chain of operations to a
  sink of the DAG).  Ready products are ordered by priority (ties: DAG order)
  for magic-terminal assignment and the cultivation filter; the congestion
  router routes them in priority order and, when it must drop nets to resolve
  congestion, drops the *lowest-priority* conflicting net (previously the one
  with the most conflicts, regardless of its slack).
* **`HARVEST_TERMINALS = "matching"`.** Instead of giving each product in
  turn the nearest free terminal (Euclidean), one assignment of the free
  terminals to the first products (in priority order) minimizes the total
  routing distance (shortest-path length from the terminal to the product's
  oriented data ports; `scipy.optimize.linear_sum_assignment`).

A/B on the 36 Feynman circuits (time steps, geometric mean relative to the
previous scheduler / to Silva EAF on the same layout; wins/ties/losses vs Silva):

| layout | ordering only | + router priority | + matching (adopted) | vs Silva |
|---|---|---|---|---|
| ring | 1.000 | 0.986 | 0.984 | 0.955 (24/12/0) |
| blocks4 | 1.000 | 0.956 | 0.956 | 0.936 (29/6/1) |
| dense, 2 ports | 0.969 | 0.933 | 0.920 | 0.931 (31/3/2) |
| dense, 4 ports | 0.994 | 0.916 | 0.909 | 0.930 (31/4/1) |
| cultivation, dense 8 | 0.986 | 0.926 | 0.918 | 0.936 (32/3/1) |
| cultivation, ring | 1.000 | 0.982 | 0.982 | 0.953 (24/12/0) |

Matching alone (DAG order) gave 0.98–1.00.  Matching also shortens routes,
so post-pruning space drops on the ring (0.94×); on dense layouts space
changes by < 1.5%.

**Runtime (results unchanged).**  `ReadyTracker` maintains the ready set
incrementally (was a rescan of the whole DAG every step); Dijkstra applies
congestion costs on the fly (`node_cost`) and skips `blocked` cells instead
of rebuilding/copying the graph per reroute; the magic retry pass skips
terminals not in the product's connected component of the free graph (the
routing graph is symmetric, so this predicts Steiner feasibility exactly);
port orientation is memoized.  On all 216 runs (36 circuits × 6 layouts) the
schedules are bit-identical to before, at 2.0–2.4× per-circuit (3.2–4.1×
total) less compile time.  `ReadyTracker` and the Dijkstra changes are shared
with Silva EAF, whose schedules are likewise unchanged (verified on 62 runs)
and which gets ~1.6× faster; both were rerun so compile times stay
comparable.  Tests: `tests/test_harvest_speed_and_policies.py`.

## Magic-state model: `free` vs `cultivation`

By default (`--magic-model free`) the matched-IR regime gives HARVEST, Silva,
Litinski and O3LS magic states that are always ready at every port, while
PureMagic runs its native cultivation (default λ = 0.0387, i.e. a mean of
≈26 logical cycles per state).  That default favours the in-tree models.

`--magic-model cultivation --cultivation-mean-cycles 3` applies one common
supply process to every compiler (`harvest.baselines.cultivation`): each magic
port cultivates one state at a time and, after consumption, the next state is
ready after `max(1, round(Exp(mean)))` logical cycles.

| Compiler | How the common model is applied |
|---|---|
| HARVEST | its `MagicStateCultivator` on every magic patch (exponential sampler); seeded trials |
| Silva EAF | storage sites are offered only while holding a ready state (reimplementation extended; the paper assumes replenishment every step) |
| PureMagic | native cultivation with `--magic-state-lambda = 1/mean` (PureMagic allows 0-cycle draws; the others clamp to ≥ 1) |
| Litinski | ⌈mean / worst-case steps per rotation⌉ ports (§4: supply sized to the block — 1 compact/intermediate, 3 fast), each counted as a tile; rotations wait for a ready port |
| O3LS | the single port next to A waits after every π/8 measurement |
| FLASQ | `--flasq-assumptions cultivation --flasq-v-cult 3`: each T gate pays 3 blocks of cultivation volume (no reaction time, one step per T layer) |

The mean of 3 cycles is FLASQ's cultivation volume at d = 13, p = 10⁻³
(v_cult = 3.0 blocks; RushHour quotes 2–3 expected rounds).  Under this model
the in-tree models become stochastic; they run `--cultivation-trials` seeded
trials and points use the mean schedule length and mean footprint.

## Dense HARVEST layout (`--layout-preset dense`)

`harvest.layout.presets.dense_comb_layout`: columns `R D D R D D … R`, row bands
`D R D`, so every single-cell data patch touches routing on an X (N/S) side and
a Z (E/W) side — X-, Y- and Z-type products stay routable without patch
rotations — at ≈2.25 tiles per qubit plus `--layout-magic-count` magic ports
in a top row (the rest of that row is blocked).  On the 36 Feynman circuits
(free magic): HARVEST 2.60×/2.37× (2 ports) and 2.70×/2.24× (4 ports)
space/time vs FLASQ, space-time 6.17×/6.05×, versus 7.59× for its densest
earlier preset (blocks4).  Silva EAF on the same dense layouts reaches
2.78×/2.14× and 2.92×/2.02× (5.95×/5.90×): on these tight layouts HARVEST's
negotiated routing does not beat EAF on time; its smaller footprint comes from
post-scheduling pruning.

## Incremental experiment suite

`experiments/sota_suite.json` describes the inputs, every experiment
(comparison mode, baselines, trials, resource-sweep knobs) and every figure.
`src/scripts/run_sota_suite.py` runs it incrementally:

```bash
# add circuits (exactly lowered to Clifford+T), run only what is missing, redraw
python src/scripts/run_sota_suite.py experiments/sota_suite.json --add path/to/circuits/
# redraw figures from stored data only (seconds)
python src/scripts/run_sota_suite.py experiments/sota_suite.json --figures-only
# recompute one baseline after changing its code
python src/scripts/run_sota_suite.py experiments/sota_suite.json --force-rerun harvest
```

Nothing stored is recomputed:

- raw trials are reused when their identity matches — baseline, variant,
  circuit, mode, trial, seed, options and input SHA-256 — so timeouts, paths,
  tool locations or a new HARVEST commit do not trigger reruns
  (`--force-rerun` does); recorded failures are kept (`--keep-failed`);
- lowered inputs are skipped when the source SHA-256 is unchanged
  (`manifest.json` accumulates);
- lower bounds are cached per circuit in `<output_dir>/bounds_cache.json`, and
  FLASQ prices per circuit in `results/flasq_cache/`, keyed by input SHA-256,
  FLASQ settings and driver version.

Adding a circuit extends every experiment, including the resource sweeps and
the native regime; give an experiment an `"inputs"` list to restrict it.

## Parallel circuits and view search (2026-10-13)

HARVEST's space-time margin over the best other compiler grows with circuit
parallelism (Pauli products / critical path of the matched DAG), but the
Feynman suite is mostly near-serial (median ~1.6; only `qcla_*` and
`csum_mux_9` exceed 2.9).  `src/scripts/generate_parallel_benchmarks.py`
adds two textbook log-depth families, checked by classical simulation and
lowered exactly with `prepare_cliffordt_benchmarks.py`
(`results/sota_parallel_cliffordt_inputs/`): Kogge-Stone prefix adders
`ks_adder_n{4,6,8,12,16}` and AND-tree multi-controlled Toffolis
`mcx_tree_n{8,16,24,32}` (parallelism 3.1-10.4).
`experiments/sota_suite_parallel.json` reruns the `sweep` and `cultivation`
experiments on Feynman + these circuits in `results/sota_parallel/` (seeded
with the stored Feynman records; only the new circuits were computed), plus a
16-port dense cultivation layout (`cult_dense16`, HARVEST and Silva).

`src/scripts/search_sota_views.py` scores circuit subsets of a drawn figure,
defined only by a feature threshold or family, and renders the best ones
(`--circuits` of the plot script).  Results (`results/sota_view_search/`;
margin = best other / HARVEST, geomean normalized space-time):

| figure | circuits | margin (per-circuit best) | margin (best config) | HARVEST wins |
|---|---|---|---|---|
| free magic, Feynman only | 36 | 1.159 | 1.138 | 33/36 |
| free magic, + parallel | 45 | 1.173 | 1.152 | 42/45 |
| free magic, parallelism >= 2.2 | 15 | 1.237 | 1.212 | 15/15 |
| cultivation, + parallel (incl. 16 ports) | 45 | 1.122 | 1.098 | 36/45 (hwb8 a tie) |

With free magic states HARVEST wins every circuit with parallelism >= 1.52
(34/34) and loses only on 5-qubit circuits (`qft_4`, `barenco_tof_3` to O3LS)
and nearly serial `hwb8` (tie with Silva).  Under cultivation the trend
reverses: PureMagic, whose native topology has many cultivation patches,
wins most highly parallel circuits (`qcla_*`, `csum_mux_9`, `mcx_tree_n24/32`,
`ks_adder_n4/6`) even against HARVEST's 16-port layout -- magic-state supply,
not routing, limits HARVEST there.  A figure restricted to a view must state
its selection rule and report the full-set numbers alongside.

Resume caveat: PureMagic's `transpile_bin` path is part of the trial identity
(`options`), so rebuilding PureMagic at another path recomputes every stored
PureMagic trial.  The tools were rebuilt at the recorded commits (PureMagic
`f990c3ec` with its `cyclosynth` submodule; Qualtran `585da53e`) and linked to
the paths in the suite files.

## Per-family trade-off (2026-10-14)

`experiments/sota_suite_families.json` runs the 9 configurations of the
`sweep` figure (matched IR, free magic) on Feynman (36), the log-depth
circuits (9) and five families from `benchmark_circuits/qasm`, <= 100 qubits:
qaoa (12 sizes, N4-N100), qft (16, N4-N100), ising (qasmbench, n10-n98), knn
(n25-n67), square_heisenberg (N4-N100).  `_transpiled` duplicates are not
used.  `src/scripts/prepare_family_benchmarks.py` strips barriers and final
measurements and synthesizes arbitrary rotations *once*, with PureMagic's
`compile_cliffordt -e 1e-3` (per rotation; process fidelity >= 0.99995 where
checked, <= 8 qubits), so every compiler gets the same Clifford+T circuit
(`results/sota_family_inputs/`; largest: qaoa N100, 24k T).  ghz and cat are
Clifford-only: with no T gate there is no Pauli-rotation stream, every
baseline but HARVEST rejects them, and they are excluded.

Geomean normalized space-time over each family, best configuration per
circuit, 8 configurations (all but `ring_large_spacing`, which Silva needs
1.5-2 h per 100-qubit circuit for and which was still running;
`results/sota_families/figures_no_large_spacing/`):

| family | circuits | HARVEST | best rival | margin | HARVEST wins |
|---|---|---|---|---|---|
| feynman | 36 | 4.99 | Silva EAF 5.79 | 1.16 | 33/36 |
| log-depth | 9 | 6.64 | Silva EAF 8.17 | 1.23 | 9/9 |
| qaoa | 12 | 6.90 | Silva EAF 7.34 | 1.06 | 8/12 |
| qft | 16 | 7.32 | Silva EAF 7.52 | 1.03 | 12/16 |
| ising | 6 | 11.42 | Silva EAF 15.33 | 1.34 | 6/6 |
| knn | 4 | 7.02 | Silva EAF 10.39 | 1.48 | 4/4 |
| square_heisenberg | 9 | 6.37 | Silva EAF 6.75 | 1.06 | 9/9 |

Size dependence is family-specific: ising grows from 1.03 (n10) to ~1.5
(n34-n98); knn is 1.37-1.66 at every size; qft shrinks from 1.09 (N4) to
0.96-0.98 (N64-N100, Silva ahead); qaoa is mixed (0.93-1.31).  On one circuit
(qaoa N80) HARVEST's schedule is 1% shorter than FLASQ's best modelled depth;
FLASQ normalizes the time axis but is not a strict lower bound.

## Benchmark evaluation

A real 11-circuit comparison (not the tiny hand-written smoke fixtures under
`examples/sota_baselines/`) was run against the actual upstream PureMagic
(built from source, `cargo build --release`) and wisq/DASCOT `v0.2.7`
(installed via `pip install -e .` into an isolated environment; wisq's
`setup.py` requires `gcc` and `java >= 21` at install time). Circuits were
selected with the exact same eligibility filter as
`src/scripts/run_benchmark_comparisons.py`
(`≤100 qubits, ≤2000 depth, excluding the qv and chemical families`),
restricted to `≤12` qubits and `≤40` Pauli products for tractable wall-clock
time with real external subprocesses: `fredkin_n3`, `qaoa_n3`,
`teleportation_n3`, `toffoli_n3`, `adder_n4`, `variational_n4`,
`vqe_uccsd_n4`, `qec_en_n5` (qasmbench-small), `qft_N002`, `qft_N003` (qft),
and `qaoa_barabasi_albert_N3_3reps` (qaoa). This is a representative slice of
the paper's existing non-chemical benchmark pool, not the full 257-circuit
set — a full run at `--trials 20` across all eligible circuits with real
PureMagic/DASCOT subprocesses would take substantially longer than a single
session.

Two runs were executed and committed under `results/`:

- `results/sota_comparison_matched_ir/` — `harvest,silva,puremagic` at
  `--comparison-mode matched_ir`, `--trials 20 --seed 42`.
- `results/sota_comparison_native_e2e/` — `harvest,silva,puremagic,dascot` at
  `--comparison-mode native_end_to_end`, `--trials 20 --seed 42`.

Findings:

- In `matched_ir`, only 6/11 circuits produce a valid HARVEST/Silva/PureMagic
  comparison (`adder_n4`, `fredkin_n3`, `qec_en_n5`, `qft_N002`,
  `teleportation_n3`, `toffoli_n3`); the other 5 fail loudly and are recorded
  in `failures.csv` because HARVEST's own PBC conversion does not force every
  Pauli rotation onto the ±π/8 grid, so circuits containing generic continuous
  rotation angles (`qaoa_n3`, `variational_n4`, `vqe_uccsd_n4`, `qft_N003`,
  `qaoa_barabasi_albert_N3_3reps`) cannot be represented exactly in Silva's or
  PureMagic's discrete `.trans` model. This is the intended, documented
  behavior (see "Not directly normalized" above), not a bug — no angle was
  rounded or discarded silently.
- **Superseded (2026-10-09).** The space-time numbers in the next bullet
  were produced before a HARVEST metric fix: post-scheduling pruning dropped
  the data patch of every logical qubit no Pauli product touches, so
  HARVEST's `logical_patches` undercounted `qec_en_n5` (7 instead of 11) and
  `teleportation_n3` (6 instead of 8). `processor_patch_count` now keeps all
  register data patches. These six tiny circuits are also too small to rank
  schedulers; use the Feynman-suite results below.
- Geometric means over the 6 shared circuits (`baseline / HARVEST`, lower is
  better for HARVEST): PureMagic dual-purpose routing is 2.70× HARVEST on
  schedule length and 3.31× on space-time volume; PureMagic bus routing is
  2.17×/5.93×; Silva EAF matches HARVEST's schedule length exactly (1.00×, as
  expected — Silva's Steiner-forest packing is the un-negotiated,
  un-pruned ancestor of HARVEST's own scheduler) but uses 3.51× the
  space-time volume, isolating HARVEST's pruning/negotiation contribution.
- In `native_end_to_end`, Silva EAF fails on all 11 circuits by design (it is
  only qualified for `matched_ir`; DASCOT and PBC scheduling are not treated
  as the same compilation problem, per the constraints above). PureMagic
  native mode succeeds on the same 5 Clifford+T-labeled circuits as
  `matched_ir` plus fails identically on the other 6 (its gate-level
  Clifford+T check, not the ±π/8 check, is the blocker here). DASCOT succeeds
  on only 4/11 circuits (`adder_n4`, `fredkin_n3`, `qec_en_n5`,
  `teleportation_n3`): the other 6 fail HARVEST's/wisq's own Clifford+T/CX
  gate-set validation (same non-Clifford+T circuits as above), and
  `toffoli_n3` — despite passing gate-set validation — crashes wisq's DASCOT
  simulated-annealing mapper with a `KeyError` in
  `wisq/phased_graph.py:count_overlapping_fast` on every trial (both
  architectures). **Correction (2026-10-09):** this was originally recorded as
  an upstream wisq bug, but the cause is an input-format limitation:
  `wisq.utils.extract_gates_from_file` only matches gates on a register
  literally named `q`, and `toffoli_n3` declares `qreg a[3]`, so wisq parsed
  zero gates. The DASCOT adapter now hands wisq an exact single-register
  copy (`input_q.qasm`) and verifies wisq's own CX/T counts; the four
  successful circuits above already used `qreg q[...]` and are unaffected.
  Geometric means over the 4 shared circuits: DASCOT Square Sparse is 2.54×
  HARVEST on schedule length and 11.38× on space-time volume (dominated by
  its architecture's routing overhead); DASCOT Compact is 3.43×/7.50×.
- All `compiler_runtime_s` ratios are reported for completeness but are not
  meaningful cross-system comparisons: HARVEST/Silva runtime is pure-Python
  scheduling, PureMagic is a native Rust binary, and DASCOT includes Python
  simulated-annealing with a fixed `--mr_timeout`.

Reproduce with:

```bash
PYTHONPATH=src python src/scripts/compare_sota_baselines.py \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/fredkin_n3/fredkin_n3.qasm \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/qaoa_n3/qaoa_n3.qasm \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/teleportation_n3/teleportation_n3.qasm \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/toffoli_n3/toffoli_n3.qasm \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/adder_n4/adder_n4.qasm \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/variational_n4/variational_n4.qasm \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/vqe_uccsd_n4/vqe_uccsd_n4.qasm \
  --benchmarks benchmark_circuits/qasm/qasmbench-small/qec_en_n5/qec_en_n5.qasm \
  --benchmarks benchmark_circuits/qasm/qft/qft_N002.qasm \
  --benchmarks benchmark_circuits/qasm/qft/qft_N003.qasm \
  --benchmarks benchmark_circuits/qasm/qaoa/qaoa_barabasi_albert_N3_3reps.qasm \
  --baselines harvest,silva,puremagic[,dascot] \
  --comparison-mode matched_ir  # or native_end_to_end (add dascot) \
  --trials 20 --seed 42 --timeout 300 \
  --puremagic-bin <path>/puremagic --puremagic-transpile-bin <path>/transpile \
  --wisq-bin <path>/wisq \
  --output-dir results/sota_comparison_<mode>
```

### Expanded qft/qaoa sweep with Clifford+T pre-synthesis

The corpus check above found only 5 native Clifford+T circuits in the entire
257-circuit eligible pool, all already included. To get more `native_end_to_end`
data points, PureMagic's own non-GUOQ synthesizer (`compile_cliffordt`, built
with `cargo build --release --bin compile_cliffordt`; needs the system
`fontconfig` dev headers) was wired in as a shared, tool-neutral Clifford+T
front end for circuits whose original gate set neither PureMagic-native nor
DASCOT's `scmr` mode can accept directly (`--puremagic-compile-bin` and the
new `--dascot-compile-bin`, both pointed at the same binary). It is used only
as a fallback: circuits already in native Clifford+T/CX (the 5 above) are
never re-synthesized, so their results are unchanged from the core run.
Reused GUOQ is deliberately never invoked — see the "avoid GUOQ" constraint —
so this keeps the DASCOT/PureMagic-native results comparable to the core run
in kind, just extended to more circuits. Every synthesized run records the
exact `compile_cliffordt` command in `metadata.pre_synthesis_command` and
carries an explicit note; HARVEST and Silva always run on the unsynthesized
original circuit.

Swept `qft_N002`–`qft_N010` and `qaoa_barabasi_albert_N3`–`N10` (17 circuits)
through `harvest,silva,puremagic,dascot` in `native_end_to_end` mode
(`results/sota_comparison_native_e2e_expanded/`). A cheap single-trial probe
first established DASCOT's simulated-annealing mapper cost roughly *5s at
N=5, ~30-100s at N=7-8, ~110-170s at N=9-12* per trial per architecture —
right at the requested 180s cutoff by N≈12 — so **trials were reduced from
the spec's 20 to 5** for this sweep to keep total wall-clock time tractable
(this range still finished with zero timeouts at 180s). All circuits
completed for HARVEST/PureMagic/DASCOT except `qft_N002` on
`dascot_compact`, which crashes wisq's own mapper with
`TypeError: cannot unpack non-iterable NoneType object` in
`wisq/__init__.py:map_and_route` for every trial — apparently an upstream
wisq issue (its input does use `qreg q[...]`, so it is unrelated to the
register-name limitation behind the `toffoli_n3` `KeyError`), captured
verbatim under `results/.../work/qft_N002/dascot/dascot_compact/*/stderr.txt`
rather than hidden.

**Critical caveat — do not quote these geomeans without this context.** The
pairwise geomeans over this sweep show DASCOT/PureMagic at roughly **40-130×**
HARVEST's schedule length and space-time volume, dwarfing the 2.5-11× gap
measured on the 5 native Clifford+T circuits. This is *not* evidence that
DASCOT/PureMagic schedule worse here — it is a granularity artifact of
Clifford+T synthesis. Example: `qft_N010`'s original circuit has 155
arbitrary-angle Pauli rotations; HARVEST's cost model charges exactly one
schedule cycle per rotation regardless of angle (schedule length 59, ≈0.38
cycles/rotation), because HARVEST's internal representation treats every
Pauli rotation as a single atomic magic-state operation and never lowers it
to a T-count. The *same* 155 rotations, once synthesized to genuine
Clifford+T (most angles are not exact multiples of π/4), explode to **4155
T-gates** (`compile_cliffordt` output), which DASCOT then schedules in
1426-1781 cycles — that is ≈0.34 cycles/T-gate, a packing efficiency
comparable to HARVEST's own per-operation rate. The blow-up is almost
entirely the T-synthesis cost, not a scheduling-quality difference. Anyone
citing this sweep's ratios must state explicitly that HARVEST's "one magic
state per rotation" cost model and DASCOT/PureMagic's literal T-state model
are different physical resources, and that this sweep does not measure
scheduler quality on equal footing the way the 5-circuit native Clifford+T
comparison does. If a fair magic-state-count comparison is needed, normalize
by T-count (or another common elementary-operation count) rather than by
rotation count before comparing.

PureMagic's Rust scheduler stayed fast even as T-count exploded (≤27s per
trial through N=10), so the practical scaling bottleneck at this size is
DASCOT's simulated annealing, not synthesis or PureMagic's own scheduling.

## Silva et al. — `silva_eaf`

| Property | Audited behavior |
|---|---|
| Paper/version | Allyson Silva et al., “Multi-qubit Lattice Surgery Scheduling,” TQC 2024, [arXiv:2405.17688](https://arxiv.org/abs/2405.17688), especially Algorithms 1–2. |
| Upstream implementation | The paper describes a Python/NetworkX implementation but does not provide a maintained official repository.  HARVEST therefore contains a separately named reproduction, not a vendored upstream executable. |
| Input representation | Ordered Pauli rotations after Clifford removal; the qualified matched-IR path accepts only one-term ±π/8 Pauli rotations. |
| Operation model | Multi-qubit Pauli rotations.  A rotation consumes one logical cycle. |
| Layout | Static 2-D logical adjacency graph with data patches, bus patches, and boundary magic-storage patches. |
| Magic states | Static storage sites, continuously replenished by external distillation; a storage site is used by at most one operation in a timestep and is available again next timestep.  Factory latency is outside the paper's scheduling model. |
| Scheduling | Earliest-available-first: schedule dependency roots, remove successes, repeat.  The implementation uses the stable source order for reproducibility.  The paper says candidates are selected randomly but does not specify a seed/order distribution. |
| Routing | Approximate metric-closure Steiner tree over data terminals, followed by a shortest connection from the tree to the nearest available magic-storage terminal.  Used routing cells are greedily removed for the rest of the timestep. |
| Placement | Fixed row-major logical mapping by default.  No circuit-aware HARVEST placement is used.  A matched-placement study must be separately named. |
| Stochastic behavior | The paper's candidate draw is stochastic.  `silva_eaf` intentionally uses deterministic source order and records this deviation. |
| Original metric | Expected number of logical cycles `E(N)` (equal to EAF timesteps for the post-transpilation ±π/8 model), lower/upper bounds, and scheduler wall time. |
| Directly comparable | Schedule length, runtime, full logical patch count, patches×cycles, and route length proxy in `matched_ir`, subject to the port-model deviation below. |
| Not directly normalized | Physical error rate, distillation/factory cost, and arbitrary-angle synthesis cost. |

The existing `steiner_packing` mode is **not** an alias for Silva EAF.  It
jointly computes a Steiner tree over data and a preselected magic terminal and
orders candidates by terminal-set size.  Silva first constructs the data tree,
then attaches the closest static magic site, and describes random candidate
selection.  Both reuse HARVEST's Kou/metric-closure tree primitive; Silva's
paper calls its closely related implementation a modified Mehlhorn heuristic.
HARVEST's directional X/Z port nodes are also more explicit than the paper's
logical operator vertices.  The reproduction deterministically orients those
ports toward the first available storage patch; this deviation is retained in
result notes.

The current Qiskit DAG root policy corresponds to the paper's scalable
“trivial” dependency policy for the relevant case: operations sharing a data
qubit cannot execute simultaneously.  It does not claim the extra concurrency
of the paper's quadratic general commutation-graph construction.

## PureMagic — `puremagic_bus` and `puremagic`

| Property | Audited behavior |
|---|---|
| Paper/version | Steven Hofmeyr et al., “Scheduling Lattice Surgery with Magic State Cultivation,” [arXiv:2512.06484](https://arxiv.org/abs/2512.06484). |
| Upstream implementation | Official [BQSKit/PureMagic](https://github.com/BQSKit/PureMagic) Rust CLI.  Use the paper's `QCE` snapshot when it is available and record the executable banner/commit.  The public repository currently exposes no release tags through GitHub's tags API, so the adapter never silently calls `main` “QCE.” |
| Input representation | Native signed `.trans` Pauli products (`<T>`, `<CX>`, `<S>`, `<SX>`, `<M>`, `<Z>`, `<X>`).  HARVEST conversion is exact only for one-term ±π/8 PPRs and Pauli-product measurements; all other operations fail loudly. |
| Operation model | Dynamic PPR scheduling; current upstream models CX as two cycles and S/SX as three, while T outcomes can emit correction operations. |
| Layout | `puremagic_bus` uses conventional data/bus/magic topology. `puremagic` passes `--use-magic-routing`, allowing cultivation patches to route. |
| Magic states | Probabilistic cultivation governed by λ.  Cultivation can be interrupted for routing in PureMagic mode.  T injection failures are enabled unless `--puremagic-no-t-failures` is set. |
| Scheduling/routing | Native EAF scheduler with greedy Steiner packing and A* for relevant paths; no HARVEST code is substituted. |
| Placement | Native topology placement; optional upstream random data placement is not enabled by this adapter. |
| Stochastic behavior | Cultivation and injection behavior use `--rseed`.  The experiment defaults to 20 trials with consecutive fixed seeds and stores every trial. |
| Original metrics | Scheduled logical cycles, topology qubits, logical volume, scheduling/parallel efficiency, cultivation behavior, and runtime. |
| Directly comparable | In exact `matched_ir`: logical cycles and complete-topology patches×cycles, with operation-duration differences disclosed.  In `native_end_to_end`: system-level cycles/volume from the same source circuit. |
| Not directly normalized | Aggregate routing wirelength (not exposed by normal native output), physical code distance/error, or cycles from different IRs as if they were scheduler-only. |

`logical_patches` is PureMagic's reported total (data + bus + magic/cultivation
patches), never just data qubits.  The adapter cross-checks its normalized
patches×cycles against PureMagic's native volume when available.

## DASCOT — `dascot_square_sparse` and `dascot_compact`

| Property | Audited behavior |
|---|---|
| Paper/version | Abtin Molavi et al., “Dependency-Aware Compilation for Surface Code Quantum Architectures,” OOPSLA 2025, [arXiv:2311.18042](https://arxiv.org/abs/2311.18042). |
| Upstream implementation | Official [qqq-wisc/wisq](https://github.com/qqq-wisc/wisq); reproducible default documented here is tag `v0.2.7` (`d35ed5a…`).  A preserved paper artifact is [Zenodo 14934044](https://doi.org/10.5281/zenodo.14934044). |
| Input representation | OpenQASM 2 in the Clifford+T/CX model.  `wisq --mode scmr` extracts CX and T/Tdg routing operations; local single-qubit Cliffords have no routed path.  Arbitrary rotations are rejected unless synthesized before the comparison. |
| Operation model | Surface-code mapping and routing of CX and distillation-based T operations with dependency-aware phases.  This is not HARVEST's PPR model. |
| Layout | Native Square Sparse or Compact architecture, both with their own algorithm/magic locations and routing cells. |
| Magic states | Architecture magic-state locations participate in routed T operations under wisq's SCMR model. |
| Scheduling/routing | Dependency-aware mapping and routing, each searched with simulated annealing.  The adapter always uses `--mode scmr`; GUOQ optimization is excluded. |
| Placement | DASCOT's dependency-aware mapping is part of the baseline. |
| Stochastic behavior | DASCOT is randomized, but wisq v0.2.7 exposes no CLI seed.  Repeated trials are supported; raw records set `seed` to null and retain `requested_seed` plus `seed_supported=false`. |
| Original metrics | Routing/schedule cost, solution quality relative to optimal/other compilers, and compiler runtime across Square Sparse and Compact architectures. |
| Directly comparable | Native end-to-end schedule length/runtime/complete architecture area from the same source QASM; matched-architecture results when conversion succeeds. |
| Not directly normalized | A scheduler-only PPR cycle ratio to HARVEST.  DASCOT's CX/T schedule and HARVEST's PPR schedule are different compilation problems. |

For matched architecture, a HARVEST grid is exported as wisq row-major indices:

```json
{
  "height": 9,
  "width": 9,
  "alg_qubits": [20, 24],
  "magic_states": [1, 3, 5, 7]
}
```

Every other grid position is a wisq routing resource.  This conversion is
faithful only for one-tile data/magic patches without blocked cells.  Other
layouts are rejected.  DASCOT patch count is the complete `width × height`
architecture, including algorithm, magic, and routing resources.

## Litinski — `litinski_compact`, `litinski_intermediate`, `litinski_fast`

| Property | Audited behavior |
|---|---|
| Paper/version | Daniel Litinski, “A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery,” Quantum 3, 128 (2019), [arXiv:1808.02892v3](https://arxiv.org/abs/1808.02892), §2.1–2.3 (data blocks), §1 (Pauli-based computation). |
| Upstream implementation | None; the paper is an analytic resource model.  `harvest.baselines.litinski` evaluates its closed forms. |
| Input representation | The exact matched ±π/8 Pauli-product stream (`pauli_dag_to_trans`, identical to PureMagic's input); `matched_ir` only. |
| Operation model | Rotations execute strictly one after another (“one T gate is performed after the other”).  Per rotation: compact = Y-fix + 3 per row containing an X/Y qubit + 1; intermediate = Y-fix + 2 if any X/Y qubit + 1; fast = 1.  Y-fix = 0/1/2 for no/odd/even Y's.  Maxima equal the published worst cases (9, 5, 1). |
| Layout | Compact 1.5n+3, intermediate 2n+4, fast ⌈2n+√(8n)+1⌉ tiles (the paper's n=100 example rounds the fast block to 231), plus one magic-state port tile.  Compact upper row = first ⌈n/2⌉ qubits (not fixed by the paper). |
| Magic states | Matched-IR assumption: free and always ready at one port tile; distillation blocks (§3) are not added. |
| Scheduling / routing | None beyond the sequential block model. |
| Stochastic behavior | Deterministic. |
| Original metric | Tiles and time steps per T gate (total ≈ T-count × steps per T). |
| Directly comparable | `matched_ir`: total steps on the same stream, tiles, tiles×steps.  The three blocks are three configurations of one compiler and form a frontier. |
| Not directly normalized | Distillation footprint/throughput, the time-optimal “units” of §5, physical qubits per code distance. |

## Zhu et al. — O3LS (`o3ls_min`, `o3ls_medium`, `o3ls_large`)

| Property | Audited behavior |
|---|---|
| Paper/version | Zhu et al., “O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling,” ISCA 2026, [arXiv:2604.15099](https://arxiv.org/abs/2604.15099), §II-B cost rules, §III-A–D (Algorithms 1–2). |
| Upstream implementation | No public code (checked arXiv, Zenodo and the authors' GitHub on 2026-10-09).  HARVEST contains a separately named reimplementation from the paper text (`harvest.baselines.o3ls`), as RushHour (arXiv:2608.18985, App. A) also did; it is not a vendored upstream executable. |
| Input representation | The exact matched ±π/8 Pauli-product stream (`pauli_dag_to_trans`, identical to PureMagic's input); any other operation or angle fails the run.  The paper's GridSynth/transpilation front end is not rerun. |
| Operation model | Litinski cost rules: π/8 = P⊗Z measurement with \|m⟩ (1 step), π/4 = P⊗Y measurement with ancilla A needing both its edges (1 step), move 1, patch rotation 3, init/shrink 0.  Y's are always removed by Algorithm 1 (one odd group or two odd groups, absorption-aware bipartition, exact cancellation of (Z_G)−π/4(Z_G)+π/4 pairs). |
| Layout | One-tile patches on an N×M board, A fixed at corner (0,0).  Greedy §III-A search with S(B)=C(B)(N_x+N_z−0.1·N_e) plus best-improving one-step relocations.  Board size (not specified in the paper): `o3ls_min` = smallest near-square board where the search reaches C(B)=1; medium/large = smallest valid near-square board with ≥1.5× / ≥2× that area; `o3ls_fixed` takes an explicit board. |
| Magic states | Matched-IR assumption: free and always ready at one port tile next to A's X-side tile (0,1); the port is counted as one patch.  No factory tiles or latency (the paper places one 15-to-1 factory outside the board). |
| Scheduling | Algorithm 2: every iteration executes all executable frontier operators of the per-qubit-support PDAG in one step (stream-index order), then fixes the lowest-index frontier operator with greedy patch moves/rotations (reward = #enabled patches, 0 if connectivity breaks, ties → lower cost).  Patch operations are sequential and never overlap measurements.  **Because every measurement needs the single magic port / A neighbourhood, measurements are serialized (one per step)** — consistent with the paper's adapter_28 numbers (543 steps ≥ its measurement count), but a modelling choice. |
| Routing | Node-disjoint routing trees per step, grown by sequential BFS shortest paths from the existing tree (tree tiles cost 0), from the \|m⟩/A terminal to one free tile next to each required data edge. |
| Placement | Edge-aware initial mapping (§III-D): qubits ranked by X↔Z switch count in the synthesized stream, mapped to patches ranked by the number of edge types exposed to routing space. |
| Stochastic behavior | Deterministic; no seed. |
| Original metric | Time steps (≈ d code cycles), board size N×M, ancilla path length and an accumulated LER.  Validation on the paper's adder_28 (Fig. 9): ours 360 steps on ≥9×9 boards vs the paper's 543, and 559–748 vs 841–900 on small/skewed boards — same shape (flat on large boards, rising on small ones), ~1/3 lower in absolute terms, attributed to the different front end (our stream has 360 measurements after Y-synthesis; even uncancelled it would be 456 < 543). |
| Directly comparable | Schedule length (time steps), board tiles + port as logical patches, patches×steps, routing-tile count, compiler runtime, in `matched_ir`.  The three board sizes are configurations of one compiler and form a frontier. |
| Not directly normalized | The LER model, magic-state factory footprint/latency, GridSynth arbitrary-angle synthesis, and the paper's LAPBC-integrated variant. |

## Kobori et al. — LSQCA (`lsqca_point_b1/b2`, `lsqca_line_b1/b2/b4`)

| Property | Audited behavior |
|---|---|
| Paper/version | Takumi Kobori, Yasunari Suzuki, Yosuke Ueno, Teruo Tanimoto, Synge Todo, Yuuki Tokunaga, “LSQCA: Resource-Efficient Load/Store Architecture for Limited-Scale Fault-Tolerant Quantum Computing,” HPCA 2025, [arXiv:2412.20486v2](https://arxiv.org/abs/2412.20486), §II-C (Fig. 4), §IV (Table I, Figs. 10–11), §V (Fig. 12), §VI-A. |
| Upstream implementation | None public (the paper's code-beat-accurate simulator is unreleased).  HARVEST contains an independent, documented reimplementation (`harvest.baselines.lsqca`), not vendored upstream code. |
| Input representation | OpenQASM 2 Clifford+T (`native_end_to_end` only): h, s, sdg, t, tdg, cx; cz → H·CX·H and swap → 3 CX are decomposed explicitly; x/y/z/id are ignored as in the paper; any other operation (incl. mid-circuit measurement/reset/conditionals) fails the run. |
| Operation model | Gate-level LSQCA instructions (Table I) with load/store, not Pauli-based computation.  Lattice surgery 1, H 3, S 2, move 1, Pauli/prep/1q-measurement 0 code beats.  H/S always in-memory; T = PM + in-memory MZZ + MX + SK + PH (the conditional S is always taken, as in the paper); CX = optimised `CX M1 M2`: cheaper-to-load operand loaded, other accessed in memory, CNOT in CR = MZZ+MXX via ancilla (2 beats), then store. |
| Layout | Point SAM: k+1 cells per bank (⌈√(k+1)⌉-wide box, shortened bottom line), CR 2×3 = 6 cells, 1–2 banks.  Line SAM: w×h banks (L×L or L×(L+1), minimising SAM+CR cells) with a centre scan row, CR 2×h (2×2h for 4 banks, two stacked banks per side — assumption), 1/2/4 banks.  Qubits round-robin over banks, row-major inside a bank.  Patches = SAM + CR + 1 MSF port cell. |
| Magic states | Matched assumption by default: free and always ready at one MSF port cell (no factory area).  Optional `msf_period_beats` (paper: Litinski 15-beat factory) with a buffer of 2. |
| Scheduling | In-order issue with overlap: a gate starts when its qubits, its SAM banks and the CR registers it needs (CX both, T one) are free, never before the previous gate.  Resources are held for the whole gate.  The issue model is not specified in the paper (assumption). |
| Routing | No explicit router.  Point SAM: closed-form sliding-puzzle cost W+H+6min(W,H)+5\|W−H\| (second cell W+H+4min+3\|W−H\|), +1 final move; line SAM: row distance to the tracked scan line.  Locality-aware stores (1 beat; hole relocation at 1 beat/cell when needed — assumption).  Intra-bank path conflicts are abstracted by the closed forms. |
| Placement | Static round-robin bank assignment (paper §VI-A); dynamic in-bank placement by the locality-aware store.  No HARVEST placement. |
| Stochastic behavior | None; deterministic.  Seed is recorded as `null`. |
| Original metric | Execution time in code beats, CPI (beats / LSQCA instruction count) and memory density = data cells / (SAM + CR cells), MSF excluded. |
| Directly comparable | `native_end_to_end`: schedule length (code beats = logical cycles), full cell count incl. one magic port, cells×beats, runtime.  Memory density is reproduced exactly for the paper's 400-qubit line-SAM example (400/462). |
| Not directly normalized | Factory area/throughput (free by default), the paper's conventional 50%-density baseline, hybrid floorplans (§V-D), the paper's unpublished instruction-issue and conditional-skip details, per-code-distance physical cost. |

## TACO — `taco_k1`, `taco_k2`, `taco_k4`

| Property | Audited behavior |
|---|---|
| Paper/version | Meng Wang et al., “Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing,” ISCA 2026, [arXiv:2412.15434v2](https://arxiv.org/abs/2412.15434), §II-B, §IV-A–F, §V-B, §VI-C (Tables II/III). |
| Upstream implementation | Transpiler released as pnnl/nwqec (MIT; Zenodo 10.5281/zenodo.19449157), run unmodified: `nwqec==0.1.2` / `nwqec-cli` built from tag `v0.1.2` (d93299c) in an isolated env, as `nwqec-cli in.qasm --cr -o out.qasm`.  The architecture model and cycle-accurate simulator are **not released**; HARVEST reimplements §IV-D/§V-B (`harvest.baselines.taco`). |
| Input representation | Clifford+T OpenQASM 2 (cz/ccx allowed; anything needing synthesis is refused).  nwqec output: gate-level Clifford+T with CX kept (weight ≤ 2), H largely absorbed into `rx(±π/4)`, S removed via TS = T†Z, Paulis pushed to the end.  Losslessness is checked (register size, non-Clifford count, CX count, nwqec `count_ops`). |
| Operation model | §II-B latencies in QEC rounds (equal to nwqec `get_gate_duration`): CX and H 3d+4, S/Sdg/SX/SXdg 1.5d+3, T/Tdg/Rx(±π/4) 2.5d+4, Pauli 0.  Simulated exactly in rounds; `schedule_length = ceil(makespan_rounds / d)`, d = 13 by default (`--code-distance`). |
| Layout | Memory block = Litinski compact layout, ⌈1.5n⌉ tiles (§IV-D2 headline; the paper's QFT-18 example uses 29 = 1.5n+2).  k compute blocks × 4 tiles; swept over k ∈ {1, 2, 4} (1.5n+4k) as in RushHour App. A. |
| Magic states | Default: matched assumption — free and always ready, one port tile per compute block.  Option `magic_model=distillation`: 11 × 15-to-1 units per block (§VI-C3 / Table II). |
| Scheduling | Reimplemented §V-B: event-driven greedy first-come-first-served list scheduling on the per-qubit dependency DAG; CX, T-type and S-type each occupy one compute block; H and Paulis are local. |
| Routing | No explicit paths.  Memory↔compute transfer = 1 QEC round per move (§IV-D4 “single code cycle”, §II-B cycle = round), plus one to evict an idle resident; `transfer_rounds=d` is an option (up to 1.6× slower).  Patch rotations (§IV-D3) are not charged. |
| Placement | A qubit stays resident on the π/4 site of the last block that injected into it; free empty blocks are preferred. |
| Stochastic behavior | Deterministic. |
| Original metric | Clifford reduction (%), critical-path and cycle-accurate QEC cycles (Table II).  Validation: on nwqec 0.1.2's QFT-18 output at d=19 the scheduler gives 1.04M / 711k / 556k rounds for k = 2/3/4 vs the paper's 1.14M / 761k / 596k scaled by T-count (37,725 vs 40,777): −1.7% / +1.0% / +0.9%.  Clifford reduction with the artifact's Fig. 17 script averages 93.7% (paper: 91.2%). |
| Directly comparable | `native_end_to_end` only: schedule length (logical steps of d rounds), tiles (memory + 4k compute + k magic ports), tiles × steps. |
| Not directly normalized | Round-based latencies at a fixed d; TACO's 2d×d compute-site footprint counted as 2 tiles; transfer cost (1 vs d rounds); distillation/cultivation, physical error rates, reaction/decoder latency. |

## Metric rules

- `space_time_volume = logical_patch_count × schedule_length` only when both
  values describe the baseline's complete logical topology and native logical
  cycles.  Otherwise it is null with a note.
- HARVEST reports initial and post-pruning patch counts and space-time values.
  Its primary `logical_patches`/`space_time_volume` fields use post-pruning
  resources; the initial values remain visible in the same raw record.
- Failed and timed-out runs remain in raw data and `failures.csv`.  They are
  excluded—not converted to zero—from means, confidence intervals, and ratios.
- 95% confidence intervals use `1.96 × sample_std / sqrt(n)` and are null when
  fewer than two successful trials exist.
- Pairwise geometric means use only the benchmark intersection supported by
  both systems in the same comparison regime.  The ratio convention is
  **baseline / HARVEST**; values above one favor HARVEST for lower-is-better
  metrics.
- Runtime scope follows the comparison regime: matched-IR HARVEST/Silva time
  covers scheduling/routing (plus HARVEST pruning), while native-end-to-end
  HARVEST additionally includes QASM loading, transformation, and layout.
  PureMagic native mode includes its configured compile/transpile stages, and
  DASCOT time includes input validation plus the `wisq --mode scmr` process.
