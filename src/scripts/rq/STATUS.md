# RQ experiments: status (2026-10-29, 15:00)

Handover note for whoever picks this up. `README.md` in this folder describes
the experiments themselves; this file says what is running and where.

## Three checkouts of the repo

| Folder (`~/Documents/PhD/Harvest_PLDI/code/`) | Branch | Content |
|---|---|---|
| `harvest_magic_state` | `NewBaselines` | Old Harvest code. Runs the remaining **baseline** jobs (EAF, Litinski, O3LS). |
| `harvest_magic_state_perf` | `harvest-perf` | Harvest improvements in Python (below). RQ1 run there was **stopped**; its results are kept. |
| `harvest_magic_state_rust` | `harvest-rust` | Same as `harvest-perf` plus the router in Rust. **All current Harvest runs.** |

`harvest-perf` and `harvest-rust` are uncommitted on purpose: the user commits
them. `results/rq/cache/placement`, the circuit inputs and `benchmark_circuits`
are symlinks into `harvest_magic_state`.

## Code changes (on both new branches)

- `HarvestConfig.terminals = "route"` (new default): the router picks each
  product's magic port among the free ready ones, starting from the min-cost
  matching, and may move it while negotiating (`TerminalChoice` in
  `layout/engine.py`).
- A* inside the Steiner search; distances are exact, only ties differ.
- Template `auto` = best volume of ring / blocks4 / dense4 / `dense4full`
  (dense comb with a magic port on every routing column, top and bottom).
  Compile time is the sum over the four. For n >= 400, `auto` gets 3
  cultivation seeds (as EAF / O3LS do), not 10.
- PureMagic's work directory now includes the magic model (instant,
  cultivation and cultivation_pm shared it at seed 42 and could read each
  other's `.schedule`).
- `harvest-rust` only: `rust/harvest_router` (PyO3; build with
  `rust/build.sh` -> `src/harvest/_router.so`), `HarvestConfig.backend =
  "rust"` (default) or `"python"`. About 9x faster scheduling, same quality
  (volume 1.001x vs Python on the 15-circuit benchmark).

## Running now

| systemd user unit | Folder | Command (`src/scripts/rq/run_all.py ...`) |
|---|---|---|
| `harvest-rq-rust` | `harvest_magic_state_rust` | `--experiments rq1,rq2,rq3,rq4,rq5 --rq1-templates auto --skip-compilers eaf,litinski,o3ls,puremagic --skip-families random --jobs 2` |
| `harvest-rq` | `harvest_magic_state` | `--skip-compilers harvest,puremagic --jobs 2` |

Both run under `systemd-inhibit` (no suspend) with `-p OOMPolicy=continue`.
Always pass that flag when restarting, or one worker killed for running out
of memory stops the whole unit. `harvest-rq-rust` has MemoryMax 9G and 2 jobs (17:25; 4 jobs under 7G ran out of memory on random 80–90q): placing
a circuit of 60+ qubits needs ~2.7 GB per worker.

Progress: `tail results/rq/run_all/run.log` in the unit's folder. At 14:45,
12,198 of 14,294 Rust runs were done. RQ2–RQ5 were waiting only on their
largest circuits; RQ1 still has its circuits of 64 qubits and more,
including Ising and wstate n118–n440.

A run that crashes twice is recorded as `status=crashed` and is **not**
retried automatically. Delete its row (under `results/rq/<exp>/<exp>.lock`)
and restart the unit.

## Results and plots

- Rust branch: `results/rq/rq1` ... `rq5` (Harvest only).
- PureMagic (all 3,200 RQ1 runs, rerun with the work-directory fix): the
  perf folder's `results/rq/rq1/rq1.csv`. Baseline rows (EAF / Litinski /
  O3LS) are in the main folder.
- RQ1 snapshot with every compiler: `python src/scripts/rq/rq1_view.py $PWD`
  -> `results/rq/rq1_view/`. RQ2–RQ5:
  `python src/scripts/rq/run_all.py --experiments rq2,rq3,rq4,rq5 --plot-only`.
- Layout pictures, PureMagic vs Harvest:
  `python src/scripts/rq/plot_layouts.py <circuit ...> --model instant|cultivation`.

## Findings so far (interim)

- RQ1, circuits up to 98 qubits: Harvest volume is 0.71x PureMagic's (instant)
  and 0.83x (cultivation), almost entirely from area: `auto` picks dense
  layouts, and pruning removes another 7–10%. Time steps are within 5%
  of PureMagic. Harvest still loses on highly parallel circuits under
  cultivation (ising, knn, swap_test): they need about 3 x parallelism magic
  ports, more than dense4full offers, and blocks4's ring costs area.
- RQ2: removing negotiation now costs about 0% volume, against +4–6% on the
  old code (the router-chosen terminals remove most conflicts). Check again
  once ising_n98 etc. are done. Labels `-matching` (now: no router choice)
  and `-retry` (now: no greedy fill) no longer match their names.
- RQ5: the validator catches 100% of injected faults and corruptions.
  `compile_time_s` includes the cached placement time, so the scaling plot
  shows a few large outliers.

## User preferences

- No hash or fingerprint columns in experiment outputs.
- After Harvest changes, rerun only Harvest (and PureMagic if needed), not
  EAF / Litinski / O3LS.
- Ideas still open and **not to be implemented without asking**:
  pruning-aware routing, wider routing lanes, routing tiles that cultivate,
  magic-port count from `mean x parallelism`. A commutation-aware DAG was
  rejected.

## 2026-10-29, 17:25

- Random circuits dropped (user: only real circuit families). The Rust unit
  runs with `--skip-families random` (new flag in `run_all.py`); their
  remaining 33 RQ1 runs are not done. `rq1_view.py` leaves the `random`
  family out for every compiler. Rows already in the CSVs are kept.

## 2026-10-29, 19:56

- The laptop rebooted at about 18:30–19:54; `systemd-run` units do not
  survive a reboot. Both units were restarted with the commands above
  (`harvest-rq` now MemoryMax 4G). After any reboot, check
  `systemctl --user is-active harvest-rq-rust harvest-rq` and rerun them;
  finished rows are skipped.
