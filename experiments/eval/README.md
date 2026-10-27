# HARVEST paper evaluation (EVAL_PLAN)

One JSON file per experiment (`e1_sota.json` … `e8_scaling.json`), the frozen
benchmark suites (`suites.json`) and external tool locations (`tools.json`).
Results go to `results/eval/<id>/`; `results/eval/INDEX.md` lists every
experiment, its paper section and its figures.

## Setup (once)

```bash
# Freeze the suites (also generates and lowers the mcx_tree scaling series).
python src/scripts/eval/freeze_suites.py --generate-scale
# Regression gate: HarvestConfig() must reproduce the reference schedules.
python -m pytest tests/test_harvest_config_regression.py tests/test_eval_harness.py
```

E1 and E8 need PureMagic (`cargo build --release` of the QCE snapshot); put the
binary paths in `tools.json`. A missing binary is not fatal: those runs are
recorded as `status=error` rows. The FLASQ time floor is read from
`results/flasq_cache` (populate it with `plot_sota_space_time_tradeoff.py
--flasq-python ...`); without it, E1 normalizes time by the critical path.

## Run

```bash
python src/scripts/eval/run_eval.py e1_sota --jobs 6        # P0
python src/scripts/eval/run_eval.py e3_ablation --jobs 6    # P0
python src/scripts/eval/run_eval.py e8_scaling              # P0 (always serial)
python src/scripts/eval/run_eval.py e4_provisioning --jobs 6
python src/scripts/eval/run_eval.py e5_magic_models --jobs 6
python src/scripts/eval/run_eval.py e6_placement --jobs 6
python src/scripts/eval/run_eval.py e2_characterize         # reads e1 data
python src/scripts/eval/run_eval.py e7_validity --jobs 6    # run last: aggregates e1-e8
python src/scripts/eval/make_paper_numbers.py               # -> results/eval/paper_numbers.tex
```

For long runs, use `nohup … > results/eval/<id>/logs/driver.log 2>&1 &`. Each
sweep point also logs to `results/eval/<id>/logs/<point>.log`.

* **Resume:** every run reuses stored trials (`compare_sota_baselines.py
  --resume --keep-failed`), so rerunning a command computes only what is
  missing. `--force-rerun harvest` recomputes a baseline after code changes.
* **Redraw:** `--figures-only` redraws the PDFs from `data.csv` /
  `summary.csv`. `--no-run` re-collects and re-summarizes the stored raw
  records.
* **Smoke test:** `--quick N` runs on the N smallest circuits of each suite and
  writes to `results/eval_quick/`.
* **Routes:** HARVEST and Silva records list every product in
  `metadata.schedule` (DAG node, qubits with their Pauli factors, time step,
  magic terminal). The exact route of each product (data ports, routing nodes
  and edges) is in the gzipped sidecar `raw/<record>.routes.json.gz`; read it
  with `harvest.eval.routes.read_routes(<record>.json)`. Records written before
  this export have no sidecar and an empty `qubits` field.
* `PYTHONHASHSEED=0` is set for every run, because HARVEST's tie-breaking
  iterates string-keyed sets and its schedules differ under other hash seeds.

## Circuit families and the Ising deep dive

Every `data.csv` row carries `family` (ising, qaoa, qft, adder, gf2_mult, ...;
`harvest.eval.families`) and `synthesis` (`exact` / `approximate`). Extra suites:

* `families`: every prepared family below 400 qubits with at least 100 Pauli
  products. Inputs come from `results/sota_family_inputs` and
  `results/eval_inputs/families` (`src/scripts/eval/prepare_new_families.py`: QASMBench,
  QV, random, Feynman leftovers). Approximate families were synthesized once with
  PureMagic `compile_cliffordt -e 1e-3`; every compiler gets the same Clifford+T file.
  Skipped inputs and reasons: `results/eval_inputs/families/prefilter_skipped.json`
  (Clifford-only, or over the 40k T budget) and `<family>/skipped.json`.
* `ising` (n10-n98) and `ising_large` (n400, n420, n440).

Experiments with `"template"` reuse another experiment's summary and plots:
`e1_families` (head-to-head per family, adds `e1_family_volume.pdf`),
`e1_ising_large`, and `e3_ablation_ising`, `e4_provisioning_ising`,
`e5_magic_models_ising`, `e6_placement_ising` (E3-E6 on the Ising suite).

## Running in the background

`results/eval/run_all.sh` (main chain, then `run_puremagic_followup.sh`) and
`results/eval/run_ising_families.sh` run as memory-capped user units, so an
out-of-memory kill hits an evaluation run rather than the desktop:

```bash
systemctl --user list-units 'harvest-*'          # what is running
journalctl --user -u harvest-main -f              # follow a unit
systemctl --user stop harvest-ising               # stop one (rerun later: it resumes)
```
