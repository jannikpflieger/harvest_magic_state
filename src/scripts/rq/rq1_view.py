"""Snapshot RQ1 view while runs are in progress: this branch's Harvest rows + the
perf folder's other compilers (PureMagic, EAF, Litinski, O3LS),
Harvest restricted to circuits whose runs (per model) are all done; random circuits
left out (real circuit families only); plots into results/rq/rq1_view."""
import csv, sys
from collections import defaultdict
from pathlib import Path

branch = Path(sys.argv[1])
perf = Path("/home/jannik/Documents/PhD/Harvest_PLDI/code/harvest_magic_state_perf/results/rq/rq1/rq1.csv")
out = branch / "results" / "rq" / "rq1_view"
out.mkdir(parents=True, exist_ok=True)

def read(path):
    with open(path, newline="") as f:
        r = csv.reader(f); header = next(r)
        return header, [dict(zip(header, row)) for row in r]

h_head, h_rows = read(branch / "results" / "rq" / "rq1" / "rq1.csv")
o_head, o_rows = read(perf)
SKIP_FAMILIES = {"random"}
harvest = [r for r in h_rows if r["compiler"] == "Harvest" and r["family"] not in SKIP_FAMILIES]
others = [r for r in o_rows if r["compiler"] != "Harvest" and r["family"] not in SKIP_FAMILIES]
# expected Harvest runs per (circuit, model): instant 1, cultivation 10 (3 for n >= 400), cultivation_pm 1
seen = defaultdict(set)
for r in harvest:
    seen[(r["circuit"], r["model"])].add(r["run_key"])
def expected(circ_rows, model):
    q = int(float(circ_rows["num_qubits"]))
    return 10 if model == "cultivation" and q < 400 else (3 if model == "cultivation" else 1)
first = {}
for r in harvest:
    first.setdefault((r["circuit"], r["model"]), r)
complete = {k for k, keys in seen.items() if len(keys) >= expected(first[k], k[1])}
harvest = [r for r in harvest if (r["circuit"], r["model"]) in complete]
header = list(dict.fromkeys(o_head + h_head))
with open(out / "rq1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, header, extrasaction="ignore"); w.writeheader()
    for r in others + harvest:
        w.writerow(r)
print(f"{branch.name}: {len(complete)} complete Harvest (circuit, model) groups, {len(harvest)} Harvest rows, {len(others)} other rows")

sys.path.insert(0, str(branch / "src" / "scripts" / "rq"))
import rq1_plots
rq1_plots.OUT = out
rq1_plots.main()
