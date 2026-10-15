#!/usr/bin/env python3
"""Search circuit subsets ("views") of a stored space/time trade-off figure.

Reads the per-circuit CSV that ``plot_sota_space_time_tradeoff.py`` wrote for
one figure of a suite (nothing is recomputed), groups the circuits by
structural features of their matched Pauli-product DAG and scores every
candidate subset by how clearly HARVEST separates from the other compilers:

* ``margin_rep``  geomean (over the subset) of each compiler's best
  space-time per circuit -- the large labelled marker of the figure --
  best other compiler / HARVEST (> 1: HARVEST lower);
* ``margin_cfg``  same, but each compiler's best *fixed configuration*
  (one point of its frontier), as drawn by the connected sweep frontier;
* ``dominated``   fraction of the other compilers' configurations whose
  geomean (space, time) is weakly dominated by some HARVEST configuration;
* ``win_rate``    fraction of circuits on which HARVEST's best space-time is
  the lowest of all compilers.

Subsets are defined only by a feature threshold or a family (never picked
circuit by circuit), and the report always lists the full set and the circuits
each view leaves out, so a figure drawn from a view can be described honestly
("circuits with parallelism >= x").  Features, from the cached lower bounds:

* ``parallelism`` = Pauli products / critical path of the matched DAG,
* ``num_qubits``, ``num_operations``, ``critical_path`` (``time_lb``),
* ``family`` (circuit name without its size suffix).

``--render N`` redraws the N best views with the figure's own experiments and
arguments plus ``--circuits``.

Example::

    PYTHONPATH=src python src/scripts/search_sota_views.py experiments/sota_suite.json \\
        --figure sweep --render 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HARVEST = "HARVEST"
FLASQ = "FLASQ model"
FEATURES = ("parallelism", "num_qubits", "num_operations", "critical_path")
QUANTILES = (0.25, 1 / 3, 0.5, 2 / 3, 0.75)


def geomean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def family(circuit: str) -> str:
    """Circuit name without its size suffix: gf2^8_mult -> gf2_mult, ham15-low -> ham."""
    name = re.sub(r"\^\d+", "", circuit)
    name = re.sub(r"-(low|med|high)$", "", name)
    name = re.sub(r"_n?\d+$", "", name)
    name = re.sub(r"\d+$", "", name)
    return name.rstrip("_-") or circuit


def load_rows(csv_path: Path) -> List[Dict[str, object]]:
    rows = []
    for r in csv.DictReader(csv_path.open()):
        if r["compiler"] == FLASQ or r.get("in_shared_subset") != "True":
            continue
        rows.append({
            "circuit": r["circuit"],
            "compiler": r["compiler"],
            "config": (r["source_dir"], r["config_id"]),
            "space": float(r["normalized_space"]),
            "time": float(r["normalized_time"]),
            "st": float(r["normalized_space_time"]),
        })
    if not rows:
        raise SystemExit(f"{csv_path}: no shared, non-FLASQ rows")
    return rows


def load_features(result_dirs: Sequence[Path], circuits: Sequence[str]) -> Dict[str, Dict]:
    features: Dict[str, Dict] = {}
    for result_dir in result_dirs:
        cache = result_dir / "bounds_cache.json"
        if not cache.exists():
            continue
        for key, bound in json.loads(cache.read_text()).items():
            if "|ppr|" not in key or bound["circuit"] in features:
                continue
            features[bound["circuit"]] = {
                "num_qubits": bound["num_qubits"],
                "num_operations": bound["num_operations"],
                "critical_path": bound["time_lb"],
                "parallelism": bound["num_operations"] / bound["time_lb"],
                "family": family(bound["circuit"]),
            }
    missing = sorted(set(circuits) - set(features))
    if missing:
        raise SystemExit(f"no cached PPR bounds for: {', '.join(missing)}")
    return features


def candidate_views(features: Dict[str, Dict], circuits: Sequence[str],
                    min_circuits: int) -> List[Tuple[str, List[str]]]:
    views: List[Tuple[str, List[str]]] = [("all circuits", sorted(circuits))]
    for feature in FEATURES:
        values = sorted(features[c][feature] for c in circuits)
        for q in QUANTILES:
            cut = values[min(len(values) - 1, int(q * len(values)))]
            fmt = f"{cut:.2f}" if feature == "parallelism" else f"{cut:g}"
            above = sorted(c for c in circuits if features[c][feature] >= cut)
            below = sorted(c for c in circuits if features[c][feature] < cut)
            views.append((f"{feature} >= {fmt}", above))
            views.append((f"{feature} < {fmt}", below))
    by_family: Dict[str, List[str]] = defaultdict(list)
    for c in circuits:
        by_family[features[c]["family"]].append(c)
    for name, members in sorted(by_family.items()):
        views.append((f"family {name}", sorted(members)))
        rest = sorted(set(circuits) - set(members))
        views.append((f"without family {name}", rest))
    unique: Dict[Tuple[str, ...], str] = {}
    for label, members in views:
        if len(members) >= min_circuits:
            unique.setdefault(tuple(members), label)
    return [(label, list(members)) for members, label in unique.items()]


def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1]


def score(rows: Sequence[Dict], members: Sequence[str]) -> Optional[Dict[str, object]]:
    subset = set(members)
    best: Dict[Tuple[str, str], float] = {}
    configs: Dict[Tuple[str, Tuple[str, str]], Dict[str, Dict]] = defaultdict(dict)
    for r in rows:
        if r["circuit"] not in subset:
            continue
        key = (r["compiler"], r["circuit"])
        best[key] = min(best.get(key, math.inf), r["st"])
        configs[(r["compiler"], r["config"])][r["circuit"]] = r
    compilers = sorted({c for c, _ in best})
    if HARVEST not in compilers or len(compilers) < 2:
        return None
    rep = {}
    for compiler in compilers:
        values = [best[(compiler, c)] for c in members if (compiler, c) in best]
        if len(values) == len(members):
            rep[compiler] = geomean(values)
    frontier: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for (compiler, _), by_circuit in configs.items():
        if len(by_circuit) != len(members):
            continue
        chosen = [by_circuit[c] for c in members]
        frontier[compiler].append((geomean([p["space"] for p in chosen]),
                                   geomean([p["time"] for p in chosen]),
                                   geomean([p["st"] for p in chosen])))
    if HARVEST not in rep or HARVEST not in frontier:
        return None
    others = [c for c in rep if c != HARVEST]
    rival = min(others, key=rep.get)
    cfg_best = {c: min(p[2] for p in pts) for c, pts in frontier.items()}
    cfg_rival = min((c for c in cfg_best if c != HARVEST), key=cfg_best.get)
    other_cfgs = [p for c, pts in frontier.items() if c != HARVEST for p in pts]
    dominated = sum(
        any(dominates(h[:2], o[:2]) for h in frontier[HARVEST]) for o in other_cfgs
    ) / max(1, len(other_cfgs))
    wins = sum(
        best[(HARVEST, c)] < min(best[(o, c)] for o in others if (o, c) in best)
        for c in members
    )
    return {
        "n": len(members),
        "margin_rep": rep[rival] / rep[HARVEST],
        "rival_rep": rival,
        "margin_cfg": cfg_best[cfg_rival] / cfg_best[HARVEST],
        "rival_cfg": cfg_rival,
        "dominated": dominated,
        "win_rate": wins / len(members),
        "harvest_st": rep[HARVEST],
    }


def clarity(s: Dict[str, object]) -> float:
    """One ranking number: the weaker of the two margins, nudged by dominance."""
    return min(s["margin_rep"], s["margin_cfg"]) * (1 + 0.25 * s["dominated"])  # type: ignore[operator]


def figure_spec(suite: Dict, name: str) -> Tuple[Dict, List[Dict]]:
    figures = {f["name"]: f for f in suite["figures"]}
    if name not in figures:
        raise SystemExit(f"unknown figure {name!r}; have {sorted(figures)}")
    figure = figures[name]
    if figure.get("kind"):
        raise SystemExit(f"figure {name!r} is a {figure['kind']} figure, not a trade-off figure")
    experiments = {e["name"]: e for e in suite["experiments"]}
    return figure, [experiments[e] for e in figure["experiments"]]


def render(view_id: str, label: str, members: Sequence[str], suite: Dict, figure: Dict,
           experiments: Sequence[Dict], out_dir: Path, extra: Sequence[str]) -> None:
    list_path = out_dir / f"{view_id}.circuits"
    list_path.write_text("\n".join(members) + "\n")
    command = [sys.executable, str(SCRIPT_DIR / "plot_sota_space_time_tradeoff.py")]
    command += [e["output_dir"] for e in experiments]
    command += ["--output-dir", str(out_dir), "--output-stem", view_id,
                "--circuits", f"@{list_path}"]
    flasq_python = suite.get("tools", {}).get("flasq_python")
    if flasq_python and Path(flasq_python).exists():
        command += ["--flasq-python", flasq_python]
    command += list(figure.get("args", [])) + list(extra)
    print(f"== render {view_id}: {label} ({len(members)} circuits)", flush=True)
    log = out_dir / f"{view_id}.log"
    with log.open("w") as handle:
        proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                              env={**__import__("os").environ,
                                   "PYTHONPATH": str(PROJECT_ROOT / "src")})
    if proc.returncode != 0:
        print(f"   failed, see {log}")


def write_report(path: Path, figure_name: str, scored: List[Tuple[str, List[str], Dict]],
                 all_circuits: Sequence[str], features: Dict[str, Dict], top: int) -> None:
    full = next(s for label, _, s in scored if label == "all circuits")
    lines = [
        f"# View search: figure `{figure_name}`",
        "",
        "Margins are best other compiler / HARVEST (geomean normalized space-time; "
        "> 1 means HARVEST is lower). `rep`: per-circuit best of each compiler; "
        "`cfg`: best fixed configuration. `dominated`: share of the other compilers' "
        "configurations dominated by a HARVEST configuration.",
        "",
        "A view is only a valid figure if its selection rule is stated in the caption; "
        "the full-set result below must be reported alongside it.",
        "",
        "| rank | view | n | margin rep (rival) | margin cfg (rival) | dominated | wins |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, (label, members, s) in enumerate(scored, 1):
        lines.append(
            f"| {rank} | {label} | {s['n']} | {s['margin_rep']:.3f} ({s['rival_rep']}) | "
            f"{s['margin_cfg']:.3f} ({s['rival_cfg']}) | {s['dominated']:.0%} | "
            f"{s['win_rate']:.0%} |"
        )
    lines += ["", f"Full set ({full['n']} circuits): margin rep {full['margin_rep']:.3f}, "
              f"margin cfg {full['margin_cfg']:.3f}, wins {full['win_rate']:.0%}.", ""]
    for rank, (label, members, s) in enumerate(scored[:top], 1):
        left_out = sorted(set(all_circuits) - set(members))
        lines += [f"## {rank}. {label}", "",
                  f"Included ({len(members)}): {', '.join(members)}", "",
                  f"Left out ({len(left_out)}): {', '.join(left_out) or '-'}", ""]
    lines += ["## Circuit features", "", "| circuit | family | qubits | products | "
              "critical path | parallelism |", "|---|---|---|---|---|---|"]
    for c in sorted(all_circuits, key=lambda c: features[c]["parallelism"]):
        f = features[c]
        lines.append(f"| {c} | {f['family']} | {f['num_qubits']} | {f['num_operations']} | "
                     f"{f['critical_path']:g} | {f['parallelism']:.2f} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("suite", type=Path)
    parser.add_argument("--figure", default="sweep", help="Trade-off figure of the suite.")
    parser.add_argument("--min-circuits", type=int, default=8,
                        help="Smallest subset considered (avoids tiny, anecdotal views).")
    parser.add_argument("--render", type=int, default=0, metavar="N",
                        help="Redraw the N best views (plus the full set).")
    parser.add_argument("--plot-args", default="",
                        help="Extra arguments for the plot script, e.g. '--per-benchmark none'.")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: results/sota_view_search/<suite stem>/<figure>.")
    args = parser.parse_args(argv)

    suite = json.loads(args.suite.read_text())
    figure, experiments = figure_spec(suite, args.figure)
    csv_path = PROJECT_ROOT / figure["output_dir"] / f"{figure['stem']}.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} missing; draw the figure first (run_sota_suite --figures-only)")
    rows = load_rows(csv_path)
    circuits = sorted({r["circuit"] for r in rows})
    features = load_features([PROJECT_ROOT / e["output_dir"] for e in experiments], circuits)

    scored = []
    for label, members in candidate_views(features, circuits, args.min_circuits):
        s = score(rows, members)
        if s is not None:
            scored.append((label, members, s))
    scored.sort(key=lambda item: (item[0] != "all circuits", -clarity(item[2])))
    full, ranked = scored[0], sorted(scored[1:], key=lambda item: -clarity(item[2]))
    scored = [full] + ranked

    out_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "sota_view_search" / args.suite.stem / args.figure
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "views.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["view", "n", "clarity", "margin_rep", "rival_rep", "margin_cfg",
                         "rival_cfg", "dominated", "win_rate", "circuits"])
        for label, members, s in scored:
            writer.writerow([label, s["n"], f"{clarity(s):.4f}", f"{s['margin_rep']:.4f}",
                             s["rival_rep"], f"{s['margin_cfg']:.4f}", s["rival_cfg"],
                             f"{s['dominated']:.3f}", f"{s['win_rate']:.3f}", " ".join(members)])
    write_report(out_dir / "views.md", args.figure, scored, circuits, features,
                 max(args.render, 5))
    print(f"{'view':<34} {'n':>3} {'rep':>6} {'cfg':>6} {'dom':>5} {'wins':>5}")
    for label, members, s in scored[:15]:
        print(f"{label:<34} {s['n']:>3} {s['margin_rep']:6.3f} {s['margin_cfg']:6.3f} "
              f"{s['dominated']:5.0%} {s['win_rate']:5.0%}")
    print(f"wrote {out_dir / 'views.md'} and {out_dir / 'views.csv'}")

    if args.render:
        extra = args.plot_args.split()
        for index, (label, members, _) in enumerate(scored[: args.render + 1]):
            view_id = "view_00_all" if index == 0 else f"view_{index:02d}"
            render(view_id, label, members, suite, figure, experiments, out_dir, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
