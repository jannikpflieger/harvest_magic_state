#!/usr/bin/env python3
"""Side-by-side layouts of PureMagic and Harvest for one circuit and magic model.

PureMagic: the topology it generates for the circuit (``--plot topo`` writes it
as ``<circuit>.topo.txt``; every non-data tile is a magic patch that also routes),
run with the RQ1 settings (magic routing, seed 42).  Harvest: the template
``auto`` picks for the circuit, after pruning; routing tiles are shaded by the
number of steps that route through them, pruned tiles are drawn as outlines.

    python src/scripts/rq/plot_layouts.py multiplier_n15 ising_n34 --model instant

Figures: ``results/rq/rq1_view/layout_<circuit>_<model>.png/.pdf``; one row per
figure (tile counts by role, time steps, volume) in ``layouts_summary.csv`` there.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import rq_plotstyle as style
from rq_common import CACHE, MODELS, RESULTS, TOOLS, load_circuit, load_suite, run_in_tree

OUT = RESULTS / "rq1_view"
DATA, MAGIC = style.SERIES[0], style.SERIES[3]
ROUTE_LIGHT, ROUTE_DARK = (0.93, 0.92, 0.90), (0.45, 0.44, 0.42)


def puremagic_layout(circ, model: str):
    """(grid rows of 'm' / 'dX' / 'dZ' / 'b' tokens, T, volume) of PureMagic's run."""
    from harvest.baselines.puremagic import write_trans_file

    _, dag = load_circuit(circ)
    trans = CACHE / "trans" / f"{circ['circuit']}.trans"
    if not trans.exists():
        write_trans_file(dag, trans, circ["num_qubits"])
    spec = MODELS[model]
    lam = 1.0 / spec["cultivation_mean_cycles"] if "cultivation_mean_cycles" in spec else 1e6
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / trans.name
        local.write_text(trans.read_text())
        cmd = [TOOLS["puremagic_bin"], "--circuit", local.name, "--magic-state-lambda", str(lam),
               "--rseed", "42", "--ancilla-rows", "1", "--use-magic-routing", "--plot", "topo"]
        if model == "instant":
            cmd.append("--no-t-failures")
        out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, check=True).stdout
        grid = [line.split() for line in (Path(tmp) / f"{local.stem}.topo.txt").read_text().splitlines()
                if line.strip()]
    m = re.search(r"Scheduled\s+\d+\s+in\s+(\d+)\s+logical cycles,\s*volume\s+(\d+)", out)
    return grid, int(m.group(1)), int(m.group(2))


def harvest_layout(circ, model: str):
    """Cells by role of the ``auto`` layout after pruning, plus the run's row."""
    sink = {}
    seed = 42 if model != "instant" else None
    row = run_in_tree(circ, "harvest", "auto", model, seed, sink=sink)
    proc, results = sink["processor"], sink["results"]
    eng = proc.eng
    use = Counter(n for r in results for n in r.get("steiner_nodes", ()) if isinstance(n, tuple))
    cells = {}  # (x, y) -> (role, uses)
    for patch in eng.patches.values():
        is_magic = any(p.port_type == "M" for p in patch.ports)
        if is_magic:
            role = "magic" if patch.name in proc.ports_by_patch else "pruned"
        else:
            index = int(patch.name.split("_")[1]) if patch.name.startswith("q_") else -1
            role = "data" if 0 <= index < circ["num_qubits"] else "pruned"
        for c in patch.cells:
            cells[c] = (role, 0)
    for x in range(eng.W):
        for y in range(eng.H):
            if (x, y) in cells or eng.occ.get((x, y)) == "BLOCKED":
                continue
            cells[(x, y)] = ("routing" if (x, y) in proc.graph else "pruned", use.get((x, y), 0))
    return cells, (eng.W, eng.H), row


def draw(ax, cells, size, title, note):
    from matplotlib.patches import Rectangle

    peak = max([u for role, u in cells.values() if role == "routing"] or [1])
    for (x, y), (role, uses) in cells.items():
        if role == "pruned":
            ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, fill=False, linestyle=(0, (2, 2)),
                                   edgecolor=style.AXIS, linewidth=0.8))
            continue
        if role == "routing":
            t = (uses / peak) ** 0.5
            color = tuple(a + (b - a) * t for a, b in zip(ROUTE_LIGHT, ROUTE_DARK))
        else:
            color = DATA if role == "data" else MAGIC
        ax.add_patch(Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, facecolor=color, edgecolor="none"))
    ax.set_xlim(-0.2, size[0] + 0.2)
    ax.set_ylim(size[1] + 0.2, -0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color=style.INK2)


def plot(name: str, model: str) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    circ = next(c for c in load_suite(["core"]) if c["circuit"] == name)
    grid, pm_t, pm_vol = puremagic_layout(circ, model)
    pm_cells = {}
    for y, row in enumerate(grid):
        for x, tok in enumerate(row):
            pm_cells[(x, y)] = ("data" if tok.startswith("d") else "magic" if tok == "m" else "routing", 0)
    pm_count = Counter(role for role, _ in pm_cells.values())
    pm_size = (max(len(r) for r in grid), len(grid))

    h_cells, h_size, row = harvest_layout(circ, model)
    h_count = Counter(role for role, _ in h_cells.values())
    h_area = h_count["data"] + h_count["magic"] + h_count["routing"]

    width = max(pm_size[0], h_size[0])
    height = max(pm_size[1], h_size[1])
    fig, axes = plt.subplots(1, 2, figsize=(2 * (width * 0.42 + 0.6), height * 0.42 + 1.9))
    draw(axes[0], pm_cells, (width, height), "PureMagic (generated layout)",
         f"{pm_count['data']} data + {pm_count['magic']} magic"
         + (f" + {pm_count['routing']} bus" if pm_count["routing"] else "")
         + f" = {len(pm_cells)} tiles\n{pm_t} time steps, volume {pm_vol:,}")
    draw(axes[1], h_cells, (width, height),
         f"Harvest ({row['template_chosen']}, chosen by auto; after pruning)",
         f"{h_count['data']} data + {h_count['routing']} routing + {h_count['magic']} magic"
         f" = {h_area} tiles ({h_count['pruned']} pruned)\n"
         f"{row['T_cycles']:.0f} time steps, volume {row['volume_tile_cycles']:,.0f}")
    handles = [Patch(facecolor=DATA, label="data patch"),
               Patch(facecolor=MAGIC, label="magic patch (PureMagic: also routes)"),
               Patch(facecolor=ROUTE_DARK, label="routing tile (shade = steps used)"),
               Patch(facecolor="none", edgecolor=style.AXIS, linestyle="--", label="pruned / unused")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               title=f"{name} ({circ['num_qubits']} qubits, {int(circ['ops'])} products), "
                     f"{model} magic states", title_fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"layout_{name}_{model}"
    style.save(fig, path)
    summary = {
        "circuit": name, "model": model, "qubits": circ["num_qubits"], "ops": int(circ["ops"]),
        "parallelism": round(float(circ["parallelism"]), 3),
        "pm_data": pm_count["data"], "pm_magic": pm_count["magic"], "pm_tiles": len(pm_cells),
        "pm_T": pm_t, "pm_volume": pm_vol, "harvest_template": row["template_chosen"],
        "harvest_data": h_count["data"], "harvest_routing": h_count["routing"],
        "harvest_magic": h_count["magic"], "harvest_tiles": h_area, "harvest_pruned": h_count["pruned"],
        "harvest_T": int(row["T_cycles"]), "harvest_volume": int(row["volume_tile_cycles"]),
    }
    log = OUT / "layouts_summary.csv"
    rows = [r for r in csv.DictReader(log.open())] if log.exists() else []
    rows = [r for r in rows if (r["circuit"], r["model"]) != (name, model)] + [summary]
    with log.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, list(summary))
        writer.writeheader()
        writer.writerows(rows)
    return path.with_suffix(".png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("circuits", nargs="+")
    parser.add_argument("--model", default="instant", choices=["instant", "cultivation"])
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    style.apply()
    for name in args.circuits:
        print(plot(name, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
