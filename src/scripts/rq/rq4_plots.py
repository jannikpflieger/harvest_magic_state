#!/usr/bin/env python3
"""RQ4 tables and figures from ``results/rq/rq4/rq4.csv``.

A: makespan / critical path against the number of magic terminals (geomean
over circuits, seeds averaged first), aware (solid) vs oblivious-replayed
(dashed), with the supply bound magic_ops * latency / terminals (dotted).
B: plug-ability table - interface lines of code per model, restriction
stability (checked here) and validator results of the aware schedules.

Outputs (``results/rq/rq4/``): rq4_curves.csv, rq4_oblivious_vs_aware.csv,
rq4_pluggability.csv/.tex, fig5_magic_terminals.pdf
"""

from __future__ import annotations

import ast
import inspect
import random
import textwrap

import numpy as np
import pandas as pd

import rq_plotstyle as style
from rq_common import MODELS, PROJECT_ROOT, RESULTS, geomean, read_rows

OUT = RESULTS / "rq4"
PANELS = ["instant", "dist4", "dist10", "cultivation", "multiplexed", "lognormal"]


def curves(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.status == "ok"].copy()
    for col in ("T_cycles", "critical_path", "magic_terminals", "supply_bound", "level",
                "v4_violations_unmodified"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["mode"].isin(["aware", "oblivious"])]
    agg = df.groupby(["circuit", "mode", "model", "level"]).agg(
        T=("T_cycles", "mean"), cp=("critical_path", "first"), m=("magic_terminals", "first"),
        bound=("supply_bound", "mean"), v4=("v4_violations_unmodified", "mean"),
        n=("T_cycles", "size")).reset_index()
    # instant needs no replay: its oblivious schedule is the aware one
    inst = agg[(agg["mode"] == "aware") & (agg.model == "instant")].assign(mode="oblivious")
    agg = pd.concat([agg, inst], ignore_index=True)
    agg["T_over_cp"] = agg["T"] / agg.cp
    agg["bound_over_cp"] = agg.bound / agg.cp
    return agg


def loc(obj) -> int:
    """Statement lines of a function/class without docstrings, comments or blank lines."""
    source = textwrap.dedent(inspect.getsource(obj))
    tree = ast.parse(source)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(node):
            d = node.body[0]
            doc_lines |= set(range(d.lineno, d.end_lineno + 1))
    return sum(1 for i, line in enumerate(source.splitlines(), 1)
               if line.strip() and not line.strip().startswith("#") and i not in doc_lines)


def restriction_stable(model: str, trials: int = 20, steps: int = 400) -> bool:
    """Same consumption trace on a subset S of terminals: the source built on S
    and the source built on all terminals report identical readiness on S."""
    from harvest.baselines.cultivation import make_magic_source

    options = MODELS[model]
    if not options:
        return True  # instant: no source, every terminal always ready
    terminals = [f"m{i}" for i in range(12)]
    for trial in range(trials):
        rng = random.Random(trial)
        subset = sorted(rng.sample(terminals, 5))
        full = make_magic_source(options, terminals, 42 + trial)
        sub = make_magic_source(options, subset, 42 + trial)
        for _ in range(steps):
            full.tick()
            sub.tick()
            ready_full = sorted(set(full.get_ready_terminals()) & set(subset))
            if ready_full != sorted(sub.get_ready_terminals()):
                return False
            for t in ready_full:
                if rng.random() < 0.5:
                    full.consume(t)
                    sub.consume(t)
    return True


def pluggability(df: pd.DataFrame) -> pd.DataFrame:
    from harvest.baselines import cultivation as cult
    from harvest.routing import magic_state_cultivator as mc
    from harvest.routing.magic_state_factory import MagicStateFactory
    from harvest.routing.magic_state_multiplexed import MultiplexedCultivator

    make_src = inspect.getsource(cult.make_magic_source)
    branch = {"multiplexed": make_src.split("if model == MULTIPLEXED:")[1].split("if model ==")[0],
              "lognormal": make_src.split("if model == LOGNORMAL:")[1].split("from harvest.routing")[0]}
    entries = [
        ("instant", "none (no source: every terminal always ready)", 0, False),
        ("dist4", "MagicStateFactory (fixed latency mu)", loc(MagicStateFactory), False),
        ("dist10", "MagicStateFactory (fixed latency mu)", loc(MagicStateFactory), False),
        ("cultivation", "MagicStateCultivator + exponential_sampler",
         loc(mc.MagicStateCultivator) + loc(mc.exponential_sampler), False),
        ("lognormal", "MagicStateCultivator + lognormal_sampler (new for this paper)",
         loc(mc.lognormal_sampler) + sum(1 for l in branch["lognormal"].splitlines() if l.strip()), True),
        ("multiplexed", "MultiplexedCultivator (new for this paper)",
         loc(MultiplexedCultivator) + sum(1 for l in branch["multiplexed"].splitlines() if l.strip()), True),
    ]
    aware = df[(df["mode"] == "aware")]
    rows = []
    for model, impl, n_loc, new in entries:
        sub = aware[aware.model == model]
        ok = sub[sub.status == "ok"]
        rows.append({
            "model": model, "implementation": impl, "new_for_paper": new,
            "interface_loc": n_loc, "scheduler_or_validator_changes_loc": 0,
            "restriction_stable": restriction_stable(model),
            "schedules": len(sub), "completed": len(ok),
            "validated_pre_pass": int((ok.validated_pre.astype(str) == "True").sum()),
            "validated_post_pass": int((ok.validated_post.astype(str) == "True").sum()),
            "violations": int(pd.to_numeric(ok.n_violations, errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    style.apply()
    import matplotlib.pyplot as plt

    df = read_rows(OUT / "rq4.csv")
    if len(df) == 0:
        print("rq4: no data yet")
        return 0
    agg = curves(df)
    agg.to_csv(OUT / "rq4_curves_per_circuit.csv", index=False)
    lines = agg.groupby(["mode", "model", "level"]).agg(
        m=("m", lambda x: geomean(x)), T_over_cp=("T_over_cp", lambda x: geomean(x)),
        bound_over_cp=("bound_over_cp", lambda x: geomean(x)), n=("circuit", "nunique")).reset_index()
    lines.to_csv(OUT / "rq4_curves.csv", index=False)

    pair = agg.pivot_table(index=["circuit", "model", "level"], columns="mode", values="T").dropna()
    pair["oblivious_over_aware"] = pair["oblivious"] / pair["aware"]
    v4 = agg[agg["mode"] == "oblivious"].set_index(["circuit", "model", "level"]).v4
    pair["v4_violations_unmodified"] = v4.reindex(pair.index)
    comp = pair.reset_index().groupby("model").agg(
        oblivious_over_aware_geomean=("oblivious_over_aware", lambda x: geomean(x)),
        oblivious_over_aware_max=("oblivious_over_aware", "max"),
        schedules=("oblivious_over_aware", "size"),
        invalid_unmodified=("v4_violations_unmodified", lambda x: int((x > 0).sum())),
        v4_violations_mean=("v4_violations_unmodified", "mean")).reset_index()
    comp.to_csv(OUT / "rq4_oblivious_vs_aware.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, model in zip(axes.flat, PANELS):
        color = style.MODEL_COLORS[model]
        for mode, ls in (("aware", "-"), ("oblivious", "--")):
            s = lines[(lines["mode"] == mode) & (lines.model == model)].sort_values("m")
            s = s[s.n >= s.n.max() / 2] if len(s) else s
            if len(s) and not (mode == "oblivious" and model in ("instant", "multiplexed", "lognormal")):
                ax.plot(s.m, s.T_over_cp, ls, marker="o", color=color, label=f"{mode}")
        s = lines[(lines["mode"] == "aware") & (lines.model == model)].sort_values("m")
        if model != "instant" and len(s):
            ax.plot(s.m, s.bound_over_cp, ":", color=style.INK2, label="supply bound T/terminals")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.axhline(1, color=style.AXIS, linewidth=1)
        ax.set_title(model)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("magic terminals")
    for ax in axes[:, 0]:
        ax.set_ylabel("makespan / critical path")
    style.save(fig, OUT / "fig5_magic_terminals")

    plug = pluggability(df)
    plug.to_csv(OUT / "rq4_pluggability.csv", index=False)
    tex = [r"\begin{tabular}{llrcr}", r"\toprule",
           r"Model & Implementation & LoC & restr.\ stable & valid schedules \\", r"\midrule"]
    for _, r in plug.iterrows():
        tex.append(f"{r.model} & {r.implementation} & {r.interface_loc} & "
                   f"{'yes' if r.restriction_stable else 'no'} & {r.validated_post_pass}/{r.schedules} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "rq4_pluggability.tex").write_text("\n".join(tex) + "\n")
    print(comp.round(3).to_string(index=False))
    print(plug.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
