#!/usr/bin/env python3
"""E7 figures: mutation kill matrix, T / critical path, validator time, trace robustness."""

from __future__ import annotations

import numpy as np

from plot_common import kind, load
from harvest.eval import plotting as P

RULES = ("V1", "V2", "V3", "V4")


def mutation_matrix(exp_dir, summary) -> None:
    mut = kind(summary, "mutation")
    fig, ax = P.figure(P.SINGLE, 2.2)
    if mut.empty:
        P.empty(ax)
    else:
        mut["row"] = mut["fault"] + " (" + mut["magic_model"] + ")"
        mut = mut.sort_values("row")
        grid = np.array([[r[f"caught_{rule}"] / r["completed"] if r["completed"] else np.nan
                          for rule in RULES] + [r["kill_rate"] if r["completed"] else np.nan]
                         for _, r in mut.iterrows()], dtype=float)
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(RULES) + 1))
        ax.set_xticklabels(list(RULES) + ["any"])
        ax.set_yticks(range(len(mut)))
        ax.set_yticklabels(mut["row"], fontsize=6)
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.0%}", ha="center", va="center", fontsize=5,
                    color="white" if not np.isnan(v) and v > 0.5 else "black")
        ax.set_title("Share of mutant runs flagged, by rule")
    P.save(fig, exp_dir / "e7_mutation_matrix.pdf")


def progress(exp_dir, summary) -> None:
    prog = kind(summary, "progress")
    fig, axes = P.figure(P.DOUBLE, 2.0, ncols=2, squeeze=False)
    if prog.empty:
        P.empty(axes[0][0])
        P.empty(axes[0][1])
    else:
        prog["group"] = prog["layout"].fillna("?") + "\n" + prog["magic_model"]
        groups = sorted(prog["group"].unique())
        for ax, col, label, bound in ((axes[0][0], "T_over_cp", "T / critical path", 1.0),
                                      (axes[0][1], "T_over_bound", "T / ((Lambda+1)|O|)", 1.0)):
            data = [prog[prog["group"] == g][col].dropna() for g in groups]
            keep = [i for i, d in enumerate(data) if len(d)]
            if not keep:
                P.empty(ax, "bound not defined for these models")
                continue
            parts = ax.violinplot([data[i] for i in keep], showmedians=True, widths=0.8)
            for body in parts["bodies"]:
                body.set_facecolor("#0072B2")
                body.set_alpha(0.5)
            ax.axhline(bound, color="#D55E00", lw=0.8, ls="--")
            ax.set_xticks(range(1, len(keep) + 1))
            ax.set_xticklabels([groups[i] for i in keep], fontsize=5)
            ax.set_yscale("log")
            ax.set_ylabel(label)
    P.save(fig, exp_dir / "e7_T_over_cp.pdf")


def validator_time(exp_dir, summary) -> None:
    vt = kind(summary, "validator_time")
    vt = vt[vt["validate_time_s"].notna() & vt["ops"].notna()] if not vt.empty else vt
    fig, ax = P.figure(P.SINGLE, 2.0)
    if vt.empty:
        P.empty(ax)
    else:
        ax.scatter(vt["ops"], vt["validate_time_s"], s=3, alpha=0.4, color="#0072B2", linewidths=0,
                   label="validator (pre + post)")
        ax.scatter(vt["ops"], vt["compile_time_s"], s=3, alpha=0.25, color="#BBBBBB", linewidths=0,
                   label="compile")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Pauli products")
        ax.set_ylabel("Time (s)")
        ax.legend(frameon=False, fontsize=6)
    P.save(fig, exp_dir / "e7_validator_time.pdf")


def trace(exp_dir) -> None:
    import pandas as pd

    path = exp_dir / "e7_trace_robustness.csv"
    fig, ax = P.figure(P.SINGLE, 2.0)
    try:
        t = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        t = pd.DataFrame()
    if t.empty:
        P.empty(ax)
    else:
        classes = [c for c in ("low", "medium", "high") if c in set(t["circuit_class"])]
        data = [t[t["circuit_class"] == c]["frac_magic_not_ready"] for c in classes]
        parts = ax.violinplot(data, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor("#D55E00")
            body.set_alpha(0.5)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([f"{c} parallelism" for c in classes])
        ax.set_ylabel("Magic ops not ready\nunder a fresh trace")
    P.save(fig, exp_dir / "e7_trace_robustness.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    mutation_matrix(exp_dir, summary)
    progress(exp_dir, summary)
    validator_time(exp_dir, summary)
    trace(exp_dir)


if __name__ == "__main__":
    main()
