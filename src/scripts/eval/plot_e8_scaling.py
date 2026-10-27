#!/usr/bin/env python3
"""E8 figures: compile time vs. Pauli products per compiler; HARVEST phase breakdown."""

from __future__ import annotations

import numpy as np

from plot_common import kind, load
from harvest.eval import plotting as P

PHASES = ("t_compose_s", "t_layout_s", "t_place_s", "t_schedule_s", "t_prune_s", "t_validate_s")
PHASE_COLORS = ("#BBBBBB", "#999999", "#56B4E9", "#0072B2", "#009E73", "#E69F00")


def compile_time(exp_dir, summary) -> None:
    ct = kind(summary, "compile_time")
    fig, ax = P.figure(P.SINGLE, 2.4)
    if ct.empty:
        P.empty(ax)
    else:
        for compiler, sub in ct.groupby("compiler"):
            sub = sub.sort_values("ops")
            ok = sub[sub["compile_time_s"].notna()]
            ax.plot(ok["ops"], ok["compile_time_s"], marker=P.marker(compiler), ms=3,
                    color=P.color(compiler), label=compiler, lw=0.8)
            to = sub[sub["timed_out"]]
            ax.scatter(to["ops"], to["timeout_s"], marker="x", s=14, color=P.color(compiler))
        ax.axhline(ct["timeout_s"].max(), color="#999999", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Pauli products")
        ax.set_ylabel("Compile time (s); x = timeout")
        ax.legend(frameon=False, fontsize=6)
    P.save(fig, exp_dir / "e8_compile_time.pdf")


def phase_breakdown(exp_dir, summary) -> None:
    ph = kind(summary, "phases")
    fig, ax = P.figure(P.SINGLE, 2.2)
    if ph.empty:
        P.empty(ax)
    else:
        ph = ph.groupby("circuit").mean(numeric_only=True).sort_values("ops")
        picks = np.unique(np.linspace(0, len(ph) - 1, min(5, len(ph))).round().astype(int))
        ph = ph.iloc[picks]
        total = ph[list(PHASES)].fillna(0).sum(axis=1)
        bottom = np.zeros(len(ph))
        x = np.arange(len(ph))
        for phase, c in zip(PHASES, PHASE_COLORS):
            share = ph[phase].fillna(0) / total
            ax.bar(x, share, bottom=bottom, color=c, label=phase[2:-2], width=0.7)
            bottom += share.to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{int(o)} ops\n{t:.0f} s" for n, o, t in
                            zip(ph.index, ph["ops"], total)], fontsize=5)
        ax.set_ylabel("Share of HARVEST compile time")
        ax.legend(frameon=False, fontsize=5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    P.save(fig, exp_dir / "e8_phase_breakdown.pdf")


def main() -> None:
    exp_dir, summary = load(__doc__)
    compile_time(exp_dir, summary)
    phase_breakdown(exp_dir, summary)


if __name__ == "__main__":
    main()
