"""Aggregation helpers: geomeans over shared subsets, confidence intervals (EVAL_PLAN A.2)."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def geomean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not pd.isna(v) and float(v) > 0]
    return math.exp(mean(math.log(v) for v in vals)) if vals else None


def ci95(values: Sequence[float]) -> Optional[float]:
    """Half-width of the 95% Student-t confidence interval of the mean."""
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    if len(vals) < 2:
        return None
    try:
        from scipy.stats import t

        quantile = float(t.ppf(0.975, len(vals) - 1))
    except ImportError:  # pragma: no cover - scipy is a project dependency
        quantile = 1.96
    return quantile * stdev(vals) / math.sqrt(len(vals))


def trial_means(df: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean, CI and trial counts per ``keys`` group over successful trials only.

    Failed/timeout trials stay visible through ``trials_total`` / ``trials_ok``.
    """
    rows = []
    for group, sub in df.groupby(list(keys), dropna=False, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        ok = sub[sub["status"] == "ok"]
        row = dict(zip(keys, group))
        row["trials_total"] = len(sub)
        row["trials_ok"] = len(ok)
        row["status"] = "ok" if len(ok) == len(sub) else (
            "partial" if len(ok) else ",".join(sorted(set(sub["status"])))
        )
        for metric in metrics:
            vals = ok[metric].dropna().astype(float).tolist() if metric in ok else []
            row[metric] = mean(vals) if vals else None
            row[f"{metric}_ci95"] = ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ratios(
    df: pd.DataFrame, system_col: str, reference: Any, metric: str, on: Sequence[str],
) -> pd.DataFrame:
    """Per-``on`` ratio ``system / reference`` for every other system, shared subset only."""
    ok = df[df[metric].notna() & (df[metric] > 0)]
    ref = ok[ok[system_col] == reference].set_index(list(on))[metric]
    out = []
    for system, sub in ok[ok[system_col] != reference].groupby(system_col):
        sub = sub.set_index(list(on))[metric]
        shared = sub.index.intersection(ref.index)
        for key in shared:
            out.append({system_col: system, **dict(zip(on, key if isinstance(key, tuple) else (key,))),
                        "ratio": float(sub[key]) / float(ref[key])})
    return pd.DataFrame(out)


def geomean_table(ratios: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows = []
    for group, sub in ratios.groupby(list(by), sort=True):
        group = group if isinstance(group, tuple) else (group,)
        rows.append({**dict(zip(by, group)), "geomean_ratio": geomean(sub["ratio"]),
                     "n_shared": len(sub),
                     "wins": int((sub["ratio"] > 1 + 1e-9).sum()),
                     "ties": int((abs(sub["ratio"] - 1) <= 1e-9).sum()),
                     "losses": int((sub["ratio"] < 1 - 1e-9).sum())})
    return pd.DataFrame(rows)


def require_single_regime(df: pd.DataFrame, column: str = "comparison_mode") -> str:
    """Refuse to aggregate across comparison regimes (EVAL_PLAN A.2 rule 1)."""
    regimes = sorted(set(df[column].dropna()))
    if len(regimes) > 1:
        raise ValueError(f"refusing to pool comparison regimes {regimes}")
    return regimes[0] if regimes else ""


def shared_circuits(df: pd.DataFrame, system_col: str, systems: Iterable[Any],
                    metric: str, circuit_col: str = "circuit") -> List[str]:
    ok = df[df[metric].notna()]
    sets = [set(ok[ok[system_col] == s][circuit_col]) for s in systems]
    return sorted(set.intersection(*sets)) if sets else []
