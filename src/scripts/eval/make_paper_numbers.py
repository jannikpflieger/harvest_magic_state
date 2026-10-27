#!/usr/bin/env python3
"""Write ``results/eval/paper_numbers.tex`` from the experiments' ``summary.csv`` (EVAL_PLAN B.9).

Every number the intro and takeaways quote is a macro defined here, so text
never goes stale.  A value whose experiment has not run yet is emitted as a
bold ``??`` with a comment saying what is missing, never guessed.

Conventions: a "gain" is HARVEST's relative reduction, ``1 - 1/ratio`` with
``ratio = baseline / HARVEST`` (geomean over the pair's shared circuits); a
negative gain means HARVEST is worse.  Where a compiler has several
configurations or layouts the most favorable one for the baseline is used
(the smallest ratio), so the claim is conservative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read(root: Path, exp: str) -> pd.DataFrame:
    try:
        return pd.read_csv(root / exp / "summary.csv", low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["kind"])


class Macros:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def put(self, name: str, value, fmt: str = "{}", note: str = "") -> None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            self.lines.append(f"\\newcommand{{\\{name}}}{{\\textbf{{??}}}}  % missing: {note}")
        else:
            self.lines.append(f"\\newcommand{{\\{name}}}{{{fmt.format(value)}}}" +
                              (f"  % {note}" if note else ""))


def pct(ratio: Optional[float]) -> Optional[float]:
    return None if ratio is None or pd.isna(ratio) else 100 * (1 - 1 / ratio)


def best_ratio(pair: pd.DataFrame, magic: str, prefix: str, metric: str):
    sub = pair[(pair["magic_model"] == magic) & pair["system"].str.startswith(prefix)]
    if prefix == "Silva EAF":
        sub = sub[~sub["system"].str.contains("pruning")]
    sub = sub[sub[metric].notna()]
    if sub.empty:
        return None, None, 0
    row = sub.loc[sub[metric].idxmin()]
    return row[metric], row["system"], int(row["n_shared"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "eval")
    args = parser.parse_args()
    m = Macros()

    suites = json.loads((PROJECT_ROOT / "experiments" / "eval" / "suites.json").read_text())
    m.put("nCoreCircuits", suites["core"]["n"], note="experiments/eval/suites.json core")

    e1 = read(args.root, "e1_sota")
    pair = e1[e1["kind"] == "pairwise"] if "kind" in e1 else e1
    for magic, tag in (("free", "Free"), ("cultivation", "Cult")):
        for prefix, name in (("PureMagic", "PureMagic"), ("Silva EAF + pruning", "EAFPruned"),
                             ("Silva EAF", "EAF"), ("Litinski", "Litinski"), ("O3LS", "OThreeLS")):
            vol, system, n = best_ratio(pair, magic, prefix, "volume_ratio_geomean")
            area, _, _ = best_ratio(pair, magic, prefix, "area_ratio_geomean")
            time, _, _ = best_ratio(pair, magic, prefix, "time_ratio_geomean")
            note = f"E1 {magic}, vs {system} (n={n})" if system else f"E1 {magic} {prefix}"
            m.put(f"volGain{name}{tag}", pct(vol), "{:.0f}\\%", note)
            m.put(f"patchGain{name}{tag}", pct(area), "{:.0f}\\%", note)
            m.put(f"timeGain{name}{tag}", pct(time), "{:.0f}\\%", note)
    ring = pair[(pair["system"] == "Silva EAF [ring]") & (pair["magic_model"] == "free")] \
        if "system" in pair else pair
    m.put("fasterThanEAFCount", int(ring["time_wins"].iloc[0]) if len(ring) else None,
          note="E1 free, ring: circuits where HARVEST is strictly faster than Silva EAF")
    m.put("fasterThanEAFTotal", int(ring["n_shared"].iloc[0]) if len(ring) else None,
          note="E1 free, ring: shared circuits")
    geo = e1[e1["kind"] == "system_geomean"] if "kind" in e1 else e1
    m.put("nSharedSOTA", int(geo["n_shared"].min()) if len(geo) else None,
          note="E1: circuits every system completed")

    e3 = read(args.root, "e3_ablation")
    abl = e3[e3["kind"] == "ablation"] if "kind" in e3 else e3
    gains = []
    for layout, sub in abl.groupby("layout") if len(abl) else []:
        t = sub.set_index("ablation")["T_rel_eaf_geomean"]
        if "pack" in t and "nego" in t:
            gains.append(100 * (1 - t["nego"] / t["pack"]))
    m.put("negoMaxGain", max(gains) if gains else None, "{:.0f}\\%",
          "E3: max over layouts of 1 - T(nego)/T(pack)")
    full = abl[abl["ablation"] == "full"] if len(abl) else abl
    m.put("fullVsEAFMaxGain", pct(1 / full["T_rel_eaf_geomean"].min()) if len(full) else None,
          "{:.0f}\\%", "E3: best layout, 1 - T(full)/T(eaf)")

    e8 = read(args.root, "e8_scaling")
    sp = e8[e8["kind"] == "speedup"] if "kind" in e8 else e8
    for compiler, name in (("PureMagic", "PureMagic"), ("Silva EAF", "EAF"),
                           ("Litinski", "Litinski"), ("O3LS", "OThreeLS")):
        row = sp[sp["compiler"] == compiler] if len(sp) else sp
        m.put(f"compileSpeedupVs{name}", row["speedup_geomean"].iloc[0] if len(row) else None,
              "{:.2g}$\\times$", f"E8: geomean compile time {compiler} / HARVEST")
    m.put("compileSpeedup", sp["speedup_geomean"].min() if len(sp) and sp["speedup_geomean"].notna().any()
          else None, "{:.2g}$\\times$", "E8: smallest speedup over any baseline (<1: slower)")

    e7 = read(args.root, "e7_validity")
    val = e7[e7["kind"] == "validity"] if "kind" in e7 else e7
    m.put("nSchedulesValidated", int(val["schedules_checked"].sum()) if len(val) else None,
          "{:,}", "E7a: HARVEST/Silva schedules checked (pre-pruning)")
    m.put("nValidatorViolations", int(val["violations"].sum()) if len(val) else None,
          note="E7a: violations on unmutated runs (expected 0)")
    mut = e7[e7["kind"] == "mutation"] if "kind" in e7 else e7
    completed = pd.to_numeric(mut["completed"]).sum() if len(mut) else 0
    detected = pd.to_numeric(mut["detected"]).sum() if len(mut) else 0
    m.put("mutationKillRate", 100 * detected / completed if completed else None,
          "{:.0f}\\%", "E7b: detected / activated mutant runs")
    m.put("nMutantRuns", int(completed) if completed else None, note="E7b: activated mutant runs")

    out = args.root / "paper_numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("% Generated by src/scripts/eval/make_paper_numbers.py -- do not edit.\n"
                   + "\n".join(m.lines) + "\n")
    print(f"wrote {out}")
    missing = [line for line in m.lines if "??" in line]
    if missing:
        print(f"{len(missing)} macro(s) missing data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
