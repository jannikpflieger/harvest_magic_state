"""``data.csv`` -> ``summary.csv`` (+ LaTeX tables) for every experiment (EVAL_PLAN B.6).

Every summary is tidy: a ``kind`` column says which table a row belongs to,
geomeans carry ``n_shared``, and nothing pools comparison regimes.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT_SIBLINGS = ("e1_sota", "e2_characterize", "e3_ablation", "e4_provisioning",
                      "e5_magic_models", "e6_placement", "e7_validity", "e8_scaling")


def geomean(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not pd.isna(v) and float(v) > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def ci95(values) -> Optional[float]:
    from harvest.eval.stats import ci95 as _ci

    return _ci(list(values))


def ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"] if "status" in df else df.iloc[0:0]


def system_label(row) -> str:
    layout = row.get("layout")
    return f"{row['config']} [{layout}]" if isinstance(layout, str) and layout else row["config"]


def trial_mean(df: pd.DataFrame, keys: List[str], metrics: List[str]) -> pd.DataFrame:
    """Mean (+CI) over successful trials per ``keys``; trial counts keep failures visible."""
    if df.empty:
        return pd.DataFrame(columns=keys + metrics)
    rows = []
    for group, sub in df.groupby(keys, dropna=False, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        good = sub[sub["status"] == "ok"]
        row = dict(zip(keys, group))
        row.update(trials_total=len(sub), trials_ok=len(good),
                   status="ok" if len(good) == len(sub) else ("partial" if len(good) else
                                                              ",".join(sorted(set(sub["status"])))))
        for metric in metrics:
            vals = good[metric].dropna().astype(float) if metric in good else pd.Series(dtype=float)
            row[metric] = vals.mean() if len(vals) else None
            row[f"{metric}_ci95"] = ci95(vals) if len(vals) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def latex_escape(text) -> str:
    text = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("^", r"\^{}")):
        text = text.replace(a, b)
    return text


# --------------------------------------------------------------------------
# E1 -- head-to-head
# --------------------------------------------------------------------------

E1_METRICS = ["T_cycles", "area_tiles", "volume_tile_cycles", "T_over_lb", "area_over_lb",
              "T_over_flasq", "compile_time_s", "volume_initial", "volume_pruned"]


def _reference_for(system: str, layout) -> str:
    """HARVEST at the same layout for in-tree baselines, HARVEST [ring] otherwise."""
    if isinstance(layout, str) and layout:
        return f"HARVEST [{layout}]"
    return "HARVEST [ring]"


def summarize_e1(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    from harvest.eval.stats import require_single_regime

    if df.empty:
        return pd.DataFrame()
    require_single_regime(df)
    df = df.copy()
    df["system"] = df.apply(system_label, axis=1)
    per = trial_mean(df, ["magic_model", "system", "compiler", "config", "layout", "circuit"],
                     E1_METRICS)
    per["kind"] = "circuit"
    rows = [per]
    for magic, sub in per[per["trials_ok"] > 0].groupby("magic_model"):
        systems = sorted(sub["system"].unique())
        sets = [set(sub[sub["system"] == s]["circuit"]) for s in systems]
        shared = sorted(set.intersection(*sets)) if sets else []
        for system in systems:
            s = sub[(sub["system"] == system) & sub["circuit"].isin(shared)]
            own = sub[sub["system"] == system]
            rows.append(pd.DataFrame([{
                "kind": "system_geomean", "magic_model": magic, "system": system,
                "compiler": own["compiler"].iloc[0], "config": own["config"].iloc[0],
                "layout": own["layout"].iloc[0],
                "area_over_lb": geomean(s["area_over_lb"]),
                "T_over_lb": geomean(s["T_over_lb"]),
                "T_over_flasq": geomean(s["T_over_flasq"]) if s["T_over_flasq"].notna().all() else None,
                "volume_over_lb": geomean(s["area_over_lb"] * s["T_over_lb"]),
                "n_shared": len(s), "n_own": len(own),
            }]))
        # Pairwise: baseline / HARVEST on each pair's shared subset.
        for system in systems:
            if system.startswith("HARVEST"):
                continue
            s = sub[sub["system"] == system].set_index("circuit")
            ref_name = _reference_for(system, s["layout"].iloc[0])
            ref = sub[sub["system"] == ref_name].set_index("circuit")
            shared_pair = sorted(set(s.index) & set(ref.index))
            if not shared_pair:
                continue
            entry = {"kind": "pairwise", "magic_model": magic, "system": system,
                     "reference": ref_name, "compiler": s["compiler"].iloc[0],
                     "n_shared": len(shared_pair)}
            for metric, name in (("T_cycles", "time"), ("area_tiles", "area"),
                                 ("volume_tile_cycles", "volume")):
                ratios = s.loc[shared_pair, metric] / ref.loc[shared_pair, metric]
                entry[f"{name}_ratio_geomean"] = geomean(ratios)
                entry[f"{name}_wins"] = int((ratios > 1 + 1e-9).sum())   # HARVEST better
                entry[f"{name}_losses"] = int((ratios < 1 - 1e-9).sum())
            rows.append(pd.DataFrame([entry]))
            for circuit in shared_pair:
                rows.append(pd.DataFrame([{
                    "kind": "ratio", "magic_model": magic, "system": system,
                    "compiler": s["compiler"].iloc[0], "reference": ref_name, "circuit": circuit,
                    "volume_ratio": s.loc[circuit, "volume_tile_cycles"] / ref.loc[circuit, "volume_tile_cycles"],
                    "time_ratio": s.loc[circuit, "T_cycles"] / ref.loc[circuit, "T_cycles"],
                }]))
    rows.append(_family_rows(df))
    summary = pd.concat(rows, ignore_index=True)
    _write_e1_table(summary, out / "e1_table.tex")
    return summary


def _family_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per circuit family: geomean ratio baseline / HARVEST (same reference rule as above)."""
    if "family" not in df:
        return pd.DataFrame()
    per = trial_mean(df, ["magic_model", "system", "compiler", "layout", "family", "circuit"],
                     ["T_cycles", "area_tiles", "volume_tile_cycles"])
    per = per[per["trials_ok"] > 0]
    out = []
    for (magic, family), sub in per.groupby(["magic_model", "family"]):
        for system, s in sub.groupby("system"):
            if system.startswith("HARVEST"):
                continue
            ref = sub[sub["system"] == _reference_for(system, s["layout"].iloc[0])].set_index("circuit")
            s = s.set_index("circuit")
            shared = sorted(set(s.index) & set(ref.index))
            if not shared:
                continue
            out.append({
                "kind": "family", "magic_model": magic, "family": family, "system": system,
                "compiler": s["compiler"].iloc[0], "n_shared": len(shared),
                **{f"{name}_ratio_geomean": geomean(s.loc[shared, metric] / ref.loc[shared, metric])
                   for metric, name in (("T_cycles", "time"), ("area_tiles", "area"),
                                        ("volume_tile_cycles", "volume"))},
            })
    return pd.DataFrame(out)


def _write_e1_table(summary: pd.DataFrame, path: Path) -> None:
    pair = summary[summary["kind"] == "pairwise"]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Magic & Baseline (vs.\ HARVEST) & Time & Area & Volume & $n$ \\", r"\midrule"]
    for _, r in pair.sort_values(["magic_model", "system"]).iterrows():
        def fmt(v):
            return "--" if v is None or pd.isna(v) else f"{v:.2f}$\\times$"
        lines.append(f"{latex_escape(r['magic_model'])} & {latex_escape(r['system'])} & "
                     f"{fmt(r['time_ratio_geomean'])} & {fmt(r['area_ratio_geomean'])} & "
                     f"{fmt(r['volume_ratio_geomean'])} & {int(r['n_shared'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              "% ratio = baseline / HARVEST (geomean over each pair's shared circuits); >1: HARVEST better"]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# E2 -- where HARVEST wins and loses
# --------------------------------------------------------------------------


def summarize_e2(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    # Rows come from the head-to-head experiment(s) named in inputs_from.
    df = df[df["exp_id"].astype(str).str.startswith("e1_") & (df["magic_model"] == "free")] \
        if not df.empty else df
    if df.empty:
        return pd.DataFrame()
    df = ok_rows(df)
    rows = []
    feats = ["feat_parallelism", "feat_t_fraction", "feat_mean_pauli_weight", "feat_y_fraction",
             "feat_ops", "num_qubits"]
    for layout, sub in df[df["layout"].notna()].groupby("layout"):
        harvest = sub[sub["config"] == "HARVEST"].groupby("circuit").mean(numeric_only=True)
        for config in ("Silva EAF", "Silva EAF + pruning"):
            eaf = sub[sub["config"] == config].groupby("circuit").mean(numeric_only=True)
            shared = sorted(set(harvest.index) & set(eaf.index))
            for c in shared:
                rows.append({
                    "kind": "circuit", "layout": layout, "baseline": config, "circuit": c,
                    "volume_ratio": eaf.loc[c, "volume_tile_cycles"] / harvest.loc[c, "volume_tile_cycles"],
                    "time_ratio": eaf.loc[c, "T_cycles"] / harvest.loc[c, "T_cycles"],
                    "area_ratio": eaf.loc[c, "area_tiles"] / harvest.loc[c, "area_tiles"],
                    **{f: harvest.loc[c, f] for f in feats if f in harvest},
                })
    per = pd.DataFrame(rows)
    out_rows = [per]
    for (layout, baseline), sub in per.groupby(["layout", "baseline"]):
        for metric in ("time_ratio", "volume_ratio"):
            r = sub[metric]
            out_rows.append(pd.DataFrame([{
                "kind": "winloss", "layout": layout, "baseline": baseline, "metric": metric,
                "wins": int((r > 1 + 1e-9).sum()), "ties": int((abs(r - 1) <= 1e-9).sum()),
                "losses": int((r < 1 - 1e-9).sum()), "n_shared": len(r),
                "geomean_ratio": geomean(r),
                "spearman_parallelism": (sub[[metric, "feat_parallelism"]].corr("spearman").iloc[0, 1]
                                         if len(sub) > 2 else None),
            }]))
    return pd.concat(out_rows, ignore_index=True)


# --------------------------------------------------------------------------
# E3 -- scheduler ablation
# --------------------------------------------------------------------------


def _ablation_name(row) -> Optional[str]:
    if row["baseline"] == "silva":
        return "eaf"
    if row.get("scheduler_mode") == "steiner_tree":
        return "seq"
    return row["variant"] if row["variant"] in ("pack", "nego", "+prio", "+match", "+fill", "full") else None


def summarize_e3(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    rows = []
    abl = df[df["stage"].isin(["ablation", "seq", "eaf"])].copy()
    if not abl.empty:
        abl["ablation"] = abl.apply(_ablation_name, axis=1)
        abl = abl[abl["ablation"].notna()]
        per = trial_mean(abl, ["layout", "ablation", "circuit"],
                         ["T_cycles", "T_over_lb", "compile_time_s", "mean_ops_per_step",
                          "mean_deferrals_per_step", "critical_path"])
        per["kind"] = "ablation_circuit"
        rows.append(per)
        for layout, sub in per[per["trials_ok"] > 0].groupby("layout"):
            variants = sorted(sub["ablation"].unique())
            sets = [set(sub[sub["ablation"] == v]["circuit"]) for v in variants]
            shared = sorted(set.intersection(*sets)) if sets else []
            eaf = sub[sub["ablation"] == "eaf"].set_index("circuit")
            for v in variants:
                s = sub[(sub["ablation"] == v) & sub["circuit"].isin(shared)].set_index("circuit")
                rel = (s["T_cycles"] / eaf.loc[s.index, "T_cycles"]) if not eaf.empty else None
                rows.append(pd.DataFrame([{
                    "kind": "ablation", "layout": layout, "ablation": v,
                    "T_rel_eaf_geomean": geomean(rel) if rel is not None else None,
                    "T_over_lb_geomean": geomean(s["T_over_lb"]),
                    "compile_time_s_geomean": geomean(s["compile_time_s"]),
                    "mean_ops_per_step": s["mean_ops_per_step"].mean(),
                    "mean_deferrals_per_step": s["mean_deferrals_per_step"].mean(),
                    "n_shared": len(s),
                }]))
            cp = sub.drop_duplicates("circuit").set_index("circuit")
            rows.append(pd.DataFrame([{
                "kind": "ablation", "layout": layout, "ablation": "critical_path",
                "T_rel_eaf_geomean": geomean(cp.loc[shared, "critical_path"] / eaf.loc[shared, "T_cycles"])
                if shared and not eaf.empty else None,
                "T_over_lb_geomean": 1.0, "n_shared": len(shared),
            }]))
    for stage, knobs in (("alpha_beta", ["cfg_alpha", "cfg_beta"]),
                         ("iters", ["cfg_max_iters", "cfg_stagnation_limit"])):
        sens = ok_rows(df[df["stage"] == stage])
        if sens.empty:
            continue
        for layout, sub in sens.groupby("layout"):
            grid = sub.groupby(knobs + ["circuit"])[["T_cycles", "compile_time_s"]].mean().reset_index()
            counts = grid.groupby(knobs)["circuit"].nunique()
            full = counts.max()
            for key, cell in grid.groupby(knobs):
                rows.append(pd.DataFrame([{
                    "kind": f"sensitivity_{stage}", "layout": layout,
                    **dict(zip(knobs, key)),
                    "T_geomean": geomean(cell["T_cycles"]),
                    "compile_time_s_geomean": geomean(cell["compile_time_s"]),
                    "n_shared": len(cell), "complete": len(cell) == full,
                }]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# E4 -- schedule large, report small
# --------------------------------------------------------------------------


def parallelism_class(p) -> str:
    if p is None or pd.isna(p):
        return "unknown"
    return "low" if p < 2 else ("medium" if p < 4 else "high")


def summarize_e4(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    df = ok_rows(df)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["circuit_class"] = df["feat_parallelism"].map(parallelism_class)
    df["volume_lb"] = df["area_lb_tiles"] * df["T_lb_cycles"]
    df["v_init_rel"] = df["volume_initial"] / df["volume_lb"]
    df["v_pruned_rel"] = df["volume_pruned"] / df["volume_lb"]
    df["a_init_rel"] = df["area_initial_tiles"] / df["area_lb_tiles"]
    df["a_pruned_rel"] = df["area_pruned_tiles"] / df["area_lb_tiles"]
    keys = ["magic_model", "circuit_class", "layout_preset", "layout"]
    rows = []
    for key, sub in df.groupby(keys):
        per_circuit = sub.groupby("circuit").mean(numeric_only=True)
        entry = dict(zip(keys, key))
        entry.update({
            "kind": "layout",
            "magic_count": sub["magic_tiles_initial"].median(),
            "area_initial_rel": geomean(per_circuit["a_init_rel"]),
            "area_pruned_rel": geomean(per_circuit["a_pruned_rel"]),
            "volume_initial_rel": geomean(per_circuit["v_init_rel"]),
            "volume_pruned_rel": geomean(per_circuit["v_pruned_rel"]),
            "T_over_lb": geomean(per_circuit["T_over_lb"]),
            "n_circuits": len(per_circuit),
            "all_validated_post": bool(sub["validated_post"].fillna(False).astype(bool).all()),
        })
        for kind in ("routing", "magic", "data"):
            init = sub[f"{kind}_tiles_initial"].sum()
            kept = sub[f"{kind}_tiles_pruned"].sum()
            entry[f"{kind}_tiles_initial_sum"] = init
            entry[f"{kind}_tiles_pruned_sum"] = kept
            entry[f"{kind}_removed_frac"] = (1 - kept / init) if init else None
        rows.append(entry)
    summary = pd.DataFrame(rows)
    # Pruning must never change T: the schedule is identical and still valid.
    if "validated_post" in df and not df["validated_post"].fillna(False).astype(bool).all():
        bad = df[~df["validated_post"].fillna(False).astype(bool)]
        raise SystemExit(f"E4: {len(bad)} pruned schedule(s) failed validation")
    return summary


# --------------------------------------------------------------------------
# E5 -- magic-state protocols
# --------------------------------------------------------------------------


def _model_label(row) -> str:
    model = row["magic_model"]
    if model == "free":
        return "free"
    mean = row["magic_mean_cycles"]
    mean = int(mean) if float(mean).is_integer() else mean
    return {"fixed": f"fixed mu={mean}", "geometric": f"geometric mean={mean}",
            "cultivation": f"exponential mean={mean}"}[model]


def summarize_e5(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["model"] = df.apply(_model_label, axis=1)
    df["capacity"] = df["magic_tiles_initial"]
    df["capacity_req"] = df["layout_magic_ring"].fillna(0).astype(int)
    per = trial_mean(df, ["model", "magic_model", "magic_mean_cycles", "capacity_req", "circuit"],
                     ["T_cycles", "T_over_lb", "steps_idle", "magic_wait_cycles",
                      "terminal_utilization", "mean_op_delay_steps", "capacity"])
    stds = (ok_rows(df).groupby(["model", "capacity_req", "circuit"])["T_cycles"].std()
            .rename("T_std").reset_index())
    per = per.merge(stds, on=["model", "capacity_req", "circuit"], how="left")
    per["kind"] = "circuit"
    rows = [per]
    for (model, cap), sub in per[per["trials_ok"] > 0].groupby(["model", "capacity_req"]):
        rows.append(pd.DataFrame([{
            "kind": "model_capacity", "model": model, "capacity_req": cap,
            "magic_model": sub["magic_model"].iloc[0],
            "magic_mean_cycles": sub["magic_mean_cycles"].iloc[0],
            "capacity_geomean": geomean(sub["capacity"]),
            "T_over_lb_geomean": geomean(sub["T_over_lb"]),
            "utilization_mean": sub["terminal_utilization"].mean(),
            "idle_steps_mean": sub["steps_idle"].mean(),
            "mean_op_delay_steps": sub["mean_op_delay_steps"].mean(),
            "T_cv_mean": (sub["T_std"] / sub["T_cycles"]).mean(),
            "n_circuits": len(sub),
        }]))
    seeds = ok_rows(df[df["magic_model"].isin(["cultivation", "geometric"])])
    if not seeds.empty:
        seeds = seeds[["model", "capacity_req", "circuit", "trial", "T_cycles", "T_over_lb"]].copy()
        means = seeds.groupby(["model", "capacity_req", "circuit"])["T_cycles"].transform("mean")
        seeds["T_rel_mean"] = seeds["T_cycles"] / means
        seeds["kind"] = "seed"
        rows.append(seeds)
    _write_modularity(out / "e5_modularity.tex")
    return pd.concat(rows, ignore_index=True)


def _class_loc(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return node.end_lineno - node.lineno + 1
    return 0


def _write_modularity(path: Path) -> None:
    src = PROJECT_ROOT / "src" / "harvest"
    scheduler = (src / "routing" / "scheduler.py").read_text()
    entries = [
        ("Free (always ready)", "--", 0),
        ("Fixed latency (distillation-like)", "MagicStateFactory",
         _class_loc(src / "routing" / "magic_state_factory.py", "MagicStateFactory")),
        ("Cultivation (any sampler)", "MagicStateCultivator",
         _class_loc(src / "routing" / "magic_state_cultivator.py", "MagicStateCultivator")),
        ("  exponential sampler", "exponential\\_sampler",
         _class_loc(src / "routing" / "magic_state_cultivator.py", "exponential_sampler")),
        ("  geometric sampler", "geometric\\_sampler",
         _class_loc(src / "routing" / "magic_state_cultivator.py", "geometric_sampler")),
    ]
    mentions = sum(scheduler.count(name) for name in
                   ("MagicStateFactory", "MagicStateCultivator", "sampler", "cultivat"))
    lines = [r"\begin{tabular}{llr}", r"\toprule", r"Supply model & Class & LoC \\", r"\midrule"]
    lines += [f"{a} & \\texttt{{{b}}} & {c} \\\\" for a, b, c in entries]
    lines += [r"\midrule",
              f"Scheduler lines referring to a specific model & & {mentions} \\\\",
              r"\bottomrule", r"\end{tabular}",
              "% scheduler.py only calls the MagicStateSource protocol (tick, get_ready_terminals, "
              "consume, record_wait_cycle)."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# E6 -- placement
# --------------------------------------------------------------------------


def summarize_e6(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    df = ok_rows(df)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["placement_label"] = df["placement"].replace({"circuit_aware_preset": "circuit_aware"})
    per = df.groupby(["layout", "placement_label", "circuit"]).agg(
        T_cycles=("T_cycles", "mean"), placement_cost=("placement_cost", "mean"),
        mean_route_len=("mean_route_len", "mean"), t_place_s=("t_place_s", "mean"),
        n=("T_cycles", "size")).reset_index()
    rows = []
    for layout, sub in per.groupby("layout"):
        idx = sub[sub["placement_label"] == "index"].set_index("circuit")
        for label in ("random", "circuit_aware"):
            other = sub[sub["placement_label"] == label].set_index("circuit")
            for c in sorted(set(idx.index) & set(other.index)):
                rows.append({
                    "kind": "pair", "layout": layout, "placement": label, "circuit": c,
                    "speedup": idx.loc[c, "T_cycles"] / other.loc[c, "T_cycles"],
                    "d_cost_rel": (other.loc[c, "placement_cost"] - idx.loc[c, "placement_cost"])
                                  / idx.loc[c, "placement_cost"] if idx.loc[c, "placement_cost"] else None,
                    "d_T_rel": (other.loc[c, "T_cycles"] - idx.loc[c, "T_cycles"]) / idx.loc[c, "T_cycles"],
                    "d_route_len": other.loc[c, "mean_route_len"] - idx.loc[c, "mean_route_len"],
                    "t_place_s": other.loc[c, "t_place_s"],
                })
    pairs = pd.DataFrame(rows)
    out_rows = [per.assign(kind="circuit"), pairs]
    for (layout, label), sub in pairs.groupby(["layout", "placement"]):
        out_rows.append(pd.DataFrame([{
            "kind": "speedup", "layout": layout, "placement": label,
            "speedup_geomean": geomean(sub["speedup"]), "n_shared": len(sub),
            "faster": int((sub["speedup"] > 1).sum()), "slower": int((sub["speedup"] < 1).sum()),
        }]))
    return pd.concat(out_rows, ignore_index=True)


# --------------------------------------------------------------------------
# E7 -- correctness and robustness
# --------------------------------------------------------------------------


def summarize_e7(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    root = out.parent
    # (a) every unmutated HARVEST / Silva schedule of the other experiments.
    checked = []
    for exp in sorted(p.name for p in root.glob("e*") if p.is_dir()):
        if exp.startswith(("e2_", "e7_")):  # aggregates of other experiments' rows
            continue
        path = root / exp / "data.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame = frame[frame["validated_pre"].notna()]
            checked.append(frame.assign(source_exp=exp))
    if checked:
        allv = pd.concat(checked, ignore_index=True)
        for exp, sub in allv.groupby("source_exp"):
            rows.append({
                "kind": "validity", "source_exp": exp,
                "schedules_checked": len(sub),
                "pruned_checked": int(sub["validated_post"].notna().sum()),
                "violations": int(sub["n_violations"].fillna(0).sum()),
                "invalid_schedules": int((sub["n_violations"].fillna(0) > 0).sum()),
                "validate_time_s_total": sub["validate_time_s"].sum(),
                "compile_time_s_total": sub["compile_time_s"].sum(),
            })
        for _, r in allv.iterrows():
            rows.append({"kind": "validator_time", "source_exp": r["source_exp"],
                         "ops": r.get("ops"), "validate_time_s": r.get("validate_time_s"),
                         "compile_time_s": r.get("compile_time_s")})
        harvest = allv[(allv["baseline"] == "harvest") & (allv["status"] == "ok")]
        for _, r in harvest.iterrows():
            lam = 0 if r["magic_model"] == "free" else (
                r["magic_mean_cycles"] if r["magic_model"] == "fixed" else None)
            rows.append({
                "kind": "progress", "source_exp": r["source_exp"], "circuit": r["circuit"],
                "layout": r.get("layout"), "magic_model": r["magic_model"],
                "T_over_cp": r["T_cycles"] / r["critical_path"] if r.get("critical_path") else None,
                "T_over_bound": (r["T_cycles"] / ((lam + 1) * r["ops"]))
                if lam is not None and r.get("ops") else None,
            })
    # (b) mutation study.
    mut = df[df.get("stage", pd.Series(dtype=str)) == "mutation"] if not df.empty else df
    if not mut.empty:
        for (fault, magic), sub in mut.groupby(["fault", "magic_model"]):
            # Kill rate over runs where the fault changed behavior; a fault that
            # never activates is an equivalent mutant, reported separately.
            activated = sub["fault_activations"].fillna(0) > 0
            completed = sub[(sub["status"] == "ok") & activated]
            detected = completed["n_violations"].fillna(0) > 0
            entry = {"kind": "mutation", "fault": fault, "magic_model": magic,
                     "runs": len(sub), "not_activated": int((~activated).sum()),
                     "crashed": int((activated & (sub["status"] != "ok")).sum()),
                     "completed": len(completed), "detected": int(detected.sum()),
                     "kill_rate": detected.mean() if len(completed) else None}
            for rule in ("V1", "V2", "V3", "V4"):
                entry[f"caught_{rule}"] = int((completed[f"violations_{rule}"].fillna(0) > 0).sum())
            entry["first_rule_mode"] = (completed.loc[detected, "first_violation_rule"].mode().iloc[0]
                                        if detected.any() else None)
            rows.append(entry)
        for _, r in mut.iterrows():
            rows.append({"kind": "mutation_run", "fault": r["fault"], "magic_model": r["magic_model"],
                         "circuit": r["circuit"], "layout": r.get("layout"), "status": r["status"],
                         "activations": r.get("fault_activations"),
                         "killed": bool((r.get("n_violations") or 0) > 0),
                         "first_violation_rule": r.get("first_violation_rule"),
                         "n_violations": r.get("n_violations")})
    # (d) trace robustness, written by run_trace_replay.py.
    trace = out / "e7_trace_robustness.csv"
    if trace.exists():
        t = pd.read_csv(trace)
        for key, sub in t.groupby(["circuit_class", "layout"]):
            rows.append({"kind": "trace", "circuit_class": key[0], "layout": key[1],
                         "frac_magic_not_ready_mean": sub["frac_magic_not_ready"].mean(),
                         "frac_steps_delayed_mean": sub["frac_steps_delayed"].mean(),
                         "replays": len(sub)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E8 -- compile time and scalability
# --------------------------------------------------------------------------


def summarize_e8(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    rows = []
    for (compiler, circuit), sub in df.groupby(["compiler", "circuit"]):
        good = sub[sub["status"] == "ok"]
        rows.append({
            "kind": "compile_time", "compiler": compiler, "circuit": circuit,
            "ops": sub["feat_ops"].iloc[0],
            # Median over a multi-configuration compiler's variants and trials.
            "compile_time_s": good["compile_time_s"].median() if len(good) else None,
            "timed_out": bool((sub["status"] == "timeout").any()),
            "status": "ok" if len(good) == len(sub) else ",".join(sorted(set(sub["status"]))),
            "timeout_s": 1800.0,
            "peak_rss_mb": sub["process_peak_rss_mb"].max() if "process_peak_rss_mb" in sub else None,
        })
    h = ok_rows(df[df["baseline"] == "harvest"])
    phases = ["t_compose_s", "t_layout_s", "t_place_s", "t_schedule_s", "t_prune_s", "t_validate_s"]
    for _, r in h.iterrows():
        rows.append({"kind": "phases", "compiler": "HARVEST", "circuit": r["circuit"],
                     "ops": r["feat_ops"], **{p: r.get(p) for p in phases}})
    summary = pd.DataFrame(rows)
    ct = summary[summary["kind"] == "compile_time"]
    ref = ct[ct["compiler"] == "HARVEST"].set_index("circuit")["compile_time_s"]
    for compiler, sub in ct[ct["compiler"] != "HARVEST"].groupby("compiler"):
        s = sub.set_index("circuit")["compile_time_s"].dropna()
        shared = sorted(set(s.index) & set(ref.dropna().index))
        summary = pd.concat([summary, pd.DataFrame([{
            "kind": "speedup", "compiler": compiler,
            "speedup_geomean": geomean(s[shared] / ref[shared]) if shared else None,
            "n_shared": len(shared),
        }])], ignore_index=True)
    return summary


SUMMARIZERS: Dict[str, Callable[[pd.DataFrame, Path], pd.DataFrame]] = {
    "e1_sota": summarize_e1, "e2_characterize": summarize_e2, "e3_ablation": summarize_e3,
    "e4_provisioning": summarize_e4, "e5_magic_models": summarize_e5,
    "e6_placement": summarize_e6, "e7_validity": summarize_e7, "e8_scaling": summarize_e8,
}


def summarize(exp_id: str, df: pd.DataFrame, out: Path) -> pd.DataFrame:
    summary = SUMMARIZERS[exp_id](df, out)
    tmp = out / "summary.csv.tmp"
    summary.to_csv(tmp, index=False)
    tmp.replace(out / "summary.csv")
    print(f"summary.csv: {len(summary)} rows")
    return summary
