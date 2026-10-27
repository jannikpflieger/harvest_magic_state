"""Shared argument parsing and CSV loading for the ``plot_<id>.py`` scripts.

Plot scripts read only ``summary.csv`` / ``data.csv`` (and CSVs written next
to them), so ``run_eval.py --figures-only`` redraws every figure without
rerunning anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--exp-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.exp_dir / "summary.csv"
    try:
        summary = pd.read_csv(summary_path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        summary = pd.DataFrame(columns=["kind"])
    return args.exp_dir, summary


def kind(summary: pd.DataFrame, name: str) -> pd.DataFrame:
    if "kind" not in summary:
        return summary.iloc[0:0]
    return summary[summary["kind"] == name].copy()
