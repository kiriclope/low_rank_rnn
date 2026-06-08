"""
analyze.py — load and summarise results.jsonl from a sweep.

Usage (script)
--------------
    python analyze.py
    python analyze.py --results ../results/dual/vanilla/results.jsonl

Usage (notebook / REPL)
-----------------------
    from analyze import load_results, summary_table
    df = load_results("../results/dual/vanilla/results.jsonl")
    print(summary_table(df))
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> pd.DataFrame:
    """
    Read results.jsonl and return a flat DataFrame.

    Each row is one completed run.  Config fields and accuracy fields are
    flattened into columns; loss histories are kept as lists.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue

            row = {"run_id": r["run_id"]}

            # Config fields → flat columns
            for k, v in r.get("config", {}).items():
                row[k] = v

            # Accuracy per stage → flat columns
            for stage, metrics in r.get("accuracy", {}).items():
                for metric, val in metrics.items():
                    row[f"{stage}/{metric}"] = val

            # Final loss per stage
            for stage, val in r.get("final_train_loss", {}).items():
                row[f"{stage}/final_train_loss"] = val
            for stage, val in r.get("final_val_loss", {}).items():
                row[f"{stage}/final_val_loss"] = val

            records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary_table(
    df: pd.DataFrame,
    group_by: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Group by config axes and report mean ± sem for key accuracy metrics.

    Default grouping: init_style × memory_lambda.
    Default metrics: the four final accuracy columns.
    """
    if group_by is None:
        group_by = [c for c in ["init_style", "memory_lambda"] if c in df.columns]

    if metrics is None:
        metrics = [c for c in df.columns if c.startswith("after_") and "/" in c]

    if not group_by:
        return df[metrics].describe().T

    rows = []
    for keys, grp in df.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_by, keys))
        row["n"] = len(grp)
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna().values
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sem"]  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=str, default="../results/dual/vanilla/results.jsonl"
    )
    parser.add_argument(
        "--group_by", type=str, nargs="*", default=None,
        help="Config columns to group by (default: init_style memory_lambda)",
    )
    args = parser.parse_args()

    df = load_results(args.results)
    print(f"Loaded {len(df)} runs from {args.results}\n")

    tbl = summary_table(df, group_by=args.group_by)
    with pd.option_context("display.max_columns", None, "display.width", 120,
                           "display.float_format", "{:.3f}".format):
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
