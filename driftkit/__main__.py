from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from . import compare_datasets, compute_stats, format_drift_report


def main() -> None:
    # Create the top-level parser
    parser = argparse.ArgumentParser(prog="driftkit", description="Lightweight Parquet-first drift tracker (v0.1).")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # driftkit stats <path>
    stats_parser = subparsers.add_parser("stats", help="Compute stats for a Parquet file.")
    stats_parser.add_argument("path", help="Path to Parquet file.")

    # driftkit compare <reference> <current>
    compare_parser = subparsers.add_parser("compare", help="Compare two Parquet files for drift.")
    compare_parser.add_argument("reference", help="Reference Parquet file.")
    compare_parser.add_argument("current", help="Current Parquet file.")

    args = parser.parse_args()

    if args.command == "stats":
        stats = compute_stats(args.path)
        print(_stats_to_json(stats))
    elif args.command == "compare":
        ref = compute_stats(args.reference)
        cur = compute_stats(args.current)
        drift = compare_datasets(ref, cur)
        print(format_drift_report(drift))


def _stats_to_json(stats) -> str:
    payload: Dict[str, Any] = {
        "row_count": stats.row_count,
        "columns": {},
    }
    for name, col in stats.columns.items():
        payload["columns"][name] = {
            "kind": col.kind,
            "count": col.count,
            "missing_count": col.missing_count,
            "missing_ratio": col.missing_ratio,
            "distinct_count": col.distinct_count,
            "mean": col.mean,
            "stddev": col.stddev,
            "minimum": col.minimum,
            "maximum": col.maximum,
            "top_values": col.top_values,
        }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    main()
