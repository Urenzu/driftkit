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
    stats_parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Optional sampling rate (0-1] to downsample rows before computing stats.",
    )
    stats_parser.add_argument(
        "--max-categories",
        type=int,
        default=100,
        help="Cap on the number of categorical values to keep in value_counts/top_values.",
    )

    # driftkit compare <reference> <current>
    compare_parser = subparsers.add_parser("compare", help="Compare two Parquet files for drift.")
    compare_parser.add_argument("reference", help="Reference Parquet file.")
    compare_parser.add_argument("current", help="Current Parquet file.")
    compare_parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Optional sampling rate (0-1] to downsample rows before computing stats.",
    )
    compare_parser.add_argument(
        "--ks-threshold",
        type=float,
        default=0.1,
        help="Kolmogorov-Smirnov threshold used to flag numeric drift.",
    )
    compare_parser.add_argument(
        "--jsd-threshold",
        type=float,
        default=0.1,
        help="Jensen-Shannon divergence threshold used to flag categorical drift.",
    )
    compare_parser.add_argument(
        "--max-categories",
        type=int,
        default=100,
        help="Cap on the number of categorical values to keep in value_counts/top_values.",
    )

    args = parser.parse_args()

    if args.command == "stats":
        stats = compute_stats(args.path, sample_rate=args.sample_rate, max_categories=args.max_categories)
        print(_stats_to_json(stats))
    elif args.command == "compare":
        ref = compute_stats(args.reference, sample_rate=args.sample_rate, max_categories=args.max_categories)
        cur = compute_stats(args.current, sample_rate=args.sample_rate, max_categories=args.max_categories)
        drift = compare_datasets(ref, cur, ks_threshold=args.ks_threshold, jsd_threshold=args.jsd_threshold)
        print(format_drift_report(drift))


def _stats_to_json(stats) -> str:
    payload: Dict[str, Any] = {
        "row_count": stats.row_count,
        "sample_count": stats.sample_count,
        "effective_sample_rate": (stats.sample_count / stats.row_count) if stats.row_count else 0.0,
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
            "value_counts_capped": col.value_counts_capped,
            "value_counts": col.value_counts,
            "day_histogram": col.day_histogram,
            "hour_histogram": col.hour_histogram,
            "max_recency_seconds": col.max_recency_seconds,
        }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    main()
