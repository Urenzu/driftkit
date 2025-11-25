from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class ColumnStats:
    name: str
    kind: str  # "numeric" or "categorical"
    count: int
    missing_count: int
    distinct_count: Optional[int]
    value_counts: Optional[Dict[str, int]] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    histogram: Optional[Tuple[List[float], List[int]]] = None  # edges, counts
    top_values: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def missing_ratio(self) -> float:
        if self.count == 0:
            return 0.0
        return self.missing_count / self.count


@dataclass
class DatasetStats:
    row_count: int
    columns: Dict[str, ColumnStats]


def compute_stats(
    path: str,
    *,
    histogram_bins: int = 10,
) -> DatasetStats:
    """
    Compute basic stats for a Parquet file using pyarrow.
    """
    table = pq.read_table(path)
    row_count = table.num_rows
    columns: Dict[str, ColumnStats] = {}

    for field in table.schema:
        name = field.name
        column = table[name]
        kind = _kind_from_type(field.type)

        non_missing = int(pc.count(column).as_py())
        missing_count = row_count - non_missing
        distinct_count = int(pc.count_distinct(column).as_py())

        if kind == "numeric":
            stats = _numeric_stats(column, name, row_count, missing_count, distinct_count, histogram_bins)
        else:
            stats = _categorical_stats(column, name, row_count, missing_count, distinct_count)

        columns[name] = stats

    return DatasetStats(row_count=row_count, columns=columns)


def _kind_from_type(pa_type: pa.DataType) -> str:
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        return "numeric"
    return "categorical"


def _numeric_stats(
    column: pa.ChunkedArray,
    name: str,
    row_count: int,
    missing_count: int,
    distinct_count: int,
    histogram_bins: int,
) -> ColumnStats:
    non_missing = row_count - missing_count
    mean = stddev = minimum = maximum = None
    histogram = None

    if non_missing > 0:
        mean = pc.mean(column).as_py()
        variance = pc.variance(column, ddof=0).as_py()
        variance = variance if variance is not None else 0.0
        stddev = math.sqrt(max(variance, 0.0))
        min_max = pc.min_max(column).as_py()
        minimum = min_max["min"]
        maximum = min_max["max"]
        histogram = _build_histogram(column, histogram_bins)

    return ColumnStats(
        name=name,
        kind="numeric",
        count=row_count,
        missing_count=missing_count,
        distinct_count=distinct_count,
        mean=mean,
        stddev=stddev,
        minimum=minimum,
        maximum=maximum,
        histogram=histogram,
    )


def _categorical_stats(
    column: pa.ChunkedArray, name: str, row_count: int, missing_count: int, distinct_count: int
) -> ColumnStats:
    counts = pc.value_counts(column)
    counts_list = counts.to_pylist()
    value_counts: Dict[str, int] = {}
    top_values: List[Tuple[str, int]] = []
    for idx, entry in enumerate(counts_list):
        val = entry["values"]
        cnt = int(entry["counts"])
        value_counts[str(val)] = cnt
        if idx < 10:
            top_values.append((str(val), cnt))

    return ColumnStats(
        name=name,
        kind="categorical",
        count=row_count,
        missing_count=missing_count,
        distinct_count=distinct_count,
        value_counts=value_counts,
        top_values=top_values,
    )


def _build_histogram(column: pa.ChunkedArray, bins: int) -> Optional[Tuple[List[float], List[int]]]:
    if bins <= 0:
        return None

    non_null = pc.drop_null(column)
    arr = non_null.to_numpy(zero_copy_only=False)
    numeric_values = np.asarray(arr, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if len(numeric_values) == 0:
        return None

    counts, edges = np.histogram(numeric_values, bins=bins)
    return edges.tolist(), counts.tolist()
