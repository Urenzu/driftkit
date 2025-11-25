from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class ColumnStats:
    name: str
    kind: str  # "numeric", "categorical", or "datetime"
    count: int  # number of sampled rows used for the stats
    missing_count: int
    distinct_count: Optional[int]
    value_counts: Optional[Dict[str, int]] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    minimum: Optional[Union[float, str]] = None
    maximum: Optional[Union[float, str]] = None
    histogram: Optional[Tuple[List[float], List[int]]] = None  # edges, counts
    top_values: List[Tuple[str, int]] = field(default_factory=list)
    value_counts_capped: bool = False
    day_histogram: Optional[Dict[str, int]] = None
    hour_histogram: Optional[Dict[str, int]] = None
    max_recency_seconds: Optional[float] = None

    @property
    def missing_ratio(self) -> float:
        if self.count == 0:
            return 0.0
        return self.missing_count / self.count


@dataclass
class DatasetStats:
    row_count: int
    sample_count: int
    columns: Dict[str, ColumnStats]


def compute_stats(
    path: str,
    *,
    histogram_bins: int = 10,
    sample_rate: float = 1.0,
    max_categories: int = 100,
) -> DatasetStats:
    """
    Compute basic stats for a Parquet file using pyarrow.

    Sampling is applied up front to the table (without replacement) so downstream
    aggregates reflect the sampled population. The effective sample size is
    returned on DatasetStats.sample_count.
    """
    table = pq.read_table(path)
    row_count = table.num_rows
    sample_count = row_count

    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("sample_rate must be in the interval (0, 1].")

    if 0 < sample_rate < 1 and row_count > 0:
        sample_count = max(1, int(row_count * sample_rate))
        if sample_count < row_count:
            indices = np.random.default_rng().choice(row_count, size=sample_count, replace=False)
            table = table.take(pa.array(indices))
    columns: Dict[str, ColumnStats] = {}

    for field in table.schema:
        name = field.name
        column = table[name]
        kind = _kind_from_type(field.type)

        non_missing = int(pc.count(column).as_py())
        missing_count = sample_count - non_missing
        distinct_count = int(pc.count_distinct(column).as_py())

        if kind == "numeric":
            stats = _numeric_stats(column, name, sample_count, missing_count, distinct_count, histogram_bins)
        elif kind == "datetime":
            stats = _datetime_stats(column, name, sample_count, missing_count, distinct_count, histogram_bins)
        else:
            stats = _categorical_stats(column, name, sample_count, missing_count, distinct_count, max_categories)

        columns[name] = stats

    return DatasetStats(row_count=row_count, sample_count=sample_count, columns=columns)


def _kind_from_type(pa_type: pa.DataType) -> str:
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        return "numeric"
    if pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type):
        return "datetime"
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
    column: pa.ChunkedArray,
    name: str,
    row_count: int,
    missing_count: int,
    distinct_count: int,
    max_categories: int,
) -> ColumnStats:
    counts = pc.value_counts(column)
    counts_list = counts.to_pylist()
    value_counts: Dict[str, int] = {}
    top_values: List[Tuple[str, int]] = []
    for idx, entry in enumerate(counts_list):
        val = entry["values"]
        cnt = int(entry["counts"])
        value_counts[str(val)] = cnt
        if idx < max_categories:
            top_values.append((str(val), cnt))

    value_counts_capped = len(value_counts) > max_categories
    return ColumnStats(
        name=name,
        kind="categorical",
        count=row_count,
        missing_count=missing_count,
        distinct_count=distinct_count,
        value_counts=value_counts if not value_counts_capped else dict(top_values),
        top_values=top_values,
        value_counts_capped=value_counts_capped,
    )


def _build_histogram(column: pa.ChunkedArray, bins: int) -> Optional[Tuple[List[float], List[int]]]:
    if bins <= 0:
        return None

    arr = _non_null_numpy(column)
    numeric_values = np.asarray(arr, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if len(numeric_values) == 0:
        return None

    counts, edges = np.histogram(numeric_values, bins=bins)
    return edges.tolist(), counts.tolist()


def _datetime_stats(
    column: pa.ChunkedArray,
    name: str,
    row_count: int,
    missing_count: int,
    distinct_count: int,
    histogram_bins: int,
) -> ColumnStats:
    non_missing = row_count - missing_count
    day_histogram: Optional[Dict[str, int]] = None
    hour_histogram: Optional[Dict[str, int]] = None
    max_recency_seconds: Optional[float] = None
    maximum = None
    minimum = None
    histogram = None

    if non_missing > 0:
        min_max = pc.min_max(column).as_py()
        minimum = _datetime_to_iso(min_max["min"])
        maximum_py = min_max["max"]
        maximum = _datetime_to_iso(maximum_py)
        if maximum_py is not None:
            max_dt = _as_datetime(maximum_py)
            if max_dt is not None:
                max_recency_seconds = (datetime.now(timezone.utc) - max_dt).total_seconds()

        histogram = _build_datetime_histogram(column, histogram_bins)
        day_histogram = _build_day_histogram(column)
        hour_histogram = _build_hour_histogram(column)

    return ColumnStats(
        name=name,
        kind="datetime",
        count=row_count,
        missing_count=missing_count,
        distinct_count=distinct_count,
        minimum=minimum,
        maximum=maximum,
        histogram=histogram,
        day_histogram=day_histogram,
        hour_histogram=hour_histogram,
        max_recency_seconds=max_recency_seconds,
    )


def _build_datetime_histogram(
    column: pa.ChunkedArray, bins: int
) -> Optional[Tuple[List[float], List[int]]]:
    if bins <= 0:
        return None

    np_values = _non_null_numpy(column)
    if np_values.size == 0:
        return None

    as_float = np.array(np_values.astype("datetime64[ns]").astype(np.float64))
    as_float = as_float[np.isfinite(as_float)]
    if len(as_float) == 0:
        return None

    counts, edges = np.histogram(as_float, bins=bins)
    return edges.tolist(), counts.tolist()


def _build_day_histogram(column: pa.ChunkedArray) -> Optional[Dict[str, int]]:
    np_values = _non_null_numpy(column)
    if np_values.size == 0:
        return None

    days = np_values.astype("datetime64[D]").astype(np.int64)
    # Monday=0 ... Sunday=6
    day_of_week = (days + 3) % 7
    counts = np.bincount(day_of_week, minlength=7)
    return {str(idx): int(cnt) for idx, cnt in enumerate(counts)}


def _build_hour_histogram(column: pa.ChunkedArray) -> Optional[Dict[str, int]]:
    if not pa.types.is_timestamp(column.type):
        return None

    np_values = _non_null_numpy(column)
    if np_values.size == 0:
        return None

    hours = np_values.astype("datetime64[h]").astype(np.int64) % 24
    counts = np.bincount(hours, minlength=24)
    return {str(idx): int(cnt) for idx, cnt in enumerate(counts)}


def _datetime_to_iso(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
    except Exception:
        return None
    return None


def _as_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
            return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def _non_null_numpy(column: pa.ChunkedArray) -> np.ndarray:
    cleaned = pc.drop_null(column)
    if hasattr(cleaned, "combine_chunks"):
        cleaned = cleaned.combine_chunks()
    return cleaned.to_numpy(zero_copy_only=False)
