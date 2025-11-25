from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .stats_api import ColumnStats, DatasetStats


@dataclass
class ColumnDrift:
    name: str
    kind: str
    missing_ratio_delta: float
    mean_delta: Optional[float] = None
    stddev_delta: Optional[float] = None
    population_stability_index: Optional[float] = None
    jensen_shannon_divergence: Optional[float] = None


@dataclass
class DriftResult:
    columns: Dict[str, ColumnDrift]


def compare_datasets(reference: DatasetStats, current: DatasetStats) -> DriftResult:
    shared = set(reference.columns.keys()) & set(current.columns.keys())
    results: Dict[str, ColumnDrift] = {}

    for name in sorted(shared):
        ref_col = reference.columns[name]
        cur_col = current.columns[name]
        if ref_col.kind == "numeric" and cur_col.kind == "numeric":
            results[name] = _compare_numeric(ref_col, cur_col)
        elif ref_col.kind == "categorical" and cur_col.kind == "categorical":
            results[name] = _compare_categorical(ref_col, cur_col)
        else:
            # Skip mismatched kinds for now.
            continue

    return DriftResult(columns=results)


def _compare_numeric(reference: ColumnStats, current: ColumnStats) -> ColumnDrift:
    missing_delta = current.missing_ratio - reference.missing_ratio
    mean_delta = _delta(reference.mean, current.mean)
    std_delta = _delta(reference.stddev, current.stddev)
    psi = _population_stability_index(reference, current)
    return ColumnDrift(
        name=reference.name,
        kind="numeric",
        missing_ratio_delta=missing_delta,
        mean_delta=mean_delta,
        stddev_delta=std_delta,
        population_stability_index=psi,
    )


def _compare_categorical(reference: ColumnStats, current: ColumnStats) -> ColumnDrift:
    missing_delta = current.missing_ratio - reference.missing_ratio
    jsd = _jensen_shannon(reference.value_counts, current.value_counts)
    return ColumnDrift(
        name=reference.name,
        kind="categorical",
        missing_ratio_delta=missing_delta,
        jensen_shannon_divergence=jsd,
    )


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def _population_stability_index(reference: ColumnStats, current: ColumnStats) -> Optional[float]:
    if not reference.histogram or not current.histogram:
        return None
    ref_edges, ref_counts = reference.histogram
    cur_edges, cur_counts = current.histogram
    if len(ref_counts) == 0 or len(ref_counts) != len(cur_counts):
        return None

    ref_total = sum(ref_counts)
    cur_total = sum(cur_counts)
    if ref_total == 0 or cur_total == 0:
        return None

    epsilon = 1e-12
    psi = 0.0
    for i in range(len(ref_counts)):
        ref_frac = max(ref_counts[i] / ref_total, epsilon)
        cur_frac = max(cur_counts[i] / cur_total, epsilon)
        psi += (ref_frac - cur_frac) * math.log(ref_frac / cur_frac)
    return psi


def _jensen_shannon(
    ref_counts: Optional[Dict[str, int]], cur_counts: Optional[Dict[str, int]]
) -> Optional[float]:
    if not ref_counts or not cur_counts:
        return None
    ref_total = sum(ref_counts.values())
    cur_total = sum(cur_counts.values())
    if ref_total == 0 or cur_total == 0:
        return None

    keys = set(ref_counts.keys()) | set(cur_counts.keys())
    ref_dist = {k: ref_counts.get(k, 0) / ref_total for k in keys}
    cur_dist = {k: cur_counts.get(k, 0) / cur_total for k in keys}
    m = {k: 0.5 * (ref_dist[k] + cur_dist[k]) for k in keys}

    return 0.5 * (_kl_divergence(ref_dist, m) + _kl_divergence(cur_dist, m))


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    epsilon = 1e-12
    divergence = 0.0
    for k, p_val in p.items():
        p_val = max(p_val, epsilon)
        q_val = max(q.get(k, 0.0), epsilon)
        divergence += p_val * math.log(p_val / q_val, 2)
    return divergence
