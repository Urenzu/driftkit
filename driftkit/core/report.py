from __future__ import annotations

from typing import List

from .drift_api import ColumnDrift, DriftResult

"""
Column: data type

missing_ratio_delta: How much the missing data has changed (more or less)

mean_delta: Change in mean value

stddev_delta: Change in standard deviation

psi: Directional shift from one distribution to another (reference -> current), specifically the distribution of a variable
(PSI >= 0.25 is significant drift, 0.1 <= PSI < 0.25 is moderate drift, PSI < 0.1 is minor drift)

kolmogorov_smirnov {numeric drift}: Maximum distance between the empirical cumulative distribution functions of two samples, maximum vertical distance

jensen_shannon {categorical drift}: A smoothed version of KL divergence, measuring difference between two probability distributions
(0 -> identical distributions, 1 -> completely different distributions)

"""


def format_drift_report(result: DriftResult) -> str:
    lines: List[str] = []
    lines.append(
        f"reference rows: {result.reference_sample_rows}/{result.reference_rows}, "
        f"current rows: {result.current_sample_rows}/{result.current_rows}"
    )
    if result.added_columns or result.removed_columns or result.type_changes:
        lines.append("Schema changes:")
        if result.added_columns:
            lines.append(f"  added: {', '.join(result.added_columns)}")
        if result.removed_columns:
            lines.append(f"  removed: {', '.join(result.removed_columns)}")
        for change in result.type_changes:
            lines.append(f"  type changed: {change.name} ({change.reference_kind} -> {change.current_kind})")

    for name, drift in result.columns.items():
        lines.append(f"Column: {name} ({drift.kind})")
        lines.append(f"  severity: {drift.severity}")
        lines.append(f"  missing_ratio_delta: {drift.missing_ratio_delta:.4f}")
        if drift.mean_delta is not None:
            lines.append(f"  mean_delta: {drift.mean_delta:.4f}")
        if drift.stddev_delta is not None:
            lines.append(f"  stddev_delta: {drift.stddev_delta:.4f}")
        if drift.population_stability_index is not None:
            lines.append(f"  psi: {drift.population_stability_index:.4f}")
        if drift.kolmogorov_smirnov is not None:
            lines.append(f"  ks: {drift.kolmogorov_smirnov:.4f} (threshold={drift.threshold})")
        if drift.jensen_shannon_divergence is not None:
            lines.append(f"  jensen_shannon: {drift.jensen_shannon_divergence:.4f} (threshold={drift.threshold})")
    return "\n".join(lines)
