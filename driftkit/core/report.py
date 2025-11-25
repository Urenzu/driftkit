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
jensen_shannon: A smoothed version of KL divergence, measuring difference between two probability distributions
(0 -> identical distributions, 1 -> completely different distributions)
"""

def format_drift_report(result: DriftResult) -> str:
    lines: List[str] = []
    for name, drift in result.columns.items():
        lines.append(f"Column: {name} ({drift.kind})")
        lines.append(f"  missing_ratio_delta: {drift.missing_ratio_delta:.4f}")
        if drift.mean_delta is not None:
            lines.append(f"  mean_delta: {drift.mean_delta:.4f}")
        if drift.stddev_delta is not None:
            lines.append(f"  stddev_delta: {drift.stddev_delta:.4f}")
        if drift.population_stability_index is not None:
            lines.append(f"  psi: {drift.population_stability_index:.4f}")
        if drift.jensen_shannon_divergence is not None:
            lines.append(f"  jensen_shannon: {drift.jensen_shannon_divergence:.4f}")
    return "\n".join(lines)
