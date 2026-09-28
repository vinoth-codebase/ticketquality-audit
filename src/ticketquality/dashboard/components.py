"""Reusable presentation helpers for the Streamlit dashboard.

Everything here is deliberately free of Streamlit imports so the same
functions can be unit-tested headlessly and reused by the CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "RISK_COLOURS",
    "risk_colour",
    "format_probability",
    "metric_cards",
    "build_triage_queue",
    "build_group_breakdown",
    "waterfall_frame",
    "summary_counts",
]

RISK_COLOURS: Dict[str, str] = {
    "LOW": "#2e7d32",
    "ELEVATED": "#ef6c00",
    "CRITICAL": "#c62828",
}


def risk_colour(level: str) -> str:
    """Hex colour for a risk band (falls back to neutral grey)."""

    return RISK_COLOURS.get(str(level).upper(), "#616161")


def format_probability(value: float) -> str:
    """Render a probability as a percentage with one decimal."""

    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def summary_counts(scored: pd.DataFrame) -> Dict[str, int]:
    """Counts per risk band plus the total."""

    if scored.empty:
        return {"total": 0, "LOW": 0, "ELEVATED": 0, "CRITICAL": 0, "flagged": 0}
    levels = scored.get("risk_level", pd.Series(dtype=str)).value_counts()
    return {
        "total": int(len(scored)),
        "LOW": int(levels.get("LOW", 0)),
        "ELEVATED": int(levels.get("ELEVATED", 0)),
        "CRITICAL": int(levels.get("CRITICAL", 0)),
        "flagged": int(scored.get("flagged_for_review", pd.Series(dtype=int)).sum()),
    }


def metric_cards(scored: pd.DataFrame, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """KPI cards for the top of the dashboard."""

    counts = summary_counts(scored)
    average = (
        float(scored["reopen_probability"].mean()) if "reopen_probability" in scored else 0.0
    )
    flagged_rate = (counts["flagged"] / counts["total"] * 100) if counts["total"] else 0.0
    return [
        {
            "label": "Tickets audited",
            "value": f"{counts['total']:,}",
            "help": "Closed tickets evaluated in this batch.",
        },
        {
            "label": "Flagged for review",
            "value": f"{counts['flagged']:,}",
            "help": f"{flagged_rate:.1f}% of the batch exceeds the "
            f"{'tuned' if threshold is None else f'{threshold:.2f}'} threshold.",
        },
        {
            "label": "Critical risk",
            "value": f"{counts['CRITICAL']:,}",
            "help": "Highest reopen probability band.",
        },
        {
            "label": "Average reopen probability",
            "value": format_probability(average),
            "help": "Mean predicted probability across the batch.",
        },
    ]


def build_triage_queue(
    scored: pd.DataFrame,
    top_n: int = 50,
    probability_column: str = "reopen_probability",
) -> pd.DataFrame:
    """Ranked review queue, highest risk first."""

    if scored.empty:
        return scored
    queue = scored.sort_values(probability_column, ascending=False).head(top_n).copy()
    if probability_column in queue.columns:
        queue["risk_pct"] = queue[probability_column].map(format_probability)
    return queue.reset_index(drop=True)


def build_group_breakdown(
    scored: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Per-group aggregation used by the team-quality charts."""

    from ..reports.scorecard import group_quality_metrics

    return group_quality_metrics(scored, group_column)


def waterfall_frame(explanation: Any) -> pd.DataFrame:
    """Turn a ticket explanation into a plot-friendly frame.

    Rows are ordered by absolute contribution, with the base value and the
    final prediction included so a waterfall can be drawn without further
    arithmetic.
    """

    drivers = getattr(explanation, "drivers", []) or []
    rows = [
        {
            "label": driver.label,
            "shap_value": float(driver.shap_value),
            "contribution_pct": float(driver.contribution_pct),
            "direction": driver.direction,
            "value": driver.value,
            "explanation": driver.sentence(),
        }
        for driver in drivers
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.reindex(
            frame["shap_value"].abs().sort_values(ascending=True).index
        ).reset_index(drop=True)
    return frame


def cumulative_contribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Cumulative running total of SHAP values, for a waterfall plot."""

    if frame.empty:
        return frame
    result = frame.copy()
    result["cumulative"] = result["shap_value"].cumsum()
    return result
