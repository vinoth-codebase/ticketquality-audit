"""Team-level QA scorecards.

Individual risk scores tell an agent what to re-check. A scorecard tells a
support manager *where the process is leaking*. This module aggregates a
batch of scored tickets into the views a QA lead actually reviews:

* an overall audit summary (volume, at-risk rate, average risk);
* premature-closure rates sliced by assignment group, category and shift;
* a ranked list of groups that most need intervention, weighted by
  volume so a 4-ticket queue with one reopen does not outrank a
  400-ticket queue with forty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "AuditScorecard",
    "build_scorecard",
    "group_quality_metrics",
    "rank_groups_by_risk",
    "DEFAULT_GROUPINGS",
]

DEFAULT_GROUPINGS: Sequence[str] = ("assignment_group", "category", "shift")


@dataclass
class AuditScorecard:
    """Aggregated audit findings for a batch of tickets."""

    summary: Dict[str, Any] = field(default_factory=dict)
    by_group: Dict[str, pd.DataFrame] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "summary": self.summary,
            "recommendations": self.recommendations,
            "breakdowns": {},
        }
        for key, frame in self.by_group.items():
            payload["breakdowns"][key] = frame.to_dict(orient="records")
        return payload

    def to_markdown(self) -> str:
        lines = ["# Support QA Audit Scorecard", ""]
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for key, value in self.summary.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
        lines.append("")
        for key, frame in self.by_group.items():
            lines.append(f"## By {key.replace('_', ' ').title()}")
            lines.append("")
            lines.append(frame.to_markdown(index=False))
            lines.append("")
        if self.recommendations:
            lines.append("## Recommended QA Actions")
            lines.append("")
            for item in self.recommendations:
                lines.append(f"- {item}")
            lines.append("")
        return "\n".join(lines)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def group_quality_metrics(
    scored: pd.DataFrame,
    group_column: str,
    probability_column: str = "reopen_probability",
    risk_column: str = "risk_level",
    actual_column: Optional[str] = "is_premature_closure",
    min_volume: int = 1,
) -> pd.DataFrame:
    """Aggregate quality metrics for one grouping column."""

    if group_column not in scored.columns:
        return pd.DataFrame(
            columns=[
                group_column,
                "tickets",
                "flagged_for_review",
                "flagged_rate",
                "avg_reopen_probability",
                "critical_tickets",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for name, block in scored.groupby(group_column, dropna=False):
        volume = len(block)
        if volume < min_volume:
            continue
        flagged = int(block.get("flagged_for_review", pd.Series(dtype=int)).sum())
        row: Dict[str, Any] = {
            group_column: str(name),
            "tickets": int(volume),
            "flagged_for_review": flagged,
            "flagged_rate": round(_safe_rate(flagged, volume), 4),
            "avg_reopen_probability": round(
                float(block[probability_column].mean()), 4
            ),
            "critical_tickets": int((block[risk_column] == "CRITICAL").sum())
            if risk_column in block.columns
            else 0,
        }
        if actual_column and actual_column in block.columns:
            observed = float(pd.to_numeric(block[actual_column], errors="coerce").mean())
            row["observed_reopen_rate"] = round(observed, 4)
            row["reopened_tickets"] = int(
                pd.to_numeric(block[actual_column], errors="coerce").fillna(0).sum()
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["flagged_rate", "tickets"], ascending=[False, False]
        ).reset_index(drop=True)
    return frame


def rank_groups_by_risk(
    scored: pd.DataFrame,
    group_column: str = "assignment_group",
    min_volume: int = 5,
) -> pd.DataFrame:
    """Rank groups by a volume-weighted risk score.

    ``risk_score = flagged_rate * log1p(tickets)`` — a group that flags
    often *and* handles real volume rises to the top, while a tiny queue
    with a single flag does not.
    """

    metrics = group_quality_metrics(scored, group_column, min_volume=min_volume)
    if metrics.empty:
        return metrics
    metrics["risk_score"] = (
        metrics["flagged_rate"] * np.log1p(metrics["tickets"])
    ).round(4)
    return metrics.sort_values("risk_score", ascending=False).reset_index(drop=True)


def build_scorecard(
    scored: pd.DataFrame,
    groupings: Sequence[str] = DEFAULT_GROUPINGS,
    threshold: Optional[float] = None,
    probability_column: str = "reopen_probability",
    actual_column: Optional[str] = "is_premature_closure",
) -> AuditScorecard:
    """Build a full :class:`AuditScorecard` from a scored ticket frame."""

    if scored.empty:
        return AuditScorecard(
            summary={"tickets_audited": 0},
            recommendations=["No tickets supplied for audit."],
        )

    total = len(scored)
    flagged = int(scored.get("flagged_for_review", pd.Series(dtype=int)).sum())
    probabilities = scored[probability_column]

    summary: Dict[str, Any] = {
        "tickets_audited": int(total),
        "flagged_for_review": flagged,
        "flagged_rate_pct": round(_safe_rate(flagged, total) * 100, 2),
        "average_reopen_probability": round(float(probabilities.mean()), 4),
        "median_reopen_probability": round(float(probabilities.median()), 4),
        "critical_risk_tickets": int((scored.get("risk_level") == "CRITICAL").sum())
        if "risk_level" in scored.columns
        else 0,
        "elevated_risk_tickets": int((scored.get("risk_level") == "ELEVATED").sum())
        if "risk_level" in scored.columns
        else 0,
        "operating_threshold": round(float(threshold), 4) if threshold is not None else None,
    }

    if actual_column and actual_column in scored.columns:
        actual = pd.to_numeric(scored[actual_column], errors="coerce").fillna(0)
        summary["observed_reopen_rate_pct"] = round(float(actual.mean()) * 100, 2)
        summary["observed_reopens"] = int(actual.sum())

    by_group: Dict[str, pd.DataFrame] = {}
    for column in groupings:
        if column in scored.columns:
            by_group[column] = group_quality_metrics(scored, column, actual_column=actual_column)

    recommendations = _recommendations(scored, summary, by_group)

    return AuditScorecard(summary=summary, by_group=by_group, recommendations=recommendations)


def _recommendations(
    scored: pd.DataFrame,
    summary: Dict[str, Any],
    by_group: Dict[str, pd.DataFrame],
) -> List[str]:
    """Derive plain-English QA actions from the scorecard."""

    notes: List[str] = []
    rate = summary.get("flagged_rate_pct", 0.0)
    notes.append(
        f"{summary['flagged_for_review']} of {summary['tickets_audited']} closed "
        f"tickets ({rate:.1f}%) are flagged for peer review before customer "
        f"notification."
    )

    if summary.get("critical_risk_tickets"):
        notes.append(
            f"{summary['critical_risk_tickets']} tickets sit in the CRITICAL band; "
            f"prioritise these for same-day QA sampling."
        )

    group_frame = by_group.get("assignment_group")
    if group_frame is not None and not group_frame.empty:
        worst = group_frame.iloc[0]
        notes.append(
            f"Assignment group '{worst['assignment_group']}' shows the highest "
            f"flagged rate ({worst['flagged_rate'] * 100:.1f}% of "
            f"{worst['tickets']} tickets); schedule a closure-quality review."
        )

    category_frame = by_group.get("category")
    if category_frame is not None and not category_frame.empty:
        worst_cat = category_frame.iloc[0]
        notes.append(
            f"Category '{worst_cat['category']}' carries the greatest reopen risk "
            f"(avg probability {worst_cat['avg_reopen_probability']:.2f}); check "
            f"for triage or knowledge-base gaps."
        )

    shift_frame = by_group.get("shift")
    if shift_frame is not None and len(shift_frame) > 1:
        worst_shift = shift_frame.iloc[0]
        notes.append(
            f"The {worst_shift['shift']} shift records the highest flagged rate "
            f"({worst_shift['flagged_rate'] * 100:.1f}%); consider coverage or "
            f"handover audits."
        )

    return notes
