"""Export audit findings to machine- and human-readable formats.

Three artefacts come out of a batch audit:

* ``report.json`` — the full scorecard plus model provenance, for CI
  artefacts and downstream tooling.
* ``triage_queue.csv`` — the ranked review queue a support lead works
  through top to bottom.
* ``report.md`` — the same scorecard as Markdown, for pasting into a
  ticket or a wiki.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .scorecard import AuditScorecard, build_scorecard

__all__ = [
    "json_default",
    "export_json",
    "export_markdown",
    "export_triage_queue",
    "build_audit_report",
    "save_audit_report",
]


def json_default(value: Any) -> Any:
    """JSON encoder fallback for numpy / pandas / datetime scalars."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def export_json(payload: Dict[str, Any], path: Path) -> Path:
    """Write ``payload`` as pretty-printed JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default, sort_keys=False)
    return destination


def export_markdown(scorecard: AuditScorecard, path: Path) -> Path:
    """Write a scorecard as Markdown."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(scorecard.to_markdown(), encoding="utf-8")
    return destination


def export_triage_queue(
    scored: pd.DataFrame,
    path: Path,
    probability_column: str = "reopen_probability",
    top_n: Optional[int] = None,
) -> Path:
    """Write the ranked review queue as CSV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    queue = scored.sort_values(probability_column, ascending=False)
    if top_n:
        queue = queue.head(top_n)

    preferred = [
        c
        for c in (
            "ticket_id",
            probability_column,
            "risk_level",
            "flagged_for_review",
            "reassignment_count",
            "resolution_minutes",
            "priority",
            "category",
            "assignment_group",
            "shift",
            "is_premature_closure",
        )
        if c in queue.columns
    ]
    remainder = [c for c in queue.columns if c not in preferred]
    queue = queue[preferred + remainder]
    queue.to_csv(destination, index=False)
    return destination


def build_audit_report(
    scored: pd.DataFrame,
    model_metrics: Optional[Dict[str, Any]] = None,
    explanations: Optional[List[Any]] = None,
    threshold: Optional[float] = None,
    model_backend: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the complete audit report payload."""

    scorecard = build_scorecard(scored, threshold=threshold)
    payload: Dict[str, Any] = {
        "report_type": "premature_closure_audit",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": source,
        "model": {
            "backend": model_backend,
            "operating_threshold": round(float(threshold), 6)
            if threshold is not None
            else None,
            "metrics": model_metrics or {},
        },
        "scorecard": scorecard.to_dict(),
    }

    if explanations:
        payload["top_risk_explanations"] = [
            explanation.to_dict()
            if hasattr(explanation, "to_dict")
            else dict(explanation)
            for explanation in explanations[:10]
        ]

    if "ticket_id" in scored.columns:
        ranked = scored.sort_values("reopen_probability", ascending=False)
        payload["triage_queue"] = ranked.head(25).to_dict(orient="records")

    return payload


def save_audit_report(
    payload: Dict[str, Any],
    path: Path,
) -> Path:
    """Persist an audit report, choosing the format from the extension."""

    destination = Path(path)
    if destination.suffix.lower() == ".md":
        scorecard = AuditScorecard(
            summary=payload.get("scorecard", {}).get("summary", {}),
            recommendations=payload.get("scorecard", {}).get("recommendations", []),
        )
        for key, rows in payload.get("scorecard", {}).get("breakdowns", {}).items():
            scorecard.by_group[key] = pd.DataFrame(rows)
        return export_markdown(scorecard, destination)
    return export_json(payload, destination)
