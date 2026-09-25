"""Ticket-level SHAP explainability.

A risk score a support agent cannot interrogate is a risk score they will
not act on. This module turns the booster's output into a ranked list of
plain-English risk drivers, e.g.::

    Reassignment count of 4 contributed +28% to reopen risk
    Resolution time of 12 min contributed +21% to reopen risk
    Caller historical reopen rate of 0.62 contributed +11% to reopen risk

Attribution uses :class:`shap.TreeExplainer`, which computes *exact*
Shapley values for tree ensembles in polynomial time — no sampling
approximation. The values are additive: they sum to (model output −
expected output), so a ticket's total risk decomposes cleanly into the
contributions listed.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guarded for minimal environments
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None  # type: ignore
    SHAP_AVAILABLE = False

__all__ = [
    "RiskDriver",
    "TicketExplanation",
    "TicketExplainer",
    "SHAP_AVAILABLE",
    "humanise_feature_name",
]

# Pretty labels for the raw column names that reach the design matrix.
_FEATURE_LABELS: Dict[str, str] = {
    "reassignment_count": "Reassignment count",
    "reassignment_rate": "Reassignment rate (handoffs/hour)",
    "modification_density": "Modification density (updates/hour)",
    "sys_mod_count": "System modification count",
    "interaction_count": "Interaction count",
    "caller_reopen_rate": "Caller historical reopen rate",
    "priority_mismatch_index": "Priority mismatch index",
    "assignment_group_dwell": "Assignment group dwell (hours)",
    "resolution_minutes": "Resolution time (minutes)",
    "resolution_hours": "Resolution time (hours)",
    "active_minutes": "Active resolution time (minutes)",
    "active_hours": "Active resolution time (hours)",
    "handle_time": "Handle time (minutes)",
    "handoff_intensity": "Handoff intensity",
    "instant_close_flag": "Instant close flag",
    "speed_per_handoff": "Speed per handoff",
    "combined_failure_mode": "Quick-close x handoff interaction",
    "priority": "Priority",
    "impact": "Impact",
    "urgency": "Urgency",
    "category": "Category",
    "assignment_group": "Assignment group",
    "contact_type": "Contact channel",
    "shift": "Support shift",
    "made_sla": "SLA met",
}


def humanise_feature_name(name: str) -> str:
    """Turn a design-matrix column name into a readable label.

    Handles one-hot expansions such as ``priority_High`` -> ``Priority: High``.
    """

    base, _, category = name.partition("_")
    # one-hot column: <feature>_<category>, but only when the base matches
    # a known categorical feature.
    if name in _FEATURE_LABELS:
        return _FEATURE_LABELS[name]
    for feature in ("priority", "impact", "urgency", "category", "assignment_group", "contact_type", "shift", "made_sla"):
        prefix = f"{feature}_"
        if name.startswith(prefix):
            value = name[len(prefix):]
            label = _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
            return f"{label}: {value}"
    # quantile / bounded prefixed names from ColumnTransformer
    for prefix in ("skewed__", "bounded__", "categorical__"):
        if name.startswith(prefix):
            return humanise_feature_name(name[len(prefix):])
    return _FEATURE_LABELS.get(name, name.replace("_", " ").strip().title())


@dataclass
class RiskDriver:
    """A single feature's contribution to one ticket's reopen risk."""

    feature: str
    label: str
    value: Any
    shap_value: float
    contribution_pct: float
    direction: str  # "increases" | "decreases"

    def sentence(self) -> str:
        """A plain-English statement of this driver."""

        verb = "increased" if self.direction == "increases" else "reduced"
        magnitude = abs(self.contribution_pct)
        value_text = self._format_value()
        return (
            f"{self.label} of {value_text} {verb} reopen risk "
            f"by {magnitude:.1f}%"
        )

    def _format_value(self) -> str:
        if isinstance(self.value, (int, np.integer)):
            return str(int(self.value))
        if isinstance(self.value, (float, np.floating)):
            if abs(self.value) >= 100:
                return f"{self.value:,.0f}"
            return f"{self.value:,.2f}"
        return str(self.value)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["explanation"] = self.sentence()
        return payload


@dataclass
class TicketExplanation:
    """The full explanation for one ticket."""

    ticket_id: str
    reopen_probability: float
    risk_level: str
    base_value: float
    raw_log_odds: float
    drivers: List[RiskDriver] = field(default_factory=list)

    def top_drivers(self, n: int = 5) -> List[RiskDriver]:
        return self.drivers[:n]

    def summary(self) -> str:
        """A one-paragraph narrative suitable for the CLI / report."""

        if not self.drivers:
            return (
                f"Ticket {self.ticket_id}: reopen probability "
                f"{self.reopen_probability:.1%} ({self.risk_level}); no dominant drivers."
            )
        lines = [
            f"Ticket {self.ticket_id} carries a {self.reopen_probability:.1%} "
            f"reopen probability ({self.risk_level}). Principal drivers:"
        ]
        for driver in self.top_drivers(3):
            lines.append(f"  - {driver.sentence()}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "reopen_probability": round(self.reopen_probability, 6),
            "risk_level": self.risk_level,
            "base_value": round(self.base_value, 6),
            "raw_log_odds": round(self.raw_log_odds, 6),
            "drivers": [d.to_dict() for d in self.drivers],
        }


class TicketExplainer:
    """Compute and translate SHAP attributions for a fitted audit model."""

    def __init__(self, model: Any, max_drivers: int = 8):
        """``model`` is a :class:`~ticketquality.models.classifier.TicketAuditModel`."""

        self.model = model
        self.max_drivers = max_drivers
        self._explainer: Any = None
        self._feature_names: List[str] = list(model.pipeline.feature_names)

    # ------------------------------------------------------------------ #
    def _ensure_explainer(self) -> Any:
        if self._explainer is not None:
            return self._explainer
        if not SHAP_AVAILABLE:
            raise RuntimeError(
                "the `shap` package is required for explanations; "
                "install it with `pip install shap`"
            )
        booster = self.model.classifier.underlying_booster()
        if booster is None:
            raise RuntimeError("the audit model must be fitted before explaining tickets")
        try:
            self._explainer = shap.TreeExplainer(booster)
        except Exception as exc:  # pragma: no cover - model-type dependent
            raise RuntimeError(f"could not build a TreeExplainer: {exc}") from exc
        return self._explainer

    # ------------------------------------------------------------------ #
    def _positive_class_values(self, X: np.ndarray) -> np.ndarray:
        """Return an ``(n, features)`` matrix of SHAP values for class 1."""

        explainer = self._ensure_explainer()
        with warnings.catch_warnings():
            # SHAP emits a benign notice about LightGBM binary-classifier
            # output shape; we handle both shapes below.
            warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
            values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values, dtype=float)
        if values.ndim == 3:
            # (n, features, classes)
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
        return values

    def _base_value(self) -> float:
        explainer = self._ensure_explainer()
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            arr = np.asarray(base, dtype=float).ravel()
            return float(arr[1]) if arr.size > 1 else float(arr[0])
        return float(base)

    # ------------------------------------------------------------------ #
    def explain_row(
        self,
        row: pd.Series,
        design_row: np.ndarray,
        probability: float,
        risk_band: str,
        ticket_id: Optional[str] = None,
    ) -> TicketExplanation:
        """Explain a single ticket given its raw row and design vector."""

        values = self._positive_class_values(design_row.reshape(1, -1))[0]
        names = self._feature_names
        if len(names) != len(values):
            names = [f"f{i}" for i in range(len(values))]

        total_abs = float(np.abs(values).sum()) or 1.0
        drivers: List[RiskDriver] = []
        for name, shap_value in zip(names, values):
            raw_name = name
            for prefix in ("skewed__", "bounded__", "categorical__"):
                if raw_name.startswith(prefix):
                    raw_name = raw_name[len(prefix):]
            label = humanise_feature_name(name)
            value = self._lookup_value(row, raw_name, name)
            contribution = float(shap_value) / total_abs * 100.0
            drivers.append(
                RiskDriver(
                    feature=raw_name,
                    label=label,
                    value=value,
                    shap_value=float(shap_value),
                    contribution_pct=round(contribution, 3),
                    direction="increases" if shap_value >= 0 else "decreases",
                )
            )

        drivers.sort(key=lambda d: abs(d.shap_value), reverse=True)
        drivers = drivers[: self.max_drivers]

        return TicketExplanation(
            ticket_id=str(ticket_id if ticket_id is not None else row.get("ticket_id", "UNKNOWN")),
            reopen_probability=float(probability),
            risk_level=str(risk_band),
            base_value=self._base_value(),
            raw_log_odds=float(self._base_value() + values.sum()),
            drivers=drivers,
        )

    @staticmethod
    def _lookup_value(row: pd.Series, raw_name: str, matrix_name: str) -> Any:
        if raw_name in row.index:
            return row[raw_name]
        # one-hot column: fall back to the category token
        for feature in (
            "priority",
            "impact",
            "urgency",
            "category",
            "assignment_group",
            "contact_type",
            "shift",
            "made_sla",
        ):
            prefix = f"{feature}_"
            if matrix_name.startswith(prefix):
                return matrix_name[len(prefix):]
        return "n/a"

    # ------------------------------------------------------------------ #
    def explain_frame(self, frame: pd.DataFrame) -> List[TicketExplanation]:
        """Explain every row of ``frame`` (already scored or not)."""

        scored = self.model.score_frame(frame)
        # Value lookup needs the *engineered* row (speed_per_handoff and
        # friends do not exist on the raw frame).
        engineered = self.model.pipeline.engineer.transform(frame)
        X = self.model.design_matrix(frame)
        explanations: List[TicketExplanation] = []
        for position, (_, row) in enumerate(scored.iterrows()):
            explanations.append(
                self.explain_row(
                    engineered.iloc[position],
                    X[position],
                    float(row["reopen_probability"]),
                    str(row["risk_level"]),
                    ticket_id=row.get("ticket_id"),
                )
            )
        return explanations

    def explain_ticket(self, frame: pd.DataFrame, ticket_id: str) -> TicketExplanation:
        """Locate a ticket by id and explain it."""

        if "ticket_id" not in frame.columns:
            raise KeyError("frame has no ticket_id column")
        matches = frame[frame["ticket_id"].astype(str) == str(ticket_id)]
        if matches.empty:
            raise KeyError(f"ticket {ticket_id!r} not found in frame")
        return self.explain_frame(matches)[0]

    # ------------------------------------------------------------------ #
    def global_importance(self, frame: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
        """Mean absolute SHAP value per feature across a sample."""

        sample = frame.head(max_rows) if len(frame) > max_rows else frame
        X = self.model.design_matrix(sample)
        values = self._positive_class_values(X)
        names = self._feature_names
        if len(names) != values.shape[1]:
            names = [f"f{i}" for i in range(values.shape[1])]
        mean_abs = np.abs(values).mean(axis=0)
        frame_out = pd.DataFrame(
            {
                "feature": [humanise_feature_name(n) for n in names],
                "mean_abs_shap": np.round(mean_abs, 6),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        frame_out["importance_pct"] = (
            frame_out["mean_abs_shap"] / frame_out["mean_abs_shap"].sum() * 100
        ).round(3)
        return frame_out.reset_index(drop=True)
