"""Model evaluation suite.

Reports the metrics a support-QA lead actually needs to trust an audit
model:

* **ROC-AUC** — global ranking power.
* **PR-AUC** — ranking power on the rare positive class (premature
  closures are ~15-25% of terminal tickets, so this is the honest metric).
* **F1 / precision / recall** — at the tuned, recall-first threshold.
* **Brier score** — calibration. A risk *probability* that is not
  calibrated is worse than useless for triage, because the dashboard sorts
  by it.
* **Confusion matrix** — the raw operational counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "EvaluationReport",
    "compute_metrics",
    "evaluate_classifier",
    "calibration_table",
]


@dataclass
class EvaluationReport:
    """Container for a full evaluation result."""

    metrics: Dict[str, float] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    threshold: float = 0.5
    n_samples: int = 0
    n_positive: int = 0
    positive_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "positive_rate": round(self.positive_rate, 6),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "confusion_matrix": {
                "labels": ["actual_ok", "actual_reopen"],
                "matrix": self.confusion,
            },
        }

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy two-column table for terminal rendering."""

        order = [
            "roc_auc",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "specificity",
            "accuracy",
            "brier_score",
        ]
        rows = []
        for key in order:
            if key in self.metrics:
                rows.append({"metric": key, "value": round(self.metrics[key], 4)})
        for key, value in self.metrics.items():
            if key not in order:
                rows.append({"metric": key, "value": round(value, 4)})
        return pd.DataFrame(rows)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the full metric bundle at ``threshold``."""

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    preds = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    metrics["f1"] = float(f1_score(y_true, preds, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, preds, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, preds, zero_division=0))
    metrics["accuracy"] = float((preds == y_true).mean())
    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        metrics["brier_score"] = float("nan")

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["false_negative_count"] = float(fn)
    metrics["false_positive_count"] = float(fp)
    metrics["true_positive_count"] = float(tp)
    metrics["true_negative_count"] = float(tn)
    return metrics


def evaluate_classifier(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: Optional[float] = None,
) -> EvaluationReport:
    """Evaluate any fitted :class:`ReopenRiskClassifier` on a design matrix."""

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    cut = threshold if threshold is not None else getattr(model, "threshold_", 0.5)

    metrics = compute_metrics(y_true, y_prob, cut)
    cm = confusion_matrix(y_true, (y_prob >= cut).astype(int), labels=[0, 1]).tolist()

    return EvaluationReport(
        metrics=metrics,
        confusion=cm,
        threshold=float(cut),
        n_samples=int(len(y_true)),
        n_positive=int(y_true.sum()),
        positive_rate=float(y_true.mean()) if len(y_true) else 0.0,
    )


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bin predictions and compare mean predicted vs observed frequency."""

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    rows = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": round(float(edges[bin_index]), 3),
                "bin_upper": round(float(edges[bin_index + 1]), 3),
                "count": int(mask.sum()),
                "mean_predicted": round(float(y_prob[mask].mean()), 4),
                "observed_rate": round(float(y_true[mask].mean()), 4),
            }
        )
    return pd.DataFrame(rows)
