"""Predictive modelling, evaluation and SHAP explainability."""

from .classifier import (
    LIGHTGBM_AVAILABLE,
    ReopenRiskClassifier,
    TicketAuditModel,
    optimize_threshold,
    risk_level,
)
from .evaluation import (
    EvaluationReport,
    calibration_table,
    compute_metrics,
    evaluate_classifier,
)
from .explainer import (
    SHAP_AVAILABLE,
    RiskDriver,
    TicketExplanation,
    TicketExplainer,
    humanise_feature_name,
)

__all__ = [
    "LIGHTGBM_AVAILABLE",
    "ReopenRiskClassifier",
    "TicketAuditModel",
    "optimize_threshold",
    "risk_level",
    "EvaluationReport",
    "calibration_table",
    "compute_metrics",
    "evaluate_classifier",
    "SHAP_AVAILABLE",
    "RiskDriver",
    "TicketExplanation",
    "TicketExplainer",
    "humanise_feature_name",
]
