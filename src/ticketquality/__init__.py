"""TicketQuality-Audit — premature closure & re-open risk forecaster.

An enterprise-grade audit engine for software support: it ingests ITSM
incident logs, engineers operational-velocity features, trains a
calibrated gradient-boosted classifier to predict premature closures,
explains each prediction with exact SHAP attributions, and surfaces the
result through a Rich CLI and a Streamlit visual-management dashboard.

Quick start::

    from ticketquality import generate_synthetic_tickets, TicketAuditModel

    frame = generate_synthetic_tickets(1000)
    model = TicketAuditModel.from_config()
    model.train(frame)
    scored = model.score_frame(frame)
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Vinothkumar N"
__email__ = "vinothofficial14@gmail.com"
__license__ = "MIT"

from .config import AuditConfig, get_config, load_config
from .data.synthetic import (
    SyntheticTicketGenerator,
    generate_synthetic_tickets,
    write_synthetic_dataset,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "AuditConfig",
    "get_config",
    "load_config",
    "SyntheticTicketGenerator",
    "generate_synthetic_tickets",
    "write_synthetic_dataset",
    "TicketAuditModel",
    "TicketExplainer",
    "build_scorecard",
]


def __getattr__(name: str):
    """Lazily expose the heavier model/report entry points.

    Importing :mod:`ticketquality` should stay cheap — it must not pull in
    LightGBM or SHAP unless the caller actually asks for a model.
    """

    if name == "TicketAuditModel":
        from .models.classifier import TicketAuditModel

        return TicketAuditModel
    if name == "TicketExplainer":
        from .models.explainer import TicketExplainer

        return TicketExplainer
    if name == "build_scorecard":
        from .reports.scorecard import build_scorecard

        return build_scorecard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
