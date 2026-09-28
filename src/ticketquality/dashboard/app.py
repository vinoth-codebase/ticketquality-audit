"""Streamlit visual-management dashboard.

Run with::

    streamlit run src/ticketquality/dashboard/app.py

Three surfaces, matching the three questions a support QA lead asks:

1. **Single Ticket Auditor** — "why is *this* ticket risky?" Enter or
   select a ticket id and see the probability, the risk band, and a SHAP
   waterfall explaining the score in plain English.
2. **Batch QA File Uploader** — "what should my reviewers look at first?"
   Upload a CSV of recently closed tickets and get a ranked triage queue.
3. **Team Quality Metrics** — "where is the process leaking?" Premature
   closure rates sliced by assignment group, category and shift.

The module is import-safe: Streamlit only executes :func:`main` when the
file is run as a script, so the helpers can be unit-tested headlessly.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import get_config
from ..data.synthetic import generate_synthetic_tickets
from ..models.classifier import TicketAuditModel
from ..models.explainer import TicketExplainer
from ..reports.scorecard import build_scorecard, group_quality_metrics
from . import components as ui

CONFIG = get_config()

# Populated by ``main`` and shared across the tab renderers.
_STATE: dict = {}


# --------------------------------------------------------------------------- #
# Data / model helpers (no Streamlit calls — unit-testable)
# --------------------------------------------------------------------------- #
def ensure_model(config=None) -> TicketAuditModel:
    """Load the persisted model, or train a fresh one on synthetic data."""

    cfg = config or get_config()
    model_path = cfg.path("model_path")
    if model_path.exists():
        try:
            return TicketAuditModel.load(model_path)
        except Exception:  # pragma: no cover - corrupt artefact fallback
            pass
    frame = generate_synthetic_tickets(cfg.data.synthetic_samples, seed=cfg.project.random_seed)
    model = TicketAuditModel.from_config(cfg)
    model.train(frame)
    return model


def read_ticket_csv(payload: bytes | str | Path) -> pd.DataFrame:
    """Read an uploaded CSV into a frame, tolerating common delimiters."""

    if isinstance(payload, bytes):
        buffer: Any = io.BytesIO(payload)
    elif isinstance(payload, (str, Path)) and Path(str(payload)).exists():
        buffer = Path(str(payload))
    else:
        buffer = io.StringIO(str(payload))
    frame = pd.read_csv(buffer, sep=None, engine="python")
    return frame


def score_uploaded_frame(model: TicketAuditModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Score an arbitrary uploaded frame, keeping identifiers intact."""

    scored = model.score_frame(frame)
    return scored


def high_risk_explanations(
    model: TicketAuditModel, scored: pd.DataFrame, top_n: int = 5
) -> list:
    """Explain the ``top_n`` riskiest tickets in a scored frame."""

    if scored.empty:
        return []
    explainer = TicketExplainer(model)
    ranked = scored.sort_values("reopen_probability", ascending=False).head(top_n)
    return explainer.explain_frame(ranked)


# --------------------------------------------------------------------------- #
# Streamlit rendering
# --------------------------------------------------------------------------- #
def render_single_ticket(model: TicketAuditModel, frame: pd.DataFrame) -> None:
    import streamlit as st

    st.subheader("Single Ticket Auditor")
    st.caption(
        "Inspect one closed ticket: predicted reopen probability, risk band, "
        "and the SHAP drivers behind the score."
    )

    scored = model.score_frame(frame)
    ids = scored["ticket_id"].astype(str).tolist() if "ticket_id" in scored else []
    if not ids:
        st.warning("The loaded dataset has no `ticket_id` column.")
        return

    default_index = int(scored["reopen_probability"].idxmax())
    selected = st.selectbox(
        "Ticket ID",
        options=ids,
        index=ids.index(str(scored.loc[default_index, "ticket_id"])) if ids else 0,
    )

    row = scored[scored["ticket_id"].astype(str) == str(selected)].iloc[0]
    probability = float(row["reopen_probability"])
    level = str(row["risk_level"])

    left, middle, right = st.columns(3)
    left.metric("Reopen probability", ui.format_probability(probability))
    middle.metric("Risk band", level)
    right.metric(
        "Flagged for review",
        "Yes" if int(row.get("flagged_for_review", 0)) == 1 else "No",
    )

    st.markdown(
        f"<div style='padding:0.5rem 0.75rem;border-left:5px solid "
        f"{ui.risk_colour(level)};background:rgba(127,127,127,0.08)'>"
        f"<b>{level} risk</b> — reopen probability {ui.format_probability(probability)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Computing SHAP attributions…"):
        explanation = TicketExplainer(model).explain_ticket(frame, selected)

    st.markdown("#### Why this ticket was flagged")
    waterfall = ui.waterfall_frame(explanation)
    if waterfall.empty:
        st.info("No feature attributions available for this ticket.")
    else:
        st.bar_chart(waterfall.set_index("label")["shap_value"])
        st.dataframe(
            waterfall[["label", "value", "explanation"]].rename(
                columns={"label": "Driver", "value": "Observed value", "explanation": "Impact"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Ticket attributes"):
        st.dataframe(pd.DataFrame([row]).T.rename(columns={0: "value"}))


def render_batch_tab(model: TicketAuditModel) -> None:
    import streamlit as st

    st.subheader("Batch QA File Uploader")
    st.caption(
        "Upload a CSV of recently closed tickets to produce a ranked triage "
        "queue of at-risk closures needing peer review."
    )

    uploaded = st.file_uploader("Closed tickets (CSV)", type=["csv"])
    if uploaded is None:
        st.info("Awaiting a CSV upload. A sample lives at `examples/sample_tickets.csv`.")
        return

    try:
        frame = read_ticket_csv(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        return

    scored = score_uploaded_frame(model, frame)
    cards = ui.metric_cards(scored, model.classifier.threshold_)
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        column.metric(card["label"], card["value"], help=card["help"])

    st.markdown("#### Ranked triage queue")
    queue = ui.build_triage_queue(scored, top_n=50)
    display_columns = [
        c
        for c in (
            "ticket_id",
            "risk_pct",
            "risk_level",
            "reassignment_count",
            "resolution_minutes",
            "priority",
            "category",
            "assignment_group",
            "shift",
        )
        if c in queue.columns
    ]
    st.dataframe(queue[display_columns], use_container_width=True, hide_index=True)

    st.download_button(
        "Download scored queue (CSV)",
        data=scored.sort_values("reopen_probability", ascending=False).to_csv(index=False),
        file_name="triage_queue.csv",
        mime="text/csv",
    )

    scorecard = build_scorecard(scored, threshold=model.classifier.threshold_)
    st.markdown("#### Recommended QA actions")
    for note in scorecard.recommendations:
        st.markdown(f"- {note}")


def render_team_tab(model: TicketAuditModel, frame: pd.DataFrame) -> None:
    import streamlit as st

    st.subheader("Team Quality Metrics")
    st.caption("Premature closure risk sliced by assignment group, category and shift.")

    scored = model.score_frame(frame)
    grouping = st.radio(
        "Group by",
        options=["assignment_group", "category", "shift"],
        horizontal=True,
    )
    if grouping not in scored.columns:
        st.warning(f"The dataset has no `{grouping}` column.")
        return

    metrics = group_quality_metrics(scored, grouping)
    if metrics.empty:
        st.info("Not enough data to build a breakdown.")
        return

    st.bar_chart(metrics.set_index(grouping)["flagged_rate"])
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.markdown("#### Highest-risk groups (volume-weighted)")
    weighted = metrics.copy()
    weighted["risk_score"] = (
        weighted["flagged_rate"] * np.log1p(weighted["tickets"])
    ).round(4)
    st.dataframe(
        weighted.sort_values("risk_score", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title=CONFIG.dashboard.page_title,
        page_icon="🎫",
        layout="wide",
    )
    st.title("TicketQuality-Audit")
    st.caption(
        "Premature closure & re-open risk forecaster for software support. "
        "Audits closed tickets and surfaces the ones that will bounce back."
    )

    with st.spinner("Loading audit model…"):
        model = ensure_model(CONFIG)

    dataset_path = os.environ.get("TICKETQUALITY_DATASET")
    if dataset_path and Path(dataset_path).exists():
        frame = pd.read_csv(dataset_path)
    else:
        frame = generate_synthetic_tickets(
            CONFIG.data.synthetic_samples, seed=CONFIG.project.random_seed
        )

    backend = getattr(model.classifier, "backend_", "unknown")
    threshold = getattr(model.classifier, "threshold_", 0.5)
    st.caption(
        f"Model backend: **{backend}** · operating threshold: **{threshold:.3f}** · "
        f"dataset rows: **{len(frame):,}**"
    )

    single_tab, batch_tab, team_tab = st.tabs(
        ["Single Ticket Auditor", "Batch QA Uploader", "Team Quality Metrics"]
    )
    with single_tab:
        render_single_ticket(model, frame)
    with batch_tab:
        render_batch_tab(model)
    with team_tab:
        render_team_tab(model, frame)


if __name__ == "__main__":
    main()
