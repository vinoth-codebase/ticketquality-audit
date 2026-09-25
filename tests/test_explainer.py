"""Tests for the SHAP explainer and the dashboard presentation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticketquality.dashboard import components as ui
from ticketquality.models.explainer import (
    RiskDriver,
    TicketExplanation,
    TicketExplainer,
    humanise_feature_name,
)

# --------------------------------------------------------------------------- #
# Feature-name humanisation
# --------------------------------------------------------------------------- #
def test_humanise_known_numeric_feature():
    assert humanise_feature_name("reassignment_count") == "Reassignment count"


def test_humanise_one_hot_feature():
    assert humanise_feature_name("priority_High") == "Priority: High"
    assert humanise_feature_name("assignment_group_Network Ops") == (
        "Assignment group: Network Ops"
    )


def test_humanise_strips_transformer_prefix():
    assert humanise_feature_name("skewed__resolution_minutes") == "Resolution time (minutes)"


def test_humanise_unknown_feature_titlecases():
    assert humanise_feature_name("some_new_column") == "Some New Column"


# --------------------------------------------------------------------------- #
# RiskDriver / TicketExplanation
# --------------------------------------------------------------------------- #
def test_risk_driver_sentence_for_increase():
    driver = RiskDriver(
        feature="reassignment_count",
        label="Reassignment count",
        value=4,
        shap_value=0.5,
        contribution_pct=28.0,
        direction="increases",
    )
    sentence = driver.sentence()
    assert "Reassignment count of 4" in sentence
    assert "increased reopen risk" in sentence
    assert "28.0%" in sentence


def test_risk_driver_sentence_for_decrease():
    driver = RiskDriver(
        feature="interaction_count",
        label="Interaction count",
        value=2.5,
        shap_value=-0.1,
        contribution_pct=-11.5,
        direction="decreases",
    )
    assert "reduced reopen risk" in driver.sentence()
    assert "11.5%" in driver.sentence()


def test_risk_driver_to_dict_includes_explanation():
    driver = RiskDriver("f", "F", 1.0, 0.2, 5.0, "increases")
    payload = driver.to_dict()
    assert payload["feature"] == "f"
    assert "explanation" in payload


def test_risk_driver_formats_large_floats():
    driver = RiskDriver("f", "F", 12345.678, 0.2, 5.0, "increases")
    assert "12,346" in driver.sentence()


def test_ticket_explanation_summary_and_dict():
    explanation = TicketExplanation(
        ticket_id="INC1",
        reopen_probability=0.82,
        risk_level="CRITICAL",
        base_value=-1.2,
        raw_log_odds=1.5,
        drivers=[
            RiskDriver("reassignment_count", "Reassignment count", 5, 0.6, 40.0, "increases"),
            RiskDriver("interaction_count", "Interaction count", 1, -0.1, -7.0, "decreases"),
        ],
    )
    assert "INC1" in explanation.summary()
    assert "CRITICAL" in explanation.summary()
    assert len(explanation.top_drivers(1)) == 1
    assert explanation.to_dict()["drivers"][0]["feature"] == "reassignment_count"


def test_ticket_explanation_summary_without_drivers():
    explanation = TicketExplanation("INC1", 0.1, "LOW", 0.0, 0.0, [])
    assert "no dominant drivers" in explanation.summary()


# --------------------------------------------------------------------------- #
# TicketExplainer
# --------------------------------------------------------------------------- #
def test_explainer_explains_frame(trained_model, ticket_frame):
    explainer = TicketExplainer(trained_model)
    explanations = explainer.explain_frame(ticket_frame.head(5))
    assert len(explanations) == 5
    for explanation in explanations:
        assert 0.0 <= explanation.reopen_probability <= 1.0
        assert explanation.risk_level in {"LOW", "ELEVATED", "CRITICAL"}
        assert explanation.drivers
        # SHAP values are additive; the decomposition must be non-trivial.
        assert any(abs(d.shap_value) > 0 for d in explanation.drivers)


def test_explainer_drivers_are_sorted_by_absolute_contribution(trained_model, ticket_frame):
    explanation = TicketExplainer(trained_model).explain_frame(ticket_frame.head(3))[0]
    magnitudes = [abs(d.shap_value) for d in explanation.drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explainer_respects_max_drivers(trained_model, ticket_frame):
    explainer = TicketExplainer(trained_model, max_drivers=3)
    explanation = explainer.explain_frame(ticket_frame.head(1))[0]
    assert len(explanation.drivers) <= 3


def test_explainer_explain_ticket_by_id(trained_model, ticket_frame):
    ticket_id = str(ticket_frame.iloc[0]["ticket_id"])
    explanation = TicketExplainer(trained_model).explain_ticket(ticket_frame, ticket_id)
    assert explanation.ticket_id == ticket_id


def test_explainer_unknown_ticket_raises(trained_model, ticket_frame):
    with pytest.raises(KeyError):
        TicketExplainer(trained_model).explain_ticket(ticket_frame, "DOES-NOT-EXIST")


def test_explainer_requires_ticket_id_column(trained_model, ticket_frame):
    with pytest.raises(KeyError):
        TicketExplainer(trained_model).explain_ticket(
            ticket_frame.drop(columns=["ticket_id"]), "INC1"
        )


def test_explainer_driver_values_are_resolved_not_na(trained_model, ticket_frame):
    explanation = TicketExplainer(trained_model).explain_frame(ticket_frame.head(3))[0]
    assert all(d.value != "n/a" for d in explanation.drivers)


def test_explainer_global_importance(trained_model, ticket_frame):
    importance = TicketExplainer(trained_model).global_importance(ticket_frame, max_rows=50)
    assert not importance.empty
    assert {"feature", "mean_abs_shap", "importance_pct"}.issubset(importance.columns)
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert importance["importance_pct"].sum() == pytest.approx(100.0, abs=0.5)


def test_explainer_base_value_is_finite(trained_model, ticket_frame):
    explanation = TicketExplainer(trained_model).explain_frame(ticket_frame.head(1))[0]
    assert np.isfinite(explanation.base_value)
    assert np.isfinite(explanation.raw_log_odds)


# --------------------------------------------------------------------------- #
# Dashboard components
# --------------------------------------------------------------------------- #
def test_risk_colour_lookup():
    assert ui.risk_colour("LOW").startswith("#")
    assert ui.risk_colour("critical") == ui.RISK_COLOURS["CRITICAL"]
    assert ui.risk_colour("UNKNOWN").startswith("#")


def test_format_probability():
    assert ui.format_probability(0.1234) == "12.3%"
    assert ui.format_probability("nonsense") == "n/a"


def test_summary_counts(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    counts = ui.summary_counts(scored)
    assert counts["total"] == len(scored)
    assert counts["LOW"] + counts["ELEVATED"] + counts["CRITICAL"] == counts["total"]


def test_summary_counts_empty():
    counts = ui.summary_counts(pd.DataFrame())
    assert counts["total"] == 0


def test_metric_cards(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    cards = ui.metric_cards(scored, threshold=trained_model.classifier.threshold_)
    assert len(cards) == 4
    assert all({"label", "value", "help"} <= set(card) for card in cards)


def test_build_triage_queue_is_ranked(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    queue = ui.build_triage_queue(scored, top_n=10)
    assert len(queue) == 10
    assert queue["reopen_probability"].is_monotonic_decreasing
    assert "risk_pct" in queue.columns


def test_build_triage_queue_empty():
    assert ui.build_triage_queue(pd.DataFrame()).empty


def test_build_group_breakdown(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    breakdown = ui.build_group_breakdown(scored, "category")
    assert not breakdown.empty
    assert "flagged_rate" in breakdown.columns


def test_waterfall_frame_is_ordered_by_magnitude(trained_model, ticket_frame):
    explanation = TicketExplainer(trained_model).explain_frame(ticket_frame.head(1))[0]
    frame = ui.waterfall_frame(explanation)
    assert not frame.empty
    magnitudes = frame["shap_value"].abs().tolist()
    assert magnitudes == sorted(magnitudes)


def test_waterfall_frame_without_drivers():
    explanation = TicketExplanation("INC1", 0.1, "LOW", 0.0, 0.0, [])
    assert ui.waterfall_frame(explanation).empty


def test_cumulative_contribution():
    frame = pd.DataFrame({"shap_value": [0.5, -0.2, 0.3]})
    result = ui.cumulative_contribution(frame)
    assert result["cumulative"].tolist() == pytest.approx([0.5, 0.3, 0.6])


def test_cumulative_contribution_empty():
    assert ui.cumulative_contribution(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# Dashboard app helpers (no Streamlit server required)
# --------------------------------------------------------------------------- #
def test_app_read_ticket_csv_from_bytes(ticket_frame):
    from ticketquality.dashboard.app import read_ticket_csv

    payload = ticket_frame.head(10).to_csv(index=False).encode()
    frame = read_ticket_csv(payload)
    assert len(frame) == 10
    assert "ticket_id" in frame.columns


def test_app_score_uploaded_frame(trained_model, ticket_frame):
    from ticketquality.dashboard.app import score_uploaded_frame

    scored = score_uploaded_frame(trained_model, ticket_frame.head(15))
    assert "reopen_probability" in scored.columns
    assert len(scored) == 15


def test_app_high_risk_explanations(trained_model, ticket_frame):
    from ticketquality.dashboard.app import high_risk_explanations

    scored = trained_model.score_frame(ticket_frame)
    explanations = high_risk_explanations(trained_model, scored, top_n=3)
    assert len(explanations) == 3


def test_app_high_risk_explanations_empty(trained_model):
    from ticketquality.dashboard.app import high_risk_explanations

    assert high_risk_explanations(trained_model, pd.DataFrame(), top_n=3) == []


def test_app_ensure_model_loads_or_trains(fast_config, tmp_path):
    from ticketquality.dashboard import app as app_module

    config = fast_config.model_copy(deep=True)
    config.paths.model_path = str(tmp_path / "missing_model.joblib")
    config.data.synthetic_samples = 200  # keep the fallback training cheap
    model = app_module.ensure_model(config)
    assert model.classifier.estimator_ is not None


def test_app_ensure_model_prefers_saved_artifact(trained_model, fast_config, tmp_path):
    from ticketquality.dashboard import app as app_module

    saved = trained_model.save(tmp_path / "saved_model.joblib")
    config = fast_config.model_copy(deep=True)
    config.paths.model_path = str(saved)
    model = app_module.ensure_model(config)
    assert model.classifier.threshold_ == pytest.approx(trained_model.classifier.threshold_)
