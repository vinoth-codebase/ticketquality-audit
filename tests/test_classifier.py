"""Tests for the classifier, evaluation suite, scorecards and exporters."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ticketquality.models.classifier import (
    LIGHTGBM_AVAILABLE,
    ReopenRiskClassifier,
    TicketAuditModel,
    optimize_threshold,
    risk_level,
)
from ticketquality.models.evaluation import (
    EvaluationReport,
    calibration_table,
    compute_metrics,
    evaluate_classifier,
)
from ticketquality.reports.exporter import (
    build_audit_report,
    export_json,
    export_markdown,
    export_triage_queue,
    json_default,
    save_audit_report,
)
from ticketquality.reports.scorecard import (
    AuditScorecard,
    build_scorecard,
    group_quality_metrics,
    rank_groups_by_risk,
)

# --------------------------------------------------------------------------- #
# Threshold optimisation
# --------------------------------------------------------------------------- #
def test_optimize_threshold_meets_recall_floor(confusion_pairs):
    y_true, y_prob = confusion_pairs
    threshold, stats = optimize_threshold(y_true, y_prob, recall_floor=0.80)
    assert stats["recall_floor_met"] is True
    assert stats["recall"] >= 0.80
    assert 0.0 < threshold < 1.0


def test_optimize_threshold_maximises_precision_under_floor():
    # Two well-separated clusters: a high threshold should keep recall high
    # while shedding low-confidence positives.
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    threshold, stats = optimize_threshold(y_true, y_prob, recall_floor=0.80)
    assert stats["precision"] > 0.9
    assert stats["recall"] >= 0.80


def test_optimize_threshold_handles_single_class():
    y_true = np.zeros(20, dtype=int)
    y_prob = np.linspace(0, 1, 20)
    threshold, stats = optimize_threshold(y_true, y_prob)
    assert threshold == 0.5
    assert stats["recall_floor_met"] is False


def test_optimize_threshold_shape_mismatch_raises():
    with pytest.raises(ValueError):
        optimize_threshold(np.array([0, 1]), np.array([0.5]))


def test_optimize_threshold_recall_floor_is_always_achievable():
    """A recall floor <= 1 is always reachable by lowering the threshold.

    This is a property of binary thresholding, not an accident: flag
    everything and recall is 1.0. The optimiser must therefore report the
    floor as met and pick the most precise cut that satisfies it.
    """

    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.concatenate([np.zeros(90), np.full(10, 0.05)])
    threshold, stats = optimize_threshold(y_true, y_prob, recall_floor=1.0)
    assert stats["recall_floor_met"] is True
    assert stats["recall"] == pytest.approx(1.0)
    assert threshold <= 0.05


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "probability,expected",
    [(0.05, "LOW"), (0.44, "ELEVATED"), (0.99, "CRITICAL")],
)
def test_risk_level_bands(probability, expected):
    assert risk_level(probability, 0.35, 0.60) == expected


def test_risk_level_boundaries_are_inclusive_lower():
    assert risk_level(0.35, 0.35, 0.60) == "ELEVATED"
    assert risk_level(0.60, 0.35, 0.60) == "CRITICAL"


# --------------------------------------------------------------------------- #
# ReopenRiskClassifier
# --------------------------------------------------------------------------- #
def test_classifier_fit_predict_and_threshold(trained_model):
    classifier = trained_model.classifier
    assert classifier.threshold_ > 0.0
    assert classifier.feature_names_
    assert classifier.n_features_in_ > 0


def test_classifier_probabilities_are_valid(trained_model, ticket_frame):
    X = trained_model.pipeline.transform(ticket_frame)
    proba = trained_model.classifier.predict_proba(X)
    assert proba.shape == (len(ticket_frame), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_classifier_predict_returns_binary(trained_model, ticket_frame):
    X = trained_model.pipeline.transform(ticket_frame)
    preds = trained_model.classifier.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_classifier_custom_threshold_changes_predictions(trained_model, ticket_frame):
    X = trained_model.pipeline.transform(ticket_frame)
    strict = trained_model.classifier.predict(X, threshold=0.99)
    loose = trained_model.classifier.predict(X, threshold=0.0)
    assert loose.sum() >= strict.sum()
    assert loose.sum() == len(loose)


def test_classifier_predict_risk(trained_model, ticket_frame):
    X = trained_model.pipeline.transform(ticket_frame)
    bands = trained_model.classifier.predict_risk(X)
    assert set(np.unique(bands)).issubset({"LOW", "ELEVATED", "CRITICAL"})


def test_classifier_feature_importances_sum(trained_model):
    importances = trained_model.classifier.feature_importances()
    assert not importances.empty
    assert importances["importance_pct"].sum() == pytest.approx(100.0, abs=0.5)


def test_classifier_unfitted_raises():
    classifier = ReopenRiskClassifier()
    with pytest.raises(RuntimeError):
        classifier.predict_proba(np.zeros((2, 3)))
    with pytest.raises(RuntimeError):
        classifier.feature_importances()


def test_classifier_rejects_row_mismatch():
    classifier = ReopenRiskClassifier(calibrate=False)
    with pytest.raises(ValueError):
        classifier.fit(np.zeros((5, 3)), np.array([0, 1]))


def test_classifier_works_without_calibration(small_frame):
    classifier = ReopenRiskClassifier(calibrate=False, params={"lightgbm": {"n_estimators": 20}})
    y = small_frame["is_premature_closure"].to_numpy()
    X = np.random.default_rng(0).normal(size=(len(small_frame), 6))
    classifier.fit(X, y)
    assert classifier.calibrator_ is None
    assert classifier.underlying_booster() is not None


def test_classifier_save_and_load_roundtrip(trained_model, tmp_path):
    path = tmp_path / "model.joblib"
    trained_model.classifier.save(path)
    reloaded = ReopenRiskClassifier.load(path)
    assert reloaded.threshold_ == pytest.approx(trained_model.classifier.threshold_)


def test_classifier_backend_resolution():
    classifier = ReopenRiskClassifier(backend="sklearn")
    assert classifier._resolve_backend() == "sklearn"
    classifier = ReopenRiskClassifier(backend="auto")
    assert classifier._resolve_backend() in {"lightgbm", "sklearn"}
    assert isinstance(LIGHTGBM_AVAILABLE, bool)


def test_classifier_sklearn_backend_trains(small_frame):
    classifier = ReopenRiskClassifier(
        backend="sklearn",
        calibrate=False,
        params={"sklearn_fallback": {"max_iter": 30, "max_leaf_nodes": 8}},
    )
    y = small_frame["is_premature_closure"].to_numpy()
    X = np.random.default_rng(1).normal(size=(len(small_frame), 5))
    classifier.fit(X, y)
    assert classifier.backend_ == "sklearn"


# --------------------------------------------------------------------------- #
# TicketAuditModel
# --------------------------------------------------------------------------- #
def test_audit_model_train_returns_metrics(trained_model):
    assert "threshold" in trained_model.metrics_
    assert trained_model.trained_at is not None


def test_audit_model_train_requires_target(ticket_frame):
    model = TicketAuditModel()
    with pytest.raises(KeyError):
        model.train(ticket_frame.drop(columns=["is_premature_closure"]))


def test_audit_model_score_frame_adds_columns(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    assert {"reopen_probability", "risk_level", "flagged_for_review"}.issubset(scored.columns)
    assert scored["reopen_probability"].between(0, 1).all()


def test_audit_model_risk_bands_are_ordered(trained_model):
    low, elevated = trained_model.risk_bands()
    assert low < elevated


def test_audit_model_risk_bands_respect_config(trained_model):
    config = trained_model.config.model_copy(deep=True)
    config.dashboard.risk_low_upper = 0.2
    config.dashboard.risk_elevated_upper = 0.8
    model = TicketAuditModel(config=config, classifier=trained_model.classifier,
                             pipeline=trained_model.pipeline)
    assert model.risk_bands() == (0.2, 0.8)


def test_audit_model_save_and_load(trained_model, ticket_frame, tmp_path):
    path = tmp_path / "audit_model.joblib"
    trained_model.save(path)
    reloaded = TicketAuditModel.load(path)
    assert reloaded.classifier.threshold_ == pytest.approx(trained_model.classifier.threshold_)
    scored = reloaded.score_frame(ticket_frame.head(20))
    assert "reopen_probability" in scored.columns


def test_audit_model_threshold_is_a_valid_probability_cut(trained_model):
    assert 0.0 < trained_model.classifier.threshold_ < 1.0
    assert trained_model.classifier.threshold_stats_["recall_floor_met"] is True


def test_audit_model_from_config_uses_config_params(fast_config):
    model = TicketAuditModel.from_config(fast_config)
    assert model.classifier.recall_floor == fast_config.model.recall_floor
    assert model.target_column == fast_config.model.target


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = compute_metrics(y_true, y_prob, 0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["brier_score"] < 0.01


def test_compute_metrics_single_class_returns_degenerate_aucs():
    metrics = compute_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10), 0.5)
    assert np.isnan(metrics["roc_auc"])
    # average_precision_score returns 0.0 (not NaN) when there are no positives.
    assert metrics["pr_auc"] == 0.0
    assert metrics["recall"] == 0.0


def test_compute_metrics_confusion_counts():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_prob, 0.5)
    assert metrics["true_positive_count"] == 1
    assert metrics["false_positive_count"] == 1
    assert metrics["false_negative_count"] == 1
    assert metrics["specificity"] == pytest.approx(0.5)


def test_evaluate_classifier_report_shape(trained_model, ticket_frame):
    X = trained_model.pipeline.transform(ticket_frame)
    report = evaluate_classifier(
        trained_model.classifier, X, ticket_frame["is_premature_closure"].to_numpy()
    )
    assert isinstance(report, EvaluationReport)
    assert report.n_samples == len(ticket_frame)
    assert len(report.confusion) == 2


def test_evaluation_report_to_dict_and_frame(trained_model, ticket_frame):
    X = trained_model.pipeline.transform(ticket_frame)
    report = evaluate_classifier(
        trained_model.classifier, X, ticket_frame["is_premature_closure"].to_numpy()
    )
    payload = report.to_dict()
    assert "metrics" in payload and "confusion_matrix" in payload
    frame = report.to_frame()
    assert set(frame.columns) == {"metric", "value"}
    assert "roc_auc" in set(frame["metric"])


def test_evaluation_report_extra_metrics_are_included():
    report = EvaluationReport(metrics={"roc_auc": 0.9, "custom_metric": 0.5})
    assert "custom_metric" in set(report.to_frame()["metric"])


def test_calibration_table_bins_sum_to_total(confusion_pairs):
    y_true, y_prob = confusion_pairs
    table = calibration_table(y_true, y_prob, n_bins=5)
    assert table["count"].sum() == len(y_true)
    assert table["mean_predicted"].between(0, 1).all()


def test_calibration_table_empty_bins_are_skipped():
    table = calibration_table(np.array([0, 1]), np.array([0.01, 0.99]), n_bins=10)
    assert len(table) <= 10


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def test_group_quality_metrics_shape(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    metrics = group_quality_metrics(scored, "assignment_group")
    assert not metrics.empty
    assert {"tickets", "flagged_rate", "avg_reopen_probability"}.issubset(metrics.columns)
    assert metrics["flagged_rate"].between(0, 1).all()


def test_group_quality_metrics_missing_column_returns_empty(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    assert group_quality_metrics(scored, "not_a_column").empty


def test_group_quality_metrics_min_volume_filter(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    metrics = group_quality_metrics(scored, "assignment_group", min_volume=10_000)
    assert metrics.empty


def test_build_scorecard_summary(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    scorecard = build_scorecard(scored, threshold=trained_model.classifier.threshold_)
    assert scorecard.summary["tickets_audited"] == len(scored)
    assert scorecard.recommendations
    assert "assignment_group" in scorecard.by_group


def test_build_scorecard_empty_frame():
    scorecard = build_scorecard(pd.DataFrame())
    assert scorecard.summary["tickets_audited"] == 0
    assert scorecard.recommendations


def test_build_scorecard_without_ground_truth(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame).drop(columns=["is_premature_closure"])
    scorecard = build_scorecard(scored, actual_column=None)
    assert "observed_reopen_rate_pct" not in scorecard.summary


def test_scorecard_to_dict_and_markdown(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    scorecard = build_scorecard(scored)
    payload = scorecard.to_dict()
    assert "summary" in payload and "breakdowns" in payload
    markdown = scorecard.to_markdown()
    assert markdown.startswith("# Support QA Audit Scorecard")


def test_rank_groups_by_risk_orders_descending(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    ranked = rank_groups_by_risk(scored, "assignment_group", min_volume=1)
    assert not ranked.empty
    assert ranked["risk_score"].is_monotonic_decreasing


def test_rank_groups_empty_when_volume_too_high(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    assert rank_groups_by_risk(scored, "assignment_group", min_volume=100_000).empty


def test_audit_scorecard_to_dict_with_breakdowns():
    scorecard = AuditScorecard(
        summary={"tickets_audited": 1},
        by_group={"category": pd.DataFrame({"category": ["X"], "tickets": [1]})},
        recommendations=["do the thing"],
    )
    payload = scorecard.to_dict()
    assert payload["breakdowns"]["category"][0]["category"] == "X"


# --------------------------------------------------------------------------- #
# Exporters
# --------------------------------------------------------------------------- #
def test_json_default_handles_numpy_and_timestamps():
    assert json_default(np.int64(3)) == 3
    assert json_default(np.float64(1.5)) == 1.5
    assert json_default(np.bool_(True)) is True
    assert json_default(np.array([1, 2])) == [1, 2]
    assert isinstance(json_default(pd.Timestamp("2025-01-01")), str)
    assert json_default(np.nan) is None


def test_json_default_rejects_unknown_types():
    with pytest.raises(TypeError):
        json_default(object())


def test_export_json_writes_valid_file(tmp_path):
    path = export_json({"a": np.int64(1), "b": np.array([1.0, 2.0])}, tmp_path / "r.json")
    payload = json.loads(path.read_text())
    assert payload["a"] == 1
    assert payload["b"] == [1.0, 2.0]


def test_export_markdown_and_triage_queue(trained_model, ticket_frame, tmp_path):
    scored = trained_model.score_frame(ticket_frame)
    scorecard = build_scorecard(scored)
    md = export_markdown(scorecard, tmp_path / "report.md")
    assert md.exists() and md.read_text().startswith("# Support QA Audit Scorecard")

    queue_path = export_triage_queue(scored, tmp_path / "queue.csv", top_n=10)
    queue = pd.read_csv(queue_path)
    assert len(queue) == 10
    assert queue["reopen_probability"].is_monotonic_decreasing


def test_build_audit_report_payload(trained_model, ticket_frame):
    scored = trained_model.score_frame(ticket_frame)
    payload = build_audit_report(
        scored,
        model_metrics=trained_model.metrics_,
        threshold=trained_model.classifier.threshold_,
        model_backend=trained_model.classifier.backend_,
        source="unit-test",
    )
    assert payload["report_type"] == "premature_closure_audit"
    assert payload["source"] == "unit-test"
    assert payload["triage_queue"]
    assert payload["model"]["backend"] == trained_model.classifier.backend_


def test_build_audit_report_with_explanations(trained_model, ticket_frame):
    from ticketquality.models.explainer import TicketExplainer

    scored = trained_model.score_frame(ticket_frame)
    top = scored.sort_values("reopen_probability", ascending=False).head(2)
    explanations = TicketExplainer(trained_model).explain_frame(top)
    payload = build_audit_report(scored, explanations=explanations)
    assert len(payload["top_risk_explanations"]) == 2


def test_save_audit_report_json_and_markdown(trained_model, ticket_frame, tmp_path):
    scored = trained_model.score_frame(ticket_frame)
    payload = build_audit_report(scored, threshold=trained_model.classifier.threshold_)

    json_path = save_audit_report(payload, tmp_path / "report.json")
    assert json.loads(json_path.read_text())["report_type"] == "premature_closure_audit"

    md_path = save_audit_report(payload, tmp_path / "report.md")
    assert "Support QA Audit Scorecard" in md_path.read_text()
